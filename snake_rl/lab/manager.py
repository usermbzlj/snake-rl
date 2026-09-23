"""ExperimentManager: lifecycle, live-patch, weight fan-out."""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing as mp
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from snake_rl.core.config import ExperimentConfig, live_field_keys
from snake_rl.lab.storage import ExperimentStore
from snake_rl.lab.worker import run_worker


@dataclass
class _WorkerHandle:
    process: Any
    cmd_q: Any
    out_q: Any
    reader: threading.Thread
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    status: str = "running"
    error: str | None = None
    env_steps: int = 0


class ExperimentManager:
    def __init__(
        self,
        store: ExperimentStore | None = None,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self.store = store or ExperimentStore()
        self._ctx = mp.get_context("spawn")
        self._workers: dict[str, _WorkerHandle] = {}
        self._latest_weights: dict[str, tuple[int, dict[str, Any], int]] = {}
        # exp_id -> (version, state_dict, env_steps)
        self._lock = threading.Lock()
        self._loop = loop
        self._closed = False

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def startup(self) -> None:
        self.store.root.mkdir(parents=True, exist_ok=True)
        self.store.mark_stale_workers_stopped()

    def shutdown(self) -> None:
        self._closed = True
        ids = list(self._workers.keys())
        for eid in ids:
            try:
                self.stop(eid, wait=True, timeout=15.0)
            except Exception:
                self._force_kill(eid)

    def _force_kill(self, exp_id: str) -> None:
        h = self._workers.get(exp_id)
        if h is None:
            return
        if h.process.is_alive():
            h.process.terminate()
            h.process.join(timeout=3.0)
            if h.process.is_alive():
                h.process.kill()
                h.process.join(timeout=2.0)
        self._workers.pop(exp_id, None)

    def list(self) -> list[dict[str, Any]]:
        return [self.store.summary(eid) for eid in self.store.list_ids()]

    def get(self, exp_id: str) -> dict[str, Any]:
        return self.store.detail(exp_id)

    def summary(self, exp_id: str) -> dict[str, Any]:
        s = self.store.summary(exp_id)
        h = self._workers.get(exp_id)
        if h is not None:
            s["status"] = h.status
            if h.env_steps:
                s["env_steps"] = h.env_steps
        return s

    def create(self, config: ExperimentConfig, *, start: bool = True) -> dict[str, Any]:
        meta = self.store.create(config)
        if start:
            self.start(meta["id"], resume=False)
        return self.summary(meta["id"])

    def start(self, exp_id: str, *, resume: bool | None = None) -> dict[str, Any]:
        if exp_id in self._workers and self._workers[exp_id].process.is_alive():
            raise RuntimeError("实验已在运行中")
        meta = self.store.read_meta(exp_id)
        status = meta.get("status")
        if resume is None:
            resume = (
                status in ("stopped", "finished", "paused", "error")
                and self.store.ckpt_path(exp_id, "latest").is_file()
            )
        if status == "running" and exp_id not in self._workers:
            # Stale status — treat as continue
            resume = self.store.ckpt_path(exp_id, "latest").is_file()

        config = self.store.read_config(exp_id)
        cmd_q = self._ctx.Queue()
        out_q = self._ctx.Queue(maxsize=64)
        proc = self._ctx.Process(
            target=run_worker,
            args=(str(self.store.exp_dir(exp_id)), config.model_dump(), bool(resume), cmd_q, out_q),
            name=f"snake-worker-{exp_id}",
            daemon=True,
        )
        handle = _WorkerHandle(process=proc, cmd_q=cmd_q, out_q=out_q, reader=threading.Thread())
        # Start reader before process so we don't miss early messages
        handle.reader = threading.Thread(
            target=self._reader_loop,
            args=(exp_id, handle),
            name=f"snake-reader-{exp_id}",
            daemon=True,
        )
        self._workers[exp_id] = handle
        handle.reader.start()
        proc.start()
        handle.status = "running"
        self.store.update_status(exp_id, "running", clear_error=True)
        return self.summary(exp_id)

    def pause(self, exp_id: str) -> dict[str, Any]:
        h = self._require_live(exp_id)
        h.cmd_q.put(("pause",))
        return self.summary(exp_id)

    def resume(self, exp_id: str) -> dict[str, Any]:
        h = self._require_live(exp_id)
        h.cmd_q.put(("resume",))
        return self.summary(exp_id)

    def stop(self, exp_id: str, *, wait: bool = True, timeout: float = 30.0) -> dict[str, Any]:
        h = self._workers.get(exp_id)
        if h is None:
            meta = self.store.read_meta(exp_id)
            if meta.get("status") in ("running", "paused"):
                self.store.update_status(exp_id, "stopped")
            return self.summary(exp_id)
        with contextlib.suppress(Exception):
            h.cmd_q.put(("stop",))
        if wait:
            h.process.join(timeout=timeout)
            if h.process.is_alive():
                h.process.terminate()
                h.process.join(timeout=5.0)
                if h.process.is_alive():
                    h.process.kill()
                    h.process.join(timeout=2.0)
            h.reader.join(timeout=2.0)
            self._workers.pop(exp_id, None)
            meta = self.store.read_meta(exp_id)
            if meta.get("status") in ("running", "paused"):
                self.store.update_status(exp_id, "stopped")
        return self.summary(exp_id)

    def live_patch(self, exp_id: str, patch: dict[str, float]) -> dict[str, Any]:
        meta = self.store.read_meta(exp_id)
        algo = meta["algo"]
        allowed = live_field_keys(algo)
        bad = [k for k in patch if k not in allowed]
        if bad:
            raise ValueError(f"不可实时调整的字段: {', '.join(bad)}")
        if not patch:
            raise ValueError("补丁为空")

        # Apply to persisted config immediately (worker also applies when live)
        cfg = ExperimentConfig.model_validate(meta["config"])
        changes: dict[str, list[Any]] = {}
        for key, new_val in patch.items():
            parts = key.split(".")
            obj: Any = cfg
            for p in parts[:-1]:
                obj = getattr(obj, p)
            old = getattr(obj, parts[-1])
            coerced = type(old)(new_val)
            setattr(obj, parts[-1], coerced)
            changes[key] = [old, coerced]

        h = self._workers.get(exp_id)
        event: dict[str, Any]
        if h is not None and h.process.is_alive():
            # Worker owns disk writes for config + event while running
            h.cmd_q.put(("live_patch", {k: float(v) for k, v in patch.items()}))
            event = {
                "t": time.time(),
                "env_steps": int(meta.get("env_steps") or h.env_steps or 0),
                "type": "live_patch",
                "data": {"changes": changes},
            }
        else:
            self.store.update_config(exp_id, cfg)
            event = self.store.append_event(
                exp_id,
                {
                    "t": time.time(),
                    "env_steps": int(meta.get("env_steps") or 0),
                    "type": "live_patch",
                    "data": {"changes": changes},
                },
            )
        return {"config": cfg.model_dump(), "event": event}

    def delete(self, exp_id: str) -> None:
        if exp_id in self._workers:
            self.stop(exp_id, wait=True, timeout=15.0)
        self._latest_weights.pop(exp_id, None)
        self.store.delete(exp_id)

    def clone(self, exp_id: str, *, name: str, with_weights: bool) -> dict[str, Any]:
        meta = self.store.clone(exp_id, name=name, with_weights=with_weights)
        return self.summary(meta["id"])

    def subscribe(self, exp_id: str) -> asyncio.Queue:
        if not self.store.exists(exp_id):
            raise FileNotFoundError(f"实验不存在: {exp_id}")
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        h = self._workers.get(exp_id)
        if h is not None:
            with self._lock:
                h.subscribers.append(q)
        return q

    def unsubscribe(self, exp_id: str, q: asyncio.Queue) -> None:
        h = self._workers.get(exp_id)
        if h is None:
            return
        with self._lock:
            if q in h.subscribers:
                h.subscribers.remove(q)

    def latest_weights(self, exp_id: str) -> tuple[int, dict[str, Any], int] | None:
        """Return (version, cpu_state_dict, env_steps) or None."""
        with self._lock:
            w = self._latest_weights.get(exp_id)
            if w is not None:
                return w
        # Fall back to disk for stopped experiments
        for name in ("latest", "best"):
            path = self.store.ckpt_path(exp_id, name)
            if path.is_file():
                try:
                    from snake_rl.core.checkpoint import load_checkpoint

                    data = load_checkpoint(path)
                    trainer = data["trainer"]
                    if "net" in trainer:
                        sd = trainer["net"]
                    elif "online" in trainer:
                        sd = trainer["online"]
                    else:
                        continue
                    steps = int(trainer.get("env_steps", 0))
                    return (0, sd, steps)
                except Exception:
                    continue
        return None

    def _require_live(self, exp_id: str) -> _WorkerHandle:
        h = self._workers.get(exp_id)
        if h is None or not h.process.is_alive():
            raise RuntimeError("实验未在运行")
        return h

    def _reader_loop(self, exp_id: str, handle: _WorkerHandle) -> None:
        import queue as queue_mod

        # Wait until the process has a PID (Windows spawn can briefly look dead).
        for _ in range(200):
            if handle.process.pid is not None or not handle.process.is_alive():
                break
            time.sleep(0.01)

        while not self._closed:
            try:
                msg = handle.out_q.get(timeout=0.25)
            except queue_mod.Empty:
                if handle.process.exitcode is not None:
                    # Process exited — drain any remaining messages
                    while True:
                        try:
                            msg = handle.out_q.get_nowait()
                        except queue_mod.Empty:
                            break
                        self._handle_msg(exp_id, handle, msg)
                    break
                continue
            self._handle_msg(exp_id, handle, msg)

        # Process ended: reconcile status if needed, keep handle until stop()/join
        if handle.process.exitcode is not None:
            try:
                meta = self.store.read_meta(exp_id)
                if meta.get("status") in ("running", "paused") and handle.status in ("running", "paused"):
                    # Unexpected exit without status update
                    if handle.error:
                        self.store.update_status(exp_id, "error", error=handle.error)
                        handle.status = "error"
                    else:
                        # Worker may have already persisted status; re-read
                        meta2 = self.store.read_meta(exp_id)
                        handle.status = meta2.get("status", "stopped")
            except Exception:
                pass
            # Only drop if still this handle and process is gone
            cur = self._workers.get(exp_id)
            if cur is handle and handle.process.exitcode is not None and not handle.process.is_alive():
                self._workers.pop(exp_id, None)

    def _handle_msg(self, exp_id: str, handle: _WorkerHandle, msg: tuple[Any, ...]) -> None:
        kind = msg[0]
        if kind == "weights":
            version, state_dict, env_steps = msg[1], msg[2], msg[3] if len(msg) > 3 else 0
            with self._lock:
                self._latest_weights[exp_id] = (int(version), state_dict, int(env_steps))
                handle.env_steps = int(env_steps)
            return
        if kind == "metrics":
            row = msg[1]
            handle.env_steps = int(row.get("env_steps", handle.env_steps))
            self._fanout(handle, {"type": "metrics", "row": row})
            return
        if kind == "event":
            self._fanout(handle, {"type": "event", "event": msg[1]})
            return
        if kind == "status":
            status = msg[1]
            err = msg[2] if len(msg) > 2 else None
            handle.status = status
            handle.error = err
            payload: dict[str, Any] = {"type": "status", "status": status}
            if err:
                payload["error"] = err
            self._fanout(handle, payload)
            return
        if kind == "error":
            handle.error = str(msg[1])
            handle.status = "error"
            self._fanout(handle, {"type": "status", "status": "error", "error": handle.error})
            return
        if kind == "config":
            return

    def _fanout(self, handle: _WorkerHandle, payload: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None:
            return
        with self._lock:
            subs = list(handle.subscribers)

        def _put(q: asyncio.Queue, item: dict[str, Any]) -> None:
            try:
                q.put_nowait(item)
            except asyncio.QueueFull:
                with contextlib.suppress(Exception):
                    _ = q.get_nowait()
                with contextlib.suppress(Exception):
                    q.put_nowait(item)

        for q in subs:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(_put, q, payload)


def experiments_root_from_env() -> Path:
    import os

    raw = os.environ.get("SNAKE_RL_EXPERIMENTS")
    if raw:
        return Path(raw)
    return ExperimentStore().root
