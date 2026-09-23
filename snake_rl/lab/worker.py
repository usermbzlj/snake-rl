"""Training worker process (multiprocessing spawn entry)."""

from __future__ import annotations

import contextlib
import logging
import multiprocessing
import queue
import time
import traceback
from pathlib import Path
from typing import Any

from snake_rl.core.checkpoint import load_checkpoint, save_checkpoint
from snake_rl.core.config import ExperimentConfig
from snake_rl.core.trainer import make_trainer, reward_weights_tensor, set_seed
from snake_rl.lab.storage import ExperimentStore

log = logging.getLogger(__name__)


def _cpu_state_dict(net_state: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in net_state.items():
        if hasattr(v, "detach"):
            out[k] = v.detach().cpu().clone()
        else:
            out[k] = v
    return out


def _put(out_q: Any, msg: tuple[Any, ...], *, droppable: bool = False) -> None:
    try:
        out_q.put_nowait(msg)
    except queue.Full:
        if droppable:
            # Drop oldest droppable (weights) if possible, then retry once
            with contextlib.suppress(queue.Empty):
                _ = out_q.get_nowait()
            with contextlib.suppress(queue.Full):
                out_q.put_nowait(msg)
            return
        # Metrics / status / events must not be lost — block briefly
        with contextlib.suppress(queue.Full):
            out_q.put(msg, timeout=5.0)


def _sanitize_metrics(row: dict[str, Any]) -> dict[str, float]:
    clean: dict[str, float] = {}
    for k, v in row.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv != fv or fv == float("inf") or fv == float("-inf"):
            continue
        clean[k] = fv
    return clean


class WorkerSession:
    """Owns one training process loop: commands, steps, checkpoints, events."""

    def __init__(
        self,
        exp_dir: str,
        config_dict: dict[str, Any],
        resume: bool,
        cmd_q: Any,
        out_q: Any,
    ) -> None:
        self.exp_path = Path(exp_dir)
        self.exp_id = self.exp_path.name
        self.store = ExperimentStore(self.exp_path.parent)
        self.config = ExperimentConfig.model_validate(config_dict)
        self.resume = resume
        self.cmd_q = cmd_q
        self.out_q = out_q

        self.paused = False
        self.stop_requested = False
        self.weight_version = 0
        self.best_eval = float("-inf")
        self.last_ckpt_t = time.perf_counter()
        self.last_weights_t = 0.0
        self.last_metric_write_t = 0.0
        self.last_elapsed_t = 0.0
        self.pending_metric: dict[str, float] | None = None
        self.metric_interval = 0.5  # ≤ ~2 rows/s
        self.started_wall = time.perf_counter()
        self.base_elapsed = 0.0
        self.trainer: Any = None

    def run(self) -> None:
        try:
            self._setup()
            finished = self._main_loop()
            if finished:
                return
            # Explicit stop, or parent-process death / loop exit — same cleanup as before.
            self._finish_stop()
        except Exception as exc:
            self._handle_error(exc)

    def _setup(self) -> None:
        meta = self.store.read_meta(self.exp_id)
        self.base_elapsed = float(meta.get("elapsed_s") or 0.0)
        if meta.get("best_eval_score") is not None:
            self.best_eval = float(meta["best_eval_score"])

        set_seed(self.config.run.seed)
        self.trainer = make_trainer(self.config)

        if self.resume:
            self._resume_from_checkpoint(meta)

        self.store.update_status(self.exp_id, "running", clear_error=True)
        self._emit_event("start", {"resume": self.resume})
        self._put_status("running")

        self._broadcast_weights()
        self.last_weights_t = time.perf_counter()

    def _resume_from_checkpoint(self, meta: dict[str, Any]) -> None:
        """Load latest.pt and sync live config from disk.

        Trainers expose ``apply_live`` (which refreshes reward weights) but no
        dedicated ``set_config`` / ``refresh_reward_weights`` API, so resume
        assigns ``trainer.config`` then rebuilds ``trainer.weights`` via
        ``reward_weights_tensor`` — same behaviour as before the refactor.
        """
        latest = self.store.ckpt_path(self.exp_id, "latest")
        if not latest.is_file():
            return
        data = load_checkpoint(latest)
        self.trainer.load_state_dict(data["trainer"])
        # Prefer live config from disk (may have been live-patched while stopped)
        self.config = ExperimentConfig.model_validate(meta["config"])
        self.trainer.config = self.config
        device = getattr(self.trainer, "device", None)
        if device is not None and hasattr(self.trainer, "weights"):
            self.trainer.weights = reward_weights_tensor(self.config, device)
        self.best_eval = max(
            self.best_eval,
            float(self.trainer.state_dict().get("best_eval", float("-inf"))),
        )

    def _main_loop(self) -> bool:
        """Return True if finished via max_env_steps."""
        parent = multiprocessing.parent_process()
        last_parent_check_t = time.perf_counter()
        trainer = self.trainer

        while not self.stop_requested:
            if time.perf_counter() - last_parent_check_t >= 2.0:
                last_parent_check_t = time.perf_counter()
                if parent is not None and not parent.is_alive():
                    break

            self._drain_commands()
            if self.stop_requested:
                break

            if self.paused:
                time.sleep(0.05)
                continue

            row = self._train_step()
            now = time.perf_counter()
            self._maybe_save_best(row)
            self._maybe_broadcast_weights(now)
            self._maybe_checkpoint(now)
            self._maybe_update_meta(now)

            max_steps = int(self.config.run.max_env_steps or 0)
            if max_steps > 0 and trainer.env_steps >= max_steps:
                self._finish_max_steps()
                return True

        return False

    def _drain_commands(self) -> None:
        while True:
            try:
                cmd = self.cmd_q.get_nowait()
            except queue.Empty:
                break
            kind = cmd[0] if isinstance(cmd, tuple | list) else cmd
            if kind == "pause":
                self._handle_pause()
            elif kind == "resume":
                self._handle_resume()
            elif kind == "stop":
                self.stop_requested = True
                break
            elif kind == "live_patch":
                patch = cmd[1] if len(cmd) > 1 else {}
                self._handle_live_patch(patch)

    def _handle_pause(self) -> None:
        self.paused = True
        self.store.update_status(self.exp_id, "paused")
        self._emit_event("pause", {})
        self._put_status("paused")

    def _handle_resume(self) -> None:
        self.paused = False
        self.store.update_status(self.exp_id, "running")
        self._emit_event("resume", {})
        self._put_status("running")

    def _handle_live_patch(self, patch: dict[str, Any]) -> None:
        trainer = self.trainer
        old_cfg = trainer.config.model_dump()
        trainer.apply_live(patch)
        self.config = trainer.config
        changes: dict[str, list[Any]] = {}
        new_cfg = self.config.model_dump()
        for key, new_val in patch.items():
            parts = key.split(".")
            o: Any = old_cfg
            for p in parts:
                o = o[p] if isinstance(o, dict) else getattr(o, p)
            changes[key] = [o, new_val]
        self.store.update_config(self.exp_id, self.config)
        self._emit_event("live_patch", {"changes": changes})
        _put(self.out_q, ("config", new_cfg))

    def _train_step(self) -> dict[str, float]:
        row = _sanitize_metrics(self.trainer.train_iteration())
        now = time.perf_counter()
        is_eval = "eval_score_mean" in row
        self.pending_metric = row
        should_write = is_eval or (now - self.last_metric_write_t) >= self.metric_interval
        if should_write and self.pending_metric is not None:
            self._flush_metric(self.pending_metric)
            self.last_metric_write_t = now
            self.pending_metric = None
        return row

    def _maybe_save_best(self, row: dict[str, float]) -> None:
        if "eval_score_mean" not in row:
            return
        score = float(row["eval_score_mean"])
        if score <= self.best_eval:
            return
        self.best_eval = score
        self._save_best()
        meta = self.store.read_meta(self.exp_id)
        meta["best_eval_score"] = self.best_eval
        self.store.write_meta(self.exp_id, meta)
        self._emit_event("best", {"eval_score_mean": self.best_eval})

    def _maybe_broadcast_weights(self, now: float) -> None:
        if now - self.last_weights_t < 1.0:
            return
        self._broadcast_weights()
        self.last_weights_t = now

    def _broadcast_weights(self) -> None:
        self.weight_version += 1
        net_sd = _cpu_state_dict(self.trainer.policy_net().state_dict())
        _put(
            self.out_q,
            ("weights", self.weight_version, net_sd, self.trainer.env_steps),
            droppable=True,
        )

    def _maybe_checkpoint(self, now: float) -> None:
        if now - self.last_ckpt_t < 60.0:
            return
        self._save_latest()
        self.last_ckpt_t = now

    def _maybe_update_meta(self, now: float) -> None:
        if now - self.last_elapsed_t < 1.0:
            return
        self._write_elapsed_meta()
        self.last_elapsed_t = now

    def _write_elapsed_meta(self) -> None:
        meta = self.store.read_meta(self.exp_id)
        meta["elapsed_s"] = self.base_elapsed + (time.perf_counter() - self.started_wall)
        meta["env_steps"] = self.trainer.env_steps
        self.store.write_meta(self.exp_id, meta)

    def _flush_pending_metric(self) -> None:
        if self.pending_metric is not None:
            self._flush_metric(self.pending_metric)
            self.pending_metric = None

    def _flush_metric(self, row: dict[str, float]) -> None:
        self.store.append_metric(self.exp_id, row)
        _put(self.out_q, ("metrics", row))

    def _finish_max_steps(self) -> None:
        self._flush_pending_metric()
        self._save_latest()
        self.store.update_status(self.exp_id, "finished")
        self._emit_event("finish", {})
        self._put_status("finished")

    def _finish_stop(self) -> None:
        self._flush_pending_metric()
        self._save_latest()
        self._write_elapsed_meta()
        self.store.update_status(self.exp_id, "stopped")
        self._emit_event("stop", {})
        self._put_status("stopped")

    def _handle_error(self, exc: BaseException) -> None:
        tb = traceback.format_exc()
        msg = f"{type(exc).__name__}: {exc}"
        log.exception("Worker %s crashed: %s", self.exp_id, msg)
        try:
            env_steps = int(getattr(self.trainer, "env_steps", 0) or 0)
            self.store.update_status(self.exp_id, "error", error=msg)
            self._emit_event("error", {"message": msg, "traceback": tb}, env_steps=env_steps)
            _put(self.out_q, ("error", msg, tb))
            self._put_status("error", msg)
        except Exception:
            log.exception("Failed to report worker error for %s", self.exp_id)

    def _emit_event(
        self,
        event_type: str,
        data: dict[str, Any],
        *,
        env_steps: int | None = None,
    ) -> None:
        steps = env_steps if env_steps is not None else int(getattr(self.trainer, "env_steps", 0) or 0)
        ev = self.store.append_event(
            self.exp_id,
            {"t": time.time(), "env_steps": steps, "type": event_type, "data": data},
        )
        _put(self.out_q, ("event", ev))

    def _put_status(self, status: str, error: str | None = None) -> None:
        if error is not None:
            _put(self.out_q, ("status", status, error))
        else:
            _put(self.out_q, ("status", status))

    def _save_latest(self) -> None:
        path = self.store.ckpt_path(self.exp_id, "latest")
        save_checkpoint(
            path,
            trainer_state=self.trainer.state_dict(),
            config=self.config,
            meta={"env_steps": self.trainer.env_steps, "iteration": self.trainer.iteration},
        )

    def _save_best(self) -> None:
        path = self.store.ckpt_path(self.exp_id, "best")
        save_checkpoint(
            path,
            trainer_state=self.trainer.state_dict(),
            config=self.config,
            meta={
                "env_steps": self.trainer.env_steps,
                "iteration": self.trainer.iteration,
                "best": True,
            },
        )


def run_worker(
    exp_dir: str,
    config_dict: dict[str, Any],
    resume: bool,
    cmd_q: Any,
    out_q: Any,
) -> None:
    """Entry point for a spawned training process."""
    WorkerSession(exp_dir, config_dict, resume, cmd_q, out_q).run()
