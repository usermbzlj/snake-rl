"""Training worker process (multiprocessing spawn entry)."""

from __future__ import annotations

import contextlib
import queue
import time
import traceback
from pathlib import Path
from typing import Any

from snake_rl.core.checkpoint import load_checkpoint, save_checkpoint
from snake_rl.core.config import ExperimentConfig
from snake_rl.core.trainer import make_trainer, set_seed
from snake_rl.lab.storage import ExperimentStore


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


def _save_latest(store: ExperimentStore, exp_id: str, trainer: Any, config: ExperimentConfig) -> None:
    path = store.ckpt_path(exp_id, "latest")
    save_checkpoint(
        path,
        trainer_state=trainer.state_dict(),
        config=config,
        meta={"env_steps": trainer.env_steps, "iteration": trainer.iteration},
    )


def _save_best(store: ExperimentStore, exp_id: str, trainer: Any, config: ExperimentConfig) -> None:
    path = store.ckpt_path(exp_id, "best")
    save_checkpoint(
        path,
        trainer_state=trainer.state_dict(),
        config=config,
        meta={"env_steps": trainer.env_steps, "iteration": trainer.iteration, "best": True},
    )


def run_worker(
    exp_dir: str,
    config_dict: dict[str, Any],
    resume: bool,
    cmd_q: Any,
    out_q: Any,
) -> None:
    """Entry point for a spawned training process."""
    exp_path = Path(exp_dir)
    exp_id = exp_path.name
    store = ExperimentStore(exp_path.parent)
    config = ExperimentConfig.model_validate(config_dict)
    paused = False
    weight_version = 0
    best_eval = float("-inf")
    last_ckpt_t = time.perf_counter()
    last_weights_t = 0.0
    last_metric_write_t = 0.0
    last_elapsed_t = 0.0
    pending_metric: dict[str, float] | None = None
    metric_interval = 0.5  # ≤ ~2 rows/s
    started_wall = time.perf_counter()
    base_elapsed = 0.0

    try:
        meta = store.read_meta(exp_id)
        base_elapsed = float(meta.get("elapsed_s") or 0.0)
        if meta.get("best_eval_score") is not None:
            best_eval = float(meta["best_eval_score"])

        set_seed(config.run.seed)
        trainer = make_trainer(config)

        if resume:
            latest = store.ckpt_path(exp_id, "latest")
            if latest.is_file():
                data = load_checkpoint(latest)
                trainer.load_state_dict(data["trainer"])
                # Prefer live config from disk (may have been live-patched)
                config = ExperimentConfig.model_validate(meta["config"])
                trainer.config = config
                from snake_rl.core.trainer import reward_weights_tensor

                device = getattr(trainer, "device", None)
                if device is not None and hasattr(trainer, "weights"):
                    trainer.weights = reward_weights_tensor(config, device)  # type: ignore[attr-defined]
                best_eval = max(best_eval, float(trainer.state_dict().get("best_eval", float("-inf"))))

        store.update_status(exp_id, "running", clear_error=True)
        ev = store.append_event(
            exp_id,
            {"t": time.time(), "env_steps": trainer.env_steps, "type": "start", "data": {"resume": resume}},
        )
        _put(out_q, ("status", "running"))
        _put(out_q, ("event", ev))

        # Initial weights for viewers
        weight_version += 1
        net_sd = _cpu_state_dict(trainer.policy_net().state_dict())
        _put(out_q, ("weights", weight_version, net_sd, trainer.env_steps), droppable=True)
        last_weights_t = time.perf_counter()

        stop_requested = False
        while not stop_requested:
            # Drain commands
            while True:
                try:
                    cmd = cmd_q.get_nowait()
                except queue.Empty:
                    break
                kind = cmd[0] if isinstance(cmd, tuple | list) else cmd
                if kind == "pause":
                    paused = True
                    store.update_status(exp_id, "paused")
                    ev = store.append_event(
                        exp_id,
                        {"t": time.time(), "env_steps": trainer.env_steps, "type": "pause", "data": {}},
                    )
                    _put(out_q, ("status", "paused"))
                    _put(out_q, ("event", ev))
                elif kind == "resume":
                    paused = False
                    store.update_status(exp_id, "running")
                    ev = store.append_event(
                        exp_id,
                        {"t": time.time(), "env_steps": trainer.env_steps, "type": "resume", "data": {}},
                    )
                    _put(out_q, ("status", "running"))
                    _put(out_q, ("event", ev))
                elif kind == "stop":
                    stop_requested = True
                    break
                elif kind == "live_patch":
                    patch = cmd[1] if len(cmd) > 1 else {}
                    old_cfg = trainer.config.model_dump()
                    trainer.apply_live(patch)
                    config = trainer.config
                    changes: dict[str, list[Any]] = {}
                    new_cfg = config.model_dump()
                    for key, new_val in patch.items():
                        parts = key.split(".")
                        o: Any = old_cfg
                        for p in parts:
                            o = o[p] if isinstance(o, dict) else getattr(o, p)
                        changes[key] = [o, new_val]
                    store.update_config(exp_id, config)
                    ev = store.append_event(
                        exp_id,
                        {
                            "t": time.time(),
                            "env_steps": trainer.env_steps,
                            "type": "live_patch",
                            "data": {"changes": changes},
                        },
                    )
                    _put(out_q, ("event", ev))
                    _put(out_q, ("config", new_cfg))

            if stop_requested:
                break

            if paused:
                time.sleep(0.05)
                continue

            row = trainer.train_iteration()
            # Sanitize NaN
            clean: dict[str, float] = {}
            for k, v in row.items():
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if fv != fv or fv == float("inf") or fv == float("-inf"):
                    continue
                clean[k] = fv
            row = clean

            now = time.perf_counter()
            is_eval = "eval_score_mean" in row
            pending_metric = row
            should_write = is_eval or (now - last_metric_write_t) >= metric_interval
            if should_write and pending_metric is not None:
                store.append_metric(exp_id, pending_metric)
                _put(out_q, ("metrics", pending_metric))
                last_metric_write_t = now
                pending_metric = None

            if is_eval:
                score = float(row["eval_score_mean"])
                if score > best_eval:
                    best_eval = score
                    _save_best(store, exp_id, trainer, config)
                    meta = store.read_meta(exp_id)
                    meta["best_eval_score"] = best_eval
                    store.write_meta(exp_id, meta)
                    ev = store.append_event(
                        exp_id,
                        {
                            "t": time.time(),
                            "env_steps": trainer.env_steps,
                            "type": "best",
                            "data": {"eval_score_mean": best_eval},
                        },
                    )
                    _put(out_q, ("event", ev))

            if now - last_weights_t >= 1.0:
                weight_version += 1
                net_sd = _cpu_state_dict(trainer.policy_net().state_dict())
                _put(out_q, ("weights", weight_version, net_sd, trainer.env_steps), droppable=True)
                last_weights_t = now

            if now - last_ckpt_t >= 60.0:
                _save_latest(store, exp_id, trainer, config)
                last_ckpt_t = now

            if now - last_elapsed_t >= 1.0:
                meta = store.read_meta(exp_id)
                meta["elapsed_s"] = base_elapsed + (time.perf_counter() - started_wall)
                meta["env_steps"] = trainer.env_steps
                store.write_meta(exp_id, meta)
                last_elapsed_t = now

            max_steps = int(config.run.max_env_steps or 0)
            if max_steps > 0 and trainer.env_steps >= max_steps:
                # Flush pending metric
                if pending_metric is not None:
                    store.append_metric(exp_id, pending_metric)
                    _put(out_q, ("metrics", pending_metric))
                _save_latest(store, exp_id, trainer, config)
                store.update_status(exp_id, "finished")
                ev = store.append_event(
                    exp_id,
                    {"t": time.time(), "env_steps": trainer.env_steps, "type": "finish", "data": {}},
                )
                _put(out_q, ("event", ev))
                _put(out_q, ("status", "finished"))
                return

        # Stop path
        if pending_metric is not None:
            store.append_metric(exp_id, pending_metric)
            _put(out_q, ("metrics", pending_metric))
        _save_latest(store, exp_id, trainer, config)
        meta = store.read_meta(exp_id)
        meta["elapsed_s"] = base_elapsed + (time.perf_counter() - started_wall)
        meta["env_steps"] = trainer.env_steps
        store.write_meta(exp_id, meta)
        store.update_status(exp_id, "stopped")
        ev = store.append_event(
            exp_id,
            {"t": time.time(), "env_steps": trainer.env_steps, "type": "stop", "data": {}},
        )
        _put(out_q, ("event", ev))
        _put(out_q, ("status", "stopped"))

    except Exception as exc:
        tb = traceback.format_exc()
        msg = f"{type(exc).__name__}: {exc}"
        try:
            store.update_status(exp_id, "error", error=msg)
            ev = store.append_event(
                exp_id,
                {
                    "t": time.time(),
                    "env_steps": 0,
                    "type": "error",
                    "data": {"message": msg, "traceback": tb},
                },
            )
            _put(out_q, ("event", ev))
            _put(out_q, ("error", msg, tb))
            _put(out_q, ("status", "error", msg))
        except Exception:
            pass
