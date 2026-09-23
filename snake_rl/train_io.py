"""Run directory, logging, and training-state persistence."""

from __future__ import annotations

import csv
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Protocol

from .agent import DDQNAgent
from .config import TrainConfig
from .replay_buffer import ReplayBuffer
from .training_state import save_training_state, training_state_path
from .versions import (
    FEATURE_SCHEMA_VERSION,
    MODEL_CHECKPOINT_SCHEMA_VERSION,
    TRAINING_STATE_SCHEMA_VERSION,
)
from .viz import LivePlotter


def prepare_run_dir(cfg: TrainConfig) -> Path:
    run_name = cfg.run_name
    if run_name == "default":
        run_name = datetime.now().strftime("ddqn_%Y%m%d_%H%M%S")
    run_dir = cfg.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "state").mkdir(parents=True, exist_ok=True)
    payload = asdict(cfg)
    payload["output_root"] = str(cfg.output_root)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    (run_dir / "run_config.json").write_text(text, encoding="utf-8")
    manifest = {
        "run_manifest_schema_version": 1,
        "model_checkpoint_schema_version": MODEL_CHECKPOINT_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "training_state_schema_version": TRAINING_STATE_SCHEMA_VERSION,
        "run_name": run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return run_dir


def persist_training_state(
    run_dir: Path,
    agent: DDQNAgent,
    replay: ReplayBuffer,
    cfg: TrainConfig,
    *,
    global_step: int,
    next_episode: int,
    best_avg_reward: float,
    completed_episodes: int | None = None,
    worker_episode_counters: list[int] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    done = int(completed_episodes) if completed_episodes is not None else int(next_episode) - 1
    meta: dict[str, Any] = {
        "global_step": int(global_step),
        "next_episode": int(next_episode),
        "completed_episodes": done,
        "best_avg_reward": float(best_avg_reward),
    }
    if worker_episode_counters is not None:
        meta["worker_episode_counters"] = [int(x) for x in worker_episode_counters]
    if extra_meta:
        meta.update(extra_meta)
    save_training_state(
        training_state_path(run_dir),
        agent=agent,
        replay=replay,
        cfg=cfg,
        meta=meta,
    )


def append_episode_csv_incremental(
    path: Path,
    episode_rows: list[dict[str, Any]],
    committed: int,
) -> int:
    if committed >= len(episode_rows):
        return committed
    new_rows = episode_rows[committed:]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(new_rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(new_rows)
    return len(episode_rows)


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_logged_episode_row(raw: dict[str, Any]) -> dict[str, Any]:
    row = dict(raw)
    for key in ("episode", "global_step", "steps", "foods", "score", "win", "board_size"):
        value = _coerce_int(row.get(key))
        if value is not None:
            row[key] = value
    for key in (
        "reward",
        "avg_reward",
        "best_avg_reward",
        "avg_steps",
        "epsilon",
        "loss",
        "q_mean",
        "target_q_mean",
        "eval_reward",
    ):
        value = _coerce_float(row.get(key))
        if value is not None:
            row[key] = value
    stage_index = row.get("stage_index")
    if stage_index in ("", None):
        row["stage_index"] = None
    else:
        normalized_stage = _coerce_int(stage_index)
        row["stage_index"] = normalized_stage if normalized_stage is not None else stage_index
    terminal_reason = row.get("terminal_reason")
    if terminal_reason is not None:
        row["terminal_reason"] = str(terminal_reason)
    return row


def load_episode_history_snapshot(run_dir: Path, moving_avg_window: int) -> dict[str, Any]:
    maxlen = max(1, int(moving_avg_window))
    reward_window: deque[float] = deque(maxlen=maxlen)
    steps_window: deque[float] = deque(maxlen=maxlen)
    terminal_reason_counter: dict[str, int] = {}
    last_row: dict[str, Any] | None = None
    episodes_logged = 0

    logs_dir = run_dir / "logs"
    jsonl_path = logs_dir / "episodes.jsonl"
    csv_path = logs_dir / "episodes.csv"

    source_rows: list[dict[str, Any]] = []
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    source_rows.append(payload)
    elif csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            source_rows.extend(csv.DictReader(f))

    for raw in source_rows:
        row = _normalize_logged_episode_row(raw)
        last_row = row
        episode = _coerce_int(row.get("episode"))
        if episode is not None:
            episodes_logged = max(episodes_logged, episode)
        reward = _coerce_float(row.get("reward"))
        if reward is not None:
            reward_window.append(reward)
        steps = _coerce_float(row.get("steps"))
        if steps is not None:
            steps_window.append(steps)
        terminal_reason = str(row.get("terminal_reason", "")).strip()
        if terminal_reason:
            terminal_reason_counter[terminal_reason] = terminal_reason_counter.get(terminal_reason, 0) + 1

    return {
        "episodes_logged": episodes_logged,
        "last_row": last_row,
        "reward_window": list(reward_window),
        "steps_window": list(steps_window),
        "terminal_reason_counter": terminal_reason_counter,
    }


class ScalarWriter(Protocol):
    def add_scalar(self, tag: str, scalar_value: Any, global_step: int | None = None) -> Any: ...

    def close(self) -> None: ...


def create_scalar_writer(
    cfg: TrainConfig, run_dir: Path, *, purge_step: int | None = None
) -> ScalarWriter | None:
    if not cfg.tensorboard:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter
    except Exception as exc:  # noqa: BLE001
        raise ImportError("未安装 tensorboard，请执行 uv sync 安装依赖。") from exc
    kwargs: dict[str, Any] = {"log_dir": str(run_dir)}
    if purge_step is not None:
        kwargs["purge_step"] = purge_step
    return TorchSummaryWriter(**kwargs)


def maybe_write_episode(
    *,
    writer: ScalarWriter | None,
    plotter: LivePlotter,
    jsonl_file: Any,
    row: dict[str, Any],
    terminal_reason_counter: dict[str, int],
    tensorboard_log_interval: int,
    jsonl_flush_interval: int,
) -> None:
    episode = int(row["episode"])
    should_log_tensorboard = episode == 1 or episode % max(1, int(tensorboard_log_interval)) == 0
    should_flush_jsonl = episode % max(1, int(jsonl_flush_interval)) == 0

    if jsonl_file is not None:
        jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        if should_flush_jsonl:
            jsonl_file.flush()

    if writer is not None and should_log_tensorboard:
        writer.add_scalar("episode/reward", row["reward"], episode)
        writer.add_scalar("episode/avg_reward", row["avg_reward"], episode)
        writer.add_scalar("episode/steps", row["steps"], episode)
        writer.add_scalar("episode/avg_steps", row["avg_steps"], episode)
        writer.add_scalar("episode/foods", row["foods"], episode)
        writer.add_scalar("episode/score", row["score"], episode)
        writer.add_scalar("episode/win", row["win"], episode)
        writer.add_scalar("episode/board_size", row["board_size"], episode)
        writer.add_scalar("train/epsilon", row["epsilon"], episode)
        if row.get("loss") is not None:
            writer.add_scalar("train/loss", row["loss"], episode)
        if row.get("q_mean") is not None:
            writer.add_scalar("train/q_mean", row["q_mean"], episode)
        if row.get("target_q_mean") is not None:
            writer.add_scalar("train/target_q_mean", row["target_q_mean"], episode)
        if row.get("eval_reward") is not None:
            writer.add_scalar("eval/reward", row["eval_reward"], episode)
        writer.add_scalar(
            f"terminal_reason/{row['terminal_reason']}",
            terminal_reason_counter[row["terminal_reason"]],
            episode,
        )
        if row.get("stage_index") is not None:
            writer.add_scalar("curriculum/stage_index", row["stage_index"], episode)

    plotter.update(
        episode=int(row["episode"]),
        reward=float(row["reward"]),
        steps=int(row["steps"]),
        foods=int(row["foods"]),
        epsilon=float(row["epsilon"]),
        loss=row.get("loss"),
    )


def finalize_run(
    *,
    cfg: TrainConfig,
    run_dir: Path,
    episode_rows: list[dict[str, Any]],
    writer: ScalarWriter | None,
    plotter: LivePlotter,
    jsonl_file: Any,
    summary: dict[str, Any],
    csv_committed: int = 0,
) -> dict[str, Any]:
    if cfg.save_csv:
        append_episode_csv_incremental(
            run_dir / "logs" / "episodes.csv", episode_rows, csv_committed
        )
    with (run_dir / "logs" / "summary.json").open("w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))

    if jsonl_file is not None:
        jsonl_file.close()
    if writer is not None:
        writer.close()
    plotter.close()
    return summary
