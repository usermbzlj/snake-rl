"""
训练 run 目录的统一元数据：GUI 与 Web 监控共用，避免状态/局数判断不一致。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# 若日志或 state 在该时间内有更新，视为「训练中」（无 summary 时）
STALE_AFTER_SECONDS = 120.0

STATUS_LABELS: dict[str, str] = {
    "completed": "完成",
    "running": "训练中",
    "interrupted": "中断",
    "empty": "空",
    "unknown": "未知",
}


def json_load_dict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _safe_mtime(path: Path) -> float | None:
    try:
        if path.exists() and path.is_file():
            return float(path.stat().st_mtime)
    except OSError:
        pass
    return None


def _max_mtime(paths: list[Path]) -> float | None:
    best: float | None = None
    for p in paths:
        m = _safe_mtime(p)
        if m is None:
            continue
        best = m if best is None else max(best, m)
    return best


def _checkpoint_mtimes(run_dir: Path) -> float | None:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    best: float | None = None
    try:
        for p in ckpt_dir.iterdir():
            if p.suffix.lower() == ".pt":
                m = _safe_mtime(p)
                if m is not None:
                    best = m if best is None else max(best, m)
    except OSError:
        return best
    return best


def read_last_lines(path: Path, max_lines: int) -> list[str]:
    if max_lines <= 0 or not path.exists():
        return []
    with path.open("rb") as f:
        f.seek(0, 2)
        file_size = f.tell()
        block_size = 4096
        data = b""
        lines: list[bytes] = []
        while file_size > 0 and len(lines) <= max_lines:
            read_size = min(block_size, file_size)
            file_size -= read_size
            f.seek(file_size)
            data = f.read(read_size) + data
            lines = data.splitlines()
    tail = lines[-max_lines:] if len(lines) > max_lines else lines
    return [line.decode("utf-8", errors="replace") for line in tail]


def parse_last_jsonl_row(path: Path) -> dict[str, Any] | None:
    for line in reversed(read_last_lines(path, max_lines=32)):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                return row
        except Exception:
            continue
    return None


def parse_last_csv_row(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            last: dict[str, str] | None = None
            for row in reader:
                last = row
    except Exception:
        return None
    if not last:
        return None
    best_avg_raw = last.get("best_avg_reward")
    best_avg_reward = None
    if best_avg_raw not in (None, ""):
        try:
            best_avg_reward = float(best_avg_raw)
        except (TypeError, ValueError):
            best_avg_reward = None

    return {
        "episode": int(last.get("episode", "0") or 0),
        "reward": float(last.get("reward", "0") or 0),
        "avg_reward": float(last.get("avg_reward", "0") or 0),
        "best_avg_reward": best_avg_reward,
        "steps": int(last.get("steps", "0") or 0),
        "foods": int(last.get("foods", "0") or 0),
        "score": int(last.get("score", "0") or 0),
        "epsilon": float(last.get("epsilon", "0") or 0),
        "stage_index": last.get("stage_index"),
    }


def latest_episode_row(run_dir: Path) -> dict[str, Any] | None:
    logs = run_dir / "logs"
    j = logs / "episodes.jsonl"
    if j.exists():
        r = parse_last_jsonl_row(j)
        if r:
            return r
    return parse_last_csv_row(logs / "episodes.csv")


def _parse_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def best_avg_reward_from_logs(run_dir: Path, *, last_row: dict[str, Any] | None = None) -> float | None:
    if last_row is not None:
        best_from_last = _parse_float(last_row.get("best_avg_reward"))
        if best_from_last is not None:
            return best_from_last

    logs_dir = run_dir / "logs"
    jsonl_path = logs_dir / "episodes.jsonl"
    best: float | None = None
    if jsonl_path.exists():
        try:
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(row, dict):
                        continue
                    candidate = _parse_float(row.get("best_avg_reward"))
                    if candidate is None:
                        candidate = _parse_float(row.get("avg_reward"))
                    if candidate is None:
                        continue
                    best = candidate if best is None else max(best, candidate)
        except Exception:
            pass
        if best is not None:
            return best

    csv_path = logs_dir / "episodes.csv"
    if csv_path.exists():
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    candidate = _parse_float(row.get("best_avg_reward"))
                    if candidate is None:
                        candidate = _parse_float(row.get("avg_reward"))
                    if candidate is None:
                        continue
                    best = candidate if best is None else max(best, candidate)
        except Exception:
            pass
    return best


def activity_timestamps(run_dir: Path) -> dict[str, float | None]:
    logs = run_dir / "logs"
    jsonl = logs / "episodes.jsonl"
    csv_path = logs / "episodes.csv"
    summary_p = logs / "summary.json"
    state_p = run_dir / "state" / "training.pt"
    return {
        "jsonl": _safe_mtime(jsonl),
        "csv": _safe_mtime(csv_path),
        "summary": _safe_mtime(summary_p),
        "training_state": _safe_mtime(state_p),
        "checkpoints": _checkpoint_mtimes(run_dir),
    }


def last_activity_ts(run_dir: Path) -> float | None:
    ts = activity_timestamps(run_dir)
    candidates = [v for v in ts.values() if v is not None]
    if not candidates:
        return None
    return max(candidates)


@dataclass(frozen=True)
class RunMeta:
    name: str
    model_type: str
    status_key: str
    status_label: str
    episodes_logged: int
    best_avg_reward: float | None
    best_avg_reward_live: float | None
    last_activity_at: str
    last_activity_ts: float | None
    has_best_checkpoint: bool
    has_latest_checkpoint: bool
    has_training_state: bool
    latest_metrics: dict[str, Any]


def build_run_meta(run_dir: Path, *, now: float | None = None) -> RunMeta:
    import time

    t_now = time.time() if now is None else float(now)
    name = run_dir.name

    run_cfg = json_load_dict(run_dir / "run_config.json")
    if not run_cfg:
        run_cfg = json_load_dict(run_dir / "train_config.json")
    model_type = str(run_cfg.get("model_type", "-"))

    logs_dir = run_dir / "logs"
    summary_path = logs_dir / "summary.json"
    summary = json_load_dict(summary_path)
    summary_ok = bool(summary)

    has_best = (run_dir / "checkpoints" / "best.pt").exists()
    has_latest = (run_dir / "checkpoints" / "latest.pt").exists()
    has_training_state = (run_dir / "state" / "training.pt").exists()
    has_checkpoint = has_best or has_latest or has_training_state

    last_row = latest_episode_row(run_dir)
    episodes_from_log = int(last_row["episode"]) if last_row and "episode" in last_row else 0

    summary_eps = 0
    best_from_summary: float | None = None
    if summary_ok:
        try:
            summary_eps = int(summary.get("episodes", 0))
        except Exception:
            summary_eps = 0
        bar = summary.get("best_avg_reward")
        if isinstance(bar, (int, float)):
            best_from_summary = float(bar)

    activity_ts = activity_timestamps(run_dir)
    act_ts = last_activity_ts(run_dir)
    last_activity_at = (
        datetime.fromtimestamp(act_ts).strftime("%Y-%m-%d %H:%M:%S") if act_ts else "-"
    )

    if summary_ok:
        status_key = "completed"
        episodes_logged = max(summary_eps, episodes_from_log)
        best_live = best_from_summary if best_from_summary is not None else best_avg_reward_from_logs(
            run_dir, last_row=last_row
        )
    else:
        runtime_ts = _max_mtime(
            [
                logs_dir / "episodes.jsonl",
                logs_dir / "episodes.csv",
                run_dir / "state" / "training.pt",
            ]
        )
        fresh_runtime = runtime_ts is not None and (t_now - runtime_ts) <= STALE_AFTER_SECONDS
        jsonl_path = logs_dir / "episodes.jsonl"
        jsonl_nonempty = False
        try:
            jsonl_nonempty = jsonl_path.is_file() and jsonl_path.stat().st_size > 0
        except OSError:
            jsonl_nonempty = False
        has_log_signal = episodes_from_log > 0 or jsonl_nonempty or (logs_dir / "episodes.csv").exists()
        has_runtime_signal = has_log_signal or activity_ts["training_state"] is not None

        if fresh_runtime and has_runtime_signal:
            status_key = "running"
        elif has_checkpoint or has_log_signal:
            status_key = "interrupted"
        else:
            status_key = "empty"

        episodes_logged = episodes_from_log
        best_live = best_avg_reward_from_logs(run_dir, last_row=last_row)

    if status_key == "completed":
        best_avg_reward_live = best_from_summary
    else:
        best_avg_reward_live = best_live

    status_label = STATUS_LABELS.get(status_key, STATUS_LABELS["unknown"])

    latest_metrics: dict[str, Any] = {}
    if last_row:
        for k in ("episode", "reward", "avg_reward", "epsilon", "stage_index"):
            if k in last_row:
                latest_metrics[k] = last_row[k]

    return RunMeta(
        name=name,
        model_type=model_type,
        status_key=status_key,
        status_label=status_label,
        episodes_logged=episodes_logged,
        best_avg_reward=best_from_summary,
        best_avg_reward_live=best_avg_reward_live,
        last_activity_at=last_activity_at,
        last_activity_ts=act_ts,
        has_best_checkpoint=has_best,
        has_latest_checkpoint=has_latest,
        has_training_state=has_training_state,
        latest_metrics=latest_metrics,
    )


def run_meta_to_api_item(meta: RunMeta) -> dict[str, Any]:
    return {
        "name": meta.name,
        "model_type": meta.model_type,
        "status": meta.status_label,
        "status_key": meta.status_key,
        "episodes": meta.episodes_logged,
        "best_avg_reward": meta.best_avg_reward if meta.status_key == "completed" else meta.best_avg_reward_live,
        "best_avg_reward_final": meta.best_avg_reward,
        "best_avg_reward_live": meta.best_avg_reward_live,
        "updated_at": meta.last_activity_at,
        "last_activity_ts": meta.last_activity_ts,
        "has_best_checkpoint": meta.has_best_checkpoint,
        "has_latest_checkpoint": meta.has_latest_checkpoint,
        "has_training_state": meta.has_training_state,
        "latest_metrics": meta.latest_metrics,
    }


def list_run_metas_sorted(runs_dir: Path) -> list[RunMeta]:
    if not runs_dir.exists():
        return []
    dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    metas = [build_run_meta(p) for p in dirs]
    metas.sort(key=lambda m: m.last_activity_ts or 0.0, reverse=True)
    return metas


def run_meta_to_gui_row(meta: RunMeta) -> dict[str, str]:
    eps = str(meta.episodes_logged) if meta.episodes_logged else "-"
    if meta.status_key == "completed" and meta.best_avg_reward is not None:
        br = f"{meta.best_avg_reward:.3f}"
    elif meta.best_avg_reward_live is not None:
        br = f"{meta.best_avg_reward_live:.3f}"
    else:
        br = "-"
    ck_parts: list[str] = []
    if meta.has_best_checkpoint:
        ck_parts.append("best.pt")
    if meta.has_latest_checkpoint:
        ck_parts.append("latest.pt")
    if meta.has_training_state:
        ck_parts.append("training.pt")
    return {
        "name": meta.name,
        "model": meta.model_type,
        "episodes": eps,
        "best_reward": br,
        "status": meta.status_label,
        "status_key": meta.status_key,
        "updated": meta.last_activity_at,
        "badges": " ".join(ck_parts) if ck_parts else "-",
    }
