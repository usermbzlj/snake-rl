from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..train_events import parse_event_line
from .state import RuntimeState

EPISODE_LINE_RE = re.compile(r"\[Episode\s+(\d+)\]")
STAGE_LINE_RE = re.compile(r"\[Stage\s+(\d+)\s+\|\s+Ep\s+(\d+)/(\d+)\]")
STAGE_HEADER_RE = re.compile(r"Curriculum Stage\s+(\d+)/(\d+)")
TOTAL_EPISODES_RE = re.compile(r"总局数上限[：:]\s*(\d+)")
AVG_REWARD_RE = re.compile(r"avg_reward=\s*([-+]?\d+(?:\.\d+)?)")
EPSILON_RE = re.compile(r"\beps=\s*([-+]?\d+(?:\.\d+)?)")


def apply_progress_event(state: RuntimeState, event: dict[str, Any]) -> None:
    etype = str(event.get("type", ""))
    if etype == "episode":
        episode = event.get("episode")
        if episode is not None:
            state.progress_current = max(state.progress_current, int(episode))
        total = event.get("episodes_total")
        if total:
            state.progress_total = max(state.progress_total, int(total))
        if event.get("avg_reward") is not None:
            state.progress_avg_reward = float(event["avg_reward"])
        if event.get("epsilon") is not None:
            state.progress_epsilon = float(event["epsilon"])
        if event.get("stage_index") is not None:
            state.progress_stage = str(event["stage_index"])
    elif etype == "stage":
        stage_index = event.get("stage_index")
        stages_total = event.get("stages_total")
        if stage_index is not None:
            state.progress_stage = f"{stage_index}/{stages_total or '?'}"
    elif etype == "eval":
        if event.get("eval_reward") is not None:
            state.progress_avg_reward = float(event["eval_reward"])


def update_progress_from_line(state: RuntimeState, line: str) -> None:
    event = parse_event_line(line)
    if event is not None:
        apply_progress_event(state, event)
        return

    total_match = TOTAL_EPISODES_RE.search(line)
    if total_match:
        try:
            state.progress_total = max(state.progress_total, int(total_match.group(1)))
        except Exception:
            pass

    episode_match = EPISODE_LINE_RE.search(line)
    if episode_match:
        try:
            state.progress_current = max(state.progress_current, int(episode_match.group(1)))
        except Exception:
            pass

    stage_match = STAGE_LINE_RE.search(line)
    if stage_match:
        try:
            stage_index = int(stage_match.group(1))
            stage_episode = int(stage_match.group(2))
            stage_total = int(stage_match.group(3))
            prefix = state.stage_prefix.get(stage_index)
            if prefix is not None:
                state.progress_current = max(
                    state.progress_current,
                    prefix + stage_episode,
                )
            state.progress_stage = f"{stage_index} ({stage_episode}/{stage_total})"
        except Exception:
            pass
    else:
        stage_header_match = STAGE_HEADER_RE.search(line)
        if stage_header_match:
            state.progress_stage = f"{stage_header_match.group(1)} (准备中)"

    avg_match = AVG_REWARD_RE.search(line)
    if avg_match:
        try:
            state.progress_avg_reward = float(avg_match.group(1))
        except Exception:
            pass

    eps_match = EPSILON_RE.search(line)
    if eps_match:
        try:
            state.progress_epsilon = float(eps_match.group(1))
        except Exception:
            pass


def progress_payload(state: RuntimeState) -> dict[str, Any]:
    total = state.progress_total
    cur = state.progress_current
    pct = 0.0
    if total > 0:
        pct = max(0.0, min(100.0, 100.0 * cur / total))
    run_name = None
    if state.training_run_dir:
        try:
            run_name = Path(state.training_run_dir).name
        except Exception:
            run_name = None
    return {
        "type": "progress",
        "episode": cur,
        "total": total,
        "percent": pct,
        "avg_reward": state.progress_avg_reward,
        "epsilon": state.progress_epsilon,
        "stage": state.progress_stage,
        "started_at": state.train_started_at,
        "training_run": run_name,
    }


def status_payload(state: RuntimeState) -> dict[str, Any]:
    run_name = None
    if state.training_run_dir:
        try:
            run_name = Path(state.training_run_dir).name
        except Exception:
            run_name = None
    return {
        "type": "status",
        "training": state.training_alive(),
        "monitor": state.monitor_alive(),
        "infer": state.infer_alive(),
        "monitor_port": state.monitor_port,
        "infer_port": state.inference_port,
        "estimating": state.estimating,
        "training_run": run_name,
        "train_started_at": state.train_started_at,
    }
