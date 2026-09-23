"""Training entry: validate, route, and re-export helpers used by CLI / estimate / tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import TrainConfig, resolve_device, validate_config
from .obs import center_pad_chw, extract_model_inputs, hwc_to_chw
from .train_factory import (
    build_env_options,
    build_initial_env,
    create_agent,
    create_replay,
    get_agent_input_size,
    set_global_seed,
)
from .train_io import load_episode_history_snapshot
from .train_loop import infer_last_global_step_from_warm_checkpoint, run_unified_training


def run_training(
    cfg: TrainConfig,
    *,
    resume_state: Path | None = None,
    warm_start: Path | None = None,
    extra_episodes: int | None = None,
    warm_start_global_step: int | None = None,
) -> dict[str, Any]:
    validate_config(cfg)
    if resume_state is not None and warm_start is not None:
        raise ValueError("不能同时使用 resume_state 与 warm_start")
    return run_unified_training(
        cfg,
        resume_state=resume_state,
        warm_start=warm_start,
        extra_episodes=extra_episodes,
        warm_start_global_step=warm_start_global_step,
    )


__all__ = [
    "build_env_options",
    "build_initial_env",
    "center_pad_chw",
    "create_agent",
    "create_replay",
    "extract_model_inputs",
    "get_agent_input_size",
    "hwc_to_chw",
    "infer_last_global_step_from_warm_checkpoint",
    "load_episode_history_snapshot",
    "resolve_device",
    "run_training",
    "set_global_seed",
    "validate_config",
]


if __name__ == "__main__":
    raise SystemExit("请使用 `snake-rl train` 或 `python -m snake_rl.cli train`，不要直接运行 train.py。")
