"""从 checkpoint 或 run 目录解析训练元数据，供评估 / 推理 / warm-start 共用。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import EnvPreset


def checkpoint_run_dir(checkpoint: Path) -> Path | None:
    """runs/<name>/checkpoints/foo.pt -> runs/<name>。"""
    resolved = checkpoint.expanduser().resolve()
    parts = resolved.parts
    if "checkpoints" in parts:
        idx = parts.index("checkpoints")
        return Path(*parts[:idx])
    return None


def load_run_config_dict(run_dir: Path) -> dict[str, Any] | None:
    """优先 run_config.json，其次兼容旧 train_config.json。"""
    for name in ("run_config.json", "train_config.json"):
        p = run_dir / name
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
    return None


def env_preset_from_run_config(payload: dict[str, Any]) -> EnvPreset | None:
    raw = payload.get("env")
    if not isinstance(raw, dict):
        return None
    return EnvPreset(
        difficulty=str(raw.get("difficulty", "normal")),
        mode=str(raw.get("mode", "classic")),
        board_size=int(raw.get("board_size", 22)),
        enable_bonus_food=bool(raw.get("enable_bonus_food", False)),
        enable_obstacles=bool(raw.get("enable_obstacles", False)),
        allow_leveling=bool(raw.get("allow_leveling", False)),
        max_steps_without_food=int(raw.get("max_steps_without_food", 250)),
        seed=raw.get("seed"),
    )


def reward_weights_from_run_config(payload: dict[str, Any]) -> dict[str, float] | None:
    rw = payload.get("reward_weights")
    if not isinstance(rw, dict):
        return None
    out: dict[str, float] = {}
    for k, v in rw.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out or None


@dataclass(slots=True)
class RunContext:
    run_dir: Path | None
    env: EnvPreset | None
    reward_weights: dict[str, float] | None
    model_type: str | None
    local_patch_size: int | None

    @classmethod
    def from_checkpoint(cls, checkpoint: Path) -> RunContext:
        run_dir = checkpoint_run_dir(checkpoint)
        if run_dir is None:
            return cls(None, None, None, None, None)
        payload = load_run_config_dict(run_dir)
        if payload is None:
            return cls(run_dir, None, None, None, None)
        env = env_preset_from_run_config(payload)
        rw = reward_weights_from_run_config(payload)
        mt = payload.get("model_type")
        model_type = str(mt) if isinstance(mt, str) else None
        lp = payload.get("local_patch_size")
        local_patch = int(lp) if lp is not None else None
        return cls(run_dir, env, rw, model_type, local_patch)
