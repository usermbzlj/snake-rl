"""Checkpoint save/load for trainers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from snake_rl.core.config import ExperimentConfig


def save_checkpoint(
    path: str | Path,
    *,
    trainer_state: dict[str, Any],
    config: ExperimentConfig,
    meta: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trainer": trainer_state,
        "config": config.model_dump(),
        "meta": meta or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = torch.load(path, map_location="cpu", weights_only=False)
    if "config" in data and not isinstance(data["config"], ExperimentConfig):
        data["config"] = ExperimentConfig.model_validate(data["config"])
    return data
