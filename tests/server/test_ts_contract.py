"""Keep the hand-written TypeScript config types in sync with the pydantic models."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import BaseModel

from snake_rl.core.config import (
    DQNConfig,
    EnvConfig,
    ExperimentConfig,
    ModelConfig,
    PPOConfig,
    RewardConfig,
    RunConfig,
)

TYPES_TS = Path(__file__).resolve().parents[2] / "web" / "src" / "api" / "types.ts"

MODELS: dict[str, type[BaseModel]] = {
    "EnvConfig": EnvConfig,
    "RewardConfig": RewardConfig,
    "ModelConfig": ModelConfig,
    "PPOConfig": PPOConfig,
    "DQNConfig": DQNConfig,
    "RunConfig": RunConfig,
    "ExperimentConfig": ExperimentConfig,
}


def _ts_interface_keys(source: str, name: str) -> set[str]:
    match = re.search(rf"export interface {name} \{{(.*?)\n\}}", source, re.S)
    assert match, f"types.ts 中缺少 interface {name}"
    return set(re.findall(r"^\s*(\w+)\??:", match.group(1), re.M))


@pytest.mark.parametrize("name", list(MODELS))
def test_ts_interface_matches_pydantic(name: str) -> None:
    ts_keys = _ts_interface_keys(TYPES_TS.read_text(encoding="utf-8"), name)
    assert ts_keys == set(MODELS[name].model_fields)
