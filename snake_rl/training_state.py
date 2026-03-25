"""完整训练状态（网络 + 回放 + 元数据），与模型 checkpoint 区分。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .agent import DDQNAgent
from .config import TrainConfig, train_config_from_dict
from .replay_buffer import ReplayBuffer
from .versions import TRAINING_STATE_SCHEMA_VERSION


@dataclass(slots=True)
class LoadedTrainingState:
    cfg: TrainConfig
    agent_payload: dict[str, Any]
    replay_state: dict[str, Any]
    meta: dict[str, Any]


def _config_to_storable_dict(cfg: TrainConfig) -> dict[str, Any]:
    d = asdict(cfg)
    d["output_root"] = str(cfg.output_root)
    return d


def save_training_state(
    path: Path,
    *,
    agent: DDQNAgent,
    replay: ReplayBuffer,
    cfg: TrainConfig,
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": TRAINING_STATE_SCHEMA_VERSION,
        "train_config": _config_to_storable_dict(cfg),
        "agent_checkpoint": agent.checkpoint_payload(),
        "replay": replay.state_dict(),
        "meta": dict(meta),
    }
    torch.save(payload, path)


def load_training_state(path: Path, device: torch.device) -> LoadedTrainingState:
    data = torch.load(path, map_location=device, weights_only=False)
    ver = int(data.get("schema_version", 0))
    if ver != TRAINING_STATE_SCHEMA_VERSION:
        raise ValueError(
            f"不支持的训练状态版本: {ver}，当前需要 {TRAINING_STATE_SCHEMA_VERSION}"
        )
    cfg = train_config_from_dict(data["train_config"])
    replay_raw = data["replay"]
    if not isinstance(replay_raw, dict):
        raise ValueError("训练状态缺少 replay")
    return LoadedTrainingState(
        cfg=cfg,
        agent_payload=data["agent_checkpoint"],
        replay_state=replay_raw,
        meta=dict(data.get("meta") or {}),
    )


def training_state_path(run_dir: Path) -> Path:
    return run_dir / "state" / "training.pt"
