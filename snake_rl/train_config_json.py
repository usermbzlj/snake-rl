"""TrainConfig 与 JSON 文本互转（供 Web UI 与测试复用）。"""

from __future__ import annotations

import json
from dataclasses import asdict

from .config import TrainConfig, train_config_from_dict
from .train import validate_config


def train_config_to_json_text(cfg: TrainConfig) -> str:
    payload = asdict(cfg)
    payload["output_root"] = str(cfg.output_root)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_and_validate_train_config_json(raw: str) -> TrainConfig:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("JSON 根必须是对象")
    cfg = train_config_from_dict(data)
    validate_config(cfg)
    return cfg
