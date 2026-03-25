from __future__ import annotations

import numpy as np
import torch

import json
from pathlib import Path

import pytest

from snake_rl.config import train_config_from_dict
from snake_rl.replay_buffer import ReplayBuffer
from snake_rl.schemes import get_config, load_custom_train_config


def test_replay_buffer_state_dict_roundtrip() -> None:
    obs_shape = (8, 10, 10)
    device = torch.device("cpu")
    buf = ReplayBuffer(64, obs_shape, device, hybrid=False)
    s = np.zeros(obs_shape, dtype=np.float32)
    buf.add(s, 1, 0.5, s, False)
    buf.add(s, 2, -0.1, s, True)

    sd = buf.state_dict()
    buf2 = ReplayBuffer.from_state_dict(sd, device)
    assert buf2._size == buf._size
    assert buf2._position == buf._position
    assert np.array_equal(buf2.actions[: buf2._size], buf.actions[: buf._size])


def test_train_config_from_dict_minimal() -> None:
    cfg = train_config_from_dict(
        {
            "episodes": 500,
            "model_type": "adaptive_cnn",
            "output_root": "runs",
            "env": {
                "difficulty": "easy",
                "mode": "wrap",
                "board_size": 18,
                "enable_bonus_food": True,
                "enable_obstacles": True,
                "allow_leveling": False,
                "max_steps_without_food": 300,
                "seed": 7,
            },
        }
    )
    assert cfg.episodes == 500
    assert cfg.model_type == "adaptive_cnn"
    assert cfg.env.board_size == 18
    assert cfg.env.mode == "wrap"
    assert cfg.env.seed == 7


def test_get_config_custom_requires_path(tmp_path: Path) -> None:
    """当默认 custom 配置文件不存在时，get_config(scheme="custom") 应抛 FileNotFoundError。"""
    # 确保不依赖项目根目录是否存在 custom_train_config.json
    import snake_rl.schemes as _schemes
    _orig = _schemes.DEFAULT_CUSTOM_CONFIG_PATH
    nonexistent = tmp_path / "no_such_file.json"
    _schemes.DEFAULT_CUSTOM_CONFIG_PATH = nonexistent
    try:
        with pytest.raises(FileNotFoundError):
            get_config(scheme="custom", custom_config_path=None)
    finally:
        _schemes.DEFAULT_CUSTOM_CONFIG_PATH = _orig


def test_get_config_custom_fallback_to_default(tmp_path: Path) -> None:
    """当 custom_config_path=None 时，若默认文件存在则自动加载。"""
    import snake_rl.schemes as _schemes
    from snake_rl.schemes import default_custom_train_config
    from snake_rl.train_config_json import train_config_to_json_text

    default_file = tmp_path / "custom_train_config.json"
    default_file.write_text(
        train_config_to_json_text(default_custom_train_config()),
        encoding="utf-8",
    )
    _orig = _schemes.DEFAULT_CUSTOM_CONFIG_PATH
    _schemes.DEFAULT_CUSTOM_CONFIG_PATH = default_file
    try:
        cfg = get_config(scheme="custom", custom_config_path=None)
        assert cfg.run_name == "custom"
    finally:
        _schemes.DEFAULT_CUSTOM_CONFIG_PATH = _orig


def test_load_custom_train_config_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "custom.json"
    p.write_text(
        json.dumps(
            {
                "episodes": 120,
                "model_type": "adaptive_cnn",
                "output_root": "runs",
                "run_name": "my_custom",
                "env": {
                    "difficulty": "normal",
                    "mode": "classic",
                    "board_size": 12,
                    "enable_bonus_food": False,
                    "enable_obstacles": False,
                    "allow_leveling": False,
                    "max_steps_without_food": 144,
                    "seed": 1,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    cfg = load_custom_train_config(p)
    assert cfg.episodes == 120
    assert cfg.run_name == "my_custom"
    cfg2 = get_config(scheme="custom", custom_config_path=p)
    assert cfg2.model_type == "adaptive_cnn"
    assert cfg2.env.board_size == 12
