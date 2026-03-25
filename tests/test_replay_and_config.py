from __future__ import annotations

import numpy as np
import torch

from snake_rl.config import train_config_from_dict
from snake_rl.replay_buffer import ReplayBuffer


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
