from __future__ import annotations

import pytest
import torch

from snake_rl.agent import AgentHyperParams, DDQNAgent
from snake_rl.versions import FEATURE_SCHEMA_VERSION


def test_hybrid_checkpoint_rejects_stale_feature_schema() -> None:
    agent = DDQNAgent(
        observation_shape=(9, 11, 11),
        num_actions=3,
        device=torch.device("cpu"),
        hp=AgentHyperParams(),
        model_type="hybrid",
    )
    ckpt = {
        "model_type": "hybrid",
        "feature_schema_version": int(FEATURE_SCHEMA_VERSION) - 1,
        "online_net": agent.online_net.state_dict(),
        "target_net": agent.target_net.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "hyper_params": {},
    }
    with pytest.raises(ValueError, match="FEATURE_SCHEMA_VERSION"):
        agent.load_checkpoint_payload(ckpt)
