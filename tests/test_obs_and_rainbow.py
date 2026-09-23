from __future__ import annotations

import numpy as np
import torch

from snake_rl.config import TrainConfig, validate_config
from snake_rl.env import SnakeEnv, SnakeEnvConfig
from snake_rl.obs import encode_transition_state, extract_model_inputs, hwc_to_chw
from snake_rl.replay_buffer import NStepAccumulator, ReplayBuffer
from snake_rl.train_events import EVENT_PREFIX, parse_event_line
from snake_rl.agent import DDQNAgent, AgentHyperParams
from snake_rl.versions import MODEL_CHECKPOINT_SCHEMA_VERSION


def test_hwc_to_chw_and_tiny_encode_keeps_signed_features() -> None:
    hwc = np.zeros((4, 4, 9), dtype=np.float32)
    hwc[1, 2, 0] = 1.0
    chw = hwc_to_chw(hwc)
    assert chw.shape == (9, 4, 4)
    assert chw[0, 1, 2] == 1.0

    feats = np.array([0.2, -0.8, 0.0, 1.0], dtype=np.float32)
    encoded = encode_transition_state(feats, "tiny")
    assert encoded.dtype == np.float32
    np.testing.assert_array_almost_equal(encoded, feats)

    image = encode_transition_state(chw, "adaptive_cnn")
    assert image.dtype == np.uint8
    assert image[0, 1, 2] == 1


def test_extract_tiny_does_not_binarize() -> None:
    env = SnakeEnv(config=SnakeEnvConfig(board_size=8, mode="classic"), seed=1)
    obs, _ = env.reset(seed=1)
    cfg = TrainConfig(model_type="tiny", local_patch_size=9)
    state, gf = extract_model_inputs(env, obs, cfg, 10)
    assert gf is None
    assert state.shape == (10,)
    assert state.dtype == np.float32
    assert np.any(state < 0) or np.any((state > 0) & (state < 1))


def test_local_patch_head_is_center_and_wrap_matches_bounds() -> None:
    env = SnakeEnv(config=SnakeEnvConfig(board_size=8, mode="classic"), seed=2)
    env.reset(seed=2)
    patch = env.get_local_patch(5)
    assert patch.shape == (5, 5, 9)
    assert patch[2, 2, 0] == 1.0

    env_wrap = SnakeEnv(config=SnakeEnvConfig(board_size=8, mode="wrap"), seed=2)
    env_wrap.reset(seed=2)
    wrap_patch = env_wrap.get_local_patch(5)
    assert wrap_patch.shape == (5, 5, 9)
    assert wrap_patch[2, 2, 0] == 1.0


def test_nstep_emits_discounted_return() -> None:
    acc = NStepAccumulator(n_step=3, gamma=0.5)
    s = np.array([1.0], dtype=np.float32)
    out = []
    out.extend(acc.push(s, 0, 1.0, s, False))
    out.extend(acc.push(s, 1, 2.0, s, False))
    out.extend(acc.push(s, 2, 4.0, s, False))
    assert len(out) == 1
    assert abs(out[0].reward - (1.0 + 0.5 * 2.0 + 0.25 * 4.0)) < 1e-6
    assert out[0].action == 0


def test_per_sample_and_priority_update() -> None:
    buf = ReplayBuffer(8, (2,), torch.device("cpu"), tiny=True, per_enabled=True, per_alpha=0.6)
    for i in range(6):
        x = np.array([float(i), -0.5], dtype=np.float32)
        buf.add(x, i % 3, 0.1 * i, x, False)
    batch = buf.sample(4, beta=0.5)
    assert batch.states.shape == (4, 2)
    assert batch.weights is not None
    assert batch.indices is not None
    buf.update_priorities(batch.indices, np.ones(len(batch.indices)))
    assert len(buf) == 6


def test_validate_rejects_small_cnn_curriculum() -> None:
    from snake_rl.config import CurriculumConfig, CurriculumStage

    cfg = TrainConfig(
        model_type="small_cnn",
        curriculum=CurriculumConfig(stages=[CurriculumStage(board_size=8, episodes=10)]),
    )
    try:
        validate_config(cfg)
    except ValueError as exc:
        assert "small_cnn" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_parse_event_line() -> None:
    line = EVENT_PREFIX + '{"type":"episode","episode":3,"avg_reward":1.25}'
    event = parse_event_line(line)
    assert event is not None
    assert event["type"] == "episode"
    assert event["episode"] == 3
    assert parse_event_line("not an event") is None


def test_new_checkpoint_schema_is_v2() -> None:
    agent = DDQNAgent(
        observation_shape=(10,),
        num_actions=3,
        device=torch.device("cpu"),
        hp=AgentHyperParams(n_step=1),
        model_type="tiny",
        dueling=True,
        noisy=False,
    )
    payload = agent.checkpoint_payload()
    assert payload["checkpoint_schema_version"] == MODEL_CHECKPOINT_SCHEMA_VERSION
    assert payload["dueling"] is True
