"""Shared observation transforms for train / eval / actors / inference."""

from __future__ import annotations

from typing import Any

import numpy as np

from .env import SnakeEnv


def hwc_to_chw(obs_hwc: np.ndarray) -> np.ndarray:
    return np.transpose(obs_hwc, (2, 0, 1)).astype(np.float32, copy=False)


def center_pad_chw(obs_chw: np.ndarray, target_size: int) -> np.ndarray:
    """Center-pad a smaller board to a fixed CHW size for a shared replay tensor."""
    channels, height, width = obs_chw.shape
    if height == target_size and width == target_size:
        return obs_chw
    if height > target_size or width > target_size:
        raise ValueError(f"观测尺寸 {obs_chw.shape} 大于目标尺寸 {target_size}")

    out = np.zeros((channels, target_size, target_size), dtype=np.float32)
    top = (target_size - height) // 2
    left = (target_size - width) // 2
    out[:, top : top + height, left : left + width] = obs_chw
    return out


def uses_spatial_padding(cfg: Any) -> bool:
    return cfg.model_type == "adaptive_cnn" and (
        getattr(cfg, "curriculum", None) is not None or getattr(cfg, "random_board", None) is not None
    )


def extract_inputs(
    env: SnakeEnv,
    obs_hwc: np.ndarray,
    *,
    model_type: str,
    local_patch_size: int,
    agent_input_size: int,
    use_padding: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    if model_type == "tiny":
        return env.get_tiny_features(), None
    if model_type == "hybrid":
        patch = env.get_local_patch(local_patch_size)
        return hwc_to_chw(patch), env.get_global_features()
    state = hwc_to_chw(obs_hwc)
    if use_padding:
        state = center_pad_chw(state, agent_input_size)
    return state, None


def extract_model_inputs(
    env: SnakeEnv,
    obs_hwc: np.ndarray,
    cfg: Any,
    agent_input_size: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    return extract_inputs(
        env,
        obs_hwc,
        model_type=str(cfg.model_type),
        local_patch_size=int(cfg.local_patch_size),
        agent_input_size=int(agent_input_size),
        use_padding=uses_spatial_padding(cfg),
    )


def encode_transition_state(state: np.ndarray, model_type: str) -> np.ndarray:
    """Pack a transition tensor for IPC / replay.

    Image observations are stored as uint8 occupancy. Tiny features are signed
    floats and must not be thresholded.
    """
    arr = np.asarray(state)
    if model_type == "tiny":
        return arr.astype(np.float32, copy=False)
    return np.asarray(arr > 0.5, dtype=np.uint8)
