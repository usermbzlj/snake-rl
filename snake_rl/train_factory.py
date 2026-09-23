"""Factories for env / agent / replay / seeds used by training and estimates."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from .agent import AgentHyperParams, DDQNAgent
from .config import EnvPreset, TrainConfig, resolve_device
from .env import TINY_FEAT_DIM, SnakeEnv, SnakeEnvConfig
from .replay_buffer import ReplayBuffer


def set_global_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_env_options(
    env: EnvPreset,
    *,
    board_size: int | None = None,
    max_steps_without_food: int | None = None,
) -> dict[str, Any]:
    return env.to_options(board_size=board_size, max_steps_without_food=max_steps_without_food)


def build_initial_env(cfg: TrainConfig) -> SnakeEnv:
    if cfg.curriculum is not None:
        first_stage = cfg.curriculum.stages[0]
        if first_stage.board_sizes:
            first_board = max(int(size) for size in first_stage.board_sizes)
            timeout = max(
                1,
                int(round(first_board * first_board * first_stage.max_steps_scale)),
            )
        else:
            first_board = first_stage.board_size
            timeout = first_stage.max_steps_without_food
            if cfg.curriculum.scale_timeout:
                timeout = first_board * first_board
        options = build_env_options(cfg.env, board_size=first_board, max_steps_without_food=timeout)
    elif cfg.random_board is not None:
        first_board = cfg.random_board.board_sizes[0]
        timeout = max(1, int(round(first_board * first_board * cfg.random_board.max_steps_scale)))
        options = build_env_options(cfg.env, board_size=first_board, max_steps_without_food=timeout)
    else:
        options = build_env_options(cfg.env)

    return SnakeEnv(config=SnakeEnvConfig(**options), seed=cfg.env.seed, reward_weights=cfg.reward_weights)


def get_agent_input_size(cfg: TrainConfig) -> int:
    if cfg.model_type == "tiny":
        return TINY_FEAT_DIM
    if cfg.model_type == "hybrid":
        return int(cfg.local_patch_size)
    if cfg.curriculum is not None:
        sizes: list[int] = []
        for stage in cfg.curriculum.stages:
            if stage.board_sizes:
                sizes.extend(int(size) for size in stage.board_sizes)
            else:
                sizes.append(int(stage.board_size))
        return max(sizes)
    if cfg.random_board is not None:
        return max(cfg.random_board.board_sizes)
    return int(cfg.env.board_size)


def create_agent(cfg: TrainConfig, device: torch.device, observation_shape: tuple[int, ...]) -> DDQNAgent:
    return DDQNAgent(
        observation_shape=observation_shape,
        num_actions=3,
        device=device,
        hp=AgentHyperParams(
            gamma=cfg.gamma,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            grad_clip_norm=cfg.grad_clip_norm,
            epsilon_start=cfg.epsilon_start,
            epsilon_end=cfg.epsilon_end,
            epsilon_decay_steps=cfg.epsilon_decay_steps,
            n_step=cfg.n_step,
            tau=cfg.tau,
            target_update=cfg.target_update,
        ),
        model_type=cfg.model_type,
        dueling=cfg.dueling,
        noisy=cfg.noisy,
    )


def create_replay(
    cfg: TrainConfig,
    device: torch.device,
    observation_shape: tuple[int, ...],
    capacity: int | None = None,
) -> ReplayBuffer:
    return ReplayBuffer(
        capacity=cfg.replay_capacity if capacity is None else int(capacity),
        observation_shape=observation_shape,
        device=device,
        hybrid=cfg.model_type == "hybrid",
        tiny=cfg.model_type == "tiny",
        per_enabled=cfg.per_enabled,
        per_alpha=cfg.per_alpha,
    )


def resolve_train_device(cfg: TrainConfig) -> torch.device:
    return torch.device(resolve_device(cfg.device))
