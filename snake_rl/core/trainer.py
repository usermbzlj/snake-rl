"""Trainer protocol, episode stats, and factory."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

from snake_rl.core.config import ExperimentConfig, live_field_keys, resolve_device
from snake_rl.core.env import Obs
from snake_rl.core.network import SnakeNet


@runtime_checkable
class Trainer(Protocol):
    config: ExperimentConfig
    env_steps: int
    iteration: int

    def train_iteration(self) -> dict[str, float]: ...
    def apply_live(self, patch: dict[str, float]) -> None: ...
    def policy_net(self) -> SnakeNet: ...
    def act(self, obs: Obs, greedy: bool = False) -> tuple[Tensor, Tensor, Tensor]: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, d: dict[str, Any]) -> None: ...


class EpisodeStats:
    """Accumulate finished-episode stats within one iteration."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.scores: list[float] = []
        self.lengths: list[float] = []
        self.steps: list[float] = []
        self.returns: list[float] = []
        self.death_wall = 0
        self.death_self = 0
        self.death_starve = 0
        self.wins = 0

    def add(
        self,
        done: Tensor,
        cause: Tensor,
        ep_score: Tensor,
        ep_length: Tensor,
        ep_steps: Tensor,
        ep_return: Tensor,
    ) -> None:
        if not done.any():
            return
        idx = done.nonzero(as_tuple=False).view(-1)
        self.scores.extend(ep_score[idx].detach().cpu().tolist())
        self.lengths.extend(ep_length[idx].detach().cpu().tolist())
        self.steps.extend(ep_steps[idx].detach().cpu().tolist())
        self.returns.extend(ep_return[idx].detach().cpu().tolist())
        causes = cause[idx].detach().cpu()
        self.death_wall += int((causes == 1).sum().item())
        self.death_self += int((causes == 2).sum().item())
        self.death_starve += int((causes == 3).sum().item())
        self.wins += int((causes == 4).sum().item())

    def as_metrics(self) -> dict[str, float]:
        n = len(self.scores)
        if n == 0:
            return {"episodes": 0.0}
        import math

        out: dict[str, float] = {
            "episodes": float(n),
            "score_mean": float(sum(self.scores) / n),
            "score_max": float(max(self.scores)),
            "length_mean": float(sum(self.lengths) / n),
            "return_mean": float(sum(self.returns) / n),
            "ep_steps_mean": float(sum(self.steps) / n),
            "death_wall": self.death_wall / n,
            "death_self": self.death_self / n,
            "death_starve": self.death_starve / n,
            "win_rate": self.wins / n,
        }
        for k, v in list(out.items()):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                del out[k]
        return out


def reward_weights_tensor(config: ExperimentConfig, device: torch.device) -> Tensor:
    return torch.tensor(config.reward.as_tensor_list(), dtype=torch.float32, device=device)


def _get_dotted(obj: Any, key: str) -> Any:
    for part in key.split("."):
        obj = getattr(obj, part)
    return obj


def sync_live_fields(trainer: Trainer, config: ExperimentConfig) -> dict[str, float]:
    """Apply the live-tunable values of ``config`` that differ from the trainer's current ones.

    Only differing keys are applied: re-applying an unchanged ``dqn.epsilon_end`` would pin
    DQN's exploration to its floor and skip the decay schedule.
    """
    patch = {
        key: float(_get_dotted(config, key))
        for key in live_field_keys(config.algo)
        if _get_dotted(config, key) != _get_dotted(trainer.config, key)
    }
    if patch:
        trainer.apply_live(patch)
    return patch


def make_trainer(config: ExperimentConfig) -> Trainer:
    device = resolve_device(config.run.device)
    if config.algo == "ppo":
        from snake_rl.core.ppo import PPOTrainer

        return PPOTrainer(config, device=device)
    if config.algo == "dqn":
        from snake_rl.core.dqn import DQNTrainer

        return DQNTrainer(config, device=device)
    raise ValueError(f"未知算法: {config.algo}")


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
