"""Shared SnakeNet backbone with PPO or dueling-DQN heads."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from snake_rl.core.env import Obs


def _ch(base: int, width: float) -> int:
    return max(8, round(base * width))


class SnakeNet(nn.Module):
    def __init__(
        self,
        mode: Literal["ppo", "dqn"] = "ppo",
        width: float = 1.0,
        resize_obs: bool = False,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.width = float(width)
        self.resize_obs = bool(resize_obs)
        c32, c64, c128 = _ch(32, width), _ch(64, width), _ch(128, width)
        self.conv = nn.Sequential(
            nn.Conv2d(4, c32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c32, c64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c64, c64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c64, c128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(5),
        )
        flat = c128 * 25
        self.scalar_mlp = nn.Sequential(nn.Linear(6, 32), nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(nn.Linear(flat + 32, 256), nn.ReLU(inplace=True))
        if mode == "ppo":
            self.policy = nn.Linear(256, 3)
            self.value = nn.Linear(256, 1)
        else:
            self.v = nn.Linear(256, 1)
            self.adv = nn.Linear(256, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d | nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if mode == "ppo":
            nn.init.orthogonal_(self.policy.weight, gain=0.01)
            nn.init.orthogonal_(self.value.weight, gain=1.0)
        else:
            nn.init.orthogonal_(self.adv.weight, gain=0.01)

    def features(self, grid: Tensor, scalars: Tensor) -> Tensor:
        if self.resize_obs and (grid.shape[-1] != 31 or grid.shape[-2] != 31):
            grid = F.interpolate(grid, size=(31, 31), mode="bilinear", align_corners=False)
        x = self.conv(grid).flatten(1)
        s = self.scalar_mlp(scalars)
        return self.trunk(torch.cat([x, s], dim=1))

    def forward(self, grid: Tensor, scalars: Tensor) -> tuple[Tensor, Tensor]:
        """Return (logits_or_q, value). For DQN, value is state-value V(s)."""
        h = self.features(grid, scalars)
        if self.mode == "ppo":
            return self.policy(h), self.value(h).squeeze(-1)
        v = self.v(h)
        adv = self.adv(h)
        q = v + adv - adv.mean(dim=1, keepdim=True)
        return q, v.squeeze(-1)

    def forward_obs(self, obs: Obs) -> tuple[Tensor, Tensor]:
        return self.forward(obs.grid, obs.scalars)
