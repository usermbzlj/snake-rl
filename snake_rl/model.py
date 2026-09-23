from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class FactorizedNoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("eps_in", torch.zeros(in_features))
        self.register_buffer("eps_out", torch.zeros(out_features))
        self.sigma0 = float(sigma0)
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma0 / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma0 / math.sqrt(self.out_features))

    def reset_noise(self) -> None:
        eps_in = torch.randn(self.in_features, device=self.weight_mu.device)
        eps_out = torch.randn(self.out_features, device=self.weight_mu.device)
        self.eps_in.copy_(eps_in.sign() * eps_in.abs().sqrt())
        self.eps_out.copy_(eps_out.sign() * eps_out.abs().sqrt())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * torch.outer(self.eps_out, self.eps_in)
            bias = self.bias_mu + self.bias_sigma * self.eps_out
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


def _linear(in_features: int, out_features: int, noisy: bool) -> nn.Module:
    if noisy:
        return FactorizedNoisyLinear(in_features, out_features)
    return nn.Linear(in_features, out_features)


class DuelingHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_actions: int, noisy: bool) -> None:
        super().__init__()
        self.value = nn.Sequential(
            _linear(in_dim, hidden, noisy),
            nn.ReLU(inplace=True),
            _linear(hidden, 1, noisy),
        )
        self.advantage = nn.Sequential(
            _linear(in_dim, hidden, noisy),
            nn.ReLU(inplace=True),
            _linear(hidden, num_actions, noisy),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.value(x)
        adv = self.advantage(x)
        return value + adv - adv.mean(dim=1, keepdim=True)


def reset_noisy_noise(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, FactorizedNoisyLinear):
            child.reset_noise()


class SmallSnakeCNN(nn.Module):
    """Fixed-size CNN (Flatten + FC). Only supports a constant board_size."""

    def __init__(
        self,
        input_channels: int,
        board_size: int,
        num_actions: int,
        *,
        dueling: bool = True,
        noisy: bool = False,
    ) -> None:
        super().__init__()
        self.dueling = dueling
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            sample = torch.zeros(1, input_channels, board_size, board_size)
            feature_dim = int(self.features(sample).flatten(1).shape[1])
        if dueling:
            self.q_head = DuelingHead(feature_dim, 256, num_actions, noisy)
        else:
            self.q_head = nn.Sequential(
                _linear(feature_dim, 256, noisy),
                nn.ReLU(inplace=True),
                _linear(256, num_actions, noisy),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        feats = self.features(x).flatten(1)
        return self.q_head(feats)


class AdaptiveCNN(nn.Module):
    """Resolution-independent CNN via global average pooling."""

    FEATURE_DIM = 64

    def __init__(
        self,
        input_channels: int,
        num_actions: int,
        *,
        dueling: bool = True,
        noisy: bool = False,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.FEATURE_DIM, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        if dueling:
            self.q_head = DuelingHead(self.FEATURE_DIM, 128, num_actions, noisy)
        else:
            self.q_head = nn.Sequential(
                _linear(self.FEATURE_DIM, 128, noisy),
                nn.ReLU(inplace=True),
                _linear(128, num_actions, noisy),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        pooled = self.gap(self.features(x)).flatten(1)
        return self.q_head(pooled)


class TinyMLP(nn.Module):
    """MLP over 10-d ray + food features."""

    FEAT_DIM = 10

    def __init__(
        self,
        feat_dim: int,
        num_actions: int,
        *,
        dueling: bool = True,
        noisy: bool = False,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            _linear(feat_dim, 64, noisy),
            nn.ReLU(inplace=True),
            _linear(64, 64, noisy),
            nn.ReLU(inplace=True),
        )
        if dueling:
            self.q_head = DuelingHead(64, 64, num_actions, noisy)
        else:
            self.q_head = _linear(64, num_actions, noisy)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        return self.q_head(self.trunk(x))


class HybridNet(nn.Module):
    """Local CNN patch + global hand-crafted features (FEATURE_SCHEMA_VERSION=2).

    global_feat layout:
        [0] food Δx / (size-1)
        [1] food Δy / (size-1)
        [2] Manhattan / max
        [3-6] wall / wrap seam features
        [7] snake_len / size²
        [8] foods_eaten / (size²/4) clipped
        [9] bonus-food flag
    """

    GLOBAL_FEAT_DIM = 10
    CNN_OUT_DIM = 64
    GLOBAL_HIDDEN = 32
    FUSED_HIDDEN = 128

    def __init__(
        self,
        input_channels: int,
        num_actions: int,
        *,
        dueling: bool = True,
        noisy: bool = False,
    ) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.CNN_OUT_DIM, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.global_branch = nn.Sequential(
            _linear(self.GLOBAL_FEAT_DIM, self.GLOBAL_HIDDEN, noisy),
            nn.ReLU(inplace=True),
        )
        fused_in = self.CNN_OUT_DIM + self.GLOBAL_HIDDEN
        if dueling:
            self.q_head = DuelingHead(fused_in, self.FUSED_HIDDEN, num_actions, noisy)
        else:
            self.q_head = nn.Sequential(
                _linear(fused_in, self.FUSED_HIDDEN, noisy),
                nn.ReLU(inplace=True),
                _linear(self.FUSED_HIDDEN, num_actions, noisy),
            )

    def forward(self, map_obs: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        if map_obs.dtype != torch.float32:
            map_obs = map_obs.float()
        if global_feat.dtype != torch.float32:
            global_feat = global_feat.float()
        cnn_out = self.gap(self.cnn(map_obs)).flatten(1)
        global_out = self.global_branch(global_feat)
        return self.q_head(torch.cat([cnn_out, global_out], dim=1))
