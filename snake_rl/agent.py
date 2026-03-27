from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from .model import AdaptiveCNN, HybridNet, SmallSnakeCNN, TinyMLP
from .versions import FEATURE_SCHEMA_VERSION, MODEL_CHECKPOINT_SCHEMA_VERSION

ModelType = Literal["small_cnn", "adaptive_cnn", "hybrid", "tiny"]


@dataclass(slots=True)
class AgentHyperParams:
    gamma: float = 0.99
    learning_rate: float = 2.5e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100000


def build_network(
    model_type: ModelType,
    input_channels: int,
    board_size: int,
    num_actions: int,
) -> nn.Module:
    """根据 model_type 实例化对应网络。"""
    if model_type == "tiny":
        return TinyMLP(input_channels, num_actions)
    if model_type == "small_cnn":
        return SmallSnakeCNN(input_channels, board_size, num_actions)
    if model_type == "adaptive_cnn":
        return AdaptiveCNN(input_channels, num_actions)
    if model_type == "hybrid":
        return HybridNet(input_channels, num_actions)
    raise ValueError(f"未知 model_type: {model_type!r}，可选: tiny / small_cnn / adaptive_cnn / hybrid")


class DDQNAgent:
    """Double DQN agent，支持四种网络架构。

    model_type:
        tiny         — 纯 MLP，输入 10 维特征，无视觉观测
        small_cnn    — 原始固定尺寸网络，select_action/update 用单一 state 张量
        adaptive_cnn — GAP 网络，同 small_cnn 接口但支持可变尺寸
        hybrid       — CNN + 手工特征，select_action/update 额外需要 global_feat
    """

    def __init__(
        self,
        observation_shape: tuple[int, ...],
        num_actions: int,
        device: torch.device,
        hp: AgentHyperParams | None = None,
        model_type: ModelType = "small_cnn",
    ) -> None:
        self.observation_shape = tuple(int(v) for v in observation_shape)
        self.num_actions = int(num_actions)
        self.device = device
        self.hp = hp or AgentHyperParams()
        self.model_type: ModelType = model_type

        if model_type == "tiny":
            feat_dim = self.observation_shape[0]
            self.online_net = build_network(model_type, feat_dim, 0, num_actions).to(device)
            self.target_net = build_network(model_type, feat_dim, 0, num_actions).to(device)
        else:
            channels, height, width = self.observation_shape
            if height != width:
                raise ValueError("期望方形地图观测（height == width）。")
            self.online_net = build_network(model_type, channels, height, num_actions).to(device)
            self.target_net = build_network(model_type, channels, height, num_actions).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(
            self.online_net.parameters(),
            lr=self.hp.learning_rate,
            weight_decay=self.hp.weight_decay,
        )
        self.loss_fn = nn.SmoothL1Loss()

    def epsilon_by_step(self, global_step: int) -> float:
        step = max(0, int(global_step))
        if self.hp.epsilon_decay_steps <= 0:
            return float(self.hp.epsilon_end)
        ratio = min(1.0, step / float(self.hp.epsilon_decay_steps))
        return float(self.hp.epsilon_start + ratio * (self.hp.epsilon_end - self.hp.epsilon_start))

    def select_action(
        self,
        state: np.ndarray,
        global_step: int,
        eval_mode: bool = False,
        global_feat: np.ndarray | None = None,
    ) -> int:
        """选择动作。

        Args:
            state       : CHW 格式的观测张量，或 tiny 模式下 shape=(10,) 的特征向量。
            global_step : 当前全局步数（用于计算 epsilon）。
            eval_mode   : True 时关闭探索（纯贪心）。
            global_feat : 仅 hybrid 模型需要，shape=(10,) 的手工特征。
        """
        epsilon = 0.0 if eval_mode else self.epsilon_by_step(global_step)
        if np.random.random() < epsilon:
            return int(np.random.randint(0, self.num_actions))

        state_t = torch.from_numpy(state).unsqueeze(0).to(self.device, dtype=torch.float32)
        self.online_net.eval()
        with torch.no_grad():
            if self.model_type == "hybrid":
                if global_feat is None:
                    raise ValueError("hybrid 模型的 select_action 需要提供 global_feat")
                gf_t = torch.from_numpy(global_feat).unsqueeze(0).to(self.device, dtype=torch.float32)
                q_values = self.online_net(state_t, gf_t)
            else:
                q_values = self.online_net(state_t)
            action = int(torch.argmax(q_values, dim=1).item())
        self.online_net.train()
        return action

    def compute_q_values(
        self,
        state: np.ndarray,
        global_feat: np.ndarray | None = None,
    ) -> np.ndarray:
        """推理调试：返回各动作的 Q 值（numpy 1D）。"""
        state_t = torch.from_numpy(state).unsqueeze(0).to(self.device, dtype=torch.float32)
        self.online_net.eval()
        with torch.no_grad():
            if self.model_type == "hybrid":
                if global_feat is None:
                    raise ValueError("hybrid 模型的 compute_q_values 需要提供 global_feat")
                gf_t = torch.from_numpy(global_feat).unsqueeze(0).to(self.device, dtype=torch.float32)
                q_values = self.online_net(state_t, gf_t)
            else:
                q_values = self.online_net(state_t)
            out = q_values.squeeze(0).detach().cpu().numpy().astype(np.float64, copy=False)
        self.online_net.train()
        return out

    def update(
        self,
        replay_buffer: Any,
        global_step: int,
        batch_size: int,
        min_replay_size: int,
        train_frequency: int,
        target_update_interval: int,
    ) -> dict[str, float] | None:
        required = max(int(min_replay_size), int(batch_size))
        if len(replay_buffer) < required:
            return None
        if int(train_frequency) > 1 and global_step % int(train_frequency) != 0:
            return None

        batch = replay_buffer.sample(batch_size)
        metrics = self._learn_from_batch(batch)

        if target_update_interval > 0 and global_step % int(target_update_interval) == 0:
            self.sync_target()
        return metrics

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    def _learn_from_batch(
        self,
        batch: tuple[torch.Tensor, ...],
    ) -> dict[str, float]:
        if self.model_type == "hybrid":
            states, actions, rewards, next_states, dones, global_feats, next_global_feats = batch
            q_values = self.online_net(states, global_feats)
            current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_actions = torch.argmax(
                    self.online_net(next_states, next_global_feats), dim=1, keepdim=True
                )
                next_target_q = self.target_net(next_states, next_global_feats).gather(
                    1, next_actions
                ).squeeze(1)
                target_q = rewards + (1.0 - dones) * self.hp.gamma * next_target_q
        else:
            states, actions, rewards, next_states, dones = batch
            q_values = self.online_net(states)
            current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_actions = torch.argmax(self.online_net(next_states), dim=1, keepdim=True)
                next_target_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
                target_q = rewards + (1.0 - dones) * self.hp.gamma * next_target_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.hp.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(self.online_net.parameters(), self.hp.grad_clip_norm)
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "q_mean": float(current_q.detach().mean().item()),
            "target_q_mean": float(target_q.detach().mean().item()),
        }

    def checkpoint_payload(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "observation_shape": self.observation_shape,
            "num_actions": self.num_actions,
            "hyper_params": asdict(self.hp),
            "model_type": self.model_type,
            "checkpoint_schema_version": MODEL_CHECKPOINT_SCHEMA_VERSION,
            "feature_schema_version": (
                int(FEATURE_SCHEMA_VERSION) if self.model_type in ("hybrid", "tiny") else None
            ),
        }
        if extra:
            payload["extra"] = extra
        return payload

    def save_checkpoint(
        self,
        path: str | Path,
        extra: dict[str, Any] | None = None,
    ) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(extra=extra), target)

    def load_checkpoint_payload(self, ckpt: dict[str, Any]) -> dict[str, Any]:
        model_type = ckpt.get("model_type", self.model_type)
        if model_type in ("hybrid", "tiny"):
            fs = ckpt.get("feature_schema_version")
            if fs is None or int(fs) < FEATURE_SCHEMA_VERSION:
                raise ValueError(
                    f"该 checkpoint 的 {model_type} 特征版本过旧或不兼容当前 FEATURE_SCHEMA_VERSION="
                    f"{FEATURE_SCHEMA_VERSION}，请用当前代码重新训练后再加载。"
                )
        hp_data = ckpt.get("hyper_params")
        if isinstance(hp_data, dict) and hp_data:
            self.hp = AgentHyperParams(**hp_data)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        for g in self.optimizer.param_groups:
            g["lr"] = float(self.hp.learning_rate)
            g["weight_decay"] = float(self.hp.weight_decay)
        self.online_net.to(self.device)
        self.target_net.to(self.device)
        self.target_net.eval()
        return ckpt.get("extra", {})

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        ckpt = torch.load(Path(path), map_location=self.device, weights_only=False)
        return self.load_checkpoint_payload(ckpt)

    def load_weights_only(self, path: str | Path) -> None:
        """仅加载 online/target 权重，保留当前超参与优化器状态（用于 warm-start）。"""
        ckpt = torch.load(Path(path), map_location=self.device, weights_only=False)
        if ckpt.get("model_type") in ("hybrid", "tiny"):
            fs = ckpt.get("feature_schema_version")
            if fs is None or int(fs) < FEATURE_SCHEMA_VERSION:
                raise ValueError(
                    f"该 checkpoint 的 {ckpt.get('model_type')} 特征版本过旧，无法用 warm-start 加载。"
                )
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.online_net.to(self.device)
        self.target_net.to(self.device)
        self.target_net.eval()

    def reset_epsilon(self, epsilon_start: float, epsilon_end: float, epsilon_decay_steps: int) -> None:
        """课程学习阶段切换时重置探索率参数。"""
        self.hp = AgentHyperParams(
            gamma=self.hp.gamma,
            learning_rate=self.hp.learning_rate,
            weight_decay=self.hp.weight_decay,
            grad_clip_norm=self.hp.grad_clip_norm,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
        )
