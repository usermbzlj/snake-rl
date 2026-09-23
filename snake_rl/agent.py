from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import ModelType
from .model import AdaptiveCNN, HybridNet, SmallSnakeCNN, TinyMLP, reset_noisy_noise
from .replay_buffer import ReplayBuffer, TransitionBatch
from .versions import FEATURE_SCHEMA_VERSION, MODEL_CHECKPOINT_SCHEMA_VERSION


@dataclass(slots=True)
class AgentHyperParams:
    gamma: float = 0.99
    learning_rate: float = 2.5e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100000
    n_step: int = 3
    tau: float = 0.005
    target_update: str = "hard"


def hyper_params_from_dict(data: dict[str, Any] | None) -> AgentHyperParams:
    if not data:
        return AgentHyperParams()
    allowed = {item.name for item in fields(AgentHyperParams)}
    return AgentHyperParams(**{k: v for k, v in data.items() if k in allowed})


def build_network(
    model_type: ModelType,
    input_channels: int,
    board_size: int,
    num_actions: int,
    *,
    dueling: bool = True,
    noisy: bool = False,
) -> nn.Module:
    if model_type == "tiny":
        return TinyMLP(input_channels, num_actions, dueling=dueling, noisy=noisy)
    if model_type == "small_cnn":
        return SmallSnakeCNN(input_channels, board_size, num_actions, dueling=dueling, noisy=noisy)
    if model_type == "adaptive_cnn":
        return AdaptiveCNN(input_channels, num_actions, dueling=dueling, noisy=noisy)
    if model_type == "hybrid":
        return HybridNet(input_channels, num_actions, dueling=dueling, noisy=noisy)
    raise ValueError(f"未知 model_type: {model_type!r}，可选: tiny / small_cnn / adaptive_cnn / hybrid")


class DDQNAgent:
    """Rainbow-lite Double DQN: dueling, n-step, optional PER / noisy / soft target."""

    def __init__(
        self,
        observation_shape: tuple[int, ...],
        num_actions: int,
        device: torch.device,
        hp: AgentHyperParams | None = None,
        model_type: ModelType = "adaptive_cnn",
        dueling: bool = True,
        noisy: bool = False,
    ) -> None:
        self.observation_shape = tuple(int(v) for v in observation_shape)
        self.num_actions = int(num_actions)
        self.device = device
        self.hp = hp or AgentHyperParams()
        self.model_type: ModelType = model_type
        self.dueling = bool(dueling)
        self.noisy = bool(noisy)

        if model_type == "tiny":
            feat_dim = self.observation_shape[0]
            self.online_net = build_network(
                model_type, feat_dim, 0, num_actions, dueling=self.dueling, noisy=self.noisy
            ).to(device)
            self.target_net = build_network(
                model_type, feat_dim, 0, num_actions, dueling=self.dueling, noisy=self.noisy
            ).to(device)
        else:
            channels, height, width = self.observation_shape
            if height != width:
                raise ValueError("期望方形地图观测（height == width）。")
            self.online_net = build_network(
                model_type, channels, height, num_actions, dueling=self.dueling, noisy=self.noisy
            ).to(device)
            self.target_net = build_network(
                model_type, channels, height, num_actions, dueling=self.dueling, noisy=self.noisy
            ).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(
            self.online_net.parameters(),
            lr=self.hp.learning_rate,
            weight_decay=self.hp.weight_decay,
        )
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

    def epsilon_by_step(self, global_step: int) -> float:
        if self.noisy:
            return 0.0 if self.hp.epsilon_end <= 0 else float(self.hp.epsilon_end)
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
        epsilon = 0.0 if eval_mode else self.epsilon_by_step(global_step)
        if (not self.noisy or eval_mode) and np.random.random() < epsilon:
            return int(np.random.randint(0, self.num_actions))

        state_t = torch.from_numpy(state).unsqueeze(0).to(self.device, dtype=torch.float32)
        was_training = self.online_net.training
        self.online_net.eval()
        with torch.no_grad():
            q_values = self._q(self.online_net, state_t, global_feat)
            action = int(torch.argmax(q_values, dim=1).item())
        if was_training:
            self.online_net.train()
        return action

    def compute_q_values(
        self,
        state: np.ndarray,
        global_feat: np.ndarray | None = None,
    ) -> np.ndarray:
        state_t = torch.from_numpy(state).unsqueeze(0).to(self.device, dtype=torch.float32)
        was_training = self.online_net.training
        self.online_net.eval()
        with torch.no_grad():
            q_values = self._q(self.online_net, state_t, global_feat)
            out = q_values.squeeze(0).detach().cpu().numpy().astype(np.float64, copy=False)
        if was_training:
            self.online_net.train()
        return out

    def _q(
        self,
        net: nn.Module,
        states: torch.Tensor,
        global_feat: np.ndarray | torch.Tensor | None,
    ) -> torch.Tensor:
        if self.model_type == "hybrid":
            if global_feat is None:
                raise ValueError("hybrid 模型需要 global_feat")
            if isinstance(global_feat, np.ndarray):
                gf_t = torch.from_numpy(global_feat).unsqueeze(0).to(self.device, dtype=torch.float32)
            else:
                gf_t = global_feat
            return net(states, gf_t)
        return net(states)

    def update(
        self,
        replay_buffer: ReplayBuffer,
        global_step: int,
        batch_size: int,
        min_replay_size: int,
        train_frequency: int,
        target_update_interval: int,
        beta: float = 0.4,
    ) -> dict[str, float] | None:
        required = max(int(min_replay_size), int(batch_size))
        if len(replay_buffer) < required:
            return None
        if int(train_frequency) > 1 and global_step % int(train_frequency) != 0:
            return None

        if self.noisy:
            reset_noisy_noise(self.online_net)
            reset_noisy_noise(self.target_net)

        batch = replay_buffer.sample(batch_size, beta=beta)
        metrics = self._learn_from_batch(batch)
        td_errors = metrics.pop("td_errors", None)
        if batch.indices is not None and td_errors is not None:
            replay_buffer.update_priorities(batch.indices, np.asarray(td_errors))

        if self.hp.target_update == "soft":
            self.sync_target(tau=self.hp.tau)
        elif target_update_interval > 0 and global_step % int(target_update_interval) == 0:
            self.sync_target()
        return metrics

    def sync_target(self, tau: float | None = None) -> None:
        if tau is None or tau >= 1.0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            return
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.mul_(1.0 - tau).add_(online_param.data, alpha=tau)

    def _learn_from_batch(self, batch: TransitionBatch) -> dict[str, float]:
        n_step = max(1, int(self.hp.n_step))
        gamma_n = self.hp.gamma**n_step

        if self.model_type == "hybrid":
            if batch.global_feats is None or batch.next_global_feats is None:
                raise ValueError("hybrid batch 缺少 global features")
            q_values = self.online_net(batch.states, batch.global_feats)
            current_q = q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_actions = torch.argmax(
                    self.online_net(batch.next_states, batch.next_global_feats), dim=1, keepdim=True
                )
                next_target_q = self.target_net(batch.next_states, batch.next_global_feats).gather(
                    1, next_actions
                ).squeeze(1)
                target_q = batch.rewards + (1.0 - batch.dones) * gamma_n * next_target_q
        else:
            q_values = self.online_net(batch.states)
            current_q = q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_actions = torch.argmax(self.online_net(batch.next_states), dim=1, keepdim=True)
                next_target_q = self.target_net(batch.next_states).gather(1, next_actions).squeeze(1)
                target_q = batch.rewards + (1.0 - batch.dones) * gamma_n * next_target_q

        td_error = target_q - current_q
        element_loss = self.loss_fn(current_q, target_q)
        if batch.weights is not None:
            loss = (element_loss * batch.weights).mean()
        else:
            loss = element_loss.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.hp.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(self.online_net.parameters(), self.hp.grad_clip_norm)
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "q_mean": float(current_q.detach().mean().item()),
            "target_q_mean": float(target_q.detach().mean().item()),
            "td_error": float(td_error.detach().abs().mean().item()),
            "td_errors": td_error.detach().abs().cpu().numpy(),
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
            "dueling": self.dueling,
            "noisy": self.noisy,
            "checkpoint_schema_version": MODEL_CHECKPOINT_SCHEMA_VERSION,
            "feature_schema_version": (
                int(FEATURE_SCHEMA_VERSION) if self.model_type in ("hybrid", "tiny") else None
            ),
        }
        if extra:
            payload["extra"] = extra
        return payload

    def save_checkpoint(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(extra=extra), target)

    def _reject_stale_checkpoint(self, ckpt: dict[str, Any], *, for_warm_start: bool = False) -> None:
        schema = ckpt.get("checkpoint_schema_version", 0)
        if int(schema or 0) < MODEL_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"该 checkpoint schema={schema} 与当前 MODEL_CHECKPOINT_SCHEMA_VERSION="
                f"{MODEL_CHECKPOINT_SCHEMA_VERSION} 不兼容，请用当前代码重新训练。"
            )
        model_type = ckpt.get("model_type", self.model_type)
        if model_type in ("hybrid", "tiny"):
            fs = ckpt.get("feature_schema_version")
            if fs is None or int(fs) < FEATURE_SCHEMA_VERSION:
                verb = "无法用 warm-start 加载" if for_warm_start else "请用当前代码重新训练后再加载"
                raise ValueError(
                    f"该 checkpoint 的 {model_type} 特征版本过旧或不兼容当前 FEATURE_SCHEMA_VERSION="
                    f"{FEATURE_SCHEMA_VERSION}，{verb}。"
                )

    def load_checkpoint_payload(self, ckpt: dict[str, Any]) -> dict[str, Any]:
        self._reject_stale_checkpoint(ckpt)
        hp_data = ckpt.get("hyper_params")
        if isinstance(hp_data, dict) and hp_data:
            self.hp = hyper_params_from_dict(hp_data)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        for group in self.optimizer.param_groups:
            group["lr"] = float(self.hp.learning_rate)
            group["weight_decay"] = float(self.hp.weight_decay)
        self.online_net.to(self.device)
        self.target_net.to(self.device)
        self.target_net.eval()
        return ckpt.get("extra", {})

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        ckpt = torch.load(Path(path), map_location=self.device, weights_only=False)
        return self.load_checkpoint_payload(ckpt)

    def load_weights_only(self, path: str | Path) -> None:
        ckpt = torch.load(Path(path), map_location=self.device, weights_only=False)
        self._reject_stale_checkpoint(ckpt, for_warm_start=True)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.online_net.to(self.device)
        self.target_net.to(self.device)
        self.target_net.eval()

    def reset_epsilon(self, epsilon_start: float, epsilon_end: float, epsilon_decay_steps: int) -> None:
        self.hp = AgentHyperParams(
            gamma=self.hp.gamma,
            learning_rate=self.hp.learning_rate,
            weight_decay=self.hp.weight_decay,
            grad_clip_norm=self.hp.grad_clip_norm,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
            n_step=self.hp.n_step,
            tau=self.hp.tau,
            target_update=self.hp.target_update,
        )
