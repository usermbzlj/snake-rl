from __future__ import annotations

from typing import Any

import numpy as np
import torch

GLOBAL_FEAT_DIM = 10  # 与 model.HybridNet.GLOBAL_FEAT_DIM 保持一致


class ReplayBuffer:
    """经验回放池，支持普通模式、hybrid 模式和 tiny 模式。

    hybrid=True 时，add() 和 sample() 额外处理 global_feat / next_global_feat。
    tiny=True 时，states 存储为 float32 标量特征（非图像）。
    """

    def __init__(
        self,
        capacity: int,
        observation_shape: tuple[int, ...],
        device: torch.device,
        hybrid: bool = False,
        tiny: bool = False,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = int(capacity)
        self.device = device
        self.observation_shape = tuple(int(v) for v in observation_shape)
        self.hybrid = hybrid
        self.tiny = tiny
        self._position = 0
        self._size = 0
        self._rng = np.random.default_rng()

        state_dtype = np.float32 if self.tiny else np.uint8
        self.states = np.zeros((self.capacity, *self.observation_shape), dtype=state_dtype)
        self.next_states = np.zeros((self.capacity, *self.observation_shape), dtype=state_dtype)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)

        if self.hybrid:
            self.global_feats = np.zeros((self.capacity, GLOBAL_FEAT_DIM), dtype=np.float32)
            self.next_global_feats = np.zeros((self.capacity, GLOBAL_FEAT_DIM), dtype=np.float32)

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        global_feat: np.ndarray | None = None,
        next_global_feat: np.ndarray | None = None,
    ) -> None:
        if self.tiny:
            self.states[self._position] = state.astype(np.float32)
            self.next_states[self._position] = next_state.astype(np.float32)
        else:
            self.states[self._position] = self._to_uint8(state)
            self.next_states[self._position] = self._to_uint8(next_state)
        self.actions[self._position] = int(action)
        self.rewards[self._position] = float(reward)
        self.dones[self._position] = 1.0 if done else 0.0

        if self.hybrid:
            if global_feat is None or next_global_feat is None:
                raise ValueError("hybrid ReplayBuffer 的 add() 需要提供 global_feat 和 next_global_feat")
            self.global_feats[self._position] = global_feat.astype(np.float32)
            self.next_global_feats[self._position] = next_global_feat.astype(np.float32)

        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        return self._size >= int(batch_size)

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, ...]:
        if not self.can_sample(batch_size):
            raise ValueError("not enough samples in replay buffer")
        indices = self._rng.integers(0, self._size, size=batch_size, endpoint=False)

        states = torch.from_numpy(self.states[indices]).to(self.device, dtype=torch.float32)
        next_states = torch.from_numpy(self.next_states[indices]).to(self.device, dtype=torch.float32)
        actions = torch.from_numpy(self.actions[indices]).to(self.device, dtype=torch.long)
        rewards = torch.from_numpy(self.rewards[indices]).to(self.device, dtype=torch.float32)
        dones = torch.from_numpy(self.dones[indices]).to(self.device, dtype=torch.float32)

        if self.hybrid:
            global_feats = torch.from_numpy(self.global_feats[indices]).to(self.device, dtype=torch.float32)
            next_global_feats = torch.from_numpy(self.next_global_feats[indices]).to(self.device, dtype=torch.float32)
            return states, actions, rewards, next_states, dones, global_feats, next_global_feats

        return states, actions, rewards, next_states, dones

    def ordered_indices(self) -> np.ndarray:
        """按时间顺序返回当前缓存中样本的索引（最旧 -> 最新）。"""
        if self._size <= 0:
            return np.zeros((0,), dtype=np.int64)
        start = (self._position - self._size) % self.capacity
        return (start + np.arange(self._size, dtype=np.int64)) % self.capacity

    def resized_copy(self, new_capacity: int) -> "ReplayBuffer":
        """创建一个新容量的回放池，并保留尽可能多的最新经验。

        用途：
        - curriculum 阶段切换时，`carry_replay=True` 但下一阶段希望更大的容量
        - 避免仅改参数却没有真正迁移旧经验
        """
        new_capacity = int(new_capacity)
        if new_capacity <= 0:
            raise ValueError("new_capacity must be > 0")

        out = ReplayBuffer(
            capacity=new_capacity,
            observation_shape=self.observation_shape,
            device=self.device,
            hybrid=self.hybrid,
            tiny=self.tiny,
        )
        if self._size <= 0:
            return out

        ordered = self.ordered_indices()
        keep = min(self._size, new_capacity)
        selected = ordered[-keep:]

        out.states[:keep] = self.states[selected]
        out.next_states[:keep] = self.next_states[selected]
        out.actions[:keep] = self.actions[selected]
        out.rewards[:keep] = self.rewards[selected]
        out.dones[:keep] = self.dones[selected]
        if self.hybrid:
            out.global_feats[:keep] = self.global_feats[selected]
            out.next_global_feats[:keep] = self.next_global_feats[selected]

        out._size = keep
        out._position = keep % new_capacity
        return out

    def _to_uint8(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs > 0.5, dtype=np.uint8)

    def state_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "capacity": self.capacity,
            "observation_shape": list(self.observation_shape),
            "hybrid": self.hybrid,
            "tiny": self.tiny,
            "_position": self._position,
            "_size": self._size,
            "states": np.ascontiguousarray(self.states),
            "next_states": np.ascontiguousarray(self.next_states),
            "actions": np.ascontiguousarray(self.actions),
            "rewards": np.ascontiguousarray(self.rewards),
            "dones": np.ascontiguousarray(self.dones),
        }
        if self.hybrid:
            d["global_feats"] = np.ascontiguousarray(self.global_feats)
            d["next_global_feats"] = np.ascontiguousarray(self.next_global_feats)
        return d

    @classmethod
    def from_state_dict(cls, data: dict[str, Any], device: torch.device) -> ReplayBuffer:
        obs_shape = tuple(int(x) for x in data["observation_shape"])
        buf = cls(
            capacity=int(data["capacity"]),
            observation_shape=obs_shape,
            device=device,
            hybrid=bool(data["hybrid"]),
            tiny=bool(data.get("tiny", False)),
        )
        buf._position = int(data["_position"])
        buf._size = int(data["_size"])
        buf.states[:] = data["states"]
        buf.next_states[:] = data["next_states"]
        buf.actions[:] = data["actions"]
        buf.rewards[:] = data["rewards"]
        buf.dones[:] = data["dones"]
        if buf.hybrid:
            buf.global_feats[:] = data["global_feats"]
            buf.next_global_feats[:] = data["next_global_feats"]
        return buf
