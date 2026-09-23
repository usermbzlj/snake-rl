"""Experience replay with optional PER and n-step accumulation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

GLOBAL_FEAT_DIM = 10


@dataclass(slots=True)
class TransitionBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    global_feats: torch.Tensor | None = None
    next_global_feats: torch.Tensor | None = None
    weights: torch.Tensor | None = None
    indices: np.ndarray | None = None


@dataclass(slots=True)
class PendingTransition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    global_feat: np.ndarray | None
    next_global_feat: np.ndarray | None


class NStepAccumulator:
    """Sliding n-step return buffer. Emits 1-step tuples when n==1."""

    def __init__(self, n_step: int, gamma: float) -> None:
        self.n_step = max(1, int(n_step))
        self.gamma = float(gamma)
        self._buf: deque[PendingTransition] = deque()

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        global_feat: np.ndarray | None = None,
        next_global_feat: np.ndarray | None = None,
    ) -> list[PendingTransition]:
        self._buf.append(
            PendingTransition(
                state=state,
                action=int(action),
                reward=float(reward),
                next_state=next_state,
                done=bool(done),
                global_feat=global_feat,
                next_global_feat=next_global_feat,
            )
        )
        emitted: list[PendingTransition] = []
        if done:
            while self._buf:
                emitted.append(self._emit(len(self._buf)))
            return emitted
        if len(self._buf) >= self.n_step:
            emitted.append(self._emit(self.n_step))
        return emitted

    def _emit(self, n: int) -> PendingTransition:
        first = self._buf[0]
        ret = 0.0
        for i in range(n):
            ret += (self.gamma**i) * self._buf[i].reward
        last = self._buf[n - 1]
        self._buf.popleft()
        return PendingTransition(
            state=first.state,
            action=first.action,
            reward=ret,
            next_state=last.next_state,
            done=last.done,
            global_feat=first.global_feat,
            next_global_feat=last.next_global_feat,
        )

    def reset(self) -> None:
        self._buf.clear()


class ReplayBuffer:
    """Uniform or prioritized replay. Supports hybrid globals and tiny features."""

    def __init__(
        self,
        capacity: int,
        observation_shape: tuple[int, ...],
        device: torch.device,
        hybrid: bool = False,
        tiny: bool = False,
        per_enabled: bool = False,
        per_alpha: float = 0.6,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = int(capacity)
        self.device = device
        self.observation_shape = tuple(int(v) for v in observation_shape)
        self.hybrid = hybrid
        self.tiny = tiny
        self.per_enabled = bool(per_enabled)
        self.per_alpha = float(per_alpha)
        self._position = 0
        self._size = 0
        self._rng = np.random.default_rng()
        self._max_priority = 1.0

        state_dtype = np.float32 if self.tiny else np.uint8
        self.states = np.zeros((self.capacity, *self.observation_shape), dtype=state_dtype)
        self.next_states = np.zeros((self.capacity, *self.observation_shape), dtype=state_dtype)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.priorities = np.zeros((self.capacity,), dtype=np.float64)

        if self.hybrid:
            self.global_feats = np.zeros((self.capacity, GLOBAL_FEAT_DIM), dtype=np.float32)
            self.next_global_feats = np.zeros((self.capacity, GLOBAL_FEAT_DIM), dtype=np.float32)

        if self.per_enabled:
            self._tree = np.zeros(2 * self.capacity, dtype=np.float64)

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
        idx = self._position
        if self.tiny:
            self.states[idx] = state.astype(np.float32)
            self.next_states[idx] = next_state.astype(np.float32)
        else:
            self.states[idx] = self._to_uint8(state)
            self.next_states[idx] = self._to_uint8(next_state)
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.dones[idx] = 1.0 if done else 0.0

        if self.hybrid:
            if global_feat is None or next_global_feat is None:
                raise ValueError("hybrid ReplayBuffer 的 add() 需要提供 global_feat 和 next_global_feat")
            self.global_feats[idx] = global_feat.astype(np.float32)
            self.next_global_feats[idx] = next_global_feat.astype(np.float32)

        priority = self._max_priority
        self.priorities[idx] = priority
        if self.per_enabled:
            self._tree_update(idx, priority**self.per_alpha)

        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        return self._size >= int(batch_size)

    def sample(self, batch_size: int, beta: float = 0.4) -> TransitionBatch:
        if not self.can_sample(batch_size):
            raise ValueError("not enough samples in replay buffer")
        batch_size = int(batch_size)
        if self.per_enabled and self._size > 0:
            indices, weights = self._sample_prioritized(batch_size, float(beta))
        else:
            indices = self._rng.integers(0, self._size, size=batch_size, endpoint=False)
            weights = np.ones(batch_size, dtype=np.float32)

        states = torch.from_numpy(self.states[indices]).to(self.device, dtype=torch.float32)
        next_states = torch.from_numpy(self.next_states[indices]).to(self.device, dtype=torch.float32)
        actions = torch.from_numpy(self.actions[indices]).to(self.device, dtype=torch.long)
        rewards = torch.from_numpy(self.rewards[indices]).to(self.device, dtype=torch.float32)
        dones = torch.from_numpy(self.dones[indices]).to(self.device, dtype=torch.float32)
        weight_t = torch.from_numpy(weights.astype(np.float32)).to(self.device)

        global_feats = None
        next_global_feats = None
        if self.hybrid:
            global_feats = torch.from_numpy(self.global_feats[indices]).to(self.device, dtype=torch.float32)
            next_global_feats = torch.from_numpy(self.next_global_feats[indices]).to(
                self.device, dtype=torch.float32
            )

        return TransitionBatch(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            global_feats=global_feats,
            next_global_feats=next_global_feats,
            weights=weight_t,
            indices=indices,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        if not self.per_enabled:
            return
        abs_err = np.abs(np.asarray(td_errors, dtype=np.float64)) + 1e-6
        for idx, err in zip(indices, abs_err, strict=False):
            i = int(idx)
            self.priorities[i] = err
            self._max_priority = max(self._max_priority, float(err))
            self._tree_update(i, err**self.per_alpha)

    def ordered_indices(self) -> np.ndarray:
        if self._size <= 0:
            return np.zeros((0,), dtype=np.int64)
        start = (self._position - self._size) % self.capacity
        return (start + np.arange(self._size, dtype=np.int64)) % self.capacity

    def resized_copy(self, new_capacity: int) -> ReplayBuffer:
        new_capacity = int(new_capacity)
        if new_capacity <= 0:
            raise ValueError("new_capacity must be > 0")

        out = ReplayBuffer(
            capacity=new_capacity,
            observation_shape=self.observation_shape,
            device=self.device,
            hybrid=self.hybrid,
            tiny=self.tiny,
            per_enabled=self.per_enabled,
            per_alpha=self.per_alpha,
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
        out.priorities[:keep] = self.priorities[selected]
        if self.hybrid:
            out.global_feats[:keep] = self.global_feats[selected]
            out.next_global_feats[:keep] = self.next_global_feats[selected]
        out._size = keep
        out._position = keep % new_capacity
        out._max_priority = float(np.max(out.priorities[:keep])) if keep else 1.0
        if out.per_enabled:
            for i in range(keep):
                out._tree_update(i, out.priorities[i] ** out.per_alpha)
        return out

    def _to_uint8(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs > 0.5, dtype=np.uint8)

    def _tree_update(self, data_idx: int, value: float) -> None:
        tree_idx = data_idx + self.capacity
        delta = float(value) - self._tree[tree_idx]
        self._tree[tree_idx] = float(value)
        while tree_idx > 1:
            tree_idx //= 2
            self._tree[tree_idx] += delta

    def _tree_total(self) -> float:
        return float(self._tree[1])

    def _tree_find(self, value: float) -> int:
        idx = 1
        while idx < self.capacity:
            left = idx * 2
            if value <= self._tree[left]:
                idx = left
            else:
                value -= self._tree[left]
                idx = left + 1
        data_idx = idx - self.capacity
        if data_idx >= self._size:
            data_idx = int(self._rng.integers(0, self._size))
        return int(data_idx)

    def _sample_prioritized(self, batch_size: int, beta: float) -> tuple[np.ndarray, np.ndarray]:
        total = self._tree_total()
        if total <= 0:
            indices = self._rng.integers(0, self._size, size=batch_size, endpoint=False)
            return indices, np.ones(batch_size, dtype=np.float32)
        segment = total / batch_size
        indices = np.empty(batch_size, dtype=np.int64)
        raw_p = np.empty(batch_size, dtype=np.float64)
        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            mass = float(self._rng.uniform(low, high))
            idx = self._tree_find(mass)
            indices[i] = idx
            raw_p[i] = max(self._tree[idx + self.capacity], 1e-12)
        probs = raw_p / total
        weights = (self._size * probs) ** (-beta)
        weights /= weights.max()
        return indices, weights.astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "capacity": self.capacity,
            "observation_shape": list(self.observation_shape),
            "hybrid": self.hybrid,
            "tiny": self.tiny,
            "per_enabled": self.per_enabled,
            "per_alpha": self.per_alpha,
            "_position": self._position,
            "_size": self._size,
            "_max_priority": self._max_priority,
            "states": np.ascontiguousarray(self.states),
            "next_states": np.ascontiguousarray(self.next_states),
            "actions": np.ascontiguousarray(self.actions),
            "rewards": np.ascontiguousarray(self.rewards),
            "dones": np.ascontiguousarray(self.dones),
            "priorities": np.ascontiguousarray(self.priorities),
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
            per_enabled=bool(data.get("per_enabled", False)),
            per_alpha=float(data.get("per_alpha", 0.6)),
        )
        buf._position = int(data["_position"])
        buf._size = int(data["_size"])
        buf.states[:] = data["states"]
        buf.next_states[:] = data["next_states"]
        buf.actions[:] = data["actions"]
        buf.rewards[:] = data["rewards"]
        buf.dones[:] = data["dones"]
        if "priorities" in data:
            buf.priorities[:] = data["priorities"]
            buf._max_priority = float(data.get("_max_priority", np.max(buf.priorities[: max(buf._size, 1)]) or 1.0))
        else:
            buf.priorities[: buf._size] = 1.0
            buf._max_priority = 1.0
        if buf.hybrid:
            buf.global_feats[:] = data["global_feats"]
            buf.next_global_feats[:] = data["next_global_feats"]
        if buf.per_enabled:
            for i in range(buf._size):
                buf._tree_update(i, buf.priorities[i] ** buf.per_alpha)
        return buf
