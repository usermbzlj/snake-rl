"""GPU replay buffer storing quantized observations and n-step component sums."""

from __future__ import annotations

import torch
from torch import Tensor


class ComponentReplayBuffer:
    """Stores transitions with n-step discounted component vectors."""

    def __init__(
        self,
        capacity: int,
        grid_shape: tuple[int, int, int],
        scalar_dim: int,
        n_components: int,
        device: torch.device,
    ) -> None:
        self.capacity = int(capacity)
        self.device = device
        c, h, w = grid_shape
        self.obs_grid = torch.zeros(capacity, c, h, w, dtype=torch.float16, device=device)
        self.obs_scalars = torch.zeros(capacity, scalar_dim, dtype=torch.float32, device=device)
        self.next_grid = torch.zeros(capacity, c, h, w, dtype=torch.float16, device=device)
        self.next_scalars = torch.zeros(capacity, scalar_dim, dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.comp_sum = torch.zeros(capacity, n_components, dtype=torch.float32, device=device)
        self.discount = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.done = torch.zeros(capacity, dtype=torch.bool, device=device)
        self._pos = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @staticmethod
    def quantize_grid(grid: Tensor) -> Tensor:
        return grid.clamp(0, 1).to(torch.float16)

    @staticmethod
    def dequantize_grid(grid_u8: Tensor) -> Tensor:
        return grid_u8.float()

    def add_batch(
        self,
        obs_grid: Tensor,
        obs_scalars: Tensor,
        actions: Tensor,
        comp_sum: Tensor,
        discount: Tensor,
        next_grid: Tensor,
        next_scalars: Tensor,
        done: Tensor,
    ) -> None:
        b = int(actions.shape[0])
        if b == 0:
            return
        idx = torch.arange(self._pos, self._pos + b, device=self.device) % self.capacity
        self.obs_grid[idx] = self.quantize_grid(obs_grid)
        self.obs_scalars[idx] = obs_scalars
        self.actions[idx] = actions.long()
        self.comp_sum[idx] = comp_sum
        self.discount[idx] = discount
        self.next_grid[idx] = self.quantize_grid(next_grid)
        self.next_scalars[idx] = next_scalars
        self.done[idx] = done
        self._pos = (self._pos + b) % self.capacity
        self._size = min(self.capacity, self._size + b)

    def sample(self, batch_size: int) -> dict[str, Tensor]:
        if self._size < batch_size:
            raise RuntimeError("replay buffer too small to sample")
        idx = torch.randint(0, self._size, (batch_size,), device=self.device)
        return {
            "obs_grid": self.dequantize_grid(self.obs_grid[idx]),
            "obs_scalars": self.obs_scalars[idx],
            "actions": self.actions[idx],
            "comp_sum": self.comp_sum[idx],
            "discount": self.discount[idx],
            "next_grid": self.dequantize_grid(self.next_grid[idx]),
            "next_scalars": self.next_scalars[idx],
            "done": self.done[idx],
        }


def n_step_component_returns(
    components: Tensor,
    dones: Tensor,
    gamma: float,
    n_step: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """n-step discounted component sums.

    Args:
        components: [T, 6] or [T, N, 6]
        dones: [T] or [T, N] — episode ended after that step
        gamma: discount factor
        n_step: horizon

    Returns:
        comp_sum, discount_prod, bootstrap_valid
        (same leading shape as inputs; N added if needed)
        bootstrap_valid is True iff all n steps exist and none ended the episode.
    """
    squeezed = False
    if components.ndim == 2:
        components = components.unsqueeze(1)
        dones = dones.unsqueeze(1)
        squeezed = True

    t, n, c = components.shape
    device = components.device
    dtype = components.dtype
    comp_sum = torch.zeros(t, n, c, device=device, dtype=dtype)
    alive = torch.ones(t, n, dtype=torch.bool, device=device)

    for k in range(n_step):
        exists = (torch.arange(t, device=device)[:, None] + k) < t
        exists = exists.expand(t, n)
        active = alive & exists
        src = torch.zeros(t, n, c, device=device, dtype=dtype)
        if k == 0:
            src = components
        else:
            src[:-k] = components[k:]
        comp_sum = comp_sum + active.unsqueeze(-1).to(dtype) * ((gamma**k) * src)
        done_k = torch.zeros(t, n, dtype=torch.bool, device=device)
        if k == 0:
            done_k = dones
        else:
            done_k[:-k] = dones[k:]
        alive = alive & exists & (~done_k)

    bootstrap_valid = alive.clone()
    discount_prod = torch.where(
        bootstrap_valid,
        torch.full((t, n), gamma**n_step, device=device, dtype=dtype),
        torch.zeros(t, n, device=device, dtype=dtype),
    )

    if squeezed:
        return comp_sum.squeeze(1), discount_prod.squeeze(1), bootstrap_valid.squeeze(1)
    return comp_sum, discount_prod, bootstrap_valid
