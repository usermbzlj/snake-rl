"""GPU-vectorized batched Snake environment with egocentric observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

_DIR_DR = torch.tensor([-1, 0, 1, 0], dtype=torch.int32)
_DIR_DC = torch.tensor([0, 1, 0, -1], dtype=torch.int32)

CAUSE_NONE = 0
CAUSE_WALL = 1
CAUSE_SELF = 2
CAUSE_STARVE = 3
CAUSE_WIN = 4

COMP_FOOD = 0
COMP_DEATH = 1
COMP_STEP = 2
COMP_APPROACH = 3
COMP_STARVE = 4
COMP_WIN = 5
N_COMPONENTS = 6


@dataclass(slots=True)
class StepResult:
    components: Tensor
    done: Tensor
    cause: Tensor
    ep_score: Tensor
    ep_length: Tensor
    ep_steps: Tensor
    ep_size: Tensor
    ep_return_components: Tensor


@dataclass(slots=True)
class Obs:
    grid: Tensor
    scalars: Tensor


def hunger_limit(size: Tensor | int, hunger_factor: float) -> Tensor | int:
    if isinstance(size, int):
        return max(int(hunger_factor * size * size), 4 * size)
    s = size.to(dtype=torch.int32)
    return torch.maximum((hunger_factor * (s * s).to(torch.float32)).to(torch.int32), 4 * s)


def _rotation_offsets(grid: int, device: torch.device) -> Tensor:
    w = 2 * grid - 1
    center = grid - 1
    ii = torch.arange(w, device=device) - center
    jj = torch.arange(w, device=device) - center
    di, dj = torch.meshgrid(ii, jj, indexing="ij")
    offsets = torch.empty(4, w, w, 2, dtype=torch.int32, device=device)
    offsets[0, ..., 0], offsets[0, ..., 1] = di, dj
    offsets[1, ..., 0], offsets[1, ..., 1] = dj, -di
    offsets[2, ..., 0], offsets[2, ..., 1] = -di, -dj
    offsets[3, ..., 0], offsets[3, ..., 1] = -dj, di
    return offsets


def window_to_board(
    direction: int | Tensor,
    head: tuple[int, int] | Tensor,
    size: int | Tensor,
    grid: int,
) -> Tensor:
    device = head.device if isinstance(head, Tensor) else torch.device("cpu")
    offsets = _rotation_offsets(grid, device)
    if isinstance(direction, int):
        off = offsets[direction]
        hr, hc = int(head[0]), int(head[1])  # type: ignore[index]
        s = int(size)
        br = hr + off[..., 0].long()
        bc = hc + off[..., 1].long()
        valid = (br >= 0) & (br < s) & (bc >= 0) & (bc < s)
        out = torch.stack([br, bc], dim=-1)
        out[~valid] = -1
        return out
    assert isinstance(head, Tensor) and isinstance(size, Tensor)
    off = offsets[direction.long()]
    br = head[:, 0, None, None].long() + off[..., 0].long()
    bc = head[:, 1, None, None].long() + off[..., 1].long()
    s = size.long()[:, None, None]
    valid = (br >= 0) & (br < s) & (bc >= 0) & (bc < s)
    out = torch.stack([br, bc], dim=-1)
    out[~valid] = -1
    return out


class BatchedSnakeEnv:
    """Vectorized Snake on a padded square grid [N, G, G]."""

    def __init__(
        self,
        num_envs: int,
        min_size: int,
        max_size: int,
        hunger_factor: float,
        device: torch.device | str,
        seed: int | None = None,
    ) -> None:
        if not 5 <= min_size <= max_size <= 32:
            raise ValueError("board size must satisfy 5 <= min_size <= max_size <= 32")
        self.num_envs = int(num_envs)
        self.min_size = int(min_size)
        self.max_size = int(max_size)
        self.hunger_factor = float(hunger_factor)
        self.device = torch.device(device)
        self.grid = self.max_size
        self._pending_min = self.min_size
        self._pending_max = self.max_size

        self._gen = torch.Generator(device=self.device)
        if seed is not None:
            self._gen.manual_seed(int(seed))
        else:
            self._gen.seed()

        n, g = self.num_envs, self.grid
        self.body = torch.zeros(n, g, g, dtype=torch.int32, device=self.device)
        self.head = torch.zeros(n, 2, dtype=torch.int32, device=self.device)
        self.dir = torch.zeros(n, dtype=torch.int32, device=self.device)
        self.length = torch.zeros(n, dtype=torch.int32, device=self.device)
        self.food = torch.zeros(n, 2, dtype=torch.int32, device=self.device)
        self.size = torch.zeros(n, dtype=torch.int32, device=self.device)
        self.steps_since_food = torch.zeros(n, dtype=torch.int32, device=self.device)
        self.steps = torch.zeros(n, dtype=torch.int32, device=self.device)
        self.score = torch.zeros(n, dtype=torch.int32, device=self.device)

        self._dir_dr = _DIR_DR.to(self.device)
        self._dir_dc = _DIR_DC.to(self.device)
        self._offsets = _rotation_offsets(self.grid, self.device).long()
        self._ep_return_comp = torch.zeros(n, N_COMPONENTS, dtype=torch.float32, device=self.device)
        self._env_ix = torch.arange(n, device=self.device)
        self._rr = torch.arange(g, device=self.device).view(1, g, 1).expand(n, g, g)
        self._cc = torch.arange(g, device=self.device).view(1, 1, g).expand(n, g, g)
        self._in_board = torch.ones(n, g, g, dtype=torch.bool, device=self.device)
        w = 2 * g - 1
        self._obs_env = self._env_ix[:, None, None].expand(n, w, w)
        self._comp_buf = torch.zeros(n, N_COMPONENTS, dtype=torch.float32, device=self.device)
        self._cause_buf = torch.zeros(n, dtype=torch.int8, device=self.device)
        self._i8 = {
            CAUSE_WALL: torch.tensor(CAUSE_WALL, dtype=torch.int8, device=self.device),
            CAUSE_SELF: torch.tensor(CAUSE_SELF, dtype=torch.int8, device=self.device),
            CAUSE_STARVE: torch.tensor(CAUSE_STARVE, dtype=torch.int8, device=self.device),
            CAUSE_WIN: torch.tensor(CAUSE_WIN, dtype=torch.int8, device=self.device),
        }
        self._graph: torch.cuda.CUDAGraph | None = None
        self._graph_actions: Tensor | None = None
        self._graph_obs: Obs | None = None
        self._graph_enabled = False

        self.reset()

    def set_board_sizes(self, min_size: int, max_size: int) -> None:
        if not 5 <= min_size <= max_size <= 32:
            raise ValueError("board size must satisfy 5 <= min_size <= max_size <= 32")
        if max_size > self.grid:
            raise ValueError(f"max_size {max_size} exceeds env grid capacity {self.grid}")
        self._pending_min = int(min_size)
        self._pending_max = int(max_size)
        self.min_size = self._pending_min
        self.max_size = self._pending_max

    def reset(self) -> None:
        self._reset_masked(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))

    def _sample_sizes(self, n: int) -> Tensor:
        lo, hi = self._pending_min, self._pending_max
        if lo == hi:
            return torch.full((n,), lo, dtype=torch.int32, device=self.device)
        if self._graph_enabled:
            return torch.randint(lo, hi + 1, (n,), device=self.device, dtype=torch.int32)
        return torch.randint(lo, hi + 1, (n,), generator=self._gen, device=self.device, dtype=torch.int32)

    def _refresh_in_board(self) -> None:
        self._in_board = (self._rr < self.size[:, None, None]) & (self._cc < self.size[:, None, None])

    def _reset_masked(self, mask: Tensor) -> None:
        """Branchless reset (CUDA-graph friendly)."""
        n = self.num_envs
        sizes = self._sample_sizes(n)
        centers = sizes // 2
        self.body.masked_fill_(mask[:, None, None], 0)
        self.size = torch.where(mask, sizes, self.size)
        self.head[:, 0] = torch.where(mask, centers, self.head[:, 0])
        self.head[:, 1] = torch.where(mask, centers, self.head[:, 1])
        self.dir = torch.where(mask, torch.ones_like(self.dir), self.dir)
        self.length = torch.where(mask, torch.full_like(self.length, 3), self.length)
        self.steps_since_food = torch.where(
            mask, torch.zeros_like(self.steps_since_food), self.steps_since_food
        )
        self.steps = torch.where(mask, torch.zeros_like(self.steps), self.steps)
        self.score = torch.where(mask, torch.zeros_like(self.score), self.score)
        self._ep_return_comp = torch.where(
            mask[:, None], torch.zeros_like(self._ep_return_comp), self._ep_return_comp
        )
        env = self._env_ix
        cr = centers.long()
        for dc, life in ((0, 3), (-1, 2), (-2, 1)):
            c = (centers + dc).long()
            old = self.body[env, cr, c]
            self.body[env, cr, c] = torch.where(mask, torch.full_like(old, life), old)
        self._in_board = (self._rr < self.size[:, None, None]) & (self._cc < self.size[:, None, None])
        self._place_food_masked(mask)

    def _place_food_on(self, idx: Tensor) -> None:
        self._place_food_masked(
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device).scatter_(0, idx, True)
        )

    def _place_food_masked(self, mask: Tensor) -> None:
        """K-candidate food placement (branchless)."""
        n, g, k = self.num_envs, self.grid, 32
        sizes_f = self.size.float()
        if self._graph_enabled:
            rr = (torch.rand(n, k, device=self.device) * sizes_f[:, None]).long().clamp(0, g - 1)
            cc = (torch.rand(n, k, device=self.device) * sizes_f[:, None]).long().clamp(0, g - 1)
        else:
            rr = (
                (torch.rand(n, k, generator=self._gen, device=self.device) * sizes_f[:, None])
                .long()
                .clamp(0, g - 1)
            )
            cc = (
                (torch.rand(n, k, generator=self._gen, device=self.device) * sizes_f[:, None])
                .long()
                .clamp(0, g - 1)
            )
        env_k = self._env_ix[:, None].expand(n, k)
        cand_empty = (self.body[env_k, rr, cc] == 0) & mask[:, None]
        any_hit = cand_empty.any(dim=1)
        chosen = cand_empty.long().argmax(dim=1)
        pick_r = rr.gather(1, chosen[:, None]).squeeze(1).to(torch.int32)
        pick_c = cc.gather(1, chosen[:, None]).squeeze(1).to(torch.int32)
        place = mask & any_hit
        self.food[:, 0] = torch.where(place, pick_r, self.food[:, 0])
        self.food[:, 1] = torch.where(place, pick_c, self.food[:, 1])

        # Fallback when all K candidates miss (not used inside CUDA graphs)
        if not self._graph_enabled:
            miss = mask & ~any_hit
            if bool(miss.any().item()):
                idx = miss.nonzero(as_tuple=False).view(-1)
                m = int(idx.shape[0])
                body = self.body[idx]
                sizes = self.size[idx]
                rows = torch.arange(g, device=self.device).view(1, g, 1).expand(m, g, g)
                cols = torch.arange(g, device=self.device).view(1, 1, g).expand(m, g, g)
                in_b = (rows < sizes[:, None, None]) & (cols < sizes[:, None, None])
                emp = (body == 0) & in_b
                has = emp.flatten(1).any(dim=1)
                if bool(has.any().item()):
                    place_idx = idx[has]
                    emp_h = emp[has]
                    scores = torch.rand(place_idx.shape[0], g, g, generator=self._gen, device=self.device)
                    scores = scores.masked_fill(~emp_h, float("-inf"))
                    pick = scores.flatten(1).argmax(dim=1)
                    self.food[place_idx, 0] = (pick // g).to(torch.int32)
                    self.food[place_idx, 1] = (pick % g).to(torch.int32)

    def step(self, actions: Tensor) -> StepResult:
        if actions.device != self.device or actions.dtype != torch.int64:
            actions = actions.to(device=self.device, dtype=torch.int64).view(-1)
        else:
            actions = actions.view(-1)
        n = self.num_envs
        if actions.shape[0] != n:
            raise ValueError(f"expected {n} actions, got {actions.shape[0]}")

        components = self._comp_buf
        components.zero_()
        components[:, COMP_STEP] = 1.0

        turn = (actions == 1).to(torch.int32) * (-1) + (actions == 2).to(torch.int32)
        self.dir = (self.dir + turn) & 3

        dr = self._dir_dr[self.dir.long()]
        dc = self._dir_dc[self.dir.long()]
        old_r, old_c = self.head[:, 0], self.head[:, 1]
        new_r = old_r + dr
        new_c = old_c + dc

        old_dist = (old_r - self.food[:, 0]).abs() + (old_c - self.food[:, 1]).abs()
        new_dist = (new_r - self.food[:, 0]).abs() + (new_c - self.food[:, 1]).abs()

        wall = (new_r < 0) | (new_c < 0) | (new_r >= self.size) | (new_c >= self.size)
        safe_r = new_r.clamp(0, self.grid - 1).long()
        safe_c = new_c.clamp(0, self.grid - 1).long()
        env_ix = self._env_ix
        self_hit = (~wall) & (self.body[env_ix, safe_r, safe_c] > 1)
        ate = (~wall) & (~self_hit) & (new_r == self.food[:, 0]) & (new_c == self.food[:, 1])
        alive = (~wall) & (~self_hit)

        moved = alive & (~ate)
        components[:, COMP_APPROACH] = moved.float() * (
            (new_dist < old_dist).float() - (new_dist > old_dist).float()
        )
        components[:, COMP_FOOD] = ate.float()

        dec = (alive & (~ate))[:, None, None]
        self.body.sub_(((self.body > 0) & dec).to(torch.int32))
        self.length.add_(ate.to(torch.int32))
        self.body[env_ix, safe_r, safe_c] = torch.where(alive, self.length, self.body[env_ix, safe_r, safe_c])
        nr32, nc32 = new_r.to(torch.int32), new_c.to(torch.int32)
        self.head[:, 0] = torch.where(alive, nr32, self.head[:, 0])
        self.head[:, 1] = torch.where(alive, nc32, self.head[:, 1])

        self.steps += 1
        self.steps_since_food = torch.where(
            ate, torch.zeros_like(self.steps_since_food), self.steps_since_food + 1
        )
        self.score.add_(ate.to(torch.int32))

        # Win if filled; else place new food for eaters that haven't won
        board_cells = self.size * self.size
        win = alive & (self.length >= board_cells)
        self._place_food_masked(ate & (~win))

        hlim = torch.maximum(
            (self.hunger_factor * board_cells.float()).to(torch.int32),
            4 * self.size,
        )
        starve = alive & (~win) & (self.steps_since_food >= hlim)

        cause = self._cause_buf
        cause.zero_()
        cause = torch.where(wall, self._i8[CAUSE_WALL], cause)
        cause = torch.where(self_hit, self._i8[CAUSE_SELF], cause)
        cause = torch.where(starve, self._i8[CAUSE_STARVE], cause)
        cause = torch.where(win, self._i8[CAUSE_WIN], cause)
        self._cause_buf = cause

        dead = wall | self_hit
        components[:, COMP_DEATH] = dead.float()
        components[:, COMP_STARVE] = starve.float()
        components[:, COMP_WIN] = win.float()
        done = dead | starve | win
        self._ep_return_comp.add_(components)

        # Snapshot episode stats before auto-reset mutates state
        if self._graph_enabled:
            # CUDA-graph path: clones are captured once; skip host-side packaging cost
            result = StepResult(
                components=components,
                done=done,
                cause=cause,
                ep_score=self.score.float(),
                ep_length=self.length.float(),
                ep_steps=self.steps.float(),
                ep_size=self.size.float(),
                ep_return_components=self._ep_return_comp,
            )
        else:
            result = StepResult(
                components=components.clone(),
                done=done.clone(),
                cause=cause.clone(),
                ep_score=self.score.float().clone(),
                ep_length=self.length.float().clone(),
                ep_steps=self.steps.float().clone(),
                ep_size=self.size.float().clone(),
                ep_return_components=self._ep_return_comp.clone(),
            )
        self._reset_masked(done)
        return result

    def warm_cuda_graph(self) -> None:
        """Capture a CUDA graph for step+observe (large-N throughput path)."""
        if self.device.type != "cuda":
            return
        self._graph_actions = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._graph_enabled = True
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.step(self._graph_actions)
                self.observe()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.step(self._graph_actions)
            self._graph_obs = self.observe()
        self._graph = g

    def step_observe(self, actions: Tensor) -> Obs:
        """Step then observe; uses CUDA graph when warmed for peak throughput."""
        if self._graph is not None and self._graph_actions is not None:
            if actions.device != self.device or actions.dtype != torch.int64:
                actions = actions.to(device=self.device, dtype=torch.int64)
            self._graph_actions.copy_(actions.view(-1))
            self._graph.replay()
            assert self._graph_obs is not None
            return self._graph_obs
        self.step(actions)
        return self.observe()

    def observe(self) -> Obs:
        g = self.grid
        off = self._offsets[self.dir.long()]
        head_r = self.head[:, 0, None, None].long()
        head_c = self.head[:, 1, None, None].long()
        br = head_r + off[..., 0]
        bc = head_c + off[..., 1]
        sizes = self.size.long()[:, None, None]
        in_board = (br >= 0) & (br < sizes) & (bc >= 0) & (bc < sizes)
        br_c = br.clamp(0, g - 1)
        bc_c = bc.clamp(0, g - 1)
        gathered = self.body[self._obs_env, br_c, bc_c].to(torch.float32)
        gathered = gathered * in_board.float()

        length_f = self.length.float().clamp(min=1.0)[:, None, None]
        body_ch = torch.where(gathered > 0, gathered / length_f, gathered)
        head_here = (br == head_r) & (bc == head_c) & in_board
        body_ch = torch.where(head_here, torch.ones_like(body_ch), body_ch)
        food_here = (
            (br == self.food[:, 0, None, None].long()) & (bc == self.food[:, 1, None, None].long()) & in_board
        )
        grid = torch.stack(
            [(~in_board).float(), body_ch, food_here.float(), (gathered == 1).float()],
            dim=1,
        )
        area = (self.size * self.size).float().clamp(min=1.0)
        hlim = (
            torch.maximum(
                (self.hunger_factor * area).to(torch.int32),
                4 * self.size,
            )
            .float()
            .clamp(min=1.0)
        )
        scalars = torch.stack(
            [
                self.length.float() / area,
                self.steps_since_food.float() / hlim,
                self.size.float() * (1.0 / 32.0),
                self.score.float() / area,
            ],
            dim=1,
        )
        return Obs(grid=grid, scalars=scalars)

    def snapshot(self, idx: int) -> dict[str, Any]:
        i = int(idx)
        size = int(self.size[i].item())
        body = self.body[i, :size, :size].cpu()
        length = int(self.length[i].item())
        cells: list[tuple[int, int, int]] = []
        for r in range(size):
            for c in range(size):
                v = int(body[r, c].item())
                if v > 0:
                    cells.append((v, r, c))
        cells.sort(key=lambda t: -t[0])
        snake = [[r, c] for _, r, c in cells]
        return {
            "size": size,
            "body": snake,
            "food": [int(self.food[i, 0].item()), int(self.food[i, 1].item())],
            "dir": int(self.dir[i].item()),
            "score": int(self.score[i].item()),
            "steps": int(self.steps[i].item()),
            "steps_since_food": int(self.steps_since_food[i].item()),
            "length": length,
        }

    def load_snapshot(self, idx: int, snap: dict[str, Any]) -> None:
        i = int(idx)
        size = int(snap["size"])
        if size > self.grid:
            raise ValueError(f"snapshot size {size} exceeds grid {self.grid}")
        body = snap["body"]
        length = len(body)
        self.body[i] = 0
        for k, (r, c) in enumerate(body):
            self.body[i, int(r), int(c)] = length - k
        self.size[i] = size
        self.length[i] = length
        self.head[i, 0] = int(body[0][0])
        self.head[i, 1] = int(body[0][1])
        self.food[i, 0] = int(snap["food"][0])
        self.food[i, 1] = int(snap["food"][1])
        self.dir[i] = int(snap["dir"])
        self.score[i] = int(snap.get("score", length - 3))
        self.steps[i] = int(snap.get("steps", 0))
        self.steps_since_food[i] = int(snap.get("steps_since_food", 0))
        self._ep_return_comp[i] = 0
        self._refresh_in_board()
