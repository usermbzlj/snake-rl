"""Environment rules, observation, and reward component tests."""

from __future__ import annotations

import torch

from snake_rl.core.env import (
    CAUSE_SELF,
    CAUSE_STARVE,
    CAUSE_WALL,
    CAUSE_WIN,
    COMP_APPROACH,
    COMP_DEATH,
    COMP_FOOD,
    COMP_STARVE,
    COMP_STEP,
    COMP_WIN,
    BatchedSnakeEnv,
    hunger_limit,
    window_to_board,
)


def _cpu_env(n: int = 1, size: int = 8, seed: int = 0) -> BatchedSnakeEnv:
    return BatchedSnakeEnv(n, size, size, 1.0, device="cpu", seed=seed)


def test_tail_follow_legal() -> None:
    env = _cpu_env(size=6)
    # Straight snake horizontal, head facing right; circling so head enters leaving tail.
    # body head->tail: (2,3),(2,2),(2,1) dir RIGHT — move right to (2,4)
    # Build a loop: length 4 in a square so next move onto current tail.
    env.load_snapshot(
        0,
        {
            "size": 6,
            "body": [[2, 2], [2, 1], [3, 1], [3, 2]],  # head at (2,2) facing...?
            "food": [0, 0],
            "dir": 1,  # RIGHT — next (2,3) empty
            "score": 1,
            "steps": 0,
            "steps_since_food": 0,
        },
    )
    # Reconfigure: snake of length 4 going right into its tail cell.
    # Positions head->tail: (1,2),(1,1),(2,1),(2,2), dir UP — straight goes to (0,2) not tail.
    # Classic: head (2,2), body (2,1),(2,0),(1,0),(1,1),(1,2) length 6, dir LEFT
    # Tail at (1,2) with life=1. Head facing left toward (2,1) occupied >1 — self.
    # Setup: head (2,3), body (2,2),(2,1),(1,1),(1,2),(1,3) dir RIGHT
    # Tail (1,3)==1. Straight -> (2,4) empty.
    # Want: head moves onto tail. head (1,3), body (1,2),(1,1),(2,1),(2,2),(2,3), dir RIGHT
    # Tail is (2,3). Straight -> (1,4).
    # head (2,3) dir DOWN, body (1,3),(1,2),(2,2),(2,3) wait duplicate.
    # Simple: length 4 square. head (1,1), (1,2), (2,2), (2,1), dir LEFT.
    # Tail=(2,1) life=1. Straight left -> (1,0).
    # Turn right (from LEFT -> UP): new head (0,1).
    # Turn left (from LEFT -> DOWN): new head (2,1) == current tail. Legal!
    env.load_snapshot(
        0,
        {
            "size": 6,
            "body": [[1, 1], [1, 2], [2, 2], [2, 1]],
            "food": [5, 5],
            "dir": 3,  # LEFT
            "score": 1,
            "steps": 10,
            "steps_since_food": 0,
        },
    )
    # action 1 = turn left -> DOWN, new head (2,1) which is tail
    result = env.step(torch.tensor([1]))
    assert not bool(result.done[0].item()), "moving onto vacating tail must be legal"
    assert int(result.cause[0].item()) == 0


def test_self_collision() -> None:
    env = _cpu_env(size=6)
    # head (2,2), body (2,1),(2,0),(1,0),(1,1),(1,2),(2,2) — wait
    # head (1,1) facing UP, body includes (0,1) with life>1
    env.load_snapshot(
        0,
        {
            "size": 6,
            "body": [[2, 2], [2, 1], [2, 0], [1, 0], [1, 1], [1, 2]],
            "food": [5, 5],
            "dir": 0,  # UP — into (1,2) which has life=1 (tail) — legal actually
            "score": 3,
            "steps": 0,
            "steps_since_food": 0,
        },
    )
    # Tail is last = (1,2). UP from (2,2) -> (1,2) == tail, legal.
    # Use dir LEFT into (2,1) which is body life=5 > 1
    env.load_snapshot(
        0,
        {
            "size": 6,
            "body": [[2, 2], [2, 1], [2, 0], [1, 0], [1, 1], [1, 2]],
            "food": [5, 5],
            "dir": 3,  # LEFT into (2,1) life=5
            "score": 3,
            "steps": 0,
            "steps_since_food": 0,
        },
    )
    result = env.step(torch.tensor([0]))  # straight
    assert bool(result.done[0].item())
    assert int(result.cause[0].item()) == CAUSE_SELF
    assert float(result.components[0, COMP_DEATH].item()) == 1.0


def test_wall_death() -> None:
    env = _cpu_env(size=5)
    env.load_snapshot(
        0,
        {
            "size": 5,
            "body": [[0, 2], [0, 1], [0, 0]],
            "food": [4, 4],
            "dir": 0,  # UP into wall
            "score": 0,
            "steps": 0,
            "steps_since_food": 0,
        },
    )
    result = env.step(torch.tensor([0]))
    assert bool(result.done[0].item())
    assert int(result.cause[0].item()) == CAUSE_WALL


def test_eating_and_growth() -> None:
    env = _cpu_env(size=8)
    env.load_snapshot(
        0,
        {
            "size": 8,
            "body": [[4, 4], [4, 3], [4, 2]],
            "food": [4, 5],
            "dir": 1,  # RIGHT onto food
            "score": 0,
            "steps": 0,
            "steps_since_food": 5,
        },
    )
    before = int(env.length[0].item())
    result = env.step(torch.tensor([0]))
    assert float(result.components[0, COMP_FOOD].item()) == 1.0
    assert float(result.components[0, COMP_APPROACH].item()) == 0.0
    assert int(env.length[0].item()) == before + 1 or bool(result.done[0].item())
    # After auto-reset only if done; otherwise length grew and score++
    if not bool(result.done[0].item()):
        assert int(env.score[0].item()) == 1
        assert int(env.steps_since_food[0].item()) == 0
        assert int(env.length[0].item()) == before + 1


def test_starve() -> None:
    env = BatchedSnakeEnv(1, 5, 5, hunger_factor=0.1, device="cpu", seed=0)
    # hunger_limit = max(int(0.1*25), 20) = max(2, 20) = 20
    lim = int(hunger_limit(5, 0.1))
    assert lim == 20
    env.load_snapshot(
        0,
        {
            "size": 5,
            "body": [[2, 2], [2, 1], [2, 0]],
            "food": [4, 4],
            "dir": 1,
            "score": 0,
            "steps": 100,
            "steps_since_food": lim - 1,
        },
    )
    # One safe step straight (not into wall/food necessarily)
    # head (2,2) right -> (2,3) empty
    result = env.step(torch.tensor([0]))
    assert bool(result.done[0].item())
    assert int(result.cause[0].item()) == CAUSE_STARVE
    assert float(result.components[0, COMP_STARVE].item()) == 1.0


def test_win_tiny_board() -> None:
    env = _cpu_env(size=5)
    # Fill almost all: length 24, one empty with food, eat to win
    cells = []
    for r in range(5):
        for c in range(5):
            cells.append([r, c])
    # Remove last cell for food; snake uses 24 cells
    food = cells.pop()
    # Order head->tail: need contiguous snake — for load_snapshot only life counters matter
    env.load_snapshot(
        0,
        {
            "size": 5,
            "body": cells,  # 24 cells
            "food": food,
            "dir": 1,
            "score": 21,
            "steps": 0,
            "steps_since_food": 0,
        },
    )
    # Place food at (0,1), head at (0,0) facing RIGHT, body fills rest
    body = [[0, 0]]
    for r in range(5):
        for c in range(5):
            if (r, c) != (0, 0) and (r, c) != (0, 1):
                body.append([r, c])
    env.load_snapshot(
        0,
        {
            "size": 5,
            "body": body,
            "food": [0, 1],
            "dir": 1,
            "score": 21,
            "steps": 0,
            "steps_since_food": 0,
        },
    )
    assert int(env.length[0].item()) == 24
    result = env.step(torch.tensor([0]))
    assert bool(result.done[0].item())
    assert int(result.cause[0].item()) == CAUSE_WIN
    assert float(result.components[0, COMP_WIN].item()) == 1.0
    assert float(result.components[0, COMP_FOOD].item()) == 1.0


def test_auto_reset() -> None:
    env = _cpu_env(size=5, seed=1)
    env.load_snapshot(
        0,
        {
            "size": 5,
            "body": [[0, 2], [0, 1], [0, 0]],
            "food": [4, 4],
            "dir": 0,
            "score": 7,
            "steps": 50,
            "steps_since_food": 3,
        },
    )
    result = env.step(torch.tensor([0]))
    assert bool(result.done[0].item())
    assert int(result.ep_score[0].item()) == 7
    # After reset: fresh game
    assert int(env.length[0].item()) == 3
    assert int(env.score[0].item()) == 0
    assert int(env.steps[0].item()) == 0


def test_reward_component_math() -> None:
    env = _cpu_env(size=8)
    env.load_snapshot(
        0,
        {
            "size": 8,
            "body": [[4, 4], [4, 3], [4, 2]],
            "food": [4, 6],
            "dir": 1,
            "score": 0,
            "steps": 0,
            "steps_since_food": 0,
        },
    )
    # Move closer to food
    result = env.step(torch.tensor([0]))
    c = result.components[0]
    assert float(c[COMP_STEP].item()) == 1.0
    assert float(c[COMP_APPROACH].item()) == 1.0
    assert float(c[COMP_FOOD].item()) == 0.0
    weights = torch.tensor([1.0, -1.0, -0.005, 0.05, -0.5, 5.0])
    r = float((c * weights).sum().item())
    assert abs(r - (-0.005 + 0.05)) < 1e-5


def test_window_to_board_mapping() -> None:
    grid = 8
    # dir RIGHT, head (4,4), size 8
    mapping = window_to_board(1, (4, 4), 8, grid)
    center = grid - 1
    # Window center maps to head
    assert mapping[center, center, 0].item() == 4
    assert mapping[center, center, 1].item() == 4
    # Window cell above center (center-1, center) is "forward" = RIGHT = (4, 5)
    assert mapping[center - 1, center, 0].item() == 4
    assert mapping[center - 1, center, 1].item() == 5
    # Left in window = UP on board when facing right
    assert mapping[center, center - 1, 0].item() == 3
    assert mapping[center, center - 1, 1].item() == 4


def _naive_observe(env: BatchedSnakeEnv, idx: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference egocentric observation for one env."""
    g = env.grid
    w = 2 * g - 1
    size = int(env.size[idx].item())
    head_r = int(env.head[idx, 0].item())
    head_c = int(env.head[idx, 1].item())
    d = int(env.dir[idx].item())
    length = int(env.length[idx].item())
    food = (int(env.food[idx, 0].item()), int(env.food[idx, 1].item()))
    body = env.body[idx].cpu()

    def rot(di: int, dj: int) -> tuple[int, int]:
        if d == 0:
            return di, dj
        if d == 1:
            return dj, -di
        if d == 2:
            return -di, -dj
        return -dj, di

    grid = torch.zeros(4, w, w)
    center = g - 1
    for i in range(w):
        for j in range(w):
            di, dj = i - center, j - center
            br, bc = head_r + rot(di, dj)[0], head_c + rot(di, dj)[1]
            if not (0 <= br < size and 0 <= bc < size):
                grid[0, i, j] = 1.0
                continue
            life = int(body[br, bc].item())
            if life > 0:
                grid[1, i, j] = 1.0 if (br == head_r and bc == head_c) else life / length
            if (br, bc) == food:
                grid[2, i, j] = 1.0
            if life == 1:
                grid[3, i, j] = 1.0

    hlim = float(hunger_limit(size, env.hunger_factor))
    area = float(size * size)
    scalars = torch.tensor(
        [
            length / area,
            int(env.steps_since_food[idx].item()) / hlim,
            size / 32.0,
            int(env.score[idx].item()) / area,
        ]
    )
    return grid, scalars


def test_observation_matches_naive() -> None:
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        env = BatchedSnakeEnv(4, 6, 10, 1.0, device=device, seed=42)
        for _ in range(5):
            actions = torch.randint(0, 3, (4,), device=device)
            env.step(actions)
        obs = env.observe()
        for i in range(4):
            ref_g, ref_s = _naive_observe(env, i)
            got_g = obs.grid[i].cpu()
            got_s = obs.scalars[i].cpu()
            assert torch.allclose(got_g, ref_g, atol=1e-5), f"grid mismatch env {i} on {device}"
            assert torch.allclose(got_s, ref_s, atol=1e-5), f"scalars mismatch env {i} on {device}"


def test_cpu_small_batch() -> None:
    for n in (1, 4, 16):
        env = BatchedSnakeEnv(n, 6, 8, 1.0, device="cpu", seed=0)
        for _ in range(10):
            env.step(torch.randint(0, 3, (n,)))
            obs = env.observe()
            assert obs.grid.shape[0] == n
