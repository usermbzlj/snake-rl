"""Episode inspect with saliency and multi-experiment compare."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from snake_rl.core.checkpoint import load_checkpoint
from snake_rl.core.config import ExperimentConfig
from snake_rl.core.env import BatchedSnakeEnv, window_to_board
from snake_rl.core.network import SnakeNet


def _load_net_from_checkpoint(path: str | Path) -> tuple[SnakeNet, ExperimentConfig, dict[str, Any]]:
    data = load_checkpoint(path)
    config = data["config"]
    if not isinstance(config, ExperimentConfig):
        config = ExperimentConfig.model_validate(config)
    trainer = data["trainer"]
    if "net" in trainer:
        sd = trainer["net"]
    elif "online" in trainer:
        sd = trainer["online"]
    else:
        raise ValueError("检查点中没有可用的网络权重")
    net = SnakeNet(mode=config.algo, width=config.model.width)
    net.load_state_dict(sd)
    net.eval()
    return net, config, trainer


def inspect_episode(
    checkpoint_path: str | Path,
    board_size: int,
    seed: int | None = None,
    greedy: bool = True,
    max_steps: int = 5000,
    saliency: bool = True,
) -> dict[str, Any]:
    net, config, _trainer = _load_net_from_checkpoint(checkpoint_path)
    if seed is None:
        seed = 0
    device = torch.device("cpu")
    env = BatchedSnakeEnv(
        num_envs=1,
        min_size=board_size,
        max_size=board_size,
        hunger_factor=config.env.hunger_factor,
        device=device,
        seed=seed,
    )
    env.reset()

    steps_out: list[dict[str, Any]] = []
    grids: list[torch.Tensor] = []
    scalars: list[torch.Tensor] = []
    actions_list: list[int] = []
    dirs: list[int] = []
    heads: list[list[int]] = []
    sizes: list[int] = []

    result_cause = 0
    result_score = 0
    result_steps = 0

    for _ in range(max_steps):
        snap = env.snapshot(0)
        obs = env.observe()
        grid = obs.grid.clone()
        sc = obs.scalars.clone()

        with torch.no_grad():
            logits_or_q, value = net(grid, sc)
            logits_or_q = logits_or_q.float()
            value = value.float()
            if config.algo == "ppo":
                probs = F.softmax(logits_or_q, dim=-1)
                action = int(probs.argmax(dim=-1).item() if greedy else torch.multinomial(probs, 1).item())
                q_out = None
            else:
                q = logits_or_q
                probs = F.softmax(q, dim=-1)
                action = int(q.argmax(dim=-1).item() if greedy else torch.multinomial(probs, 1).item())
                q_out = [float(x) for x in q[0].tolist()]

        step = env.step(torch.tensor([action], dtype=torch.long))
        comps = [float(x) for x in step.components[0].tolist()]

        entry: dict[str, Any] = {
            "snake": snap["body"],
            "food": snap["food"],
            "dir": snap["dir"],
            "action": action,
            "probs": [float(x) for x in probs[0].tolist()],
            "value": float(value[0].item()),
            "reward_components": comps,
            "score": snap["score"],
        }
        if q_out is not None:
            entry["q"] = q_out
        steps_out.append(entry)

        grids.append(grid[0].cpu())
        scalars.append(sc[0].cpu())
        actions_list.append(action)
        dirs.append(int(snap["dir"]))
        heads.append([int(snap["body"][0][0]), int(snap["body"][0][1])])
        sizes.append(int(snap["size"]))

        if bool(step.done[0].item()):
            result_cause = int(step.cause[0].item())
            result_score = int(step.ep_score[0].item())
            result_steps = int(step.ep_steps[0].item())
            break
    else:
        result_score = int(env.score[0].item())
        result_steps = int(env.steps[0].item())
        result_cause = 0

    sal: list[list[float]] | None = None
    if saliency and steps_out:
        sal = _compute_saliency(
            net,
            config.algo,
            grids,
            scalars,
            actions_list,
            dirs,
            heads,
            sizes,
            grid_cap=env.grid,
            board_size=board_size,
        )

    return {
        "board_size": board_size,
        "seed": seed,
        "steps": steps_out,
        "result": {"score": result_score, "cause": result_cause, "steps": result_steps},
        "saliency": sal,
    }


def _compute_saliency(
    net: SnakeNet,
    algo: str,
    grids: list[torch.Tensor],
    scalars: list[torch.Tensor],
    actions: list[int],
    dirs: list[int],
    heads: list[list[int]],
    sizes: list[int],
    *,
    grid_cap: int,
    board_size: int,
    batch_size: int = 32,
) -> list[list[float]]:
    net.eval()
    n = len(grids)
    out: list[list[float] | None] = [None] * n
    s = board_size

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        g = torch.stack(grids[start:end]).requires_grad_(True)
        sc = torch.stack(scalars[start:end])
        acts = actions[start:end]

        logits_or_q, _value = net(g, sc)
        logits_or_q = logits_or_q.float()
        chosen = logits_or_q[torch.arange(end - start), acts].sum()
        net.zero_grad(set_to_none=True)
        if g.grad is not None:
            g.grad = None
        chosen.backward()
        assert g.grad is not None
        # abs-sum over channels → [B, W, W]
        heat = g.grad.abs().sum(dim=1)

        for bi in range(end - start):
            gi = start + bi
            mapping = window_to_board(dirs[gi], (heads[gi][0], heads[gi][1]), sizes[gi], grid_cap)
            # mapping: [W, W, 2]
            board = torch.zeros(s * s, dtype=torch.float32)
            hmap = heat[bi]
            w = hmap.shape[0]
            for i in range(w):
                for j in range(w):
                    br = int(mapping[i, j, 0].item())
                    bc = int(mapping[i, j, 1].item())
                    if br < 0 or bc < 0 or br >= s or bc >= s:
                        continue
                    board[br * s + bc] += float(hmap[i, j].item())
            mx = float(board.max().item())
            if mx > 0:
                board = board / mx
            out[gi] = [round(float(x), 3) for x in board.tolist()]

    return [x if x is not None else [0.0] * (s * s) for x in out]


def compare(
    entries: Sequence[tuple[str | Path, str]],
    board_size: int,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """entries: list of (checkpoint_path, label). Same seed, no saliency."""
    if seed is None:
        seed = 0
    trajs: list[dict[str, Any]] = []
    for path, label in entries:
        traj = inspect_episode(path, board_size, seed=seed, greedy=True, saliency=False)
        traj["name"] = label
        trajs.append(traj)
    return trajs
