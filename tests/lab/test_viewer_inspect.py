"""Viewer and inspect unit tests."""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path

import torch

from snake_rl.core.checkpoint import save_checkpoint
from snake_rl.core.config import EnvConfig, ExperimentConfig, ModelConfig, PPOConfig, RunConfig
from snake_rl.core.env import BatchedSnakeEnv
from snake_rl.core.network import SnakeNet
from snake_rl.core.trainer import make_trainer
from snake_rl.lab.inspect import compare, inspect_episode
from snake_rl.lab.manager import ExperimentManager
from snake_rl.lab.storage import ExperimentStore
from snake_rl.lab.viewer import WatchSession, _death_frame


def _tiny_cfg() -> ExperimentConfig:
    return ExperimentConfig(
        name="inspect-tiny",
        algo="ppo",
        env=EnvConfig(min_size=6, max_size=6),
        model=ModelConfig(width=0.5),
        ppo=PPOConfig(num_envs=4, rollout=8, epochs=1, minibatches=1),
        run=RunConfig(device="cpu", seed=0, eval_every_s=999),
    )


def _make_ckpt(path: Path) -> Path:
    cfg = _tiny_cfg()
    trainer = make_trainer(cfg)
    # one tiny iteration optional — random weights fine
    save_checkpoint(path, trainer_state=trainer.state_dict(), config=cfg, meta={"env_steps": 0})
    return path


def test_inspect_saliency(tmp_path: Path):
    ckpt = _make_ckpt(tmp_path / "latest.pt")
    traj = inspect_episode(ckpt, board_size=6, seed=1, greedy=True, max_steps=50, saliency=True)
    assert traj["board_size"] == 6
    assert traj["seed"] == 1
    assert len(traj["steps"]) >= 1
    assert traj["saliency"] is not None
    assert len(traj["saliency"]) == len(traj["steps"])
    assert len(traj["saliency"][0]) == 6 * 6
    assert all(0.0 <= v <= 1.0 for v in traj["saliency"][0])
    step0 = traj["steps"][0]
    assert len(step0["probs"]) == 3
    assert len(step0["reward_components"]) == 11


def test_compare(tmp_path: Path):
    a = _make_ckpt(tmp_path / "a.pt")
    b = _make_ckpt(tmp_path / "b.pt")
    trajs = compare([(a, "A"), (b, "B")], board_size=6, seed=2)
    assert len(trajs) == 2
    assert trajs[0]["saliency"] is None
    assert trajs[0]["name"] == "A"


def test_viewer_produces_frames(tmp_path: Path):
    store = ExperimentStore(tmp_path)
    mgr = ExperimentManager(store)
    mgr.startup()
    cfg = _tiny_cfg()
    meta = store.create(cfg)
    eid = meta["id"]
    # Seed weights into manager
    net = SnakeNet(mode="ppo", width=0.5)
    mgr._latest_weights[eid] = (1, net.state_dict(), 0)

    frames: list[dict] = []

    async def send_json(obj: dict) -> None:
        frames.append(obj)

    session = WatchSession(mgr, eid, send_json=send_json)
    session.games = 1
    session.board_size = 6
    session.speed = 30

    async def run_few():
        task = asyncio.create_task(session.run())
        await asyncio.sleep(0.8)
        session.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(run_few())
    mgr.shutdown()

    frame_msgs = [f for f in frames if f.get("type") == "frame"]
    assert len(frame_msgs) >= 1
    g = frame_msgs[0]["games"][0]
    assert "snake" in g and "probs" in g and len(g["probs"]) == 3
    assert "food" in g and "value" in g


def test_watch_hold_survives_out_of_board_death_frame(tmp_path: Path):
    store = ExperimentStore(tmp_path)
    session = WatchSession(ExperimentManager(store), "unused", send_json=lambda _obj: _async_none())
    session.games = 1
    session.board_size = 8
    session._algo = "ppo"
    session._width = 0.5
    env = BatchedSnakeEnv(1, 8, 8, 1.0, torch.device("cpu"), seed=0)
    session._env = env
    net = SnakeNet(mode="ppo", width=0.5)
    net.eval()
    session._net = net
    session._model_version = 1
    pre = {
        "size": 8,
        "body": [[4, 7], [4, 6], [4, 5]],
        "food": [0, 0],
        "dir": 1,
        "score": 0,
        "steps": 10,
        "steps_since_food": 1,
    }
    final = _death_frame(pre, 0, 1, 0)
    session._holding[0] = (time.perf_counter() + 30, final, 1, [1.0, 0.0, 0.0], 0.0, 0)
    frame = session._step_frame(env, net)
    assert frame["games"][0]["dead"] is True
    snap = env.snapshot(0)
    assert all(0 <= r < 8 and 0 <= c < 8 for r, c in snap["body"])


async def _async_none() -> None:
    return None
