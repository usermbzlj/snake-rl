"""Viewer and inspect unit tests."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

from snake_rl.core.checkpoint import save_checkpoint
from snake_rl.core.config import EnvConfig, ExperimentConfig, ModelConfig, PPOConfig, RunConfig
from snake_rl.core.network import SnakeNet
from snake_rl.core.trainer import make_trainer
from snake_rl.lab.inspect import compare, inspect_episode
from snake_rl.lab.manager import ExperimentManager
from snake_rl.lab.storage import ExperimentStore
from snake_rl.lab.viewer import WatchSession


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
    assert len(step0["reward_components"]) == 6


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
