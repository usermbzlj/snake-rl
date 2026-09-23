"""ExperimentManager lifecycle with a real worker process."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import pytest

from snake_rl.core.config import EnvConfig, ExperimentConfig, ModelConfig, PPOConfig, RunConfig
from snake_rl.lab.manager import ExperimentManager
from snake_rl.lab.storage import ExperimentStore


def _tiny_ppo(device: Literal["auto", "cuda", "cpu"] = "cpu") -> ExperimentConfig:
    return ExperimentConfig(
        name="tiny-ppo",
        algo="ppo",
        env=EnvConfig(min_size=6, max_size=6),
        model=ModelConfig(width=0.5),
        ppo=PPOConfig(num_envs=8, rollout=16, epochs=1, minibatches=1, lr=3e-4),
        run=RunConfig(device=device, eval_every_s=5.0, max_env_steps=0, seed=0),
    )


def _wait_metrics(store: ExperimentStore, eid: str, *, min_rows: int = 1, timeout: float = 60.0) -> list:
    t0 = time.time()
    while time.time() - t0 < timeout:
        rows = store.read_metrics(eid)
        if len(rows) >= min_rows:
            return rows
        time.sleep(0.2)
    pytest.fail(f"metrics did not arrive within {timeout}s (got {len(store.read_metrics(eid))})")


@pytest.fixture()
def manager(tmp_path: Path):
    store = ExperimentStore(tmp_path)
    mgr = ExperimentManager(store)
    mgr.startup()
    yield mgr
    mgr.shutdown()


def test_manager_lifecycle(manager: ExperimentManager):
    cfg = _tiny_ppo("cpu")
    summary = manager.create(cfg, start=True)
    eid = summary["id"]
    assert summary["status"] in ("running", "created")

    rows = _wait_metrics(manager.store, eid, min_rows=1, timeout=90.0)
    assert rows[0]["env_steps"] > 0
    steps1 = int(rows[-1]["env_steps"])

    # live patch
    res = manager.live_patch(eid, {"reward.food": 2.0})
    assert res["config"]["reward"]["food"] == 2.0

    # wait a bit for event
    time.sleep(0.5)
    events = manager.store.read_events(eid)
    assert any(e["type"] == "live_patch" for e in events)

    manager.pause(eid)
    time.sleep(0.8)
    # status should become paused (async via worker)
    t0 = time.time()
    while time.time() - t0 < 15:
        st = manager.store.read_meta(eid)["status"]
        if st == "paused":
            break
        time.sleep(0.2)
    assert manager.store.read_meta(eid)["status"] == "paused"

    manager.resume(eid)
    t0 = time.time()
    while time.time() - t0 < 15:
        if manager.store.read_meta(eid)["status"] == "running":
            break
        time.sleep(0.2)

    manager.stop(eid, wait=True, timeout=30.0)
    assert manager.store.read_meta(eid)["status"] == "stopped"
    assert manager.store.ckpt_path(eid, "latest").is_file()

    steps_before = int(manager.store.read_meta(eid).get("env_steps") or steps1)

    # continue
    manager.start(eid)
    rows2 = _wait_metrics(manager.store, eid, min_rows=len(rows) + 1, timeout=90.0)
    assert int(rows2[-1]["env_steps"]) >= steps_before

    manager.stop(eid, wait=True, timeout=30.0)

    cloned = manager.clone(eid, name="clone-me", with_weights=True)
    assert cloned["name"] == "clone-me"
    assert manager.store.ckpt_path(cloned["id"], "latest").is_file()

    manager.delete(eid)
    assert not manager.store.exists(eid)
    manager.delete(cloned["id"])
