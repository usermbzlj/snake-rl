"""FastAPI TestClient coverage for REST + WS + SPA."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import pytest
from fastapi.testclient import TestClient

from snake_rl.core.config import EnvConfig, ExperimentConfig, ModelConfig, PPOConfig, RunConfig, get_preset
from snake_rl.lab.manager import ExperimentManager
from snake_rl.lab.storage import ExperimentStore
from snake_rl.server.app import create_app


def _tiny(device: Literal["auto", "cuda", "cpu"] = "cpu") -> ExperimentConfig:
    return ExperimentConfig(
        name="api-tiny",
        algo="ppo",
        env=EnvConfig(min_size=6, max_size=6),
        model=ModelConfig(width=0.5),
        ppo=PPOConfig(num_envs=8, rollout=16, epochs=1, minibatches=1),
        run=RunConfig(device=device, eval_every_s=8.0, seed=1),
    )


@pytest.fixture()
def client(tmp_path: Path):
    store = ExperimentStore(tmp_path)
    mgr = ExperimentManager(store)
    app = create_app(manager=mgr, experiments_root=tmp_path, port=7860)
    with TestClient(app) as c:
        yield c, mgr, store


def test_meta_and_schema(client):
    c, _mgr, _store = client
    r = c.get("/api/meta")
    assert r.status_code == 200
    body = r.json()
    assert "version" in body and "device" in body and "port" in body
    r = c.get("/api/config-schema")
    assert r.status_code == 200
    schema = r.json()
    assert "groups" in schema and "presets" in schema
    assert len(schema["presets"]) >= 1


def test_spa_fallback(client):
    c, _mgr, _store = client
    r = c.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    r = c.get("/exp/does-not-matter")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")


def test_crud_and_actions(client):
    c, _mgr, store = client
    cfg = _tiny()
    r = c.post("/api/experiments", json={"config": cfg.model_dump(), "start": True})
    assert r.status_code == 200, r.text
    summary = r.json()
    eid = summary["id"]

    r = c.get("/api/experiments")
    assert r.status_code == 200
    assert any(x["id"] == eid for x in r.json())

    # wait for some metrics
    t0 = time.time()
    while time.time() - t0 < 60:
        detail = c.get(f"/api/experiments/{eid}").json()
        if detail["metrics"]:
            break
        time.sleep(0.3)
    assert c.get(f"/api/experiments/{eid}").json()["metrics"]

    r = c.patch(f"/api/experiments/{eid}/live", json={"patch": {"reward.approach": 0.1}})
    assert r.status_code == 200, r.text
    assert r.json()["config"]["reward"]["approach"] == 0.1

    r = c.post(f"/api/experiments/{eid}/pause")
    assert r.status_code == 200
    time.sleep(0.5)
    r = c.post(f"/api/experiments/{eid}/resume")
    assert r.status_code == 200

    r = c.post(f"/api/experiments/{eid}/stop")
    assert r.status_code == 200
    time.sleep(0.5)

    # start again (continue)
    r = c.post(f"/api/experiments/{eid}/start")
    assert r.status_code == 200
    time.sleep(1.0)
    c.post(f"/api/experiments/{eid}/stop")

    # Ensure checkpoint exists for inspect
    t0 = time.time()
    while time.time() - t0 < 20:
        if store.ckpt_path(eid, "latest").is_file():
            break
        time.sleep(0.2)

    r = c.get(f"/api/experiments/{eid}/checkpoints")
    assert r.status_code == 200

    if store.ckpt_path(eid, "latest").is_file():
        r = c.post(
            f"/api/experiments/{eid}/inspect",
            json={"checkpoint": "latest", "board_size": 6, "seed": 3, "greedy": True},
        )
        assert r.status_code == 200, r.text
        traj = r.json()
        assert traj["saliency"] is not None
        assert len(traj["saliency"][0]) == 36

    r = c.post(f"/api/experiments/{eid}/clone", json={"name": "cloned", "with_weights": True})
    assert r.status_code == 200
    eid2 = r.json()["id"]

    # compare needs 2 ckpts
    if store.ckpt_path(eid, "latest").is_file() and store.ckpt_path(eid2, "latest").is_file():
        r = c.post(
            "/api/compare",
            json={
                "entries": [
                    {"experiment_id": eid, "checkpoint": "latest"},
                    {"experiment_id": eid2, "checkpoint": "latest"},
                ],
                "board_size": 6,
                "seed": 1,
            },
        )
        assert r.status_code == 200, r.text
        assert len(r.json()["trajectories"]) == 2

    r = c.delete(f"/api/experiments/{eid}")
    assert r.status_code == 204
    r = c.delete(f"/api/experiments/{eid2}")
    assert r.status_code == 204

    r = c.get("/api/experiments/nope")
    assert r.status_code == 404
    assert "detail" in r.json()


def test_websockets(client):
    c, _mgr, _store = client
    cfg = _tiny()
    # create without start, inject weights for watch
    r = c.post("/api/experiments", json={"config": cfg.model_dump(), "start": True})
    eid = r.json()["id"]

    with c.websocket_connect(f"/ws/experiments/{eid}") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        # wait for a metrics message
        got = False
        t0 = time.time()
        while time.time() - t0 < 45:
            try:
                msg = ws.receive_json()
            except Exception:
                break
            if msg.get("type") == "metrics":
                got = True
                break
        assert got, "expected metrics over WS"

    with c.websocket_connect(f"/ws/experiments/{eid}/watch") as ws:
        ws.send_json({"type": "config", "games": 1, "board_size": 6, "speed": 20, "greedy": True})
        got_frame = False
        t0 = time.time()
        while time.time() - t0 < 30:
            msg = ws.receive_json()
            if msg.get("type") == "frame":
                got_frame = True
                assert len(msg["games"]) == 1
                break
            if msg.get("type") == "info":
                continue
        assert got_frame, "expected watch frame"

    c.post(f"/api/experiments/{eid}/stop")
    c.delete(f"/api/experiments/{eid}")


def test_create_from_preset_shape(client):
    c, _mgr, _store = client
    preset = get_preset("quick_8x8")
    # Don't actually start full preset (too heavy); override to tiny
    cfg = preset.model_copy(deep=True)
    cfg.ppo.num_envs = 4
    cfg.ppo.rollout = 8
    cfg.ppo.epochs = 1
    cfg.ppo.minibatches = 1
    cfg.model.width = 0.5
    cfg.run.device = "cpu"
    cfg.run.eval_every_s = 999
    r = c.post("/api/experiments", json={"config": cfg.model_dump(), "start": False})
    assert r.status_code == 200
    eid = r.json()["id"]
    c.delete(f"/api/experiments/{eid}")
