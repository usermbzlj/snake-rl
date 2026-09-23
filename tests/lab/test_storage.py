"""ExperimentStore round-trip tests."""

from __future__ import annotations

from pathlib import Path

from snake_rl.core.config import ExperimentConfig, get_preset
from snake_rl.lab.storage import ExperimentStore, downsample_evenly, make_experiment_id, sparkline


def test_make_id_slug():
    eid = make_experiment_id("快速入门 · 8×8")
    assert "8" in eid or "8x8" in eid.lower() or "入门" in eid or eid.count("-") >= 2


def test_storage_roundtrip(tmp_path: Path):
    store = ExperimentStore(tmp_path)
    cfg = get_preset("quick_8x8")
    meta = store.create(cfg, notes="hello")
    eid = meta["id"]
    assert store.exists(eid)
    assert meta["status"] == "created"
    assert meta["notes"] == "hello"

    store.update_status(eid, "running")
    store.append_metric(eid, {"iter": 1, "env_steps": 1000, "score_mean": 1.5, "sps": 100.0})
    store.append_metric(eid, {"iter": 2, "env_steps": 2000, "score_mean": 2.0, "eval_score_mean": 3.0})
    ev = store.append_event(eid, {"env_steps": 1000, "type": "start", "data": {}})
    assert "t" in ev

    detail = store.detail(eid)
    assert len(detail["metrics"]) == 2
    assert detail["events"][0]["type"] == "start"
    assert detail["experiment"]["status"] == "running"

    summary = store.summary(eid)
    assert summary["env_steps"] == 2000
    assert summary["best_eval_score"] == 3.0
    assert summary["last"]["score_mean"] == 2.0
    assert len(summary["spark"]) >= 1

    cfg2 = store.read_config(eid)
    assert isinstance(cfg2, ExperimentConfig)
    assert cfg2.env.min_size == 8

    cloned = store.clone(eid, name="克隆", with_weights=False)
    assert cloned["parent_id"] == eid
    assert cloned["name"] == "克隆"

    store.mark_stale_workers_stopped()
    assert store.read_meta(eid)["status"] == "stopped"

    store.delete(eid)
    assert not store.exists(eid)


def test_downsample_and_spark():
    rows = [{"score_mean": float(i), "i": i} for i in range(100)]
    ds = downsample_evenly(rows, 10)
    assert len(ds) == 10
    assert ds[0]["i"] == 0
    assert ds[-1]["i"] == 99
    sp = sparkline(rows, limit=20)
    assert len(sp) == 20
