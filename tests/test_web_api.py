from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from snake_rl.web_console.monitor import (
    monitor_logdir_for,
    normalize_run_ref,
    should_restart_monitor,
)
from snake_rl.web_server import app


def test_normalize_run_ref_accepts_object_or_name() -> None:
    assert normalize_run_ref({"name": "exp1", "model": "tiny"}) == "exp1"
    assert normalize_run_ref("exp1") == "exp1"
    assert normalize_run_ref(None) == ""


def test_should_restart_monitor_when_logdir_changes(tmp_path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    first = monitor_logdir_for("a", runs)
    second = monitor_logdir_for("b", runs)
    assert not should_restart_monitor(str(first), first, True)
    assert should_restart_monitor(str(first), second, True)
    assert not should_restart_monitor(str(first), second, False)


def test_form_meta_and_schemes_endpoints() -> None:
    client = TestClient(app)
    meta = client.get("/api/form-meta")
    assert meta.status_code == 200
    keys = {
        field["key"]
        for sec in meta.json()["sections"]
        for group in sec["groups"]
        for field in group["fields"]
    }
    assert "n_step" in keys
    assert "curriculum_enabled" in keys
    assert "random_board.board_sizes" in keys
    essentials = {
        field["key"]
        for sec in meta.json()["sections"]
        for group in sec["groups"]
        for field in group["fields"]
        if field.get("essential")
    }
    assert {"run_name", "episodes", "model_type", "learning_rate"} <= essentials
    schemes = client.get("/api/schemes")
    assert schemes.status_code == 200
    assert "custom" in schemes.json()
    state = client.get("/api/state")
    assert state.status_code == 200
    body = state.json()
    assert "scheme" in body
    assert "lan_ip" in body
    assert "progress" in body
    assert "started_at" in body["progress"]
    with patch("snake_rl.web_console.app.tcp_port_open", return_value=False):
        proxy = client.get("/api/infer/proxy/v1/status")
    assert proxy.status_code == 503
    assert "推理" in str(proxy.json().get("detail", ""))
    with (
        patch("snake_rl.web_console.app.tcp_port_open", return_value=True),
        patch("snake_rl.web_console.app.http_get_json", return_value={"detail": "Not Found"}),
    ):
        foreign = client.get("/api/infer/proxy/v1/status")
    assert foreign.status_code == 503
    fav = client.get("/favicon.ico")
    assert fav.status_code == 200
