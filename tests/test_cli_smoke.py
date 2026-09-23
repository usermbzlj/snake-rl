from __future__ import annotations

import os
import subprocess
import sys


def test_cli_train_help() -> None:
    r = subprocess.run(
        [sys.executable, "-m", "snake_rl.cli", "train", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0
    assert "--scheme" in r.stdout
    assert "--custom-config" in r.stdout


def test_cli_estimate_help() -> None:
    r = subprocess.run(
        [sys.executable, "-m", "snake_rl.cli", "estimate", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0
    assert "--custom-config" in r.stdout


def test_cli_eval_help() -> None:
    r = subprocess.run(
        [sys.executable, "-m", "snake_rl.cli", "eval", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0
    assert "--checkpoint" in r.stdout


def test_cli_train_custom_file_not_found(tmp_path) -> None:
    """--scheme custom 指定不存在的文件时，应以 exit code 2 退出并给出友好错误，不抛 traceback。"""
    nonexistent = tmp_path / "no_such_config.json"
    r = subprocess.run(
        [
            sys.executable, "-m", "snake_rl.cli", "train",
            "--scheme", "custom",
            "--custom-config", str(nonexistent),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 2
    combined = r.stdout + r.stderr
    assert "错误" in combined
    # 不应含 Python traceback
    assert "Traceback" not in combined


def test_cli_train_env_residue_custom_no_config(tmp_path) -> None:
    """环境变量 SNAKE_TRAIN_SCHEME=custom 残留、不传 --scheme 也不传 --custom-config 时，
    若默认文件不存在应以 exit code 2 退出并给出友好错误（非 traceback）。"""
    env = {**os.environ, "SNAKE_TRAIN_SCHEME": "custom", "SNAKE_DEFAULT_CUSTOM_OVERRIDE": str(tmp_path / "no.json")}
    # 直接传一个不存在的 --custom-config 来模拟"无默认文件且有 env 残留"的场景
    r = subprocess.run(
        [
            sys.executable, "-m", "snake_rl.cli", "train",
            "--custom-config", str(tmp_path / "missing.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 2
    combined = r.stdout + r.stderr
    assert "错误" in combined
    assert "Traceback" not in combined


def test_cli_train_custom_config_with_wrong_scheme(tmp_path, monkeypatch, capsys) -> None:
    """--scheme scheme1 同时传 --custom-config 时，应打印警告且不启动真实训练。"""
    import argparse

    dummy = tmp_path / "dummy.json"
    dummy.write_text("{}", encoding="utf-8")
    called = {}

    def fake_run_training(cfg, **kwargs):
        called["ok"] = True
        return {"run_dir": str(tmp_path), "episodes": 0}

    monkeypatch.setattr("snake_rl.train.run_training", fake_run_training)
    from snake_rl import cli

    args = argparse.Namespace(
        scheme="scheme1",
        custom_config=dummy,
        resume_state=None,
        warm_start=None,
        extra_episodes=None,
        warm_start_global_step=None,
        parallel=False,
        parallel_workers=None,
        parallel_queue_capacity=None,
        parallel_sync_interval=None,
        parallel_actor_sleep_ms=None,
        parallel_actor_seed_stride=None,
        parallel_actor_device=None,
    )
    cli.cmd_train(args)
    err = capsys.readouterr().err
    assert "警告" in err
    assert "custom-config" in err
    assert called.get("ok") is True


def test_cli_estimate_env_residue_custom_no_config(tmp_path) -> None:
    """estimate 子命令：环境变量残留 custom + 没有配置文件，应给出友好错误。"""
    env = {**os.environ, "SNAKE_TRAIN_SCHEME": "custom"}
    r = subprocess.run(
        [
            sys.executable, "-m", "snake_rl.cli", "estimate",
            "--custom-config", str(tmp_path / "missing.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 2
    combined = r.stdout + r.stderr
    assert "错误" in combined
    assert "Traceback" not in combined
