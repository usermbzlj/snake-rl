from __future__ import annotations

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


def test_cli_eval_help() -> None:
    r = subprocess.run(
        [sys.executable, "-m", "snake_rl.cli", "eval", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0
    assert "--checkpoint" in r.stdout
