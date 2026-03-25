from __future__ import annotations

import json
from pathlib import Path

from snake_rl.run_context import (
    RunContext,
    checkpoint_run_dir,
    load_run_config_dict,
)


def test_checkpoint_run_dir_resolves_parent() -> None:
    p = Path("runs/my_run/checkpoints/best.pt")
    rd = checkpoint_run_dir(p)
    assert rd is not None
    assert rd.name == "my_run"


def test_load_run_config_prefers_run_config(tmp_path: Path) -> None:
    run_dir = tmp_path / "r1"
    run_dir.mkdir()
    (run_dir / "train_config.json").write_text(
        json.dumps({"model_type": "old"}), encoding="utf-8"
    )
    (run_dir / "run_config.json").write_text(
        json.dumps({"model_type": "new"}), encoding="utf-8"
    )
    d = load_run_config_dict(run_dir)
    assert d is not None
    assert d["model_type"] == "new"


def test_run_context_from_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "r2"
    (run_dir / "checkpoints").mkdir(parents=True)
    ckpt = run_dir / "checkpoints" / "latest.pt"
    ckpt.write_bytes(b"")
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "model_type": "hybrid",
                "local_patch_size": 11,
                "env": {
                    "difficulty": "hard",
                    "mode": "wrap",
                    "board_size": 16,
                    "enable_bonus_food": True,
                    "enable_obstacles": False,
                    "allow_leveling": True,
                    "max_steps_without_food": 200,
                },
                "reward_weights": {"food": 1.0},
            }
        ),
        encoding="utf-8",
    )
    ctx = RunContext.from_checkpoint(ckpt.resolve())
    assert ctx.run_dir == run_dir.resolve()
    assert ctx.model_type == "hybrid"
    assert ctx.local_patch_size == 11
    assert ctx.env is not None
    assert ctx.env.mode == "wrap"
    assert ctx.env.board_size == 16
    assert ctx.reward_weights == {"food": 1.0}
