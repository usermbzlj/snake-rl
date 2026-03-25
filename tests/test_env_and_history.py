from __future__ import annotations

import csv
import json
from pathlib import Path

from snake_rl.env import SnakeEnv, SnakeEnvConfig
from snake_rl.run_meta import parse_last_csv_row
from snake_rl.train import load_episode_history_snapshot


def test_env_distance_shaping_reward_matches_default_formula() -> None:
    env = SnakeEnv(
        config=SnakeEnvConfig(
            difficulty="normal",
            mode="classic",
            board_size=8,
            enable_bonus_food=False,
            enable_obstacles=False,
            allow_leveling=False,
            max_steps_without_food=0,
        ),
        seed=1,
    )
    env.set_state(
        {
            "config": {
                "difficulty": "normal",
                "mode": "classic",
                "board_size": 8,
                "enable_bonus_food": False,
                "enable_obstacles": False,
                "allow_leveling": False,
                "max_steps_without_food": 0,
            },
            "state": "running",
            "direction": "right",
            "snake": [{"x": 4, "y": 4}, {"x": 3, "y": 4}, {"x": 2, "y": 4}],
            "food": {"x": 6, "y": 4},
            "bonus_food": None,
            "obstacles": [],
            "score": 0,
            "level": 1,
            "foods_eaten": 0,
            "steps_since_last_food": 0,
            "episode_index": 1,
            "episode_steps": 0,
        }
    )

    _, reward, done, info = env.step(0)

    expected = -0.01 + 0.4 * (2 - 1) / (2 * (8 - 1))
    assert done is False
    assert info["ate_food"] is False
    assert abs(reward - expected) < 1e-6


def test_load_episode_history_snapshot_restores_recent_windows_and_counts(tmp_path: Path) -> None:
    logs_dir = tmp_path / "run" / "logs"
    logs_dir.mkdir(parents=True)
    rows = [
        {"episode": 1, "reward": 1.0, "steps": 10, "terminal_reason": "wall"},
        {"episode": 2, "reward": 2.0, "steps": 20, "terminal_reason": "self"},
        {"episode": 3, "reward": 3.0, "steps": 30, "terminal_reason": "timeout"},
    ]
    with (logs_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    snapshot = load_episode_history_snapshot(logs_dir.parent, moving_avg_window=2)

    assert snapshot["episodes_logged"] == 3
    assert snapshot["reward_window"] == [2.0, 3.0]
    assert snapshot["steps_window"] == [20.0, 30.0]
    assert snapshot["terminal_reason_counter"] == {"wall": 1, "self": 1, "timeout": 1}
    assert snapshot["last_row"]["episode"] == 3


def test_parse_last_csv_row_keeps_best_avg_reward_without_loading_all_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "episodes.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "reward",
                "avg_reward",
                "best_avg_reward",
                "steps",
                "foods",
                "score",
                "epsilon",
                "stage_index",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "episode": 1,
                "reward": 1.0,
                "avg_reward": 1.0,
                "best_avg_reward": 1.0,
                "steps": 10,
                "foods": 1,
                "score": 10,
                "epsilon": 0.9,
                "stage_index": "",
            }
        )
        writer.writerow(
            {
                "episode": 2,
                "reward": 0.5,
                "avg_reward": 0.75,
                "best_avg_reward": 1.0,
                "steps": 12,
                "foods": 0,
                "score": 10,
                "epsilon": 0.8,
                "stage_index": "",
            }
        )

    row = parse_last_csv_row(csv_path)

    assert row is not None
    assert row["episode"] == 2
    assert row["best_avg_reward"] == 1.0
