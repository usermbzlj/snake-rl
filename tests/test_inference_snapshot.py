from __future__ import annotations

from snake_rl.env import SnakeEnv, SnakeEnvConfig
from snake_rl.inference_server import browser_state_to_python_snapshot


def test_browser_snapshot_roundtrip_keeps_snake_and_food() -> None:
    env = SnakeEnv(config=SnakeEnvConfig(board_size=8, mode="classic"), seed=3)
    env.reset(seed=3)
    browser_state = {
        "envConfig": {
            "difficulty": "normal",
            "mode": "classic",
            "boardSize": 8,
            "enableBonusFood": False,
            "enableObstacles": False,
            "allowLeveling": False,
            "maxStepsWithoutFood": 16,
        },
        "rewardWeights": {"food": 1.0},
        "seed": 3,
        "state": "running",
        "direction": "right",
        "snake": [{"x": 3, "y": 4}, {"x": 2, "y": 4}, {"x": 1, "y": 4}],
        "food": {"x": 6, "y": 4},
        "bonusFood": None,
        "obstacles": [],
        "score": 2,
        "level": 1,
        "foodsEaten": 1,
        "stepsSinceLastFood": 0,
        "episodeStats": {"episode": 1, "steps": 4, "totalReward": 0.2, "foods": 1},
        "lastTerminalReason": "",
    }
    snapshot = browser_state_to_python_snapshot(browser_state)
    env.set_state(snapshot)
    assert env.snake[0] == (3, 4)
    assert env.food == (6, 4)
    assert env.direction == "right"
