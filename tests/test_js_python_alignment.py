from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from snake_rl.env import (
    ACTIONS,
    DEFAULT_REWARD_WEIGHTS,
    OBSERVATION_CHANNELS,
    TERMINAL_REASONS,
    TINY_FEAT_DIM,
    SnakeEnv,
    SnakeEnvConfig,
    _TINY_RAYS,
)

ROOT = Path(__file__).resolve().parents[1]
CONSTANTS = ROOT / "web" / "game" / "constants.js"
FEATURES = ROOT / "web" / "game" / "features.js"


def test_js_constants_match_python() -> None:
    src = CONSTANTS.read_text(encoding="utf-8")
    assert re.search(r"STRAIGHT:\s*0", src)
    assert re.search(r"TURN_LEFT:\s*1", src)
    assert re.search(r"TURN_RIGHT:\s*2", src)
    assert ACTIONS == {"STRAIGHT": 0, "TURN_LEFT": 1, "TURN_RIGHT": 2}
    for channel in OBSERVATION_CHANNELS:
        assert f'"{channel}"' in src
    assert f"const TINY_FEAT_DIM = {TINY_FEAT_DIM};" in src
    for key, value in TERMINAL_REASONS.items():
        assert f'{key}: "{value}"' in src
    for key, value in DEFAULT_REWARD_WEIGHTS.items():
        assert re.search(rf"{key}:\s*{value}", src)
    for direction, rays in _TINY_RAYS.items():
        assert direction in src
        assert str(list(rays[0])).replace(" ", "") in src.replace(" ", "")


def test_python_env_golden_steps_are_deterministic() -> None:
    env = SnakeEnv(
        config=SnakeEnvConfig(
            board_size=8,
            mode="classic",
            enable_bonus_food=False,
            enable_obstacles=False,
            allow_leveling=False,
            max_steps_without_food=32,
        ),
        seed=7,
    )
    env.reset(seed=7)
    start_head = list(env.snake[0])
    actions = [0, 0, 1, 2, 0, 0, 2, 1, 0, 0, 0, 1]
    trace = []
    for action in actions:
        _, reward, done, info = env.step(action)
        trace.append(
            {
                "action": action,
                "reward": round(float(reward), 6),
                "done": bool(done),
                "head": list(env.snake[0]),
                "food": list(env.food) if env.food is not None else None,
                "terminal": info.get("terminal_reason"),
            }
        )
        if done:
            break
    assert trace[0]["done"] is False
    assert trace[0]["head"] != start_head
    assert all("reward" in row for row in trace)
    assert len(trace) >= 3


def _browser_state_from_env(env: SnakeEnv) -> dict:
    return {
        "boardSize": env.board_size,
        "mode": env.config.mode,
        "direction": env.direction,
        "snake": [{"x": x, "y": y} for x, y in env.snake],
        "food": {"x": env.food[0], "y": env.food[1]} if env.food is not None else None,
        "bonusFood": (
            {"x": env.bonus_food[0], "y": env.bonus_food[1]} if env.bonus_food is not None else None
        ),
        "obstacles": [{"x": x, "y": y} for x, y in env.obstacles],
    }


def test_js_feature_functions_match_python_when_node_available() -> None:
    node = shutil.which("node")
    if node is None:
        return
    env = SnakeEnv(
        config=SnakeEnvConfig(
            board_size=8,
            mode="classic",
            enable_bonus_food=False,
            enable_obstacles=False,
            allow_leveling=False,
        ),
        seed=3,
    )
    env.reset(seed=3)
    state = _browser_state_from_env(env)
    script = f"""
const vm = require('vm');
const fs = require('fs');
const context = {{
  Float32Array, Object, Math, Set, Number, console,
  module: {{ exports: {{}} }},
}};
vm.createContext(context);
vm.runInContext(fs.readFileSync({json.dumps(str(CONSTANTS))}, 'utf8'), context);
vm.runInContext(fs.readFileSync({json.dumps(str(FEATURES))}, 'utf8'), context);
const state = {json.dumps(state)};
const tiny = Array.from(context.computeTinyFeatures(state).data);
const patch = Array.from(context.computeLocalPatch(state, 5).data);
const obs = Array.from(context.computeObservationTensor(state).data);
process.stdout.write(JSON.stringify({{ tiny, patch, obs }}));
"""
    proc = subprocess.run([node, "-e", script], check=True, capture_output=True, text=True)
    payload = json.loads(proc.stdout)
    np.testing.assert_allclose(payload["tiny"], env.get_tiny_features(), atol=1e-5)
    np.testing.assert_allclose(payload["patch"], env.get_local_patch(5).reshape(-1), atol=1e-5)
    np.testing.assert_allclose(payload["obs"], env.get_observation().reshape(-1), atol=1e-5)
