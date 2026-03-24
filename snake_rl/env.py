from __future__ import annotations

from dataclasses import dataclass, asdict
import random
import time
from typing import Any

import numpy as np

DIRS: dict[str, tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}

DIRECTION_ORDER = ("up", "right", "down", "left")
DIRECTION_INDEX = {"up": 0, "right": 1, "down": 2, "left": 3}

ACTIONS = {
    "STRAIGHT": 0,
    "TURN_LEFT": 1,
    "TURN_RIGHT": 2,
}

OBSERVATION_CHANNELS = (
    "snakeHead",
    "snakeBody",
    "food",
    "bonusFood",
    "obstacle",
    "dirUp",
    "dirRight",
    "dirDown",
    "dirLeft",
)

DEFAULT_REWARD_WEIGHTS = {
    "alive": -0.01,
    "food": 1.0,
    "bonusFood": 1.5,
    "death": -1.5,
    "timeout": -1.0,
    "levelUp": 0.2,
    "victory": 5.0,
    "foodDistanceK": 0.4,
}

TERMINAL_REASONS = {
    "WALL": "wall",
    "OBSTACLE": "obstacle",
    "SELF": "self",
    "BOARD_FULL": "board_full",
    "TIMEOUT": "timeout",
    "NOT_RUNNING": "not_running",
}

TERMINAL_REASON_LABELS = {
    "wall": "撞墙了",
    "obstacle": "撞到障碍物了",
    "self": "咬到自己了",
    "board_full": "地图已填满",
    "timeout": "长时间未吃到食物",
    "not_running": "回合已结束",
}

DIFFICULTY_CONFIG: dict[str, dict[str, float]] = {
    "easy": {
        "baseTick": 180,
        "perLevelFaster": 4,
        "minTick": 105,
        "levelStepByFoods": 7,
        "bonusChance": 0.28,
        "maxObstacles": 8,
        "bonusLifeMs": 9000,
    },
    "normal": {
        "baseTick": 145,
        "perLevelFaster": 5,
        "minTick": 90,
        "levelStepByFoods": 6,
        "bonusChance": 0.32,
        "maxObstacles": 12,
        "bonusLifeMs": 8000,
    },
    "hard": {
        "baseTick": 120,
        "perLevelFaster": 6,
        "minTick": 78,
        "levelStepByFoods": 5,
        "bonusChance": 0.35,
        "maxObstacles": 16,
        "bonusLifeMs": 7200,
    },
    "expert": {
        "baseTick": 98,
        "perLevelFaster": 6,
        "minTick": 68,
        "levelStepByFoods": 4,
        "bonusChance": 0.4,
        "maxObstacles": 20,
        "bonusLifeMs": 6500,
    },
}


@dataclass(slots=True)
class SnakeEnvConfig:
    """Environment config mirroring the JavaScript implementation."""

    difficulty: str = "normal"
    mode: str = "classic"
    board_size: int = 22
    enable_bonus_food: bool = True
    enable_obstacles: bool = True
    allow_leveling: bool = True
    max_steps_without_food: int = 0

    def normalized(self) -> "SnakeEnvConfig":
        difficulty = self.difficulty if self.difficulty in DIFFICULTY_CONFIG else "normal"
        mode = "wrap" if self.mode == "wrap" else "classic"
        board_size = int(round(float(self.board_size)))
        board_size = max(8, min(64, board_size))
        timeout = int(round(float(self.max_steps_without_food)))
        max_steps_without_food = timeout if timeout > 0 else 0
        return SnakeEnvConfig(
            difficulty=difficulty,
            mode=mode,
            board_size=board_size,
            enable_bonus_food=bool(self.enable_bonus_food),
            enable_obstacles=bool(self.enable_obstacles),
            allow_leveling=bool(self.allow_leveling),
            max_steps_without_food=max_steps_without_food,
        )


class SnakeEnv:
    """Pure Python snake environment for DDQN training."""

    def __init__(
        self,
        config: SnakeEnvConfig | None = None,
        reward_weights: dict[str, float] | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = (config or SnakeEnvConfig()).normalized()
        self.reward_weights = dict(DEFAULT_REWARD_WEIGHTS)
        if reward_weights:
            for key, value in reward_weights.items():
                if key in self.reward_weights:
                    self.reward_weights[key] = float(value)

        self._rng = random.Random()
        self._seed: int | None = None
        self.state = "ready"
        self.episode_index = 0
        self._episode_start_time = 0.0
        self._episode_steps = 0
        self._episode_total_reward = 0.0
        self._episode_foods = 0
        self._episode_bonus_foods = 0
        self._episode_level_ups = 0
        self._episode_max_length = 0
        self._episode_terminal_reason = ""

        self.direction = "right"
        self.snake: list[tuple[int, int]] = []
        self.food: tuple[int, int] | None = None
        self.bonus_food: tuple[int, int] | None = None
        self.bonus_expires_step = 0
        self.obstacles: set[tuple[int, int]] = set()
        self.score = 0
        self.level = 1
        self.foods_eaten = 0
        self.steps_since_last_food = 0
        self.last_terminal_reason = ""
        self.last_info: dict[str, Any] | None = None

        self.set_seed(seed)
        self.reset()

    @property
    def board_size(self) -> int:
        return self.config.board_size

    def set_seed(self, seed: int | None) -> int | None:
        if seed is None:
            self._seed = None
            self._rng = random.Random()
            return None
        normalized = int(seed)
        normalized = (normalized & 0xFFFFFFFF) or 1
        self._seed = normalized
        self._rng = random.Random(normalized)
        return self._seed

    def get_seed(self) -> int | None:
        return self._seed

    def configure(self, options: dict[str, Any]) -> SnakeEnvConfig:
        mapped = self._map_config_keys(options)
        merged = asdict(self.config)
        merged.update(mapped)
        self.config = SnakeEnvConfig(**merged).normalized()
        return self.config

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.set_seed(seed)
        if options:
            self.configure(options)

        center = self.board_size // 2
        self.snake = [
            (center, center),
            (center - 1, center),
            (center - 2, center),
        ]
        self.direction = "right"
        self.score = 0
        self.level = 1
        self.foods_eaten = 0
        self.steps_since_last_food = 0
        self.food = None
        self.bonus_food = None
        self.bonus_expires_step = 0
        self.obstacles = set()
        self.last_terminal_reason = ""
        self.last_info = None

        self._spawn_food()
        self.state = "running"
        self.episode_index += 1
        self._episode_start_time = time.time()
        self._episode_steps = 0
        self._episode_total_reward = 0.0
        self._episode_foods = 0
        self._episode_bonus_foods = 0
        self._episode_level_ups = 0
        self._episode_max_length = len(self.snake)
        self._episode_terminal_reason = ""

        observation = self.get_observation()
        info = {
            "episode": self.episode_index,
            "reset": True,
            "step": 0,
            "action": ACTIONS["STRAIGHT"],
            "score_before": 0,
            "score_after": self.score,
            "length_before": len(self.snake),
            "length_after": len(self.snake),
            "level_before": self.level,
            "level_after": self.level,
            "foods_eaten": self.foods_eaten,
            "steps_since_food_before": 0,
            "steps_since_food_after": 0,
            "terminal_reason": "",
            "terminal_reason_label": "",
        }
        self.last_info = info
        return observation, info

    def _build_step_info(
        self,
        *,
        step: int,
        action: int,
        lightweight: bool,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            "episode": self.episode_index,
            "step": int(step),
            "action": int(action),
            "ate_food": False,
            "ate_bonus_food": False,
            "level_up": False,
            "terminal_reason": "",
        }
        if lightweight:
            return info

        info.update(
            {
                "score_gain": 0,
                "score_before": self.score,
                "score_after": self.score,
                "length_before": len(self.snake),
                "length_after": len(self.snake),
                "level_before": self.level,
                "level_after": self.level,
                "foods_eaten": self.foods_eaten,
                "steps_since_food_before": self.steps_since_last_food,
                "steps_since_food_after": self.steps_since_last_food,
                "terminal_reason_label": "",
            }
        )
        return info

    def step(
        self,
        action: int | str,
        *,
        lightweight_info: bool = False,
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self.state != "running":
            observation = self.get_observation()
            info = self._build_step_info(
                step=self._episode_steps,
                action=self._normalize_action(action),
                lightweight=lightweight_info,
            )
            info["terminal_reason"] = self.last_terminal_reason or TERMINAL_REASONS["NOT_RUNNING"]
            if not lightweight_info:
                info["terminal_reason_label"] = TERMINAL_REASON_LABELS.get(
                    self.last_terminal_reason or TERMINAL_REASONS["NOT_RUNNING"],
                    TERMINAL_REASONS["NOT_RUNNING"],
                )
                info["score_after"] = self.score
                info["length_after"] = len(self.snake)
                info["level_after"] = self.level
                info["foods_eaten"] = self.foods_eaten
                info["steps_since_food_after"] = self.steps_since_last_food
            self.last_info = info
            return observation, 0.0, True, info

        step_index = self._episode_steps + 1
        normalized_action = self._apply_relative_action(action)
        reward = float(self.reward_weights["alive"])
        done = False
        info = self._build_step_info(
            step=step_index,
            action=normalized_action,
            lightweight=lightweight_info,
        )

        def set_terminal(reason: str, extra_reward: float = 0.0) -> None:
            nonlocal done, reward
            if done:
                return
            done = True
            reward += float(extra_reward)
            info["terminal_reason"] = reason
            if not lightweight_info:
                info["terminal_reason_label"] = TERMINAL_REASON_LABELS.get(reason, reason)

        head_x, head_y = self.snake[0]
        _old_food_dist: int | None = None
        if self.food is not None:
            _old_food_dist = self._manhattan_distance(head_x, head_y, self.food[0], self.food[1])
        move_x, move_y = DIRS[self.direction]
        next_x = head_x + move_x
        next_y = head_y + move_y

        if self.config.mode == "wrap":
            next_x = (next_x + self.board_size) % self.board_size
            next_y = (next_y + self.board_size) % self.board_size
        elif not self._in_bounds(next_x, next_y):
            set_terminal(TERMINAL_REASONS["WALL"], self.reward_weights["death"])

        next_cell = (next_x, next_y)
        if not done and next_cell in self.obstacles:
            set_terminal(TERMINAL_REASONS["OBSTACLE"], self.reward_weights["death"])

        if not done:
            hit_food = self.food is not None and next_cell == self.food
            hit_bonus = self.bonus_food is not None and next_cell == self.bonus_food
            will_grow = bool(hit_food or hit_bonus)
            body_for_collision = self.snake if will_grow else self.snake[:-1]
            if next_cell in body_for_collision:
                set_terminal(TERMINAL_REASONS["SELF"], self.reward_weights["death"])
            else:
                self.snake.insert(0, next_cell)
                self.steps_since_last_food += 1
                if hit_food:
                    eat_info = self._on_eat_food(step_index)
                    info["ate_food"] = True
                    if not lightweight_info:
                        info["score_gain"] += eat_info["score_gain"]
                    info["level_up"] = eat_info["level_up"]
                    reward += float(self.reward_weights["food"])
                    self.steps_since_last_food = 0
                    if eat_info["level_up"]:
                        reward += float(self.reward_weights["levelUp"])
                    if eat_info["board_filled"]:
                        set_terminal(TERMINAL_REASONS["BOARD_FULL"], self.reward_weights["victory"])
                elif hit_bonus:
                    bonus_info = self._on_eat_bonus(step_index)
                    info["ate_bonus_food"] = True
                    if not lightweight_info:
                        info["score_gain"] += bonus_info["score_gain"]
                    reward += float(self.reward_weights["bonusFood"])
                    self.steps_since_last_food = 0
                else:
                    self.snake.pop()

        if not done and not info["ate_food"] and not info["ate_bonus_food"]:
            if self.food is not None and _old_food_dist is not None:
                _hx, _hy = self.snake[0]
                _new_dist = self._manhattan_distance(_hx, _hy, self.food[0], self.food[1])
                _max_dist = max(1.0, 2.0 * (self.board_size - 1))
                _k = float(self.reward_weights.get("foodDistanceK", 0.0))
                reward += _k * (_old_food_dist - _new_dist) / _max_dist

        if (
            not done
            and self.config.max_steps_without_food > 0
            and self.steps_since_last_food >= self.config.max_steps_without_food
        ):
            set_terminal(TERMINAL_REASONS["TIMEOUT"], self.reward_weights["timeout"])

        if self.bonus_food is not None and step_index >= self.bonus_expires_step:
            self.bonus_food = None
            self.bonus_expires_step = 0

        self._episode_steps = step_index
        if not lightweight_info:
            info["score_after"] = self.score
            info["length_after"] = len(self.snake)
            info["level_after"] = self.level
            info["foods_eaten"] = self.foods_eaten
            info["steps_since_food_after"] = self.steps_since_last_food
        observation = self.get_observation()

        self._record_transition(reward=reward, done=done, info=info)
        if done:
            self.state = "over"
            self.last_terminal_reason = str(info["terminal_reason"])
            self._episode_terminal_reason = self.last_terminal_reason

        self.last_info = info
        return observation, reward, done, info

    def sample_action(self) -> int:
        return self._rng.randrange(3)

    def get_observation(self) -> np.ndarray:
        size = self.board_size
        channels = len(OBSERVATION_CHANNELS)
        observation = np.zeros((size, size, channels), dtype=np.float32)

        for idx, (x, y) in enumerate(self.snake):
            channel = 0 if idx == 0 else 1
            observation[y, x, channel] = 1.0

        if self.food is not None:
            x, y = self.food
            observation[y, x, 2] = 1.0

        if self.bonus_food is not None:
            x, y = self.bonus_food
            observation[y, x, 3] = 1.0

        for x, y in self.obstacles:
            observation[y, x, 4] = 1.0

        direction_idx = DIRECTION_INDEX[self.direction]
        observation[:, :, 5 + direction_idx] = 1.0
        return observation

    def get_local_patch(self, patch_size: int = 11) -> np.ndarray:
        """以蛇头为中心提取局部观测 patch，供 hybrid 模型使用。

        - `patch_size` 必须为奇数，例如 9 / 11 / 13
        - 经典模式下，越界区域填 0
        - 穿墙模式下，局部窗口会按环绕规则取值
        """
        patch_size = int(patch_size)
        if patch_size <= 0 or patch_size % 2 == 0:
            raise ValueError("patch_size 必须是正奇数")

        radius = patch_size // 2
        patch = np.zeros((patch_size, patch_size, len(OBSERVATION_CHANNELS)), dtype=np.float32)
        head_x, head_y = self.snake[0]
        head = (head_x, head_y)
        snake_body = set(self.snake[1:])
        direction_idx = DIRECTION_INDEX[self.direction]
        patch[:, :, 5 + direction_idx] = 1.0
        wrap_mode = self.config.mode == "wrap"

        for py in range(patch_size):
            dy = py - radius
            for px in range(patch_size):
                dx = px - radius
                src_x = head_x + dx
                src_y = head_y + dy

                if wrap_mode:
                    cell = (src_x % self.board_size, src_y % self.board_size)
                elif self._in_bounds(src_x, src_y):
                    cell = (src_x, src_y)
                else:
                    continue

                if cell == head:
                    patch[py, px, 0] = 1.0
                elif cell in snake_body:
                    patch[py, px, 1] = 1.0

                if self.food is not None and cell == self.food:
                    patch[py, px, 2] = 1.0
                if self.bonus_food is not None and cell == self.bonus_food:
                    patch[py, px, 3] = 1.0
                if cell in self.obstacles:
                    patch[py, px, 4] = 1.0

        return patch

    def get_global_features(self) -> np.ndarray:
        """返回 10 维归一化全局特征向量，供 HybridNet 使用。

        特征定义（索引固定，与 model.HybridNet.GLOBAL_FEAT_DIM=10 对应）：
            [0] 食物相对方向 x，归一化到 [-1, 1]；无食物时为 0
            [1] 食物相对方向 y，归一化到 [-1, 1]；无食物时为 0
            [2] 食物曼哈顿距离，归一化到 [0, 1]（除以 2*(board_size-1)）
            [3] 蛇头到上方墙的格数，归一化到 [0, 1]
            [4] 蛇头到下方墙的格数，归一化到 [0, 1]
            [5] 蛇头到左方墙的格数，归一化到 [0, 1]
            [6] 蛇头到右方墙的格数，归一化到 [0, 1]
            [7] 蛇身长度，归一化到 [0, 1]（除以 board_size²）
            [8] 蛇占地图格子比例，[0, 1]
            [9] 是否存在奖励食物（0.0 或 1.0）
        """
        feat = np.zeros(10, dtype=np.float32)
        size = self.board_size
        max_dist = 2.0 * (size - 1) if size > 1 else 1.0
        max_cells = float(size * size)

        head_x, head_y = self.snake[0]

        if self.food is not None:
            fx, fy = self.food
            dx = fx - head_x
            dy = fy - head_y
            feat[0] = dx / (size - 1) if size > 1 else 0.0
            feat[1] = dy / (size - 1) if size > 1 else 0.0
            feat[2] = (abs(dx) + abs(dy)) / max_dist

        feat[3] = head_y / (size - 1) if size > 1 else 0.0            # 到上方墙
        feat[4] = (size - 1 - head_y) / (size - 1) if size > 1 else 0.0  # 到下方墙
        feat[5] = head_x / (size - 1) if size > 1 else 0.0            # 到左方墙
        feat[6] = (size - 1 - head_x) / (size - 1) if size > 1 else 0.0  # 到右方墙

        snake_len = len(self.snake)
        feat[7] = snake_len / max_cells
        feat[8] = snake_len / max_cells
        feat[9] = 1.0 if self.bonus_food is not None else 0.0

        return feat

    def get_action_space(self) -> dict[str, Any]:
        return {
            "type": "discrete",
            "size": 3,
            "semantics": "relative_turn",
            "actions": {
                "straight": ACTIONS["STRAIGHT"],
                "turnLeft": ACTIONS["TURN_LEFT"],
                "turnRight": ACTIONS["TURN_RIGHT"],
            },
        }

    def get_observation_space(self) -> dict[str, Any]:
        return {
            "type": "tensor",
            "dtype": "float32",
            "layout": "HWC",
            "shape": [self.board_size, self.board_size, len(OBSERVATION_CHANNELS)],
            "channels": list(OBSERVATION_CHANNELS),
        }

    def get_metadata(self) -> dict[str, Any]:
        return {
            "mode": self.config.mode,
            "difficulty": self.config.difficulty,
            "board_size": self.config.board_size,
            "seed": self._seed,
            "env_config": asdict(self.config),
            "reward_weights": dict(self.reward_weights),
            "action_space": self.get_action_space(),
            "observation_space": self.get_observation_space(),
        }

    def get_episode_stats(self) -> dict[str, Any]:
        now = time.time()
        duration_ms = max(0, int((now - self._episode_start_time) * 1000))
        return {
            "episode": self.episode_index,
            "steps": self._episode_steps,
            "total_reward": self._episode_total_reward,
            "foods": self._episode_foods,
            "bonus_foods": self._episode_bonus_foods,
            "level_ups": self._episode_level_ups,
            "max_length": self._episode_max_length,
            "score_end": self.score,
            "done": self.state != "running",
            "terminal_reason": self._episode_terminal_reason,
            "duration_ms": duration_ms,
        }

    def get_state(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "reward_weights": dict(self.reward_weights),
            "seed": self._seed,
            "state": self.state,
            "direction": self.direction,
            "snake": [{"x": x, "y": y} for x, y in self.snake],
            "food": None if self.food is None else {"x": self.food[0], "y": self.food[1]},
            "bonus_food": None
            if self.bonus_food is None
            else {
                "x": self.bonus_food[0],
                "y": self.bonus_food[1],
                "expires_step": self.bonus_expires_step,
            },
            "obstacles": [{"x": x, "y": y} for x, y in sorted(self.obstacles)],
            "score": self.score,
            "level": self.level,
            "foods_eaten": self.foods_eaten,
            "steps_since_last_food": self.steps_since_last_food,
            "episode_index": self.episode_index,
            "episode_steps": self._episode_steps,
            "episode_total_reward": self._episode_total_reward,
            "episode_foods": self._episode_foods,
            "episode_bonus_foods": self._episode_bonus_foods,
            "episode_level_ups": self._episode_level_ups,
            "episode_max_length": self._episode_max_length,
            "episode_terminal_reason": self._episode_terminal_reason,
        }

    def set_state(self, snapshot: dict[str, Any]) -> None:
        if not isinstance(snapshot, dict):
            raise TypeError("snapshot must be a dict")
        if "config" in snapshot:
            self.configure(snapshot["config"])
        if "reward_weights" in snapshot:
            self.set_reward_weights(snapshot["reward_weights"])
        if "seed" in snapshot:
            self.set_seed(snapshot["seed"])

        self.state = str(snapshot.get("state", "running"))
        self.direction = str(snapshot.get("direction", "right"))
        self.snake = [(int(cell["x"]), int(cell["y"])) for cell in snapshot.get("snake", [])]
        if not self.snake:
            raise ValueError("snapshot snake cannot be empty")
        food = snapshot.get("food")
        self.food = None if food is None else (int(food["x"]), int(food["y"]))
        bonus = snapshot.get("bonus_food")
        if bonus is None:
            self.bonus_food = None
            self.bonus_expires_step = 0
        else:
            self.bonus_food = (int(bonus["x"]), int(bonus["y"]))
            self.bonus_expires_step = int(bonus.get("expires_step", 0))
        self.obstacles = {
            (int(cell["x"]), int(cell["y"])) for cell in snapshot.get("obstacles", [])
        }
        self.score = int(snapshot.get("score", 0))
        self.level = int(snapshot.get("level", 1))
        self.foods_eaten = int(snapshot.get("foods_eaten", 0))
        self.steps_since_last_food = int(snapshot.get("steps_since_last_food", 0))
        self.episode_index = int(snapshot.get("episode_index", self.episode_index))
        self._episode_steps = int(snapshot.get("episode_steps", 0))
        self._episode_total_reward = float(snapshot.get("episode_total_reward", 0.0))
        self._episode_foods = int(snapshot.get("episode_foods", 0))
        self._episode_bonus_foods = int(snapshot.get("episode_bonus_foods", 0))
        self._episode_level_ups = int(snapshot.get("episode_level_ups", 0))
        self._episode_max_length = int(snapshot.get("episode_max_length", len(self.snake)))
        self._episode_terminal_reason = str(snapshot.get("episode_terminal_reason", ""))

    def set_reward_weights(self, weights: dict[str, float]) -> dict[str, float]:
        for key, value in weights.items():
            if key in self.reward_weights:
                self.reward_weights[key] = float(value)
        return dict(self.reward_weights)

    def render(self, mode: str = "ansi") -> str | np.ndarray | None:
        if mode == "rgb_array":
            return self._render_rgb_array()
        rows = [["." for _ in range(self.board_size)] for _ in range(self.board_size)]
        for x, y in self.obstacles:
            rows[y][x] = "#"
        for idx, (x, y) in enumerate(self.snake):
            rows[y][x] = "H" if idx == 0 else "s"
        if self.food is not None:
            x, y = self.food
            rows[y][x] = "F"
        if self.bonus_food is not None:
            x, y = self.bonus_food
            rows[y][x] = "B"

        lines = [" ".join(row) for row in rows]
        board = "\n".join(lines)
        if mode == "human":
            print(board)
            return None
        return board

    def close(self) -> None:
        self.state = "over"

    def _record_transition(self, reward: float, done: bool, info: dict[str, Any]) -> None:
        self._episode_total_reward += float(reward)
        self._episode_max_length = max(self._episode_max_length, len(self.snake))
        if info["ate_food"]:
            self._episode_foods += 1
        if info["ate_bonus_food"]:
            self._episode_bonus_foods += 1
        if info["level_up"]:
            self._episode_level_ups += 1
        if done:
            self._episode_terminal_reason = str(info["terminal_reason"])

    def _map_config_keys(self, options: dict[str, Any]) -> dict[str, Any]:
        key_map = {
            "boardSize": "board_size",
            "enableBonusFood": "enable_bonus_food",
            "enableObstacles": "enable_obstacles",
            "allowLeveling": "allow_leveling",
            "maxStepsWithoutFood": "max_steps_without_food",
        }
        out: dict[str, Any] = {}
        for key, value in options.items():
            out[key_map.get(key, key)] = value
        return out

    def _normalize_action(self, action: int | str) -> int:
        if action in (ACTIONS["STRAIGHT"], ACTIONS["TURN_LEFT"], ACTIONS["TURN_RIGHT"]):
            return int(action)
        if isinstance(action, str):
            value = action.strip().lower()
            if value in {"0", "straight", "forward"}:
                return ACTIONS["STRAIGHT"]
            if value in {"1", "left", "turn_left"}:
                return ACTIONS["TURN_LEFT"]
            if value in {"2", "right", "turn_right"}:
                return ACTIONS["TURN_RIGHT"]
        return ACTIONS["STRAIGHT"]

    def _apply_relative_action(self, action: int | str) -> int:
        normalized = self._normalize_action(action)
        direction_idx = DIRECTION_ORDER.index(self.direction)
        if normalized == ACTIONS["TURN_LEFT"]:
            direction_idx = (direction_idx - 1) % len(DIRECTION_ORDER)
        elif normalized == ACTIONS["TURN_RIGHT"]:
            direction_idx = (direction_idx + 1) % len(DIRECTION_ORDER)
        self.direction = DIRECTION_ORDER[direction_idx]
        return normalized

    def _on_eat_food(self, step_index: int) -> dict[str, Any]:
        score_gain = 10 * self.level
        self.foods_eaten += 1
        self.score += score_gain

        level_up = self._recalculate_level()
        self._spawn_food()
        board_filled = self.food is None

        if (
            not board_filled
            and self.config.enable_bonus_food
            and self.bonus_food is None
            and self.foods_eaten >= 3
            and self.foods_eaten % 3 == 0
            and self._rng.random() < float(DIFFICULTY_CONFIG[self.config.difficulty]["bonusChance"])
        ):
            self._spawn_bonus_food(step_index)

        return {
            "score_gain": score_gain,
            "level_up": level_up,
            "board_filled": board_filled,
        }

    def _on_eat_bonus(self, step_index: int) -> dict[str, Any]:
        if self.bonus_food is None:
            return {"score_gain": 0}
        remain_steps = max(0, self.bonus_expires_step - step_index)
        bonus_score = 20 * self.level + remain_steps
        self.score += bonus_score
        self.bonus_food = None
        self.bonus_expires_step = 0
        return {"score_gain": bonus_score}

    def _recalculate_level(self) -> bool:
        if not self.config.allow_leveling:
            return False
        level_step = int(DIFFICULTY_CONFIG[self.config.difficulty]["levelStepByFoods"])
        next_level = 1 + (self.foods_eaten // level_step)
        if next_level <= self.level:
            return False
        delta = next_level - self.level
        self.level = next_level
        if self.config.enable_obstacles:
            for _ in range(delta):
                self._try_spawn_obstacle()
        return True

    def _try_spawn_obstacle(self) -> None:
        max_obstacles = int(DIFFICULTY_CONFIG[self.config.difficulty]["maxObstacles"])
        if len(self.obstacles) >= max_obstacles:
            return

        for _ in range(self.board_size * self.board_size):
            candidate = self._random_cell()
            if candidate is None:
                return
            if candidate in self.obstacles:
                continue
            if candidate in self.snake:
                continue
            if self.food is not None and candidate == self.food:
                continue
            if self.bonus_food is not None and candidate == self.bonus_food:
                continue

            head_x, head_y = self.snake[0]
            c_x, c_y = candidate
            distance = abs(c_x - head_x) + abs(c_y - head_y)
            if distance < 4:
                continue
            self.obstacles.add(candidate)
            return

    def _spawn_food(self) -> None:
        self.food = self._random_empty_cell()

    def _spawn_bonus_food(self, step_index: int) -> None:
        cell = self._random_empty_cell()
        if cell is None:
            return
        self.bonus_food = cell
        bonus_life_ms = float(DIFFICULTY_CONFIG[self.config.difficulty]["bonusLifeMs"])
        tick_ms = max(1.0, float(self._get_tick_duration()))
        bonus_life_steps = max(1, int(round(bonus_life_ms / tick_ms)))
        self.bonus_expires_step = step_index + bonus_life_steps

    def _random_empty_cell(self) -> tuple[int, int] | None:
        max_try = self.board_size * self.board_size * 2
        for _ in range(max_try):
            candidate = self._random_cell()
            if candidate is None:
                return None
            if candidate in self.snake:
                continue
            if candidate in self.obstacles:
                continue
            if self.food is not None and candidate == self.food:
                continue
            if self.bonus_food is not None and candidate == self.bonus_food:
                continue
            return candidate
        return None

    def _random_cell(self) -> tuple[int, int] | None:
        if self.board_size <= 0:
            return None
        return (self._rng.randrange(self.board_size), self._rng.randrange(self.board_size))

    def _manhattan_distance(self, x1: int, y1: int, x2: int, y2: int) -> int:
        """Manhattan distance, respecting wrap mode."""
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        if self.config.mode == "wrap":
            dx = min(dx, self.board_size - dx)
            dy = min(dy, self.board_size - dy)
        return dx + dy

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.board_size and 0 <= y < self.board_size

    def _get_tick_duration(self) -> float:
        cfg = DIFFICULTY_CONFIG[self.config.difficulty]
        tick = float(cfg["baseTick"]) - (self.level - 1) * float(cfg["perLevelFaster"])
        return max(float(cfg["minTick"]), tick)

    def _render_rgb_array(self) -> np.ndarray:
        colors = np.zeros((self.board_size, self.board_size, 3), dtype=np.uint8)
        if self.food is not None:
            x, y = self.food
            colors[y, x] = np.array([220, 50, 47], dtype=np.uint8)
        if self.bonus_food is not None:
            x, y = self.bonus_food
            colors[y, x] = np.array([211, 54, 130], dtype=np.uint8)
        for x, y in self.obstacles:
            colors[y, x] = np.array([88, 110, 117], dtype=np.uint8)
        for idx, (x, y) in enumerate(self.snake):
            colors[y, x] = np.array([38, 139, 210], dtype=np.uint8)
            if idx == 0:
                colors[y, x] = np.array([42, 161, 152], dtype=np.uint8)
        return colors
