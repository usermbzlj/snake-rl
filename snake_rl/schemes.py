"""
训练方案注册表：所有 scheme 与共享超参的唯一来源。

GUI / CLI 通过本模块获取方案列表与 `TrainConfig`，勿再依赖根目录独立脚本。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .config import (
    CurriculumConfig,
    CurriculumStage,
    EnvPreset,
    ParallelRolloutConfig,
    RandomBoardConfig,
    TrainConfig,
    train_config_from_dict,
)

ACTIVE_SCHEME = os.environ.get("SNAKE_TRAIN_SCHEME", "custom")

# 项目根目录下的默认自定义配置文件，scheme=custom 且未指定路径时自动使用
DEFAULT_CUSTOM_CONFIG_PATH = Path(__file__).resolve().parent.parent / "custom_train_config.json"

COMMON_ENV = EnvPreset(
    difficulty="normal",
    mode="classic",
    board_size=14,
    enable_bonus_food=False,
    enable_obstacles=False,
    allow_leveling=False,
    max_steps_without_food=196,
    seed=42,
)

IMPROVED_REWARDS = {
    "alive": -0.01,
    "food": 1.0,
    "bonusFood": 1.5,
    "death": -1.5,
    "timeout": -1.0,
    "levelUp": 0.2,
    "victory": 5.0,
    "foodDistanceK": 0.4,
}

SCHEME_INFO: dict[str, str] = {
    "custom": "自定义（主推）—— 从 JSON 文件加载完整 TrainConfig，全参数自由配置，使用 GUI 编辑器修改后「保存到文件」即可生效",
    "scheme1": "课程学习 —— 从小地图逐步放大，表现达标后自动晋升，加入接近食物奖励塑形",
    "scheme2": "随机地图 —— 每局随机地图大小，纯泛化训练",
    "scheme3": "Hybrid —— 局部 patch + 全局特征，随机地图，跨尺寸泛化",
    "scheme4": "课程 + 随机 + Hybrid —— 兼顾稳定性和泛化，带表现门槛",
}


def scheme_ids() -> list[str]:
    return list(SCHEME_INFO.keys())


def _base_train_config() -> TrainConfig:
    return TrainConfig(
        episodes=30000,
        max_steps_per_episode=3000,
        gamma=0.99,
        learning_rate=1e-4,
        weight_decay=1e-5,
        batch_size=128,
        replay_capacity=50000,
        min_replay_size=3000,
        train_frequency=4,
        target_update_interval=2000,
        grad_clip_norm=5.0,
        epsilon_start=1.0,
        epsilon_end=0.03,
        epsilon_decay_steps=200000,
        moving_avg_window=100,
        log_interval=10,
        checkpoint_interval=500,
        live_plot=False,
        tensorboard=True,
        save_csv=True,
        save_jsonl=True,
        run_name="default",
        output_root=Path("runs"),
        device="auto",
        env=COMMON_ENV,
        local_patch_size=11,
        parallel=ParallelRolloutConfig(
            enabled=False,
            num_workers=4,
            queue_capacity=8192,
            weight_sync_interval_steps=512,
            actor_loop_sleep_ms=0,
            actor_seed_stride=100_000,
            actor_device="cpu",
        ),
        reward_weights=dict(IMPROVED_REWARDS),
    )


def build_scheme1_curriculum() -> TrainConfig:
    cfg = _base_train_config()
    cfg.model_type = "adaptive_cnn"
    cfg.run_name = "scheme1_curriculum"
    cfg.curriculum = CurriculumConfig(
        carry_replay=True,
        scale_timeout=True,
        stages=[
            CurriculumStage(
                board_size=8,
                episodes=8000,
                epsilon_start=1.0,
                epsilon_end=0.05,
                epsilon_decay_steps=400000,
                replay_capacity=50000,
                min_replay_size=3000,
                promotion_threshold_foods=3.0,
                promotion_window=100,
                promotion_min_episodes=1000,
            ),
            CurriculumStage(
                board_size=10,
                episodes=12000,
                epsilon_start=0.30,
                epsilon_end=0.04,
                epsilon_decay_steps=800000,
                replay_capacity=80000,
                min_replay_size=5000,
                promotion_threshold_foods=2.5,
                promotion_window=100,
                promotion_min_episodes=2000,
            ),
            CurriculumStage(
                board_size=14,
                episodes=20000,
                epsilon_start=0.20,
                epsilon_end=0.03,
                epsilon_decay_steps=1500000,
                replay_capacity=100000,
                min_replay_size=8000,
                promotion_threshold_foods=2.0,
                promotion_window=100,
                promotion_min_episodes=3000,
            ),
            CurriculumStage(
                board_size=20,
                episodes=25000,
                epsilon_start=0.15,
                epsilon_end=0.02,
                epsilon_decay_steps=2500000,
                replay_capacity=150000,
                min_replay_size=10000,
            ),
        ],
    )
    return cfg


def build_scheme2_random_board() -> TrainConfig:
    cfg = _base_train_config()
    cfg.model_type = "adaptive_cnn"
    cfg.run_name = "scheme2_random_board"
    cfg.episodes = 50000
    cfg.replay_capacity = 100000
    cfg.min_replay_size = 5000
    cfg.epsilon_decay_steps = 3000000
    cfg.random_board = RandomBoardConfig(
        board_sizes=[8, 10, 12, 14, 16],
        weights=[1.0, 2.0, 3.0, 2.0, 1.0],
        max_steps_scale=1.0,
    )
    return cfg


def build_scheme3_hybrid() -> TrainConfig:
    cfg = _base_train_config()
    cfg.model_type = "hybrid"
    cfg.run_name = "scheme3_hybrid"
    cfg.episodes = 50000
    cfg.local_patch_size = 11
    cfg.replay_capacity = 100000
    cfg.min_replay_size = 5000
    cfg.epsilon_decay_steps = 3500000
    cfg.random_board = RandomBoardConfig(
        board_sizes=[8, 10, 12, 16, 20],
        weights=[1.0, 2.0, 3.0, 2.0, 1.0],
        max_steps_scale=1.0,
    )
    return cfg


def build_scheme4_curriculum_random_hybrid() -> TrainConfig:
    cfg = _base_train_config()
    cfg.model_type = "hybrid"
    cfg.run_name = "scheme4_curriculum_random_hybrid"
    cfg.local_patch_size = 11
    cfg.curriculum = CurriculumConfig(
        carry_replay=True,
        scale_timeout=False,
        stages=[
            CurriculumStage(
                board_sizes=[8, 10],
                weights=[3.0, 1.0],
                episodes=9000,
                max_steps_scale=1.0,
                epsilon_start=1.0,
                epsilon_end=0.05,
                epsilon_decay_steps=500000,
                replay_capacity=40000,
                min_replay_size=2000,
                promotion_threshold_foods=3.0,
                promotion_window=100,
                promotion_min_episodes=1500,
            ),
            CurriculumStage(
                board_sizes=[10, 12, 14],
                weights=[1.0, 2.0, 1.0],
                episodes=14000,
                max_steps_scale=1.0,
                epsilon_start=0.30,
                epsilon_end=0.04,
                epsilon_decay_steps=900000,
                replay_capacity=60000,
                min_replay_size=4000,
                promotion_threshold_foods=2.0,
                promotion_window=100,
                promotion_min_episodes=2500,
            ),
            CurriculumStage(
                board_sizes=[14, 16, 20],
                weights=[2.0, 3.0, 1.0],
                episodes=18000,
                max_steps_scale=1.0,
                epsilon_start=0.20,
                epsilon_end=0.02,
                epsilon_decay_steps=1600000,
                replay_capacity=100000,
                min_replay_size=6000,
            ),
        ],
    )
    return cfg


def default_custom_train_config() -> TrainConfig:
    """用于 GUI 自定义模式的初始模板（与内置基线一致，可再导出为 JSON）。"""
    cfg = _base_train_config()
    cfg.run_name = "custom"
    return cfg


def load_custom_train_config(path: Path | str) -> TrainConfig:
    """从 JSON 文件反序列化 `TrainConfig`（与 `run_config.json` 结构一致）。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"自定义配置不存在: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("自定义配置 JSON 根必须是对象")
    return train_config_from_dict(raw)


def get_config(scheme: str | None = None, *, custom_config_path: Path | str | None = None) -> TrainConfig:
    """返回指定方案的配置。scheme 默认取环境变量 SNAKE_TRAIN_SCHEME 或 ACTIVE_SCHEME（默认 custom）。

    ``scheme=custom`` 时：
    - 若提供了 ``custom_config_path``，加载该文件；
    - 若未提供，自动查找项目根目录的 ``custom_train_config.json``；
    - 若默认文件也不存在，抛出 ``FileNotFoundError`` 并给出明确提示。
    """
    name = scheme if scheme is not None else os.environ.get("SNAKE_TRAIN_SCHEME", ACTIVE_SCHEME)
    if name == "custom":
        if custom_config_path is None:
            if DEFAULT_CUSTOM_CONFIG_PATH.is_file():
                custom_config_path = DEFAULT_CUSTOM_CONFIG_PATH
            else:
                raise FileNotFoundError(
                    f"scheme=custom 时需要自定义配置文件，但未指定路径且默认文件不存在: "
                    f"{DEFAULT_CUSTOM_CONFIG_PATH}\n"
                    "请通过 --custom-config 指定路径，或在项目根目录创建 custom_train_config.json"
                )
        return load_custom_train_config(custom_config_path)
    if name == "scheme1":
        return build_scheme1_curriculum()
    if name == "scheme2":
        return build_scheme2_random_board()
    if name == "scheme3":
        return build_scheme3_hybrid()
    if name == "scheme4":
        return build_scheme4_curriculum_random_hybrid()
    raise ValueError(f"未知训练方案: {name!r}")
