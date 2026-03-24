"""
统一训练配置文件。

四种训练方案：
1. 方案 1：课程学习 + 表现门槛晋升（推荐）
2. 方案 2：每个 episode 随机地图大小
3. 方案 3：局部 patch + 全局手工特征（Hybrid）
4. 方案 4：课程学习 + 阶段内随机地图 + Hybrid

建议：
- 初学者优先从方案 1 开始
- 想做跨尺寸泛化，再尝试方案 4
- 顶部只需要改 ACTIVE_SCHEME（或通过 GUI / 环境变量 SNAKE_TRAIN_SCHEME 覆盖）
"""

import os
from pathlib import Path

from snake_rl.config import (
    CurriculumConfig,
    CurriculumStage,
    EnvPreset,
    RandomBoardConfig,
    TrainConfig,
)


ACTIVE_SCHEME = os.environ.get("SNAKE_TRAIN_SCHEME", "scheme1")


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


def _base_train_config() -> TrainConfig:
    """所有方案共享的基础超参数。"""
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
        reward_weights=dict(IMPROVED_REWARDS),
    )


def build_scheme1_curriculum() -> TrainConfig:
    """方案 1：课程学习 + 表现门槛晋升。

    改进点：
    - 每个阶段设置晋升门槛（promotion_threshold_foods）
    - 只有最近 N 局平均吃到足够食物，才允许进入更大地图
    - 同时保留最大局数上限，避免永远卡在某阶段
    - 加入接近食物的奖励塑形，让早期学习信号更密集
    """
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
    """方案 2：每个 episode 随机选地图大小。"""
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
    """方案 3：局部 patch + 全局手工特征（Hybrid）。"""
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
    """方案 4：课程学习 + 阶段内随机地图 + Hybrid + 表现门槛。"""
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


def get_config() -> TrainConfig:
    """根据 ACTIVE_SCHEME 返回当前要跑的配置。

    优先级：环境变量 SNAKE_TRAIN_SCHEME > 文件顶部 ACTIVE_SCHEME。
    """
    scheme = os.environ.get("SNAKE_TRAIN_SCHEME", ACTIVE_SCHEME)
    if scheme == "scheme1":
        return build_scheme1_curriculum()
    if scheme == "scheme2":
        return build_scheme2_random_board()
    if scheme == "scheme3":
        return build_scheme3_hybrid()
    if scheme == "scheme4":
        return build_scheme4_curriculum_random_hybrid()
    raise ValueError(f"未知 ACTIVE_SCHEME: {scheme!r}")
