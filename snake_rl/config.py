from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


ModelType = Literal["small_cnn", "adaptive_cnn", "hybrid"]
"""
三种网络架构：
  small_cnn    : 原始固定尺寸 CNN（Flatten+FC），只能用于固定 board_size。
  adaptive_cnn : Global Average Pooling CNN，支持任意尺寸，用于方案1/2。
  hybrid       : CNN + 手工全局特征融合，跨尺寸泛化最强，用于方案3/4。
"""


@dataclass(slots=True)
class EnvPreset:
    difficulty: str = "normal"
    mode: str = "classic"
    board_size: int = 22
    enable_bonus_food: bool = False
    enable_obstacles: bool = False
    allow_leveling: bool = False
    max_steps_without_food: int = 250
    seed: int | None = 42


@dataclass(slots=True)
class CurriculumStage:
    """课程学习的单个阶段配置。

    每个阶段可以：
    - 在单一 `board_size` 上训练
    - 或在 `board_sizes` 指定的一组尺寸中随机采样训练

    结束后将权重迁移到下一阶段继续训练（而不是从零开始）。

    Attributes:
        board_size         : 本阶段固定地图尺寸。
        board_sizes        : 若提供，则本阶段每个 episode 从该列表随机采样尺寸。
        weights            : 对应 `board_sizes` 的采样权重（None = 均匀）。
        episodes           : 本阶段训练回合数。
        max_steps_without_food: 超时步数，建议约为 board_size²。
        max_steps_scale    : 当使用 `board_sizes` 随机采样时，
                             超时步数 = board_size² * max_steps_scale。
        epsilon_start      : 本阶段起始探索率。
                             第一阶段用 1.0；后续阶段可适当降低（0.3~0.5），
                             因为之前的权重已有一定策略，不需要完全重新探索。
        epsilon_end        : 本阶段结束时的最低探索率。
        epsilon_decay_steps: 本阶段内 epsilon 衰减所需步数。
        replay_capacity    : 本阶段经验池容量（尺寸越大建议越大）。
        min_replay_size    : 开始训练所需的最小经验数。
        promotion_threshold_foods : 晋升门槛（0=关闭，正数=最近N局平均食物数达到此值才晋升）。
        promotion_window   : 晋升检查的滑动窗口大小（局数）。
        promotion_min_episodes : 本阶段至少训练多少局后才开始检查晋升条件。
    """
    board_size: int = 14
    board_sizes: list[int] | None = None
    weights: list[float] | None = None
    episodes: int = 1000
    max_steps_without_food: int = 200
    max_steps_scale: float = 1.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50000
    replay_capacity: int = 15000
    min_replay_size: int = 1500
    promotion_threshold_foods: float = 0.0
    promotion_window: int = 100
    promotion_min_episodes: int = 200


@dataclass(slots=True)
class CurriculumConfig:
    """课程学习总配置（方案1：逐步放大地图）。

    Attributes:
        stages         : 按顺序执行的训练阶段列表。
        carry_replay   : 是否把上一阶段的经验池迁移到下一阶段。
                         True  = 迁移（热启动，收敛更快，但旧经验分布不同）。
                         False = 每阶段清空重建（更干净，推荐）。
        scale_timeout  : 若为 True，自动将每阶段的 max_steps_without_food
                         覆盖为 board_size²，无视 CurriculumStage 中的设置。
    """
    stages: list[CurriculumStage] = field(default_factory=list)
    carry_replay: bool = False
    scale_timeout: bool = True


@dataclass(slots=True)
class RandomBoardConfig:
    """随机地图尺寸训练配置（方案2）。

    每个 episode 从 board_sizes 中随机（或按权重）抽取一个尺寸，
    要求网络必须是 adaptive_cnn 或 hybrid（支持可变输入）。

    Attributes:
        board_sizes    : 可选地图尺寸列表。
        weights        : 对应每个尺寸的采样权重（None = 均匀）。
                         例如 [1, 2, 2, 1] 表示中间尺寸出现频率更高。
        max_steps_scale: max_steps_without_food = board_size * board_size * max_steps_scale。
    """
    board_sizes: list[int] = field(default_factory=lambda: [8, 10, 12, 16])
    weights: list[float] | None = None
    max_steps_scale: float = 1.0


@dataclass(slots=True)
class ParallelRolloutConfig:
    """并行采样配置（actor 多进程 + learner 单进程）。"""

    enabled: bool = False
    num_workers: int = 4
    queue_capacity: int = 8192
    weight_sync_interval_steps: int = 512
    actor_loop_sleep_ms: int = 0
    actor_seed_stride: int = 100_000
    actor_device: str = "cpu"


@dataclass(slots=True)
class TrainConfig:
    episodes: int = 3000
    max_steps_per_episode: int = 2500
    gamma: float = 0.99
    learning_rate: float = 2.5e-4
    weight_decay: float = 0.0
    batch_size: int = 128
    replay_capacity: int = 20000
    min_replay_size: int = 2000
    train_frequency: int = 4
    target_update_interval: int = 1000
    grad_clip_norm: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100000
    eval_episodes: int = 20
    moving_avg_window: int = 100
    log_interval: int = 10
    checkpoint_interval: int = 100
    tensorboard_log_interval: int = 5
    jsonl_flush_interval: int = 20
    run_name: str = "default"
    output_root: Path = Path("runs")
    live_plot: bool = False
    tensorboard: bool = True
    save_csv: bool = True
    save_jsonl: bool = True
    device: str = "auto"
    lightweight_step_info: bool = True
    # 网络架构选择
    model_type: ModelType = "small_cnn"
    # hybrid 模型使用的局部 patch 大小，必须是奇数，例如 9/11/13
    local_patch_size: int = 11
    # 课程学习（方案1），非 None 时忽略 episodes/env.board_size 等顶层参数
    curriculum: CurriculumConfig | None = None
    # 随机地图（方案2），非 None 时要求 model_type 为 adaptive_cnn 或 hybrid
    random_board: RandomBoardConfig | None = None
    # 并行采样配置（默认关闭，保持原有串行行为）
    parallel: ParallelRolloutConfig = field(default_factory=ParallelRolloutConfig)
    reward_weights: dict[str, float] | None = None
    env: EnvPreset = field(default_factory=EnvPreset)


def resolve_device(device: str) -> str:
    """Resolve requested device name."""
    if device != "auto":
        return device
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_config_from_dict(data: dict[str, Any]) -> TrainConfig:
    """从 run_config.json / training state 反序列化 TrainConfig。"""
    env_raw = data.get("env") or {}
    seed_raw = env_raw.get("seed")
    env = EnvPreset(
        difficulty=str(env_raw.get("difficulty", "normal")),
        mode=str(env_raw.get("mode", "classic")),
        board_size=int(env_raw.get("board_size", 22)),
        enable_bonus_food=bool(env_raw.get("enable_bonus_food", False)),
        enable_obstacles=bool(env_raw.get("enable_obstacles", False)),
        allow_leveling=bool(env_raw.get("allow_leveling", False)),
        max_steps_without_food=int(env_raw.get("max_steps_without_food", 250)),
        seed=int(seed_raw) if seed_raw is not None else None,
    )

    curriculum = None
    cur_raw = data.get("curriculum")
    if isinstance(cur_raw, dict) and cur_raw.get("stages"):
        stages: list[CurriculumStage] = []
        for s in cur_raw["stages"]:
            if not isinstance(s, dict):
                continue
            stages.append(
                CurriculumStage(
                    board_size=int(s.get("board_size", 14)),
                    board_sizes=list(s["board_sizes"]) if s.get("board_sizes") else None,
                    weights=list(s["weights"]) if s.get("weights") is not None else None,
                    episodes=int(s.get("episodes", 1000)),
                    max_steps_without_food=int(s.get("max_steps_without_food", 200)),
                    max_steps_scale=float(s.get("max_steps_scale", 1.0)),
                    epsilon_start=float(s.get("epsilon_start", 1.0)),
                    epsilon_end=float(s.get("epsilon_end", 0.05)),
                    epsilon_decay_steps=int(s.get("epsilon_decay_steps", 50000)),
                    replay_capacity=int(s.get("replay_capacity", 15000)),
                    min_replay_size=int(s.get("min_replay_size", 1500)),
                    promotion_threshold_foods=float(s.get("promotion_threshold_foods", 0.0)),
                    promotion_window=int(s.get("promotion_window", 100)),
                    promotion_min_episodes=int(s.get("promotion_min_episodes", 200)),
                )
            )
        curriculum = CurriculumConfig(
            stages=stages,
            carry_replay=bool(cur_raw.get("carry_replay", False)),
            scale_timeout=bool(cur_raw.get("scale_timeout", True)),
        )

    random_board = None
    rb_raw = data.get("random_board")
    if isinstance(rb_raw, dict) and rb_raw.get("board_sizes"):
        random_board = RandomBoardConfig(
            board_sizes=[int(x) for x in rb_raw["board_sizes"]],
            weights=list(rb_raw["weights"]) if rb_raw.get("weights") is not None else None,
            max_steps_scale=float(rb_raw.get("max_steps_scale", 1.0)),
        )

    par_raw = data.get("parallel") or {}
    parallel = ParallelRolloutConfig(
        enabled=bool(par_raw.get("enabled", False)),
        num_workers=int(par_raw.get("num_workers", 4)),
        queue_capacity=int(par_raw.get("queue_capacity", 8192)),
        weight_sync_interval_steps=int(par_raw.get("weight_sync_interval_steps", 512)),
        actor_loop_sleep_ms=int(par_raw.get("actor_loop_sleep_ms", 0)),
        actor_seed_stride=int(par_raw.get("actor_seed_stride", 100_000)),
        actor_device=str(par_raw.get("actor_device", "cpu")),
    )

    rw = data.get("reward_weights")
    reward_weights: dict[str, float] | None = None
    if isinstance(rw, dict):
        reward_weights = {str(k): float(v) for k, v in rw.items()}

    out_root = data.get("output_root", "runs")
    return TrainConfig(
        episodes=int(data.get("episodes", 3000)),
        max_steps_per_episode=int(data.get("max_steps_per_episode", 2500)),
        gamma=float(data.get("gamma", 0.99)),
        learning_rate=float(data.get("learning_rate", 2.5e-4)),
        weight_decay=float(data.get("weight_decay", 0.0)),
        batch_size=int(data.get("batch_size", 128)),
        replay_capacity=int(data.get("replay_capacity", 20000)),
        min_replay_size=int(data.get("min_replay_size", 2000)),
        train_frequency=int(data.get("train_frequency", 4)),
        target_update_interval=int(data.get("target_update_interval", 1000)),
        grad_clip_norm=float(data.get("grad_clip_norm", 10.0)),
        epsilon_start=float(data.get("epsilon_start", 1.0)),
        epsilon_end=float(data.get("epsilon_end", 0.05)),
        epsilon_decay_steps=int(data.get("epsilon_decay_steps", 100000)),
        eval_episodes=int(data.get("eval_episodes", 20)),
        moving_avg_window=int(data.get("moving_avg_window", 100)),
        log_interval=int(data.get("log_interval", 10)),
        checkpoint_interval=int(data.get("checkpoint_interval", 100)),
        tensorboard_log_interval=int(data.get("tensorboard_log_interval", 5)),
        jsonl_flush_interval=int(data.get("jsonl_flush_interval", 20)),
        run_name=str(data.get("run_name", "default")),
        output_root=Path(str(out_root)),
        live_plot=bool(data.get("live_plot", True)),
        tensorboard=bool(data.get("tensorboard", False)),
        save_csv=bool(data.get("save_csv", True)),
        save_jsonl=bool(data.get("save_jsonl", True)),
        device=str(data.get("device", "auto")),
        lightweight_step_info=bool(data.get("lightweight_step_info", True)),
        model_type=data.get("model_type", "small_cnn"),  # type: ignore[arg-type]
        local_patch_size=int(data.get("local_patch_size", 11)),
        curriculum=curriculum,
        random_board=random_board,
        parallel=parallel,
        reward_weights=reward_weights,
        env=env,
    )
