"""Experiment configuration — single source of truth for trainers and the UI."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _meta(
    *,
    label: str,
    help: str,
    advanced: bool = False,
    live: bool = False,
    min: float | None = None,
    max: float | None = None,
    step: float | None = None,
    unit: str | None = None,
    choices: list[Any] | None = None,
) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "label": label,
        "help": help,
        "advanced": advanced,
        "live": live,
    }
    if min is not None:
        extra["min"] = min
    if max is not None:
        extra["max"] = max
    if step is not None:
        extra["step"] = step
    if unit is not None:
        extra["unit"] = unit
    if choices is not None:
        extra["choices"] = choices
    return extra


class EnvConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_size: int = Field(
        default=8,
        json_schema_extra=_meta(
            label="最小棋盘边长",
            help="训练时随机棋盘的下限。调大则环境更难、早期学习变慢；调小则更容易得分、适合入门。",
            min=5,
            max=32,
            step=1,
        ),
    )
    max_size: int = Field(
        default=8,
        json_schema_extra=_meta(
            label="最大棋盘边长",
            help="训练时随机棋盘的上限，也是默认评估尺寸。调大提升泛化但更难；调小收敛更快。",
            min=5,
            max=32,
            step=1,
        ),
    )
    hunger_factor: float = Field(
        default=1.0,
        json_schema_extra=_meta(
            label="饥饿系数",
            help="多久没吃到食物算饿死（与棋盘面积相关）。调大更宽容、少饿死；调小逼迫主动找食物。",
            advanced=True,
            min=0.25,
            max=4.0,
            step=0.05,
        ),
    )

    @field_validator("min_size", "max_size")
    @classmethod
    def _size_range(cls, v: int) -> int:
        if not 5 <= v <= 32:
            raise ValueError("棋盘边长必须在 5 到 32 之间")
        return v

    @model_validator(mode="after")
    def _min_le_max(self) -> EnvConfig:
        if self.min_size > self.max_size:
            raise ValueError("最小棋盘边长不能大于最大棋盘边长")
        return self


class RewardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    food: float = Field(
        default=1.0,
        json_schema_extra=_meta(
            label="吃到食物奖励",
            help="吃到食物时加的分。调大更积极追食物；调小则更在意别死、别饿死。",
            live=True,
            min=-5.0,
            max=10.0,
            step=0.05,
        ),
    )
    death: float = Field(
        default=-1.0,
        json_schema_extra=_meta(
            label="撞墙/咬自己惩罚",
            help="撞墙或咬到自己时的惩罚。调得更负会更保守避障；调小（接近 0）则更敢冒险。",
            live=True,
            min=-10.0,
            max=0.0,
            step=0.05,
        ),
    )
    step: float = Field(
        default=-0.005,
        json_schema_extra=_meta(
            label="每步存活代价",
            help="每走一步的小惩罚，鼓励尽快吃到食物。调得更负更急；调接近 0 则更愿绕路。",
            live=True,
            advanced=True,
            min=-0.1,
            max=0.05,
            step=0.0005,
        ),
    )
    approach: float = Field(
        default=0.05,
        json_schema_extra=_meta(
            label="靠近食物奖励",
            help="曼哈顿距离靠近食物得正、远离得负。调大早期学得更快；调小或为 0 则更依赖远期回报。",
            live=True,
            min=-0.5,
            max=0.5,
            step=0.005,
        ),
    )
    starve: float = Field(
        default=-0.5,
        json_schema_extra=_meta(
            label="饿死惩罚",
            help="长时间不吃食物饿死时的惩罚。调得更负更怕饿死；调小则允许更多探索绕路。",
            live=True,
            advanced=True,
            min=-5.0,
            max=0.0,
            step=0.05,
        ),
    )
    win: float = Field(
        default=5.0,
        json_schema_extra=_meta(
            label="通关奖励",
            help="蛇填满棋盘时的大奖。调大更追求完美通关；调小则主要优化普通得分。",
            live=True,
            advanced=True,
            min=0.0,
            max=50.0,
            step=0.5,
        ),
    )

    def as_tensor_list(self) -> list[float]:
        return [self.food, self.death, self.step, self.approach, self.starve, self.win]


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: float = Field(
        default=1.0,
        json_schema_extra=_meta(
            label="网络宽度倍率",
            help="卷积/全连接通道数倍率。调大（如 1.5–2）表达力更强但更慢更吃显存；调小（如 0.5）更快但上限更低。",
            advanced=True,
            min=0.5,
            max=2.0,
            step=0.5,
            choices=[0.5, 1.0, 1.5, 2.0],
        ),
    )
    resize_obs: bool = Field(
        default=False,
        json_schema_extra=_meta(
            label="观测缩放到固定分辨率",
            help="把自我中心窗口双线性缩放到 31×31 再进网络。打开后多尺寸泛化更好，但固定小棋盘会变慢；通才预设建议打开，快速入门保持关闭。",
            advanced=True,
        ),
    )


class PPOConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_envs: int = Field(
        default=1024,
        json_schema_extra=_meta(
            label="并行环境数",
            help="同时模拟多少条蛇。调大吞吐更高、梯度更稳但显存占用更大；调小更省显存、更新更噪。",
            min=1,
            max=8192,
            step=1,
        ),
    )
    rollout: int = Field(
        default=128,
        json_schema_extra=_meta(
            label="每次采集步数",
            help="每次更新前每个环境走多少步。调大样本更远、更新更少；调小更新更勤、样本相关性更高。",
            advanced=True,
            min=16,
            max=512,
            step=8,
        ),
    )
    epochs: int = Field(
        default=4,
        json_schema_extra=_meta(
            label="每批训练轮数",
            help="同一批轨迹重复训练几轮。调大样本利用率高但易过拟合；调小更保守。",
            advanced=True,
            min=1,
            max=16,
            step=1,
        ),
    )
    minibatches: int = Field(
        default=8,
        json_schema_extra=_meta(
            label="小批量份数",
            help="把一次采集切成几份做梯度更新。调大每步更稳更省显存；调小步子更大、更快但不稳。",
            advanced=True,
            min=1,
            max=64,
            step=1,
        ),
    )
    lr: float = Field(
        default=3e-4,
        json_schema_extra=_meta(
            label="学习率",
            help="参数更新步长。调大学得快但可能发散；调小更稳但更慢。",
            live=True,
            min=1e-6,
            max=1e-2,
            step=1e-5,
        ),
    )
    lr_end: float = Field(
        default=5e-5,
        json_schema_extra=_meta(
            label="学习率终点",
            help="后期学习率衰减到的值。调小后期更细调、少破坏已学策略；设成与起始学习率相同等于不衰减。",
            advanced=True,
            live=True,
            min=1e-7,
            max=1e-2,
            step=1e-5,
        ),
    )
    lr_anneal_steps: int = Field(
        default=10_000_000,
        json_schema_extra=_meta(
            label="学习率衰减环境步数",
            help="从起始学习率线性降到终点所需环境步。调大前期保持高学习率更久；设 0 关闭衰减。",
            advanced=True,
            min=0,
            max=100_000_000,
            step=100_000,
        ),
    )
    gamma: float = Field(
        default=0.99,
        json_schema_extra=_meta(
            label="折扣因子 γ",
            help="未来奖励的重视程度。调大更看长远得分；调小更关注眼前几步。",
            live=True,
            advanced=True,
            min=0.8,
            max=0.999,
            step=0.001,
        ),
    )
    gae_lambda: float = Field(
        default=0.95,
        json_schema_extra=_meta(
            label="GAE λ",
            help="优势估计的偏差-方差权衡。调大偏差小方差大；调小更平滑但有偏。",
            advanced=True,
            min=0.5,
            max=1.0,
            step=0.01,
        ),
    )
    clip: float = Field(
        default=0.2,
        json_schema_extra=_meta(
            label="PPO 裁剪范围",
            help="限制策略更新幅度。调大允许更剧烈更新；调小更保守防崩溃。",
            live=True,
            advanced=True,
            min=0.05,
            max=0.5,
            step=0.01,
        ),
    )
    ent_coef: float = Field(
        default=0.01,
        json_schema_extra=_meta(
            label="熵正则系数",
            help="鼓励探索的程度。调大动作更随机、少早熟；调小更贪心利用已学策略。",
            live=True,
            min=0.0,
            max=0.1,
            step=0.001,
        ),
    )
    ent_coef_end: float = Field(
        default=0.001,
        json_schema_extra=_meta(
            label="熵系数终点",
            help="训练后期熵系数衰减到的值。调小后期更敢贪心走满棋盘；调大则一直保持探索。",
            advanced=True,
            live=True,
            min=0.0,
            max=0.05,
            step=0.0005,
        ),
    )
    ent_anneal_steps: int = Field(
        default=8_000_000,
        json_schema_extra=_meta(
            label="熵衰减环境步数",
            help="从起始熵系数线性降到终点所需环境步。调大探索更久；调小更快转入利用。设 0 关闭衰减。",
            advanced=True,
            min=0,
            max=100_000_000,
            step=100_000,
        ),
    )
    vf_coef: float = Field(
        default=0.5,
        json_schema_extra=_meta(
            label="价值损失系数",
            help="价值网络损失权重。调大更重视估值得准；调小更偏重策略改进。",
            advanced=True,
            min=0.0,
            max=2.0,
            step=0.05,
        ),
    )
    max_grad_norm: float = Field(
        default=0.5,
        json_schema_extra=_meta(
            label="梯度裁剪阈值",
            help="限制单次更新梯度范数。调大允许更大更新；调小训练更稳。",
            advanced=True,
            min=0.1,
            max=5.0,
            step=0.1,
        ),
    )
    normalize_returns: bool = Field(
        default=True,
        json_schema_extra=_meta(
            label="回报归一化",
            help="用批次回报方差缩放价值损失，让价值学习更稳且不改变价值量纲。打开通常更稳；关掉更接近教科书写法。",
            advanced=True,
        ),
    )


class DQNConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_envs: int = Field(
        default=256,
        json_schema_extra=_meta(
            label="并行环境数",
            help="同时采集经验的环境数。调大填充回放更快；调小更省显存。",
            min=1,
            max=2048,
            step=1,
        ),
    )
    replay_size: int = Field(
        default=200_000,
        json_schema_extra=_meta(
            label="回放缓冲区大小",
            help="GPU 上保存多少条转移。调大样本更多样但更吃显存；调小更新更依赖近期经验。",
            advanced=True,
            min=10_000,
            max=1_000_000,
            step=10_000,
        ),
    )
    n_step: int = Field(
        default=3,
        json_schema_extra=_meta(
            label="n-step 步数",
            help="多步回报长度。调大信用分配更远、学吃食物更快；调小更稳但偏短视。",
            advanced=True,
            min=1,
            max=10,
            step=1,
        ),
    )
    batch_size: int = Field(
        default=512,
        json_schema_extra=_meta(
            label="训练批量",
            help="每次梯度更新抽多少样本。调大更稳更吃显存；调小噪声大、更新快。",
            advanced=True,
            min=32,
            max=4096,
            step=32,
        ),
    )
    lr: float = Field(
        default=2.5e-4,
        json_schema_extra=_meta(
            label="学习率",
            help="Q 网络更新步长。调大学得快易抖；调小更稳更慢。",
            live=True,
            min=1e-6,
            max=1e-2,
            step=1e-5,
        ),
    )
    gamma: float = Field(
        default=0.99,
        json_schema_extra=_meta(
            label="折扣因子 γ",
            help="未来奖励权重。调大更看长远；调小更看眼前。",
            live=True,
            advanced=True,
            min=0.8,
            max=0.999,
            step=0.001,
        ),
    )
    tau: float = Field(
        default=0.01,
        json_schema_extra=_meta(
            label="目标网络软更新 τ",
            help="目标网络向在线网络靠拢的速度。调大目标更新更快但不稳；调小更稳更慢。",
            advanced=True,
            min=0.001,
            max=0.1,
            step=0.001,
        ),
    )
    epsilon_start: float = Field(
        default=1.0,
        json_schema_extra=_meta(
            label="探索率起点",
            help="训练开始时的随机动作比例。调大开局更乱更探索；调小开局更跟策略。",
            advanced=True,
            min=0.0,
            max=1.0,
            step=0.05,
        ),
    )
    epsilon_end: float = Field(
        default=0.02,
        json_schema_extra=_meta(
            label="探索率终点",
            help="衰减结束后的最小随机比例。调大始终保留探索；调小后期更贪心。",
            live=True,
            min=0.0,
            max=0.5,
            step=0.01,
        ),
    )
    epsilon_decay_steps: int = Field(
        default=1_000_000,
        json_schema_extra=_meta(
            label="探索衰减步数",
            help="从起点衰减到终点所需环境步数。调大探索更久；调小更快转入利用。",
            advanced=True,
            min=10_000,
            max=20_000_000,
            step=10_000,
        ),
    )
    train_freq: int = Field(
        default=16,
        json_schema_extra=_meta(
            label="更新频率（环境步/次）",
            help="大约每多少环境步做一次梯度更新。调大更新更稀；调小更新更勤、更吃算力。",
            advanced=True,
            min=1,
            max=256,
            step=1,
        ),
    )
    learning_starts: int = Field(
        default=10_000,
        json_schema_extra=_meta(
            label="开始学习步数",
            help="回放攒够多少步后才开始更新。调大初期更随机；调小更早开始学。",
            advanced=True,
            min=1000,
            max=200_000,
            step=1000,
        ),
    )
    steps_per_iteration: int = Field(
        default=256,
        json_schema_extra=_meta(
            label="每轮采集步数",
            help="一次 train_iteration 里每个环境走多少步。调大每轮更长；调小 UI 刷新更勤。",
            advanced=True,
            min=16,
            max=2048,
            step=16,
        ),
    )


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_env_steps: int = Field(
        default=0,
        json_schema_extra=_meta(
            label="最大环境步数",
            help="到步数自动结束；0 表示一直跑到你手动停止。调大训练更久；调小更快结束。",
            min=0,
            max=1_000_000_000,
            step=100_000,
        ),
    )
    seed: int | None = Field(
        default=None,
        json_schema_extra=_meta(
            label="随机种子",
            help="固定后可复现同一训练过程。设为具体整数可复现；留空则每次不同。",
            advanced=True,
            min=0,
            max=2_147_483_647,
            step=1,
        ),
    )
    device: Literal["auto", "cuda", "cpu"] = Field(
        default="auto",
        json_schema_extra=_meta(
            label="计算设备",
            help="auto 优先用 GPU。选 cuda 强制 GPU（无卡会报错）；选 cpu 更慢但可调试。",
            advanced=True,
            choices=[
                {"value": "auto", "label": "自动（有显卡就用显卡）"},
                {"value": "cuda", "label": "显卡 GPU"},
                {"value": "cpu", "label": "处理器 CPU"},
            ],
        ),
    )
    eval_every_s: float = Field(
        default=30.0,
        json_schema_extra=_meta(
            label="评估间隔（秒）",
            help="每隔多少秒做一次贪心评估并可能保存 best。调小评估更勤更耗时；调大更专注训练。",
            advanced=True,
            min=5.0,
            max=600.0,
            step=5.0,
            unit="s",
        ),
    )
    compile: bool = Field(
        default=False,
        json_schema_extra=_meta(
            label="torch.compile",
            help="尝试编译网络加速。打开可能更快但 Windows 上常不稳定；关闭更稳妥。",
            advanced=True,
        ),
    )


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        default="实验",
        json_schema_extra=_meta(
            label="实验名称",
            help="显示在实验室列表里的名字。改成好认的名字方便对比；留默认也行。",
        ),
    )
    algo: Literal["ppo", "dqn"] = Field(
        default="ppo",
        json_schema_extra=_meta(
            label="算法",
            help="PPO 通常更稳、适合边看边调；DQN 样本效率不同，适合对照。改算法会重建训练器。",
            choices=[
                {"value": "ppo", "label": "PPO（推荐入门）"},
                {"value": "dqn", "label": "DQN（对照组）"},
            ],
        ),
    )
    env: EnvConfig = Field(default_factory=EnvConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    dqn: DQNConfig = Field(default_factory=DQNConfig)
    run: RunConfig = Field(default_factory=RunConfig)


def live_field_keys(algo: str | None = None) -> set[str]:
    keys = {
        "reward.food",
        "reward.death",
        "reward.step",
        "reward.approach",
        "reward.starve",
        "reward.win",
    }
    if algo is None or algo == "ppo":
        keys |= {"ppo.lr", "ppo.lr_end", "ppo.ent_coef", "ppo.ent_coef_end", "ppo.gamma", "ppo.clip"}
    if algo is None or algo == "dqn":
        keys |= {"dqn.lr", "dqn.gamma", "dqn.epsilon_end"}
    return keys


def resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求使用 CUDA，但当前环境没有可用的 GPU")
    return device


_GROUP_DEFS: list[tuple[str, str, str, type[BaseModel], str | None]] = [
    ("env", "环境", "棋盘大小与饥饿规则", EnvConfig, None),
    ("reward", "奖励", "奖励分量权重（可训练中实时调整）", RewardConfig, None),
    ("model", "模型", "神经网络容量", ModelConfig, None),
    ("ppo", "PPO", "近端策略优化超参数", PPOConfig, "ppo"),
    ("dqn", "DQN", "深度 Q 网络超参数", DQNConfig, "dqn"),
    ("run", "运行", "设备、种子与评估节奏", RunConfig, None),
]


def _field_schema(prefix: str, name: str, field_info: Any, algo: str | None) -> dict[str, Any]:
    extra = field_info.json_schema_extra or {}
    if callable(extra):
        extra = {}
    ann = field_info.annotation
    typ = "number"
    choices = extra.get("choices")
    if ann is bool:
        typ = "boolean"
    elif ann is int:
        typ = "integer"
    elif ann is str:
        typ = "string"
    elif choices and all(isinstance(c, str | dict) for c in choices):
        typ = "select"

    default = field_info.default
    if default is None and getattr(field_info, "default_factory", None) is not None:
        default = field_info.default_factory()

    out: dict[str, Any] = {
        "key": f"{prefix}.{name}" if prefix else name,
        "label": extra.get("label", name),
        "help": extra.get("help", ""),
        "type": typ,
        "default": default,
        "advanced": bool(extra.get("advanced", False)),
        "live": bool(extra.get("live", False)),
        "algo": algo,
    }
    for k in ("min", "max", "step", "unit"):
        if k in extra:
            out[k] = extra[k]
    if choices is not None:
        # Frontend FieldControl expects {value, label} objects
        out["choices"] = [c if isinstance(c, dict) else {"value": c, "label": str(c)} for c in choices]
    return out


def ui_schema() -> dict[str, Any]:
    groups: list[dict[str, Any]] = [
        {
            "key": "meta",
            "label": "基本信息",
            "description": "实验名称与算法",
            "fields": [
                _field_schema("", "name", ExperimentConfig.model_fields["name"], None),
                _field_schema("", "algo", ExperimentConfig.model_fields["algo"], None),
            ],
        }
    ]
    for key, label, desc, model_cls, algo in _GROUP_DEFS:
        fields = [_field_schema(key, fname, finfo, algo) for fname, finfo in model_cls.model_fields.items()]
        groups.append({"key": key, "label": label, "description": desc, "fields": fields})

    return {"groups": groups, "presets": [p.model_dump() for p in PRESETS]}


class Preset(BaseModel):
    id: str
    name: str
    description: str
    config: ExperimentConfig


PRESETS: list[Preset] = [
    Preset(
        id="quick_8x8",
        name="快速入门 · 8×8",
        description=(
            "固定 8×8，优先让新手尽快看到分数爬升。"
            "约 2–3 分钟平均能吃到 20 个左右；约 8 分钟平均约 40；"
            "约 15 分钟可接近满分 61 并开始偶发通关。后期几乎只死于撞自己。"
        ),
        config=ExperimentConfig(
            name="快速入门 · 8×8",
            algo="ppo",
            env=EnvConfig(min_size=8, max_size=8),
        ),
    ),
    Preset(
        id="champion_10x10",
        name="长训冠军 · 10×10",
        description=(
            "固定 10×10，加宽网络与更长回报视野，适合挂机冲高分。"
            "约 12 分钟平均约 50（满分 97）；15 分钟以上仍主要死于撞自己，"
            "纯 RL 很难稳定填满。不适合第一次上手。"
        ),
        config=ExperimentConfig(
            name="长训冠军 · 10×10",
            algo="ppo",
            env=EnvConfig(min_size=10, max_size=10),
            model=ModelConfig(width=1.5),
            ppo=PPOConfig(
                ent_coef=0.012,
                ent_coef_end=0.0005,
                ent_anneal_steps=10_000_000,
                gamma=0.995,
                lr_end=5e-5,
                lr_anneal_steps=12_000_000,
            ),
        ),
    ),
    Preset(
        id="generalist_6_16",
        name="多尺寸通才 · 6–16",
        description=(
            "在 6–16 随机尺寸上训练，并开启固定分辨率观测以便跨尺寸泛化。"
            "约 12–20 分钟后，训练范围内与未见的 20×20 都能吃到几十个；"
            "更大如 24×24 仍偏弱。比固定棋盘学得慢，吞吐也更低。"
        ),
        config=ExperimentConfig(
            name="多尺寸通才 · 6–16",
            algo="ppo",
            env=EnvConfig(min_size=6, max_size=16, hunger_factor=1.25),
            model=ModelConfig(width=1.5, resize_obs=True),
            ppo=PPOConfig(
                num_envs=768,
                rollout=128,
                ent_coef=0.015,
                ent_coef_end=0.001,
                ent_anneal_steps=10_000_000,
                gamma=0.995,
                lr_end=5e-5,
                lr_anneal_steps=12_000_000,
            ),
        ),
    ),
    Preset(
        id="dqn_8x8",
        name="DQN 对照组 · 8×8",
        description="同一 8×8 设定下的 Double Dueling DQN，方便和 PPO 对比学习曲线，不是冲分首选。",
        config=ExperimentConfig(
            name="DQN 对照组 · 8×8",
            algo="dqn",
            env=EnvConfig(min_size=8, max_size=8),
        ),
    ),
    Preset(
        id="challenge_20",
        name="挑战 · 20×20",
        description=(
            "大棋盘高压训练：加宽网络、提高饥饿宽容。"
            "约 15 分钟通常只能平均吃到几个；要看到明显进步往往需要 1 小时以上。"
            "想冲分请先用 8×8 / 10×10。"
        ),
        config=ExperimentConfig(
            name="挑战 · 20×20",
            algo="ppo",
            env=EnvConfig(min_size=20, max_size=20, hunger_factor=1.5),
            reward=RewardConfig(approach=0.03, step=-0.003),
            model=ModelConfig(width=1.5),
            ppo=PPOConfig(
                num_envs=512,
                rollout=256,
                ent_coef=0.02,
                ent_coef_end=0.002,
                ent_anneal_steps=12_000_000,
                gamma=0.995,
                lr=2.5e-4,
                lr_end=5e-5,
                lr_anneal_steps=15_000_000,
            ),
        ),
    ),
]


def get_preset(preset_id: str) -> ExperimentConfig:
    for p in PRESETS:
        if p.id == preset_id:
            return p.config.model_copy(deep=True)
    raise KeyError(f"未知预设: {preset_id}")
