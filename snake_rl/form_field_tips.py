"""自定义配置表单字段元数据（供 Web UI /api/form-meta 使用）。"""

from __future__ import annotations

from typing import Any, Literal

FieldType = Literal["text", "number", "float", "bool", "select"]

# 与旧 GUI 一致的中文说明
FIELD_TIPS: dict[str, tuple[str, str]] = {
    "run_name": (
        "实验名称",
        "输出到 runs/<名称>/，每次实验用不同名字方便 TensorBoard 横向对比，如 exp01_lr1e4",
    ),
    "output_root": (
        "输出根目录",
        "训练产物的存储根目录（相对项目根），默认 runs，通常不必修改",
    ),
    "device": (
        "计算设备",
        "auto = 有 GPU 就用，没有就用 CPU；也可手动指定 cuda 或 cpu",
    ),
    "episodes": (
        "训练总局数",
        "蛇一共玩多少局。14×14 地图推荐 20000~50000 局，tiny 模型可适当增多",
    ),
    "max_steps_per_episode": (
        "每局最大步数",
        "单局步数硬上限，防止蛇无限转圈。建议设为 board_size² × 10",
    ),
    "model_type": (
        "网络类型",
        "tiny = 极速验证（~5K参数）；adaptive_cnn = 通用推荐（默认）；"
        "hybrid = 最强泛化；small_cnn = 固定地图入门。"
        "【热加载时不生效，由继承的权重决定】",
    ),
    "local_patch_size": (
        "局部感受野",
        "仅 hybrid 模型生效。蛇头周围的观察窗口大小，必须为奇数（如 7、9）。"
        "【热加载时不生效，由继承的权重决定】",
    ),
    "learning_rate": (
        "学习率",
        "最关键的超参数。推荐 1e-4 起步；训练抖动严重时降低 10 倍",
    ),
    "weight_decay": (
        "L2 正则化",
        "防过拟合系数。1e-5 即可，设为 0 关闭。一般不必修改",
    ),
    "gamma": (
        "折扣因子",
        "控制 AI 看重当前还是未来。0.99 表示很看重长远奖励，一般不改",
    ),
    "grad_clip_norm": (
        "梯度裁剪",
        "防止训练时梯度爆炸。loss 暴增时可降到 1.0",
    ),
    "batch_size": (
        "批大小",
        "每次学习从回放池中取多少条经验。128 是平衡速度和稳定性的起点",
    ),
    "replay_capacity": (
        "回放池容量",
        "最多保存多少条历史经验。越大越稳定但占内存。14×14 建议 50000~100000",
    ),
    "min_replay_size": (
        "最少经验数",
        "池子里至少存够多少条才开始学习。建议 batch_size 的 20 倍",
    ),
    "train_frequency": (
        "更新频率",
        "每走 N 步做一次学习更新。4 是常用值，降低可加速学习但不稳定",
    ),
    "target_update_interval": (
        "目标网更新步",
        "每隔多少步把当前网络参数复制给目标网络。1000~3000 都合理",
    ),
    "epsilon_start": (
        "初始探索率",
        "训练初期随机行动的概率。1.0 = 完全随机，让 AI 先广泛尝试",
    ),
    "epsilon_end": (
        "最低探索率",
        "训练后期保留的随机性下限。0.01~0.05，保证 AI 不固化",
    ),
    "epsilon_decay_steps": (
        "探索衰减步数",
        "随机率从最高降到最低需要多少步（注意是步数不是局数！）",
    ),
    "moving_avg_window": (
        "平均窗口",
        "用最近 N 局的平均奖励判断模型好坏并保存 best.pt",
    ),
    "eval_episodes": (
        "评估局数",
        "训练结束后自动评估几局。0 = 不评估，直接结束",
    ),
    "checkpoint_interval": (
        "检查点频率",
        "每隔多少局保存一次模型快照（ep_XXXXX.pt）。建议 200~1000",
    ),
    "log_interval": (
        "日志打印频率",
        "每隔多少局在终端打印一行训练进度",
    ),
    "tensorboard_log_interval": (
        "TensorBoard 频率",
        "每隔多少局写一条 TensorBoard 记录。5 就够了",
    ),
    "jsonl_flush_interval": (
        "JSONL 刷新频率",
        "每隔多少局把详细日志刷到磁盘。防中断丢数据",
    ),
    "tensorboard": (
        "TensorBoard",
        "是否生成 TensorBoard 事件文件。强烈建议开启，方便看训练曲线",
    ),
    "save_csv": (
        "保存 CSV",
        "训练完成后保存 episodes.csv，方便用 Excel / pandas 分析",
    ),
    "save_jsonl": (
        "保存 JSONL",
        "实时写入每局详细日志。即使训练中断，已完成的数据也不会丢",
    ),
    "live_plot": (
        "实时曲线窗口",
        "弹出 Matplotlib 窗口实时显示曲线。服务器或无显示器环境必须关闭",
    ),
    "lightweight_step_info": (
        "轻量统计模式",
        "只在每局结束时收集统计（跳过每步统计），提升训练速度",
    ),
    "env.board_size": (
        "地图尺寸",
        "NxN 的方形地图。入门推荐 10~14，越大越难",
    ),
    "env.difficulty": (
        "难度",
        "easy = 初始蛇短容错高；normal = 标准；hard = 初始蛇长更有挑战",
    ),
    "env.mode": (
        "边界模式",
        "classic = 碰墙即死（推荐入门）；wrap = 穿越到对侧无墙壁",
    ),
    "env.max_steps_without_food": (
        "无食物超时",
        "连续多少步没吃到食物就判超时结束。建议设为 board_size 的平方",
    ),
    "env.enable_bonus_food": (
        "奖励食物",
        "开启后会短暂出现高价值食物。入门阶段建议关闭",
    ),
    "env.enable_obstacles": (
        "随机障碍物",
        "开启后地图随机出现障碍物。入门阶段建议关闭",
    ),
    "env.allow_leveling": (
        "升级系统",
        "开启后达到特定长度触发升级。进阶玩法，入门建议关闭",
    ),
    "env.seed": (
        "随机种子",
        "固定数字可复现每一局。留空则每次随机，适合泛化训练",
    ),
    "rw.alive": (
        "每步存活",
        "每走一步给的奖励。设负值(-0.01)催促蛇别磨蹭快去找食物",
    ),
    "rw.food": (
        "吃到食物",
        "吃到普通食物的奖励，作为基准信号。一般保持 1.0",
    ),
    "rw.bonusFood": (
        "奖励食物",
        "吃到限时高价值食物的奖励。需开启奖励食物功能才生效",
    ),
    "rw.death": (
        "死亡惩罚",
        "碰墙或自咬的惩罚（负值）。-1.5 让蛇学会避险",
    ),
    "rw.timeout": (
        "超时惩罚",
        "长时间找不到食物的惩罚。-1.0 让蛇别原地打转",
    ),
    "rw.levelUp": (
        "升级奖励",
        "触发升级时的奖励。需开启升级系统才生效",
    ),
    "rw.victory": (
        "铺满全图",
        "蛇填满整张地图的终极奖励。罕见但意义重大",
    ),
    "rw.foodDistanceK": (
        "靠近食物系数",
        "每步靠近食物给正奖励、远离给负奖励。0.3~0.5 加速早期学习，过大(>0.6)会让蛇过于贪心",
    ),
}


def form_meta() -> dict[str, Any]:
    """返回前端动态渲染表单所需的结构化描述。"""
    sections: list[dict[str, Any]] = [
        {
            "tab": "basic",
            "tabTitle": "基础",
            "groups": [
                {
                    "title": "实验标识",
                    "fields": [
                        _f("run_name", "text"),
                        _f("output_root", "text"),
                        _f("device", "select", choices=["auto", "cpu", "cuda"]),
                    ],
                },
                {
                    "title": "训练规模",
                    "fields": [
                        _f("episodes", "number"),
                        _f("max_steps_per_episode", "number"),
                        _f("eval_episodes", "number"),
                    ],
                },
                {
                    "title": "网络结构",
                    "fields": [
                        _f(
                            "model_type",
                            "select",
                            choices=["small_cnn", "adaptive_cnn", "hybrid", "tiny"],
                        ),
                        _f("local_patch_size", "number"),
                    ],
                },
                {
                    "title": "探索策略（ε-greedy）",
                    "fields": [
                        _f("epsilon_start", "float"),
                        _f("epsilon_end", "float"),
                        _f("epsilon_decay_steps", "number"),
                    ],
                },
            ],
        },
        {
            "tab": "env",
            "tabTitle": "环境",
            "groups": [
                {
                    "title": "地图设置",
                    "fields": [
                        _f("env.board_size", "number"),
                        _f("env.mode", "select", choices=["classic", "wrap"]),
                        _f("env.difficulty", "select", choices=["easy", "normal", "hard"]),
                        _f("env.max_steps_without_food", "number"),
                        _f("env.seed", "text"),
                    ],
                },
                {
                    "title": "进阶功能（入门建议全部关闭）",
                    "fields": [
                        _f("env.enable_bonus_food", "bool"),
                        _f("env.enable_obstacles", "bool"),
                        _f("env.allow_leveling", "bool"),
                    ],
                },
            ],
        },
        {
            "tab": "reward",
            "tabTitle": "奖励",
            "groups": [
                {
                    "title": "奖励权重",
                    "fields": [
                        _f("rw.alive", "float"),
                        _f("rw.food", "float"),
                        _f("rw.bonusFood", "float"),
                        _f("rw.death", "float"),
                        _f("rw.timeout", "float"),
                        _f("rw.levelUp", "float"),
                        _f("rw.victory", "float"),
                        _f("rw.foodDistanceK", "float"),
                    ],
                },
            ],
        },
        {
            "tab": "advanced",
            "tabTitle": "高级",
            "groups": [
                {
                    "title": "学习超参数",
                    "fields": [
                        _f("learning_rate", "float"),
                        _f("weight_decay", "float"),
                        _f("gamma", "float"),
                        _f("grad_clip_norm", "float"),
                        _f("target_update_interval", "number"),
                    ],
                },
                {
                    "title": "经验回放",
                    "fields": [
                        _f("batch_size", "number"),
                        _f("replay_capacity", "number"),
                        _f("min_replay_size", "number"),
                        _f("train_frequency", "number"),
                    ],
                },
                {
                    "title": "日志与输出",
                    "fields": [
                        _f("checkpoint_interval", "number"),
                        _f("log_interval", "number"),
                        _f("tensorboard_log_interval", "number"),
                        _f("jsonl_flush_interval", "number"),
                        _f("moving_avg_window", "number"),
                        _f("tensorboard", "bool"),
                        _f("save_csv", "bool"),
                        _f("save_jsonl", "bool"),
                        _f("live_plot", "bool"),
                        _f("lightweight_step_info", "bool"),
                    ],
                },
            ],
        },
    ]
    return {"sections": sections}


def _f(
    key: str,
    typ: FieldType,
    *,
    choices: list[str] | None = None,
) -> dict[str, Any]:
    label, tip = FIELD_TIPS.get(key, (key, ""))
    d: dict[str, Any] = {"key": key, "label": label, "tip": tip, "type": typ}
    if choices is not None:
        d["choices"] = choices
    return d
