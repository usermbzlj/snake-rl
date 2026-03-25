"""自定义配置表单字段元数据（供 Web UI /api/form-meta 使用）。"""

from __future__ import annotations

from typing import Any, Literal

FieldType = Literal["text", "number", "float", "bool", "select"]

# 与旧 GUI 一致的中文说明
FIELD_TIPS: dict[str, tuple[str, str]] = {
    "run_name": ("实验名称", "输出目录 runs/<名称>/，每次实验建议用不同名字方便 TensorBoard 对比"),
    "output_root": ("输出根目录", "训练产物存储位置（相对项目根），通常不必修改"),
    "device": ("计算设备", "auto = 有 GPU 自动用，否则用 CPU；也可填 cuda / cpu"),
    "episodes": ("训练总局数", "训练跑多少局；14×14 地图建议 20000~50000"),
    "max_steps_per_episode": ("每局最大步数", "硬上限，防止蛇无限绕圈；建议 board_size² × 10 左右"),
    "model_type": ("网络类型", "small_cnn=入门首选；adaptive_cnn/hybrid 支持可变地图"),
    "local_patch_size": ("局部感受野", "仅 hybrid 模型有效，蛇头周围观察范围（奇数）"),
    "learning_rate": ("学习率", "最重要超参；推荐 1e-4，不稳定时降低 10 倍"),
    "weight_decay": ("L2 正则化", "防过拟合，1e-5 即可；0 = 关闭"),
    "gamma": ("折扣因子", "越接近 1 越看重长远奖励；贪吃蛇推荐 0.99，一般不改"),
    "grad_clip_norm": ("梯度裁剪", "防梯度爆炸；loss 暴增时可降低到 1.0"),
    "batch_size": ("批大小", "每次更新采样多少条经验；128 是入门平衡点"),
    "replay_capacity": ("回放池容量", "最多保存多少条历史经验；14×14 建议 50000~100000"),
    "min_replay_size": ("最少经验数", "收集够这么多条经验后才开始更新；建议 batch_size × 20"),
    "train_frequency": ("更新频率", "每采集 N 步做一次反向传播；通常保持 4"),
    "target_update_interval": ("目标网更新步", "每隔多少全局步同步目标网络；建议 1000~3000"),
    "epsilon_start": ("初始探索率", "ε-greedy 初始值；1.0 = 完全随机探索"),
    "epsilon_end": ("最低探索率", "训练末期保留的最小随机性；推荐 0.01~0.05"),
    "epsilon_decay_steps": ("探索衰减步数", "ε 从初始值线性降到最低所需的全局步数（不是局数！）"),
    "moving_avg_window": ("平均窗口", "计算平均奖励使用的最近 N 局窗口大小"),
    "eval_episodes": ("评估局数", "训练完成后自动评估的局数；0 = 不评估"),
    "checkpoint_interval": ("检查点频率", "每隔多少局保存一次模型并增量写入 CSV；建议 200~1000"),
    "log_interval": ("日志打印频率", "每隔多少局在终端打印一行进度"),
    "tensorboard_log_interval": ("TensorBoard 频率", "每隔多少局写一次 TFEvents；5 即可"),
    "jsonl_flush_interval": ("JSONL 刷新频率", "每隔多少局强制 flush 一次详细日志"),
    "tensorboard": ("TensorBoard", "写 TFEvents 供实时监控；强烈建议保持开启"),
    "save_csv": ("保存 CSV", "训练完成后存 episodes.csv 供数据分析"),
    "save_jsonl": ("保存 JSONL", "实时写每局详细日志，中断也不丢数据"),
    "live_plot": ("实时曲线", "弹 Matplotlib 窗口；服务器/无头环境必须关闭"),
    "lightweight_step_info": ("轻量统计", "true = 只在局结束时收集统计（更快）"),
    "env.board_size": ("地图尺寸", "格子数（宽=高）；越大越难；入门推荐 10~14"),
    "env.difficulty": ("难度", "easy/normal/hard，影响蛇的初始长度"),
    "env.mode": ("边界模式", "classic = 碰墙即死；wrap = 穿越到对侧"),
    "env.max_steps_without_food": ("无食超时步", "连续多少步没吃食物则超时；建议 board_size²"),
    "env.enable_bonus_food": ("奖励食物", "短暂出现的高奖励食物；入门建议关闭"),
    "env.enable_obstacles": ("障碍物", "随机障碍物；入门建议关闭"),
    "env.allow_leveling": ("升级系统", "进阶功能；入门建议关闭"),
    "env.seed": ("随机种子", "用于复现结果；留空 = 每次随机"),
    "rw.alive": ("存活奖励", "每步存活的奖励（负值=惩罚）；推荐 -0.01"),
    "rw.food": ("吃食物", "吃到普通食物的奖励；作为基准设 1.0"),
    "rw.bonusFood": ("奖励食物", "吃到奖励食物的奖励"),
    "rw.death": ("死亡惩罚", "碰墙/自咬的惩罚（负值）；推荐 -1.5"),
    "rw.timeout": ("超时惩罚", "无食超时的惩罚（负值）；推荐 -1.0"),
    "rw.levelUp": ("升级奖励", "升级事件奖励"),
    "rw.victory": ("填满奖励", "填满整张地图的大奖励"),
    "rw.foodDistanceK": ("靠近食物系数", "每步靠近食物给正奖励；0~0.5；过大会抖动"),
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
                            choices=["small_cnn", "adaptive_cnn", "hybrid"],
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
