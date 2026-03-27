# Custom Train Config Handbook

`custom` 模式读取 `custom_train_config.json`，这是本项目最推荐的训练入口。配置文件带有 JSON Schema 约束（`$schema` 字段），VS Code 等编辑器可即时校验。

关联文件：

- `custom_train_config.json`
- `custom_train_config.schema.json`
- `snake_rl/config.py`
- `snake_rl/schemes.py`
- `snake_rl/train.py`

## How To Tune（最小调参流程）

1. 先用默认配置完整跑一轮，确认管道通畅。
2. 每次只改 1~2 个参数，观察曲线变化。
3. 每轮改 `run_name`，便于 TensorBoard 横向对比。
4. 用 `snake-rl estimate` 提前估算耗时，避免长训后才发现配置不合理。

## High-Impact Fields

| 字段 | 作用 | 推荐起点 |
| --- | --- | --- |
| `run_name` | 输出目录名 | 每次实验唯一，如 `exp01_lr1e4` |
| `episodes` | 训练总局数 | `30000` |
| `model_type` | 网络结构 | 新手用 `adaptive_cnn`（默认）；极速迭代用 `tiny`；最强泛化用 `hybrid` |
| `learning_rate` | 学习率 | `1e-4` |
| `batch_size` | 每次更新采样数 | `128` |
| `epsilon_start/end/decay_steps` | 探索策略 | `1.0 / 0.03 / 200000` |
| `replay_capacity` | 回放池容量 | `50000` 起 |
| `target_update_interval` | 目标网络同步步数 | `2000` |
| `reward_weights.foodDistanceK` | 靠近食物 shaping 强度 | `0.4` |

## Model Fields

### `model_type`

| 值 | 输入 | 支持可变地图 | 适用场景 |
| --- | --- | --- | --- |
| `tiny` | 10 维标量（射线距离 + 食物方向 + 蛇长） | 是 | 参数 ~5K，秒级训练验证，适合教学和快速原型 |
| `small_cnn` | 固定 `[H, W, 9]` 图像 | 否 | 固定地图入门，不支持 curriculum / random_board |
| `adaptive_cnn` | 可变 `[H, W, 9]`，全局平均池化 | 是 | **多数场景通用推荐**（默认） |
| `hybrid` | 局部 patch CNN + 10 维全局特征融合 | 是 | 跨尺寸泛化最优，计算略重 |

> **如何选**：不确定就选 `adaptive_cnn`；只想跑通看看效果选 `tiny`；要求最好的跨地图泛化选 `hybrid`。

### `local_patch_size`

仅 `hybrid` 有效。必须为正奇数（如 `7`、`9`、`11`）。越大感受野越大，但计算量也上升。建议从 `7` 开始。

## Exploration Fields

```json
"epsilon_start": 1.0,
"epsilon_end": 0.03,
"epsilon_decay_steps": 200000
```

经验规则：

- 收敛太慢 → 适度减小 `epsilon_decay_steps`（减少探索阶段长度）
- 后期表现不稳定 → 适度增大 `epsilon_decay_steps`（延长充分探索期）
- `epsilon_end` 一般不低于 `0.01`，保留少量随机性

## Replay + Update Fields

```json
"replay_capacity": 50000,
"min_replay_size": 3000,
"batch_size": 128,
"train_frequency": 4,
"target_update_interval": 2000
```

经验规则：

- 显存/内存紧张 → 先降 `replay_capacity`，再降 `batch_size`
- 训练抖动大 → 尝试降 `learning_rate` 或增大 `target_update_interval`
- `min_replay_size` 应小于 `replay_capacity`，建议为 `capacity` 的 5%~10%

## Env Fields

```json
"env": {
  "board_size": 14,
  "difficulty": "normal",
  "mode": "classic",
  "max_steps_without_food": 196,
  "enable_bonus_food": false,
  "enable_obstacles": false,
  "allow_leveling": false,
  "seed": 42
}
```

建议：

- 入门：`board_size=10~14`，先关闭 `enable_bonus_food`、`enable_obstacles`、`allow_leveling`
- `max_steps_without_food` 常设为 `board_size²`（如 `14×14=196`）
- `mode: "wrap"` 启用边界环绕（无墙），适合泛化训练
- `seed` 控制初始局面随机性，设为 `null` 则每局随机

## Reward Fields

默认奖励权重：

```json
"reward_weights": {
  "alive": -0.01,
  "food": 1.0,
  "bonusFood": 1.5,
  "death": -1.5,
  "timeout": -1.0,
  "levelUp": 0.2,
  "victory": 5.0,
  "foodDistanceK": 0.4
}
```

| 字段 | 含义 | 调优方向 |
| --- | --- | --- |
| `alive` | 每步存活惩罚（负值） | 绝对值越大越催促蛇快速找食物 |
| `food` | 吃到普通食物奖励 | 核心信号，轻易别改 |
| `bonusFood` | 吃到奖励食物 | 仅 `enable_bonus_food=true` 时生效 |
| `death` | 死亡惩罚 | 负值，绝对值越大越害怕死亡 |
| `timeout` | 超时惩罚 | 负值 |
| `levelUp` | 升级奖励 | 仅 `allow_leveling=true` 时生效 |
| `victory` | 铺满棋盘胜利奖励 | 一般不需要修改 |
| `foodDistanceK` | 靠近食物 shaping 系数 | 前期加速收敛，过大会干扰策略 |

优先调这 3 个：`death`、`alive`、`foodDistanceK`。

## Curriculum / Random Board

`curriculum` 与 `random_board` 互斥，开启任意一个时 `model_type` 必须为支持可变地图的类型：`adaptive_cnn`、`hybrid` 或 `tiny`。

### `curriculum`

分阶段放大地图，每阶段独立设置 `episodes`、`epsilon_*`、`replay_capacity`。适合从小图稳定收敛后再迁移到大图的场景。

```json
"curriculum": {
  "stages": [
    {"board_size": 8, "episodes": 5000, "epsilon_start": 1.0, "epsilon_end": 0.1, "epsilon_decay_steps": 30000, "replay_capacity": 20000},
    {"board_size": 12, "episodes": 10000, "epsilon_start": 0.3, "epsilon_end": 0.03, "epsilon_decay_steps": 80000, "replay_capacity": 40000},
    {"board_size": 16, "episodes": 15000, "epsilon_start": 0.1, "epsilon_end": 0.03, "epsilon_decay_steps": 100000, "replay_capacity": 50000}
  ]
}
```

> 课程学习目前不支持 `--resume-state`，中断后请用 `--warm-start` 继续。

### `random_board`

每局从给定尺寸列表中随机采样，可通过 `weights` 控制各尺寸出现概率：

```json
"random_board": {
  "sizes": [8, 10, 12, 14, 16],
  "weights": [1, 2, 3, 3, 1]
}
```

## Parallel Section

```json
"parallel": {
  "enabled": false,
  "num_workers": 4,
  "queue_capacity": 8192,
  "weight_sync_interval_steps": 512,
  "actor_loop_sleep_ms": 0,
  "actor_seed_stride": 100000,
  "actor_device": "cpu"
}
```

建议：先串行验证曲线正常，再开启并行。worker 数不超过 CPU 核心数的 75%。

## Logging / Checkpoint Fields

| 字段 | 作用 | 默认值 |
| --- | --- | --- |
| `log_interval` | 每隔多少局打印/记录一次日志 | `10` |
| `checkpoint_interval` | 每隔多少局保存一次周期检查点 | `500` |
| `tensorboard_log_interval` | 每隔多少局写入 TensorBoard | `5` |
| `jsonl_flush_interval` | 每隔多少局刷新 JSONL 文件 | `20` |
| `moving_avg_window` | 计算最佳平均奖励的滑动窗口 | `100` |
| `eval_episodes` | 评估时运行的局数 | `20` |
| `live_plot` | 是否开启 Matplotlib 实时曲线 | `false` |
| `tensorboard` | 是否写入 TensorBoard | `true` |

## Validation

配置会经过 JSON Schema + Python 侧双重校验。建议先做快速估算再启动正式训练：

```bash
uv run snake-rl estimate --scheme custom --custom-config custom_train_config.json
```

## Common Failures

| 报错 | 原因 | 解决 |
| --- | --- | --- |
| `可变尺寸不支持` | `small_cnn` + `curriculum` 或 `random_board` | 改用 `adaptive_cnn`、`hybrid` 或 `tiny` |
| `local_patch_size 非法` | `hybrid` 时 `local_patch_size` 不是正奇数 | 改为 `7`、`9`、`11` 等正奇数 |
| `curriculum 和 random_board 互斥` | 同时设置了两者 | 只保留一个，另一个设为 `null` |
| 曲线不涨 | 探索不足或奖励信号太弱 | 检查 `epsilon_decay_steps`、`learning_rate`、`foodDistanceK` |
| 训练结束后 `best.pt` 未更新 | `moving_avg_window` 局数内奖励未超过历史最优 | 适当减小 `moving_avg_window` 或延长训练 |
