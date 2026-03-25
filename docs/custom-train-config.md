# Custom Train Config Handbook

`custom` 模式读取 `custom_train_config.json`，这是本项目最推荐的训练入口。

关联文件：

- `custom_train_config.json`
- `custom_train_config.schema.json`
- `snake_rl/config.py`
- `snake_rl/schemes.py`
- `snake_rl/train.py`

## How To Tune (Minimal Process)

1. 先用默认配置完整跑一轮。
2. 每次只改 1~2 个参数。
3. 每轮改 `run_name`，便于 TensorBoard 横向对比。

## High-Impact Fields

| 字段 | 作用 | 推荐起点 |
| --- | --- | --- |
| `episodes` | 训练总局数 | `30000` |
| `model_type` | 网络结构 | `adaptive_cnn` 或 `hybrid` |
| `learning_rate` | 学习率 | `1e-4` |
| `batch_size` | 每次更新采样数 | `128` |
| `epsilon_start/end/decay_steps` | 探索策略 | `1.0 / 0.03 / 200000` |
| `replay_capacity` | 回放池容量 | `50000` 起 |
| `target_update_interval` | 目标网络同步步数 | `2000` |
| `run_name` | 输出目录名 | 每次实验唯一 |

## Model Fields

### `model_type`

- `small_cnn`：固定尺寸输入，不适合 curriculum/random_board。
- `adaptive_cnn`：支持可变尺寸，通用推荐。
- `hybrid`：可变尺寸 + 更强泛化，计算略重。

### `local_patch_size`

仅 `hybrid` 有效，必须正奇数（如 `11`）。

## Exploration Fields

```json
"epsilon_start": 1.0,
"epsilon_end": 0.03,
"epsilon_decay_steps": 200000
```

经验规则：

- 收敛太慢：适度减小 `epsilon_decay_steps`。
- 后期质量差：适度增大 `epsilon_decay_steps`。

## Replay + Update Fields

```json
"replay_capacity": 50000,
"min_replay_size": 3000,
"batch_size": 128,
"train_frequency": 4,
"target_update_interval": 2000
```

经验规则：

- 显存/内存紧张：先降 `replay_capacity`，再降 `batch_size`。
- 训练抖动大：尝试降 `learning_rate` 或增 `target_update_interval`。

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

- 入门：`board_size=10~14`，复杂机制先关。
- `max_steps_without_food` 常设为 `board_size^2`。

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

优先调这 3 个：

- `death`：死亡惩罚强度
- `alive`：存活步惩罚强度
- `foodDistanceK`：靠近食物的 shaping 强度

## Curriculum / Random Board

- `curriculum` 与 `random_board` 互斥。
- 开启任意一个时，`model_type` 必须为 `adaptive_cnn` 或 `hybrid`。

### `curriculum`

用于分阶段放大地图，按阶段设置 `episodes`、`epsilon_*`、`replay_capacity`。

### `random_board`

用于每局随机尺寸训练，可通过 `weights` 控制采样分布。

## Parallel Section

```json
"parallel": {
  "enabled": false,
  "num_workers": 4,
  "weight_sync_interval_steps": 512,
  "actor_device": "cpu"
}
```

建议：先串行验证稳定，再开启并行。

## Validation

- 配置会经过 schema + Python 侧校验。
- 推荐先跑：

```bash
uv run snake-rl estimate --scheme custom --custom-config custom_train_config.json
```

再启动正式训练。

## Common Failures

- 报“可变尺寸不支持”：检查是否 `small_cnn + curriculum/random_board` 组合。
- 报“local_patch_size 非法”：改成正奇数。
- 曲线不涨：优先检查 `epsilon_decay_steps`、`learning_rate`、奖励权重。
