# JS-Python Rule Mapping

本文记录浏览器环境 `web/game.js` 与 Python 环境 `snake_rl/env.py` 的规则对齐关系。两侧保持语义一致是模型可直接用于浏览器推理的基础。

关联文件：

- `web/game.js`（`SnakeGame` 类，常量 `ACTIONS`、`OBSERVATION_CHANNELS`、`TERMINAL_REASONS`、`DEFAULT_REWARD_WEIGHTS`）
- `snake_rl/env.py`（`SnakeEnv`，模块级常量与 `SnakeEnvConfig`）
- `snake_rl/inference_server.py`（`browser_state_to_python_snapshot`，状态映射层）

## Action Mapping

两侧动作语义完全一致（相对转向，与当前蛇头朝向无关）：

| 值 | JS 常量 | Python 常量 | 语义 |
| --- | --- | --- | --- |
| `0` | `AGENT_ACTIONS.STRAIGHT` | `ACTIONS.STRAIGHT` | 直走 |
| `1` | `AGENT_ACTIONS.TURN_LEFT` | `ACTIONS.TURN_LEFT` | 左转 |
| `2` | `AGENT_ACTIONS.TURN_RIGHT` | `ACTIONS.TURN_RIGHT` | 右转 |

## Observation Mapping

两侧观测格式完全一致：

- `shape: [H, W, 9]`（HWC 布局）
- `dtype: float32`
- 通道值范围：`[0.0, 1.0]`

通道顺序（`AGENT_OBSERVATION_CHANNELS` = `OBSERVATION_CHANNELS`）：

| 索引 | 名称 | JS | Python | 含义 |
| --- | --- | --- | --- | --- |
| 0 | `snakeHead` | ✓ | ✓ | 蛇头格子为 1 |
| 1 | `snakeBody` | ✓ | ✓ | 蛇身格子为 1 |
| 2 | `food` | ✓ | ✓ | 食物格子为 1 |
| 3 | `bonusFood` | ✓ | ✓ | 奖励食物格子为 1（不活跃时全 0） |
| 4 | `obstacle` | ✓ | ✓ | 障碍物格子为 1（不启用时全 0） |
| 5 | `dirUp` | ✓ | ✓ | 当前方向为上时全图为 1 |
| 6 | `dirRight` | ✓ | ✓ | 当前方向为右时全图为 1 |
| 7 | `dirDown` | ✓ | ✓ | 当前方向为下时全图为 1 |
| 8 | `dirLeft` | ✓ | ✓ | 当前方向为左时全图为 1 |

## Reward Mapping

默认奖励权重两侧一致（`DEFAULT_REWARD_WEIGHTS`）：

| 字段 | 默认值 | 触发条件 |
| --- | --- | --- |
| `alive` | `-0.01` | 每步存活 |
| `food` | `+1.0` | 吃到普通食物 |
| `bonusFood` | `+1.5` | 吃到奖励食物（`enable_bonus_food=true`） |
| `death` | `-1.5` | 碰墙/自撞/障碍物致死 |
| `timeout` | `-1.0` | 超时终止 |
| `levelUp` | `+0.2` | 升级（`allow_leveling=true`） |
| `victory` | `+5.0` | 铺满棋盘 |
| `foodDistanceK` | `+0.4` | 距离塑形系数（见下） |

**距离塑形逻辑**（两侧一致）：仅在未吃到食物/奖励食物时，根据蛇头与食物的 Manhattan 距离变化计算 shaping reward：

```
shaping = foodDistanceK * (prev_dist - curr_dist)
```

靠近食物时为正，远离时为负；吃到食物那步不加 shaping（避免双重奖励）。

## Terminal Reason Mapping

终止原因编码两侧一致（`TERMINAL_REASONS`）：

| 值 | 触发条件 |
| --- | --- |
| `wall` | `classic` 模式下撞墙 |
| `self` | 撞到自身蛇身 |
| `obstacle` | 撞到障碍物（`enable_obstacles=true`） |
| `board_full` | 蛇铺满棋盘（胜利） |
| `timeout` | `steps_since_last_food >= max_steps_without_food` |
| `not_running` | 游戏未处于运行状态时调用 `step()` |

## Core Rule Alignment

| 规则 | JS | Python | 对齐状态 |
| --- | --- | --- | --- |
| 相对转向（3 动作） | ✓ | ✓ | ✓ 一致 |
| `classic` 模式：撞墙死亡 | ✓ | ✓ | ✓ 一致 |
| `wrap` 模式：边界环绕 | ✓ | ✓ | ✓ 一致 |
| 自撞检测（不成长时排除尾巴） | ✓ | ✓ | ✓ 一致（避免尾巴误判） |
| 超时计数（`steps_since_last_food`） | ✓ | ✓ | ✓ 一致 |
| 奖励食物生成策略（按难度） | ✓ | ✓ | ✓ 一致 |
| 障碍物生成策略（按难度） | ✓ | ✓ | ✓ 一致 |
| 升级逻辑 | ✓ | ✓ | ✓ 一致 |
| 随机种子控制 | ✓ | ✓ | ✓ 一致 |

## Inference Snapshot Mapping

`snake-rl serve-model` 通过 `browser_state_to_python_snapshot()` 将浏览器 `getState()` 快照转换为 Python `SnakeEnv.set_state()` 所需格式，保证推理时两侧状态完全一致：

```
浏览器 getState() → HTTP POST /v1/act → browser_state_to_python_snapshot() → SnakeEnv.set_state() → model.forward()
```

## Why This Matters

- **零漂移推理**：模型在 Python 训练，直接用于浏览器演示，不需要额外适配层。
- **统一调试**：两侧用相同的终止原因、奖励权重、通道顺序，日志可直接对比。
- **可扩展性**：新增机制（如新障碍类型）只需同步修改 `game.js` 和 `env.py` 的对应位置，`OBSERVATION_CHANNELS` 版本常量（`FEATURE_SCHEMA_VERSION`）防止 checkpoint 与代码版本错配。
