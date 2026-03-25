# JS-Python Rule Mapping

本文记录浏览器环境 `web/game.js` 与 Python 环境 `snake_rl/env.py` 的规则对齐关系。

关联文件：

- `web/game.js`
- `snake_rl/env.py`
- `snake_rl/inference_server.py`

## Action Mapping

动作语义完全一致（相对转向）：

- `STRAIGHT = 0`
- `TURN_LEFT = 1`
- `TURN_RIGHT = 2`

## Observation Mapping

两侧观测格式一致：

- `shape: [H, W, 9]`
- `dtype: float32`
- `layout: HWC`

通道顺序一致：

1. `snakeHead`
2. `snakeBody`
3. `food`
4. `bonusFood`
5. `obstacle`
6. `dirUp`
7. `dirRight`
8. `dirDown`
9. `dirLeft`

## Reward Mapping

默认权重一致：

- `alive=-0.01`
- `food=+1.0`
- `bonusFood=+1.5`
- `death=-1.5`
- `timeout=-1.0`
- `levelUp=+0.2`
- `victory=+5.0`
- `foodDistanceK=+0.4`

距离塑形逻辑一致：只在未吃到食物/奖励食物时，按“距离变化”加减 shaping reward。

## Terminal Reason Mapping

终止原因编码一致：

- `wall`
- `obstacle`
- `self`
- `board_full`
- `timeout`
- `not_running`

## Core Rule Alignment

- `classic`：撞墙结束。
- `wrap`：边界环绕。
- 自撞检测：不成长时排除尾巴，避免误判。
- 超时终止：`steps_since_last_food >= max_steps_without_food`。
- 奖励食物、升级与障碍生成策略均按难度参数对齐。

## Why This Matters

- 浏览器演示与 Python 训练可直接复用策略，不会因为规则漂移导致表现错位。
- `snake-rl serve-model` 可直接把网页状态快照映射回 Python 环境做一致推理。
