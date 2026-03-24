# Python 版贪吃蛇 DDQN 训练说明

本文说明如何使用 `snake_rl` 目录中的纯 Python 环境与 PyTorch 训练器。

## 1. 安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## 2. 快速训练

```bash
python train_pytorch.py --episodes 300 --board-size 22 --difficulty normal --mode classic
```

默认会使用简化训练配置（更容易先学会）：
- `enable_bonus_food = false`
- `enable_obstacles = false`
- `allow_leveling = false`
- `max_steps_without_food = 250`

### 常用参数

- `--episodes`：训练回合数
- `--max-steps-per-episode`：单回合最大步数
- `--learning-rate`：学习率
- `--batch-size`：批大小
- `--replay-capacity`：经验回放容量
- `--no-live-plot`：关闭 Matplotlib 实时曲线
- `--no-tensorboard`：关闭 TensorBoard 日志
- `--run-name xxx`：指定输出目录名
- `--resume runs/<run_name>/checkpoints/latest.pt`：从 checkpoint 继续训练

## 3. 训练过程可视化

### 3.1 Matplotlib 实时窗口

训练时默认自动弹出，包含：
- 每回合奖励与滑动平均奖励
- 每回合步数与吃到食物数
- loss 曲线
- epsilon 变化曲线

### 3.2 TensorBoard

训练默认会写入：
- `episode/reward`
- `episode/avg_reward`
- `episode/steps`
- `episode/foods`
- `episode/score`
- `train/loss`
- `train/q_mean`
- `terminal_reason/*`

启动命令：

```bash
tensorboard --logdir runs
```

浏览器打开 `http://localhost:6006`。

## 4. 评估与自动演示

```bash
python eval_pytorch.py --checkpoint runs/<run_name>/checkpoints/best.pt --episodes 30
```

若想在终端逐步查看棋盘：

```bash
python eval_pytorch.py --checkpoint runs/<run_name>/checkpoints/best.pt --episodes 3 --render --render-sleep-ms 120
```

## 5. 输出目录说明

以 `runs/ddqn_YYYYMMDD_HHMMSS` 为例：

- `train_config.json`：训练配置快照
- `checkpoints/latest.pt`：最近一次 checkpoint
- `checkpoints/best.pt`：当前最佳移动平均奖励模型
- `checkpoints/ep_XXXXX.pt`：定期回合快照
- `logs/episodes.jsonl`：逐回合结构化日志
- `logs/episodes.csv`：逐回合 CSV
- `logs/summary.json`：训练总结
- `logs/tensorboard/`：TensorBoard 事件文件

## 6. 与网页版规则一致性

规则映射文档见：
- `snake_rl/JS_RULE_MAPPING.md`

核心环境实现见：
- `snake_rl/env.py`

其中动作空间、9 通道观测、奖励默认值、终止原因均对齐 `game.js`。
