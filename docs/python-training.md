# Python 版贪吃蛇 DDQN 训练说明

本文说明如何使用 `snake_rl` 目录中的纯 Python 环境与 PyTorch 训练器。依赖与入口以 **`pyproject.toml` + `uv`** 为准（与 README 一致）。

## 1. 安装依赖

### 推荐：uv（与仓库默认一致）

需先安装 [uv](https://docs.astral.sh/uv/)，在项目根目录执行：

```bash
uv sync
```

可选：一并安装开发依赖（如 `pytest`）：

```bash
uv sync --extra dev
```

之后所有命令建议用 **`uv run …`** 前缀，这样无需手动激活虚拟环境，也能保证用到当前项目锁定的解释器与依赖。

### 备选：venv + pip

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
# source .venv/bin/activate

pip install -e .
pip install -e ".[dev]"   # 可选：测试依赖
```

正式依赖列表以 `pyproject.toml` 为准。

## 2. 快速训练

按方案训练（与 GUI 一致）：

```bash
uv run snake-rl train
uv run snake-rl train --scheme scheme1
```

逐项指定超参（不经过 schemes）：

```bash
uv run python -m snake_rl.train --episodes 300 --board-size 22 --difficulty normal --mode classic
```

若已通过 `pip install -e .` / `uv sync` 把可执行脚本装进了当前激活环境，也可直接写 `snake-rl train`（等价于 `uv run snake-rl train`）。

`python -m snake_rl.train`（建议配合 `uv run`）默认会使用简化训练配置（更容易先学会）：
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
- `--resume-state runs/<run_name>/state/training.pt`：完整恢复训练状态（回放、优化器、步数等）
- `--warm-start runs/<run_name>/checkpoints/latest.pt`：只加载网络权重，清空回放；课程学习不支持 `--resume-state`

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

启动方式任选其一：

```bash
# 与项目统一 CLI 一致（默认 logdir=runs，端口可改）
uv run snake-rl monitor --logdir runs --port 6006
```

若当前环境已激活且能直接找到 `tensorboard` 可执行文件：

```bash
tensorboard --logdir runs
```

浏览器打开 `http://localhost:6006`（若改了 `--port`，请改用对应端口）。

## 4. 评估与自动演示

```bash
uv run snake-rl eval --checkpoint runs/<run_name>/checkpoints/best.pt --episodes 30
```

若想在终端逐步查看棋盘：

```bash
uv run snake-rl eval --checkpoint runs/<run_name>/checkpoints/best.pt --episodes 3 --render --render-sleep-ms 120
```

## 5. 输出目录说明

以 `runs/ddqn_YYYYMMDD_HHMMSS` 为例：

- `run_config.json` / `train_config.json`：训练配置快照（内容一致）
- `run_manifest.json`：产物 schema 版本等
- `state/training.pt`：完整训练状态（供 `--resume-state`）
- `checkpoints/latest.pt`：最近一次 checkpoint
- `checkpoints/best.pt`：当前最佳移动平均奖励模型
- `checkpoints/ep_XXXXX.pt`：定期回合快照
- `logs/episodes.jsonl`：逐回合结构化日志
- `logs/episodes.csv`：逐回合 CSV
- `logs/summary.json`：训练总结
- `logs/tensorboard/`：TensorBoard 事件文件

## 6. 与网页版规则一致性

规则映射文档见：

- [js-rule-mapping.md](js-rule-mapping.md)

核心环境实现见：

- `snake_rl/env.py`

其中动作空间、9 通道观测、奖励默认值、终止原因均对齐 `web/game.js`。
