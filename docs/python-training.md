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

### 主推：custom 模式（全参数自定义）

```bash
# 首次运行会自动生成 custom_train_config.json，直接修改后训练
uv run snake-rl train
# 或指定自定义配置文件
uv run snake-rl train --scheme custom --custom-config my_config.json
# 估算训练耗时（同样基于 custom_train_config.json）
uv run snake-rl estimate
```

`custom_train_config.json` 包含所有可调超参数，结构与训练完成后生成的 `run_config.json` 完全一致。常用字段：

| 字段 | 说明 | 示例值 |
|---|---|---|
| `episodes` | 总训练局数 | `30000` |
| `model_type` | 模型类型（`adaptive_cnn` / `hybrid`） | `"adaptive_cnn"` |
| `learning_rate` | 学习率 | `1e-4` |
| `batch_size` | 批大小 | `128` |
| `epsilon_start` / `epsilon_end` | ε-greedy 范围 | `1.0` / `0.03` |
| `curriculum` | 课程学习配置（可选） | `null` 或配置对象 |
| `random_board` | 随机地图配置（可选） | `null` 或配置对象 |

### 内置方案训练

```bash
uv run snake-rl train --scheme scheme1  # 课程学习
uv run snake-rl train --scheme scheme4  # 课程+随机+Hybrid
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
- `--no-live-plot`：关闭 Matplotlib 实时曲线（默认关闭弹窗）
- `--no-tensorboard`：关闭 TensorBoard 事件文件写入（默认开启）
- `--run-name xxx`：指定输出目录名
- `--resume-state runs/<run_name>/state/training.pt`：完整恢复训练状态（回放、优化器、步数等）
- `--warm-start runs/<run_name>/checkpoints/latest.pt`：只加载网络权重，清空回放；课程学习不支持 `--resume-state`

## 3. 训练过程可视化

### 3.1 TensorBoard（默认）

训练默认将标量写入各 run 根目录下的 `events.out.tfevents.*`，`--logdir runs` 即可在 TensorBoard 中对比多次运行。

典型标量：`episode/reward`、`episode/avg_reward`、`train/loss`、`train/epsilon` 等。

启动方式（与项目 CLI 一致）：

```bash
uv run snake-rl monitor --runs-dir runs --port 6006
```

浏览器打开 `http://localhost:6006/`（若改了 `--port`，请改用对应端口）。在左侧选择 run 名称（即 `runs/` 下子目录名）。

### 3.2 Matplotlib 实时窗口（可选）

需要本地弹窗时，在配置中打开 `live_plot`，或使用 `python -m snake_rl.train` 且**不要**加 `--no-live-plot`。默认关闭，避免无显示环境报错。

### 3.3 结构化日志

`logs/episodes.jsonl` 与 `logs/episodes.csv` 仍逐回合记录，便于脚本分析；CSV 会在每次 checkpoint 时增量追加，中断训练也可保留已写入部分。

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
- `events.out.tfevents.*`：TensorBoard 事件（run 根目录）
- `logs/episodes.jsonl`：逐回合结构化日志
- `logs/episodes.csv`：逐回合 CSV（checkpoint 时追加）
- `logs/summary.json`：训练总结

## 6. 与网页版规则一致性

规则映射文档见：

- [js-rule-mapping.md](js-rule-mapping.md)

核心环境实现见：

- `snake_rl/env.py`

其中动作空间、9 通道观测、奖励默认值、终止原因均对齐 `web/game.js`。
