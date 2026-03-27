# Python Training Guide

面向使用 `snake_rl` 训练栈的开发者。覆盖"可执行、可复现、可排错"的完整路径。

## Scope

| 命令 | 入口模块 |
| --- | --- |
| `snake-rl train` | `snake_rl/train.py` |
| `snake-rl eval` | `snake_rl/evaluate.py` |
| `snake-rl monitor` | `snake_rl/monitor_server.py` |
| `snake-rl estimate` | `snake_rl/estimate_time.py` |
| `snake-rl serve-model` | `snake_rl/inference_server.py` |

关联文件：`snake_rl/cli.py`、`snake_rl/schemes.py`、`snake_rl/config.py`、`snake_rl/agent.py`、`snake_rl/replay_buffer.py`、`snake_rl/training_state.py`

## Setup

```bash
# 安装依赖（与 uv.lock 锁文件对齐）
uv sync

# 包含开发依赖（pytest）
uv sync --extra dev
```

## Quick Start

### 1) 默认 custom 训练

```bash
uv run snake-rl train
```

等价于 `--scheme custom`，自动读取 `custom_train_config.json`。

### 2) 指定自定义配置文件

```bash
uv run snake-rl train --scheme custom --custom-config custom_train_config.json
```

### 3) 使用内置方案

```bash
uv run snake-rl train --scheme scheme1   # 课程学习（小图→大图）
uv run snake-rl train --scheme scheme2   # 每局随机地图
uv run snake-rl train --scheme scheme3   # hybrid + 随机地图
uv run snake-rl train --scheme scheme4   # 课程 + 随机 + hybrid（推荐长训）
```

### 4) 评估模型

```bash
uv run snake-rl eval --checkpoint runs/<run_name>/checkpoints/best.pt
uv run snake-rl eval --checkpoint runs/<run_name>/checkpoints/best.pt --episodes 30 --render
```

### 5) 启动 TensorBoard

```bash
uv run snake-rl monitor --runs-dir runs --port 6006
```

### 6) 估算训练耗时

```bash
# 完整基准测试
uv run snake-rl estimate

# 快速粗估（采样点更少）
uv run snake-rl estimate --quick

# 针对 custom 配置估算
uv run snake-rl estimate --scheme custom --custom-config custom_train_config.json
```

输出会告知瓶颈是 `env-bound`（优先考虑并行）还是 `compute-bound`（优先考虑 GPU/batch 调优）。

### 7) 启动推理服务

```bash
uv run snake-rl serve-model --port 8765 --checkpoint runs/<run_name>/checkpoints/best.pt
```

服务启动后，浏览器 `web/index.html` 中填入推理 URL 即可观看 AI 自动对局。

## Resume Strategy

### 完整恢复（同实验继续训练，推荐）

保留优化器状态、回放池内容、epsilon 进度、episode 计数：

```bash
uv run snake-rl train --resume-state runs/<run_name>/state/training.pt
```

### 热加载权重（迁移到新配置）

仅加载模型权重，其他状态从头初始化：

```bash
uv run snake-rl train --warm-start runs/<run_name>/checkpoints/latest.pt
```

> **注意**：课程学习（`scheme1` / `scheme4`）目前不支持 `--resume-state`，请用 `--warm-start`。

## Parallel Rollout

多进程 actor 并行收集经验，主进程负责学习更新：

```bash
uv run snake-rl train --parallel --parallel-workers 4 --parallel-sync-interval 512
```

也可在 `custom_train_config.json` 中配置 `parallel` 块：

```json
"parallel": {
  "enabled": true,
  "num_workers": 4,
  "weight_sync_interval_steps": 512,
  "actor_device": "cpu"
}
```

建议流程：

1. 先串行（`parallel.enabled = false`）跑通并确认曲线正常。
2. 用 `snake-rl estimate` 判断是否 `env-bound`。
3. 确认 `env-bound` 后再开启并行。

## Run Artifacts

每次训练输出在 `runs/<run_name>/`：

| 文件 | 用途 |
| --- | --- |
| `checkpoints/best.pt` | 最佳平均奖励时保存，评估/演示首选 |
| `checkpoints/latest.pt` | 最近周期检查点，快速调试或热启动 |
| `checkpoints/ep_XXXXX.pt` | 周期性存档 |
| `state/training.pt` | 完整恢复状态（优化器 + 回放池 + 进度） |
| `run_config.json` | 复现实验的最关键配置快照 |
| `train_config.json` | 完整训练配置归档 |
| `events.out.tfevents.*` | TensorBoard 事件文件 |
| `logs/episodes.csv` | 每局结构化 CSV 日志 |
| `logs/episodes.jsonl` | 详细 JSONL 日志（支持流式追加） |
| `logs/summary.json` | 运行摘要 |

## Model Types

| `model_type` | 输入 | 支持可变地图 | 适用场景 |
| --- | --- | --- | --- |
| `tiny` | 10 维标量特征（7 方向射线 + 食物方向 + 蛇长） | 是 | 快速实验、教学、资源极有限 |
| `small_cnn` | 固定尺寸 `[H, W, 9]` 图像 | 否（训练 / 推理尺寸必须一致） | 固定尺寸入门验证 |
| `adaptive_cnn` | 可变尺寸 `[H, W, 9]`，全局平均池化 | 是 | **多数场景推荐** |
| `hybrid` | 局部 patch + 10 维全局特征 | 是（`local_patch_size` 必须正奇数） | 跨尺寸泛化、长期训练 |

> 新手建议从 `adaptive_cnn`（默认）开始；想极速迭代可试 `tiny`；追求最强泛化选 `hybrid`。

## Troubleshooting

| 症状 | 原因 | 解决 |
| --- | --- | --- |
| `custom 配置文件不存在` | 路径错误或文件缺失 | 确认 `custom_train_config.json` 路径，或显式传 `--custom-config <path>` |
| 评估时报输入尺寸不匹配 | `small_cnn` 是固定尺寸 | 保证评估时 `board_size` 与训练一致，或换 `adaptive_cnn` |
| 训练速度慢 | 环境 / 计算瓶颈 | 先跑 `snake-rl estimate` 看是 `env-bound` 还是 `compute-bound` |
| 课程学习无法 `--resume-state` | 课程暂不支持完整恢复 | 改用 `--warm-start` |
| `hybrid` / `tiny` 加载报特征版本错误 | checkpoint 与代码版本不兼容 | 用当前代码重新训练 |
| 并行模式下曲线异常 | 并行引入噪声或同步问题 | 先串行验证，再逐步增加 worker 数 |
| 曲线长时间不涨 | 探索不足或奖励信号太弱 | 检查 `epsilon_decay_steps`、`learning_rate`、`foodDistanceK` |
| `tiny` 模型表现弱于 CNN | tiny 信息量有限（无视觉） | 正常现象，tiny 适合快速迭代而非极致表现 |

## Related Docs

- [custom-train-config.md](custom-train-config.md)：配置字段详细说明
- [browser-agent-api.md](browser-agent-api.md)：网页侧 RL API
- [js-rule-mapping.md](js-rule-mapping.md)：JS 与 Python 规则对齐
