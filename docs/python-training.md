# Python Training Guide

面向使用 `snake_rl` 训练栈的开发者。本文只覆盖“可执行、可复现、可排错”的主路径，风格与根目录 `README` 保持一致。

## Scope

- 训练入口：`snake-rl train`
- 评估入口：`snake-rl eval`
- 监控入口：`snake-rl monitor`
- 估算入口：`snake-rl estimate`
- 推理服务：`snake-rl serve-model`

关联文件：

- `snake_rl/cli.py`
- `snake_rl/train.py`
- `snake_rl/schemes.py`
- `snake_rl/monitor_server.py`
- `snake_rl/inference_server.py`

## Setup

推荐使用 `uv`（和仓库锁文件一致）：

```bash
uv sync
```

开发依赖：

```bash
uv sync --extra dev
```

## Quick Start

### 1) 默认 custom 训练

```bash
uv run snake-rl train
```

### 2) 指定 custom 配置文件

```bash
uv run snake-rl train --scheme custom --custom-config custom_train_config.json
```

### 3) 使用内置方案

```bash
uv run snake-rl train --scheme scheme1
uv run snake-rl train --scheme scheme4
```

### 4) 评估模型

```bash
uv run snake-rl eval --checkpoint runs/<run_name>/checkpoints/best.pt
```

### 5) 启动 TensorBoard

```bash
uv run snake-rl monitor --runs-dir runs --port 6006
```

### 6) 估算训练耗时

```bash
uv run snake-rl estimate
uv run snake-rl estimate --quick
```

## Resume Strategy

完整恢复训练状态（推荐用于同一实验继续训练）：

```bash
uv run snake-rl train --resume-state runs/<run_name>/state/training.pt
```

仅加载权重热启动（用于迁移到新配置）：

```bash
uv run snake-rl train --warm-start runs/<run_name>/checkpoints/latest.pt
```

注意：课程学习目前不支持 `--resume-state`，请用 `--warm-start`。

## Parallel Rollout

```bash
uv run snake-rl train --parallel --parallel-workers 4 --parallel-sync-interval 512
```

建议：

- 先串行跑通，再开并行。
- 先用 `snake-rl estimate` 判断是否 `env-bound`，再决定是否并行。

## Run Artifacts

每次训练输出在 `runs/<run_name>/`：

- `checkpoints/best.pt`：最佳平均奖励
- `checkpoints/latest.pt`：最近检查点
- `state/training.pt`：完整恢复状态
- `run_config.json`：最关键配置快照
- `events.out.tfevents.*`：TensorBoard 事件
- `logs/episodes.jsonl` / `logs/episodes.csv`：结构化日志

## Troubleshooting

- 报错 “custom 配置文件不存在”：确认 `custom_train_config.json` 路径，或显式传 `--custom-config`。
- 评估时报尺寸不匹配：`small_cnn` 需要与训练时同尺寸地图。
- 训练速度慢：先看 `snake-rl estimate` 输出的 `env-bound/compute-bound`。

## Related Docs

- [custom-train-config.md](custom-train-config.md)
- [browser-agent-api.md](browser-agent-api.md)
- [js-rule-mapping.md](js-rule-mapping.md)
