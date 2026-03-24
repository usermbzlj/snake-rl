# 成熟版贪吃蛇

一个双轨并行的贪吃蛇项目：

- 前端部分可直接双击 `index.html` 开玩
- Python 部分内置完整的 Double DQN 训练栈
- 支持课程学习、随机地图、Hybrid 跨尺寸泛化等多种训练方案
- 内置一体化 GUI，可一站式完成训练、管理、演示

---

## 快速开始

### 直接游玩

双击 `index.html` 即可运行，无需安装依赖。

### 使用 GUI（推荐）

```bash
uv sync
uv run python gui.py
```

GUI 提供：训练控制、记录管理、一键演示、TensorBoard 集成、训练时间估算。

### 命令行训练

```bash
# 1. 安装依赖
uv sync

# 2. 启动训练（默认 scheme1，或通过 --scheme 指定）
uv run python train_with_config.py
uv run python train_with_config.py --scheme scheme4

# 3. 评估模型
uv run python eval_pytorch.py --checkpoint runs/<run_name>/checkpoints/best.pt

# 4. TensorBoard 监控
uv run python serve_training_monitor.py --port 6006

# 5. 估算训练耗时
uv run python estimate_training_time.py
```

---

## 四种训练方案

所有方案集中在 `train_config.py` 管理，可通过 GUI 选择、命令行 `--scheme` 参数或环境变量 `SNAKE_TRAIN_SCHEME` 切换。

| 方案 | 核心思路 | 推荐度 |
|---|---|---|
| `scheme1` | 课程学习 + 表现门槛晋升 | 最高（推荐首选） |
| `scheme2` | 每局随机地图大小 | 中 |
| `scheme3` | Hybrid：局部 patch + 全局特征 | 高 |
| `scheme4` | 课程 + 随机 + Hybrid + 表现门槛 | 很高 |

### 课程学习的晋升机制（scheme1 / scheme4）

不再是"跑满 N 局就升级"，而是基于表现门槛：

- 最近 100 局的平均食物数达到阈值才允许晋级到更大地图
- 同时保留最大局数上限，防止永远卡在某阶段
- 例如 scheme1：8×8 需平均吃到 3 个食物 → 10×10 需 2.5 个 → 14×14 需 2 个 → 20×20

---

## 奖励机制

### 基础奖励

| 事件 | 奖励 |
|---|---|
| 每步存活 | `-0.01` |
| 吃普通食物 | `+1.0` |
| 吃奖励食物 | `+1.5` |
| 升级 | `+0.2` |
| 死亡 | `-1.5` |
| 超时未吃食物 | `-1.0` |
| 填满地图通关 | `+5.0` |

### 距离奖励塑形

每步根据蛇头到食物的距离变化给予额外奖励：

```
reward += foodDistanceK * (old_dist - new_dist) / max_dist
```

- `foodDistanceK = 0.4`（默认值）
- `max_dist = 2 * (board_size - 1)`，按棋盘大小归一化
- 靠近食物得正奖励，远离食物得负奖励，完全对称
- 归一化保证不同地图尺寸下 shaping 强度一致
- wrap 模式下使用环形 Manhattan 距离

单步量级参考：8×8 约 ±0.029，14×14 约 ±0.015，20×20 约 ±0.011。

---

## 模型结构

| `model_type` | 结构 | 支持可变尺寸 | 说明 |
|---|---|---|---|
| `small_cnn` | 卷积 + Flatten + FC | 否 | 旧版固定尺寸模型 |
| `adaptive_cnn` | CNN + Global Average Pooling | 是 | 课程学习、随机地图 |
| `hybrid` | 局部 patch CNN + 全局手工特征 | 是 | 跨尺寸泛化最强 |

---

## 项目结构

```text
gui.py                  # 一体化训练管理 GUI
train_config.py         # 所有训练方案与参数
train_with_config.py    # 配置文件启动训练（支持 --scheme）
train_pytorch.py        # 命令行参数启动训练
eval_pytorch.py         # 评估已训练模型
estimate_training_time.py  # 训练耗时估算
serve_training_monitor.py  # TensorBoard 远程监控
serve_model_inference.py   # 模型推理 HTTP 服务
index.html / game.js       # 浏览器游戏
snake_rl/
  config.py             # 配置 dataclass
  env.py                # Python 环境（对齐 game.js 规则）
  model.py              # SmallSnakeCNN / AdaptiveCNN / HybridNet
  agent.py              # Double DQN Agent
  replay_buffer.py      # 经验回放池（支持扩容迁移）
  train.py              # 训练主循环（标准 / 课程学习）
  evaluate.py           # 评估逻辑
  viz.py                # matplotlib 可视化（可选）
```

### 训练产物

```text
runs/<run_name>/
  checkpoints/
    best.pt             # 最佳平均奖励时的模型
    latest.pt           # 最新检查点
    ep_XXXXX.pt         # 定期检查点
  logs/
    tensorboard/
    episodes.csv
    episodes.jsonl
    summary.json
  train_config.json
```

---

## 训练参数

| 参数 | 默认值 |
|---|---|
| `learning_rate` | `1e-4` |
| `weight_decay` | `1e-5` |
| `batch_size` | `128` |
| `replay_capacity` | `50000`（按阶段递增） |
| `target_update_interval` | `2000` |
| `grad_clip_norm` | `5.0` |
| `epsilon_end` | `0.03` |
| `checkpoint_interval` | `500` |

训练可视化默认使用 TensorBoard（不弹出本地窗口）。

---

## 远程观察训练

### 局域网

```bash
uv run python serve_training_monitor.py --port 6006
```

同一局域网内用 `http://<训练机IP>:6006` 访问。

### 外网（内网穿透）

支持 frp / ngrok / cloudflared / Tailscale 等工具，将本地 `6006` 端口映射到公网。

frp 示例：

```toml
# frpc.toml
serverAddr = "你的服务器IP"
serverPort = 7000
auth.token = "你的token"

[[proxies]]
name = "tensorboard"
type = "tcp"
localIP = "127.0.0.1"
localPort = 6006
remotePort = 6006
```

---

## 让模型接管游戏

### 通过 GUI

在 GUI 的训练记录列表中选择一个运行，点击"用此模型演示"即可自动启动推理服务并打开游戏。

### 通过命令行

```bash
# 启动推理服务
uv run python serve_model_inference.py --port 8765 --checkpoint "runs/你的运行目录/checkpoints/best.pt"
```

打开 `index.html`，在 AI 接管面板中连接 `http://127.0.0.1:8765`。

---

## 游戏

### 特性

- 多难度：休闲 / 标准 / 困难 / 大师
- 双模式：经典（撞墙结束）/ 穿墙（边界环绕）
- 成长系统：吃食物升级、速度提升、障碍生成
- 奖励食物：限时高分食物

### 键位

| 按键 | 动作 |
|---|---|
| `A` / `←` | 左转 |
| `D` / `→` | 右转 |
| `空格` / `P` | 暂停 / 继续 |
| `Enter` | 开始 |
| `R` | 重开 |

### 浏览器 Agent API

页面加载后暴露 `window.snakeAgentAPI`，完整接口说明见 `train.md`。
