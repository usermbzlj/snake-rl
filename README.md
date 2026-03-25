# 成熟版贪吃蛇

一个双轨并行的贪吃蛇项目：

- 前端静态页在 `web/`，可直接双击 `web/index.html` 开玩
- Python 部分内置完整的 Double DQN 训练栈
- 支持课程学习、随机地图、Hybrid 跨尺寸泛化等多种训练方案
- 内置 Web 控制台（FastAPI + Vue），可一站式完成训练、管理、演示

---

## 快速开始

### 直接游玩

双击 `web/index.html` 即可运行，无需安装依赖。

### 使用 Web 控制台（推荐）

```bash
uv sync
uv run snake-webui
```

默认在 `http://127.0.0.1:7860/` 打开浏览器。可选参数：`--port 8080`、`--no-open`（不自动弹出浏览器）。

控制台提供四个页签：**训练中心**（方案与并行选项、custom 表单化参数编辑与说明、进度条与 WebSocket 实时日志）、**运行记录**（筛选、删除、演示、TensorBoard、打开目录）、**服务管理**（TensorBoard / 推理端口与启停）、**参数手册**（内嵌 `docs/custom-train-config.html`）。游戏演示页在同源下的 `/play/`。

### 命令行（统一入口 `snake-rl`）

```bash
# 1. 安装依赖
uv sync

# 2. 按方案训练（默认 custom，编辑 custom_train_config.json 后直接运行）
uv run snake-rl train
# 指定其他内置方案：
uv run snake-rl train --scheme scheme4
# custom 模式指定自定义配置文件：
uv run snake-rl train --scheme custom --custom-config my_config.json

# 完整恢复训练状态（含回放、优化器等）：指向某次 run 的 state/training.pt
# 仅从 .pt 加载权重、清空回放：--warm-start runs/.../checkpoints/latest.pt
# 注意：课程学习暂不支持 --resume-state，需用 warm-start 或新开 run

# 3. 评估模型（默认继承同 run 的 run_config.json 环境与奖励）
uv run snake-rl eval --checkpoint runs/<run_name>/checkpoints/best.pt

# 4. TensorBoard（--logdir runs，聚合各次 run 根目录下的 tfevents）
uv run snake-rl monitor --port 6006

# 5. 估算训练耗时
uv run snake-rl estimate

# 6. 推理服务（供浏览器 AI 面板）
uv run snake-rl serve-model --port 8765 --checkpoint runs/<run_name>/checkpoints/best.pt
```

底层若需逐项覆盖训练超参，可使用：`uv run python -m snake_rl.train --help`。

---

## 五种训练方案

所有方案集中在 `snake_rl/schemes.py`，可通过 Web 控制台、`snake-rl train --scheme` 或环境变量 `SNAKE_TRAIN_SCHEME` 切换。

| 方案 | 核心思路 | 推荐度 |
|---|---|---|
| `custom` | **全参数自定义**，从 JSON 文件加载完整 TrainConfig | **★★★★★ 主推** |
| `scheme1` | 课程学习 + 表现门槛晋升 | ★★★★ |
| `scheme2` | 每局随机地图大小 | ★★★ |
| `scheme3` | Hybrid：局部 patch + 全局特征 | ★★★★ |
| `scheme4` | 课程 + 随机 + Hybrid + 表现门槛 | ★★★★ |

### 课程学习的晋升机制（scheme1 / scheme4）

不再是"跑满 N 局就升级"，而是基于表现门槛：

- 最近 100 局的平均食物数达到阈值才允许晋级到更大地图
- 同时保留最大局数上限，防止永远卡在某阶段
- 例如 scheme1：8×8 需平均吃到 3 个食物 → 10×10 需 2.5 个 → 14×14 需 2 个 → 20×20

### custom 模式使用方法（主推）

custom 模式允许完全自由地配置所有超参数，适合调参实验和个性化训练：

1. **默认配置**：仓库根目录的 `custom_train_config.json` 已纳入版本库，与内置基线一致，克隆后可直接 `uv run snake-rl train`；若本地删除了该文件，启动 Web 控制台时会按模板自动重建
2. **修改配置**：直接编辑 `custom_train_config.json`，或在 Web 控制台「训练中心」里用带说明的表单编辑后点击「验证并保存到文件」
3. **开始训练**：`uv run snake-rl train`（不带 --scheme 即默认使用 custom）
4. **指定其他配置文件**：`uv run snake-rl train --scheme custom --custom-config path/to/config.json`

配置文件结构与训练完成后生成的 `runs/<run_name>/run_config.json` 完全一致，可以直接将已有 run 的配置文件作为起点二次调整。

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
pyproject.toml / uv.lock   # 依赖与可执行入口（snake-rl / snake-webui）
web/                       # 浏览器游戏 + Web 控制台（index.html、app.html 等）
docs/                      # 说明文档（Python 训练、浏览器 API、JS↔Python 规则对照）
snake_rl/
  cli.py                   # 统一 CLI（snake-rl 命令）
  web_server.py            # Web 控制台（snake-webui：FastAPI + WebSocket）
  schemes.py               # 训练方案注册与 get_config
  run_context.py           # 从 run / checkpoint 解析环境与奖励元数据
  config.py                # 配置 dataclass
  env.py                   # Python 环境（对齐 web/game.js 规则）
  model.py                 # SmallSnakeCNN / AdaptiveCNN / HybridNet
  agent.py                 # Double DQN Agent
  replay_buffer.py         # 经验回放池（支持扩容迁移）
  train.py                 # 训练主循环（标准 / 课程 / 并行）
  evaluate.py              # 评估逻辑
  inference_server.py      # 模型推理 HTTP
  monitor_server.py        # 启动 TensorBoard（--logdir runs）
  run_meta.py              # run 目录元数据与状态（Web 列表与状态）
  estimate_time.py         # 训练耗时估算
  process_supervisor.py    # 子进程温和停止（Windows CTRL_BREAK 等）
  viz.py                   # matplotlib 可视化（可选）
tests/                     # pytest
```

### 延伸阅读

- [docs/python-training.md](docs/python-training.md)：Python / uv 安装与训练说明
- [docs/browser-agent-api.md](docs/browser-agent-api.md)：`window.snakeAgentAPI` 与推理对接
- [docs/js-rule-mapping.md](docs/js-rule-mapping.md)：网页与 `snake_rl/env.py` 规则对照

### 训练产物

```text
runs/<run_name>/
  events.out.tfevents.*   # TensorBoard 标量（默认写入 run 根目录）
  checkpoints/
    best.pt             # 最佳平均奖励时的模型
    latest.pt           # 最新检查点
    ep_XXXXX.pt         # 定期检查点
  state/
    training.pt         # 完整训练状态（resume-state 使用）
  logs/
    episodes.csv        # 与 checkpoint 同步增量追加
    episodes.jsonl
    summary.json
  run_config.json       # 与 train_config.json 内容一致（配置快照）
  train_config.json     # 兼容旧工具
  run_manifest.json     # 产物 schema 与创建时间等元数据
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

训练默认开启 TensorBoard 事件写入（`runs/<run_name>/` 下的 `tfevents` 文件）；`snake-rl monitor` 即启动 TensorBoard 指向整个 `runs/` 目录。

---

## 远程观察训练

### 局域网

```bash
uv run snake-rl monitor --port 6006
```

同一局域网内用 `http://<训练机IP>:6006/` 打开 TensorBoard，在左侧 run 列表中选择对应运行目录名。Web 控制台通过检测根路径 HTTP 200 判断服务就绪。

### 外网（内网穿透）

支持 frp / ngrok / cloudflared / Tailscale 等工具，将本地 `6006` 端口映射到公网。

frp 示例：

```toml
# frpc.toml
serverAddr = "你的服务器IP"
serverPort = 7000
auth.token = "你的token"

[[proxies]]
name = "snake-monitor"
type = "tcp"
localIP = "127.0.0.1"
localPort = 6006
remotePort = 6006
```

---

## 让模型接管游戏

### 通过 Web 控制台

在「运行记录」页选中一次运行，点击「用此模型演示」会在配置的推理端口启动推理服务并打开 `/play/` 游戏页。「在 TensorBoard 中查看」会启动或复用 TensorBoard 并在浏览器中打开（在界面左侧按运行目录名选择 run）。

### 通过命令行

```bash
uv run snake-rl serve-model --port 8765 --checkpoint "runs/你的运行目录/checkpoints/best.pt"
```

打开 `web/index.html`，在 AI 接管面板中连接 `http://127.0.0.1:8765`。若服务启动时已带 `--checkpoint` 预加载，页面会在加载时请求 `/v1/status` 并自动同步模型信息。

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

页面加载后暴露 `window.snakeAgentAPI`，完整接口说明见 [docs/browser-agent-api.md](docs/browser-agent-api.md)。
