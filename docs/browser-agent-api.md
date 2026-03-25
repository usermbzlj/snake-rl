# Browser Agent API Guide

本文说明网页侧 `window.snakeAgentAPI` 的稳定用法，对齐当前 Python 推理服务与训练流程。

关联文件：

- `web/game.js`（`createAgentAPI()` 与 `SnakeGame` 类）
- `snake_rl/env.py`（Python 对等环境）
- `snake_rl/inference_server.py`（推理 HTTP 服务）
- `docs/js-rule-mapping.md`（JS/Python 规则对齐细节）

## Quick Start

```javascript
const env = window.snakeAgentAPI;

// 关闭渲染提升速度
env.setRenderEnabled(false);
env.setSeed(42);

// 重置环境
let t = env.reset({
  difficulty: "normal",
  mode: "classic",
  boardSize: 14,
  maxStepsWithoutFood: 196,
});

// 对局循环
while (!t.done) {
  const action = env.sampleAction();    // 随机动作
  t = env.step(action);
}

console.log("得分：", t.info.scoreAfter);
```

## Action Space

相对转向（离散 3 动作），与蛇当前方向无关：

| 值 | 常量 | 语义 |
| --- | --- | --- |
| `0` | `env.ACTIONS.STRAIGHT` | 直走 |
| `1` | `env.ACTIONS.TURN_LEFT` | 向左转 |
| `2` | `env.ACTIONS.TURN_RIGHT` | 向右转 |

## Observation Space

`env.getObservationSpace()` 返回：

```json
{
  "layout": "HWC",
  "dtype": "float32",
  "shape": [boardSize, boardSize, 9]
}
```

9 通道顺序（与 Python `OBSERVATION_CHANNELS` 完全对应）：

| 索引 | 通道名 | 含义 |
| --- | --- | --- |
| 0 | `snakeHead` | 蛇头位置 |
| 1 | `snakeBody` | 蛇身位置 |
| 2 | `food` | 食物位置 |
| 3 | `bonusFood` | 奖励食物位置 |
| 4 | `obstacle` | 障碍物位置 |
| 5 | `dirUp` | 当前方向是否向上（全图广播） |
| 6 | `dirRight` | 当前方向是否向右 |
| 7 | `dirDown` | 当前方向是否向下 |
| 8 | `dirLeft` | 当前方向是否向左 |

## Core APIs

### 环境控制

| 方法 | 说明 |
| --- | --- |
| `reset(options?)` | 重置并开始新局，返回 `{observation, done, info}` |
| `step(action, options?)` | 执行一步，返回 `{observation, reward, done, info}` |
| `stepBatch(actions, options?)` | 批量执行多步 |
| `configure(options)` | 运行时更新环境配置（不重置） |

### 状态与观测

| 方法 | 说明 |
| --- | --- |
| `getObservation()` | 返回 HWC Float32Array |
| `getObservationFlat()` | 返回展平的 Float32Array |
| `getActionSpace()` | 返回动作空间描述 |
| `getObservationSpace()` | 返回观测空间描述（含 `shape`） |
| `getState()` | 返回完整环境状态快照（可传给推理服务） |
| `setState(snapshot, options?)` | 从快照恢复状态 |
| `getEpisodeStats()` | 返回当前局统计（步数、得分、死亡原因等） |
| `getLastTransition()` | 返回上一步 `(obs, action, reward, next_obs, done)` |

### 随机性与奖励

| 方法 | 说明 |
| --- | --- |
| `setSeed(seed)` / `getSeed()` | 设置/读取随机种子 |
| `sampleAction()` | 均匀随机采样一个合法动作 |
| `setRewardWeights(weights)` | 运行时更新奖励权重 |
| `getRewardWeights()` | 读取当前奖励权重 |

### 渲染与事件

| 方法 | 说明 |
| --- | --- |
| `setRenderEnabled(enabled)` | 开/关画布渲染（关闭可大幅提速） |
| `subscribe(eventName, handler)` | 订阅游戏事件（如 `episodeDone`） |

## `step()` Return Contract

```javascript
const { observation, reward, done, info } = env.step(action);
```

`info` 对象包含：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `terminalReason` | string \| null | 终止原因：`wall` / `self` / `obstacle` / `timeout` / `board_full` / `not_running` |
| `foodsEaten` | number | 本局已吃食物数 |
| `scoreAfter` | number | 执行动作后的当前得分 |
| `stepCount` | number | 本局总步数 |

`options` 常用项：

| 选项 | 说明 |
| --- | --- |
| `repeat` | 重复执行同一动作的步数 |
| `returnAllTransitions` | 返回所有中间步的 transition 列表 |

## Python Inference Integration

### 启动推理服务

```bash
uv run snake-rl serve-model --port 8765 --checkpoint runs/<run_name>/checkpoints/best.pt
```

### HTTP 接口

| 路径 | 方法 | 说明 |
| --- | --- | --- |
| `/health` | GET | 健康检查 |
| `/v1/status` | GET | 服务状态与已加载模型信息 |
| `/v1/load` | POST | 加载指定 checkpoint（`{"checkpoint": "path/to/best.pt"}`） |
| `/v1/act` | POST | 传入状态快照，返回动作（`{"action": 0/1/2}`） |

### 集成工作流

```javascript
const state = env.getState();              // 1. 获取当前状态快照
const resp = await fetch("http://localhost:8765/v1/act", {
  method: "POST",
  body: JSON.stringify(state),
  headers: { "Content-Type": "application/json" }
});
const { action } = await resp.json();      // 2. 获取模型动作
const t = env.step(action);                // 3. 执行动作
```

或使用内置的 `window.remoteInferenceController`，它封装了上述流程并支持自动对局循环。

## Model Compatibility

| `model_type` | 约束 | 说明 |
| --- | --- | --- |
| `small_cnn` | 页面 `boardSize` 必须与训练时完全一致 | 固定尺寸，不可变图 |
| `adaptive_cnn` | 任意 `boardSize` | 全局平均池化，通用 |
| `hybrid` | 任意 `boardSize`，推理时额外传全局特征 | 局部 patch + 全局特征 |

## Stability Tips

- 训练/批量对局时优先关闭渲染：`setRenderEnabled(false)`
- 复现实验请固定 `setSeed(42)`
- 收集 `terminalReason` 分布做失败原因分析（`wall` 多 vs `timeout` 多意义不同）
- 可变尺寸场景优先使用 `adaptive_cnn` 或 `hybrid`
- 并发对局时注意 `snakeAgentAPI` 是单例，每次 `reset()` 覆盖上一局状态
