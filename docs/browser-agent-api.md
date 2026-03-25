# Browser Agent API Guide

本文说明网页侧 `window.snakeAgentAPI` 的稳定用法，并对齐当前 Python 推理服务与训练流程。

关联文件：

- `web/game.js`
- `snake_rl/env.py`
- `snake_rl/inference_server.py`
- `docs/js-rule-mapping.md`

## Quick Start

```javascript
const env = window.snakeAgentAPI;
env.setRenderEnabled(false);
env.setSeed(42);

let t = env.reset({
  difficulty: "normal",
  mode: "classic",
  boardSize: 14,
  maxStepsWithoutFood: 196,
});

while (!t.done) {
  const action = env.sampleAction();
  t = env.step(action);
}
```

## Action Space

相对转向（离散 3 动作）：

- `env.ACTIONS.STRAIGHT = 0`
- `env.ACTIONS.TURN_LEFT = 1`
- `env.ACTIONS.TURN_RIGHT = 2`

## Observation Space

`env.getObservationSpace()`：

- `layout: HWC`
- `dtype: float32`
- `shape: [boardSize, boardSize, 9]`

9 通道顺序：

1. `snakeHead`
2. `snakeBody`
3. `food`
4. `bonusFood`
5. `obstacle`
6. `dirUp`
7. `dirRight`
8. `dirDown`
9. `dirLeft`

## Core APIs

环境控制：

- `reset(options?)`
- `step(action, options?)`
- `stepBatch(actions, options?)`
- `configure(options)`

状态与观测：

- `getObservation()` / `getObservationFlat()`
- `getActionSpace()` / `getObservationSpace()`
- `getState()` / `setState(snapshot, options?)`
- `getEpisodeStats()` / `getLastTransition()`

随机性与奖励：

- `setSeed(seed)` / `getSeed()`
- `sampleAction()`
- `setRewardWeights(weights)` / `getRewardWeights()`

渲染与事件：

- `setRenderEnabled(enabled)`
- `subscribe(eventName, handler)`

## `step()` Return Contract

`env.step(...)` 返回：

- `observation`
- `reward`
- `done`
- `info`（包含 `terminalReason`、`foodsEaten`、`scoreAfter` 等）

`options` 常用项：

- `repeat`
- `returnAllTransitions`

## Python Inference Integration

启动服务：

```bash
uv run snake-rl serve-model --port 8765 --checkpoint runs/<run_name>/checkpoints/best.pt
```

接口：

- `GET /health`
- `GET /v1/status`
- `POST /v1/load`
- `POST /v1/act`

工作流：

1. 浏览器 `getState()` 取状态快照。
2. 发给 Python `/v1/act`。
3. Python 返回 `action`（0/1/2）。
4. 浏览器调用 `step(action)` 执行。

## Model Compatibility

- `small_cnn`：固定尺寸，页面 `boardSize` 必须匹配训练尺寸。
- `adaptive_cnn`：支持可变尺寸。
- `hybrid`：支持可变尺寸，使用局部 patch + 全局特征。

## Stability Tips

- 训练时优先关闭渲染：`setRenderEnabled(false)`。
- 复现实验请固定 seed。
- 收集 `terminalReason` 分布做错误分析。
- 可变尺寸训练优先使用 `adaptive_cnn` 或 `hybrid`。
