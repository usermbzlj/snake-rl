# 贪吃蛇训练对接指南（Double DQN + 固定尺寸 / 可变尺寸 CNN）

本文面向训练代码接入者，目标是让你在浏览器环境中直接使用 `window.snakeAgentAPI` 跑强化学习训练。

说明：

- 本文主要描述 JS 环境接口本身
- Python 侧现在已经支持三类模型：
  - `small_cnn`：固定尺寸
  - `adaptive_cnn`：可变尺寸
  - `hybrid`：局部 patch + 全局特征
- 如果你只关心 Python 训练入口，请优先看 [README.md](../README.md) 与 `snake_rl/schemes.py`

---

## 1. 快速开始

1. 打开 `web/index.html`（可直接双击，或由 `snake-webui` 服务管理里的「打开游戏页」访问 `/play/`）
2. 在浏览器控制台获取接口对象：

```javascript
const env = window.snakeAgentAPI;
```

3. 建议训练前先关闭渲染并固定随机种子：

```javascript
env.setRenderEnabled(false);
env.setSeed(42);
```

4. 重置环境并开始 rollout：

```javascript
let transition = env.reset({
  difficulty: "normal",
  mode: "classic",
  boardSize: 22,
  maxStepsWithoutFood: 250,
});

while (!transition.done) {
  const action = env.sampleAction(); // 先随机动作，后续替换为Q网络策略
  transition = env.step(action);
}
```

---

## 2. 动作空间定义（离散 3 动作）

动作是**相对转向**，不是绝对方向：

- `env.ACTIONS.STRAIGHT` = `0`：保持当前朝向
- `env.ACTIONS.TURN_LEFT` = `1`：相对左转
- `env.ACTIONS.TURN_RIGHT` = `2`：相对右转

这和手动控制一致：`A` 左转、`D` 右转。

---

## 3. 观测空间定义（CNN 输入）

`env.getObservationSpace()` 返回：

- `layout`: `HWC`
- `shape`: `[boardSize, boardSize, 9]`
- `dtype`: `float32`

9 个通道顺序固定：

1. `snakeHead`
2. `snakeBody`
3. `food`
4. `bonusFood`
5. `obstacle`
6. `dirUp`
7. `dirRight`
8. `dirDown`
9. `dirLeft`

获取观测：

```javascript
const obs = env.getObservation();
// obs.data: Float32Array
// obs.shape: [H, W, C]
```

如需扁平向量（例如调试）：

```javascript
const flat = env.getObservationFlat();
```

---

## 4. step 返回结构（核心）

`env.step(action, options?)` 返回：

```javascript
{
  observation, // next_state
  reward,      // 标量奖励
  done,        // 是否终止
  info: {
    episode,
    step,
    action,
    ateFood,
    ateBonusFood,
    levelUp,
    scoreGain,
    scoreBefore,
    scoreAfter,
    lengthBefore,
    lengthAfter,
    levelBefore,
    levelAfter,
    foodsEaten,
    stepsSinceFoodBefore,
    stepsSinceFoodAfter,
    terminalReason,
    terminalReasonLabel
  }
}
```

`options` 支持：

- `repeat`：同一调用内连续执行 N 步（默认 1）
- `returnAllTransitions`：设为 `true` 时返回整个 transition 数组

可直接用于经验回放条目：

`(state, action, reward, next_state, done)`

---

## 5. reset / configure / seed

### 5.1 `reset(options?)`

常用可选项：

- `difficulty`: `easy | normal | hard | expert`
- `mode`: `classic | wrap`
- `boardSize`: `8~64`（建议固定，比如 22）
- `enableBonusFood`: `true/false`
- `enableObstacles`: `true/false`
- `allowLeveling`: `true/false`
- `maxStepsWithoutFood`: `0` 表示关闭；>0 可防止死循环
- `rewardWeights`: 奖励权重覆盖
- `seed`: 固定随机种子
- `renderEnabled`: 是否渲染

是否需要固定 `boardSize`，取决于你使用的模型：

- `small_cnn`：必须固定尺寸
- `adaptive_cnn`：可以变尺寸
- `hybrid`：可以变尺寸，局部 patch 大小固定即可

### 5.2 `configure(options)`

更新环境配置（不重置回合），通常配合 `reset()` 使用。

可用 `env.getSupportedConfigs()` 查看可选难度、模式和 boardSize 范围。

### 5.3 `setSeed(seed)` / `getSeed()`

用于可复现实验。设置后，食物/障碍随机过程使用同一伪随机序列。

---

## 6. 奖励设计

默认权重（`env.DEFAULT_REWARD_WEIGHTS`）：

- `alive`: `-0.01`
- `food`: `+1.0`
- `bonusFood`: `+1.5`
- `death`: `-1.0`
- `timeout`: `-0.6`
- `levelUp`: `+0.2`
- `victory`: `+2.0`

修改奖励：

```javascript
env.setRewardWeights({
  alive: -0.005,
  food: 1.2,
  death: -1.2
});
```

---

## 7. 训练循环模板（Double DQN）

```javascript
const env = window.snakeAgentAPI;
env.setRenderEnabled(false);
env.setSeed(123);

for (let episode = 0; episode < 10000; episode += 1) {
  let t = env.reset({
    boardSize: 22,
    difficulty: "normal",
    mode: "classic",
    maxStepsWithoutFood: 250
  });

  let state = t.observation;
  while (!t.done) {
    // 1) epsilon-greedy 选动作（0/1/2）
    const action = selectActionWithDoubleDQN(state);

    // 2) 与环境交互
    t = env.step(action);
    const nextState = t.observation;

    // 3) 存经验
    replayBuffer.push({
      state,
      action,
      reward: t.reward,
      nextState,
      done: t.done
    });

    // 4) 训练在线Q网络 + 定期同步target网络
    trainDoubleDQNIfReady(replayBuffer);

    state = nextState;
  }

  const stat = env.getEpisodeStats();
  console.log(
    `ep=${stat.episode}, steps=${stat.steps}, reward=${stat.totalReward.toFixed(2)}, score=${stat.scoreEnd}, reason=${stat.terminalReason}`
  );
}
```

---

## 8.1 可变尺寸训练说明

如果你在浏览器端自己实现训练器，需要注意：

- 原始观测是整图 `HWC`
- 若模型是固定尺寸 CNN，训练期间不要切换 `boardSize`
- 若模型是 `adaptive_cnn` 风格，可先把观测转为 `CHW`，再按最大尺寸做居中 padding
- 若模型是 `hybrid` 风格，可以：
  - 从整图观测中以蛇头为中心裁一个固定尺寸 patch
  - 另外拼接少量手工特征，例如食物相对方向、距离、长度、占图比例

这也是当前 Python 训练栈里 `scheme3` / `scheme4` 的实现思路。

---

## 9. 进阶接口（批量、状态恢复、事件）

### 9.1 批量步进

```javascript
const transitions = env.stepBatch([0, 1, 0, 2], { stopOnDone: true });
```

### 9.2 状态保存/恢复（断点训练、复现实验）

```javascript
const snapshot = env.getState();
// ...保存到文件/内存...
env.setState(snapshot, { agentControlled: true, render: false });
```

### 9.3 订阅事件

```javascript
const unsubscribe = env.subscribe("transition", ({ transition, episodeStats }) => {
  // 监控训练过程
});

// 不再需要时取消
unsubscribe();
```

事件名：

- `reset`
- `transition`
- `done`

---

## 10. 训练稳定性建议

- `small_cnn` 才需要固定输入尺寸：`boardSize` 不要在训练中切换
- `adaptive_cnn` / `hybrid` 可以训练可变尺寸地图
- 如果做课程学习，推荐从小地图逐步放大
- 如果做随机地图，推荐让中等尺寸出现频率更高
- 如果做 `hybrid`，局部 patch 大小保持固定，例如 `11x11`
- 先关复杂机制：早期可 `enableBonusFood=false`, `enableObstacles=false`
- 增加超时终止：`maxStepsWithoutFood` 推荐 `150~400`
- 先关闭渲染：`setRenderEnabled(false)` 可显著提速
- 记录终止原因：用 `info.terminalReason` 分析策略崩溃模式
- 需要复现时固定 seed：`setSeed(固定值)`

---

## 11. Python 侧推荐训练策略

当前项目里更推荐的训练顺序是：

1. `scheme1`：课程学习，先把策略练稳定
2. `scheme4`：课程学习 + 阶段内随机地图 + `hybrid`，进一步提升跨尺寸泛化

如果只是想快速验证接口是否正常：

- 固定 `boardSize=8` 或 `10`
- 关闭复杂机制
- 用随机动作先跑完整 rollout
- 再换成自己的 DQN / DDQN 策略

---

## 12. 训练观测与耗时估算

如果你使用的是项目内置 Python 训练栈，还可以配合统一 CLI：

### 12.1 远程观察训练

```bash
uv run snake-rl monitor --port 6006
```

用途：

- 在同一局域网内打开 TensorBoard（根路径 `/`），查看 `runs/` 下各次训练的标量曲线
- 或者把 `6006` 端口通过 frp / ngrok / cloudflared / Tailscale 映射到外网

### 12.2 快速估算当前配置耗时

```bash
uv run snake-rl estimate
```

会对当前方案做「环境+前向+写回放」与「反向更新」分离的微基准，再合成 steps/s，并输出保守 / 中等 / 偏长三档训练时长估算；可加 `--quick` 加快粗略估算。

---

## 13. 让训练好的 Python 模型接管网页游戏

如果你已经训练出了 `.pt` 模型，现在可以直接让它接管正常的浏览器游戏画面。

### 13.1 启动推理服务

```bash
uv run snake-rl serve-model --port 8765
```

也可以直接预加载模型：

```bash
uv run snake-rl serve-model --port 8765 --checkpoint "runs/你的运行目录/checkpoints/best.pt"
```

### 13.2 页面操作

打开 `web/index.html` 后，使用页面新增的 `AI 接管` 面板：

1. `推理服务` 填 `http://127.0.0.1:8765`
2. `模型路径` 填你的 `.pt` 路径
3. 点击 `加载模型`
4. 点击 `AI 接管`

你还可以：

- 调整 `每步延迟`，决定演示速度
- 勾选 `循环演示`，让它一局结束后自动重开

### 13.3 模型类型差异

- `small_cnn`：固定输入尺寸，网页地图必须与训练尺寸一致
- `adaptive_cnn`：支持可变地图
- `hybrid`：支持可变地图，且更适合跨尺寸泛化

### 13.4 工作原理

网页不会直接运行 `.pt`。

真实流程是：

1. 浏览器通过 `window.snakeAgentAPI.getState()` 读取当前游戏状态
2. 状态通过 HTTP 发给本地 `snake-rl serve-model`（`snake_rl/inference_server.py`）
3. Python 端还原环境状态，并按模型类型构造输入
4. 模型输出动作 `0/1/2`
5. 浏览器再调用 `window.snakeAgentAPI.step(action)` 执行动作

这样做的好处是：

- 不需要把 PyTorch 模型转成浏览器端格式
- `adaptive_cnn` / `hybrid` 都能直接复用已有训练代码
- 同一台训练机上的模型，可以通过局域网 IP 提供给别的设备页面调用

---

## 14. 全接口清单

- 常量
  - `API_VERSION`
  - `ACTIONS`
  - `TERMINAL_REASONS`
  - `DEFAULT_REWARD_WEIGHTS`
  - `DEFAULT_ENV_CONFIG`
  - `OBSERVATION_CHANNELS`
- 环境控制
  - `reset(options?)`
  - `step(action, options?)`
  - `stepBatch(actions, options?)`
  - `configure(options)`
  - `close()`
- 状态与观测
  - `getObservation()`
  - `getObservationFlat()`
  - `getObservationSpace()`
  - `getActionSpace()`
  - `getSupportedConfigs()`
  - `getMetadata()`
  - `getState()`
  - `setState(snapshot, options?)`
  - `getLastTransition()`
  - `getEpisodeStats()`
- 随机性与奖励
  - `setSeed(seed)`
  - `getSeed()`
  - `sampleAction()`
  - `setRewardWeights(weights)`
  - `getRewardWeights()`
- 渲染与交互
  - `setRenderEnabled(enabled)`
  - `renderFrame()`
  - `resumeHumanControl()`
  - `startHumanGame()`
  - `subscribe(eventName, handler)`

---

如果你后续要加 frame stacking（如 4 帧堆叠）或改成 N-step return，也可以在这个接口层继续扩，不需要改游戏核心渲染逻辑。
