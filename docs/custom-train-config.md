# 自定义训练配置（custom 模式）参数手册

> 本文面向没有深度学习背景的入门用户，用大白话解释每个参数的意义和调参建议。
>
> **快速使用**：直接打开项目根目录的 `custom_train_config.json`，在 Cursor / VSCode 中悬停任意字段即可看到提示——本文档是更详细的参考版本。

---

## 目录

1. [先读这里：参数怎么调](#1-先读这里参数怎么调)
2. [基础训练参数](#2-基础训练参数)
3. [网络结构](#3-网络结构)
4. [学习超参数](#4-学习超参数)
5. [探索策略（ε-greedy）](#5-探索策略ε-greedy)
6. [经验回放](#6-经验回放)
7. [日志与输出](#7-日志与输出)
8. [游戏环境（env）](#8-游戏环境env)
9. [奖励权重（reward_weights）](#9-奖励权重reward_weights)
10. [课程学习（curriculum）](#10-课程学习curriculum)
11. [随机地图训练（random_board）](#11-随机地图训练random_board)
12. [并行采样（parallel）](#12-并行采样parallel)
13. [常见问题与调参食谱](#13-常见问题与调参食谱)

---

## 1. 先读这里：参数怎么调

RL 训练参数繁多，初学者容易陷入"乱调"的困境。以下是**最小化改动原则**：

1. **先跑通默认配置**（`custom_train_config.json` 原样）——确认程序正常、曲线有上升趋势。
2. **只改一个参数**，跑几百~几千局，观察效果变化。
3. **对比实验**：每次实验前改 `run_name`，结果会存到不同目录，TensorBoard 里可以对比。

> 如果什么都没收敛，先检查：奖励设计是否合理 → 探索率衰减是否太快 → 学习率是否过大。

---

## 2. 基础训练参数

### `episodes`（训练总局数）

```json
"episodes": 30000
```

训练跑多少局（一局 = 蛇从出生到死亡）。

| 地图尺寸 | 建议局数 |
|----------|----------|
| 8×8      | 5000 ~ 10000 |
| 10×10    | 10000 ~ 20000 |
| 14×14    | 20000 ~ 50000 |
| 20×20    | 50000 ~ 100000 |

> **注意**：使用 curriculum（课程学习）时，`episodes` 被各阶段的局数替代，此字段无效。

---

### `max_steps_per_episode`（每局最大步数）

```json
"max_steps_per_episode": 3000
```

单局步数硬上限（无论蛇有没有吃到食物）。主要防止"蛇长到很长但仍在绕圈"。

建议：`board_size² × 5` ~ `board_size² × 15`。

---

### `device`（计算设备）

```json
"device": "auto"
```

- `"auto"` → 有 NVIDIA GPU 时自动用 cuda，否则用 cpu。
- `"cpu"` → 强制用 CPU（对于小模型有时 CPU 反而更快）。
- `"cuda"` / `"cuda:0"` → 强制用指定 GPU。

---

### `run_name`（实验名称）

```json
"run_name": "custom"
```

输出目录为 `runs/<run_name>/`。
**强烈建议**每次实验给一个有意义的名字，方便在 TensorBoard 中对比，例如：

```json
"run_name": "board14_lr1e4_eps200k"
```

---

## 3. 网络结构

### `model_type`（神经网络类型）

```json
"model_type": "small_cnn"
```

三种选择：

| 类型 | 特点 | 适用场景 |
|------|------|----------|
| `small_cnn` | 最简单，固定尺寸 CNN | **入门首选**，单一地图训练 |
| `adaptive_cnn` | Global Avg Pool，支持可变尺寸 | curriculum / random_board |
| `hybrid` | CNN + 手工全局特征 | 跨尺寸泛化最强，但参数量最大 |

> 如果你只用单一地图训练，用 `small_cnn` 即可。

---

### `local_patch_size`（局部感受野）

```json
"local_patch_size": 11
```

仅在 `model_type = "hybrid"` 时有效，表示模型观察蛇头周围 11×11 格子的局部视野。
必须是奇数，通常 9 ~ 13 效果相当。

---

## 4. 学习超参数

### `learning_rate`（学习率）

```json
"learning_rate": 0.0001
```

**最重要的超参数之一。**

| 现象 | 可能原因 | 调整方向 |
|------|----------|----------|
| Loss 剧烈震荡、Q 值爆炸 | 学习率太大 | 降低 10 倍（1e-4 → 1e-5） |
| 训练很久没有明显提升 | 学习率太小 | 提高 3~10 倍 |
| 训练初期进展快但后期不稳定 | 学习率偏大 | 轻微下调 |

入门建议：先用 `1e-4`，出问题再调整。

---

### `gamma`（折扣因子）

```json
"gamma": 0.99
```

决定 AI 有多"重视未来奖励"。

- `0.99`：非常看重长远回报，适合贪吃蛇（目标是长期存活+吃食物）。
- `0.9`：更关注即时奖励（适合短视策略，不推荐）。

> 入门建议：不要修改这个值。

---

### `weight_decay`（L2 正则化）

```json
"weight_decay": 1e-05
```

防止网络过拟合的轻微约束，通常 `1e-5` 即可。设为 `0` 关闭。

---

### `grad_clip_norm`（梯度裁剪）

```json
"grad_clip_norm": 5.0
```

防止某次更新时梯度过大导致参数"飞掉"。
通常不需要修改；如果训练中 loss 突然暴增，可以尝试降低到 `1.0`。

---

### `target_update_interval`（目标网络更新频率）

```json
"target_update_interval": 2000
```

DQN 用两个网络：主网络（实时更新）和目标网络（延迟同步）。
这个值表示每隔多少全局步把主网络的权重复制给目标网络。

- 太小（如 100）：目标不稳定，训练容易震荡。
- 太大（如 10000）：目标过时，学习缓慢。
- 建议范围：`1000 ~ 3000`。

---

## 5. 探索策略（ε-greedy）

AI 用 **ε-greedy 策略** 平衡"探索新行为"和"利用已学知识"：
- 以概率 ε 随机行动（探索）
- 以概率 1-ε 按网络最优估计行动（利用）

```json
"epsilon_start": 1.0,
"epsilon_end": 0.03,
"epsilon_decay_steps": 200000
```

三个参数配合使用：

### `epsilon_start`

初始探索率，通常固定为 `1.0`（完全随机，尽量探索各种情况）。

### `epsilon_end`

最低探索率。`0.03` = 即使训练到最后，仍有 3% 的概率随机行动（防止完全贪婪陷入局部最优）。
通常设在 `0.01 ~ 0.05` 之间。

### `epsilon_decay_steps`

从 `epsilon_start` 线性衰减到 `epsilon_end` 需要的**全局步数**（不是局数！）。

粗略估算：
```
epsilon_decay_steps ≈ episodes × 平均每局步数 × 0.6
```

例如：预计训练 30000 局，平均每局 50 步 → 约 1,500,000 步 × 0.6 ≈ `900000`。

| 问题 | 可能原因 |
|------|----------|
| AI 一直随机行动不收敛 | `epsilon_decay_steps` 太大，ε 还没降下来 |
| 训练后期 AI 表现很差 | `epsilon_decay_steps` 太小，ε 过早降低，早期数据太少 |

---

## 6. 经验回放

DQN 将历史交互存入"经验回放池"，每次训练从中随机采样，打破样本相关性。

### `replay_capacity`（回放池容量）

```json
"replay_capacity": 50000
```

最多保存多少条历史记录（先进先出）。
容量越大，样本越多样，但内存占用也越大。

| 地图尺寸 | 建议容量 |
|----------|----------|
| ≤ 10×10  | 30000 ~ 50000 |
| 14×14    | 50000 ~ 100000 |
| ≥ 18×18  | 100000 ~ 200000 |

### `min_replay_size`（开始训练的最少经验数）

```json
"min_replay_size": 3000
```

收集这么多条经验后才开始更新网络。
太小 → 早期大量重复利用少量数据；太大 → 开训前等待时间太长。

建议：`batch_size × 20` ~ `batch_size × 50`，通常 `2000 ~ 5000`。

### `batch_size`（批大小）

```json
"batch_size": 128
```

每次更新从回放池中采样多少条记录。

- 较小（64）：更新更频繁，但方差大。
- 较大（256）：梯度更稳定，但每步计算量更大。
- 建议：`128` 是入门平衡点。

### `train_frequency`（每多少步更新一次）

```json
"train_frequency": 4
```

每采集 4 步数据，才做一次反向传播更新。
降低值（如 1）→ 更新更频繁，计算开销大；提高值（如 8）→ 更新更少，数据利用率低。
通常保持 `4`。

---

## 7. 日志与输出

### `checkpoint_interval`（检查点保存频率）

```json
"checkpoint_interval": 500
```

每隔多少局保存一次模型（`ep_XXXXX.pt`），同时增量写入 `episodes.csv`。
训练被中断时，可以从最近的检查点恢复。
建议：`200 ~ 1000`，不要太小（磁盘消耗）也不要太大（中断损失）。

### `tensorboard`、`save_csv`、`save_jsonl`

```json
"tensorboard": true,
"save_csv": true,
"save_jsonl": true
```

全部建议保持 `true`：
- `tensorboard`：写 TFEvents 文件，用 `snake-rl monitor` 实时观察曲线。
- `save_csv`：训练完成后用 Excel/Python 做数据分析。
- `save_jsonl`：详细日志，训练中断时已有数据不丢。

### `live_plot`

```json
"live_plot": false
```

是否弹出 Matplotlib 实时曲线窗口。桌面环境可以打开，服务器/无头环境必须关闭。

---

## 8. 游戏环境（env）

```json
"env": {
  "board_size": 14,
  "difficulty": "normal",
  "mode": "classic",
  "max_steps_without_food": 196,
  "enable_bonus_food": false,
  "enable_obstacles": false,
  "allow_leveling": false,
  "seed": 42
}
```

### `board_size`（地图尺寸）

地图格子数（宽 = 高）。越大越难收敛，建议：

- 入门：`10` 或 `12`
- 进阶：`14`（默认）
- 挑战：`18` 或 `20`

### `mode`

- `"classic"`：碰墙即死（标准规则）
- `"wrap"`：穿越到对侧，蛇不会因为碰墙死亡，更容易存活

### `max_steps_without_food`

连续多少步没吃到食物则超时（本局结束）。
建议 = `board_size²`，例如 14×14 设 `196`。

### `difficulty`

- `"easy"`：蛇的初始长度较短
- `"normal"`：标准
- `"hard"`：初始较长，更难存活

### `enable_bonus_food` / `enable_obstacles` / `allow_leveling`

进阶功能，入门阶段全部保持 `false`，简化学习目标。

### `seed`

随机种子，用于复现结果。设 `null` 则每次随机。

---

## 9. 奖励权重（reward_weights）

```json
"reward_weights": {
  "alive": -0.01,
  "food": 1.0,
  "bonusFood": 1.5,
  "death": -1.5,
  "timeout": -1.0,
  "levelUp": 0.2,
  "victory": 5.0,
  "foodDistanceK": 0.4
}
```

奖励设计直接决定 AI 学到什么行为。

| 字段 | 作用 | 调参建议 |
|------|------|----------|
| `alive` | 每步存活的小惩罚（负值） | `-0.001 ~ -0.02`；太大导致蛇倾向快速死亡 |
| `food` | 吃到食物的奖励 | 作为基准设 `1.0`，其他奖励以此为参照 |
| `death` | 死亡惩罚（负值） | `-1.0 ~ -2.0`；太小蛇不怕死 |
| `timeout` | 超时惩罚（负值） | `-0.5 ~ -1.0`；防止蛇无限绕圈 |
| `foodDistanceK` | 每步靠近食物的奖励系数 | `0 ~ 0.5`；帮助蛇学会追食物，但过大会抖动 |
| `victory` | 填满地图的大奖励 | 极少触发，适当大即可 |

**常见问题排查**：

- **蛇倾向快速死亡**：`death` 惩罚太小，或 `alive` 惩罚太大。
- **蛇一直绕圈不吃食物**：降低 `alive` 惩罚绝对值，或提高 `foodDistanceK`。
- **蛇太激进，乱冲死亡**：降低 `foodDistanceK`，提高 `death` 惩罚绝对值。

---

## 10. 课程学习（curriculum）

```json
"curriculum": null
```

**课程学习 = 先在小地图上训练，再迁移到大地图**，模拟人类"先学简单再学难"的方式。

要启用，将 `null` 替换为：

```json
"curriculum": {
  "carry_replay": false,
  "scale_timeout": true,
  "stages": [
    {
      "board_size": 8,
      "episodes": 5000,
      "epsilon_start": 1.0,
      "epsilon_end": 0.1,
      "epsilon_decay_steps": 50000,
      "promotion_threshold_foods": 3.0,
      "promotion_window": 100,
      "promotion_min_episodes": 500
    },
    {
      "board_size": 12,
      "episodes": 10000,
      "epsilon_start": 0.3,
      "epsilon_end": 0.05,
      "epsilon_decay_steps": 80000,
      "promotion_threshold_foods": 4.0,
      "promotion_window": 100,
      "promotion_min_episodes": 1000
    },
    {
      "board_size": 16,
      "episodes": 20000,
      "epsilon_start": 0.3,
      "epsilon_end": 0.03,
      "epsilon_decay_steps": 150000
    }
  ]
}
```

**注意**：启用 curriculum 时，`model_type` 必须改为 `"adaptive_cnn"` 或 `"hybrid"`，因为 `small_cnn` 不支持可变地图尺寸。

### 关键字段说明

| 字段 | 说明 |
|------|------|
| `carry_replay` | 是否把上一阶段的经验迁移过来（建议 `false`，干净重建） |
| `scale_timeout` | 自动让大地图有更多超时步数（建议 `true`） |
| `promotion_threshold_foods` | 晋升条件：最近 `promotion_window` 局平均吃到食物数 ≥ 此值 |
| `promotion_min_episodes` | 至少训练多少局后才开始检查晋升 |

---

## 11. 随机地图训练（random_board）

```json
"random_board": null
```

每局随机从给定尺寸列表中抽取地图，提升模型对不同尺寸的泛化能力。
**与 curriculum 互斥**（不能同时启用）。

示例：

```json
"random_board": {
  "board_sizes": [8, 10, 12, 14, 16],
  "weights": [1, 2, 3, 2, 1],
  "max_steps_scale": 1.0
}
```

`weights` 让中间尺寸（12）出现频率更高。

---

## 12. 并行采样（parallel）

```json
"parallel": {
  "enabled": false,
  "num_workers": 4,
  ...
}
```

多个 actor 进程同时采集经验，单个 learner 进程负责更新网络。
适合 CPU 核心较多（≥ 8 核）且环境步成为瓶颈的情况。

**入门建议**：先关闭（`enabled: false`），用串行训练熟悉整个流程。
运行 `snake-rl estimate --quick` 查看是"env-bound"还是"compute-bound"；
只有 env-bound 时开启并行才有明显收益。

| 字段 | 说明 |
|------|------|
| `num_workers` | actor 数量，建议 ≤ CPU 核心数 - 1 |
| `weight_sync_interval_steps` | 策略同步频率（步），建议 `256 ~ 1024` |
| `actor_device` | actor 设备，建议 `"cpu"`（只做前向推断） |

---

## 13. 常见问题与调参食谱

### Q：训练了很久，奖励曲线完全没有上升趋势

1. 检查 `min_replay_size` 是否远大于 `batch_size`（应 ≥ batch_size）。
2. `epsilon_decay_steps` 是否太大（ε 还没降下来，一直在随机行动）。
3. `learning_rate` 是否过小（尝试 `3e-4`）。
4. 奖励中 `food` 和 `death` 比例是否合理（`food:death` 建议约 `1 : 1.5`）。

---

### Q：训练初期收敛很快，但后期曲线开始下降

1. `epsilon_decay_steps` 太小，探索过早停止。
2. `learning_rate` 偏大，后期震荡。尝试训练后期用更小的 lr（或直接调小）。
3. `target_update_interval` 太小，目标不稳定。

---

### Q：蛇学会吃食物了，但时不时突然"自杀"

1. `death` 惩罚绝对值不够大，蛇对死亡不够敏感。
2. `replay_capacity` 太小，早期糟糕的经验被频繁采样。
3. 尝试适当增大 `grad_clip_norm`（如 `10.0`）检查是否梯度爆炸导致。

---

### Q：想做"只调一个参数"的对比实验

每次实验修改 `run_name`（例如 `"lr3e4_exp1"`），保持其他参数不变，
用 `snake-rl monitor` 在同一个 TensorBoard 界面对比所有曲线。

---

### 推荐的入门调参顺序

```
1. 跑通默认配置（30000 局，14×14）
2. 若收敛慢 → 检查 epsilon_decay_steps，适当调小
3. 若不稳定 → 适当降低 learning_rate
4. 若蛇不追食物 → 适当增大 foodDistanceK（试 0.5~0.8）
5. 想挑战更大地图 → 改用 curriculum，从小地图开始
```
