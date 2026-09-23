"""训练产物与观测契约的版本号（用于拒绝不兼容的旧 checkpoint）。"""

# 模型权重文件（best.pt / latest.pt）里写入的 schema
# v2: dueling / noisy 头与 Rainbow-lite 超参进入 checkpoint
MODEL_CHECKPOINT_SCHEMA_VERSION = 2

# hybrid / tiny 全局特征向量语义版本
FEATURE_SCHEMA_VERSION = 2

# state/training.pt 训练状态包版本
# v2: 课程进度 + PER 优先级
TRAINING_STATE_SCHEMA_VERSION = 2
