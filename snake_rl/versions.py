"""训练产物与观测契约的版本号（用于拒绝不兼容的旧 checkpoint）。"""

# 模型权重文件（best.pt / latest.pt）里写入的 schema
MODEL_CHECKPOINT_SCHEMA_VERSION = 1

# hybrid 全局特征向量语义版本；与旧模型不兼容时升高此值
FEATURE_SCHEMA_VERSION = 2

# state/training.pt 训练状态包版本
TRAINING_STATE_SCHEMA_VERSION = 1
