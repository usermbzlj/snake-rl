from __future__ import annotations

import torch
from torch import nn


class SmallSnakeCNN(nn.Module):
    """原始固定尺寸 CNN（Flatten + FC），仅支持训练时固定 board_size。"""

    def __init__(self, input_channels: int, board_size: int, num_actions: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            sample = torch.zeros(1, input_channels, board_size, board_size)
            feature_dim = int(self.features(sample).flatten(1).shape[1])

        self.q_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        feats = self.features(x)
        return self.q_head(feats)


class AdaptiveCNN(nn.Module):
    """支持任意地图尺寸的 CNN。

    用 Global Average Pooling 替换 Flatten，卷积输出维度固定为 64，
    与输入分辨率完全解耦。这样同一套权重可以在 8×8、14×14、22×22
    等不同尺寸地图上直接推理，是课程学习（逐步放大地图）和
    随机地图尺寸训练的基础。

    结构：
        Conv(9→32) → ReLU
        Conv(32→64) → ReLU
        Conv(64→64) → ReLU
        GlobalAveragePooling  # 输出恒为 (B, 64)，与 H/W 无关
        Linear(64→128) → ReLU
        Linear(128→num_actions)
    """

    FEATURE_DIM = 64

    def __init__(self, input_channels: int, num_actions: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.FEATURE_DIM, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Global Average Pooling: (B, C, H, W) → (B, C)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.q_head = nn.Sequential(
            nn.Linear(self.FEATURE_DIM, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        feats = self.features(x)          # (B, 64, H, W)
        pooled = self.gap(feats).flatten(1)  # (B, 64)
        return self.q_head(pooled)


class TinyMLP(nn.Module):
    """纯 MLP，输入 10 维射线+食物方向特征，无视觉输入。

    输入：(B, 10) 归一化标量特征
        [0-6] 7 方向射线距离（相对蛇头朝向）
        [7]   食物前向分量
        [8]   食物侧向分量
        [9]   蛇身长度归一化

    结构：Linear(10→64) → ReLU → Linear(64→64) → ReLU → Linear(64→num_actions)
    """

    FEAT_DIM = 10

    def __init__(self, feat_dim: int, num_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        return self.net(x)


class HybridNet(nn.Module):
    """局部 CNN patch + 全局手工特征的混合网络。

    输入：
        map_obs    : (B, C, H, W)  蛇头周围的局部 patch（例如 11x11）
        global_feat: (B, F)        手工提取的全局特征向量

    全局特征由 env.get_global_features() 返回，共 F=10 维：
        [0] 食物相对方向 x（归一化到 [-1, 1]）
        [1] 食物相对方向 y
        [2] 食物曼哈顿距离（归一化）
        [3] 到上方墙距离（归一化）
        [4] 到下方墙距离
        [5] 到左方墙距离
        [6] 到右方墙距离
        [7] 蛇身长度（归一化）
        [8] 蛇占地图比例
        [9] 是否有奖励食物（0/1）

    结构：
        CNN branch  : 局部 patch CNN backbone → GAP → 64 维
        Global branch: Linear(10→32) → ReLU
        Fusion      : concat(64, 32) → Linear(96→128) → ReLU → Linear(128→actions)

    优势：
    - CNN 捕捉蛇头附近的局部拓扑结构
    - 手工特征提供方向感、距离感，不依赖地图绝对大小
    - 跨尺寸泛化能力最强，且参数量比纯 CNN 更小
    """

    GLOBAL_FEAT_DIM = 10
    CNN_OUT_DIM = 64
    GLOBAL_HIDDEN = 32
    FUSED_HIDDEN = 128

    def __init__(self, input_channels: int, num_actions: int) -> None:
        super().__init__()

        # CNN backbone（与 AdaptiveCNN 相同）
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.CNN_OUT_DIM, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 手工特征分支
        self.global_branch = nn.Sequential(
            nn.Linear(self.GLOBAL_FEAT_DIM, self.GLOBAL_HIDDEN),
            nn.ReLU(inplace=True),
        )

        # 融合 Q 头
        fused_in = self.CNN_OUT_DIM + self.GLOBAL_HIDDEN
        self.q_head = nn.Sequential(
            nn.Linear(fused_in, self.FUSED_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Linear(self.FUSED_HIDDEN, num_actions),
        )

    def forward(
        self,
        map_obs: torch.Tensor,
        global_feat: torch.Tensor,
    ) -> torch.Tensor:
        if map_obs.dtype != torch.float32:
            map_obs = map_obs.float()
        if global_feat.dtype != torch.float32:
            global_feat = global_feat.float()

        cnn_out = self.gap(self.cnn(map_obs)).flatten(1)   # (B, 64)
        global_out = self.global_branch(global_feat)        # (B, 32)
        fused = torch.cat([cnn_out, global_out], dim=1)    # (B, 96)
        return self.q_head(fused)
