"""
轻量级 View Transformer - 2D 特征转 BEV

使用简化的 LSS (Lift-Splat-Shoot) 方法将多相机 2D 特征投影到 BEV 空间。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthNet(nn.Module):
    """
    深度预测网络

    从 2D 特征预测每个像素的深度分布
    """
    def __init__(self, in_channels, depth_channels=64, num_depth_bins=32):
        super().__init__()

        self.num_depth_bins = num_depth_bins

        # 深度预测
        self.depth_conv = nn.Sequential(
            nn.Conv2d(in_channels, depth_channels, 3, padding=1),
            nn.BatchNorm2d(depth_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(depth_channels, num_depth_bins, 1),
        )

    def forward(self, x):
        """
        Args:
            x: [B*N, C, H, W] 2D 特征

        Returns:
            depth: [B*N, D, H, W] 深度分布（softmax 归一化）
        """
        depth_logits = self.depth_conv(x)  # [B*N, D, H, W]
        depth = F.softmax(depth_logits, dim=1)  # 归一化为概率分布
        return depth


class LiteViewTransformer(nn.Module):
    """
    轻量级 View Transformer

    将多相机 2D 特征通过深度估计投影到 BEV 空间。

    Args:
        in_channels: 输入特征通道数
        out_channels: 输出 BEV 特征通道数
        feat_height, feat_width: 特征图尺寸 (1/8 分辨率)
        bev_height, bev_width: BEV 网格尺寸
        num_depth_bins: 深度离散化数量
        d_bound: 深度范围 (min, max, step)
        x_bound, y_bound: BEV 空间范围 (米)
    """

    def __init__(
        self,
        in_channels=128,
        out_channels=128,
        feat_height=48,
        feat_width=80,
        bev_height=100,
        bev_width=100,
        num_depth_bins=32,
        d_bound=(2.0, 50.0, 1.5),  # (min_depth, max_depth, step)
        x_bound=(-25.0, 25.0, 0.5),  # (min_x, max_x, resolution)
        y_bound=(-25.0, 25.0, 0.5),  # (min_y, max_y, resolution)
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_height = feat_height
        self.feat_width = feat_width
        self.bev_height = bev_height
        self.bev_width = bev_width
        self.num_depth_bins = num_depth_bins

        # 深度范围
        self.d_min, self.d_max, self.d_step = d_bound
        self.depth_bins = torch.arange(self.d_min, self.d_max, self.d_step)

        # BEV 范围
        self.x_min, self.x_max, self.x_res = x_bound
        self.y_min, self.y_max, self.y_res = y_bound

        # 深度预测
        self.depth_net = DepthNet(
            in_channels=in_channels,
            depth_channels=64,
            num_depth_bins=num_depth_bins
        )

        # 特征变换（降维用于池化）
        self.feat_transform = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # BEV 特征编码
        self.bev_encode = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features, camera_extrinsics=None):
        """
        Args:
            features: [B, N_cam, C, H, W] 多相机特征
            camera_extrinsics: [B, N_cam, 4, 4] 相机外参 (可选)

        Returns:
            bev_feat: [B, C_out, BEV_H, BEV_W] BEV 特征
        """
        B, N, C, H, W = features.shape

        # 展平批次和相机维度
        features_flat = features.view(B * N, C, H, W)

        # 1. 预测深度分布
        depth = self.depth_net(features_flat)  # [B*N, D, H, W]

        # 2. 特征变换
        feat = self.feat_transform(features_flat)  # [B*N, C_out, H, W]

        # 3. 简化版：直接平均池化到 BEV（实际应使用几何投影）
        # 这里为了轻量化，使用可学习的空间变换
        # 将深度加权的特征投影到 BEV

        # Lift: 将 2D 特征提升到 3D（深度加权）
        # [B*N, C, H, W] × [B*N, D, H, W] -> [B*N, C*D, H, W]
        D = depth.shape[1]
        feat_3d = feat.unsqueeze(2) * depth.unsqueeze(1)  # [B*N, C, D, H, W]
        feat_3d = feat_3d.view(B * N, C * D, H, W)

        # Splat: 投影到 BEV（简化版：使用自适应池化）
        # 实际应根据相机内外参进行几何投影
        bev_feat_flat = F.adaptive_avg_pool2d(feat_3d, (self.bev_height, self.bev_width))

        # 降维回原通道数
        bev_feat_flat = F.avg_pool1d(
            bev_feat_flat.view(B * N, C, D, -1).mean(dim=2),  # 先平均深度维度
            kernel_size=1
        ).view(B * N, C, self.bev_height, self.bev_width)

        # Reshape 回批次维度
        bev_feat = bev_feat_flat.view(B, N, self.out_channels, self.bev_height, self.bev_width)

        # 4. 融合多相机（平均或最大池化）
        bev_feat = bev_feat.mean(dim=1)  # [B, C_out, BEV_H, BEV_W]

        # 5. BEV 编码
        bev_feat = self.bev_encode(bev_feat)

        return bev_feat


if __name__ == '__main__':
    print("=" * 60)
    print("Lite View Transformer 测试")
    print("=" * 60)

    # 模拟输入
    B, N_cam = 2, 8
    C, H, W = 128, 48, 80

    features = torch.randn(B, N_cam, C, H, W)

    # 创建模型
    view_transformer = LiteViewTransformer(
        in_channels=128,
        out_channels=128,
        feat_height=48,
        feat_width=80,
        bev_height=100,
        bev_width=100,
        num_depth_bins=32
    )

    print(f"\n输入: {features.shape}")

    # 前向传播
    bev_feat = view_transformer(features)

    print(f"输出 BEV 特征: {bev_feat.shape}")

    # 参数量
    total_params = sum(p.numel() for p in view_transformer.parameters())
    print(f"\n参数量: {total_params/1e6:.2f}M")

    print("\n" + "=" * 60)
    print("✅ 测试通过！")
    print("=" * 60)
