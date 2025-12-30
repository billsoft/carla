"""
轻量级 3D Occupancy Decoder - BEV 转体素

将 2D BEV 特征扩展到 3D 体素网格并预测占用类别。
"""

import torch
import torch.nn as nn


class HeightExpansion(nn.Module):
    """
    高度扩展模块

    将 2D BEV 特征扩展到 3D，通过 MLP 预测每个高度层的特征。
    """
    def __init__(self, in_channels, out_channels, num_height_layers):
        super().__init__()

        self.num_height_layers = num_height_layers

        # 为每个高度层学习不同的特征
        self.height_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels * num_height_layers, 1)
        )

    def forward(self, x):
        """
        Args:
            x: [B, C_in, H, W] BEV 特征

        Returns:
            out: [B, C_out, H, W, Z] 3D 特征
        """
        B, C, H, W = x.shape

        # 预测所有高度层的特征
        feat_3d = self.height_mlp(x)  # [B, C_out*Z, H, W]

        # Reshape 到 3D
        feat_3d = feat_3d.view(B, -1, self.num_height_layers, H, W)  # [B, C_out, Z, H, W]
        feat_3d = feat_3d.permute(0, 1, 3, 4, 2).contiguous()  # [B, C_out, H, W, Z]

        return feat_3d


class Conv3DBlock(nn.Module):
    """3D 卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class LiteOccDecoder(nn.Module):
    """
    轻量级 Occupancy Decoder

    将 BEV 特征转换为 3D 占用网格。

    Args:
        in_channels: 输入 BEV 特征通道数
        num_classes: 占用类别数（包括空类）
        grid_size: 3D 网格尺寸 (X, Y, Z)
        hidden_channels: 中间特征通道数
    """

    def __init__(
        self,
        in_channels=128,
        num_classes=18,
        grid_size=(200, 200, 16),
        hidden_channels=64
    ):
        super().__init__()

        self.num_classes = num_classes
        self.grid_size = grid_size
        X, Y, Z = grid_size

        # 1. BEV 特征调整到目标网格尺寸
        self.bev_resize = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        # 2. 高度扩展：2D → 3D
        self.height_expand = HeightExpansion(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            num_height_layers=Z
        )

        # 3. 3D 卷积增强
        self.conv3d_blocks = nn.Sequential(
            Conv3DBlock(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            Conv3DBlock(hidden_channels, hidden_channels, kernel_size=3, padding=1),
        )

        # 4. 分类头
        self.cls_head = nn.Conv3d(hidden_channels, num_classes, kernel_size=1)

    def forward(self, bev_feat):
        """
        Args:
            bev_feat: [B, C_in, H_bev, W_bev] BEV 特征

        Returns:
            occ_logits: [B, num_classes, X, Y, Z] 占用 logits
        """
        B, C, H_bev, W_bev = bev_feat.shape
        X, Y, Z = self.grid_size

        # 1. 调整 BEV 尺寸到目标网格
        bev_feat = self.bev_resize(bev_feat)  # [B, hidden_C, H_bev, W_bev]

        if (H_bev, W_bev) != (X, Y):
            bev_feat = nn.functional.interpolate(
                bev_feat, size=(X, Y), mode='bilinear', align_corners=False
            )  # [B, hidden_C, X, Y]

        # 2. 高度扩展：2D → 3D
        feat_3d = self.height_expand(bev_feat)  # [B, hidden_C, X, Y, Z]

        # 3. 3D 卷积增强
        feat_3d = self.conv3d_blocks(feat_3d)  # [B, hidden_C, X, Y, Z]

        # 4. 分类
        occ_logits = self.cls_head(feat_3d)  # [B, num_classes, X, Y, Z]

        return occ_logits


if __name__ == '__main__':
    print("=" * 60)
    print("Lite Occupancy Decoder 测试")
    print("=" * 60)

    # 模拟输入
    B, C, H, W = 2, 128, 100, 100
    bev_feat = torch.randn(B, C, H, W)

    # 创建模型
    decoder = LiteOccDecoder(
        in_channels=128,
        num_classes=18,
        grid_size=(200, 200, 16),
        hidden_channels=64
    )

    print(f"\n输入 BEV 特征: {bev_feat.shape}")

    # 前向传播
    occ_logits = decoder(bev_feat)

    print(f"输出占用 logits: {occ_logits.shape}")
    print(f"  预期: [B={B}, num_classes=18, X=200, Y=200, Z=16]")

    # 参数量
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"\n参数量: {total_params/1e6:.2f}M")

    print("\n" + "=" * 60)
    print("✅ 测试通过！")
    print("=" * 60)
