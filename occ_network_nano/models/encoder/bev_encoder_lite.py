"""
轻量级 BEV Encoder - 增强 BEV 特征

使用轻量级卷积块增强 BEV 特征的空间信息。
"""

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积（MobileNet 风格）"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BEVResBlock(nn.Module):
    """BEV 残差块"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels)
        self.conv2 = DepthwiseSeparableConv(channels, channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        return out


class LiteBEVEncoder(nn.Module):
    """
    轻量级 BEV Encoder

    增强 BEV 特征的空间关系和上下文信息。

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        num_blocks: 残差块数量
    """

    def __init__(self, in_channels=128, out_channels=128, num_blocks=2):
        super().__init__()

        # 入口卷积
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 残差块
        self.blocks = nn.ModuleList([
            BEVResBlock(out_channels) for _ in range(num_blocks)
        ])

        # 输出卷积
        self.head = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: [B, C_in, H, W] BEV 特征

        Returns:
            out: [B, C_out, H, W] 增强后的 BEV 特征
        """
        x = self.stem(x)

        for block in self.blocks:
            x = block(x)

        x = self.head(x)

        return x


if __name__ == '__main__':
    print("=" * 60)
    print("Lite BEV Encoder 测试")
    print("=" * 60)

    # 模拟输入
    B, C, H, W = 2, 128, 100, 100
    x = torch.randn(B, C, H, W)

    # 创建模型
    bev_encoder = LiteBEVEncoder(in_channels=128, out_channels=128, num_blocks=2)

    print(f"\n输入: {x.shape}")

    # 前向传播
    out = bev_encoder(x)

    print(f"输出: {out.shape}")

    # 参数量
    total_params = sum(p.numel() for p in bev_encoder.parameters())
    print(f"\n参数量: {total_params/1e6:.2f}M")

    print("\n" + "=" * 60)
    print("✅ 测试通过！")
    print("=" * 60)
