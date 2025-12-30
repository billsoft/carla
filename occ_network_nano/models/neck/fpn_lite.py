"""
轻量级 FPN Neck - 融合多尺度特征

将 Backbone 输出的 C3, C4, C5 特征融合为统一通道数的特征图。
"""

import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    """卷积 + BN + ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class LiteFPN(nn.Module):
    """
    轻量级 FPN（特征金字塔网络）

    输入：
        - C3: [B, 96, H/8, W/8]
        - C4: [B, 128, H/16, W/16]
        - C5: [B, 256, H/32, W/32]

    输出：
        - [B, out_channels, H/8, W/8] 统一尺度特征

    Args:
        in_channels: dict, 输入通道数 {'C3': 96, 'C4': 128, 'C5': 256}
        out_channels: int, 输出通道数（默认 128）
    """

    def __init__(self, in_channels, out_channels=128):
        super().__init__()

        # 1x1 卷积调整通道数
        self.lateral_c5 = ConvBNReLU(in_channels['C5'], out_channels, kernel_size=1)
        self.lateral_c4 = ConvBNReLU(in_channels['C4'], out_channels, kernel_size=1)
        self.lateral_c3 = ConvBNReLU(in_channels['C3'], out_channels, kernel_size=1)

        # 3x3 卷积平滑特征（减少上采样伪影）
        self.smooth_c4 = ConvBNReLU(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth_c3 = ConvBNReLU(out_channels, out_channels, kernel_size=3, padding=1)

        # 最终融合
        self.fusion = ConvBNReLU(out_channels * 3, out_channels, kernel_size=1)

    def forward(self, features):
        """
        Args:
            features: dict, {
                'C3': [B, 96, H/8, W/8],
                'C4': [B, 128, H/16, W/16],
                'C5': [B, 256, H/32, W/32]
            }

        Returns:
            out: [B, out_channels, H/8, W/8]
        """
        c3, c4, c5 = features['C3'], features['C4'], features['C5']

        # 自顶向下路径
        # C5 -> C4
        p5 = self.lateral_c5(c5)
        p5_up = nn.functional.interpolate(p5, size=c4.shape[-2:], mode='bilinear', align_corners=False)
        p4 = self.lateral_c4(c4) + p5_up
        p4 = self.smooth_c4(p4)

        # C4 -> C3
        p4_up = nn.functional.interpolate(p4, size=c3.shape[-2:], mode='bilinear', align_corners=False)
        p3 = self.lateral_c3(c3) + p4_up
        p3 = self.smooth_c3(p3)

        # 将 C5, C4 上采样到 C3 尺度并拼接
        p5_up_to_c3 = nn.functional.interpolate(p5, size=c3.shape[-2:], mode='bilinear', align_corners=False)
        p4_up_to_c3 = nn.functional.interpolate(p4, size=c3.shape[-2:], mode='bilinear', align_corners=False)

        # 融合所有尺度
        fused = torch.cat([p3, p4_up_to_c3, p5_up_to_c3], dim=1)  # [B, 3*out_channels, H/8, W/8]
        out = self.fusion(fused)  # [B, out_channels, H/8, W/8]

        return out


if __name__ == '__main__':
    print("=" * 60)
    print("Lite FPN 测试")
    print("=" * 60)

    # 模拟 Backbone 输出
    B = 2
    features = {
        'C3': torch.randn(B, 96, 48, 80),    # 1/8
        'C4': torch.randn(B, 128, 24, 40),   # 1/16
        'C5': torch.randn(B, 256, 12, 20),   # 1/32
    }

    in_channels = {'C3': 96, 'C4': 128, 'C5': 256}
    fpn = LiteFPN(in_channels=in_channels, out_channels=128)

    print(f"\n输入:")
    for k, v in features.items():
        print(f"  {k}: {v.shape}")

    out = fpn(features)

    print(f"\n输出:")
    print(f"  {out.shape}")

    # 参数量
    total_params = sum(p.numel() for p in fpn.parameters())
    print(f"\n参数量: {total_params/1e6:.2f}M")

    print("\n" + "=" * 60)
    print("✅ 测试通过！")
    print("=" * 60)
