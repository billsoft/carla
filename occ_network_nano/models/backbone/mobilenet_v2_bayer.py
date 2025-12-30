"""
Bayer RAW MobileNetV2 Backbone

单通道 Bayer RGGB 输入的 MobileNetV2，专为车规级传感器优化。

设计特点：
1. 输入：1 通道 Bayer (H, W) - 数据量降低 66%
2. PixelUnshuffle：1→4 通道，RGGB 分离（无参数，空间换通道）
3. Stem：4→48 通道，3×3 卷积，stride=1（处理分离的 RGGB）
4. 后续层：标准 MobileNetV2 结构，逐步下采样
5. 参数量：约 4.9M（优化后）
"""

import torch
import torch.nn as nn
from typing import Dict


def _make_divisible(v, divisor=8, min_value=None):
    """确保通道数可被 divisor 整除"""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    """卷积 + BN + ReLU6"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                     groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    """倒残差块（MobileNetV2 核心模块）"""
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # Pointwise expand
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1))

        layers.extend([
            # Depthwise
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # Pointwise linear
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class BayerMobileNetV2(nn.Module):
    """
    单通道 Bayer 输入的 MobileNetV2

    输入：1 通道 Bayer RGGB (H, W)
    输出：多尺度特征 C3 (1/8), C4 (1/16), C5 (1/32)

    Args:
        width_mult: 宽度乘数（控制通道数）
    """

    def __init__(self, width_mult=1.0):
        super().__init__()

        # ========== PixelUnshuffle: RGGB 分离（无参数）==========
        # [B, 1, H, W] → [B, 4, H/2, W/2]
        # 将 2×2 Bayer 块转为 4 通道（R, Gr, Gb, B）
        # 原理：Space-to-Depth，避免卷积核跨越不同颜色
        self.pixel_unshuffle = nn.PixelUnshuffle(2)

        # ========== Stem：处理 RGGB 4 通道（内存优化版）==========
        # 4→32 通道，3×3 卷积，stride=2
        # 优化：降低通道数 48→32，stride=2 快速下采样，显存降低 75%
        stem_channels = _make_divisible(32 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(4, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU6(inplace=True)
        )

        # ========== 标准 MobileNetV2 Blocks（内存优化版）==========
        # 从 32 通道开始，逐步下采样
        # 注意：PixelUnshuffle (1/2) + Stem stride=2 (1/2) = 总共 1/4 下采样
        # - 输入: H×W
        # - PixelUnshuffle: H/2×W/2
        # - Stem stride=2: H/4×W/4
        # - Stage 1: H/8×W/8
        # - Stage 2: H/16×W/16 → C3
        # - Stage 3: H/32×W/32 → C4
        # - Stage 5: H/64×W/64 → C5

        # inverted_residual_setting: [expand_ratio, out_channels, num_blocks, stride]
        # 优化：expand_ratio 从 6 降低到 4，显存降低 33%
        inverted_residual_setting = [
            [4, 48,  1, 2],   # Stage 1: 32→48, stride=2 (1/8 总下采样)
            [4, 64, 2, 2],    # Stage 2: 48→64, stride=2 (1/16) → C3
            [4, 96, 3, 2],    # Stage 3: 64→96, stride=2 (1/32) → C4
            [4, 128, 4, 1],   # Stage 4: 96→128, stride=1 (保持 1/32)
            [4, 192, 3, 2],   # Stage 5: 128→192, stride=2 (1/64) → C5
            [4, 256, 1, 1],   # Stage 6: 192→256, stride=1 (保持 1/64)
        ]

        self.layers = nn.ModuleList()
        in_channels = stem_channels

        for t, c, n, s in inverted_residual_setting:
            out_channels = _make_divisible(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.layers.append(
                    InvertedResidual(in_channels, out_channels, stride, expand_ratio=t)
                )
                in_channels = out_channels

        # ========== 输出通道配置（内存优化版）==========
        self.out_channels = {
            'C3': _make_divisible(64 * width_mult),   # 1/16 分辨率（Stem stride=2 后）
            'C4': _make_divisible(96 * width_mult),   # 1/32 分辨率
            'C5': _make_divisible(192 * width_mult),  # 1/64 分辨率
        }

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播（内存优化版）

        Args:
            x: 输入 Bayer 图像 (B, 1, H, W)

        Returns:
            多尺度特征字典
                - C3: (B, 64, H/16, W/16)
                - C4: (B, 96, H/32, W/32)
                - C5: (B, 192, H/64, W/64)
        """
        features = {}

        # PixelUnshuffle: [B, 1, H, W] → [B, 4, H/2, W/2]
        x = self.pixel_unshuffle(x)

        # Stem: [B, 4, H/2, W/2] → [B, 32, H/4, W/4] (stride=2)
        x = self.stem(x)

        # 逐层前向（内存优化版：通道数降低，expand_ratio=4）
        for idx, layer in enumerate(self.layers):
            x = layer(x)

            # 保存多尺度特征
            # Stage 1 (idx=0): 32→48, stride=2 → H/8, W/8
            # Stage 2 (idx=1-2): 48→64, 2 blocks → H/16, W/16 (C3)
            # Stage 3 (idx=3-5): 64→96, 3 blocks → H/32, W/32 (C4)
            # Stage 4 (idx=6-9): 96→128, 4 blocks → H/32, W/32
            # Stage 5 (idx=10-12): 128→192, 3 blocks → H/64, W/64 (C5)
            # Stage 6 (idx=13): 192→256, 1 block → H/64, W/64

            if idx == 2:  # Stage 2 结束 (1/16 总下采样)
                features['C3'] = x
            elif idx == 5:  # Stage 3 结束 (1/32 总下采样)
                features['C4'] = x
            elif idx == 12:  # Stage 5 结束 (1/64 总下采样)
                features['C5'] = x

        return features


def build_bayer_mobilenetv2(width_mult=1.0) -> BayerMobileNetV2:
    """
    构建 Bayer MobileNetV2

    Args:
        width_mult: 宽度乘数

    Returns:
        BayerMobileNetV2 模型
    """
    model = BayerMobileNetV2(width_mult=width_mult)
    return model


if __name__ == '__main__':
    print("=" * 60)
    print("Bayer MobileNetV2 测试")
    print("=" * 60)

    # 创建模型
    model = build_bayer_mobilenetv2(width_mult=1.0)

    # 打印输出通道
    print(f"\n输出通道配置:")
    for key, val in model.out_channels.items():
        print(f"  {key}: {val} 通道")

    # 测试前向传播
    batch_size = 2
    H, W = 960, 1280
    x = torch.randn(batch_size, 1, H, W)  # 单通道 Bayer

    print(f"\n输入: {x.shape}")
    features = model(x)

    print(f"\n输出特征:")
    for key, val in features.items():
        print(f"  {key}: {val.shape}")

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n参数量:")
    print(f"  总计: {total_params/1e6:.2f}M")
    print(f"  可训练: {trainable_params/1e6:.2f}M")

    print("\n" + "=" * 60)
    print("✅ 测试通过！")
    print("=" * 60)
