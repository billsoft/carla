# models/neck.py
"""
特征金字塔网络 (FPN Neck)

将 Backbone 输出的多尺度特征融合为统一通道数的特征图
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class FPNNeck(nn.Module):
    """
    Feature Pyramid Network
    
    自顶向下融合多尺度特征，输出统一通道数的特征图
    
    架构:
        C5 (1/32) → P5 → Upsample → 
        C4 (1/16) → Lateral → Add → P4 → Upsample →
        C3 (1/8)  → Lateral → Add → P3
    """
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        num_outs: int = 1,
        upsample_mode: str = 'bilinear',
    ):
        """
        Args:
            in_channels: 各层输入通道数，如 [256, 512, 1024]
            out_channels: 输出统一的通道数
            num_outs: 输出几个尺度的特征 (1=只输出最大分辨率)
            upsample_mode: 上采样方式 ('bilinear' 或 'nearest')
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.upsample_mode = upsample_mode
        
        # Lateral connections (1x1 conv to unify channels)
        self.lateral_convs = nn.ModuleList()
        for in_ch in in_channels:
            self.lateral_convs.append(
                nn.Conv2d(in_ch, out_channels, kernel_size=1)
            )
        
        # Output convolutions (3x3 conv to smooth features)
        self.output_convs = nn.ModuleList()
        for _ in range(len(in_channels)):
            self.output_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        前向传播
        
        Args:
            features: Backbone 输出的多尺度特征 [C3, C4, C5]
            
        Returns:
            FPN 输出的特征 [P3, P4, P5] 或只返回 [P3]
        """
        assert len(features) == len(self.in_channels)
        
        # 1. Lateral connections
        laterals = [
            lateral_conv(feat) 
            for feat, lateral_conv in zip(features, self.lateral_convs)
        ]
        
        # 2. Top-down pathway
        # 从最高层开始，逐层向下融合
        for i in range(len(laterals) - 1, 0, -1):
            # 上采样高层特征
            upsampled = F.interpolate(
                laterals[i],
                size=laterals[i-1].shape[-2:],
                mode=self.upsample_mode,
                align_corners=False if self.upsample_mode == 'bilinear' else None,
            )
            # 与低层特征相加
            laterals[i-1] = laterals[i-1] + upsampled
            
        # 3. Output convolutions
        outputs = [
            output_conv(lateral)
            for lateral, output_conv in zip(laterals, self.output_convs)
        ]
        
        # 4. 根据 num_outs 返回
        if self.num_outs == 1:
            # 只返回最大分辨率的特征 (P3)
            return [outputs[0]]
        else:
            return outputs[:self.num_outs]


class BiFPNNeck(nn.Module):
    """
    双向特征金字塔网络 (BiFPN)
    
    相比 FPN，增加了自底向上的路径和加权融合
    """
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        
        # 输入投影
        self.input_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels
        ])
        
        # BiFPN 层
        self.bifpn_layers = nn.ModuleList([
            BiFPNLayer(out_channels, len(in_channels))
            for _ in range(num_layers)
        ])
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """前向传播"""
        # 输入投影
        features = [
            conv(feat) for feat, conv in zip(features, self.input_convs)
        ]
        
        # BiFPN 层
        for bifpn_layer in self.bifpn_layers:
            features = bifpn_layer(features)
            
        return features


class BiFPNLayer(nn.Module):
    """单个 BiFPN 层"""
    
    def __init__(self, channels: int, num_levels: int):
        super().__init__()
        
        self.num_levels = num_levels
        
        # 自顶向下的融合权重（可学习）
        self.td_weights = nn.ParameterList([
            nn.Parameter(torch.ones(2))
            for _ in range(num_levels - 1)
        ])
        
        # 自底向上的融合权重（可学习）
        self.bu_weights = nn.ParameterList([
            nn.Parameter(torch.ones(3 if i > 0 else 2))
            for i in range(num_levels - 1)
        ])
        
        # 融合后的卷积
        self.td_convs = nn.ModuleList([
            self._make_conv(channels) for _ in range(num_levels - 1)
        ])
        
        self.bu_convs = nn.ModuleList([
            self._make_conv(channels) for _ in range(num_levels - 1)
        ])
        
        self.epsilon = 1e-4
        
    def _make_conv(self, channels: int):
        """创建深度可分离卷积"""
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """前向传播"""
        # 自顶向下
        td_features = [features[-1]]  # 从最高层开始
        
        for i in range(self.num_levels - 2, -1, -1):
            w = F.relu(self.td_weights[i])
            w = w / (w.sum() + self.epsilon)
            
            upsampled = F.interpolate(
                td_features[0], size=features[i].shape[-2:], mode='nearest'
            )
            fused = w[0] * features[i] + w[1] * upsampled
            fused = self.td_convs[self.num_levels - 2 - i](fused)
            td_features.insert(0, fused)
            
        # 自底向上
        bu_features = [td_features[0]]  # 从最低层开始
        
        for i in range(1, self.num_levels):
            w = F.relu(self.bu_weights[i-1])
            w = w / (w.sum() + self.epsilon)
            
            downsampled = F.interpolate(
                bu_features[-1], size=td_features[i].shape[-2:], mode='nearest'
            )
            
            if i < self.num_levels - 1:
                # 中间层：融合三个来源
                fused = w[0] * features[i] + w[1] * td_features[i] + w[2] * downsampled
            else:
                # 最高层：融合两个来源
                fused = w[0] * td_features[i] + w[1] * downsampled
                
            fused = self.bu_convs[i-1](fused)
            bu_features.append(fused)
            
        return bu_features


# 测试代码
if __name__ == '__main__':
    print("Testing FPN Neck...")
    
    # 模拟 Backbone 输出
    c3 = torch.randn(2, 256, 48, 80)   # 1/8
    c4 = torch.randn(2, 512, 24, 40)   # 1/16
    c5 = torch.randn(2, 1024, 12, 20)  # 1/32
    
    features = [c3, c4, c5]
    
    # 测试 FPN
    fpn = FPNNeck(in_channels=[256, 512, 1024], out_channels=256, num_outs=1)
    outputs = fpn(features)
    
    print(f"Input shapes: {[f.shape for f in features]}")
    print(f"FPN output shapes: {[o.shape for o in outputs]}")
    
    # 测试 BiFPN
    print("\nTesting BiFPN Neck...")
    bifpn = BiFPNNeck(in_channels=[256, 512, 1024], out_channels=256, num_layers=2)
    outputs = bifpn(features)
    print(f"BiFPN output shapes: {[o.shape for o in outputs]}")
