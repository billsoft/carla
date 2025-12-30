# models/bev_encoder.py
"""
BEV Encoder 模块

对 BEV 特征进行进一步编码，增强空间上下文理解
使用 ResNet 风格的 2D 卷积
"""

import torch
import torch.nn as nn
from typing import Optional


class BasicBlock2D(nn.Module):
    """
    基础残差块 (2D)
    
    包含两个 3x3 卷积和一个跳跃连接
    """
    
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out


class BEVEncoder(nn.Module):
    """
    BEV 特征编码器
    
    使用多层残差块增强 BEV 特征的表达能力
    
    架构:
        Input [C, H, W] 
          -> Layer1 (保持分辨率)
          -> Layer2 (可选下采样)
          -> Layer3 (可选下采样)
          -> Output [C_out, H_out, W_out]
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        num_layers: int = 4,
        layer_channels: Optional[list] = None,
    ):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            num_layers: 残差块数量
            layer_channels: 每层的通道数列表
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 默认通道数配置
        if layer_channels is None:
            layer_channels = [in_channels] * num_layers
            
        # 构建层
        layers = []
        current_channels = in_channels
        
        for i, ch in enumerate(layer_channels):
            # 第一个块可能需要调整通道数
            if ch != current_channels:
                downsample = nn.Sequential(
                    nn.Conv2d(current_channels, ch, 1, bias=False),
                    nn.BatchNorm2d(ch),
                )
            else:
                downsample = None
                
            layers.append(BasicBlock2D(current_channels, ch, downsample=downsample))
            current_channels = ch
            
        self.layers = nn.Sequential(*layers)
        
        # 输出投影
        if current_channels != out_channels:
            self.output_proj = nn.Sequential(
                nn.Conv2d(current_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.output_proj = nn.Identity()
            
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [B, C, H, W] BEV 特征
            
        Returns:
            output: [B, C_out, H, W] 编码后的 BEV 特征
        """
        x = self.layers(x)
        x = self.output_proj(x)
        return x


class MultiscaleBEVEncoder(nn.Module):
    """
    多尺度 BEV 编码器
    
    类似 U-Net 的结构，先下采样再上采样，融合多尺度特征
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        base_channels: int = 64,
    ):
        super().__init__()
        
        # 下采样路径
        self.down1 = self._make_layer(in_channels, base_channels * 2)
        self.down2 = self._make_layer(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)
        
        # 瓶颈层
        self.bottleneck = self._make_layer(base_channels * 4, base_channels * 4)
        
        # 上采样路径
        self.up2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, 
            kernel_size=2, stride=2
        )
        self.conv_up2 = self._make_layer(base_channels * 4, base_channels * 2)
        
        self.up1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, 
            kernel_size=2, stride=2
        )
        self.conv_up1 = self._make_layer(base_channels * 2 + in_channels, base_channels)
        
        # 输出层
        self.output_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
    def _make_layer(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 保存输入用于跳跃连接
        x0 = x
        
        # 下采样
        x1 = self.down1(x)          # [B, 128, H, W]
        x1_pool = self.pool(x1)     # [B, 128, H/2, W/2]
        
        x2 = self.down2(x1_pool)    # [B, 256, H/2, W/2]
        x2_pool = self.pool(x2)     # [B, 256, H/4, W/4]
        
        # 瓶颈
        x_bottle = self.bottleneck(x2_pool)  # [B, 256, H/4, W/4]
        
        # 上采样
        x_up2 = self.up2(x_bottle)  # [B, 128, H/2, W/2]
        x_up2 = torch.cat([x_up2, x2], dim=1)  # [B, 384, H/2, W/2]
        x_up2 = self.conv_up2(x_up2)  # [B, 128, H/2, W/2]
        
        x_up1 = self.up1(x_up2)  # [B, 64, H, W]
        x_up1 = torch.cat([x_up1, x1, x0], dim=1)  # 拼接跳跃连接
        x_up1 = self.conv_up1(x_up1)  # [B, 64, H, W]
        
        # 输出
        output = self.output_conv(x_up1)  # [B, out_channels, H, W]
        
        return output


# 测试代码
if __name__ == '__main__':
    print("Testing BEV Encoder...")
    
    # 配置
    B = 2
    in_channels = 256
    out_channels = 256
    H, W = 200, 200
    
    # 测试基础编码器
    print("\n1. Testing Basic BEV Encoder...")
    encoder = BEVEncoder(in_channels, out_channels, num_layers=4)
    
    x = torch.randn(B, in_channels, H, W)
    out = encoder(x)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    
    params = sum(p.numel() for p in encoder.parameters())
    print(f"   Parameters: {params / 1e6:.2f}M")
    
    # 测试多尺度编码器
    print("\n2. Testing Multiscale BEV Encoder...")
    ms_encoder = MultiscaleBEVEncoder(in_channels, out_channels)
    
    out_ms = ms_encoder(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out_ms.shape}")
    
    params_ms = sum(p.numel() for p in ms_encoder.parameters())
    print(f"   Parameters: {params_ms / 1e6:.2f}M")
    
    print("\n✓ All tests passed!")
