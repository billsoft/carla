# models/occ_decoder.py
"""
Occupancy Decoder 模块

将 2D BEV 特征提升到 3D 体素空间

核心思想:
1. 为每个 BEV 位置预测 Z 方向上的特征分布
2. 使用 3D 卷积精炼体素特征
3. 输出每个体素的类别概率

架构:
    BEV [B, C, H, W] 
      -> Height MLP: 预测每个高度层的特征
      -> 3D Conv: 精炼体素特征
      -> Classification Head: 输出类别 logits
      -> Output [B, num_classes, H, W, Z]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class HeightMLP(nn.Module):
    """
    高度 MLP
    
    将 2D BEV 特征扩展到 3D，为每个高度层生成特征
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 128,
        num_heights: int = 16,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heights = num_heights
        
        # 为每个高度层生成特征
        # 输入: [B, C, H, W]
        # 输出: [B, C_out * num_heights, H, W] -> reshape -> [B, C_out, H, W, Z]
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels * num_heights, kernel_size=1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [B, C, H, W] BEV 特征
            
        Returns:
            output: [B, C_out, H, W, Z] 3D 体素特征
        """
        B, C, H, W = x.shape
        
        # MLP 预测
        out = self.mlp(x)  # [B, C_out * Z, H, W]
        
        # 重塑为 3D
        out = out.view(B, self.out_channels, self.num_heights, H, W)
        # [B, C_out, Z, H, W]
        
        # 调整维度顺序: [B, C_out, H, W, Z]
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        
        return out


class Conv3DBlock(nn.Module):
    """
    3D 卷积块
    
    包含 3D 卷积、归一化和激活
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResBlock3D(nn.Module):
    """
    3D 残差块
    """
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class OccDecoder(nn.Module):
    """
    Occupancy Decoder
    
    完整的 2D→3D 解码器，输出体素语义预测
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 128,
        num_classes: int = 18,
        num_heights: int = 16,
        use_3d_conv: bool = True,
        num_3d_layers: int = 2,
    ):
        """
        Args:
            in_channels: BEV 特征通道数
            hidden_channels: 3D 特征通道数
            num_classes: 输出类别数
            num_heights: Z 方向的离散高度数
            use_3d_conv: 是否使用 3D 卷积精炼
            num_3d_layers: 3D 卷积层数
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.num_heights = num_heights
        self.use_3d_conv = use_3d_conv
        
        # 1. 高度 MLP: 2D -> 3D
        self.height_mlp = HeightMLP(
            in_channels=in_channels,
            out_channels=hidden_channels,
            num_heights=num_heights,
        )
        
        # 2. 3D 卷积精炼（可选）
        if use_3d_conv:
            self.conv3d_layers = nn.Sequential(
                Conv3DBlock(hidden_channels, hidden_channels),
                *[ResBlock3D(hidden_channels) for _ in range(num_3d_layers)],
            )
        else:
            self.conv3d_layers = nn.Identity()
            
        # 3. 分类头
        self.cls_head = nn.Conv3d(hidden_channels, num_classes, kernel_size=1)
        
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, bev_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            bev_features: [B, C, H, W] BEV 特征
            
        Returns:
            occ_logits: [B, num_classes, H, W, Z] 体素类别 logits
        """
        # 1. 2D -> 3D
        x = self.height_mlp(bev_features)  # [B, hidden_channels, H, W, Z]
        
        # 调整为 3D 卷积格式: [B, C, D, H, W] (D=Z)
        x = x.permute(0, 1, 4, 2, 3)  # [B, C, Z, H, W]
        
        # 2. 3D 卷积精炼
        x = self.conv3d_layers(x)  # [B, C, Z, H, W]
        
        # 3. 分类
        logits = self.cls_head(x)  # [B, num_classes, Z, H, W]
        
        # 调整维度顺序: [B, num_classes, H, W, Z]
        logits = logits.permute(0, 1, 3, 4, 2).contiguous()
        
        return logits


class OccDecoderWithUpsample(nn.Module):
    """
    带上采样的 Occupancy Decoder
    
    训练时输出低分辨率 (200×200×16)
    推理时可上采样到完整分辨率 (500×500×40)
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 128,
        num_classes: int = 18,
        train_grid_size: Tuple[int, int, int] = (200, 200, 16),
        full_grid_size: Tuple[int, int, int] = (500, 500, 40),
        use_3d_conv: bool = True,
    ):
        super().__init__()
        
        self.train_grid_size = train_grid_size
        self.full_grid_size = full_grid_size
        
        # 基础解码器
        self.decoder = OccDecoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            num_heights=train_grid_size[2],
            use_3d_conv=use_3d_conv,
        )
        
    def forward(
        self, 
        bev_features: torch.Tensor,
        upsample: bool = False,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            bev_features: [B, C, H, W] BEV 特征
            upsample: 是否上采样到完整分辨率
            
        Returns:
            occ_logits: [B, num_classes, H, W, Z]
        """
        # 基础解码
        logits = self.decoder(bev_features)
        # [B, num_classes, 200, 200, 16]
        
        if upsample and self.train_grid_size != self.full_grid_size:
            # 上采样到完整分辨率
            B, C, H, W, Z = logits.shape
            
            # 重塑为 5D 张量进行 3D 插值
            # [B, C, H, W, Z] -> [B, C, Z, H, W]
            logits = logits.permute(0, 1, 4, 2, 3)
            
            # 3D 插值
            logits = F.interpolate(
                logits,
                size=self.full_grid_size[::-1],  # (Z, H, W) -> (40, 500, 500)
                mode='trilinear',
                align_corners=False,
            )
            
            # [B, C, Z, H, W] -> [B, C, H, W, Z]
            logits = logits.permute(0, 1, 3, 4, 2).contiguous()
            
        return logits


class LightweightOccDecoder(nn.Module):
    """
    轻量级 Occupancy Decoder
    
    不使用 3D 卷积，适合资源受限场景
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 18,
        num_heights: int = 16,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_heights = num_heights
        
        # 直接用 2D 卷积预测每个高度层的类别
        self.predictor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes * num_heights, 1),
        )
        
    def forward(self, bev_features: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        B, C, H, W = bev_features.shape
        
        # 预测
        out = self.predictor(bev_features)
        # [B, num_classes * num_heights, H, W]
        
        # 重塑
        out = out.view(B, self.num_classes, self.num_heights, H, W)
        # [B, num_classes, Z, H, W]
        
        # 调整维度: [B, num_classes, H, W, Z]
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        
        return out


# 测试代码
if __name__ == '__main__':
    print("Testing Occ Decoder...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 配置
    B = 2
    in_channels = 256
    H, W = 200, 200
    num_classes = 18
    num_heights = 16
    
    # 1. 测试基础解码器
    print("\n1. Testing Basic Occ Decoder...")
    decoder = OccDecoder(
        in_channels=in_channels,
        hidden_channels=128,
        num_classes=num_classes,
        num_heights=num_heights,
        use_3d_conv=True,
    ).to(device)
    
    bev = torch.randn(B, in_channels, H, W).to(device)
    
    with torch.no_grad():
        logits = decoder(bev)
    
    print(f"   Input shape: {bev.shape}")
    print(f"   Output shape: {logits.shape}")  # [2, 18, 200, 200, 16]
    
    params = sum(p.numel() for p in decoder.parameters())
    print(f"   Parameters: {params / 1e6:.2f}M")
    
    # 2. 测试带上采样的解码器
    print("\n2. Testing Occ Decoder with Upsample...")
    decoder_up = OccDecoderWithUpsample(
        in_channels=in_channels,
        hidden_channels=128,
        num_classes=num_classes,
        train_grid_size=(200, 200, 16),
        full_grid_size=(500, 500, 40),
    ).to(device)
    
    with torch.no_grad():
        logits_train = decoder_up(bev, upsample=False)
        logits_full = decoder_up(bev, upsample=True)
    
    print(f"   Train output shape: {logits_train.shape}")  # [2, 18, 200, 200, 16]
    print(f"   Full output shape: {logits_full.shape}")    # [2, 18, 500, 500, 40]
    
    # 3. 测试轻量级解码器
    print("\n3. Testing Lightweight Occ Decoder...")
    decoder_light = LightweightOccDecoder(
        in_channels=in_channels,
        num_classes=num_classes,
        num_heights=num_heights,
    ).to(device)
    
    with torch.no_grad():
        logits_light = decoder_light(bev)
    
    print(f"   Output shape: {logits_light.shape}")
    
    params_light = sum(p.numel() for p in decoder_light.parameters())
    print(f"   Parameters: {params_light / 1e6:.2f}M")
    
    print("\n✓ All tests passed!")
