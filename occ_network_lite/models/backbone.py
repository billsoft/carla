# models/backbone.py
"""
2D 图像编码器 (Backbone)

支持的架构:
- ResNet50/101
- EfficientNet-B4

输出多尺度特征用于后续 FPN 融合
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Tuple, Optional


class ResNetBackbone(nn.Module):
    """
    ResNet Backbone
    
    输出多尺度特征:
    - C3: 1/8 分辨率, 256 channels (ResNet50) 
    - C4: 1/16 分辨率, 512 channels
    - C5: 1/32 分辨率, 1024 channels
    """
    
    def __init__(
        self,
        depth: int = 50,
        pretrained: bool = True,
        frozen_stages: int = 1,
        out_indices: Tuple[int, ...] = (1, 2, 3),
    ):
        """
        Args:
            depth: ResNet 深度 (50 或 101)
            pretrained: 是否使用 ImageNet 预训练权重
            frozen_stages: 冻结前几个 stage (0-4)
            out_indices: 输出哪些 stage 的特征 (0=stem, 1=layer1, ...)
        """
        super().__init__()
        
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        
        # 加载预训练模型
        if depth == 50:
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            resnet = models.resnet50(weights=weights)
            self.out_channels = [256, 512, 1024, 2048]
        elif depth == 101:
            weights = models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            resnet = models.resnet101(weights=weights)
            self.out_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported ResNet depth: {depth}")
        
        # 分解 ResNet 各层
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        
        self.layer1 = resnet.layer1  # C2: 1/4, 256ch
        self.layer2 = resnet.layer2  # C3: 1/8, 512ch
        self.layer3 = resnet.layer3  # C4: 1/16, 1024ch
        self.layer4 = resnet.layer4  # C5: 1/32, 2048ch
        
        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        
        # 冻结指定层
        self._freeze_stages()
        
    def _freeze_stages(self):
        """冻结指定的 stage"""
        if self.frozen_stages >= 0:
            # 冻结 stem
            for param in self.stem.parameters():
                param.requires_grad = False
                
        for i in range(self.frozen_stages):
            layer = self.layers[i]
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False
                
    def train(self, mode: bool = True):
        """重写 train 方法，保持冻结层为 eval 模式"""
        super().train(mode)
        self._freeze_stages()
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        前向传播
        
        Args:
            x: [B, 3, H, W] 输入图像
            
        Returns:
            多尺度特征列表，根据 out_indices 返回对应层的输出
        """
        outputs = []
        
        # Stem
        x = self.stem(x)
        if 0 in self.out_indices:
            outputs.append(x)
            
        # 4 个 stage
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if (i + 1) in self.out_indices:
                outputs.append(x)
                
        return outputs


class EfficientNetBackbone(nn.Module):
    """
    EfficientNet Backbone
    
    更高效的特征提取，适合边缘部署
    """
    
    def __init__(
        self,
        variant: str = 'b4',
        pretrained: bool = True,
        out_indices: Tuple[int, ...] = (2, 3, 4),
    ):
        super().__init__()
        
        self.out_indices = out_indices
        
        # 加载预训练模型
        if variant == 'b0':
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            efficientnet = models.efficientnet_b0(weights=weights)
            self.out_channels = [24, 40, 112, 320]
        elif variant == 'b4':
            weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            efficientnet = models.efficientnet_b4(weights=weights)
            self.out_channels = [32, 56, 160, 448]
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {variant}")
            
        # 提取特征提取部分
        self.features = efficientnet.features
        
        # EfficientNet 的 stage 边界（大致）
        # Stage 0: features[0:2]
        # Stage 1: features[2:3]
        # Stage 2: features[3:4]
        # Stage 3: features[4:6]
        # Stage 4: features[6:8]
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """前向传播"""
        outputs = []
        
        # 逐层提取特征
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_indices:
                outputs.append(x)
                
        return outputs


def build_backbone(config) -> nn.Module:
    """
    构建 Backbone
    
    Args:
        config: BackboneConfig 配置对象
        
    Returns:
        Backbone 模块
    """
    if config.type.startswith('resnet'):
        depth = int(config.type.replace('resnet', ''))
        return ResNetBackbone(
            depth=depth,
            pretrained=config.pretrained,
            frozen_stages=config.frozen_stages,
            out_indices=config.out_indices,
        )
    elif config.type.startswith('efficientnet'):
        variant = config.type.replace('efficientnet_', '')
        return EfficientNetBackbone(
            variant=variant,
            pretrained=config.pretrained,
            out_indices=config.out_indices,
        )
    else:
        raise ValueError(f"Unsupported backbone type: {config.type}")


# 测试代码
if __name__ == '__main__':
    # 测试 ResNet Backbone
    print("Testing ResNet50 Backbone...")
    backbone = ResNetBackbone(depth=50, pretrained=False)
    
    x = torch.randn(2, 3, 384, 640)
    outputs = backbone(x)
    
    print(f"Input shape: {x.shape}")
    for i, feat in enumerate(outputs):
        print(f"Output {i} shape: {feat.shape}")
    
    # 预期输出:
    # Output 0 shape: torch.Size([2, 256, 48, 80])   # C3: 1/8
    # Output 1 shape: torch.Size([2, 512, 24, 40])   # C4: 1/16
    # Output 2 shape: torch.Size([2, 1024, 12, 20])  # C5: 1/32
