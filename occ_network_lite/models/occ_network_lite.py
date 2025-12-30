# models/occ_network_lite.py
"""
轻量级 Occupancy Network

优化措施:
1. MobileNetV2 Backbone (3.4M vs ResNet50 25M)
2. 降低 BEV 分辨率 (100×100 vs 200×200)
3. 减少特征维度 (128 vs 256)
4. 减少 Transformer 层数 (2 vs 6)
5. LSS 替代 Cross Attention（可选）
6. 深度可分离卷积

目标: 在 8GB 显存 GPU 上训练 batch_size=2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math


# ============== 轻量级 Backbone ==============

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积：减少参数量和计算量"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
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


class InvertedResidual(nn.Module):
    """MobileNetV2 倒残差块"""
    
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(in_channels * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            # 扩展
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        
        layers.extend([
            # 深度卷积
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # 投影
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2Backbone(nn.Module):
    """
    MobileNetV2 Backbone
    
    参数量: ~3.4M (vs ResNet50 ~25M)
    输出: 多尺度特征 [1/8, 1/16, 1/32]
    """
    
    def __init__(self, pretrained: bool = True, width_mult: float = 1.0):
        super().__init__()
        
        # MobileNetV2 配置
        # t: 扩展因子, c: 输出通道, n: 重复次数, s: stride
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],    # 0
            [6, 24, 2, 2],    # 1 -> 输出 1/4
            [6, 32, 3, 2],    # 2 -> 输出 1/8  ← 输出
            [6, 64, 4, 2],    # 3 -> 输出 1/16 ← 输出
            [6, 96, 3, 1],    # 4
            [6, 160, 3, 2],   # 5 -> 输出 1/32 ← 输出
            [6, 320, 1, 1],   # 6
        ]
        
        # 输出通道 (用于 FPN)
        self.out_channels = [
            int(32 * width_mult),   # 1/8
            int(64 * width_mult),   # 1/16
            int(160 * width_mult),  # 1/32
        ]
        
        # 首层
        input_channel = int(32 * width_mult)
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        )
        
        # 构建倒残差块
        self.layers = nn.ModuleList()
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            layer = []
            for i in range(n):
                stride = s if i == 0 else 1
                layer.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel
            self.layers.append(nn.Sequential(*layer))
        
        # 输出索引 (对应 1/8, 1/16, 1/32)
        self.out_indices = [2, 3, 5]
        
        self._init_weights()
        
        if pretrained:
            self._load_pretrained()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
    def _load_pretrained(self):
        """加载预训练权重 (键名映射版本)"""
        try:
            from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
            pretrained_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

            # 键名映射: torchvision -> custom
            # torchvision: features.0.0 -> custom: first_conv.0
            # torchvision: features.X.conv.Y.Z -> custom: layers.A.B.conv.C
            pretrained_dict = pretrained_model.state_dict()
            model_dict = self.state_dict()

            # 构建映射字典
            mapped_dict = {}

            # 映射 first_conv (features.0)
            for k, v in pretrained_dict.items():
                if k.startswith('features.0.'):
                    new_key = k.replace('features.0.', 'first_conv.')
                    if new_key in model_dict and v.shape == model_dict[new_key].shape:
                        mapped_dict[new_key] = v

            # 映射 layers (features.1-18)
            # torchvision features.1-18 对应 7 个 layer 组
            # 映射关系: features[1] -> layers[0][0], features[2-3] -> layers[1][0-1], etc.
            feature_to_layer = [
                (1, 1, 0, 0),      # features.1 -> layers.0.0
                (2, 3, 1, 0),      # features.2-3 -> layers.1.0-1
                (4, 6, 2, 0),      # features.4-6 -> layers.2.0-2
                (7, 10, 3, 0),     # features.7-10 -> layers.3.0-3
                (11, 13, 4, 0),    # features.11-13 -> layers.4.0-2
                (14, 16, 5, 0),    # features.14-16 -> layers.5.0-2
                (17, 17, 6, 0),    # features.17 -> layers.6.0
            ]

            for feat_start, feat_end, layer_idx, block_start in feature_to_layer:
                block_idx = block_start
                for feat_idx in range(feat_start, feat_end + 1):
                    for k, v in pretrained_dict.items():
                        prefix = f'features.{feat_idx}.'
                        if k.startswith(prefix):
                            # features.X.conv.Y.Z -> layers.A.B.conv.C
                            suffix = k[len(prefix):]
                            new_key = f'layers.{layer_idx}.{block_idx}.{suffix}'
                            if new_key in model_dict and v.shape == model_dict[new_key].shape:
                                mapped_dict[new_key] = v
                    block_idx += 1

            # 更新模型权重
            if mapped_dict:
                model_dict.update(mapped_dict)
                self.load_state_dict(model_dict, strict=False)
                print(f"✅ Loaded {len(mapped_dict)}/{len(model_dict)} pretrained weights")
            else:
                print(f"⚠️  Warning: No matching pretrained weights found")

        except Exception as e:
            print(f"⚠️  Warning: Could not load pretrained weights: {e}")
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            多尺度特征列表 [1/8, 1/16, 1/32]
        """
        outputs = []
        
        x = self.first_conv(x)  # 1/2
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outputs.append(x)
        
        return outputs


# ============== 轻量级 FPN ==============

class LiteFPN(nn.Module):
    """轻量级 FPN，使用深度可分离卷积"""
    
    def __init__(
        self,
        in_channels: List[int],  # [32, 64, 160]
        out_channels: int = 128,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 横向连接 (1x1 卷积统一通道)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels
        ])
        
        # 输出卷积 (深度可分离)
        self.output_convs = nn.ModuleList([
            DepthwiseSeparableConv(out_channels, out_channels)
            for _ in in_channels
        ])
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: 多尺度特征 [C3, C4, C5]
        Returns:
            融合后的特征 (最大分辨率)
        """
        # 横向连接
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        
        # 自顶向下
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i-1] = laterals[i-1] + F.interpolate(
                laterals[i], size=laterals[i-1].shape[-2:], mode='nearest'
            )
        
        # 输出卷积
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        
        # 返回最大分辨率的特征
        return outputs[0]


# ============== 轻量级 View Transformer ==============

class LiteViewTransformer(nn.Module):
    """
    轻量级 View Transformer
    
    使用简化的注意力机制:
    1. 降低 Query 数量 (100×100 vs 200×200)
    2. 减少注意力头和层数
    3. 使用高效的 Key-Value 压缩
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        bev_h: int = 100,
        bev_w: int = 100,
        num_cameras: int = 8,
        feature_h: int = 32,  # 256/8
        feature_w: int = 56,  # 448/8
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_cameras = num_cameras
        
        # Key-Value 压缩: 减少 Key 数量
        # 原始: 8 × 32 × 56 = 14336
        # 压缩后: 8 × 8 × 14 = 896 (减少 16 倍)
        self.kv_compress = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 4, stride=4),  # 4x 下采样
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        
        compressed_h = feature_h // 4
        compressed_w = feature_w // 4
        self.num_kv = num_cameras * compressed_h * compressed_w
        
        # BEV Query (可学习)
        self.bev_query = nn.Parameter(torch.randn(bev_h * bev_w, embed_dim) * 0.02)
        
        # 位置编码
        self.query_pos = nn.Parameter(torch.randn(bev_h * bev_w, embed_dim) * 0.02)
        self.key_pos = nn.Parameter(torch.randn(self.num_kv, embed_dim) * 0.02)
        
        # 简化的 Transformer 层
        self.layers = nn.ModuleList([
            LiteTransformerLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        
    def forward(
        self,
        img_features: torch.Tensor,  # [B, N, C, H, W]
    ) -> torch.Tensor:
        """
        Args:
            img_features: [B, num_cameras, embed_dim, H, W]
        Returns:
            bev_features: [B, embed_dim, bev_h, bev_w]
        """
        B, N, C, H, W = img_features.shape
        device = img_features.device
        
        # 1. 压缩 Key-Value
        # [B, N, C, H, W] -> [B*N, C, H, W] -> [B*N, C, H/4, W/4]
        img_flat = img_features.flatten(0, 1)
        kv_compressed = self.kv_compress(img_flat)
        
        # [B*N, C, H', W'] -> [B, N*H'*W', C]
        kv = kv_compressed.flatten(2).permute(0, 2, 1)  # [B*N, H'*W', C]
        kv = kv.contiguous().view(B, -1, C)  # [B, N*H'*W', C]
        
        # 2. 准备 Query
        query = self.bev_query.unsqueeze(0).expand(B, -1, -1)  # [B, H*W, C]
        
        # 3. Transformer 层
        for layer in self.layers:
            query = layer(
                query=query,
                key=kv,
                value=kv,
                query_pos=self.query_pos,
                key_pos=self.key_pos[:kv.shape[1]],
            )
        
        # 4. 重塑为 BEV
        bev = query.view(B, self.bev_h, self.bev_w, C).permute(0, 3, 1, 2)
        bev = self.output_proj(bev)
        
        return bev


class LiteTransformerLayer(nn.Module):
    """轻量级 Transformer 层"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        # 使用 PyTorch 内置的高效注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # 轻量级 FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, query, key, value, query_pos, key_pos):
        # Cross Attention
        q = query + query_pos
        k = key + key_pos if key_pos is not None else key
        
        # 修复 view/reshape 问题：确保内存连续
        q = q.contiguous()
        k = k.contiguous()
        value = value.contiguous()
        
        attn_out, _ = self.cross_attn(q, k, value)
        query = self.norm1(query + attn_out)
        
        # FFN
        ffn_out = self.ffn(query)
        query = self.norm2(query + ffn_out)
        
        return query


# ============== 轻量级 BEV Encoder ==============

class LiteBEVEncoder(nn.Module):
    """轻量级 BEV 编码器，使用深度可分离卷积"""
    
    def __init__(self, channels: int = 128, num_layers: int = 2):
        super().__init__()
        
        layers = []
        for _ in range(num_layers):
            layers.extend([
                DepthwiseSeparableConv(channels, channels),
                DepthwiseSeparableConv(channels, channels),
            ])
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ============== 轻量级 Occ Decoder ==============

class LiteOccDecoder(nn.Module):
    """
    轻量级 Occupancy 解码器
    
    不使用 3D 卷积，直接用 2D 卷积预测每个高度层
    """
    
    def __init__(
        self,
        in_channels: int = 128,
        num_classes: int = 18,
        num_heights: int = 8,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_heights = num_heights
        
        # 高度预测: 每个高度层的特征
        self.height_pred = nn.Sequential(
            DepthwiseSeparableConv(in_channels, in_channels),
            nn.Conv2d(in_channels, in_channels * num_heights, 1),
        )
        
        # 分类头: 共享的轻量级头
        self.cls_head = nn.Conv2d(in_channels, num_classes, 1)
        
    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev: [B, C, H, W]
        Returns:
            logits: [B, num_classes, H, W, Z]
        """
        B, C, H, W = bev.shape
        
        # 预测每个高度层的特征
        height_features = self.height_pred(bev)
        # [B, C*Z, H, W]
        
        height_features = height_features.view(B, C, self.num_heights, H, W)
        # [B, C, Z, H, W]
        
        # 分类
        logits = []
        for z in range(self.num_heights):
            z_feat = height_features[:, :, z, :, :]  # [B, C, H, W]
            z_logits = self.cls_head(z_feat)  # [B, num_classes, H, W]
            logits.append(z_logits)
        
        logits = torch.stack(logits, dim=-1)  # [B, num_classes, H, W, Z]
        
        return logits


# ============== 完整的轻量级网络 ==============

class OccupancyNetworkLite(nn.Module):
    """
    轻量级 Occupancy Network
    
    预计配置:
    - 参数量: ~15M (vs 原版 ~95M)
    - 显存 (BS=1): ~2-3 GB (vs 原版 ~10 GB)
    - 速度: 快 3-4 倍
    """
    
    def __init__(
        self,
        # 图像配置
        num_cameras: int = 8,
        img_size: Tuple[int, int] = (256, 448),
        
        # 特征维度
        embed_dim: int = 128,
        
        # BEV 配置
        bev_h: int = 100,
        bev_w: int = 100,
        
        # Occupancy 配置
        num_classes: int = 18,
        num_heights: int = 8,
        
        # Transformer 配置
        num_heads: int = 4,
        num_transformer_layers: int = 2,
    ):
        super().__init__()
        
        self.num_cameras = num_cameras
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_classes = num_classes
        
        # 特征图尺寸
        feature_h = img_size[0] // 8
        feature_w = img_size[1] // 8
        
        # 1. MobileNetV2 Backbone
        self.backbone = MobileNetV2Backbone(pretrained=True)
        
        # 2. 轻量级 FPN
        self.neck = LiteFPN(
            in_channels=self.backbone.out_channels,
            out_channels=embed_dim,
        )
        
        # 3. 轻量级 View Transformer
        self.view_transformer = LiteViewTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            bev_h=bev_h,
            bev_w=bev_w,
            num_cameras=num_cameras,
            feature_h=feature_h,
            feature_w=feature_w,
        )
        
        # 4. BEV 编码器
        self.bev_encoder = LiteBEVEncoder(embed_dim, num_layers=2)
        
        # 5. Occupancy 解码器
        self.occ_decoder = LiteOccDecoder(
            in_channels=embed_dim,
            num_classes=num_classes,
            num_heights=num_heights,
        )
        
    def forward(
        self,
        images: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            images: [B, num_cameras, 3, H, W]
            
        Returns:
            dict with 'occ_logits': [B, num_classes, bev_h, bev_w, num_heights]
        """
        B, N, C, H, W = images.shape
        
        # 1. Backbone
        imgs_flat = images.flatten(0, 1)  # [B*N, 3, H, W]
        features = self.backbone(imgs_flat)  # List of multi-scale features
        
        # 2. FPN
        fused_feat = self.neck(features)  # [B*N, embed_dim, H', W']
        
        # 恢复相机维度
        _, C_f, H_f, W_f = fused_feat.shape
        fused_feat = fused_feat.view(B, N, C_f, H_f, W_f)
        
        # 3. View Transformer
        bev = self.view_transformer(fused_feat)  # [B, embed_dim, bev_h, bev_w]
        
        # 4. BEV Encoder
        bev = self.bev_encoder(bev)
        
        # 5. Occ Decoder
        occ_logits = self.occ_decoder(bev)  # [B, num_classes, H, W, Z]
        
        return {
            'occ_logits': occ_logits,
            'bev_features': bev,
        }
    
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """推理接口"""
        outputs = self.forward(images)
        return outputs['occ_logits'].argmax(dim=1)


# ============== 配置 ==============

class LiteConfig:
    """轻量级网络配置"""
    
    # 图像
    num_cameras = 8
    img_size = (256, 448)  # (H, W)
    
    # Backbone
    backbone = 'mobilenetv2'
    
    # 特征
    embed_dim = 128
    num_heads = 4
    num_transformer_layers = 2
    
    # BEV
    bev_h = 100
    bev_w = 100
    
    # Occupancy
    num_classes = 18
    num_heights = 8
    
    # 训练
    batch_size = 2
    lr = 2e-4


def build_lite_model(config: LiteConfig = None) -> OccupancyNetworkLite:
    """构建轻量级模型"""
    if config is None:
        config = LiteConfig()
    
    model = OccupancyNetworkLite(
        num_cameras=config.num_cameras,
        img_size=config.img_size,
        embed_dim=config.embed_dim,
        bev_h=config.bev_h,
        bev_w=config.bev_w,
        num_classes=config.num_classes,
        num_heights=config.num_heights,
        num_heads=config.num_heads,
        num_transformer_layers=config.num_transformer_layers,
    )
    
    return model


# ============== 测试 ==============

if __name__ == '__main__':
    print("Testing OccupancyNetworkLite...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    model = build_lite_model()
    model = model.to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # 测试前向传播
    batch_size = 2
    num_cameras = 8
    img_h, img_w = 256, 448
    
    images = torch.randn(batch_size, num_cameras, 3, img_h, img_w).to(device)
    
    print(f"\nInput shape: {images.shape}")
    
    # 测试显存
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    
    model.eval()
    with torch.no_grad():
        outputs = model(images)
    
    print(f"Output shape: {outputs['occ_logits'].shape}")
    print(f"BEV shape: {outputs['bev_features'].shape}")
    
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nPeak GPU memory (inference): {peak_memory:.2f} GB")
    
    # 测试训练模式
    print("\nTesting training mode...")
    model.train()
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    
    outputs = model(images)
    loss = outputs['occ_logits'].mean()
    loss.backward()
    
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak GPU memory (training): {peak_memory:.2f} GB")
    
    print("\n✓ All tests passed!")
