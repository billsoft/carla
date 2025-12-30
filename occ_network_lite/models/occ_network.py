# models/occ_network.py
"""
完整的 Occupancy Network

整合所有模块:
1. Backbone: 图像特征提取
2. Neck: 多尺度特征融合
3. Position Encoding: 位置编码
4. View Transformer: 多相机→BEV 变换
5. BEV Encoder: BEV 特征编码
6. Occ Decoder: 3D 体素解码

输入: 8个相机的 RGB 图像
输出: 3D 语义占用网格
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from .backbone import build_backbone, ResNetBackbone
from .neck import FPNNeck
from .positional_encoding import PositionalEncoder
from .view_transformer import ViewTransformer
from .bev_encoder import BEVEncoder
from .occ_decoder import OccDecoderWithUpsample


class OccupancyNetwork(nn.Module):
    """
    完整的 Occupancy Network
    
    数据流:
        8×RGB [B, 8, 3, H, W]
            │
            ▼
        Backbone (per camera)
            │
            ▼
        FPN Neck
            │
            ▼
        View Transformer + Position Encoding
            │
            ▼
        BEV Features [B, C, Hbev, Wbev]
            │
            ▼
        BEV Encoder
            │
            ▼
        Occ Decoder
            │
            ▼
        Occupancy [B, num_classes, H, W, Z]
    """
    
    def __init__(
        self,
        # 图像配置
        num_cameras: int = 8,
        img_size: Tuple[int, int] = (384, 640),
        
        # Backbone 配置
        backbone_type: str = 'resnet50',
        backbone_pretrained: bool = True,
        
        # 特征配置
        embed_dim: int = 256,
        
        # View Transformer 配置
        num_heads: int = 8,
        num_transformer_layers: int = 6,
        
        # BEV 配置
        bev_h: int = 200,
        bev_w: int = 200,
        
        # Occupancy 配置
        num_classes: int = 18,
        num_heights: int = 16,
        full_grid_size: Tuple[int, int, int] = (500, 500, 40),
        
        # 其他配置
        dropout: float = 0.1,
        use_checkpoint: bool = False,  # 梯度检查点
    ):
        super().__init__()
        
        self.num_cameras = num_cameras
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        
        # 计算特征图尺寸（假设 1/8 下采样）
        self.feature_h = img_size[0] // 8
        self.feature_w = img_size[1] // 8
        
        # ========== 1. Backbone ==========
        self.backbone = ResNetBackbone(
            depth=int(backbone_type.replace('resnet', '')),
            pretrained=backbone_pretrained,
            frozen_stages=1,
            out_indices=(1, 2, 3),  # C3, C4, C5
        )
        
        # 获取 backbone 输出通道数
        backbone_channels = [256, 512, 1024]  # ResNet50 的 C3, C4, C5
        
        # ========== 2. FPN Neck ==========
        self.neck = FPNNeck(
            in_channels=backbone_channels,
            out_channels=embed_dim,
            num_outs=1,
        )
        
        # ========== 3. Position Encoding ==========
        self.pos_encoder = PositionalEncoder(
            embed_dim=embed_dim,
            num_cameras=num_cameras,
            bev_h=bev_h,
            bev_w=bev_w,
        )
        
        # ========== 4. View Transformer ==========
        self.view_transformer = ViewTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            ffn_dim=embed_dim * 4,
            dropout=dropout,
            bev_h=bev_h,
            bev_w=bev_w,
            num_cameras=num_cameras,
            feature_h=self.feature_h,
            feature_w=self.feature_w,
        )
        
        # ========== 5. BEV Encoder ==========
        self.bev_encoder = BEVEncoder(
            in_channels=embed_dim,
            out_channels=embed_dim,
            num_layers=4,
        )
        
        # ========== 6. Occ Decoder ==========
        self.occ_decoder = OccDecoderWithUpsample(
            in_channels=embed_dim,
            hidden_channels=embed_dim // 2,
            num_classes=num_classes,
            train_grid_size=(bev_h, bev_w, num_heights),
            full_grid_size=full_grid_size,
            use_3d_conv=True,
        )
        
        # 相机外参占位符（实际使用时从数据集加载）
        self.register_buffer(
            'default_extrinsics',
            torch.eye(4).unsqueeze(0).expand(num_cameras, -1, -1)
        )
        
    def extract_img_features(
        self,
        images: torch.Tensor,  # [B, num_cameras, 3, H, W]
    ) -> torch.Tensor:
        """
        提取图像特征
        
        Args:
            images: [B, N, 3, H, W] 多相机图像
            
        Returns:
            features: [B, N, C, H', W'] 图像特征
        """
        B, N, C, H, W = images.shape
        
        # 合并 batch 和相机维度: [B*N, 3, H, W]
        images_flat = images.flatten(0, 1)
        
        # Backbone 提取特征
        backbone_feats = self.backbone(images_flat)
        # List of [B*N, C_i, H_i, W_i]
        
        # FPN 融合
        fpn_feats = self.neck(backbone_feats)
        # List of [B*N, embed_dim, H', W']
        
        # 取第一个输出（最大分辨率）
        feat = fpn_feats[0]  # [B*N, C, H', W']
        
        # 恢复 batch 和相机维度
        _, C_feat, H_feat, W_feat = feat.shape
        feat = feat.view(B, N, C_feat, H_feat, W_feat)
        
        return feat
    
    def forward(
        self,
        images: torch.Tensor,
        extrinsics: Optional[torch.Tensor] = None,
        upsample: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            images: [B, num_cameras, 3, H, W] 多相机图像
            extrinsics: [num_cameras, 4, 4] 相机外参，如果为 None 使用默认值
            upsample: 是否上采样到完整分辨率
            
        Returns:
            dict with:
                - 'occ_logits': [B, num_classes, H, W, Z] 体素类别 logits
                - 'bev_features': [B, C, Hbev, Wbev] BEV 特征（可选）
        """
        B = images.shape[0]
        device = images.device
        
        # 使用默认外参或传入的外参
        if extrinsics is None:
            extrinsics = self.default_extrinsics
        extrinsics = extrinsics.to(device)
        
        # 1. 提取图像特征
        img_features = self.extract_img_features(images)
        # [B, num_cameras, embed_dim, H', W']
        
        # 2. 生成位置编码
        img_pos_embed = self.pos_encoder.get_image_pos_encoding(
            height=self.feature_h,
            width=self.feature_w,
            extrinsics=extrinsics,
            device=device,
        )
        # [num_cameras, H', W', embed_dim]
        
        # 3. View Transformer: 多相机 → BEV
        bev_features = self.view_transformer(img_features, img_pos_embed)
        # [B, embed_dim, bev_h, bev_w]
        
        # 4. BEV Encoder
        bev_features = self.bev_encoder(bev_features)
        # [B, embed_dim, bev_h, bev_w]
        
        # 5. Occ Decoder: BEV → 3D
        occ_logits = self.occ_decoder(bev_features, upsample=upsample)
        # [B, num_classes, H, W, Z]
        
        return {
            'occ_logits': occ_logits,
            'bev_features': bev_features,
        }
    
    def predict(
        self,
        images: torch.Tensor,
        extrinsics: Optional[torch.Tensor] = None,
        upsample: bool = True,
    ) -> torch.Tensor:
        """
        推理接口
        
        Returns:
            occ_pred: [B, H, W, Z] 体素类别预测 (argmax)
        """
        outputs = self.forward(images, extrinsics, upsample=upsample)
        occ_logits = outputs['occ_logits']
        occ_pred = occ_logits.argmax(dim=1)
        return occ_pred


# OccupancyNetworkLite 已删除
# 原因: 与 occ_network_lite.py 中的完整实现重复
# 请使用 models/occ_network_lite.py 中的轻量级版本


def build_occ_network(config) -> OccupancyNetwork:
    """
    根据配置构建 Occupancy Network
    
    Args:
        config: 配置对象
        
    Returns:
        OccupancyNetwork 实例
    """
    model = OccupancyNetwork(
        num_cameras=config.camera.num_cameras,
        img_size=config.camera.input_size,
        backbone_type=config.backbone.type,
        backbone_pretrained=config.backbone.pretrained,
        embed_dim=config.view_transformer.embed_dim,
        num_heads=config.view_transformer.num_heads,
        num_transformer_layers=config.view_transformer.num_layers,
        bev_h=config.view_transformer.bev_h,
        bev_w=config.view_transformer.bev_w,
        num_classes=config.occupancy.num_classes,
        num_heights=config.occupancy.train_grid_size[2],
        full_grid_size=config.occupancy.full_grid_size,
        dropout=config.view_transformer.dropout,
    )
    
    return model


# 测试代码
if __name__ == '__main__':
    print("Testing Occupancy Network...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 配置
    B = 1
    num_cameras = 8
    img_h, img_w = 384, 640
    
    # 创建模型
    print("\n1. Creating model...")
    model = OccupancyNetwork(
        num_cameras=num_cameras,
        img_size=(img_h, img_w),
        backbone_type='resnet50',
        backbone_pretrained=False,  # 测试时不加载预训练
        embed_dim=256,
        num_heads=8,
        num_transformer_layers=2,  # 减少层数
        bev_h=100,  # 减小 BEV 尺寸
        bev_w=100,
        num_classes=18,
        num_heights=8,
    ).to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params / 1e6:.2f}M")
    print(f"   Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # 测试前向传播
    print("\n2. Testing forward pass...")
    images = torch.randn(B, num_cameras, 3, img_h, img_w).to(device)
    
    print(f"   Input shape: {images.shape}")
    
    model.eval()
    with torch.no_grad():
        outputs = model(images, upsample=False)
    
    occ_logits = outputs['occ_logits']
    bev_features = outputs['bev_features']
    
    print(f"   BEV features shape: {bev_features.shape}")
    print(f"   Occ logits shape: {occ_logits.shape}")
    
    # 测试预测
    print("\n3. Testing prediction...")
    with torch.no_grad():
        occ_pred = model.predict(images, upsample=False)
    
    print(f"   Prediction shape: {occ_pred.shape}")
    print(f"   Unique classes: {torch.unique(occ_pred).tolist()}")
    
    print("\n✓ All tests passed!")
