# models/transformer_occ/transformer_occ_nano_net.py
"""
Transformer Occupancy Network (Nano Version)

专为低显存环境优化 (<4GB)
核心策略：
1. 激进下采样: 快速降低空间分辨率
2. 极短序列: 9600 -> 560 patches
3. 超低分辨率 BEV: 25x25 queries
4. 渐进式上采样: PixelShuffle 恢复分辨率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math
import sys
from pathlib import Path

# 添加项目根目录到 sys.path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

try:
    from models.transformer_occ.position_encoding import CameraPositionEncoding, Spatial2DPositionEncoding
    from models.transformer_occ.encoder import TransformerEncoder
    from models.transformer_occ.voxel_query import BEVQueries
except ImportError:
    from .position_encoding import CameraPositionEncoding, Spatial2DPositionEncoding
    from .encoder import TransformerEncoder
    from .voxel_query import BEVQueries


class NanoBackbone(nn.Module):
    """
    Nano-V2 Backbone
    
    1. PixelUnshuffle: 保留 Bayer 信息
    2. 温和下采样: 3层 CNN, stride=2
    
    输入: [B, 1, 960, 1280]
    输出: [B, 128, 60, 80]
    """
    
    def __init__(self, in_channels: int = 1):
        super().__init__()
        
        # 1. PixelUnshuffle (Bayer -> RGGB)
        # [B, 1, H, W] -> [B, 4, H/2, W/2]
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=2)
        
        # 2. CNN Downsample
        # Input: [B, 4, 480, 640]
        self.layer1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        ) # -> [B, 32, 240, 320]
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ) # -> [B, 64, 120, 160]
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        ) # -> [B, 128, 60, 80]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pixel_unshuffle(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class NanoDecoder(nn.Module):
    """
    极简 Decoder
    
    单层 Cross-Attention
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        # Cross-Attention
        residual = query
        query = self.norm1(query)
        
        # MultiheadAttention forward: (query, key, value)
        attn_out, _ = self.cross_attn(query, key_value, key_value)
        
        query = residual + self.dropout(attn_out)
        
        # FFN
        residual = query
        query = self.norm2(query)
        query = self.ffn(query)
        query = residual + query
        
        return query


class ProgressiveUpsample(nn.Module):
    """
    渐进式上采样 (PixelShuffle) - 自适应版
    """
    
    def __init__(
        self,
        in_dim: int = 128,
        num_classes: int = 18,
        bev_start_size: int = 50,
        target_size: Tuple[int, int, int] = (200, 200, 16),
        start_height: int = 4
    ):
        super().__init__()
        
        self.target_size = target_size
        self.start_height = start_height
        
        # Calculate number of upsample stages needed
        # Assume square BEV
        current_size = bev_start_size
        target_bev_size = target_size[0]
        
        self.layers = nn.ModuleList()
        current_dim = in_dim * start_height
        
        while current_size < target_bev_size:
            # 2x upsample
            self.layers.append(nn.Sequential(
                nn.Conv2d(current_dim, current_dim * 2, kernel_size=3, padding=1),
                nn.PixelShuffle(2), # -> current_dim/2 channels, 2x size
                nn.BatchNorm2d(current_dim // 2),
                nn.ReLU(inplace=True)
            ))
            current_dim = current_dim // 2
            current_size *= 2
            
        self.head = nn.Conv2d(current_dim, num_classes * target_size[2], kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D, H, W, Z_start]
        """
        B, D, H, W, Z = x.shape
        
        # 将 Z 维度合并到 Channel
        # [B, D, H, W, Z] -> [B, D, Z, H, W] -> [B, D*Z, H, W]
        x = x.permute(0, 1, 4, 2, 3).reshape(B, D * Z, H, W)
        
        # 2D 上采样
        for layer in self.layers:
            x = layer(x)
        
        # 预测
        x = self.head(x) # [B, num_classes * Z_target, H_target, W_target]
        
        # 重塑为 3D
        B, C_total, H_target, W_target = x.shape
        Z_target = self.target_size[2]
        num_classes = C_total // Z_target
        
        x = x.reshape(B, num_classes, Z_target, H_target, W_target)
        x = x.permute(0, 1, 3, 4, 2) # [B, C, X, Y, Z]
        
        return x


class TransformerOccNetNano(nn.Module):
    """
    Nano 版本 Occupancy Network
    
    配置:
    - 输入: 8相机 960x1280
    - Embed Dim: 128
    - Encoder: 2层
    - Decoder: 1层
    - BEV Size: 25x25
    """
    
    def __init__(
        self,
        num_cameras: int = 8,
        img_size: Tuple[int, int] = (960, 1280),
        embed_dim: int = 128,
        encoder_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 256,
        bev_size: Tuple[int, int] = (50, 50),
        num_height_levels: int = 4,
        output_grid_size: Tuple[int, int, int] = (200, 200, 16),
        num_classes: int = 18,
        dropout: float = 0.0,
        use_checkpoint: bool = True, # 默认开启 Gradient Checkpointing
    ):
        super().__init__()
        
        self.num_cameras = num_cameras
        self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint
        
        # 1. Nano Backbone (Bayer -> 128ch)
        self.backbone = NanoBackbone(in_channels=1)
        # Output: [B, 128, 60, 80]
        
        # 2. Patch Embedding (Conv 4x4 stride 4)
        # [B, 128, 60, 80] -> [B, 128, 15, 20]
        self.patch_proj = nn.Conv2d(128, embed_dim, kernel_size=4, stride=4)
        # Output: [B, 128, 15, 20] -> 300 patches/cam
        
        # 3. Position Encoding
        # 计算 grid size: 60/4=15, 80/4=20
        self.grid_h = 15
        self.grid_w = 20
        
        self.camera_pe = CameraPositionEncoding(
            embed_dim=embed_dim,
            num_cameras=num_cameras,
            grid_size=(self.grid_h, self.grid_w)
        )
        
        # 4. Encoder (3层 Window Attention)
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_layers=3, # 增加到3层
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            use_window_attn=True, # 开启窗口注意力
            window_size=5, # 窗口大小 5x5
            use_checkpoint=use_checkpoint
        )
        
        # 5. BEV Queries
        self.bev_queries = BEVQueries(
            bev_size=bev_size,
            num_height_levels=num_height_levels,
            embed_dim=embed_dim
        )
        
        # 6. Decoder
        self.decoder = NanoDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 7. Upsample
        self.upsample = ProgressiveUpsample(
            in_dim=embed_dim,
            num_classes=num_classes,
            bev_start_size=bev_size[0],
            target_size=output_grid_size,
            start_height=num_height_levels
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        images: torch.Tensor,
        camera_intrinsics: Optional[torch.Tensor] = None,
        camera_extrinsics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            images: [B, N_cam, 1, H, W]
        """
        B, N_cam, C, H, W = images.shape
        device = images.device
        
        # 顺序处理每个相机以节省显存 (Sequential Processing)
        # 避免一次性分配 [B*8, C, H, W] 的大张量
        feats_list = []
        for i in range(N_cam):
            # 取出单个相机的图像 [B, C, H, W]
            cam_img = images[:, i] 
            
            # 1. 激进下采样 [B, 1, 960, 1280] -> [B, 128, 60, 80]
            feat = self.backbone(cam_img)
            
            # 2. Patch Embedding [B, 128, 60, 80] -> [B, 128, 15, 20]
            feat = self.patch_proj(feat)
            
            # Flatten [B, 128, 15, 20] -> [B, 128, 300] -> [B, 300, 128]
            feat = feat.flatten(2).transpose(1, 2)
            
            feats_list.append(feat)
            
        # [B, 8, 300, 128]
        feat = torch.stack(feats_list, dim=1)
        
        # 3. Position Encoding
        all_pe = []
        for cam_idx in range(N_cam):
            cam_pe = self.camera_pe(cam_idx, B, device=device) # [B, 300, 128]
            all_pe.append(cam_pe)
        
        pe = torch.stack(all_pe, dim=1) # [B, 8, 300, 128]
        feat = feat + pe
        
        # 合并所有 tokens
        # [B, 2400, 128]
        feat = feat.reshape(B, -1, self.embed_dim)
        
        # 4. Encoder
        # 需要传入 H, W 给 Window Attention
        # 视为 H x (W*8) = 15 x 160
        feat = self.encoder(feat, H=self.grid_h, W=self.grid_w * N_cam)
        
        # 5. BEV Queries
        # [B, 2500, 128]
        queries, _, _ = self.bev_queries(B)
        
        # 6. Decoder
        # [B, 2500, 128]
        bev_feat = self.decoder(query=queries, key_value=feat)
        
        # 扩展到 3D (低分辨率)
        # [B, 50, 50, 128, 4]
        feat_3d = self.bev_queries.expand_to_3d(bev_feat)
        
        # 7. Upsample
        occ_logits = self.upsample(feat_3d)
        
        return occ_logits
        
    def get_params_summary(self) -> Dict[str, float]:
        def count_params(module):
            return sum(p.numel() for p in module.parameters()) / 1e6
        return {
            'backbone': count_params(self.backbone),
            'encoder': count_params(self.encoder),
            'bev_queries': count_params(self.bev_queries),
            'decoder': count_params(self.decoder),
            'upsample': count_params(self.upsample),
            'total': count_params(self)
        }


if __name__ == '__main__':
    print("=" * 60)
    print("Transformer Occupancy Network Nano-V2 测试")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = TransformerOccNetNano(
        num_cameras=8,
        img_size=(960, 1280),
        embed_dim=128,
        bev_size=(50, 50),
        num_height_levels=4
    ).to(device)
    
    # 参数量
    params = model.get_params_summary()
    print("\n参数量统计:")
    for name, value in params.items():
        print(f"  {name}: {value:.2f}M")
        
    # 显存测试
    print("\n显存测试 (BS=1):")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
    images = torch.randn(1, 8, 1, 960, 1280, device=device)
    
    with torch.no_grad():
        out = model(images)
        
    print(f"  输入: {images.shape}")
    print(f"  输出: {out.shape}")
    
    if device.type == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"  显存峰值: {peak_mem:.2f} MB")
        
    print("\n✅ 测试通过！")
