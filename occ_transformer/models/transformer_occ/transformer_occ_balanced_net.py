import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import torch.utils.checkpoint as checkpoint

from .patch_embed import MultiCameraPatchEmbed
from .position_encoding import Spatial2DPositionEncoding, CameraPositionEncoding
from .encoder import TransformerEncoder
from .voxel_query import BEVQueries
from .decoder import BalancedDecoder


class TransformerOccNetBalanced(nn.Module):
    """
    Transformer Occupancy Network - Balanced 版本
    
    特点：
    1. 序列长度可控 (9,600 tokens)
    2. 可变形 Cross-Attention 节省显存
    3. BEV 75×75 精度合理
    4. 3 层 Decoder 平衡深度与效率
    """
    
    def __init__(
        self,
        num_cameras: int = 8,
        img_size: Tuple[int, int] = (960, 1280),
        patch_size: int = 16,
        embed_dim: int = 256,
        encoder_layers: int = 5,
        decoder_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        bev_size: Tuple[int, int] = (50, 50),
        num_height_levels: int = 8,
        num_deform_points: int = 6,
        output_grid_size: Tuple[int, int, int] = (200, 200, 16),
        num_classes: int = 18,
        dropout: float = 0.1,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        
        self.num_cameras = num_cameras
        self.embed_dim = embed_dim
        self.bev_size = bev_size
        self.num_height_levels = num_height_levels
        self.use_checkpoint = use_checkpoint
        
        # 1. Patch Embedding
        self.patch_embed = MultiCameraPatchEmbed(
            num_cameras=num_cameras,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim
        )
        self.grid_h = self.patch_embed.grid_size[0]  # 30
        self.grid_w = self.patch_embed.grid_size[1]  # 40
        
        # 2. Position Encoding
        self.spatial_pe = Spatial2DPositionEncoding(
            grid_size=self.patch_embed.grid_size,
            embed_dim=embed_dim
        )
        self.camera_pe = CameraPositionEncoding(
            embed_dim=embed_dim,
            num_cameras=num_cameras,
            grid_size=self.patch_embed.grid_size
        )
        
        # 3. Transformer Encoder (窗口注意力)
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_layers=encoder_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            use_window_attn=True,
            window_size=8,
            use_checkpoint=use_checkpoint
        )
        
        # 4. BEV Queries
        self.bev_queries = BEVQueries(
            bev_size=bev_size,
            num_height_levels=num_height_levels,
            embed_dim=embed_dim
        )
        
        # 5. Decoder (可变形注意力)
        self.decoder = BalancedDecoder(
            embed_dim=embed_dim,
            num_layers=decoder_layers,
            num_heads=num_heads,
            bev_size=bev_size,
            num_height_levels=num_height_levels,
            output_grid_size=output_grid_size,
            num_classes=num_classes,
            use_deformable=True,
            num_deform_points=num_deform_points,
            dropout=dropout,
            use_checkpoint=use_checkpoint
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
            elif isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                # 3D 卷积初始化优化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
            images: [B, 8, 1, H, W] 多相机 Bayer 图像
        Returns:
            occ_logits: [B, num_classes, X, Y, Z]
        """
        B, N_cam = images.shape[:2]
        device = images.device
        
        # 1. Patch Embedding
        patches, _ = self.patch_embed(images)  # [B, 9600, 256]
        
        # 2. Add Position Encoding
        spatial_pe = self.spatial_pe()
        all_pe = []
        for cam_idx in range(N_cam):
            cam_pe = self.camera_pe(cam_idx, B, device=device)
            combined_pe = spatial_pe.unsqueeze(0).expand(B, -1, -1) + cam_pe
            all_pe.append(combined_pe)
        position_embed = torch.cat(all_pe, dim=1)  # [B, 9600, 256]
        
        patches = patches + position_embed
        
        # 3. Encoder
        total_W = self.grid_w * N_cam  # 40 * 8 = 320
        encoded = self.encoder(patches, H=self.grid_h, W=total_W)
        
        # 4. Get BEV Queries
        queries, query_pos, ref_points = self.bev_queries(B)
        
        # 5. Decoder
        # 注意: memory_spatial_shapes 应该是 [N_levels, 2]，这里只有一层
        spatial_shapes = torch.tensor([[self.grid_h, total_W]], device=device)
        
        occ_logits = self.decoder(
            query=queries,
            memory=encoded,
            query_pos=query_pos,
            reference_points=ref_points,
            memory_spatial_shapes=spatial_shapes
        )
        
        return occ_logits
    
    def get_params_summary(self) -> Dict[str, float]:
        def count_params(module):
            return sum(p.numel() for p in module.parameters()) / 1e6
        
        return {
            'patch_embed': count_params(self.patch_embed),
            'position_encoding': count_params(self.spatial_pe) + count_params(self.camera_pe),
            'encoder': count_params(self.encoder),
            'bev_queries': count_params(self.bev_queries),
            'decoder': count_params(self.decoder),
            'total': count_params(self)
        }
