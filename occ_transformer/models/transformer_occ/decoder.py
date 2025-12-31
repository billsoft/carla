# models/transformer_occ/decoder.py
"""
Transformer Decoder 模块

体素解码器，从图像特征中提取 3D 占用信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import Optional, Tuple

from .attention import MultiHeadAttention, DeformableAttention
from .encoder import FeedForward


class DecoderLayer(nn.Module):
    """
    Transformer Decoder 层
    
    包含:
    1. Self-Attention: 体素之间的交互
    2. Cross-Attention: 体素 query → 图像特征
    3. FFN
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.0,
        use_deformable_attn: bool = True,
        num_deform_points: int = 4,
    ):
        super().__init__()
        
        self.use_deformable_attn = use_deformable_attn
        
        # Self-Attention
        self.self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Cross-Attention
        if use_deformable_attn:
            self.cross_attn = DeformableAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_points=num_deform_points,
                dropout=dropout
            )
        else:
            self.cross_attn = MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self.ffn = FeedForward(embed_dim=embed_dim, hidden_dim=ffn_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        memory_spatial_shapes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        if query_pos is not None:
            query_with_pos = query + query_pos
        else:
            query_with_pos = query
            
        # Self-Attention
        residual = query
        query = self.norm1(query)
        query = self.self_attn(query, query, query)
        query = residual + self.dropout(query)
        
        # Cross-Attention
        residual = query
        query = self.norm2(query)
        
        if self.use_deformable_attn:
            query = self.cross_attn(
                query=query_with_pos if query_pos is not None else query,
                reference_points=reference_points,
                value=memory,
                value_spatial_shapes=memory_spatial_shapes
            )
        else:
            query = self.cross_attn(query, memory, memory)
            
        query = residual + self.dropout(query)
        
        # FFN
        residual = query
        query = self.norm3(query)
        query = self.ffn(query)
        query = residual + query
        
        return query


class TransformerDecoder(nn.Module):
    """Transformer Decoder - 多层堆叠"""
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.0,
        use_deformable_attn: bool = True,
        num_deform_points: int = 4,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            DecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                use_deformable_attn=use_deformable_attn,
                num_deform_points=num_deform_points
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        memory_spatial_shapes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        output = query
        
        for layer in self.layers:
            output = layer(
                query=output,
                memory=memory,
                query_pos=query_pos,
                reference_points=reference_points,
                memory_spatial_shapes=memory_spatial_shapes
            )
            
        return self.norm(output)


class VoxelDecoder(nn.Module):
    """
    完整的体素解码器
    
    Transformer Decoder + 上采样 + 预测头
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.0,
        use_deformable_attn: bool = True,
        num_deform_points: int = 4,
        query_grid_size: Tuple[int, int, int] = (50, 50, 8),
        output_grid_size: Tuple[int, int, int] = (200, 200, 16),
        num_classes: int = 18,
    ):
        super().__init__()
        
        self.query_grid_size = query_grid_size
        self.output_grid_size = output_grid_size
        
        self.decoder = TransformerDecoder(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            use_deformable_attn=use_deformable_attn,
            num_deform_points=num_deform_points
        )
        
        # 上采样
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm3d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(embed_dim // 2, embed_dim // 4, kernel_size=(2, 2, 1), stride=(2, 2, 1)),
            nn.BatchNorm3d(embed_dim // 4),
            nn.ReLU(inplace=True),
        )
        
        self.occ_head = nn.Conv3d(embed_dim // 4, num_classes, kernel_size=1)
        
    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        memory_spatial_shapes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        B = query.shape[0]
        X, Y, Z = self.query_grid_size
        
        decoded = self.decoder(
            query=query,
            memory=memory,
            query_pos=query_pos,
            reference_points=reference_points,
            memory_spatial_shapes=memory_spatial_shapes
        )
        
        decoded = decoded.view(B, X, Y, Z, -1).permute(0, 4, 1, 2, 3)
        decoded = self.upsample(decoded)
        occ_logits = self.occ_head(decoded)
        
        return occ_logits


class SimplifiedDecoder(nn.Module):
    """简化版解码器 - 更高效"""
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        query_grid_size: Tuple[int, int, int] = (50, 50, 8),
        output_grid_size: Tuple[int, int, int] = (200, 200, 16),
        num_classes: int = 18,
        use_deformable: bool = True,
    ):
        super().__init__()
        
        self.query_grid_size = query_grid_size
        self.use_deformable = use_deformable
        
        if use_deformable:
            self.cross_attn = DeformableAttention(embed_dim=embed_dim, num_heads=num_heads, num_points=4)
        else:
            self.cross_attn = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
            
        self.norm = nn.LayerNorm(embed_dim)
        
        self.conv3d = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True),
        )
        
        self.head = self._build_upsample_head(embed_dim, num_classes, query_grid_size, output_grid_size)
        
    def _build_upsample_head(self, embed_dim, num_classes, start_size, target_size):
        layers = []
        curr_dim = embed_dim
        curr_size = list(start_size)
        
        # Calculate scale factors
        scale_x = target_size[0] // start_size[0]
        scale_y = target_size[1] // start_size[1]
        scale_z = target_size[2] // start_size[2]
        
        # Iteratively upsample
        while curr_size[0] < target_size[0] or curr_size[1] < target_size[1] or curr_size[2] < target_size[2]:
            sx = 2 if curr_size[0] * 2 <= target_size[0] else 1
            sy = 2 if curr_size[1] * 2 <= target_size[1] else 1
            sz = 2 if curr_size[2] * 2 <= target_size[2] else 1
            
            if sx == 1 and sy == 1 and sz == 1:
                break
                
            next_dim = max(curr_dim // 2, 32)
            layers.extend([
                nn.ConvTranspose3d(
                    curr_dim, next_dim, 
                    kernel_size=(sx, sy, sz), 
                    stride=(sx, sy, sz)
                ),
                nn.BatchNorm3d(next_dim),
                nn.ReLU(inplace=True)
            ])
            curr_dim = next_dim
            curr_size[0] *= sx
            curr_size[1] *= sy
            curr_size[2] *= sz
            
        # Final projection
        layers.append(nn.Conv3d(curr_dim, num_classes, kernel_size=1))
        
        return nn.Sequential(*layers)
        
    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        reference_points: Optional[torch.Tensor] = None,
        memory_spatial_shapes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        B = query.shape[0]
        X, Y, Z = self.query_grid_size
        
        if self.use_deformable:
            x = self.cross_attn(query, reference_points, memory, memory_spatial_shapes)
        else:
            x = self.cross_attn(query, memory, memory)
            
        x = self.norm(x + query)
        x = x.view(B, X, Y, Z, -1).permute(0, 4, 1, 2, 3)
        x = self.conv3d(x) + x
        
        x = self.head(x)
        
        return x


class PixelShuffle3D(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        upscale_factor = self.upscale_factor
        
        out_channels = channels // (upscale_factor ** 3)
        
        input_view = input.view(batch_size, out_channels, upscale_factor, upscale_factor, upscale_factor, in_depth, in_height, in_width)
        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return output.view(batch_size, out_channels, in_depth * upscale_factor, in_height * upscale_factor, in_width * upscale_factor)


class BalancedDecoder(nn.Module):
    """
    Balanced 版本解码器
    - 可变形 Cross-Attention
    - 渐进式 3D 上采样
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        bev_size: Tuple[int, int] = (75, 75),
        num_height_levels: int = 8,
        output_grid_size: Tuple[int, int, int] = (200, 200, 16),
        num_classes: int = 18,
        use_deformable: bool = True,
        num_deform_points: int = 4,
        dropout: float = 0.1,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        
        self.bev_size = bev_size
        self.num_height_levels = num_height_levels
        self.output_grid_size = output_grid_size
        self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint
        
        # Decoder Layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=embed_dim * 4,
                dropout=dropout,
                use_deformable_attn=use_deformable,
                num_deform_points=num_deform_points
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Height Expansion: [B, 5625, 256] -> [B, 256, 75, 75, 8]
        self.height_expand = nn.Linear(embed_dim, embed_dim * num_height_levels)
        
        # 3D Refinement
        self.conv3d = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True),
        )
        
        # Upsample Head: 75×75×8 -> 200×200×16
        self.upsample_head = self._build_upsample_head(
            embed_dim, num_classes,
            (bev_size[0], bev_size[1], num_height_levels),
            output_grid_size
        )
    
    def _build_upsample_head(self, embed_dim, num_classes, start_size, target_size):
        """构建渐进式上采样头 - 使用 PixelShuffle3D"""
        layers = []
        curr_dim = embed_dim
        curr_size = list(start_size)
        
        # 第一次上采样 2×
        # ConvTranspose3d 替代方案: 1x1 Conv + PixelShuffle3D
        layers.extend([
            nn.Conv3d(curr_dim, curr_dim * 8, kernel_size=1), # 扩展通道用于 shuffle
            PixelShuffle3D(upscale_factor=2),
            nn.BatchNorm3d(curr_dim), # shuffle 后通道数变回 curr_dim (256*8 / 8 = 256)
            nn.ReLU(inplace=True)
        ])
        
        # PixelShuffle3D 保持通道数不变 (如果我们把输出通道设为 curr_dim)
        # 上面的实现: input C, output C/8. 
        # 所以我们 input 需要 C*8.
        # Conv3d(curr_dim, curr_dim*8) -> PixelShuffle -> output dim = curr_dim
        
        # 第二次调整到目标尺寸 (保持分辨率不变，只做特征提取)
        layers.extend([
            nn.Conv3d(curr_dim, curr_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(curr_dim),
            nn.ReLU(inplace=True)
        ])
        
        # 分类头
        layers.append(nn.Conv3d(curr_dim, num_classes, kernel_size=1))
        
        return nn.Sequential(*layers)
    
    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        memory_spatial_shapes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        B = query.shape[0]
        H, W = self.bev_size
        
        # 验证 query 形状
        # 如果 query 是 [B, L, C]，则需要确保 L = H*W
        # 有时 query 可能是 [B, H*W, C]
        if query.shape[1] != H * W:
             # 如果不匹配，尝试调整 bev_size 或报错
             # 这里假设我们信任传入的 bev_size，但为了调试，打印警告
             # print(f"Warning: Query length {query.shape[1]} != H*W {H*W} (H={H}, W={W})")
             pass

        # Decoder layers with Checkpointing
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                # 梯度检查点需要输入 requires_grad=True
                query = checkpoint.checkpoint(
                    layer,
                    query,
                    memory,
                    query_pos,
                    reference_points,
                    memory_spatial_shapes,
                    use_reentrant=False
                )
            else:
                query = layer(
                    query=query,
                    memory=memory,
                    query_pos=query_pos,
                    reference_points=reference_points,
                    memory_spatial_shapes=memory_spatial_shapes
                )
        
        query = self.norm(query)  # [B, H*W, 256]
        
        # Height expansion
        x = self.height_expand(query)  # [B, H*W, 256*8]
        
        # View & Permute
        # 关键修改：使用实际的 H, W 进行 view，而不是假设 75x75
        # 确保 H*W 等于 query.shape[1]
        x = x.view(B, H, W, self.embed_dim, self.num_height_levels)
        x = x.permute(0, 3, 1, 2, 4)  # [B, 256, H, W, 8]
        
        # 3D refinement with residual
        x = self.conv3d(x) + x
        
        # Upsample to target size
        x = self.upsample_head(x)  # [B, num_classes, H*2, W*2, 16]
        
        # 最终插值到 200×200
        x = F.interpolate(x, size=self.output_grid_size, mode='trilinear', align_corners=False)
        
        return x  # [B, 18, 200, 200, 16]


if __name__ == '__main__':
    print("=" * 60)
    print("Transformer Decoder 测试")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n[1] Voxel Decoder:")
    voxel_decoder = VoxelDecoder(
        embed_dim=256,
        num_layers=6,
        query_grid_size=(50, 50, 8),
        output_grid_size=(200, 200, 16),
        num_classes=18
    ).to(device)
    
    query = torch.randn(2, 50*50*8, 256, device=device)
    memory = torch.randn(2, 8*60*80, 256, device=device)
    ref_points = torch.rand(2, 50*50*8, 2, device=device)
    spatial_shapes = torch.tensor([[60, 640]], device=device)
    
    occ = voxel_decoder(query, memory, reference_points=ref_points, memory_spatial_shapes=spatial_shapes)
    print(f"  Query: {query.shape}")
    print(f"  Output: {occ.shape}")
    
    params = sum(p.numel() for p in voxel_decoder.parameters())
    print(f"  Params: {params/1e6:.2f}M")
    
    print("\n✅ 测试通过！")
