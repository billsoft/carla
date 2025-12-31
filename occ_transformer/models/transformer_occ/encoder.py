# models/transformer_occ/encoder.py
"""
Transformer Encoder 模块

图像特征编码器，处理多相机图像 patches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .attention import MultiHeadAttention, WindowAttention


class FeedForward(nn.Module):
    """
    前馈网络 (FFN)
    
    FFN(x) = Linear(GELU(Linear(x)))
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class EncoderLayer(nn.Module):
    """
    Transformer Encoder 层
    
    包含:
    1. Self-Attention (窗口注意力或高效注意力)
    2. FFN
    
    使用 Pre-Norm 结构
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.0,
        attention_type: str = 'window',  # 'standard', 'window', 'efficient'
        window_size: int = 8,
    ):
        super().__init__()
        
        self.attention_type = attention_type
        
        # Self-Attention
        if attention_type == 'standard':
            self.self_attn = MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        elif attention_type == 'window':
            self.self_attn = WindowAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                window_size=window_size,
                dropout=dropout
            )
        else:
            # 默认使用窗口注意力
            self.self_attn = WindowAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                window_size=window_size,
                dropout=dropout
            )
            
        # LayerNorm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # FFN
        self.ffn = FeedForward(
            embed_dim=embed_dim,
            hidden_dim=ffn_dim,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        H: Optional[int] = None,
        W: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] 输入序列
            H, W: 空间尺寸（窗口注意力需要）
            
        Returns:
            out: [B, N, D]
        """
        # Self-Attention (Pre-Norm)
        residual = x
        x = self.norm1(x)
        
        if self.attention_type == 'standard':
            x = self.self_attn(x, x, x)
        else:
            x = self.self_attn(x, H, W)
            
        x = self.dropout(x)
        x = residual + x
        
        # FFN (Pre-Norm)
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder
    
    多层 EncoderLayer 堆叠
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.0,
        use_window_attn: bool = True,
        window_size: int = 8,
        grid_size: Tuple[int, int] = None,  # 用于窗口注意力
        num_cameras: int = 8,  # 多相机数量
        use_checkpoint: bool = False,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_cameras = num_cameras
        self.use_checkpoint = use_checkpoint
        
        attention_type = 'window' if use_window_attn else 'standard'
        
        # Encoder 层
        self.layers = nn.ModuleList([
            EncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                attention_type=attention_type,
                window_size=window_size
            )
            for _ in range(num_layers)
        ])
        
        # 最终 LayerNorm
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        H: Optional[int] = None,
        W: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] 输入序列
            H, W: 空间尺寸
            
        Returns:
            out: [B, N, D] 编码后的序列
        """
        import torch.utils.checkpoint as checkpoint
        
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                # Checkpoint requires input to require grad, which x usually does
                x = checkpoint.checkpoint(layer, x, H, W, use_reentrant=False)
            else:
                x = layer(x, H, W)
            
        x = self.norm(x)
        
        return x


class HierarchicalEncoder(nn.Module):
    """
    分层 Encoder
    
    类似 Swin Transformer，逐层下采样
    可以处理更长的序列
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_layers: Tuple[int, ...] = (2, 2, 2),
        num_heads: Tuple[int, ...] = (4, 8, 16),
        window_size: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.num_stages = len(num_layers)
        
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        current_dim = embed_dim
        
        for stage_idx in range(self.num_stages):
            # Encoder 层
            stage = nn.ModuleList([
                EncoderLayer(
                    embed_dim=current_dim,
                    num_heads=num_heads[stage_idx],
                    ffn_dim=current_dim * 4,
                    dropout=dropout,
                    attention_type='window',
                    window_size=window_size
                )
                for _ in range(num_layers[stage_idx])
            ])
            self.stages.append(stage)
            
            # 下采样（除了最后一层）
            if stage_idx < self.num_stages - 1:
                downsample = PatchMerging(current_dim, current_dim * 2)
                self.downsamples.append(downsample)
                current_dim = current_dim * 2
                
        self.norm = nn.LayerNorm(current_dim)
        self.output_dim = current_dim
        
    def forward(
        self,
        x: torch.Tensor,
        H: int,
        W: int,
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: [B, H*W, D]
            H, W: 空间尺寸
            
        Returns:
            out: [B, H'*W', D'] 编码后的序列
            H', W': 新的空间尺寸
        """
        for stage_idx, stage in enumerate(self.stages):
            for layer in stage:
                x = layer(x, H, W)
                
            if stage_idx < self.num_stages - 1:
                x, H, W = self.downsamples[stage_idx](x, H, W)
                
        x = self.norm(x)
        
        return x, H, W


class PatchMerging(nn.Module):
    """
    Patch 合并（下采样）
    
    将 2×2 的相邻 patches 合并为一个
    """
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        
        self.reduction = nn.Linear(4 * in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(4 * in_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        H: int,
        W: int,
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: [B, H*W, D]
            
        Returns:
            out: [B, H/2*W/2, 2D]
        """
        B, N, D = x.shape
        assert N == H * W, f"Input size mismatch: {N} != {H}*{W}"
        
        x = x.view(B, H, W, D)
        
        # Padding
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H = H + pad_h
            W = W + pad_w
            
        # 2×2 合并
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, D]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # [B, H/2, W/2, 4D]
        x = x.view(B, -1, 4 * D)  # [B, H/2*W/2, 4D]
        
        x = self.norm(x)
        x = self.reduction(x)
        
        return x, H // 2, W // 2


class MultiCameraEncoder(nn.Module):
    """
    多相机 Encoder
    
    处理多相机图像，支持相机内和相机间注意力
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.0,
        window_size: int = 8,
        num_cameras: int = 8,
        cross_camera_layers: int = 2,
    ):
        super().__init__()
        
        self.num_cameras = num_cameras
        
        # 相机内 Encoder（共享权重）
        self.intra_encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_layers=num_layers - cross_camera_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            use_window_attn=True,
            window_size=window_size
        )
        
        # 相机间 Encoder（使用窗口注意力处理跨相机）
        self.inter_encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_layers=cross_camera_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            use_window_attn=True,
            window_size=window_size
        )
        
    def forward(
        self,
        x: torch.Tensor,
        camera_ids: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N_total, D] 所有相机的 patches
            camera_ids: [B, N_total] 每个 patch 的相机 ID
            H, W: 单相机的空间尺寸
            
        Returns:
            out: [B, N_total, D]
        """
        B, N_total, D = x.shape
        N_per_cam = H * W
        
        # 1. 相机内编码（逐相机处理）
        encoded_cameras = []
        for cam_idx in range(self.num_cameras):
            # 提取单相机 patches
            start_idx = cam_idx * N_per_cam
            end_idx = start_idx + N_per_cam
            cam_patches = x[:, start_idx:end_idx, :]  # [B, N_per_cam, D]
            
            # 相机内编码
            cam_encoded = self.intra_encoder(cam_patches, H, W)
            encoded_cameras.append(cam_encoded)
            
        # 拼接
        x = torch.cat(encoded_cameras, dim=1)  # [B, N_total, D]
        
        # 2. 相机间编码（全局）
        # 使用 efficient attention 处理跨相机交互
        # 将所有相机视为一个大的空间
        total_H = H
        total_W = W * self.num_cameras
        x = self.inter_encoder(x, total_H, total_W)
        
        return x


if __name__ == '__main__':
    print("=" * 60)
    print("Transformer Encoder 测试")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试单层 Encoder
    print("\n[1] Encoder Layer:")
    layer = EncoderLayer(
        embed_dim=256,
        num_heads=8,
        attention_type='window',
        window_size=8
    ).to(device)
    
    x = torch.randn(2, 60 * 80, 256, device=device)
    out = layer(x, H=60, W=80)
    print(f"  Input: {x.shape}")
    print(f"  Output: {out.shape}")
    
    # 测试完整 Encoder
    print("\n[2] Transformer Encoder:")
    encoder = TransformerEncoder(
        embed_dim=256,
        num_layers=6,
        num_heads=8,
        attention_type='window',
        window_size=8
    ).to(device)
    
    x = torch.randn(2, 60 * 80, 256, device=device)
    out = encoder(x, H=60, W=80)
    print(f"  Input: {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Layers: 6")
    
    # 测试分层 Encoder
    print("\n[3] Hierarchical Encoder:")
    hier_encoder = HierarchicalEncoder(
        embed_dim=256,
        num_layers=(2, 2, 2),
        num_heads=(4, 8, 16),
        window_size=8
    ).to(device)
    
    x = torch.randn(2, 60 * 80, 256, device=device)
    out, H_out, W_out = hier_encoder(x, H=60, W=80)
    print(f"  Input: {x.shape}, H=60, W=80")
    print(f"  Output: {out.shape}, H={H_out}, W={W_out}")
    
    # 测试多相机 Encoder
    print("\n[4] Multi-Camera Encoder:")
    multi_encoder = MultiCameraEncoder(
        embed_dim=256,
        num_layers=6,
        num_heads=8,
        num_cameras=8,
        cross_camera_layers=2
    ).to(device)
    
    x = torch.randn(2, 8 * 60 * 80, 256, device=device)
    camera_ids = torch.arange(8, device=device).repeat_interleave(60 * 80).unsqueeze(0).expand(2, -1)
    out = multi_encoder(x, camera_ids, H=60, W=80)
    print(f"  Input: {x.shape} (8 cameras)")
    print(f"  Output: {out.shape}")
    
    # 参数量
    print("\n参数量:")
    params = sum(p.numel() for p in encoder.parameters())
    print(f"  Transformer Encoder (6 layers): {params/1e6:.2f}M")
    params = sum(p.numel() for p in hier_encoder.parameters())
    print(f"  Hierarchical Encoder: {params/1e6:.2f}M")
    params = sum(p.numel() for p in multi_encoder.parameters())
    print(f"  Multi-Camera Encoder: {params/1e6:.2f}M")
    
    print("\n" + "=" * 60)
    print("✅ 测试通过！")
    print("=" * 60)
