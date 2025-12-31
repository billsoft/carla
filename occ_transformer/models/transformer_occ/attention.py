# models/transformer_occ/attention.py
"""
注意力机制模块

1. MultiHeadAttention: 标准多头注意力
2. WindowAttention: 窗口注意力（降低复杂度）
3. DeformableAttention: 可变形注意力（用于 Cross-Attention）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    标准多头注意力
    
    Attention(Q, K, V) = softmax(QK^T / √d) V
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [B, N_q, D]
            key: [B, N_k, D]
            value: [B, N_k, D]
            attn_mask: [B, N_q, N_k] 或 [N_q, N_k]
            
        Returns:
            out: [B, N_q, D]
        """
        B, N_q, _ = query.shape
        N_k = key.shape[1]
        
        # 投影
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # [B, num_heads, N, head_dim]
        
        # 注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N_q, N_k]
        
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn = attn + attn_mask
            
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 加权求和
        out = (attn @ v).transpose(1, 2).reshape(B, N_q, self.embed_dim)
        out = self.out_proj(out)
        
        return out


class WindowAttention(nn.Module):
    """
    窗口注意力 (Swin Transformer 风格)
    
    将序列划分为窗口，只在窗口内计算注意力
    复杂度: O(N * window_size²) 而非 O(N²)
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        window_size: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 相对位置偏置
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        # 计算相对位置索引
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size),
            torch.arange(window_size),
            indexing='ij'
        ))  # [2, ws, ws]
        coords_flatten = coords.flatten(1)  # [2, ws*ws]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, ws*ws, ws*ws]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [ws*ws, ws*ws, 2]
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # [ws*ws, ws*ws]
        self.register_buffer('relative_position_index', relative_position_index)
        
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: [B, H*W, D] 输入序列
            H, W: 空间尺寸
            
        Returns:
            out: [B, H*W, D]
        """
        B, N, C = x.shape
        ws = self.window_size
        
        # 重塑为 2D
        x = x.view(B, H, W, C)
        
        # Padding 使得 H, W 能被 window_size 整除
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        Hp, Wp = H + pad_h, W + pad_w
        
        # 划分窗口 [B, H, W, C] -> [B*num_windows, ws, ws, C]
        x = x.view(B, Hp // ws, ws, Wp // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(-1, ws * ws, C)  # [B*num_windows, ws*ws, C]
        
        # QKV
        qkv = self.qkv(x).reshape(-1, ws * ws, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*nw, num_heads, ws*ws, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # 相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(ws * ws, ws * ws, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)
        attn = attn + relative_position_bias
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 输出
        x = (attn @ v).transpose(1, 2).reshape(-1, ws * ws, C)
        x = self.proj(x)
        
        # 还原窗口
        num_windows = (Hp // ws) * (Wp // ws)
        x = x.view(B, Hp // ws, Wp // ws, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, Hp, Wp, C)
        
        # 去除 padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :]
            
        x = x.view(B, H * W, C)
        
        return x


class DeformableAttention(nn.Module):
    """
    可变形注意力
    
    每个 query 只关注 K 个可学习的参考点位置
    复杂度: O(N_q * K) 而非 O(N_q * N_k)
    
    适用于 Cross-Attention: 体素 query → 图像 patches
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 1,
        num_points: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads
        
        # 采样点偏移量预测
        self.sampling_offsets = nn.Linear(
            embed_dim,
            num_heads * num_levels * num_points * 2  # 2D 偏移
        )
        
        # 注意力权重预测
        self.attention_weights = nn.Linear(
            embed_dim,
            num_heads * num_levels * num_points
        )
        
        # Value 投影
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        nn.init.xavier_uniform_(self.attention_weights.weight)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)
        
    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        value_spatial_shapes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: [B, N_q, D] 查询向量
            reference_points: [B, N_q, 2] 参考点坐标 (归一化到 [0, 1])
            value: [B, N_v, D] 值向量
            value_spatial_shapes: [num_levels, 2] 每层的空间尺寸
            
        Returns:
            out: [B, N_q, D]
        """
        B, N_q, _ = query.shape
        N_v = value.shape[1]
        
        # Value 投影
        value = self.value_proj(value)
        value = value.view(B, N_v, self.num_heads, self.head_dim)
        
        # 预测采样偏移
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            B, N_q, self.num_heads, self.num_levels, self.num_points, 2
        )
        
        # 预测注意力权重
        attention_weights = self.attention_weights(query)
        attention_weights = attention_weights.view(
            B, N_q, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.view(
            B, N_q, self.num_heads, self.num_levels, self.num_points
        )
        
        # 计算采样位置
        # reference_points: [B, N_q, 2] -> [B, N_q, 1, 1, 1, 2]
        reference_points = reference_points.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        
        # 偏移量缩放
        offset_normalizer = torch.tensor(
            [value_spatial_shapes[0, 1], value_spatial_shapes[0, 0]],
            device=query.device, dtype=torch.float32
        )
        sampling_locations = reference_points + sampling_offsets / offset_normalizer
        
        # 双线性插值采样
        # 简化版：使用 grid_sample
        output = self._sample_and_aggregate(
            value, sampling_locations, attention_weights, value_spatial_shapes
        )
        
        output = self.output_proj(output)
        
        return output
    
    def _sample_and_aggregate(
        self,
        value: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
        value_spatial_shapes: torch.Tensor
    ) -> torch.Tensor:
        """
        采样和聚合 (简化实现)
        
        Args:
            value: [B, N_v, num_heads, head_dim]
            sampling_locations: [B, N_q, num_heads, num_levels, num_points, 2]
            attention_weights: [B, N_q, num_heads, num_levels, num_points]
            value_spatial_shapes: [num_levels, 2]
            
        Returns:
            output: [B, N_q, C]
        """
        B, N_v, num_heads, head_dim = value.shape
        _, N_q, _, num_levels, num_points, _ = sampling_locations.shape
        
        # 1. 准备 Value: [B*heads, head_dim, H, W]
        H, W = value_spatial_shapes[0]
        # value: [B, H*W, heads, dim] -> [B, heads, dim, H, W]
        # 使用 reshape 替代 view 以避免 stride 问题
        value = value.reshape(B, H, W, num_heads, head_dim).permute(0, 3, 4, 1, 2)
        value = value.reshape(B * num_heads, head_dim, H, W)
        
        # 2. 准备 Sampling Locations: [B*heads, N_q*points, 1, 2]
        # locs: [B, N_q, heads, levels, points, 2] -> [B, heads, N_q, points, 2] (assume levels=1)
        locs = sampling_locations.squeeze(3).permute(0, 2, 1, 3, 4)
        locs = locs.reshape(B * num_heads, N_q * num_points, 1, 2)
        
        # 转换坐标范围 [0, 1] -> [-1, 1]
        locs = locs * 2.0 - 1.0
        
        # 3. Grid Sample
        # output: [B*heads, head_dim, N_q*points, 1]
        sampled_value = F.grid_sample(
            value, 
            locs, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=False
        )
        
        # 4. 重塑回 [B, N_q, heads, points, head_dim]
        # [B*heads, dim, N_q*points, 1] -> [B, heads, dim, N_q, points]
        sampled_value = sampled_value.view(B, num_heads, head_dim, N_q, num_points)
        # -> [B, N_q, heads, points, dim]
        sampled_value = sampled_value.permute(0, 3, 1, 4, 2)
        
        # 5. 加权聚合
        # weights: [B, N_q, heads, levels, points] -> [B, N_q, heads, points]
        weights = attention_weights.squeeze(3).unsqueeze(-1) # [B, N_q, heads, points, 1]
        
        output = (sampled_value * weights).sum(dim=3)  # [B, N_q, heads, dim]
        
        # 6. Flatten heads
        output = output.flatten(2)  # [B, N_q, heads*dim]
        
        return output


class EfficientAttention(nn.Module):
    """
    高效注意力 - 组合使用窗口注意力和全局注意力
    
    用于处理长序列
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        window_size: int = 8,
        use_global: bool = True,
        global_ratio: float = 0.1,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.use_global = use_global
        
        # 窗口注意力
        self.window_attn = WindowAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout
        )
        
        if use_global:
            # 全局注意力 (使用下采样的 tokens)
            self.global_attn = MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            self.global_ratio = global_ratio
            
            # 用于下采样的卷积
            self.downsample = nn.Conv2d(
                embed_dim, embed_dim,
                kernel_size=3, stride=2, padding=1
            )
            
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: [B, H*W, D]
            H, W: 空间尺寸
        """
        # 窗口注意力
        x_window = self.window_attn(x, H, W)
        
        if self.use_global:
            # 下采样进行全局注意力
            B, N, D = x.shape
            x_2d = x.view(B, H, W, D).permute(0, 3, 1, 2)  # [B, D, H, W]
            x_down = self.downsample(x_2d)  # [B, D, H/2, W/2]
            x_down = x_down.flatten(2).transpose(1, 2)  # [B, H/2*W/2, D]
            
            # 全局注意力
            x_global = self.global_attn(x_down, x_down, x_down)
            
            # 上采样回原尺寸
            H_down, W_down = H // 2, W // 2
            x_global = x_global.view(B, H_down, W_down, D).permute(0, 3, 1, 2)
            x_global = F.interpolate(x_global, size=(H, W), mode='bilinear', align_corners=False)
            x_global = x_global.flatten(2).transpose(1, 2)  # [B, H*W, D]
            
            # 融合
            x = x_window + self.global_ratio * x_global
        else:
            x = x_window
            
        return x


if __name__ == '__main__':
    print("=" * 60)
    print("注意力模块测试")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试标准多头注意力
    print("\n[1] 标准多头注意力:")
    mha = MultiHeadAttention(embed_dim=256, num_heads=8).to(device)
    q = torch.randn(2, 100, 256, device=device)
    k = v = torch.randn(2, 200, 256, device=device)
    out = mha(q, k, v)
    print(f"  Query: {q.shape}, Key/Value: {k.shape}")
    print(f"  Output: {out.shape}")
    
    # 测试窗口注意力
    print("\n[2] 窗口注意力:")
    window_attn = WindowAttention(embed_dim=256, num_heads=8, window_size=8).to(device)
    x = torch.randn(2, 60 * 80, 256, device=device)
    out = window_attn(x, H=60, W=80)
    print(f"  Input: {x.shape}, H=60, W=80")
    print(f"  Output: {out.shape}")
    
    # 测试可变形注意力
    print("\n[3] 可变形注意力:")
    deform_attn = DeformableAttention(
        embed_dim=256, num_heads=8, num_points=4
    ).to(device)
    query = torch.randn(2, 1000, 256, device=device)
    ref_points = torch.rand(2, 1000, 2, device=device)
    value = torch.randn(2, 60 * 80, 256, device=device)
    spatial_shapes = torch.tensor([[60, 80]], device=device)
    out = deform_attn(query, ref_points, value, spatial_shapes)
    print(f"  Query: {query.shape}")
    print(f"  Reference Points: {ref_points.shape}")
    print(f"  Value: {value.shape}")
    print(f"  Output: {out.shape}")
    
    # 测试高效注意力
    print("\n[4] 高效注意力:")
    efficient_attn = EfficientAttention(
        embed_dim=256, num_heads=8, window_size=8, use_global=True
    ).to(device)
    x = torch.randn(2, 60 * 80, 256, device=device)
    out = efficient_attn(x, H=60, W=80)
    print(f"  Input: {x.shape}")
    print(f"  Output: {out.shape}")
    
    # 参数量
    print("\n参数量:")
    for name, module in [('MHA', mha), ('Window', window_attn), ('Deformable', deform_attn)]:
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {params/1e3:.1f}K")
    
    print("\n" + "=" * 60)
    print("✅ 测试通过！")
    print("=" * 60)
