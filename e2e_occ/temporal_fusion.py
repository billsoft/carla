"""
时序融合模块 (工业级优化版)
特性:
1. Ego-Motion Alignment: 基于自车运动的特征对齐
2. FlashAttention: 高效注意力计算
3. Checkpointing: 训练显存优化
4. GRU Memory: 循环记忆更新
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class EfficientTemporalAttention(nn.Module):
    """
    基于 FlashAttention 的高效时序注意力
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        
        self.dropout = dropout
        
    def forward(self, query, key, value):
        """
        query: [B, Q, C]
        key:   [B, Q, C]
        value: [B, Q, C]
        """
        B, Q, C = query.shape
        
        # Projections
        q = self.q_proj(query).view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)
        
        # FlashAttention (PyTorch 2.0+)
        # scaled_dot_product_attention 自动选择最优 kernel (FlashAttn, MemEff, etc.)
        output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0
        )
        
        output = output.transpose(1, 2).contiguous().view(B, Q, C)
        return self.o_proj(output)

class GRUGate(nn.Module):
    """
    GRU 门控更新
    """
    def __init__(self, dim):
        super().__init__()
        self.update_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.reset_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.candidate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Tanh())
    
    def forward(self, current, memory):
        concat = torch.cat([current, memory], dim=-1)
        z = self.update_gate(concat)
        r = self.reset_gate(concat)
        
        concat_reset = torch.cat([current, r * memory], dim=-1)
        h_candidate = self.candidate(concat_reset)
        
        return (1 - z) * memory + z * h_candidate

class TemporalFusionModule(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1, use_checkpoint=True, config=None):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.config = config
        
        # Attention & Gate
        self.temporal_attn = EfficientTemporalAttention(dim, num_heads, dropout)
        self.gate = GRUGate(dim)
        
        # Norms & FFN
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout)
        )
        
    def _init_memory(self, B, Q, C, device):
        return torch.zeros(B, Q, C, device=device)
    
    def align_memory(self, memory, ego_motion, spatial_shape):
        """
        memory: [B, Q, C], Q = H*W*D，上一帧（t-1）的体素特征
        ego_motion: [B, 4, 4]，C_{t-1}→C_t 变换矩阵（上一帧体素坐标→当前帧体素坐标）
                    由 train.py 计算：inv(extrinsics_t[:,0]) @ extrinsics_{t-1}[:,0]
                    平移单位：米（与 voxel_range 一致）
        spatial_shape: (H, W, D) 对应体素空间 (X, Y, Z)
        目标：将上一帧 memory 中的特征，warp 到当前帧坐标系下，以便与当前帧特征对齐
        实现：对当前帧体素网格的每个点 p_t，反查其在上一帧中的位置 p_{t-1} = ego_motion_inv @ p_t
        """
        if ego_motion is None:
            return memory

        B, Q, C = memory.shape
        H, W, D = spatial_shape
        device = memory.device

        # 体素空间的真实范围，从 config.voxel_range 读取
        if self.config is not None:
            xmin, ymin, zmin, xmax, ymax, zmax = self.config.voxel_range
        else:
            xmin, ymin, zmin, xmax, ymax, zmax = -40.0, -40.0, -1.0, 40.0, 40.0, 5.4
        x_range = (xmin, xmax)
        y_range = (ymin, ymax)
        z_range = (zmin, zmax)

        # 坐标系转换用的 scale 和 offset（4D 齐次坐标，w 分量保持 1）
        scale = torch.tensor([
            (x_range[1] - x_range[0]) / 2,   # x: [-1,1] -> 40m
            (y_range[1] - y_range[0]) / 2,   # y: [-1,1] -> 40m
            (z_range[1] - z_range[0]) / 2,   # z: [-1,1] -> 3.2m
            1.0
        ], device=device)
        offset = torch.tensor([
            (x_range[0] + x_range[1]) / 2,   # x center = 0m
            (y_range[0] + y_range[1]) / 2,   # y center = 0m
            (z_range[0] + z_range[1]) / 2,   # z center = 2.2m
            0.0
        ], device=device)

        # 1. 重塑为 3D 体积 [B, C, D, H, W]
        # OccDecoder 传入的 memory 是 [B, H*W*D, C]，由 view(B,H,W,D,C) flatten 而来
        mem_vol = memory.view(B, H, W, D, C).permute(0, 4, 3, 1, 2)  # [B, C, D, H, W]

        # 2. 创建归一化 3D 采样网格 [-1, 1]
        # grid_sample 的坐标顺序为 (x, y, z)，对应 (W, H, D)
        zs = torch.linspace(-1, 1, D, device=device)
        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W, device=device)
        grid_d, grid_h, grid_w = torch.meshgrid(zs, ys, xs, indexing='ij')  # [D, H, W]
        ones = torch.ones_like(grid_d)
        grid_homo = torch.stack([grid_w, grid_h, grid_d, ones], dim=-1)      # [D, H, W, 4]
        grid_homo = grid_homo.unsqueeze(0).expand(B, -1, -1, -1, -1)         # [B, D, H, W, 4]
        grid_flat = grid_homo.view(B, -1, 4)                                  # [B, D*H*W, 4]

        # 3. 归一化坐标 -> 世界米坐标
        # ego_motion 的平移单位是米，必须在同一坐标系下做矩阵乘法
        grid_flat_world = grid_flat * scale + offset  # [-1,1] -> 米

        # 4. 反查当前帧体素点 p_t 在上一帧坐标系中的位置
        # ego_motion: C_{t-1}→C_t，故 inv = C_t→C_{t-1}
        # p_{t-1} = inv(ego_motion) @ p_t
        T_inv = torch.linalg.inv(ego_motion)  # [B, 4, 4]
        grid_warped_world = torch.bmm(grid_flat_world, T_inv.transpose(1, 2))  # [B, D*H*W, 4]

        # 5. 世界米坐标 -> 归一化坐标（供 grid_sample 使用）
        grid_warped_norm = (grid_warped_world - offset) / scale                # 米 -> [-1,1]
        grid_warped = grid_warped_norm[..., :3].view(B, D, H, W, 3)           # [B, D, H, W, 3]

        # 6. 3D 采样
        aligned_vol = F.grid_sample(
            mem_vol, grid_warped,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )  # [B, C, D, H, W]

        # 7. 还原为 [B, Q, C]
        # [B, C, D, H, W] -> permute(0,3,4,2,1) -> [B, H, W, D, C] -> reshape -> [B, Q, C]
        aligned = aligned_vol.permute(0, 3, 4, 2, 1).reshape(B, Q, C)

        return aligned

    def _forward_impl(self, current, memory, ego_motion=None, spatial_shape=None):
        # 1. Align Memory (Warping)
        if memory is not None and ego_motion is not None and spatial_shape is not None:
            memory = self.align_memory(memory, ego_motion, spatial_shape)
            
        if memory is None:
            B, Q, C = current.shape
            memory = self._init_memory(B, Q, C, current.device)
            
        # 2. Attention
        q = self.norm1(current)
        k = self.norm1(memory)
        v = k # value is aligned memory
        
        attn_out = self.temporal_attn(q, k, v)
        fused = current + attn_out
        
        # 3. FFN
        fused = fused + self.ffn(self.norm2(fused))
        
        # 4. Gate Update
        new_memory = self.gate(fused, memory)
        
        return fused, new_memory

    def forward(self, current, memory=None, ego_motion=None, spatial_shape=None):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, current, memory, ego_motion, spatial_shape, use_reentrant=False)
        else:
            return self._forward_impl(current, memory, ego_motion, spatial_shape)
