"""
时序融合模块

1. Ego-Motion 对齐：把上一帧记忆 warp 到当前帧车辆坐标系（同时得到"有效记忆"掩码）
2. 时序注意力：
   - 'deformable'（默认）：对齐后在每个查询所在格子附近做局部 3D 可变形采样（BEVFormer 时序自注意力思路）
   - 'global'：原 5000×5000 全局注意力（SDPA，自动选 FlashAttention）
3. 帧间隔时间嵌入
4. GRU 门控更新记忆
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from .geometry import voxel_centers, metric_to_grid_sample
    from .layers import make_norm, FeedForward
except ImportError:
    from geometry import voxel_centers, metric_to_grid_sample
    from layers import make_norm, FeedForward


class EfficientTemporalAttention(nn.Module):
    """全局时序注意力（Q=当前帧，K=V=对齐后的历史帧）。"""

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, query, key, value, **kwargs):
        B, Q, C = query.shape
        q = self.q_proj(query).view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.o_proj(out.transpose(1, 2).reshape(B, Q, C))


class DeformableTemporalAttention(nn.Module):
    """局部 3D 可变形时序注意力。

    记忆已经按 ego_motion 对齐到当前帧，每个查询只需在自己所在格子附近找对应特征：
    在参考位置（自身格子中心）周围采样 temporal_points 个点（偏移可学习，最大 offset_cells 格），
    每头 softmax 加权。复杂度 O(Q·K)，相比全局注意力 O(Q²) 小两个数量级，
    也不会被远处无关特征干扰。
    """

    def __init__(self, dim, num_heads, num_points, offset_cells, spatial_shape):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_points = num_points
        self.offset_cells = offset_cells
        self.spatial_shape = tuple(spatial_shape)
        self.sampling_offsets = nn.Linear(dim, num_heads * num_points * 3)
        self.attention_weights = nn.Linear(dim, num_heads * num_points)
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.uniform_(self.sampling_offsets.bias, -0.5, 0.5)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)

    def forward(self, query, key, value, ref_norm=None):
        """query: [B,Q,C]；value: 对齐后记忆 [B,Q,C]（与 query 同一网格）；
        ref_norm: [Q,3] 每个查询自身格子中心的 grid_sample 归一化坐标 (x,y,z)。"""
        B, Q, C = query.shape
        X, Y, Z = self.spatial_shape
        h, hd, P = self.num_heads, self.head_dim, self.num_points
        v = self.value_proj(value).view(B, X, Y, Z, h, hd)
        v = v.permute(0, 4, 5, 3, 2, 1).reshape(B * h, hd, Z, Y, X)       # [B*h, hd, D=Z, H=Y, W=X]
        cell = torch.tensor([2.0 / X, 2.0 / Y, 2.0 / Z], device=query.device, dtype=query.dtype)
        off = self.sampling_offsets(query).view(B, Q, h, P, 3).tanh() * self.offset_cells * cell
        loc = ref_norm.view(1, Q, 1, 1, 3).to(off.dtype) + off               # [B, Q, h, P, 3]
        loc = loc.permute(0, 2, 1, 3, 4).reshape(B * h, Q, P, 1, 3).to(v.dtype)
        s = F.grid_sample(v, loc, mode='bilinear', padding_mode='zeros', align_corners=False)  # [B*h, hd, Q, P, 1]
        w = self.attention_weights(query).view(B, Q, h, P).softmax(-1)
        w = w.permute(0, 2, 1, 3).reshape(B * h, 1, Q, P, 1).to(s.dtype)
        out = (s * w).sum(dim=(3, 4)).view(B, h, hd, Q).permute(0, 3, 1, 2).reshape(B, Q, C)
        return self.output_proj(out)


class GRUGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.update_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.reset_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.candidate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Tanh())

    def forward(self, current, memory):
        concat = torch.cat([current, memory], dim=-1)
        z = self.update_gate(concat)
        r = self.reset_gate(concat)
        h_candidate = self.candidate(torch.cat([current, r * memory], dim=-1))
        return (1 - z) * memory + z * h_candidate


class TimeEmbedding(nn.Module):
    def __init__(self, dim, num_freqs=8):
        super().__init__()
        self.register_buffer('freqs', 2.0 ** torch.arange(num_freqs).float(), persistent=False)
        self.mlp = nn.Sequential(nn.Linear(num_freqs * 2, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, dt):
        """dt: [B] 秒 → [B, 1, dim]"""
        x = dt.float().view(-1, 1) * self.freqs.view(1, -1) * 10.0
        return self.mlp(torch.cat([x.sin(), x.cos()], dim=-1)).unsqueeze(1)


class TemporalFusionModule(nn.Module):
    def __init__(self, config, use_checkpoint=True):
        super().__init__()
        dim = config.embed_dim
        self.config = config
        self.use_checkpoint = use_checkpoint
        self.spatial_shape = tuple(config.coarse_size)
        if config.temporal_attn_type == 'deformable':
            self.temporal_attn = DeformableTemporalAttention(
                dim, config.num_heads, config.temporal_points, config.temporal_offset_cells, self.spatial_shape)
        else:
            self.temporal_attn = EfficientTemporalAttention(dim, config.num_heads, config.dropout)
        if config.use_time_embedding:
            self.time_embed = TimeEmbedding(dim)
        self.gate = GRUGate(dim)
        self.mem_out = nn.Linear(dim, dim)
        self.norm_q = make_norm(dim, config.norm_type)
        self.norm_kv = make_norm(dim, config.norm_type)
        self.norm2 = make_norm(dim, config.norm_type)
        self.ffn = FeedForward(dim, 4.0, config.dropout, config.ffn_type)

        centers = voxel_centers(self.spatial_shape, config.voxel_range)          # [Q, 3] 米
        self.register_buffer('centers', centers, persistent=False)
        self.register_buffer('centers_norm', metric_to_grid_sample(centers, config.voxel_range), persistent=False)

    def align_memory(self, memory, ego_motion):
        """把上一帧记忆 warp 到当前帧坐标系。

        memory:     [B, Q, C]，Q = X*Y*Z，按 [X, Y, Z] 行优先展开（上一帧车辆坐标系）
        ego_motion: [B, 4, 4]，上一帧车辆系 → 当前帧车辆系（= inv(pose_t) @ pose_{t-1}）
        返回 aligned [B, Q, C]、valid [B, Q, 1]（该位置在上一帧感知范围内为 1，新进入视野为 0）

        做法：对当前帧每个体素中心 p_t，反查 p_{t-1} = inv(ego_motion) @ p_t，在上一帧记忆
        体积中三线性采样。体积排列成 [B, C, Z, Y, X]，使 grid_sample 的 (x, y, z) 分量
        恰好对应车辆 X/Y/Z 轴（修复了旧实现把 X/Y 对调的问题）。
        """
        B, Q, C = memory.shape
        X, Y, Z = self.spatial_shape
        vol = memory.view(B, X, Y, Z, C).permute(0, 4, 3, 2, 1)                # [B, C, Z, Y, X]
        ones = torch.ones(B, 1, Z, Y, X, device=memory.device, dtype=memory.dtype)
        vol = torch.cat([vol, ones], dim=1)

        T_inv = torch.linalg.inv(ego_motion.float())                             # 当前 → 上一帧
        p_t = self.centers.unsqueeze(0).expand(B, -1, -1)                        # [B, Q, 3]
        p_prev = torch.einsum('bij,bqj->bqi', T_inv[:, :3, :3], p_t) + T_inv[:, None, :3, 3]
        grid = metric_to_grid_sample(p_prev, self.config.voxel_range)            # [B, Q, 3] (x,y,z)
        # 输出体积按 [Z, Y, X] 排列，grid 也要按 [B, Z, Y, X, 3] 组织
        grid = grid.view(B, X, Y, Z, 3).permute(0, 3, 2, 1, 4).to(vol.dtype)
        out = F.grid_sample(vol, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        out = out.permute(0, 4, 3, 2, 1).reshape(B, Q, C + 1)                    # 回到 [X, Y, Z] 展开
        return out[..., :C], out[..., C:]

    def _forward_impl(self, current, memory, ego_motion, time_delta):
        B, Q, C = current.shape
        if memory is None:
            memory = current.new_zeros(B, Q, C)
            valid = current.new_zeros(B, Q, 1)
        elif ego_motion is not None and self.config.use_ego_motion:
            memory, valid = self.align_memory(memory, ego_motion)
        else:
            valid = current.new_ones(B, Q, 1)

        kv = memory
        if self.config.use_time_embedding and time_delta is not None:
            kv = kv + self.time_embed(time_delta).to(kv.dtype)
        q = self.norm_q(current)
        kv = self.norm_kv(kv)
        attn_out = self.temporal_attn(q, kv, kv, ref_norm=self.centers_norm)
        fused = current + attn_out * valid.to(attn_out.dtype)                   # 无有效历史的位置不引入时序项
        fused = fused + self.ffn(self.norm2(fused))
        new_memory = self.gate(fused, memory)
        # 新记忆也经 mem_out（零初始化）回注到当前帧输出：否则 GRU 的输出只在下一帧被用到，
        # 而流式训练（--tbptt 1）每帧 detach 记忆，GRU 门控将永远拿不到梯度
        fused = fused + self.mem_out(new_memory)
        return fused, new_memory

    def forward(self, current, memory=None, ego_motion=None, time_delta=None):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, current, memory, ego_motion, time_delta, use_reentrant=False)
        return self._forward_impl(current, memory, ego_motion, time_delta)
