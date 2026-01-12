import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .attention import CrossAttention, DeformableAttention

class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim=None, drop=0.):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class BEVQueries(nn.Module):
    def __init__(self, bev_h, bev_w, embed_dim):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_queries = nn.Parameter(torch.zeros(bev_h * bev_w, embed_dim))
        self.bev_pos = nn.Parameter(torch.zeros(bev_h * bev_w, embed_dim))
        nn.init.normal_(self.bev_queries, std=0.02)
        nn.init.normal_(self.bev_pos, std=0.02)
        x = torch.linspace(0.5, bev_w - 0.5, bev_w) / bev_w
        y = torch.linspace(0.5, bev_h - 0.5, bev_h) / bev_h
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        self.register_buffer('reference_points', torch.stack([xx.flatten(), yy.flatten()], dim=-1))

    def forward(self, batch_size, device):
        queries = self.bev_queries.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        pos = self.bev_pos.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        ref_points = self.reference_points.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        return queries, pos, ref_points

class DecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, num_points=4, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = CrossAttention(dim, num_heads, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = DeformableAttention(dim, num_heads, num_levels=1, num_points=num_points)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)

    def forward(self, query, query_pos, memory, ref_points, spatial_shapes, level_start_index):
        q = query + query_pos
        query = query + self.self_attn(self.norm1(q), self.norm1(q))
        q = query + query_pos
        query = query + self.cross_attn(self.norm2(q), ref_points, memory, spatial_shapes, level_start_index)
        return query + self.mlp(self.norm3(query))

class BEVDecoder(nn.Module):
    def __init__(self, dim, num_heads, num_layers, bev_h, bev_w, num_points=4, mlp_ratio=4., drop=0., attn_drop=0., use_checkpoint=True):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.use_checkpoint = use_checkpoint
        self.bev_queries = BEVQueries(bev_h, bev_w, dim)
        self.layers = nn.ModuleList([DecoderLayer(dim, num_heads, num_points, mlp_ratio, drop, attn_drop) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, memory, spatial_shapes):
        B = memory.shape[0]
        device = memory.device
        queries, query_pos, ref_points = self.bev_queries(B, device)
        level_start_index = torch.tensor([0], device=device)
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                queries = checkpoint(layer, queries, query_pos, memory, ref_points, spatial_shapes, level_start_index, use_reentrant=False)
            else:
                queries = layer(queries, query_pos, memory, ref_points, spatial_shapes, level_start_index)
        return self.norm(queries).transpose(1, 2).view(B, -1, self.bev_h, self.bev_w)

class CoarseHeightExpansion(nn.Module):
    def __init__(self, dim, num_heights):
        super().__init__()
        self.num_heights = num_heights
        self.expand = nn.Linear(dim, dim * num_heights)

    def forward(self, bev):
        B, C, H, W = bev.shape
        x = bev.flatten(2).transpose(1, 2)
        x = self.expand(x)
        return x.view(B, H, W, self.num_heights, C).permute(0, 4, 1, 2, 3)

class LightweightUpsampler(nn.Module):
    def __init__(self, in_channels, out_channels, target_size, use_checkpoint=True):
        super().__init__()
        self.target_size = target_size
        self.use_checkpoint = use_checkpoint
        self.up = nn.Sequential(nn.Conv3d(in_channels, in_channels // 2, 3, 1, 1, bias=False), nn.BatchNorm3d(in_channels // 2), nn.GELU())
        self.out = nn.Conv3d(in_channels // 2, out_channels, 1)

    def _forward_impl(self, x):
        x = self.up(x)
        if x.shape[2:] != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode='trilinear', align_corners=False)
        return self.out(x)

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)
