import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class VoxelHead(nn.Module):
    """稠密体素输出头（交织降维 + 上采样，分类推迟到 400×400 全分辨率）。

        256@80 → 128@80 → 64@80
               → 64→32@200（插值 + 3×3 精化 + 1×1 残差）
               → 32→18@400（插值 + 3×3 精化 + 1×1 残差，直接输出 logits）
    use_flow_head 时在 200×200×32 的 32 通道特征上并联一个 3 通道流场回归分支，
    再三线性上采样到全分辨率。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.embed_dim
        nc = config.num_classes
        self.conv1 = nn.Sequential(nn.Conv3d(dim, dim // 2, 3, padding=1), nn.BatchNorm3d(dim // 2), nn.GELU())
        self.conv2 = nn.Sequential(nn.Conv3d(dim // 2, dim // 4, 3, padding=1), nn.BatchNorm3d(dim // 4), nn.GELU())
        self.refine1 = nn.Sequential(nn.Conv3d(dim // 4, dim // 8, 3, padding=1), nn.BatchNorm3d(dim // 8), nn.GELU())
        self.skip1 = nn.Conv3d(dim // 4, dim // 8, kernel_size=1)
        self.refine2 = nn.Sequential(nn.Conv3d(dim // 8, nc, 3, padding=1), nn.BatchNorm3d(nc))
        self.skip2 = nn.Conv3d(dim // 8, nc, kernel_size=1)
        if config.use_flow_head:
            self.flow = nn.Sequential(
                nn.Conv3d(dim // 8, 16, 3, padding=1), nn.GELU(), nn.Conv3d(16, 3, kernel_size=1))

    def _stage_mid(self, x):
        vx, vy, vz = self.config.voxel_size
        x = self.conv2(self.conv1(x))                                        # [B, 64, 80, 80, 16]
        x = F.interpolate(x, size=(vx // 2, vy // 2, vz), mode='trilinear', align_corners=False)
        return self.refine1(x) + self.skip1(x)                               # [B, 32, 200, 200, 32]

    def _stage_full(self, x):
        x = F.interpolate(x, size=tuple(self.config.voxel_size), mode='trilinear', align_corners=False)
        return self.refine2(x) + self.skip2(x)                               # [B, 18, 400, 400, 32]

    def _stage_flow(self, x):
        return F.interpolate(self.flow(x), size=tuple(self.config.voxel_size), mode='trilinear', align_corners=False)

    def forward(self, x, query_points=None):
        # x: [B, fx, fy, fz, C]
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        # 训练时对各阶段做梯度检查点：400×400×32 全分辨率上的中间激活（插值结果、BN 输出等）
        # 单帧就有数 GB，不保存、反向时重算，用约 1 次额外前向的计算换显存
        ckpt = self.training and self.config.checkpoint_head
        run = (lambda f, t: checkpoint(f, t, use_reentrant=False)) if ckpt else (lambda f, t: f(t))
        x = run(self._stage_mid, x)
        flow = run(self._stage_flow, x) if self.config.use_flow_head else None
        x = run(self._stage_full, x)
        return {'semantic': x, 'flow': flow}


class ImplicitVoxelHead(nn.Module):
    """可查询的隐式输出头（Tesla Occupancy 公开分享中的 "queryable" 解码器思路）。

    低分辨率体素特征 → 3D 卷积降维到 64 通道 → 对任意连续坐标三线性插值取特征 →
    拼接坐标的傅里叶编码 → MLP → 18 类 logits（+ 3 维流场）。

    训练时只对采样点（train.py 按"非空体素 + 随机体素"采样）计算损失，避免在
    400×400×32 上做稠密 3D 卷积；推理时对整个网格分块查询，输出与稠密头相同的形状。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.embed_dim
        hidden = config.implicit_hidden
        self.reduce = nn.Sequential(
            nn.Conv3d(dim, dim // 2, 3, padding=1), nn.BatchNorm3d(dim // 2), nn.GELU(),
            nn.Conv3d(dim // 2, 64, 3, padding=1), nn.BatchNorm3d(64), nn.GELU(),
        )
        self.num_freqs = 6
        in_dim = 64 + 3 + 3 * 2 * self.num_freqs
        self.out_dim = config.num_classes + (3 if config.use_flow_head else 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, self.out_dim),
        )
        self.chunk = 524288

    def _fourier(self, p):
        freqs = (2.0 ** torch.arange(self.num_freqs, device=p.device, dtype=p.dtype)) * torch.pi
        x = p.unsqueeze(-1) * freqs                                            # [B, P, 3, F]
        return torch.cat([p, x.sin().flatten(-2), x.cos().flatten(-2)], dim=-1)

    def query(self, vol, points):
        """vol: [B, 64, Z, Y, X]；points: [B, P, 3] grid_sample 归一化坐标 (x, y, z) → [B, P, out_dim]"""
        B, P, _ = points.shape
        grid = points.view(B, P, 1, 1, 3).to(vol.dtype)
        f = F.grid_sample(vol, grid, mode='bilinear', padding_mode='border', align_corners=False)
        f = f.view(B, 64, P).transpose(1, 2)
        return self.mlp(torch.cat([f, self._fourier(points.to(f.dtype))], dim=-1))

    @staticmethod
    def full_grid_points(voxel_size, device):
        X, Y, Z = voxel_size
        axes = [(torch.arange(n, device=device, dtype=torch.float32) + 0.5) * 2.0 / n - 1.0 for n in (X, Y, Z)]
        gx, gy, gz = torch.meshgrid(*axes, indexing='ij')
        return torch.stack([gx, gy, gz], dim=-1).view(-1, 3)                   # [X*Y*Z, 3]，[X,Y,Z] 行优先

    def forward(self, x, query_points=None):
        vol = self.reduce(x.permute(0, 4, 1, 2, 3).contiguous())               # [B, 64, X, Y, Z]
        vol = vol.permute(0, 1, 4, 3, 2).contiguous()                         # → [B, 64, Z, Y, X]
        nc = self.config.num_classes
        if query_points is not None:
            out = self.query(vol, query_points)                                # [B, P, out_dim]
            return {'semantic_points': out[..., :nc],
                    'flow_points': out[..., nc:] if self.config.use_flow_head else None}
        B = x.shape[0]
        X, Y, Z = self.config.voxel_size
        pts = self.full_grid_points(self.config.voxel_size, x.device)
        outs = [self.query(vol, pts[i:i + self.chunk].unsqueeze(0).expand(B, -1, -1))
                for i in range(0, pts.shape[0], self.chunk)]
        out = torch.cat(outs, dim=1).view(B, X, Y, Z, self.out_dim).permute(0, 4, 1, 2, 3)
        return {'semantic': out[:, :nc], 'flow': out[:, nc:] if self.config.use_flow_head else None}


def build_head(config):
    return ImplicitVoxelHead(config) if config.head_type == 'implicit' else VoxelHead(config)
