import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from .position_encoding import SineCosinePositionEncoding3D
    from .deformable_attention import DeformableDecoderLayer
    from .temporal_fusion import TemporalFusionModule
    from .geometry import voxel_centers
    from .layers import FeedForward, make_norm
except ImportError:
    from position_encoding import SineCosinePositionEncoding3D
    from deformable_attention import DeformableDecoderLayer
    from temporal_fusion import TemporalFusionModule
    from geometry import voxel_centers
    from layers import FeedForward, make_norm


class OccupancyDecoder(nn.Module):
    """粗 → 时序融合 → 细 两阶段 3D 查询解码器。

    查询点 = 体素中心（车辆坐标系，米，由 config.voxel_range 计算），不再是 [0,1] 边界网格。
    稀疏模式（use_sparse_fine）：粗阶段额外预测每格的占用 logit，细阶段只对上采样后
    占用概率最高的 fine_keep_ratio 比例细体素做交叉注意力，其余保留粗特征上采样的初值，
    最后 scatter 回稠密网格。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.embed_dim
        self.coarse_query = nn.Parameter(torch.randn(1, config.num_coarse_queries, dim) * 0.02)
        self.pos_3d = SineCosinePositionEncoding3D(dim)

        self.coarse_layers = nn.ModuleList([
            DeformableDecoderLayer(config, use_self_attn=config.use_self_attention)
            for _ in range(config.decoder_layers)
        ])
        self.coarse_to_fine = nn.Sequential(
            make_norm(dim, config.norm_type),
            FeedForward(dim, 2.0, config.dropout, config.ffn_type),
        )
        self.coarse_to_fine_norm = make_norm(dim, config.norm_type)
        self.fine_layers = nn.ModuleList([
            DeformableDecoderLayer(config, use_self_attn=config.use_fine_self_attention)
            for _ in range(config.decoder_layers)
        ])

        self.register_buffer('coarse_pts', voxel_centers(config.coarse_size, config.voxel_range), persistent=False)
        self.register_buffer('fine_pts', voxel_centers(config.fine_size, config.voxel_range), persistent=False)
        # 'mixed' 选点的近场分数：离自车越近越高
        self.register_buffer('fine_near', -self.fine_pts[:, :2].norm(dim=-1), persistent=False)

        if config.use_temporal:
            self.temporal_fusion = TemporalFusionModule(config, use_checkpoint=True)

        # 粗阶段占用概率：稀疏细化的选择依据（同时作为辅助监督）
        self.coarse_occ_head = nn.Sequential(
            make_norm(dim, config.norm_type), nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))

        self.fine_spatial_conv = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm3d(dim),
            nn.GELU(),
        )
        self.checkpoint_coarse = False
        self.checkpoint_fine = True

    def _select_fine(self, score, K):
        """score: [B, Qf] 细体素上插值得到的粗占用 logit → 被细化的 K 个索引 [B, K]。"""
        if self.config.fine_select_mode == 'topk':
            return score.topk(K, dim=1).indices
        q_occ, q_unc, _ = self.config.fine_select_quota
        k_occ, k_unc = int(K * q_occ), int(K * q_unc)
        k_near = K - k_occ - k_unc
        taken = torch.zeros_like(score, dtype=torch.bool)
        picks = []
        for key, k in ((score, k_occ), (-score.abs(), k_unc),
                       (self.fine_near.unsqueeze(0).expand_as(score), k_near)):
            if k <= 0:
                continue
            idx = key.masked_fill(taken, float('-inf')).topk(k, dim=1).indices
            taken.scatter_(1, idx, True)
            picks.append(idx)
        return torch.cat(picks, dim=1)

    def _run_layers(self, layers, query, pts, image_feats, intrinsics, extrinsics, use_ckpt):
        for layer in layers:
            if use_ckpt and self.training:
                query = checkpoint(layer, query, pts, image_feats, intrinsics, extrinsics, use_reentrant=False)
            else:
                query = layer(query, pts, image_feats, intrinsics, extrinsics)
        return query

    def forward(self, image_feats, intrinsics, extrinsics, memory=None, ego_motion=None, time_delta=None):
        """
        image_feats: {stride: [B, N, C, H, W]}
        返回 dict:
          fine_feats        [B, fx, fy, fz, C]
          memory            [B, 5000, C] 或 None
          coarse_occ_logits [B, cx, cy, cz]
          fine_selected     [B, K] 稀疏模式下被细化的细体素索引（否则 None）
        """
        B = next(iter(image_feats.values())).shape[0]
        device = self.coarse_query.device
        cx, cy, cz = self.config.coarse_size
        dim = self.config.embed_dim

        # ---- 粗阶段 ----
        query = self.coarse_query.expand(B, -1, -1) + self.pos_3d(cx, cy, cz, device).unsqueeze(0)
        pts = self.coarse_pts.unsqueeze(0).expand(B, -1, -1)
        query = self._run_layers(self.coarse_layers, query, pts, image_feats, intrinsics, extrinsics,
                                 self.checkpoint_coarse)

        # ---- 时序融合 ----
        new_memory = None
        if self.config.use_temporal:
            query, new_memory = self.temporal_fusion(query, memory, ego_motion=ego_motion, time_delta=time_delta)

        coarse_occ_logits = self.coarse_occ_head(query).view(B, cx, cy, cz)
        coarse_feats = query.view(B, cx, cy, cz, dim).permute(0, 4, 1, 2, 3)

        # ---- 细阶段初值 ----
        fx, fy, fz = self.config.fine_size
        fine = F.interpolate(coarse_feats, size=(fx, fy, fz), mode='trilinear', align_corners=False)
        fine = fine.permute(0, 2, 3, 4, 1).reshape(B, -1, dim)
        fine = self.coarse_to_fine_norm(fine + self.coarse_to_fine(fine))
        fine = fine + self.pos_3d(fx, fy, fz, device).unsqueeze(0)
        fine_pts = self.fine_pts.unsqueeze(0).expand(B, -1, -1)

        selected = None
        if self.config.use_sparse_fine:
            Qf = fine.shape[1]
            K = max(1, int(Qf * self.config.fine_keep_ratio))
            score = F.interpolate(coarse_occ_logits.detach().unsqueeze(1).float(), size=(fx, fy, fz),
                                  mode='trilinear', align_corners=False).view(B, -1)
            selected = self._select_fine(score, K)                                     # [B, K]
            idx_c = selected.unsqueeze(-1).expand(-1, -1, dim)
            q_sel = fine.gather(1, idx_c)
            p_sel = fine_pts.gather(1, selected.unsqueeze(-1).expand(-1, -1, 3))
            q_sel = self._run_layers(self.fine_layers, q_sel, p_sel, image_feats, intrinsics, extrinsics,
                                     self.checkpoint_fine)
            fine = fine.scatter(1, idx_c, q_sel.to(fine.dtype))
        else:
            fine = self._run_layers(self.fine_layers, fine, fine_pts, image_feats, intrinsics, extrinsics,
                                    self.checkpoint_fine)

        # ---- 空间一致性（Depthwise Conv3D 残差）----
        vol = fine.view(B, fx, fy, fz, dim).permute(0, 4, 1, 2, 3).contiguous()
        vol = vol + self.fine_spatial_conv(vol)
        return {
            'fine_feats': vol.permute(0, 2, 3, 4, 1),
            'memory': new_memory,
            'coarse_occ_logits': coarse_occ_logits,
            'fine_selected': selected,
        }
