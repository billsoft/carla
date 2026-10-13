import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .geometry import equidistant_project, image_to_grid_sample
    from .layers import make_norm, FeedForward, MultiHeadSelfAttention
except ImportError:
    from geometry import equidistant_project, image_to_grid_sample
    from layers import make_norm, FeedForward, MultiHeadSelfAttention


def get_reference_points(points, intrinsics, extrinsics, image_hw):
    """3D 点（车辆系，米）→ 各相机的 grid_sample 归一化参考点 + 可见性掩码。

    points:     [B, Q, 3]
    intrinsics: [B, N, 3, 3]（原图分辨率）
    extrinsics: [B, N, 4, 4]，Camera(OpenCV)→Vehicle
    返回 ref [B, N, Q, 2]（归一化，层级无关），valid [B, N, Q] bool

    等距投影正投影，和 position_encoding.pixel_rays_vehicle（反投影）共用 geometry.py。
    """
    R = extrinsics[..., :3, :3]                                       # [B, N, 3, 3]
    t = extrinsics[..., :3, 3]                                        # [B, N, 3]
    # Vehicle → Camera: p_cam = Rᵀ (p - t)
    diff = points.unsqueeze(1) - t.unsqueeze(2)                       # [B, N, Q, 3]
    p_cam = torch.einsum('bnji,bnqj->bnqi', R, diff)
    fx = intrinsics[..., 0, 0, None]
    fy = intrinsics[..., 1, 1, None]
    cx = intrinsics[..., 0, 2, None]
    cy = intrinsics[..., 1, 2, None]
    u, v, _ = equidistant_project(p_cam, fx, fy, cx, cy)
    un, vn = image_to_grid_sample(u, v, image_hw)
    valid = (un.abs() <= 1.0) & (vn.abs() <= 1.0)
    return torch.stack([un, vn], dim=-1), valid


class DeformableCrossAttention(nn.Module):
    """多相机、多尺度可变形交叉注意力（3D 查询 → 2D 图像特征）。

    对每个查询、每个相机、每个头、每个层级，在几何参考点附近采样 num_points 个点：
        loc = ref + tanh(Linear(q)) · offset_px（换算到该层级的归一化坐标）
    同一 (相机, 头) 内对 层级×采样点 做 softmax 加权。
    cam_visibility_norm=True 时，投影落在图像外的相机贡献置零，并按命中相机数取平均
    （BEVFormer 做法），避免特征幅值随"被几个相机看到"而变化。
    """

    def __init__(self, dim, num_heads, num_cameras, strides, image_hw, num_points=4,
                 offset_px=4.0, dropout=0.1, visibility_norm=True, query_chunk=32768):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_cameras = num_cameras
        self.strides = tuple(strides)
        self.num_levels = len(self.strides)
        self.num_points = num_points
        self.head_dim = dim // num_heads
        self.image_hw = tuple(image_hw)
        self.offset_px = offset_px
        self.visibility_norm = visibility_norm
        self.query_chunk = query_chunk

        n_off = num_cameras * num_heads * self.num_levels * num_points
        self.sampling_offsets = nn.Linear(dim, n_off * 2)
        self.attention_weights = nn.Linear(dim, n_off)
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        # 偏移权重为 0：训练初期严格按几何参考点采样；bias 让各头/各点初始散开成一个小环
        # （Deformable DETR 的做法），避免所有采样点重合。
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        N, H, L, P = self.num_cameras, self.num_heads, self.num_levels, self.num_points
        ang = torch.arange(H, dtype=torch.float32) * (2.0 * math.pi / H)
        grid = torch.stack([ang.cos(), ang.sin()], -1)                  # [H, 2]
        radius = (torch.arange(P, dtype=torch.float32) + 1.0) * 0.5     # 0.5,1,1.5,2 px
        frac = (radius / self.offset_px).clamp(max=0.9)
        bias = grid.view(1, H, 1, 1, 2) * torch.atanh(frac).view(1, 1, 1, P, 1)
        bias = bias.expand(N, H, L, P, 2).contiguous()
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(bias.view(-1))
        nn.init.xavier_uniform_(self.attention_weights.weight)
        nn.init.constant_(self.attention_weights.bias, 0.0)

    def forward(self, query, points, image_feats, intrinsics, extrinsics):
        """
        query:       [B, Q, C]
        points:      [B, Q, 3]  查询点车辆系坐标（米）
        image_feats: {stride: [B, N, C, H_s, W_s]}
        """
        B, Q, C = query.shape
        N, Hh, L, P, hd = self.num_cameras, self.num_heads, self.num_levels, self.num_points, self.head_dim

        ref, valid = get_reference_points(points, intrinsics, extrinsics, self.image_hw)

        off = self.sampling_offsets(query).view(B, Q, N, Hh, L, P, 2).tanh() * self.offset_px
        scale = torch.tensor([[2.0 / image_feats[s].shape[-1], 2.0 / image_feats[s].shape[-2]]
                              for s in self.strides], device=query.device, dtype=off.dtype)
        off = off * scale.view(1, 1, 1, 1, L, 1, 2)                     # 层级像素 → 归一化坐标
        w = self.attention_weights(query).view(B, Q, N, Hh, L * P).softmax(-1).view(B, Q, N, Hh, L, P)

        values = []
        for s in self.strides:
            f = image_feats[s]                                            # [B, N, C, H, W]
            Hs, Ws = f.shape[-2:]
            v = self.value_proj(f.permute(0, 1, 3, 4, 2))                 # [B, N, H, W, C]
            v = v.view(B, N, Hs, Ws, Hh, hd).permute(0, 1, 4, 5, 2, 3)  # [B, N, h, hd, H, W]
            values.append(v)

        output = query.new_zeros(B, Hh, hd, Q)
        for cam in range(N):
            out_cam = query.new_zeros(B, Hh, hd, Q)
            for q0 in range(0, Q, self.query_chunk):
                q1 = min(Q, q0 + self.query_chunk)
                Qc = q1 - q0
                ref_c = ref[:, cam, q0:q1]                                # [B, Qc, 2]
                for li in range(L):
                    v = values[li][:, cam].reshape(B * Hh, hd, *values[li].shape[-2:])
                    loc = ref_c.view(B, Qc, 1, 1, 2) + off[:, q0:q1, cam, :, li]   # [B, Qc, h, P, 2]
                    loc = loc.permute(0, 2, 1, 3, 4).reshape(B * Hh, Qc, P, 2).to(v.dtype)
                    s_ = F.grid_sample(v, loc, mode='bilinear', padding_mode='zeros', align_corners=False)
                    wl = w[:, q0:q1, cam, :, li].permute(0, 2, 1, 3).reshape(B * Hh, 1, Qc, P)
                    out_cam[..., q0:q1] += (s_ * wl.to(s_.dtype)).sum(-1).view(B, Hh, hd, Qc).to(out_cam.dtype)
            if self.visibility_norm:
                out_cam = out_cam * valid[:, cam].view(B, 1, 1, Q).to(out_cam.dtype)
            output = output + out_cam

        if self.visibility_norm:
            count = valid.sum(dim=1).clamp(min=1).to(output.dtype)        # [B, Q]
            output = output / count.view(B, 1, 1, Q)
        output = output.permute(0, 3, 1, 2).reshape(B, Q, C)
        return self.dropout(self.output_proj(output))


class DeformableDecoderLayer(nn.Module):
    """[可选 3D-RoPE 自注意力] → 可变形交叉注意力 → FFN，均为 Pre-Norm 残差。"""

    def __init__(self, config, use_self_attn=True):
        super().__init__()
        dim, heads = config.embed_dim, config.num_heads
        self.use_self_attn = use_self_attn
        if use_self_attn:
            self.norm_self = make_norm(dim, config.norm_type)
            self.self_attn = MultiHeadSelfAttention(dim, heads, config.dropout, use_rope=config.use_rope_3d)
        self.norm_cross = make_norm(dim, config.norm_type)
        self.cross_attn = DeformableCrossAttention(
            dim, heads, config.num_cameras, config.level_strides, config.image_size,
            num_points=config.num_sample_points, offset_px=config.deform_offset_px,
            dropout=config.dropout, visibility_norm=config.cam_visibility_norm)
        self.norm_mlp = make_norm(dim, config.norm_type)
        self.mlp = FeedForward(dim, 4.0, config.dropout, config.ffn_type)

    def forward(self, query, points, image_feats, intrinsics, extrinsics):
        if self.use_self_attn:
            query = query + self.self_attn(self.norm_self(query), coords=points)
        query = query + self.cross_attn(self.norm_cross(query), points, image_feats, intrinsics, extrinsics)
        query = query + self.mlp(self.norm_mlp(query))
        return query
