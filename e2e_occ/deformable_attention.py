import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .position_encoding import rescale_focal_to_feature_map
except (ImportError, ValueError):
    from position_encoding import rescale_focal_to_feature_map

class DeformableCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, num_cameras, num_points=1, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_cameras = num_cameras
        self.num_points = num_points
        self.head_dim = dim // num_heads
        
        self.sampling_offsets = nn.Linear(dim, num_cameras * num_heads * num_points * 2)
        self.attention_weights = nn.Linear(dim, num_cameras * num_heads * num_points)
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        nn.init.xavier_uniform_(self.attention_weights.weight)
        nn.init.constant_(self.attention_weights.bias, 0.0)
    
    def get_reference_points(self, query_coords, intrinsics, extrinsics, H, W):
        """
        Project 3D query coordinates to 2D image reference points.
        query_coords: [B, Q, 3] in [0, 1]

        等距投影(equidistant)正向投影，必须和 RayDirectionEncoding.get_rays_from_params
        的反投影是同一个相机模型的正/逆变换——那边决定"这个像素对应哪个方向的入射光线"，
        这边决定"这个 3D 点该去图像哪里采样特征"，两者不一致会导致几何自相矛盾。
        公式: theta = acos(Z/|P|)（与光轴夹角），phi = atan2(Y, X)，r = f * theta（像素半径）。
        """
        B, Q, _ = query_coords.shape
        N = self.num_cameras

        # Map [0,1] to World Coords [-40, 40] ...
        real_x = query_coords[..., 0] * 80.0 - 40.0
        real_y = query_coords[..., 1] * 80.0 - 40.0
        real_z = query_coords[..., 2] * 6.4 - 1.0
        world_points = torch.stack([real_x, real_y, real_z, torch.ones_like(real_x)], dim=-1) # [B, Q, 4]

        # Expand for cameras
        world_points = world_points.unsqueeze(1).expand(-1, N, -1, -1) # [B, N, Q, 4]

        # World to Camera
        inv_extrinsics = torch.inverse(extrinsics) # [B, N, 4, 4]
        cam_points = torch.matmul(inv_extrinsics.unsqueeze(2), world_points.unsqueeze(-1)).squeeze(-1)

        # Camera to Image：等距投影正向投影
        cam_points_3d = cam_points[..., :3] # [B, N, Q, 3]
        r3 = torch.linalg.norm(cam_points_3d, dim=-1).clamp_min(1e-6)  # [B, N, Q]
        cos_theta = (cam_points_3d[..., 2] / r3).clamp(-1.0, 1.0)
        theta = torch.acos(cos_theta)  # 与光轴 (Z) 夹角
        phi = torch.atan2(cam_points_3d[..., 1], cam_points_3d[..., 0])

        f, cx, cy = rescale_focal_to_feature_map(intrinsics, H, W)  # 各 [B, N]
        f = f.unsqueeze(-1)   # [B, N, 1]
        cx = cx.unsqueeze(-1)  # [B, N, 1]
        cy = cy.unsqueeze(-1)  # [B, N, 1]
        r_img = f * theta  # [B, N, Q]，等距投影下的像素半径

        u = cx + r_img * torch.cos(phi)
        v = cy + r_img * torch.sin(phi)

        # Normalize to [-1, 1]，align_corners=False 约定（像素中心在 (i+0.5)/size），
        # 必须和下面 forward() 里 F.grid_sample(..., align_corners=False) 保持一致——
        # 之前这里用的是 align_corners=True 的 /(W-1) 公式，和实际采样时的 False 约定
        # 对不上，会带来系统性的径向缩放/偏移误差。
        u_norm = 2.0 * (u + 0.5) / W - 1.0
        v_norm = 2.0 * (v + 0.5) / H - 1.0

        ref_points = torch.stack([u_norm, v_norm], dim=-1) # [B, N, Q, 2]

        return ref_points

    def forward(self, query, query_coords, image_feats, intrinsics=None, extrinsics=None):
        """
        Serial Camera Loop Implementation for Memory Efficiency
        """
        B, Q, C = query.shape
        _, N, _, H, W = image_feats.shape
        
        # 1. Get Reference Points
        if intrinsics is not None and extrinsics is not None:
            reference_points = self.get_reference_points(query_coords, intrinsics, extrinsics, H, W)
        else:
            reference_points = torch.zeros(B, N, Q, 2, device=query.device)
            
        # 2. Predict Offsets & Weights
        offsets = self.sampling_offsets(query) 
        offsets = offsets.view(B, Q, N, self.num_heads, self.num_points, 2)
        offsets = offsets.tanh() * 0.5 
        
        attn_weights = self.attention_weights(query)
        attn_weights = attn_weights.view(B, Q, N, self.num_heads, self.num_points)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 3. Project Values
        value_proj = self.value_proj(image_feats.permute(0, 1, 3, 4, 2)) 
        value_proj = value_proj.view(B, N, H, W, self.num_heads, self.head_dim)
        
        # 4. Serial Sampling Loop
        output = torch.zeros(B, Q, self.num_heads, self.head_dim, device=query.device)
        
        for cam in range(N):
            ref_cam = reference_points[:, cam] 
            off_cam = offsets[:, :, cam]
            w_cam = attn_weights[:, :, cam]
            v_cam = value_proj[:, cam]
            
            locs = ref_cam.unsqueeze(2).unsqueeze(3) + off_cam
            locs_flat = locs.view(B, Q * self.num_heads * self.num_points, 1, 2)
            
            # Optimization: Loop over Heads
            sampled_list = []
            for h in range(self.num_heads):
                v_h = v_cam[..., h, :].permute(0, 3, 1, 2) 
                
                locs_h = locs[:, :, h, :]
                locs_h = locs_h.contiguous().view(B, Q * self.num_points, 1, 2)
                
                s_h = F.grid_sample(v_h, locs_h, mode='bilinear', align_corners=False, padding_mode='zeros')
                s_h = s_h.view(B, self.head_dim, Q, self.num_points).permute(0, 2, 3, 1) 
                
                w_h = w_cam[:, :, h, :].unsqueeze(-1)
                
                weighted = (s_h * w_h).sum(dim=2)
                sampled_list.append(weighted)
            
            output_cam = torch.stack(sampled_list, dim=2)
            output = output + output_cam
            
        output = output.view(B, Q, C)
        output = self.output_proj(output)
        
        return self.dropout(output)

class DeformableDecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, num_cameras, num_points=1, mlp_ratio=4.0, dropout=0.1, use_self_attn=True):
        super().__init__()
        
        self.use_self_attn = use_self_attn
        if self.use_self_attn:
            self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
            self.norm_self = nn.LayerNorm(dim)

        self.cross_attn = DeformableCrossAttention(dim, num_heads, num_cameras, num_points, dropout)
        self.norm_cross = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, query, query_coords, image_feats, intrinsics=None, extrinsics=None):
        if self.use_self_attn:
            q = self.norm_self(query)
            query = query + self.self_attn(q, q, q)[0]

        # Cross-Attention
        query = query + self.cross_attn(self.norm_cross(query), query_coords, image_feats, intrinsics, extrinsics)

        query = query + self.mlp(self.norm_mlp(query))
        return query
