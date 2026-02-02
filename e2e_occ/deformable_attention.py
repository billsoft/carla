import torch
import torch.nn as nn
import torch.nn.functional as F

class DeformableCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, num_cameras, num_points=2, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_cameras = num_cameras
        self.num_points = num_points
        self.head_dim = dim // num_heads
        
        # Optim: Shared offsets/weights for all cameras? 
        # No, geometry is different. Keep per-camera logic but executed serially.
        # But we can predict offsets for ALL cameras at once (small tensor),
        # then loop for sampling.
        
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
        """
        B, Q, _ = query_coords.shape
        N = self.num_cameras
        
        # Map [0,1] to World Coords [-40, 40] ...
        # NOTE: Ideally this range should come from config, but for now hardcoded to match common settings
        real_x = query_coords[..., 0] * 80.0 - 40.0
        real_y = query_coords[..., 1] * 80.0 - 40.0
        real_z = query_coords[..., 2] * 6.4 - 1.0
        world_points = torch.stack([real_x, real_y, real_z, torch.ones_like(real_x)], dim=-1) # [B, Q, 4]
        
        # Expand for cameras
        world_points = world_points.unsqueeze(1).expand(-1, N, -1, -1) # [B, N, Q, 4]
        
        # World to Camera
        inv_extrinsics = torch.inverse(extrinsics) # [B, N, 4, 4]
        cam_points = torch.matmul(inv_extrinsics.unsqueeze(2), world_points.unsqueeze(-1)).squeeze(-1)
        
        # Camera to Image
        cam_points_3d = cam_points[..., :3] # [B, N, Q, 3]
        img_points_h = torch.matmul(intrinsics.unsqueeze(2), cam_points_3d.unsqueeze(-1)).squeeze(-1)
        
        # Normalize
        depth = img_points_h[..., 2] + 1e-6
        u = img_points_h[..., 0] / depth
        v = img_points_h[..., 1] / depth
        
        # Normalize to [-1, 1]
        u_norm = 2.0 * u / (W - 1) - 1.0
        v_norm = 2.0 * v / (H - 1) - 1.0
        
        ref_points = torch.stack([u_norm, v_norm], dim=-1) # [B, N, Q, 2]
        
        # Valid mask (optional, but good for learning)
        # valid = (depth > 0) & (u_norm > -1.1) & (u_norm < 1.1) & (v_norm > -1.1) & (v_norm < 1.1)
        
        return ref_points

    def forward(self, query, query_coords, image_feats, intrinsics=None, extrinsics=None):
        """
        Serial Camera Loop Implementation for Memory Efficiency
        """
        B, Q, C = query.shape
        _, N, _, H, W = image_feats.shape
        
        # 1. Get Reference Points (Lightweight)
        if intrinsics is not None and extrinsics is not None:
            reference_points = self.get_reference_points(query_coords, intrinsics, extrinsics, H, W)
        else:
            reference_points = torch.zeros(B, N, Q, 2, device=query.device)
            
        # 2. Predict Offsets & Weights (Once for all cameras)
        offsets = self.sampling_offsets(query) # [B, Q, N*H*P*2]
        offsets = offsets.view(B, Q, N, self.num_heads, self.num_points, 2)
        offsets = offsets.tanh() * 0.5 
        
        attn_weights = self.attention_weights(query) # [B, Q, N*H*P]
        attn_weights = attn_weights.view(B, Q, N, self.num_heads, self.num_points)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 3. Project Values (Can be done per camera or batched)
        # Batched projection is fine (N*C*H*W is not too huge compared to sampling grid)
        # [B, N, C, H, W] -> [B, N, H, W, C]
        value_proj = self.value_proj(image_feats.permute(0, 1, 3, 4, 2)) 
        value_proj = value_proj.view(B, N, H, W, self.num_heads, self.head_dim)
        
        # 4. Serial Sampling Loop (The Core Optimization)
        output = torch.zeros(B, Q, self.num_heads, self.head_dim, device=query.device)
        
        for cam in range(N):
            # Per-camera data
            # Ref: [B, Q, 2]
            ref_cam = reference_points[:, cam] 
            # Offsets: [B, Q, Heads, Points, 2]
            off_cam = offsets[:, :, cam]
            # Weights: [B, Q, Heads, Points]
            w_cam = attn_weights[:, :, cam]
            # Value: [B, H, W, Heads, HeadDim]
            v_cam = value_proj[:, cam]
            
            # Sampling Locations
            # [B, Q, 1, 1, 2] + [B, Q, Heads, Points, 2] -> [B, Q, Heads, Points, 2]
            locs = ref_cam.unsqueeze(2).unsqueeze(3) + off_cam
            
            # Flatten for grid_sample
            # locs: [B, Q*Heads*Points, 1, 2]
            locs_flat = locs.view(B, Q * self.num_heads * self.num_points, 1, 2)
            
            # Value: [B, Heads*HeadDim, H, W]
            v_cam_flat = v_cam.permute(0, 3, 4, 1, 2).reshape(B, C, H, W)
            
            # Sample
            # sampled: [B, C, L, 1]
            sampled = F.grid_sample(v_cam_flat, locs_flat, mode='bilinear', align_corners=False, padding_mode='zeros')
            
            # Reshape back
            # [B, Heads, HeadDim, Q, Heads, Points] -> This is wrong mapping
            # v_cam_flat channels are (Head1_D1...D32, Head2_D1...D32)
            # locs_flat spatial are (Q1_H1_P1, Q1_H1_P2, ..., Q1_H2_P1...)
            # We need to pick correct head channels for correct head locs?
            # Standard grid_sample samples ALL channels at ALL locs.
            # So at loc (Q1_H1_P1), we get channels for H1, H2, ... H8.
            # But we ONLY care about H1 channels for H1 locs.
            # This implies we fetch N_Heads * N_Heads data, wasting memory?
            # Yes. Standard Multi-Head DeformAttn usually does:
            # Grouped Grid Sample or Loop over Heads.
            
            # Optimization: Loop over Heads (Since NumHeads=4 is small)
            # This avoids the "All Channels at All Locs" waste.
            
            sampled_list = []
            for h in range(self.num_heads):
                # Value for head h: [B, HeadDim, H, W]
                v_h = v_cam[..., h, :].permute(0, 3, 1, 2) 
                
                # Locs for head h: [B, Q, Points, 2]
                locs_h = locs[:, :, h, :]
                locs_h = locs_h.contiguous().view(B, Q * self.num_points, 1, 2)
                
                # Sample: [B, HeadDim, Q*Points, 1]
                s_h = F.grid_sample(v_h, locs_h, mode='bilinear', align_corners=False, padding_mode='zeros')
                s_h = s_h.view(B, self.head_dim, Q, self.num_points).permute(0, 2, 3, 1) # [B, Q, Points, HeadDim]
                
                # Weight: [B, Q, Points, 1]
                w_h = w_cam[:, :, h, :].unsqueeze(-1)
                
                # Weighted Sum: [B, Q, HeadDim]
                weighted = (s_h * w_h).sum(dim=2)
                
                sampled_list.append(weighted)
            
            # Stack Heads: [B, Q, Heads, HeadDim]
            output_cam = torch.stack(sampled_list, dim=2)
            
            # Accumulate across cameras
            output = output + output_cam
            
        # Final Projection
        output = output.view(B, Q, C)
        output = self.output_proj(output)
        
        return self.dropout(output)

class DeformableDecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, num_cameras, num_points=2, mlp_ratio=4.0, dropout=0.1, use_self_attn=True):
        super().__init__()
        self.use_self_attn = use_self_attn
        if self.use_self_attn:
            self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
            self.norm1 = nn.LayerNorm(dim)
            
        self.cross_attn = DeformableCrossAttention(dim, num_heads, num_cameras, num_points, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, query, query_coords, image_feats, intrinsics=None, extrinsics=None):
        if self.use_self_attn:
            q = self.norm1(query)
            query = query + self.self_attn(q, q, q)[0]
            
        # Pass query_coords (ref_3d) and camera params
        query = query + self.cross_attn(self.norm2(query), query_coords, image_feats, intrinsics, extrinsics)
        query = query + self.mlp(self.norm3(query))
        return query
