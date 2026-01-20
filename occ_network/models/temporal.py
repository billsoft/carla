import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionCompensation(nn.Module):
    def __init__(self, bev_h, bev_w, pc_range):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        x = torch.linspace(-1, 1, bev_w)
        y = torch.linspace(-1, 1, bev_h)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        self.register_buffer('base_grid', torch.stack([xx, yy], dim=-1))

    def forward(self, bev_features, ego_motion):
        B, C, H, W = bev_features.shape
        device = bev_features.device
        grid = self.base_grid.unsqueeze(0).expand(B, -1, -1, -1).to(device)
        rot = ego_motion[:, :2, :2]
        trans = ego_motion[:, :2, 3:4]
        grid_flat = grid.view(B, -1, 2)
        new_grid = torch.bmm(grid_flat, rot.transpose(1, 2))
        scale_x = (self.pc_range[3] - self.pc_range[0]) / 2
        scale_y = (self.pc_range[4] - self.pc_range[1]) / 2
        trans_norm = trans.squeeze(-1) / torch.tensor([scale_x, scale_y], device=device)
        new_grid = (new_grid + trans_norm.unsqueeze(1)).view(B, H, W, 2)
        return F.grid_sample(bev_features, new_grid, mode='bilinear', padding_mode='zeros', align_corners=False)


class TemporalTransformerFusion(nn.Module):
    """
    Transformer时序融合 (改进版)

    关键改进:
    1. 5帧历史 (比原来2帧更多上下文)
    2. Transformer注意力替代简单门控
    3. 可学习的时序位置编码
    4. 支持变长历史 (自动处理不够5帧的情况)
    """
    def __init__(self, dim, num_frames=5, bev_h=128, bev_w=128, pc_range=None, num_heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_frames = num_frames
        self.bev_h = bev_h
        self.bev_w = bev_w

        # 运动补偿
        if pc_range is None:
            pc_range = [-40, -40, -1, 40, 40, 5.4]
        self.motion_comp = MotionCompensation(bev_h, bev_w, pc_range)

        # 时序位置编码 (可学习)
        self.temporal_pos = nn.Parameter(torch.randn(num_frames, dim) * 0.02)

        # 轻量级时序注意力 (每个BEV位置独立)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 输出归一化和投影
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Conv2d(dim, dim, 1)

        # 历史缓存
        self.history_bevs = []
        self.history_poses = []

    def reset(self):
        """场景切换时重置历史"""
        self.history_bevs.clear()
        self.history_poses.clear()

    def forward(self, current_bev, ego_motion=None, current_pose=None):
        """
        Args:
            current_bev: [B, C, H, W] 当前帧BEV特征
            ego_motion: [B, 4, 4] 自车运动
            current_pose: [B, 4, 4] 当前帧世界位姿

        Returns:
            fused_bev: [B, C, H, W] 时序融合后的BEV特征
        """
        B, C, H, W = current_bev.shape
        device = current_bev.device
        dtype = current_bev.dtype

        # 收集所有帧 (当前帧 + 历史帧)
        all_bevs = [current_bev]
        all_poses = [current_pose]

        # 对齐历史帧到当前坐标系
        for i, (hist_bev, hist_pose) in enumerate(zip(reversed(self.history_bevs), reversed(self.history_poses))):
            if current_pose is not None and hist_pose is not None:
                # 计算相对位姿
                dtype_orig = current_pose.dtype
                current_pose_fp32 = current_pose.float()
                hist_pose_fp32 = hist_pose.float()
                rel_pose = torch.bmm(torch.inverse(current_pose_fp32), hist_pose_fp32).to(dtype_orig)
            else:
                rel_pose = ego_motion if ego_motion is not None else torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)

            # 运动补偿对齐
            aligned_bev = self.motion_comp(hist_bev.to(device), rel_pose)
            all_bevs.append(aligned_bev)
            all_poses.append(hist_pose)

        num_frames_actual = len(all_bevs)

        # 如果只有当前帧，直接返回
        if num_frames_actual == 1:
            self._update_history(current_bev, current_pose)
            return current_bev

        # 堆叠所有帧: [B, T, C, H, W]
        stacked_bevs = torch.stack(all_bevs, dim=1)

        # Reshape for attention: [B*H*W, T, C]
        stacked_bevs = stacked_bevs.permute(0, 3, 4, 1, 2).contiguous()  # [B, H, W, T, C]
        stacked_bevs = stacked_bevs.view(B * H * W, num_frames_actual, C)

        # 添加时序位置编码
        temporal_pos = self.temporal_pos[:num_frames_actual].unsqueeze(0)  # [1, T, C]
        stacked_bevs = stacked_bevs + temporal_pos.to(dtype)

        # 时序自注意力
        # Query: 当前帧, Key/Value: 所有帧
        query = stacked_bevs[:, 0:1, :]  # [B*H*W, 1, C]
        attn_out, _ = self.temporal_attn(query, stacked_bevs, stacked_bevs)
        attn_out = attn_out.squeeze(1)  # [B*H*W, C]

        # Reshape back: [B, C, H, W]
        attn_out = attn_out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # 残差连接 + 归一化
        output = current_bev + self.out_proj(attn_out)

        # 更新历史
        self._update_history(current_bev, current_pose)

        return output

    def _update_history(self, bev, pose):
        """更新历史缓存"""
        self.history_bevs.append(bev.detach().clone())
        self.history_poses.append(pose.detach().clone() if pose is not None else None)

        # 保留 num_frames - 1 帧历史 (当前帧不计入)
        max_history = self.num_frames - 1
        if len(self.history_bevs) > max_history:
            self.history_bevs.pop(0)
            self.history_poses.pop(0)


class LightweightTemporalFusion(nn.Module):
    """
    轻量级时序融合 (原版，保留兼容性)

    默认使用 Transformer 版本，可通过 use_transformer=False 切换到门控版本
    """
    def __init__(self, dim, num_frames, bev_h, bev_w, pc_range, use_transformer=True):
        super().__init__()
        self.dim = dim
        self.num_frames = num_frames
        self.use_transformer = use_transformer

        if use_transformer:
            # 使用 Transformer 时序融合 (改进版)
            self.fusion = TemporalTransformerFusion(
                dim=dim,
                num_frames=num_frames,
                bev_h=bev_h,
                bev_w=bev_w,
                pc_range=pc_range,
            )
        else:
            # 原始门控融合 (兼容)
            self.motion_comp = MotionCompensation(bev_h, bev_w, pc_range)
            self.fuse = nn.Conv2d(dim * 2, dim, 1)
            self.gate = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.Sigmoid())
            self.history = []
            self.history_poses = []

    def reset(self):
        if self.use_transformer:
            self.fusion.reset()
        else:
            self.history = []
            self.history_poses = []

    def forward(self, current_bev, ego_motion=None, current_pose=None):
        if self.use_transformer:
            return self.fusion(current_bev, ego_motion, current_pose)

        # 原始门控融合逻辑
        if len(self.history) == 0:
            self._update_history(current_bev, current_pose)
            return current_bev

        B, C, H, W = current_bev.shape
        device = current_bev.device
        hist_bev, hist_pose = self.history[-1], self.history_poses[-1]

        if current_pose is not None and hist_pose is not None:
            dtype = current_pose.dtype
            current_pose_fp32 = current_pose.float()
            hist_pose_fp32 = hist_pose.float()
            rel_pose = torch.bmm(torch.inverse(current_pose_fp32), hist_pose_fp32).to(dtype)
        else:
            rel_pose = ego_motion if ego_motion is not None else torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1)

        aligned = self.motion_comp(hist_bev.to(device), rel_pose)
        concat = torch.cat([current_bev, aligned], dim=1)
        fused = self.fuse(concat)
        gate = self.gate(concat)
        output = gate * fused + (1 - gate) * current_bev
        self._update_history(current_bev, current_pose)
        return output

    def _update_history(self, bev, pose):
        self.history.append(bev.detach().clone())
        self.history_poses.append(pose.detach().clone() if pose is not None else None)
        if len(self.history) > self.num_frames - 1:
            self.history.pop(0)
            self.history_poses.pop(0)
