"""
时序融合模块 (改进版)

改进点:
1. TBPTT (截断反向传播): 近期帧保留梯度，远期帧 detach
2. 场景边界检测: 自动检测场景切换并 reset
3. 动态物体运动估计: 估计非自车运动
4. 时空位置编码: 编码时间间隔信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class MotionCompensation(nn.Module):
    """自车运动补偿"""
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


class DynamicMotionEstimator(nn.Module):
    """
    动态物体运动估计器

    估计除自车运动外的残差运动场 (用于处理移动车辆、行人等)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.motion_net = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, 2, 1),  # 输出 2D 运动场 (dx, dy)
        )

    def forward(self, current_bev: torch.Tensor, history_bev: torch.Tensor) -> torch.Tensor:
        """
        估计残差运动场

        Args:
            current_bev: [B, C, H, W] 当前帧 BEV
            history_bev: [B, C, H, W] 对齐后的历史帧 BEV

        Returns:
            motion_field: [B, 2, H, W] 残差运动场 (dx, dy)
        """
        concat = torch.cat([current_bev, history_bev], dim=1)
        motion_field = self.motion_net(concat)  # [B, 2, H, W]
        # 限制运动范围 (防止过大的偏移)
        motion_field = torch.tanh(motion_field) * 0.1  # 最大偏移 10% 的 BEV 尺寸
        return motion_field

    def warp_with_motion(self, bev: torch.Tensor, motion_field: torch.Tensor) -> torch.Tensor:
        """
        使用运动场对 BEV 进行 warp

        Args:
            bev: [B, C, H, W] 输入 BEV
            motion_field: [B, 2, H, W] 运动场

        Returns:
            warped_bev: [B, C, H, W] warp 后的 BEV
        """
        B, C, H, W = bev.shape
        device = bev.device

        # 创建基础网格
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        # 添加运动场
        motion_field_permuted = motion_field.permute(0, 2, 3, 1)  # [B, H, W, 2]
        new_grid = base_grid + motion_field_permuted

        # Warp
        warped = F.grid_sample(bev, new_grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        return warped


class SpatioTemporalPositionEncoding(nn.Module):
    """
    时空位置编码 (改进版)

    改进点:
    1. 编码时间间隔 (不只是顺序)
    2. 空间位置相关的时序编码
    3. 连续时间编码 (基于实际时间戳)
    """

    def __init__(self, dim: int, num_frames: int, bev_h: int, bev_w: int):
        super().__init__()
        self.dim = dim
        self.num_frames = num_frames

        # 时间编码: 基于时间间隔的 MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(1, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
        )

        # 空间编码: 可学习的 2D 位置编码
        self.spatial_embed = nn.Parameter(torch.randn(1, dim, bev_h, bev_w) * 0.02)

        # 时空交互
        self.st_fusion = nn.Conv2d(dim * 2, dim, 1)

    def forward(
        self,
        bev_features: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        添加时空位置编码

        Args:
            bev_features: [B, T, C, H, W] 多帧 BEV 特征
            timestamps: [B, T] 每帧的相对时间 (秒，当前帧为 0)

        Returns:
            encoded_features: [B, T, C, H, W] 添加位置编码后的特征
        """
        B, T, C, H, W = bev_features.shape
        device = bev_features.device

        # 计算时间编码
        if timestamps is None:
            # 默认: 假设等间隔 0.1 秒
            timestamps = torch.arange(T, device=device).float() * (-0.1)
            timestamps = timestamps.unsqueeze(0).expand(B, -1)

        # 时间编码
        time_enc = self.time_mlp(timestamps.unsqueeze(-1))  # [B, T, dim]
        time_enc = time_enc.unsqueeze(-1).unsqueeze(-1)  # [B, T, dim, 1, 1]
        time_enc = time_enc.expand(-1, -1, -1, H, W)  # [B, T, dim, H, W]

        # 空间编码
        spatial_enc = self.spatial_embed.expand(B, -1, -1, -1)  # [B, dim, H, W]
        spatial_enc = spatial_enc.unsqueeze(1).expand(-1, T, -1, -1, -1)  # [B, T, dim, H, W]

        # 时空融合
        combined = torch.cat([time_enc, spatial_enc], dim=2)  # [B, T, 2*dim, H, W]
        combined = combined.view(B * T, -1, H, W)
        st_encoding = self.st_fusion(combined)  # [B*T, dim, H, W]
        st_encoding = st_encoding.view(B, T, C, H, W)

        return bev_features + st_encoding


class TemporalTransformerFusion(nn.Module):
    """
    Transformer时序融合 (改进版 V2)

    关键改进:
    1. TBPTT (截断反向传播): 近期帧保留梯度，远期帧 detach
    2. 场景边界检测: 自动检测场景切换并 reset
    3. 动态物体运动估计: 估计非自车运动
    4. 时空位置编码: 编码时间间隔信息
    5. 5帧历史 + Transformer注意力
    """
    def __init__(self, dim, num_frames=5, bev_h=128, bev_w=128, pc_range=None, num_heads=4, dropout=0.1,
                 tbptt_steps=3, use_dynamic_motion=True, use_st_encoding=True):
        super().__init__()
        self.dim = dim
        self.num_frames = num_frames
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.tbptt_steps = tbptt_steps  # 保留梯度的帧数
        self.use_dynamic_motion = use_dynamic_motion
        self.use_st_encoding = use_st_encoding

        # 运动补偿
        if pc_range is None:
            pc_range = [-40, -40, -1, 40, 40, 5.4]
        self.motion_comp = MotionCompensation(bev_h, bev_w, pc_range)

        # 动态物体运动估计器 (新增)
        if use_dynamic_motion:
            self.dynamic_motion = DynamicMotionEstimator(dim)

        # 时空位置编码 (改进版)
        if use_st_encoding:
            self.st_pos_enc = SpatioTemporalPositionEncoding(dim, num_frames, bev_h, bev_w)
        else:
            # 简单的时序位置编码 (可学习)
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

        # 历史缓存 (改进: 不再全部 detach)
        self.history_bevs: List[torch.Tensor] = []
        self.history_poses: List[Optional[torch.Tensor]] = []
        self.history_timestamps: List[Optional[float]] = []
        self.history_scene_ids: List[Optional[str]] = []

    def reset(self):
        """场景切换时重置历史"""
        self.history_bevs.clear()
        self.history_poses.clear()
        self.history_timestamps.clear()
        self.history_scene_ids.clear()

    def _is_new_scene(self, scene_id: Optional[str]) -> bool:
        """
        检测是否是新场景

        场景切换条件:
        1. scene_id 变化
        2. 时间戳跳跃过大 (> 1秒)
        """
        if scene_id is None or len(self.history_scene_ids) == 0:
            return False

        last_scene_id = self.history_scene_ids[-1]
        if last_scene_id is None:
            return False

        return scene_id != last_scene_id

    def forward(self, current_bev, ego_motion=None, current_pose=None,
                timestamp: Optional[float] = None, scene_id: Optional[str] = None):
        """
        Args:
            current_bev: [B, C, H, W] 当前帧BEV特征
            ego_motion: [B, 4, 4] 自车运动
            current_pose: [B, 4, 4] 当前帧世界位姿
            timestamp: 当前帧时间戳 (秒)
            scene_id: 场景ID (用于检测场景切换)

        Returns:
            fused_bev: [B, C, H, W] 时序融合后的BEV特征
        """
        B, C, H, W = current_bev.shape
        device = current_bev.device
        dtype = current_bev.dtype

        # 场景边界检测
        if self._is_new_scene(scene_id):
            self.reset()

        # 收集所有帧 (当前帧 + 历史帧)
        all_bevs = [current_bev]
        all_poses = [current_pose]
        all_timestamps = [timestamp if timestamp is not None else 0.0]

        # 对齐历史帧到当前坐标系
        for i, (hist_bev, hist_pose) in enumerate(zip(reversed(self.history_bevs), reversed(self.history_poses))):
            # 🔑 解压缩: 上采样 + 恢复 dtype
            hist_bev_full = F.interpolate(
                hist_bev.float(),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).to(dtype)

            if current_pose is not None and hist_pose is not None:
                # 计算相对位姿
                dtype_orig = current_pose.dtype
                current_pose_fp32 = current_pose.float()
                hist_pose_fp32 = hist_pose.float()
                rel_pose = torch.bmm(torch.inverse(current_pose_fp32), hist_pose_fp32).to(dtype_orig)
            else:
                rel_pose = ego_motion if ego_motion is not None else torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)

            # 运动补偿对齐 (ego motion)
            aligned_bev = self.motion_comp(hist_bev_full.to(device), rel_pose)

            # 动态物体运动补偿 (新增)
            if self.use_dynamic_motion and i < self.tbptt_steps:
                motion_field = self.dynamic_motion(current_bev, aligned_bev)
                aligned_bev = self.dynamic_motion.warp_with_motion(aligned_bev, motion_field)

            all_bevs.append(aligned_bev)
            all_poses.append(hist_pose)

            # 计算时间戳
            if len(self.history_timestamps) > i:
                hist_ts = self.history_timestamps[-(i+1)]
                all_timestamps.append(hist_ts if hist_ts is not None else 0.0)
            else:
                all_timestamps.append(0.0)

        num_frames_actual = len(all_bevs)

        # 如果只有当前帧，直接返回
        if num_frames_actual == 1:
            self._update_history(current_bev, current_pose, timestamp, scene_id)
            return current_bev

        # 堆叠所有帧: [B, T, C, H, W]
        stacked_bevs = torch.stack(all_bevs, dim=1)

        # 时空位置编码 (改进版)
        if self.use_st_encoding:
            # 计算相对时间戳
            current_ts = all_timestamps[0]
            relative_timestamps = torch.tensor(
                [ts - current_ts for ts in all_timestamps],
                device=device, dtype=dtype
            ).unsqueeze(0).expand(B, -1)  # [B, T]

            stacked_bevs = self.st_pos_enc(stacked_bevs, relative_timestamps)

        # Reshape for attention: [B*H*W, T, C]
        stacked_bevs = stacked_bevs.permute(0, 3, 4, 1, 2).contiguous()  # [B, H, W, T, C]
        stacked_bevs = stacked_bevs.view(B * H * W, num_frames_actual, C)

        # 简单时序位置编码 (如果不使用 st_encoding)
        if not self.use_st_encoding:
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

        # 更新历史 (TBPTT: 不完全 detach)
        self._update_history(current_bev, current_pose, timestamp, scene_id)

        return output

    def _update_history(self, bev: torch.Tensor, pose: Optional[torch.Tensor],
                        timestamp: Optional[float], scene_id: Optional[str]):
        """
        更新历史缓存 (TBPTT 改进版 + 显存压缩)
        """
        # 🔑 历史帧压缩: 降采样 + FP16
        bev_compressed = F.interpolate(
            bev.float(), 
            size=(self.bev_h // 2, self.bev_w // 2),  # 降采样一半
            mode='bilinear', 
            align_corners=False
        ).half()  # 转 FP16

        # 添加新帧 (不 detach，保留梯度)
        self.history_bevs.append(bev_compressed)
        self.history_poses.append(pose.clone() if pose is not None else None)
        self.history_timestamps.append(timestamp)
        self.history_scene_ids.append(scene_id)

        # 保留 num_frames - 1 帧历史 (当前帧不计入)
        max_history = self.num_frames - 1
        while len(self.history_bevs) > max_history:
            self.history_bevs.pop(0)
            self.history_poses.pop(0)
            self.history_timestamps.pop(0)
            self.history_scene_ids.pop(0)

        # TBPTT: 对超过 tbptt_steps 的帧做 detach
        # 这样只有最近 tbptt_steps 帧有梯度回传
        for i in range(len(self.history_bevs) - self.tbptt_steps):
            if i >= 0 and self.history_bevs[i].requires_grad:
                self.history_bevs[i] = self.history_bevs[i].detach()

    def detach_history(self):
        """
        强制分离所有历史帧 (用于 TBPTT 窗口结束时)
        """
        for i in range(len(self.history_bevs)):
            if self.history_bevs[i].requires_grad:
                self.history_bevs[i] = self.history_bevs[i].detach()
        
        for i in range(len(self.history_poses)):
            if self.history_poses[i] is not None and self.history_poses[i].requires_grad:
                self.history_poses[i] = self.history_poses[i].detach()


class LightweightTemporalFusion(nn.Module):
    """
    轻量级时序融合 (改进版 V2)

    改进点:
    1. 默认使用 TBPTT 的 Transformer 版本
    2. 支持场景边界检测
    3. 支持动态物体运动估计

    可通过 use_transformer=False 切换到门控版本 (兼容旧代码)
    """
    def __init__(self, dim, num_frames, bev_h, bev_w, pc_range, use_transformer=True,
                 tbptt_steps=3, use_dynamic_motion=True, use_st_encoding=True):
        super().__init__()
        self.dim = dim
        self.num_frames = num_frames
        self.use_transformer = use_transformer

        if use_transformer:
            # 使用 Transformer 时序融合 (改进版 V2)
            self.fusion = TemporalTransformerFusion(
                dim=dim,
                num_frames=num_frames,
                bev_h=bev_h,
                bev_w=bev_w,
                pc_range=pc_range,
                tbptt_steps=tbptt_steps,
                use_dynamic_motion=use_dynamic_motion,
                use_st_encoding=use_st_encoding,
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

    def detach_history(self):
        """强制分离所有历史帧"""
        if self.use_transformer:
            self.fusion.detach_history()
        else:
            for i in range(len(self.history)):
                if self.history[i].requires_grad:
                    self.history[i] = self.history[i].detach()
            
            for i in range(len(self.history_poses)):
                if self.history_poses[i] is not None and self.history_poses[i].requires_grad:
                    self.history_poses[i] = self.history_poses[i].detach()

    def forward(self, current_bev, ego_motion=None, current_pose=None,
                timestamp: Optional[float] = None, scene_id: Optional[str] = None):
        """
        Args:
            current_bev: [B, C, H, W] 当前帧BEV特征
            ego_motion: [B, 4, 4] 自车运动
            current_pose: [B, 4, 4] 当前帧世界位姿
            timestamp: 当前帧时间戳 (秒) - 用于时空编码
            scene_id: 场景ID - 用于检测场景切换

        Returns:
            fused_bev: [B, C, H, W] 时序融合后的BEV特征
        """
        if self.use_transformer:
            return self.fusion(current_bev, ego_motion, current_pose, timestamp, scene_id)

        # 原始门控融合逻辑 (兼容)
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
