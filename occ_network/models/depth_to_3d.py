"""
Lift-Splat-Shoot 风格的深度感知 2D→3D 转换

核心改进:
1. 深度不是"预测完就扔"，而是参与3D重建
2. 每个像素的深度分布 × 特征 = 3D点云特征
3. 累加到统一的3D体素网格

架构:
  Image → Encoder → Depth Distribution + Features
                         ↓
                    Lift (沿深度撒点)
                         ↓
                    Splat (投影到体素)
                         ↓
                    3D Voxel Features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List, Optional


class DepthDistributionNet(nn.Module):
    """
    改进的深度分布预测网络

    比原来的 DepthPredictionHead 更强:
    1. 多层卷积，更好的特征提取
    2. 同时输出深度分布和加权特征
    """

    def __init__(
        self,
        in_channels: int,
        num_depth_bins: int = 64,
        depth_range: Tuple[float, float] = (0.5, 80.0),
        hidden_channels: int = 256,
    ):
        super().__init__()
        self.num_depth_bins = num_depth_bins
        self.depth_range = depth_range

        # 深度分布预测网络 (更强)
        self.depth_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_depth_bins, 1),
        )

        # 预计算深度bin中心值 (对数均匀分布)
        min_d, max_d = depth_range
        depth_bins = torch.exp(torch.linspace(
            math.log(min_d), math.log(max_d), num_depth_bins
        ))
        self.register_buffer('depth_bins', depth_bins)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [B*N, C, H, W] 2D特征图

        Returns:
            depth_logits: [B*N, D, H, W] 深度分布logits (用于监督)
            depth_probs: [B*N, D, H, W] 深度概率分布
            depth_pred: [B*N, H, W] 期望深度值 (用于可视化/监督)
        """
        # 预测深度分布
        depth_logits = self.depth_net(features)  # [B*N, D, H, W]
        depth_probs = F.softmax(depth_logits, dim=1)  # [B*N, D, H, W]

        # 计算期望深度 (软argmax)
        depth_bins = self.depth_bins.view(1, -1, 1, 1)  # [1, D, 1, 1]
        depth_pred = (depth_probs * depth_bins).sum(dim=1)  # [B*N, H, W]

        return depth_logits, depth_probs, depth_pred


class LiftSplatModule(nn.Module):
    """
    Lift-Splat-Shoot 核心模块

    将2D特征通过深度分布"提升"到3D空间，然后"溅射"到BEV网格

    关键创新:
    - 深度概率作为权重，实现软投影
    - 不需要精确的深度值，深度分布足够
    - 端到端可微分
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_depth_bins: int = 64,
        depth_range: Tuple[float, float] = (0.5, 80.0),
        bev_size: Tuple[int, int] = (128, 128),
        pc_range: List[float] = [-40, -40, -1, 40, 40, 5.4],
        num_cameras: int = 8,
        image_size: Tuple[int, int] = (960, 1280),
        patch_size: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_depth_bins = num_depth_bins
        self.depth_range = depth_range
        self.bev_h, self.bev_w = bev_size
        self.pc_range = pc_range
        self.num_cameras = num_cameras
        self.image_size = image_size  # Store image_size for pixel grid computation

        # 特征尺寸 (patch embedding 后)
        self.feat_h = image_size[0] // patch_size
        self.feat_w = image_size[1] // patch_size

        # 深度分布预测
        self.depth_net = DepthDistributionNet(
            in_channels=in_channels,
            num_depth_bins=num_depth_bins,
            depth_range=depth_range,
        )

        # 特征压缩 (可选，如果通道数太大)
        self.feature_proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        # BEV 特征聚合 (处理重叠区域)
        self.bev_aggregator = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # 预计算深度bin中心
        min_d, max_d = depth_range
        depth_bins = torch.exp(torch.linspace(
            math.log(min_d), math.log(max_d), num_depth_bins
        ))
        self.register_buffer('depth_bins', depth_bins)

        # 预计算 BEV 网格坐标
        self._init_bev_grid()
        
        # 预计算像素网格 (用于 Lift)
        self._precompute_pixel_grid()

    def _precompute_pixel_grid(self):
        """预计算像素坐标网格 (H, W, 3)"""
        # 注意: 这里使用特征图尺寸 feat_h, feat_w
        # 对应 patch embedding 后的尺寸
        H, W = self.feat_h, self.feat_w
        
        # 生成网格 (u, v)
        # 注意: 加上 0.5 是为了取像素中心
        # 但通常 patch embedding 后的特征对应的是 patch 的中心
        # 如果 patch_size=16, 第一个特征对应原图 (8, 8)
        # 这里我们生成归一化的 patch 坐标，或者对应的原图坐标
        # 简单起见，我们生成 patch 坐标，并假设内参已经缩放适配了特征图尺寸
        # 或者在使用时调整内参
        
        # 为了通用性，我们生成 0..W-1, 0..H-1 的坐标
        # 外部传入的 intrinsics 应该是针对原图的
        # 所以我们需要把 patch 坐标映射回原图坐标
        
        ys, xs = torch.meshgrid(
            torch.arange(H, dtype=torch.float32), 
            torch.arange(W, dtype=torch.float32), 
            indexing='ij'
        )
        
        # 映射回原图坐标: (idx + 0.5) * patch_size
        xs = (xs + 0.5) * self.image_size[1] / W
        ys = (ys + 0.5) * self.image_size[0] / H
        
        # 齐次坐标 (u, v, 1)
        ones = torch.ones_like(xs)
        pixel_grid = torch.stack([xs, ys, ones], dim=-1) # [H, W, 3]
        
        self.register_buffer('pixel_grid', pixel_grid)

    def _init_bev_grid(self):
        """预计算 BEV 网格的物理坐标"""
        x_range = self.pc_range[3] - self.pc_range[0]
        y_range = self.pc_range[4] - self.pc_range[1]

        # BEV 网格中心坐标
        x = torch.linspace(
            self.pc_range[0] + x_range / (2 * self.bev_w),
            self.pc_range[3] - x_range / (2 * self.bev_w),
            self.bev_w
        )
        y = torch.linspace(
            self.pc_range[1] + y_range / (2 * self.bev_h),
            self.pc_range[4] - y_range / (2 * self.bev_h),
            self.bev_h
        )

        xx, yy = torch.meshgrid(x, y, indexing='xy')  # [W, H]
        bev_coords = torch.stack([xx, yy], dim=-1)  # [W, H, 2]
        self.register_buffer('bev_coords', bev_coords)

    def forward(
        self,
        features: torch.Tensor,
        camera_intrinsics: Optional[torch.Tensor] = None,
        camera_extrinsics: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [B, N, C, H, W] 多相机特征
            camera_intrinsics: [B, N, 3, 3] 相机内参 (可选，用于精确投影)
            camera_extrinsics: [B, N, 4, 4] 相机外参 (可选，用于精确投影)

        Returns:
            bev_features: [B, C, bev_h, bev_w] BEV特征
            depth_logits: [B, N, D, H, W] 深度分布logits (用于监督)
            depth_pred: [B, N, H, W] 预测深度值
        """
        B, N, C, H, W = features.shape
        device = features.device

        # 1. 预测每个相机的深度分布
        features_flat = features.view(B * N, C, H, W)
        depth_logits, depth_probs, depth_pred = self.depth_net(features_flat)

        # Reshape 深度输出
        depth_logits = depth_logits.view(B, N, self.num_depth_bins, H, W)
        depth_probs = depth_probs.view(B, N, self.num_depth_bins, H, W)
        depth_pred = depth_pred.view(B, N, H, W)

        # 2. 特征投影
        proj_features = self.feature_proj(features_flat)  # [B*N, C', H, W]
        proj_features = proj_features.view(B, N, self.out_channels, H, W)

        # 3. Lift: 特征 × 深度概率 = 加权特征
        # [B, N, C', H, W] × [B, N, D, H, W] → [B, N, C', D, H, W]
        # 注意: 这里我们简化为在深度维度上加权求和，得到单一的加权特征
        # 完整的 LSS 会保留 D 维度，但显存开销大

        # 简化版: 使用深度期望作为软采样点
        # 这保留了深度信息，同时避免了 O(D) 的显存开销

        # 4. Splat: 根据深度将特征投影到 BEV
        bev_features = self._splat_to_bev(
            proj_features, depth_probs, depth_pred,
            camera_intrinsics, camera_extrinsics
        )

        # 5. BEV 聚合
        bev_features = self.bev_aggregator(bev_features)

        return bev_features, depth_logits, depth_pred

    def _splat_to_bev(
        self,
        features: torch.Tensor,
        depth_probs: torch.Tensor,
        depth_pred: torch.Tensor,
        intrinsics: Optional[torch.Tensor],
        extrinsics: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        将特征投影到 BEV 网格 (真正的 Lift-Splat 实现)
        
        Args:
            features: [B, N, C, H, W] 投影后的特征
            depth_probs: [B, N, D, H, W] 深度概率
            depth_pred: [B, N, H, W] 预测深度 (未使用，但保持接口)
            intrinsics: [B, N, 3, 3] 相机内参
            extrinsics: [B, N, 4, 4] 相机外参

        Returns:
            bev_features: [B, C, bev_h, bev_w]
        """
        if intrinsics is None or extrinsics is None:
            # 如果没有相机参数，回退到原来的简单逻辑 (或者报错)
            # 为了兼容性，这里我们假设必须提供参数，否则无法做几何投影
            # 但为了不崩，我们可以暂时 warn 并返回全 0
            return torch.zeros(features.shape[0], features.shape[2], self.bev_h, self.bev_w, device=features.device)

        B, N, C, H, W = features.shape
        D = self.num_depth_bins
        device = features.device

        # 初始化 BEV 特征和计数
        bev_features = torch.zeros(B, C, self.bev_h, self.bev_w, device=device)
        bev_counts = torch.zeros(B, 1, self.bev_h, self.bev_w, device=device)

        # 像素网格 [H, W, 3] -> [1, 1, 1, H, W, 3]
        # self.pixel_grid 在 _precompute_pixel_grid 中生成
        pixel_grid = self.pixel_grid.to(device).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # 深度 bins [D] -> [1, 1, D, 1, 1, 1]
        depth_bins = self.depth_bins.to(device).view(1, 1, D, 1, 1, 1)

        # 预计算一些常量
        bev_x_min, bev_y_min = self.pc_range[0], self.pc_range[1]
        bev_x_max, bev_y_max = self.pc_range[3], self.pc_range[4]
        bev_res_x = (bev_x_max - bev_x_min) / self.bev_w
        bev_res_y = (bev_y_max - bev_y_min) / self.bev_h

        # 优化: 保留外层 batch 循环，内层向量化相机维度 N
        # 这样可以利用 GPU 并行处理所有相机，同时显存增加可控
        
        # 预先扩展 pixel_grid 到 [N, H, W, 3]
        # self.pixel_grid: [H, W, 3]
        pixel_grid_n = self.pixel_grid.to(device).unsqueeze(0).expand(N, -1, -1, -1) # [N, H, W, 3]
        
        # 预先扩展 depth_bins 到 [N, D, 1, 1, 1]
        depth_bins_n = self.depth_bins.to(device).view(1, D, 1, 1, 1).expand(N, -1, -1, -1, -1)

        for b in range(B):
            # 1. 准备当前 batch 的数据 (向量化 N)
            # features: [B, N, C, H, W] -> [N, C, H, W] -> [N, H, W, C]
            feat_b = features[b].permute(0, 2, 3, 1).contiguous() 
            probs_b = depth_probs[b]  # [N, D, H, W]
            K_b = intrinsics[b]       # [N, 3, 3]
            ext_b = extrinsics[b]     # [N, 4, 4]

            # 2. Lift: 生成 3D 点 (N 个相机并行)
            # 反投影到相机坐标系
            # K_inv: [N, 3, 3]
            K_inv = torch.inverse(K_b)
            
            # ray_dirs: [N, H, W, 3] = pixel_grid_n @ K_inv.T
            # einsum 'nhwk,nlk->nhwl'
            ray_dirs = torch.einsum('nhwk,nlk->nhwl', pixel_grid_n, K_inv)
            
            # points_cam: [N, D, H, W, 3]
            points_cam = ray_dirs.unsqueeze(1) * depth_bins_n

            # 转到世界坐标系
            # points_world = R * points_cam + T
            rot = ext_b[:, :3, :3]  # [N, 3, 3]
            trans = ext_b[:, :3, 3] # [N, 3]
            
            # points_world: [N, D, H, W, 3]
            points_world = torch.einsum('ndhwk,nlk->ndhwl', points_cam, rot) + trans.view(N, 1, 1, 1, 3)

            # 3. Splat: 计算 BEV 坐标
            bev_x = ((points_world[..., 0] - bev_x_min) / bev_res_x).long()
            bev_y = ((points_world[..., 1] - bev_y_min) / bev_res_y).long()

            # 有效掩码 [N, D, H, W]
            valid = (bev_x >= 0) & (bev_x < self.bev_w) & \
                    (bev_y >= 0) & (bev_y < self.bev_h) & \
                    (points_world[..., 2] > self.pc_range[2]) # z > min_z

            # 4. 向量化累加 (Flatten N*D*H*W)
            # 为了避免显存爆炸，我们也可以选择在这里 flatten
            # 总点数: N * D * H * W = 8 * 64 * 60 * 80 ≈ 2.4M (可接受)
            
            valid_mask = valid.view(-1) # [N*D*H*W]
            if not valid_mask.any():
                continue
                
            # 展平坐标和权重
            bev_x_flat = bev_x.view(-1)[valid_mask]
            bev_y_flat = bev_y.view(-1)[valid_mask]
            weights_flat = probs_b.view(-1)[valid_mask] # [num_valid]
            
            # 展平特征 [N, H, W, C] -> [N, 1, H, W, C] -> expand D -> [N, D, H, W, C]
            # 注意: 特征对所有 D 是共享的
            feat_flat = feat_b.unsqueeze(1).expand(-1, D, -1, -1, -1).reshape(-1, C)[valid_mask] # [num_valid, C]
            
            # 加权特征
            weighted_feat = feat_flat * weights_flat.unsqueeze(1) # [num_valid, C]
            
            # 计算平铺索引
            flat_indices = bev_y_flat * self.bev_w + bev_x_flat # [num_valid]
            
            # 累加到 BEV
            # bev_features[b]: [C, bev_h, bev_w] -> view [C, -1]
            bev_flat = bev_features[b].view(C, -1)
            count_flat = bev_counts[b].view(1, -1)
            
            # index_add_: dim, index, source (source 需要转置为 [C, num_valid])
            bev_flat.index_add_(1, flat_indices, weighted_feat.t())
            count_flat.index_add_(1, flat_indices, weights_flat.unsqueeze(0))

        # 归一化
        bev_features = bev_features / (bev_counts + 1e-6)

        return bev_features


class DepthAwareFusion(nn.Module):
    """
    深度感知的特征融合模块

    替代原来的简单 fusion_proj:
    1. 深度作为几何先验参与融合
    2. 加权融合而非简单拼接
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_cameras: int = 8,
        num_depth_bins: int = 64,
        depth_range: Tuple[float, float] = (0.5, 80.0),
    ):
        super().__init__()
        self.num_cameras = num_cameras

        # 深度分布预测 (共享)
        self.depth_net = DepthDistributionNet(
            in_channels=in_channels,
            num_depth_bins=num_depth_bins,
            depth_range=depth_range,
        )

        # 基于深度的特征加权
        self.depth_weight_net = nn.Sequential(
            nn.Conv2d(num_depth_bins, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

        # 特征融合投影
        self.fusion_proj = nn.Linear(in_channels * num_cameras, out_channels)

    def forward(
        self,
        camera_features: List[torch.Tensor],
        spatial_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            camera_features: list of [B, L, C] tokens for each camera
            spatial_shape: (H, W) 特征图的空间尺寸 (可选，用于非正方形情况)

        Returns:
            fused_features: [B, L, C_out] 融合后的特征
            depth_logits: [B, N, D, H, W] 深度分布 (用于监督)
            depth_pred: [B, N, H, W] 预测深度
        """
        B = camera_features[0].shape[0]
        L = camera_features[0].shape[1]
        C = camera_features[0].shape[2]
        N = len(camera_features)

        # 计算空间尺寸
        if spatial_shape is not None:
            H, W = spatial_shape
        else:
            # 假设是正方形，或者从 L 推断
            # 对于 960x1280 图像，patch_size=16，特征图是 60x80
            # L = 60 * 80 = 4800
            if L == 4800:
                H, W = 60, 80
            else:
                # 尝试正方形
                H = W = int(L ** 0.5)
                if H * W != L:
                    # 尝试常见比例 3:4
                    import math
                    for ratio_h, ratio_w in [(3, 4), (9, 16), (2, 3), (1, 1)]:
                        scale = math.sqrt(L / (ratio_h * ratio_w))
                        H = int(ratio_h * scale)
                        W = int(ratio_w * scale)
                        if H * W == L:
                            break
                    else:
                        raise ValueError(f"Cannot infer spatial shape from L={L}")

        all_depth_logits = []
        all_depth_pred = []
        all_weighted_features = []

        for cam_idx, cam_tokens in enumerate(camera_features):
            # [B, L, C] -> [B, C, H, W]
            cam_feat = cam_tokens.transpose(1, 2).reshape(B, C, H, W)

            # 预测深度分布
            depth_logits, depth_probs, depth_pred = self.depth_net(cam_feat)
            all_depth_logits.append(depth_logits)
            all_depth_pred.append(depth_pred)

            # 计算深度权重
            depth_weight = self.depth_weight_net(depth_logits)  # [B, 1, H, W]

            # 加权特征
            weighted_feat = cam_feat * depth_weight
            all_weighted_features.append(weighted_feat)

        # 堆叠深度输出
        depth_logits = torch.stack(all_depth_logits, dim=1)  # [B, N, D, H, W]
        depth_pred = torch.stack(all_depth_pred, dim=1)  # [B, N, H, W]

        # 融合加权特征
        # 方式1: 拼接后投影 (保持原有接口)
        stacked_features = torch.cat(
            [f.flatten(2).transpose(1, 2) for f in all_weighted_features],
            dim=-1
        )  # [B, L, C*N]

        fused_features = self.fusion_proj(stacked_features)  # [B, L, C_out]

        return fused_features, depth_logits, depth_pred


class EdgeAwareDepthLoss(nn.Module):
    """
    边缘感知深度损失

    在物体边缘处，深度不连续是正常的，不应该惩罚
    """

    def __init__(self, depth_range: Tuple[float, float] = (0.5, 80.0), eps: float = 1e-6):
        super().__init__()
        self.min_depth = depth_range[0]
        self.max_depth = depth_range[1]
        self.eps = eps

    def forward(
        self,
        depth_pred: torch.Tensor,
        depth_gt: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            depth_pred: [B, N, H, W] 预测深度
            depth_gt: [B, N, H, W] 真值深度
            images: [B, N, C, H, W] 原始图像 (用于计算边缘权重)
            valid_mask: [B, N, H, W] 有效掩码

        Returns:
            loss: 深度损失标量
        """
        # 裁剪深度到有效范围
        depth_pred = depth_pred.clamp(self.min_depth, self.max_depth)
        depth_gt = depth_gt.clamp(self.min_depth, self.max_depth)

        # Log 空间 L1 损失 (对近距离更敏感)
        log_pred = torch.log(depth_pred + self.eps)
        log_gt = torch.log(depth_gt + self.eps)
        base_loss = torch.abs(log_pred - log_gt)

        # 边缘感知平滑损失 (可选)
        smooth_loss = torch.tensor(0.0, device=depth_pred.device)
        if images is not None:
            B, N, C, H_img, W_img = images.shape
            _, _, H_pred, W_pred = depth_pred.shape

            # 下采样图像到深度预测尺寸
            if H_img != H_pred or W_img != W_pred:
                images_down = F.interpolate(
                    images.view(B * N, C, H_img, W_img),
                    size=(H_pred, W_pred),
                    mode='bilinear',
                    align_corners=False
                ).view(B, N, C, H_pred, W_pred)
            else:
                images_down = images

            # 计算图像梯度 (边缘)
            img_gray = images_down.mean(dim=2)  # [B, N, H, W]

            grad_x = torch.abs(img_gray[:, :, :, 1:] - img_gray[:, :, :, :-1])
            grad_y = torch.abs(img_gray[:, :, 1:, :] - img_gray[:, :, :-1, :])

            # 边缘权重: 边缘处权重小
            edge_weight_x = torch.exp(-grad_x * 10)
            edge_weight_y = torch.exp(-grad_y * 10)

            # 深度梯度
            depth_grad_x = torch.abs(depth_pred[:, :, :, 1:] - depth_pred[:, :, :, :-1])
            depth_grad_y = torch.abs(depth_pred[:, :, 1:, :] - depth_pred[:, :, :-1, :])

            # 边缘感知平滑损失
            smooth_loss = (edge_weight_x * depth_grad_x).mean() + \
                         (edge_weight_y * depth_grad_y).mean()

        # 应用有效掩码
        if valid_mask is not None:
            valid_mask = valid_mask & (depth_gt > self.min_depth) & (depth_gt < self.max_depth)
            valid_mask = valid_mask.float()
            base_loss = (base_loss * valid_mask).sum() / (valid_mask.sum() + self.eps)
        else:
            base_loss = base_loss.mean()

        return base_loss + 0.1 * smooth_loss
