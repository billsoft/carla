# OccNetV3 进阶优化方案详解

> **写在前面**：这三个优化不是"必须做"，而是"做了更好"。当前网络已经可以训练，这些是锦上添花。

---

## 目录

1. [优化1: 增加时序帧数](#一优化1-增加时序帧数)
2. [优化2: 射线方向编码 (重点)](#二优化2-射线方向编码-重点)
3. [优化3: 距离感知损失](#三优化3-距离感知损失)
4. [三种编码的关系](#四三种编码的关系)

---

## 一、优化1: 增加时序帧数

### 1.1 为什么 2 帧不够？

先做个简单计算：

```
假设:
- 帧率: 10 FPS (每帧 100ms)
- 车速: 60 km/h = 16.7 m/s
- 行人速度: 5 km/h = 1.4 m/s

2帧时序窗口 (100ms):
- 自车移动: 16.7 × 0.1 = 1.67m
- 行人移动: 1.4 × 0.1 = 0.14m

问题:
- 如果行人被遮挡 1 帧，只有 1 帧历史信息
- 高速运动物体 (摩托车 80km/h) 移动 2.2m，可能跨越多个体素
```

**2帧 = 只能看到"刚刚发生了什么"，无法建立运动趋势。**

### 1.2 3-4 帧能带来什么？

```
3帧时序 (200ms):
- 可以计算加速度 (需要至少3个点)
- 遮挡容忍度提升
- 运动轨迹更平滑

4帧时序 (300ms):
- 接近人类反应时间
- 可以预测短期未来位置
- Tesla 据传使用 ~500ms 窗口
```

### 1.3 代码修改

```python
# configs/default.py
num_frames = 3  # 或 4

# models/temporal.py - 需要小改
class LightweightTemporalFusion(nn.Module):
    def __init__(self, dim, num_frames, ...):
        # 多帧融合改为加权平均或注意力
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads=4)
        
    def forward(self, current_bev, ego_motion, current_pose):
        if len(self.history) < self.num_frames - 1:
            self._update_history(current_bev, current_pose)
            return current_bev
        
        # 对齐所有历史帧
        aligned_history = []
        for i, (hist_bev, hist_pose) in enumerate(zip(self.history, self.history_poses)):
            rel_pose = compute_relative_pose(current_pose, hist_pose)
            aligned = self.motion_comp(hist_bev, rel_pose)
            aligned_history.append(aligned)
        
        # 时序注意力融合 (比简单拼接更好)
        all_frames = torch.stack([current_bev] + aligned_history, dim=1)  # [B, T, C, H, W]
        B, T, C, H, W = all_frames.shape
        
        # Flatten spatial → 注意力 → Reshape
        all_frames_flat = all_frames.view(B, T, C * H * W).permute(1, 0, 2)  # [T, B, C*H*W]
        fused, _ = self.temporal_attn(
            all_frames_flat[-1:],  # query: 当前帧
            all_frames_flat,       # key: 所有帧
            all_frames_flat        # value: 所有帧
        )
        
        return fused.permute(1, 0, 2).view(B, C, H, W)
```

### 1.4 显存影响

| 配置 | 额外显存 | 建议 |
|:----|:--------|:----|
| 2→3 帧 | +~100MB | ✅ 推荐 |
| 2→4 帧 | +~200MB | 🟡 可选 |
| 2→5 帧 | +~300MB | ⚠️ 谨慎 |

---

## 二、优化2: 射线方向编码 (重点)

### 2.1 先讲一个故事

想象你是一个盲人，手里拿着 8 根不同方向的"探测棒"（8个相机）。

每根棒上有很多"触点"（像素），当触点碰到物体时会告诉你"这里有东西"。

**问题来了**：你怎么知道"这里"是 3D 空间的哪个位置？

- **没有射线编码**：你只知道"第3根棒的第47个触点碰到了东西"
- **有射线编码**：你知道"从我肩膀出发，朝向东北偏上 23° 的射线，在某个距离处碰到了东西"

**射线方向编码，就是告诉模型每个像素"指向"3D空间的哪个方向。**

### 2.2 数学原理

#### 2.2.1 相机成像模型回顾

```
3D点 P = (X, Y, Z)  →  2D像素 p = (u, v)

投影公式:
┌   ┐   ┌           ┐ ┌   ┐
│ u │   │ fx  0  cx │ │X/Z│
│ v │ = │ 0  fy  cy │ │Y/Z│
│ 1 │   │ 0   0   1 │ │ 1 │
└   ┘   └           ┘ └   ┘

其中:
- (fx, fy): 焦距 (像素单位)
- (cx, cy): 主点 (图像中心)
```

#### 2.2.2 反投影：像素 → 射线方向

**关键洞察**：给定一个像素 (u, v)，我们不知道深度 Z，但我们知道**射线方向**！

```
射线方向公式:

        ┌           ┐
        │ (u - cx)/fx │
d(u,v) = │ (v - cy)/fy │
        │      1      │
        └           ┘

然后归一化:
d_normalized = d / ||d||
```

**直观理解**：
- 图像中心 (cx, cy) → 射线指向正前方 (0, 0, 1)
- 图像左边 (u < cx) → 射线偏左
- 图像上边 (v < cy) → 射线偏上

#### 2.2.3 考虑相机外参

上面的射线是在**相机坐标系**中。要转到**车辆坐标系**，需要乘以旋转矩阵：

```
d_world = R_camera @ d_camera

其中 R_camera 是相机的旋转矩阵 (从 config.cameras 获取)
```

### 2.3 一个具体的数值例子

```python
# 假设前视相机参数
fx = fy = 800  # 焦距
cx, cy = 640, 480  # 图像中心 (1280×960 图像)
FOV = 50°

# 计算三个特殊像素的射线方向

# 像素1: 图像中心 (640, 480)
d1 = [(640-640)/800, (480-480)/800, 1] = [0, 0, 1]
d1_norm = [0, 0, 1]  # 指向正前方 ✓

# 像素2: 图像左边缘 (0, 480)  
d2 = [(0-640)/800, (480-480)/800, 1] = [-0.8, 0, 1]
||d2|| = sqrt(0.64 + 0 + 1) = 1.28
d2_norm = [-0.625, 0, 0.781]  # 指向左前方 ✓

# 像素3: 图像右上角 (1280, 0)
d3 = [(1280-640)/800, (0-480)/800, 1] = [0.8, -0.6, 1]
||d3|| = sqrt(0.64 + 0.36 + 1) = 1.41
d3_norm = [0.567, -0.425, 0.709]  # 指向右上前方 ✓

# 转换到角度 (更直观)
# d1: yaw=0°, pitch=0°   (正前方)
# d2: yaw=-39°, pitch=0° (左前方39度)
# d3: yaw=39°, pitch=-31° (右上方)
```

### 2.4 为什么这很重要？

#### 场景1: 同一物体在不同相机中

```
一辆车在你的右前方 45°

前视相机看到: 图像右边缘的一团像素
右侧相机看到: 图像左边缘的一团像素

没有射线编码:
- 模型: "前视相机右边有东西，右侧相机左边有东西...是两个物体？"

有射线编码:
- 模型: "前视相机指向(0.7, 0.7, 0)的射线有东西"
        "右侧相机指向(0.7, 0.7, 0)的射线也有东西"
        "是同一个物体！"
```

#### 场景2: 深度估计的隐式约束

```
射线方向 + 已知车辆在地面 → 可以推断距离

例如:
- 射线方向: (0, -0.3, 0.95)  (略微向下)
- 车辆轮胎必然在地面 (Z ≈ 0)
- 地面交点: Z / 0.95 = 0 / (-0.3) × distance
- 可以大致估算距离
```

### 2.5 代码实现

```python
# models/position_encoding.py - 添加射线方向编码

class RayDirectionEncoding(nn.Module):
    """
    射线方向编码
    
    将每个像素的 3D 射线方向编码为特征向量
    """
    
    def __init__(
        self, 
        dim: int,
        image_size: Tuple[int, int],
        camera_configs: Dict,
        patch_size: int = 16,
        temperature: float = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.image_size = image_size
        self.patch_size = patch_size
        
        # 预计算每个相机的射线方向
        self.ray_directions = nn.ParameterDict()
        
        for cam_name, cfg in camera_configs.items():
            cam_id = cfg['id']
            rays = self._compute_ray_directions(
                cfg['fov'], 
                cfg['rotation'],
                image_size,
                patch_size
            )
            # rays: [H_patches, W_patches, 3]
            self.register_buffer(f'rays_{cam_id}', rays)
        
        # 射线方向 → 特征向量
        # 使用 MLP 而非正弦编码，因为射线方向是连续的
        self.ray_mlp = nn.Sequential(
            nn.Linear(3, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim),
        )
        
        # 或者使用正弦编码 (更轻量)
        self.use_sinusoidal = True
        if self.use_sinusoidal:
            inv_freq = 1.0 / (temperature ** (torch.arange(0, dim, 6).float() / dim))
            self.register_buffer('inv_freq', inv_freq)
    
    def _compute_ray_directions(
        self, 
        fov: float, 
        rotation: List[float],
        image_size: Tuple[int, int],
        patch_size: int
    ) -> torch.Tensor:
        """
        计算 patch 中心点的射线方向
        
        Returns:
            rays: [H_patches, W_patches, 3] 归一化射线方向
        """
        H, W = image_size
        H_p, W_p = H // patch_size, W // patch_size
        
        # 计算内参
        fx = W / (2 * math.tan(math.radians(fov / 2)))
        fy = fx  # 假设正方形像素
        cx, cy = W / 2, H / 2
        
        # Patch 中心坐标
        u = torch.linspace(patch_size/2, W - patch_size/2, W_p)
        v = torch.linspace(patch_size/2, H - patch_size/2, H_p)
        vv, uu = torch.meshgrid(v, u, indexing='ij')  # [H_p, W_p]
        
        # 相机坐标系下的射线方向
        dx = (uu - cx) / fx
        dy = (vv - cy) / fy
        dz = torch.ones_like(dx)
        
        rays_cam = torch.stack([dx, dy, dz], dim=-1)  # [H_p, W_p, 3]
        
        # 归一化
        rays_cam = rays_cam / rays_cam.norm(dim=-1, keepdim=True)
        
        # 转换到车辆坐标系
        R = self._rotation_matrix(rotation)  # [3, 3]
        rays_world = torch.einsum('ij,hwj->hwi', R, rays_cam)
        
        return rays_world
    
    def _rotation_matrix(self, rotation: List[float]) -> torch.Tensor:
        """
        从欧拉角计算旋转矩阵
        rotation: [pitch, roll, yaw] in degrees
        """
        pitch, roll, yaw = [math.radians(r) for r in rotation]
        
        # Rz @ Ry @ Rx
        Rx = torch.tensor([
            [1, 0, 0],
            [0, math.cos(pitch), -math.sin(pitch)],
            [0, math.sin(pitch), math.cos(pitch)]
        ], dtype=torch.float32)
        
        Ry = torch.tensor([
            [math.cos(roll), 0, math.sin(roll)],
            [0, 1, 0],
            [-math.sin(roll), 0, math.cos(roll)]
        ], dtype=torch.float32)
        
        Rz = torch.tensor([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        return Rz @ Ry @ Rx
    
    def _sinusoidal_encode(self, rays: torch.Tensor) -> torch.Tensor:
        """
        正弦编码射线方向
        
        rays: [..., 3] 射线方向向量
        returns: [..., dim] 编码后特征
        """
        # rays: [B, N, 3] or [H, W, 3]
        shape = rays.shape[:-1]
        rays_flat = rays.view(-1, 3)  # [*, 3]
        
        # 每个维度单独编码
        # freq: [dim // 6]
        # rays_flat: [*, 3]
        # 输出: [*, 3, dim // 6, 2] → [*, dim]
        
        encodings = []
        for i in range(3):  # x, y, z
            coord = rays_flat[:, i:i+1]  # [*, 1]
            freq = coord * self.inv_freq  # [*, dim//6]
            enc = torch.cat([freq.sin(), freq.cos()], dim=-1)  # [*, dim//3]
            encodings.append(enc)
        
        encoded = torch.cat(encodings, dim=-1)  # [*, dim]
        
        # 如果 dim 不能被 6 整除，截断或补零
        if encoded.shape[-1] > self.dim:
            encoded = encoded[..., :self.dim]
        elif encoded.shape[-1] < self.dim:
            padding = torch.zeros(*encoded.shape[:-1], self.dim - encoded.shape[-1], 
                                  device=encoded.device, dtype=encoded.dtype)
            encoded = torch.cat([encoded, padding], dim=-1)
        
        return encoded.view(*shape, self.dim)
    
    def forward(self, camera_id: int, batch_size: int) -> torch.Tensor:
        """
        获取指定相机的射线编码
        
        Args:
            camera_id: 相机ID
            batch_size: batch大小
            
        Returns:
            ray_encoding: [B, H_p * W_p, dim]
        """
        rays = getattr(self, f'rays_{camera_id}')  # [H_p, W_p, 3]
        H_p, W_p, _ = rays.shape
        
        # 编码
        if self.use_sinusoidal:
            encoded = self._sinusoidal_encode(rays)  # [H_p, W_p, dim]
        else:
            encoded = self.ray_mlp(rays)  # [H_p, W_p, dim]
        
        # Flatten 并 batch expand
        encoded = encoded.view(H_p * W_p, self.dim)  # [N, dim]
        encoded = encoded.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, dim]
        
        return encoded


class EnhancedCameraPositionEncoding(nn.Module):
    """
    增强版位置编码 = 原有编码 + 射线方向编码
    """
    
    def __init__(self, dim, num_cameras, image_size, camera_configs, patch_size=16):
        super().__init__()
        
        # 原有编码
        self.base_encoder = MultiCameraPositionEncoding(
            dim, num_cameras, image_size, camera_configs, patch_size
        )
        
        # 新增: 射线方向编码
        self.ray_encoder = RayDirectionEncoding(
            dim, image_size, camera_configs, patch_size
        )
        
        # 融合层 (可选: 直接加，或用门控)
        self.use_gate = True
        if self.use_gate:
            self.gate = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.Sigmoid()
            )
            self.fuse = nn.Linear(dim * 2, dim)
    
    def forward(self, x: torch.Tensor, camera_id: int) -> torch.Tensor:
        """
        添加位置编码到特征
        
        Args:
            x: [B, N, dim] 特征
            camera_id: 相机ID
        """
        B, N, D = x.shape
        
        # 原有编码
        x_base = self.base_encoder.add_pixel_pe(x)
        
        # 射线编码
        ray_enc = self.ray_encoder(camera_id, B)  # [B, N, dim]
        
        if self.use_gate:
            # 门控融合
            concat = torch.cat([x_base, ray_enc], dim=-1)  # [B, N, 2*dim]
            gate = self.gate(concat)  # [B, N, dim], values in [0, 1]
            fused = self.fuse(concat)  # [B, N, dim]
            return x + gate * fused
        else:
            # 简单相加
            return x_base + ray_enc
    
    def encode_qk_single_camera(self, q, k, camera_id):
        """保持原有接口"""
        return self.base_encoder.encode_qk_single_camera(q, k, camera_id)
```

### 2.6 射线编码的效果可视化

```
假设 BEV 网格中有一个点 (x=10m, y=5m)

                    前视相机 FOV
                    ↙     ↓     ↘
                  /       |       \
                /    [x]  |        \  ← 物体在这里
              /           |          \
            /             |            \
      ----[车辆]----------------------------
           \              |              /
             \            |            /
               \   右侧相机 FOV     /
                 \        |       /
                   ↘      ↓     ↙

前视相机某像素的射线: d1 = (0.89, 0.45, 0.0)  ← 指向右前方
右侧相机某像素的射线: d2 = (0.89, 0.45, 0.0)  ← 也指向右前方 (相同！)

模型学到: 不同相机的像素，如果射线方向相同，可能看到同一物体
```

---

## 三、优化3: 距离感知损失

### 3.1 为什么近距离更重要？

**安全角度**：
```
10m 内漏检行人 → 0.6秒后可能撞上 (60km/h)
40m 内漏检行人 → 2.4秒后可能撞上 (有时间反应)

风险比例: 4:1 (保守估计)
实际应该: 10:1 或更高
```

**感知难度**：
```
近距离:
- 物体占用更多像素 → 更容易检测
- 但对精度要求更高 (1m 误差在 10m 距离很致命)

远距离:
- 物体很小 → 难检测
- 但容忍更大误差 (1m 误差在 40m 距离可接受)
```

### 3.2 距离权重函数设计

```python
# 指数衰减 (推荐)
weight(d) = exp(-d / λ) + base

其中:
- d: 距离 (m)
- λ: 衰减常数 (推荐 20-30)
- base: 基础权重 (推荐 0.5)

示例 (λ=20, base=0.5):
- d=0m:  weight = exp(0) + 0.5 = 1.5
- d=10m: weight = exp(-0.5) + 0.5 ≈ 1.1
- d=20m: weight = exp(-1) + 0.5 ≈ 0.87
- d=40m: weight = exp(-2) + 0.5 ≈ 0.64
```

### 3.3 代码实现

```python
# losses/losses.py - 添加距离感知损失

class DistanceAwareLoss(nn.Module):
    """
    距离感知损失加权
    
    近距离体素的损失权重更高
    """
    
    def __init__(
        self,
        voxel_size: Tuple[int, int, int] = (400, 400, 32),
        pc_range: List[float] = [-40, -40, -1, 40, 40, 5.4],
        decay_lambda: float = 20.0,
        base_weight: float = 0.5,
        max_weight: float = 3.0,
    ):
        super().__init__()
        
        self.decay_lambda = decay_lambda
        self.base_weight = base_weight
        self.max_weight = max_weight
        
        # 预计算距离权重图
        X, Y, Z = voxel_size
        
        # 体素中心坐标
        x_range = pc_range[3] - pc_range[0]  # 80m
        y_range = pc_range[4] - pc_range[1]  # 80m
        
        x = torch.linspace(pc_range[0] + x_range/(2*X), 
                          pc_range[3] - x_range/(2*X), X)
        y = torch.linspace(pc_range[1] + y_range/(2*Y), 
                          pc_range[4] - y_range/(2*Y), Y)
        
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        distance = torch.sqrt(xx**2 + yy**2)  # [X, Y]
        
        # 计算权重
        weight = torch.exp(-distance / decay_lambda) + base_weight
        weight = weight.clamp(max=max_weight)
        
        # 扩展到 Z 维度 (所有高度使用相同权重)
        weight = weight.unsqueeze(-1).expand(-1, -1, Z)  # [X, Y, Z]
        
        self.register_buffer('distance_weight', weight)
    
    def forward(
        self, 
        pred: torch.Tensor,      # [B, C, X, Y, Z]
        target: torch.Tensor,    # [B, X, Y, Z]
        base_loss_fn: nn.Module  # 基础损失函数
    ) -> torch.Tensor:
        """
        计算距离加权损失
        """
        B = pred.shape[0]
        
        # 计算逐体素损失
        # 先不 reduce
        loss_per_voxel = self._compute_unreduced_loss(pred, target, base_loss_fn)
        # loss_per_voxel: [B, X, Y, Z]
        
        # 应用距离权重
        weight = self.distance_weight.unsqueeze(0).expand(B, -1, -1, -1)
        weighted_loss = loss_per_voxel * weight
        
        # 归一化 (保持损失量级稳定)
        return weighted_loss.sum() / weight.sum()
    
    def _compute_unreduced_loss(self, pred, target, base_loss_fn):
        """计算不 reduce 的逐体素损失"""
        B, C, X, Y, Z = pred.shape
        
        # Reshape for cross entropy
        pred_flat = pred.permute(0, 2, 3, 4, 1).reshape(-1, C)
        target_flat = target.reshape(-1)
        
        loss_flat = F.cross_entropy(pred_flat, target_flat, reduction='none')
        
        return loss_flat.view(B, X, Y, Z)


class EnhancedOccLoss(nn.Module):
    """
    增强版 Occupancy Loss
    
    = Focal + Dice + Coarse-to-Fine + Distance-Aware
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.focal_loss = FocalLoss(
            alpha=config.focal_alpha,
            gamma=config.focal_gamma,
            class_weights=config.class_weights
        )
        
        self.dice_loss = DiceLoss()
        
        self.flow_loss = FlowLoss()
        
        self.distance_aware = DistanceAwareLoss(
            voxel_size=config.voxel_size,
            pc_range=config.pc_range,
            decay_lambda=20.0,
            base_weight=0.5,
        )
        
        # 损失权重
        self.w_focal = 0.4
        self.w_dice = 0.2
        self.w_distance = 0.2
        self.w_flow = config.flow_loss_weight
        self.w_coarse = config.coarse_loss_weight
    
    def forward(self, outputs, targets):
        losses = {}
        
        semantic_pred = outputs['semantic']
        semantic_gt = targets['semantic']
        
        # 1. Focal Loss
        losses['focal'] = self.focal_loss(semantic_pred, semantic_gt) * self.w_focal
        
        # 2. Dice Loss
        losses['dice'] = self.dice_loss(semantic_pred, semantic_gt) * self.w_dice
        
        # 3. Distance-Aware Loss (使用 CE 作为基础)
        losses['distance'] = self.distance_aware(
            semantic_pred, semantic_gt, 
            nn.CrossEntropyLoss(reduction='none')
        ) * self.w_distance
        
        # 4. Coarse Loss (如果有)
        if 'coarse_semantic' in outputs:
            coarse_gt = F.interpolate(
                semantic_gt.unsqueeze(1).float(),
                size=outputs['coarse_semantic'].shape[2:],
                mode='nearest'
            ).squeeze(1).long()
            losses['coarse'] = self.focal_loss(
                outputs['coarse_semantic'], coarse_gt
            ) * self.w_coarse
        
        # 5. Flow Loss (如果有)
        if 'flow' in outputs and 'flow' in targets:
            losses['flow'] = self.flow_loss(
                outputs['flow'], 
                targets['flow'],
                targets.get('flow_mask')
            ) * self.w_flow
        
        losses['total'] = sum(losses.values())
        return losses
```

### 3.4 距离权重可视化

```
俯视图 (BEV), 车辆在中心:

              ← 40m →
         ┌─────────────────┐
         │  0.64  0.64     │  ← 远距离: 权重 ~0.64
         │                 │
         │   0.87    0.87  │  ← 20m: 权重 ~0.87
    40m  │                 │
         │    1.1    1.1   │  ← 10m: 权重 ~1.1
         │                 │
         │      [车]  1.5  │  ← 近距离: 权重 ~1.5
         │                 │
         └─────────────────┘

颜色越深 = 权重越高 = 漏检惩罚越大
```

---

## 四、三种编码的关系

### 4.1 现有编码回顾

| 编码 | 作用 | 已有？ |
|:----|:----|:------|
| **Spatial 2D PE** | patch 在图像中的位置 | ✅ 有 |
| **Camera RoPE** | 相机朝向 (yaw 角度) | ✅ 有 |
| **FOV Hyperbolic** | FOV 差异 (尺度感知) | ✅ 有 |
| **Ray Direction** | 像素指向的 3D 方向 | ❌ **建议添加** |

### 4.2 它们是替换还是配合？

**答案：配合使用，不是替换！**

```
┌─────────────────────────────────────────────────────────────────┐
│                      位置编码层次结构                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Level 1: Spatial 2D PE                                         │
│  └── "这个 patch 在图像的哪个位置" (左上/右下/中心...)          │
│                                                                  │
│  Level 2: Camera RoPE                                           │
│  └── "这个相机朝向哪个方向" (前/左/右/后...)                    │
│                                                                  │
│  Level 3: FOV Hyperbolic                                        │
│  └── "这个相机是广角还是长焦" (120°/50°/35°...)                 │
│                                                                  │
│  Level 4: Ray Direction  ← 新增                                 │
│  └── "这个像素指向 3D 空间的哪个方向" (精确向量)                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

它们编码的信息是互补的:
- Level 1-3: 离散/粗糙的几何信息
- Level 4: 连续/精确的几何信息
```

### 4.3 融合方式

```python
# 方式1: 简单相加 (推荐，显存友好)
x = x + spatial_pe + camera_rope(x) + fov_enc(x) + ray_enc

# 方式2: 门控融合 (效果更好，但稍复杂)
all_pe = torch.cat([spatial_pe, camera_rope(x), fov_enc(x), ray_enc], dim=-1)
gate = sigmoid(linear(all_pe))
x = x + gate * linear(all_pe)

# 方式3: 分层注入 (最灵活)
# Spatial PE: 在 Patch Embed 之后加
# Camera RoPE: 在 Attention 的 Q/K 上加
# FOV Enc: 在 Attention 的 Q/K 上加
# Ray Direction: 在 Cross-Attention 时加 (BEV Query 时)
```

### 4.4 推荐配置

```python
# configs/default.py

# 位置编码配置
position_encoding = {
    'spatial_2d': True,        # 保持
    'camera_rope': True,       # 保持
    'fov_hyperbolic': True,    # 保持
    'ray_direction': True,     # 新增
    
    # 融合方式
    'fusion_method': 'add',    # 'add' 或 'gate'
    
    # Ray encoding 参数
    'ray_use_sinusoidal': True,  # True=轻量, False=MLP
}
```

---

## 五、总结

### 5.1 三个优化的性价比

| 优化 | 效果 | 代价 | 优先级 |
|:----|:----|:----|:------|
| 增加时序帧数 | ⭐⭐⭐ | ~100MB | 🟡 可选 |
| 射线方向编码 | ⭐⭐⭐⭐ | ~50MB | ✅ **推荐** |
| 距离感知损失 | ⭐⭐⭐ | ~0 | ✅ **推荐** |

### 5.2 实施顺序建议

```
Phase 1: 先用当前配置训练 baseline
         → 记录各类别 IoU, 特别是 pedestrian/bicycle

Phase 2: 添加距离感知损失 (零成本)
         → 预期近距离物体 IoU 提升 3-5%

Phase 3: 添加射线方向编码 (低成本)
         → 预期多视角融合质量提升

Phase 4: 增加时序帧数 (如有必要)
         → 针对高速动态物体
```

### 5.3 最后一句话

这三个优化都是**锦上添花**，不是**雪中送炭**。

你的网络底子很好，先跑起来，用数据说话，再决定要不要加这些优化。

**过早优化是万恶之源。** —— Donald Knuth

---

*有问题欢迎评论区讨论！*