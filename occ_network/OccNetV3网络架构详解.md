# OccNetV3 网络架构详解：从 8 个摄像头到 3D 世界的完整实现

> 本文详细解析 OccNetV3 的网络架构，一个面向自动驾驶场景的 3D 占用预测网络。如果你正在学习 BEV 感知、3D 重建或时序融合，这篇文章会帮你建立完整的技术认知。

---

## 一、问题定义：我们要做什么？

想象你坐在一辆自动驾驶汽车里，车身周围安装了 **8 个摄像头**，覆盖 360° 视野。网络的任务是：

```
输入: 8 张 2D 图像 (960×1280 灰度 RAW)
输出: 3D 体素网格 (400×400×32)，每个体素标注语义类别
```

用一个比喻来说：**把 8 张照片拼成一个 3D 乐高模型**，每块乐高积木都标注了"这是车"、"这是行人"、"这是道路"。

```
              ┌─────────────────────────────────────────┐
              │          OccNetV3 整体目标               │
              ├─────────────────────────────────────────┤
              │                                         │
              │   🎥 × 8                   🧊 400³      │
              │   ┌───┐                    ┌─────┐      │
              │   │cam│ ──────────────────→│体素 │      │
              │   └───┘   2D → 3D          │网格 │      │
              │                            └─────┘      │
              │   输入: 8 张 960×1280       输出: 3D    │
              │         灰度 RAW 图像            语义   │
              │                                         │
              └─────────────────────────────────────────┘
```

---

## 二、架构总览：八大模块流水线

OccNetV3 采用**端到端的编码器-解码器架构**，可以拆解为 8 个核心模块：

```
┌──────────────────────────────────────────────────────────────────────┐
│                        OccNetV3 数据流水线                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  [1. Patch Embed]  →  [2. Ray Encoding]  →  [3. Swin Encoder]       │
│       ↓                      ↓                     ↓                 │
│   图像切块            射线方向编码            特征提取                │
│  960×1280→60×80                                                      │
│                                                                      │
│  [4. Depth Fusion]  →  [5. BEV Decoder]  →  [6. Temporal Fusion]    │
│       ↓                      ↓                     ↓                 │
│  深度感知融合           BEV 特征生成           时序融合               │
│  (Lift-Splat)          128×128                Memory Cell            │
│                                                                      │
│  [7. Height Expand]  →  [8. Prediction Head]                        │
│       ↓                      ↓                                       │
│   2D→3D 提升             语义预测                                    │
│  BEV→Voxel              400×400×32                                   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 数据形状变化

| 阶段 | 输入形状 | 输出形状 | 说明 |
|------|----------|----------|------|
| 原始输入 | `[B, 8, 1, 960, 1280]` | - | 8 相机灰度图 |
| Patch Embed | `[B, 8, 1, 960, 1280]` | `8×[B, 4800, 192]` | 每相机 4800 tokens |
| Swin Encoder | `8×[B, 4800, 192]` | `8×[B, 4800, 192]` | 特征增强 |
| Depth Fusion | `[B, 8, 192, 60, 80]` | `[B, 192, 128, 128]` | 8 相机→BEV |
| BEV Decoder | `[B, 192, 128, 128]` | `[B, 192, 128, 128]` | BEV 精炼 |
| Temporal | `[B, 192, 128, 128]` | `[B, 192, 128, 128]` | 时序融合 |
| Height Expand | `[B, 192, 128, 128]` | `[B, 192, 128, 128, 8]` | 2D→3D |
| Upsampler | `[B, 192, 128, 128, 8]` | `[B, 96, 400, 400, 32]` | 上采样 |
| Head | `[B, 96, 400, 400, 32]` | `[B, 18, 400, 400, 32]` | 18 类语义 |

---

## 三、模块详解

### 3.1 Patch Embedding：图像切块

**目标**：将高分辨率图像切成小块，转换为 Transformer 可处理的 token 序列。

**核心思想**：不直接用单层卷积，而是用 **Hybrid CNN Stem**（4 阶段渐进下采样），更好地保留 Bayer RAW 图像的细节。

```python
# models/patch_embed.py

class HybridPatchEmbed(nn.Module):
    """
    Bayer RAW 专用的混合 Patch Embedding
    
    为什么不用简单的 Conv2d(stride=16)？
    - RAW 图像有 RGGB Bayer 模式，需要特殊处理
    - 渐进式下采样保留更多边缘信息
    """
    
    def __init__(self, img_size=(960, 1280), patch_size=16, in_channels=1, embed_dim=192):
        super().__init__()
        
        # Stage 1: RGGB 合并 (Stride 2)
        # [B, 1, H, W] → [B, 32, H/2, W/2]
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        
        # Stage 2: Stride 4
        # [B, 32, H/2, W/2] → [B, 64, H/4, W/4]
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        
        # Stage 3: Stride 8
        # [B, 64, H/4, W/4] → [B, 128, H/8, W/8]
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        
        # Stage 4: 投影层 (Stride 16)
        # [B, 128, H/8, W/8] → [B, embed_dim, H/16, W/16]
        self.proj = nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(embed_dim)
```

**数据流**：

```
原始图像: [B, 1, 960, 1280]
    ↓ stage1 (stride=2)
[B, 32, 480, 640]
    ↓ stage2 (stride=2)
[B, 64, 240, 320]
    ↓ stage3 (stride=2)
[B, 128, 120, 160]
    ↓ proj (stride=2)
[B, 192, 60, 80]
    ↓ flatten + LayerNorm
[B, 4800, 192]  ← 最终输出：4800 个 token，每个 192 维
```

---

### 3.2 Ray Direction Encoding：射线方向编码

**目标**：告诉网络每个像素对应的 3D 射线方向。

**核心创新**：**统一使用等距投影模型**，无论是针孔相机还是鱼眼相机，都映射到同一个球面坐标系。

```
┌─────────────────────────────────────────────────────────────┐
│                   统一等距投影模型                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│     小 FOV (35°)           大 FOV (120°)                    │
│     长焦相机                广角相机                         │
│                                                             │
│        ┌──┐                  ┌────────┐                     │
│        │  │                  │        │                     │
│        └──┘                  │        │                     │
│     球面小区域              │        │                     │
│                              └────────┘                     │
│                            球面大区域                        │
│                                                             │
│     数学公式: θ = r / f                                     │
│     - θ: 入射角                                             │
│     - r: 像素到图像中心的距离                                │
│     - f: 焦距 = W / FOV_rad                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
# models/position_encoding.py

class RayDirectionEncoding(nn.Module):
    """
    射线方向编码
    
    设计决策: 所有相机统一使用等距投影
    - 概念统一，无需区分 pinhole/fisheye
    - FOV 决定在球面上截取的区域大小
    - 小角度时 equidistant ≈ pinhole，误差可由网络学习补偿
    """
    
    def _compute_ray_directions(self, fov, rotation, image_size, patch_size):
        H, W = image_size
        H_p, W_p = H // patch_size, W // patch_size
        
        # Patch 中心坐标
        u = torch.linspace(patch_size/2, W - patch_size/2, W_p)
        v = torch.linspace(patch_size/2, H - patch_size/2, H_p)
        vv, uu = torch.meshgrid(v, u, indexing='ij')
        
        cx, cy = W / 2, H / 2
        
        # 1. 计算像素到光心的距离 r
        dx = uu - cx
        dy = vv - cy
        r = torch.sqrt(dx**2 + dy**2)
        phi_img = torch.atan2(dy, dx)  # 方位角
        
        # 2. 等距投影: θ = r / f
        fov_rad = math.radians(fov)
        f = W / fov_rad
        theta = r / f  # 入射角
        
        # 3. 球面坐标 → 笛卡尔射线方向
        ray_z = torch.cos(theta)
        sin_theta = torch.sin(theta)
        ray_x = sin_theta * torch.cos(phi_img)
        ray_y = sin_theta * torch.sin(phi_img)
        
        rays_cam = torch.stack([ray_x, ray_y, ray_z], dim=-1)
        rays_cam = rays_cam / rays_cam.norm(dim=-1, keepdim=True)
        
        # 4. 旋转到世界坐标系
        R = self._rotation_matrix(rotation)
        rays_world = torch.einsum('ij,hwj->hwi', R, rays_cam)
        
        return rays_world
```

**为什么这样设计？**

| 传统方案 | 本方案 |
|----------|--------|
| 针孔相机用 pinhole 模型 | 统一用等距投影 |
| 鱼眼相机用 equidistant | 统一用等距投影 |
| 需要配置每个相机的投影类型 | 只需配置 FOV |
| 多个位置编码模块 | 单一编码模块 |

---

### 3.3 Swin Transformer Encoder：特征提取

**目标**：对每个相机的 token 序列进行自注意力编码，提取丰富的语义特征。

**核心结构**：Window Attention + Shifted Window（经典 Swin Transformer 设计）

```
┌─────────────────────────────────────────────────────────────┐
│              Swin Transformer Encoder Layer                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    输入: [B, 4800, 192]                                     │
│          ↓                                                  │
│    ┌─────────────────┐                                      │
│    │  Window Attn    │  ← 8×8 窗口内自注意力                │
│    │  (不 shift)     │                                      │
│    └────────┬────────┘                                      │
│             ↓                                               │
│    ┌─────────────────┐                                      │
│    │  Window Attn    │  ← 窗口移动半格后自注意力            │
│    │  (shift 4)      │    解决窗口边界信息不流通问题        │
│    └────────┬────────┘                                      │
│             ↓                                               │
│    输出: [B, 4800, 192]                                     │
│                                                             │
│    重复 4 次 (num_encoder_layers=4)                         │
└─────────────────────────────────────────────────────────────┘
```

```python
# models/encoder.py

class WindowTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift=False):
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0
        
        self.attn = FlashWindowAttention(dim, num_heads, window_size)
        self.mlp = Mlp(dim, int(dim * 4))
    
    def forward(self, x, h, w):
        # 1. 窗口划分
        windows, Hp, Wp = self._window_partition(x, h, w)
        
        # 2. 窗口内自注意力 (使用 Flash Attention 加速)
        windows = self.attn(windows)
        
        # 3. 窗口还原
        x = self._window_reverse(windows, Hp, Wp, h, w, B)
        
        # 4. FFN
        x = x + self.mlp(self.norm2(x))
        return x
```

**Flash Attention 优化**：

```python
# models/attention.py

class FlashWindowAttention(nn.Module):
    def forward(self, x, mask=None):
        # 使用 PyTorch 2.0+ 的 scaled_dot_product_attention
        # 自动启用 Flash Attention，显存和速度都有显著提升
        if hasattr(F, 'scaled_dot_product_attention'):
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, 
                enable_math=True, 
                enable_mem_efficient=True
            ):
                x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
```

---

### 3.4 Depth-Aware Fusion：深度感知融合

**这是 V2 版本最重要的改进！**

**问题背景**：传统方案预测深度后就扔掉了，深度信息没有参与 3D 重建。

**解决方案**：Lift-Splat-Shoot 风格的深度感知融合。

```
┌─────────────────────────────────────────────────────────────┐
│                  Lift-Splat-Shoot 原理                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  传统方案 (深度预测完就扔):                                  │
│  ┌───────┐                                                  │
│  │ Image │ → Encoder → Depth → [扔掉]                       │
│  └───────┘              ↓                                   │
│                    只用于监督损失                            │
│                                                             │
│  Lift-Splat (深度参与重建):                                  │
│  ┌───────┐                                                  │
│  │ Image │ → Encoder → Depth Distribution [B,N,64,H,W]      │
│  └───────┘              ↓                                   │
│                    × Features [B,N,C,H,W]                   │
│                         ↓                                   │
│                 = 3D 点云特征 (按深度分布加权)               │
│                         ↓                                   │
│                    投影到 BEV                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
# models/depth_to_3d.py

class LiftSplatModule(nn.Module):
    """
    Lift-Splat-Shoot 核心模块
    
    将 2D 特征通过深度分布"提升"到 3D 空间，然后"溅射"到 BEV 网格
    """
    
    def forward(self, features, camera_intrinsics=None, camera_extrinsics=None):
        """
        Args:
            features: [B, N, C, H, W] 多相机特征
        
        Returns:
            bev_features: [B, C, bev_h, bev_w] BEV 特征
            depth_logits: [B, N, D, H, W] 深度分布 (用于监督)
            depth_pred: [B, N, H, W] 预测深度值
        """
        B, N, C, H, W = features.shape
        
        # 1. 预测每个相机的深度分布 (64 个 bin)
        features_flat = features.view(B * N, C, H, W)
        depth_logits, depth_probs, depth_pred = self.depth_net(features_flat)
        # depth_probs: [B, N, 64, H, W] - 每个像素的深度概率分布
        
        # 2. 特征投影
        proj_features = self.feature_proj(features_flat)  # [B*N, C', H, W]
        
        # 3. Lift: 特征 × 深度概率 = 加权特征
        # 4. Splat: 根据深度将特征投影到 BEV
        bev_features = self._splat_to_bev(proj_features, depth_probs, depth_pred)
        
        # 5. BEV 聚合
        bev_features = self.bev_aggregator(bev_features)
        
        return bev_features, depth_logits, depth_pred
```

**深度分布 vs 单一深度值**：

| 单一深度值 | 深度分布 |
|-----------|---------|
| 每像素一个深度 | 每像素 64 个概率 |
| 边界模糊 | 边界清晰 |
| 不确定区域无法表达 | 可以表达"可能在 10m 也可能在 20m" |
| 硬投影 | 软投影（加权累加） |

---

### 3.5 BEV Decoder：BEV 特征精炼

**目标**：从融合后的 token 生成高质量的 BEV 特征图。

**核心机制**：Deformable Attention（可变形注意力），让 BEV Query 能自适应地采样图像特征。

```
┌─────────────────────────────────────────────────────────────┐
│                    BEV Decoder 工作原理                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  BEV Queries: [128×128, 192]  可学习的查询向量               │
│       ↓                                                     │
│  ┌─────────────────────────────────────────┐                │
│  │  Self-Attention                         │                │
│  │  BEV 位置之间的信息交互                  │                │
│  └─────────────────────────────────────────┘                │
│       ↓                                                     │
│  ┌─────────────────────────────────────────┐                │
│  │  Deformable Cross-Attention             │                │
│  │  每个 BEV 位置采样 4 个图像特征点         │                │
│  │  采样位置是可学习的偏移量                 │                │
│  └─────────────────────────────────────────┘                │
│       ↓                                                     │
│  ┌─────────────────────────────────────────┐                │
│  │  FFN                                    │                │
│  └─────────────────────────────────────────┘                │
│       ↓                                                     │
│  输出: [B, 192, 128, 128] BEV 特征图                        │
│                                                             │
│  重复 3 次 (num_decoder_layers=3)                           │
└─────────────────────────────────────────────────────────────┘
```

```python
# models/decoder.py

class BEVDecoder(nn.Module):
    def __init__(self, dim, num_heads, num_layers, bev_h, bev_w, num_points=4):
        # 可学习的 BEV Queries
        self.bev_queries = BEVQueries(bev_h, bev_w, dim)
        
        # Decoder 层
        self.layers = nn.ModuleList([
            DecoderLayer(dim, num_heads, num_points)
            for _ in range(num_layers)
        ])
    
    def forward(self, memory, spatial_shapes):
        B = memory.shape[0]
        
        # 获取 BEV Queries
        queries, query_pos, ref_points = self.bev_queries(B, device)
        
        # 逐层解码
        for layer in self.layers:
            queries = layer(queries, query_pos, memory, ref_points, spatial_shapes)
        
        # 重塑为 2D 特征图
        return queries.view(B, -1, self.bev_h, self.bev_w)
```

---

### 3.6 Temporal Fusion：时序融合（Memory Cell）

**这是网络的另一个核心创新！**

**问题背景**：传统 TBPTT（截断反向传播）需要保留多帧计算图，显存爆炸。

**解决方案**：Memory Cell（基于 ConvGRU 的时序压缩）

```
┌─────────────────────────────────────────────────────────────┐
│              时序融合方案对比                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  传统 5 帧 Transformer (显存爆炸):                           │
│  ┌─────────────────────────────────────────┐                │
│  │ bev_t-4 ─┐                              │                │
│  │ bev_t-3 ─┼─→ Attention([5帧])           │                │
│  │ bev_t-2 ─┤                              │                │
│  │ bev_t-1 ─┤                              │                │
│  │ bev_t   ─┘                              │                │
│  │                                         │                │
│  │ 存储: 5帧×12.6MB = 63MB                 │                │
│  │ TBPTT计算图: 5帧×全网络 ≈ 12GB          │                │
│  └─────────────────────────────────────────┘                │
│                                                             │
│  Memory Cell (显存友好):                                     │
│  ┌─────────────────────────────────────────┐                │
│  │              ┌──────────┐               │                │
│  │ bev_t ──→ Compress ──→│        │        │                │
│  │          (32×32×64)   │ ConvGRU │──→ 输出│                │
│  │                       │        │        │                │
│  │ memory_t-1 ─────────→│        │        │                │
│  │ (32×32×64)            └──────────┘       │                │
│  │                                         │                │
│  │ 存储: 1×0.26MB = 0.26MB (240x 压缩!)    │                │
│  │ TBPTT计算图: 只有 GRU cell ≈ 10MB       │                │
│  └─────────────────────────────────────────┘                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
# models/temporal.py

class TemporalMemoryCell(nn.Module):
    """
    基于 Memory Cell 的时序融合
    
    原理:
    1. 将当前 BEV 压缩到低维 bottleneck (128×128×192 → 32×32×64)
    2. 用 ConvGRU 更新 memory state
    3. 解压回原始分辨率
    
    显存对比:
    - 原始 5 帧: 5 × 128 × 128 × 192 × 4 = 62.9 MB (TBPTT: ~12GB)
    - Memory Cell: 1 × 32 × 32 × 64 × 4 = 0.26 MB (TBPTT: ~10MB)
    - 压缩比: 240x (显存), 1200x (TBPTT计算图)
    """
    
    def __init__(self, bev_dim=192, bev_size=(128,128), memory_dim=64, memory_size=(32,32)):
        super().__init__()
        
        # Encoder: BEV → Memory Space (128×128×192 → 32×32×64)
        self.encoder = nn.Sequential(
            nn.Conv2d(bev_dim, 128, 3, stride=2, padding=1),  # 64×64
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, memory_dim, 3, stride=2, padding=1),  # 32×32
            nn.BatchNorm2d(memory_dim), nn.GELU(),
        )
        
        # ConvGRU: 时序记忆更新
        self.gru = ConvGRUCell(memory_dim, memory_dim, kernel_size=3)
        
        # Decoder: Memory Space → BEV (32×32×64 → 128×128×192)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(memory_dim, 128, 4, stride=2, padding=1),  # 64×64
            nn.BatchNorm2d(128), nn.GELU(),
            nn.ConvTranspose2d(128, bev_dim, 4, stride=2, padding=1),  # 128×128
            nn.BatchNorm2d(bev_dim),
        )
        
        # Fusion: 合并当前帧和记忆
        self.fusion = nn.Sequential(
            nn.Conv2d(bev_dim * 2, bev_dim, 1),
            nn.BatchNorm2d(bev_dim), nn.GELU(),
            nn.Conv2d(bev_dim, bev_dim, 3, padding=1),
        )
        
        # Memory State
        self.memory = None
        self.memory_pose = None
    
    def forward(self, current_bev, ego_motion=None, current_pose=None, **kwargs):
        B, C, H, W = current_bev.shape
        
        # 1. 压缩当前 BEV 到 memory space
        current_compressed = self.encoder(current_bev)  # [B, 64, 32, 32]
        
        # 2. 第一帧：初始化记忆
        if self.memory is None:
            self.memory = current_compressed.detach()
            self.memory_pose = current_pose.detach() if current_pose is not None else None
            return current_bev
        
        # 3. 运动补偿对齐记忆
        memory_aligned = self.motion_comp(self.memory, rel_pose)
        
        # 4. GRU 更新记忆
        new_memory = self.gru(current_compressed, memory_aligned)
        
        # 5. 解码记忆到 BEV 空间
        memory_decoded = self.decoder(new_memory)
        
        # 6. 融合当前帧和记忆
        fused = self.fusion(torch.cat([current_bev, memory_decoded], dim=1))
        output = current_bev + fused  # 残差连接
        
        # 7. 更新记忆状态 (detach: 每帧独立 backward)
        self.memory = new_memory.detach()
        self.memory_pose = current_pose.detach()
        
        return output
```

**ConvGRU Cell 详解**：

```python
class ConvGRUCell(nn.Module):
    """
    2D 卷积 GRU 单元
    
    比 ConvLSTM 更轻量，效果相当
    """
    
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        
        # Reset gate: 决定遗忘多少旧记忆
        self.reset_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        
        # Update gate: 决定更新多少新信息
        self.update_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        
        # Candidate: 候选新记忆
        self.candidate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
    
    def forward(self, x, h):
        """
        x: [B, C_in, H, W] 当前输入
        h: [B, C_hidden, H, W] 上一时刻隐状态
        
        GRU 公式:
        r = σ(W_r · [x, h])           # reset gate
        z = σ(W_z · [x, h])           # update gate
        h̃ = tanh(W · [x, r*h])        # candidate
        h_new = (1-z) * h + z * h̃     # 新隐状态
        """
        combined = torch.cat([x, h], dim=1)
        
        r = torch.sigmoid(self.reset_gate(combined))   # reset gate
        z = torch.sigmoid(self.update_gate(combined))  # update gate
        
        combined_r = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.candidate(combined_r))  # candidate
        
        h_new = (1 - z) * h + z * h_tilde
        
        return h_new
```

---

### 3.7 Height Expansion & Upsampler：2D→3D 提升

**目标**：将 2D BEV 特征扩展为 3D 体素特征。

```
┌─────────────────────────────────────────────────────────────┐
│                  2D → 3D 提升过程                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  BEV: [B, 192, 128, 128]                                    │
│       ↓ Height Expansion (线性投影)                          │
│  Coarse Voxel: [B, 192, 128, 128, 8]   ← 8 个高度层          │
│       ↓ Upsampler (3D 卷积 + 三线性插值)                     │
│  Fine Voxel: [B, 96, 400, 400, 32]     ← 目标分辨率          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
# models/decoder.py

class CoarseHeightExpansion(nn.Module):
    """高度扩展：将 BEV 特征扩展到 Z 维度"""
    
    def __init__(self, dim, num_heights):
        super().__init__()
        self.num_heights = num_heights
        # 线性投影：每个 BEV 位置生成 num_heights 个特征
        self.expand = nn.Linear(dim, dim * num_heights)
    
    def forward(self, bev):
        B, C, H, W = bev.shape
        x = bev.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x = self.expand(x)  # [B, H*W, C*num_heights]
        return x.view(B, H, W, self.num_heights, C).permute(0, 4, 1, 2, 3)
        # 输出: [B, C, H, W, num_heights]


class LightweightUpsampler(nn.Module):
    """轻量级上采样器：3D 卷积 + 三线性插值"""
    
    def __init__(self, in_channels, out_channels, target_size):
        super().__init__()
        self.target_size = target_size
        
        self.up = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, 3, 1, 1, bias=False),
            nn.BatchNorm3d(in_channels // 2),
            nn.GELU()
        )
        self.out = nn.Conv3d(in_channels // 2, out_channels, 1)
    
    def forward(self, x):
        x = self.up(x)
        # 三线性插值到目标尺寸
        if x.shape[2:] != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode='trilinear', align_corners=False)
        return self.out(x)
```

---

### 3.8 Prediction Head：语义预测

**目标**：预测每个体素的语义类别（18 类）。

**核心设计**：Coarse-to-Fine（由粗到细）两阶段预测。

```
┌─────────────────────────────────────────────────────────────┐
│                 Coarse-to-Fine 预测                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入: [B, 96, 400, 400, 32]                                │
│       ↓                                                     │
│  ┌─────────────────────────────────────────┐                │
│  │  Coarse 预测 (下采样后预测)              │                │
│  │  [B, 96, 400, 400, 32]                  │                │
│  │       ↓ 下采样到 100×100×8              │                │
│  │  [B, 96, 100, 100, 8]                   │                │
│  │       ↓ 卷积预测                        │                │
│  │  [B, 18, 100, 100, 8]  ← coarse_semantic│                │
│  └─────────────────────────────────────────┘                │
│       ↓ 上采样回 400×400×32                                  │
│       ↓ 拼接原始特征                                         │
│  ┌─────────────────────────────────────────┐                │
│  │  Fine 预测 (精细化)                      │                │
│  │  [B, 96+18, 400, 400, 32]              │                │
│  │       ↓ 卷积精炼                        │                │
│  │  [B, 18, 400, 400, 32]  ← 最终语义预测   │                │
│  └─────────────────────────────────────────┘                │
│                                                             │
│  为什么要 Coarse-to-Fine？                                   │
│  1. Coarse 阶段提供全局语义先验                              │
│  2. Fine 阶段在先验基础上精细化                              │
│  3. 多尺度监督，训练更稳定                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
# models/heads.py

class CoarseToFineHead(nn.Module):
    def __init__(self, in_channels, num_classes, coarse_size, fine_size, use_flow=True):
        super().__init__()
        self.coarse_size = coarse_size  # (100, 100, 8)
        self.fine_size = fine_size      # (400, 400, 32)
        
        # Coarse 预测
        self.coarse_semantic = nn.Sequential(
            ConvBlock3D(in_channels, in_channels // 2),
            nn.Conv3d(in_channels // 2, num_classes, 1)
        )
        
        # Fine 精炼
        self.refine_conv = ConvBlock3D(num_classes + in_channels, in_channels // 2)
        self.refine_out = nn.Conv3d(in_channels // 2, num_classes, 1)
        
        # 可选: 运动流预测
        if use_flow:
            self.coarse_flow = nn.Sequential(...)
            self.refine_flow = nn.Sequential(...)
    
    def forward(self, x):
        # 1. Coarse 预测
        x_coarse = F.interpolate(x, size=self.coarse_size, mode='trilinear')
        coarse_sem = self.coarse_semantic(x_coarse)  # [B, 18, 100, 100, 8]
        
        # 2. 上采样 coarse 结果
        coarse_sem_up = F.interpolate(coarse_sem, size=self.fine_size, mode='trilinear')
        
        # 3. Fine 精炼
        x_fine = F.interpolate(x, size=self.fine_size, mode='trilinear')
        concat = torch.cat([coarse_sem_up, x_fine], dim=1)
        fine_sem = self.refine_out(self.refine_conv(concat))
        
        return {
            'semantic': fine_sem,          # [B, 18, 400, 400, 32]
            'coarse_semantic': coarse_sem  # [B, 18, 100, 100, 8]
        }
```

---

## 四、损失函数设计

OccNetV3 使用**多任务联合损失**：

```
Total Loss = Focal Loss + Dice Loss + Distance Loss + Depth Loss + Flow Loss + Coarse Loss
```

```
┌─────────────────────────────────────────────────────────────┐
│                    损失函数组成                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐                                            │
│  │ Focal Loss  │ ← 处理类别不平衡 (empty 占 90%+)           │
│  │ α=0.25,γ=2  │   降低简单样本权重，聚焦困难样本            │
│  └─────────────┘                                            │
│        +                                                    │
│  ┌─────────────┐                                            │
│  │ Dice Loss   │ ← 关注形状一致性                           │
│  │             │   对前景/背景比例不敏感                     │
│  └─────────────┘                                            │
│        +                                                    │
│  ┌─────────────┐                                            │
│  │ Distance    │ ← 近距离物体更重要 (安全相关)              │
│  │ Aware Loss  │   权重 = exp(-d/20) + 0.5                  │
│  └─────────────┘                                            │
│        +                                                    │
│  ┌─────────────┐                                            │
│  │ Depth Loss  │ ← 深度监督 (边缘感知)                      │
│  │ Log L1      │   边缘处不强制平滑                         │
│  └─────────────┘                                            │
│        +                                                    │
│  ┌─────────────┐                                            │
│  │ Flow Loss   │ ← 运动流监督 (可选)                        │
│  │ L1          │   预测动态物体的运动方向                    │
│  └─────────────┘                                            │
│        +                                                    │
│  ┌─────────────┐                                            │
│  │ Coarse Loss │ ← 多尺度监督                               │
│  │ ×0.3        │   Coarse 阶段的 Focal + Dice               │
│  └─────────────┘                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**距离感知损失权重可视化**：

```
         近 ←────────────────────────→ 远
         │                            │
权重 3.0 │▓▓▓▓▓                       │
         │▓▓▓▓▓▓▓                     │
     2.0 │▓▓▓▓▓▓▓▓▓▓                  │
         │▓▓▓▓▓▓▓▓▓▓▓▓▓               │
     1.0 │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓         │
         │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓    │
     0.5 │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
         └────────────────────────────┘
         0m         20m        40m    80m
```

---

## 4.1 深度监督详解

深度监督是 OccNetV3 的关键辅助任务，帮助网络学习 2D→3D 的几何映射。

### 为什么需要深度监督？

```
┌─────────────────────────────────────────────────────────────┐
│                深度监督的作用                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  没有深度监督:                                               │
│  ┌─────────────────────────────────────────┐                │
│  │ 网络需要"凭空"学习 2D→3D 投影            │                │
│  │ - 收敛慢                                │                │
│  │ - 深度估计不准                          │                │
│  │ - 远处物体位置偏差大                     │                │
│  └─────────────────────────────────────────┘                │
│                                                             │
│  有深度监督:                                                 │
│  ┌─────────────────────────────────────────┐                │
│  │ 显式告诉网络"每个像素对应多远"            │                │
│  │ - 收敛快 (2-3x)                         │                │
│  │ - 深度估计准确                          │                │
│  │ - 3D 重建质量高                         │                │
│  └─────────────────────────────────────────┘                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 深度损失设计：三个关键改进

#### 改进 1：Log 空间 L1 损失

```python
# losses/losses.py

# 为什么用 Log 空间？
# - 近距离物体：1m vs 2m 的差异很重要 (100% 误差)
# - 远距离物体：50m vs 51m 的差异不重要 (2% 误差)
# - Log 空间自动实现"近处敏感，远处宽容"

log_pred = torch.log(depth_pred + eps)  # eps=1e-6 防止 log(0)
log_gt = torch.log(depth_gt + eps)
base_loss = torch.abs(log_pred - log_gt)  # L1 损失

# 对比:
# 线性空间: |1 - 2| = 1,   |50 - 51| = 1   (同等惩罚)
# Log 空间: |0 - 0.69| = 0.69, |3.9 - 3.93| = 0.03 (近处惩罚更大)
```

#### 改进 2：边缘感知平滑损失

```
┌─────────────────────────────────────────────────────────────┐
│                边缘感知深度损失                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  问题: 传统平滑损失在物体边缘会出错                           │
│                                                             │
│  图像:          深度真值:        传统平滑损失:                │
│  ┌────┬────┐    ┌────┬────┐     ┌────┬────┐                │
│  │车  │背景│    │10m │50m │     │惩罚│惩罚│ ← 边缘被错误惩罚 │
│  │    │    │    │    │    │     │    │    │                │
│  └────┴────┘    └────┴────┘     └────┴────┘                │
│                                                             │
│  边缘感知平滑损失:                                           │
│  1. 计算图像梯度 (边缘检测)                                  │
│  2. 边缘处降低平滑权重                                       │
│  3. 只在平坦区域强制深度连续                                  │
│                                                             │
│  ┌────┬────┐                                                │
│  │不罚│不罚│ ← 边缘处允许深度不连续                          │
│  │平滑│平滑│ ← 平坦区域强制平滑                              │
│  └────┴────┘                                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
# losses/losses.py - EdgeAwareDepthLoss

def _compute_edge_aware_smooth_loss(self, depth_pred, images):
    # 1. 计算图像梯度 (边缘)
    img_gray = images.mean(dim=2)  # 转灰度
    grad_x = torch.abs(img_gray[:, :, :, 1:] - img_gray[:, :, :, :-1])
    grad_y = torch.abs(img_gray[:, :, 1:, :] - img_gray[:, :, :-1, :])
    
    # 2. 边缘权重: 边缘处权重小 (不惩罚深度不连续)
    edge_weight_x = torch.exp(-grad_x * 10)  # 梯度大 → 权重小
    edge_weight_y = torch.exp(-grad_y * 10)
    
    # 3. 深度梯度
    depth_grad_x = torch.abs(depth_pred[:, :, :, 1:] - depth_pred[:, :, :, :-1])
    depth_grad_y = torch.abs(depth_pred[:, :, 1:, :] - depth_pred[:, :, :-1, :])
    
    # 4. 加权平滑损失
    smooth_loss = (edge_weight_x * depth_grad_x).mean() + \
                  (edge_weight_y * depth_grad_y).mean()
    
    return smooth_loss
```

#### 改进 3：深度 GT 下采样策略

```
┌─────────────────────────────────────────────────────────────┐
│             深度 GT 下采样问题                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  原始深度图: 960×1280                                        │
│  特征图尺寸: 60×80 (下采样 16x)                              │
│                                                             │
│  问题: 如何下采样深度 GT？                                   │
│                                                             │
│  方案 1: 双线性插值 (Bilinear) ❌                            │
│  ┌─────────────────────────────────────────┐                │
│  │ 物体边缘: 10m 和 50m 平均 = 30m         │                │
│  │ → 边缘深度被严重模糊                    │                │
│  │ → 网络学习错误的边缘深度                │                │
│  └─────────────────────────────────────────┘                │
│                                                             │
│  方案 2: 最近邻插值 (Nearest) ✅                             │
│  ┌─────────────────────────────────────────┐                │
│  │ 物体边缘: 取最近的值 (10m 或 50m)        │                │
│  │ → 边缘深度保持清晰                      │                │
│  │ → 虽然有阶梯感，但比模糊好              │                │
│  └─────────────────────────────────────────┘                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
# losses/losses.py - CoarseToFineLoss.forward()

# 深度 GT 下采样
if H_gt != H_pred or W_gt != W_pred:
    depth_gt_reshaped = depth_gt.view(B * N, 1, H_gt, W_gt)
    
    # ✅ 使用最近邻下采样 (保留边缘)
    depth_gt_down = F.interpolate(
        depth_gt_reshaped, 
        size=(H_pred, W_pred), 
        mode='nearest'  # 关键: 不是 'bilinear'
    )
    
    depth_gt = depth_gt_down.view(B, N, H_pred, W_pred)
```

### 深度监督数据流

```
┌─────────────────────────────────────────────────────────────┐
│                深度监督完整流程                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  训练时:                                                     │
│  ┌───────────┐                                              │
│  │ 8 相机图像 │                                              │
│  └─────┬─────┘                                              │
│        ↓                                                    │
│  ┌───────────────────────────────────────┐                  │
│  │ Encoder + Depth Fusion                │                  │
│  │                                       │                  │
│  │ 输出:                                 │                  │
│  │ - depth_logits [B,8,64,60,80] 深度分布│ ← 用于损失计算   │
│  │ - depth_pred [B,8,60,80] 期望深度     │ ← 用于可视化     │
│  │ - bev_features [B,192,128,128]        │ ← 主任务         │
│  └───────────────────────────────────────┘                  │
│        ↓                                                    │
│  ┌───────────────────────────────────────┐                  │
│  │ 深度损失计算                          │                  │
│  │                                       │                  │
│  │ depth_gt ──→ nearest 下采样 ──→ [B,8,60,80]              │
│  │                       ↓                                  │
│  │ depth_loss = LogL1(pred, gt) + 0.1 * EdgeSmooth          │
│  └───────────────────────────────────────┘                  │
│                                                             │
│  推理时:                                                     │
│  - 不需要深度 GT                                            │
│  - 不需要深度传感器                                          │
│  - 网络已学会从图像估计深度                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 深度监督配置

```python
# configs/default.py

# 深度监督开关
use_depth_supervision = True   # 启用深度监督
use_depth_aware_fusion = True  # 深度参与 3D 重建

# 深度参数
depth_range = (0.5, 80.0)      # 有效深度范围 (米)
num_depth_bins = 64            # 深度分布 bin 数量
depth_loss_weight = 0.5        # 深度损失权重

# 深度 bin 分布 (对数均匀)
# 近处密集: 0.5, 0.6, 0.7, 0.8, ...
# 远处稀疏: 40, 50, 63, 80
depth_bins = torch.exp(torch.linspace(
    math.log(0.5),   # min_depth
    math.log(80.0),  # max_depth
    64               # num_bins
))
```

---

## 五、训练策略

### 5.1 三种训练模式

```bash
# 1. Memory Cell (推荐，显存友好)
python train_tbptt.py --dataset /path/to/data --mode memory_cell --amp

# 2. Gradient Accumulation (等效大 batch)
python train_tbptt.py --dataset /path/to/data --mode grad_accum --window 3 --amp

# 3. Classic TBPTT (高显存，效果最好)
python train_tbptt.py --dataset /path/to/data --mode classic --window 3 --amp
```

### 5.2 训练流程

```
┌─────────────────────────────────────────────────────────────┐
│                     训练流程                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Warmup (5 epochs)                                       │
│     - 学习率从 0 线性增长到 1e-4                             │
│     - 模型逐步适应数据                                       │
│                                                             │
│  2. Main Training (95 epochs)                               │
│     - Cosine Annealing 学习率调度                           │
│     - 梯度裁剪 (max_norm=1.0)                               │
│     - AMP 混合精度训练                                       │
│                                                             │
│  3. 场景边界处理                                             │
│     - 检测 scene_id 变化                                    │
│     - 自动 reset 时序状态                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 显存估算

| 模式 | Batch Size | 显存占用 |
|------|------------|----------|
| 普通训练 | 1 | ~6-8 GB |
| Memory Cell | 1 | ~6-8 GB (无额外增长) |
| Grad Accum (window=3) | 1 | ~6-8 GB |
| Classic TBPTT (window=3) | 1 | ~15-18 GB |

---

## 六、配置参数速查

```python
# configs/default.py 关键参数

# === 输入 ===
image_size = (960, 1280)      # 图像尺寸
patch_size = 16               # Patch 大小
num_cameras = 8               # 相机数量
in_channels = 1               # 输入通道 (灰度 RAW)

# === 输出 ===
voxel_size = (400, 400, 32)   # 体素网格尺寸
num_classes = 18              # 语义类别数
pc_range = [-40, -40, -1, 40, 40, 5.4]  # 物理范围 (米)

# === 网络结构 ===
embed_dim = 192               # 特征维度
num_heads = 6                 # 注意力头数
num_encoder_layers = 4        # Encoder 层数
num_decoder_layers = 3        # Decoder 层数
bev_size = (128, 128)         # BEV 特征图尺寸

# === 时序融合 ===
use_memory_cell = True        # 使用 Memory Cell (推荐)
memory_dim = 64               # Memory 通道数
memory_size = (32, 32)        # Memory 空间尺寸
num_frames = 5                # 历史帧数

# === 深度 ===
use_depth_aware_fusion = True # 深度感知融合
num_depth_bins = 64           # 深度 bin 数量
depth_range = (0.5, 80.0)     # 深度范围 (米)

# === 训练 ===
batch_size = 1
lr = 1e-4
max_epochs = 100
use_amp = True                # 混合精度
grad_clip = 1.0               # 梯度裁剪
```

---

## 七、总结

OccNetV3 的核心创新点：

| 创新点 | 解决的问题 | 技术方案 |
|--------|-----------|----------|
| **统一射线编码** | 多种相机投影模型 | 等距投影统一建模 |
| **深度感知融合** | 深度信息浪费 | Lift-Splat-Shoot |
| **Memory Cell** | TBPTT 显存爆炸 | ConvGRU 时序压缩 |
| **Coarse-to-Fine** | 多尺度物体检测 | 两阶段预测 |
| **距离感知损失** | 安全优先级 | 近距离高权重 |

```
最终效果:
- 输入: 8 张 960×1280 灰度图
- 输出: 400×400×32 语义体素网格
- 显存: ~6-8 GB (Memory Cell 模式)
- 速度: 可满足实时性要求 (配合优化)
```

---

> 作者注：本文档基于 OccNetV3 代码库编写，如有疑问欢迎交流。代码仓库包含完整实现，可直接运行训练和推理。