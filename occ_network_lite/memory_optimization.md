# Occupancy Network 显存优化方案

## 一、当前网络显存消耗分析

### 1.1 显存消耗分布（Batch Size=1, 输入 384×640）

| 模块 | 参数量 | 激活内存 | 瓶颈原因 |
|------|--------|---------|---------|
| **Backbone (ResNet50)** | 25M | ~1.5 GB | 8相机分别提取，中间特征大 |
| **FPN Neck** | 5M | ~0.3 GB | 多尺度特征融合 |
| **View Transformer** | 50M | **~4-6 GB** | ⚠️ **最大瓶颈**：Attention 矩阵巨大 |
| **BEV Encoder** | 5M | ~0.5 GB | 2D 卷积，相对较小 |
| **Occ Decoder (3D Conv)** | 10M | ~1.5 GB | 3D 卷积中间激活 |
| **总计** | ~95M | **~8-10 GB** | 超出大多数消费级 GPU |

### 1.2 View Transformer 为什么是瓶颈？

```
当前配置：
- Query 数量: 200 × 200 = 40,000
- Key/Value 数量: 8 × 48 × 80 = 30,720
- Attention 矩阵: 40,000 × 30,720 = 1.23 Billion 元素！

显存计算 (FP16):
- Attention 矩阵: 1.23B × 2 bytes = 2.46 GB (仅单层单头)
- 多头 (8头): 2.46 GB × 8 = 19.7 GB (理论值，实际有优化)
- 多层 (6层): 更加恐怖
```

**结论**：全量 Cross Attention 在高分辨率下根本无法训练。

---

## 二、优化方案总览

### 方案对比表

| 方案 | 显存节省 | 精度影响 | 实现难度 | 推荐优先级 |
|------|---------|---------|---------|-----------|
| **A. 轻量级 Backbone** | 30-50% | 轻微下降 | ⭐ 简单 | ⭐⭐⭐⭐⭐ |
| **B. 降低 BEV 分辨率** | 60-75% | 中等下降 | ⭐ 简单 | ⭐⭐⭐⭐⭐ |
| **C. 减少特征维度** | 50% | 轻微下降 | ⭐ 简单 | ⭐⭐⭐⭐ |
| **D. Deformable Attention** | 70-80% | 几乎无 | ⭐⭐⭐ 中等 | ⭐⭐⭐⭐ |
| **E. 减少 Transformer 层数** | 40% | 中等下降 | ⭐ 简单 | ⭐⭐⭐ |
| **F. 梯度检查点** | 30-50% | 无 | ⭐⭐ 简单 | ⭐⭐⭐ |
| **G. LSS 替代方案** | 80%+ | 可能更好 | ⭐⭐⭐ 中等 | ⭐⭐⭐⭐⭐ |

---

## 三、具体优化方案

### 方案 A：轻量级 Backbone

#### A1. MobileNetV2 (推荐)

```python
# 参数量: 3.4M (vs ResNet50 25M) - 减少 86%
# 输出通道: [24, 32, 96, 320]

class MobileNetV2Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        mobilenet = mobilenet_v2(weights=weights)
        
        # 提取特征层
        self.features = mobilenet.features
        
        # 输出通道: 适配不同 stage
        # Stage 0-1: 32 channels
        # Stage 2-3: 24 channels  
        # Stage 4-6: 32 channels
        # Stage 7-13: 96 channels
        # Stage 14-17: 320 channels
        self.out_channels = [32, 96, 320]  # 对应 1/8, 1/16, 1/32
        
    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in [3, 6, 17]:  # 选择输出层
                outputs.append(x)
        return outputs
```

#### A2. MobileNetV3-Small (更轻量)

```python
# 参数量: 2.5M - 最轻量
# 适合边缘部署

class MobileNetV3SmallBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        mobilenet = mobilenet_v3_small(weights=weights)
        
        self.features = mobilenet.features
        self.out_channels = [24, 48, 96]
```

#### A3. EfficientNet-B0 (平衡选择)

```python
# 参数量: 5.3M
# 精度和效率的平衡

class EfficientNetB0Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        efficientnet = efficientnet_b0(weights=weights)
        
        self.features = efficientnet.features
        self.out_channels = [24, 40, 112]  # B0 的通道数
```

### 方案 B：降低 BEV/体素分辨率

#### 分辨率配置对比

| 配置 | BEV 尺寸 | Query 数 | Attention 矩阵 | 显存估计 |
|------|---------|---------|---------------|---------|
| 原始 | 200×200 | 40,000 | 40K×30K=1.2B | ~10 GB |
| **推荐** | 100×100 | 10,000 | 10K×30K=300M | ~3 GB |
| 轻量 | 50×50 | 2,500 | 2.5K×30K=75M | ~1 GB |

```python
# 配置修改
class LiteConfig:
    bev_h = 100  # 原 200
    bev_w = 100  # 原 200
    num_heights = 8  # 原 16
    
    # 对应物理分辨率
    # 100m / 100 = 1m per voxel (仍然足够)
```

### 方案 C：减少特征维度

```python
# 原配置: embed_dim = 256
# 优化配置: embed_dim = 128 或 64

class LiteDimConfig:
    embed_dim = 128      # 原 256，减少 50%
    num_heads = 4        # 原 8
    ffn_dim = 512        # 原 1024
    hidden_channels = 64 # 原 128
```

### 方案 D：Deformable Attention (重要优化)

**核心思想**：每个 Query 不关注所有 30K 个像素，只关注 K 个采样点（如 K=4）。

```python
class DeformableCrossAttention(nn.Module):
    """
    可变形交叉注意力
    
    复杂度: O(N_q × K) 而非 O(N_q × N_kv)
    当 K=4, N_kv=30720 时，减少 7680 倍！
    """
    def __init__(
        self,
        embed_dim=128,
        num_heads=4,
        num_points=4,  # 每个 Query 采样的点数
        num_levels=1,  # 多尺度层数
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_points = num_points
        
        # 采样偏移预测
        self.sampling_offsets = nn.Linear(
            embed_dim, 
            num_heads * num_levels * num_points * 2
        )
        
        # 注意力权重
        self.attention_weights = nn.Linear(
            embed_dim,
            num_heads * num_levels * num_points
        )
        
        # 值投影
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, reference_points, value, spatial_shapes):
        """
        Args:
            query: [B, N_q, C] BEV Query
            reference_points: [B, N_q, 2] 归一化参考点坐标
            value: [B, N_kv, C] 图像特征
            spatial_shapes: 特征图尺寸
        """
        B, N_q, C = query.shape
        
        # 预测采样偏移
        offsets = self.sampling_offsets(query)
        # [B, N_q, num_heads * num_points * 2]
        
        # 预测注意力权重
        attn_weights = self.attention_weights(query)
        attn_weights = F.softmax(attn_weights, dim=-1)
        # [B, N_q, num_heads * num_points]
        
        # 计算采样点位置
        sampling_locations = reference_points[:, :, None, :] + offsets.view(
            B, N_q, self.num_heads * self.num_points, 2
        )
        
        # 双线性插值采样
        # ... (需要 grid_sample 实现)
        
        return output
```

### 方案 E：LSS (Lift-Splat-Shoot) 替代方案

**LSS 比 Cross Attention 更省显存**，因为它不需要计算大的注意力矩阵。

```python
class LSSViewTransformer(nn.Module):
    """
    Lift-Splat-Shoot: 显式深度估计 + 体素投射
    
    优势:
    1. 无需 Attention 矩阵
    2. 几何可解释
    3. 显存友好
    """
    def __init__(
        self,
        in_channels=128,
        out_channels=128,
        num_depth_bins=64,  # 深度离散化数量
        depth_range=(1.0, 60.0),
        bev_h=100,
        bev_w=100,
    ):
        super().__init__()
        
        # 深度估计网络
        self.depth_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_depth_bins, 1),
        )
        
        # 深度 bins
        self.depth_bins = nn.Parameter(
            torch.linspace(depth_range[0], depth_range[1], num_depth_bins),
            requires_grad=False
        )
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_depth_bins = num_depth_bins
        
    def forward(self, img_features, intrinsics, extrinsics):
        """
        Args:
            img_features: [B, N, C, H, W] 多相机特征
            intrinsics: [B, N, 3, 3] 内参
            extrinsics: [B, N, 4, 4] 外参
        """
        B, N, C, H, W = img_features.shape
        
        # 1. 预测深度分布
        depth_logits = self.depth_net(img_features.flatten(0, 1))
        depth_probs = F.softmax(depth_logits, dim=1)  # [B*N, D, H, W]
        
        # 2. Lift: 将 2D 特征提升到 3D 视锥体
        # 每个像素 × 每个深度 bin = 一个 3D 点
        frustum_features = img_features.flatten(0, 1)[:, :, None, :, :] * \
                          depth_probs[:, None, :, :, :]
        # [B*N, C, D, H, W]
        
        # 3. Splat: 将 3D 点投射到 BEV 网格
        bev_features = self.voxel_pooling(
            frustum_features, intrinsics, extrinsics
        )
        # [B, C, bev_h, bev_w]
        
        return bev_features
```

### 方案 F：梯度检查点

```python
from torch.utils.checkpoint import checkpoint

class OccupancyNetworkWithCheckpoint(OccupancyNetwork):
    def forward(self, images, extrinsics=None, upsample=False):
        # 使用 checkpoint 减少显存
        img_features = checkpoint(
            self.extract_img_features, 
            images,
            use_reentrant=False
        )
        
        # View Transformer 也使用 checkpoint
        bev_features = checkpoint(
            self.view_transformer,
            img_features,
            self.pos_encoder.get_image_pos_encoding(...),
            use_reentrant=False
        )
        
        # ... 其余部分
```

---

## 四、推荐的轻量级网络配置

### 配置方案：OccNet-Lite

```python
# 综合应用多种优化

class OccNetLiteConfig:
    # 图像
    img_size = (256, 448)         # 原 (384, 640)，减少 40%
    
    # Backbone
    backbone = 'mobilenetv2'      # 原 resnet50
    
    # 特征维度
    embed_dim = 128               # 原 256
    num_heads = 4                 # 原 8
    
    # BEV 分辨率
    bev_h = 100                   # 原 200
    bev_w = 100                   # 原 200
    
    # Transformer
    num_transformer_layers = 2    # 原 6
    ffn_dim = 256                 # 原 1024
    
    # 体素
    num_heights = 8               # 原 16
    
    # 预计显存: ~2-3 GB (Batch Size=1)
```

### 预计效果对比

| 配置 | 参数量 | 显存 (BS=1) | mIoU 预估 |
|------|--------|------------|-----------|
| 原版 | ~95M | ~10 GB | 基准 |
| **Lite** | ~15M | ~2-3 GB | -5~10% |
| Ultra-Lite | ~8M | ~1.5 GB | -10~15% |

---

## 五、实现计划

### 第一阶段：快速降低显存（立即可用）

1. **降低 BEV 分辨率**: 200→100
2. **降低特征维度**: 256→128
3. **减少 Transformer 层数**: 6→2
4. **降低输入分辨率**: 384×640→256×448

### 第二阶段：更换 Backbone（中期）

1. 实现 MobileNetV2 Backbone
2. 调整 FPN 通道数匹配

### 第三阶段：架构优化（长期）

1. 实现 LSS View Transformer
2. 或实现 Deformable Attention
3. 多尺度特征融合优化

---

## 六、代码实现

见 `models/occ_network_lite.py`（下一个文件）
