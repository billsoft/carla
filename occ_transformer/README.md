# Transformer Occupancy Network

统一 Transformer 架构的 3D 占用网络，将多视角 2D 图像"翻译"为 3D 体素占用网格。

## 核心思想

```
多视角 2D 像素序列 → Transformer → 3D 体素序列
     (源语言)                      (目标语言)
```

**关键创新**：
- **相机参数 = 位置编码**：射线方向 + 相机位置作为几何先验
- **Cross-Attention = 像素-体素对应**：自动学习投影关系，无需显式几何投影

## 目录结构

```
occ_network_nano/
├── models/
│   ├── __init__.py                      # 模型模块入口
│   └── transformer_occ/                 # Transformer OccNet 核心模块
│       ├── __init__.py                  # 模块导出
│       ├── patch_embed.py               # Bayer Patch Embedding
│       ├── position_encoding.py         # 位置编码 (Camera PE, 3D PE)
│       ├── attention.py                 # 注意力机制 (标准/窗口/可变形)
│       ├── encoder.py                   # Transformer Encoder
│       ├── voxel_query.py               # 3D/BEV 体素查询
│       ├── decoder.py                   # Transformer Decoder
│       └── transformer_occ_net.py       # 主网络 (Standard/Lite)
├── data/
│   ├── __init__.py
│   └── carla_dataset_bayer.py           # CARLA Bayer 数据集
├── utils/
│   ├── __init__.py
│   └── loss.py                          # 损失函数
├── train_transformer.py                 # 训练脚本
├── inference_transformer.py             # 推理脚本
├── verify_transformer_network.py        # 网络验证脚本
└── README.md                            # 本文件
```

## 模块说明

### 1. Patch Embedding (`patch_embed.py`)
- `BayerPatchEmbed`: 单相机 Bayer → Patches
- `MultiCameraPatchEmbed`: 多相机处理，共享权重

### 2. Position Encoding (`position_encoding.py`)
- `CameraPositionEncoding`: 相机位置编码（核心创新）
  - 像素坐标 (u, v)
  - 射线方向 (dx, dy, dz)
  - 相机位置 (cx, cy, cz)
- `Voxel3DPositionEncoding`: 3D 体素位置编码
- `Spatial2DPositionEncoding`: 2D 空间位置编码

### 3. Attention (`attention.py`)
- `MultiHeadAttention`: 标准多头注意力
- `WindowAttention`: 窗口注意力 (Swin 风格)
- `DeformableAttention`: 可变形注意力
- `EfficientAttention`: 高效注意力

### 4. Encoder (`encoder.py`)
- `TransformerEncoder`: 标准 Transformer Encoder
- `HierarchicalEncoder`: 分层 Encoder (Swin 风格)
- `MultiCameraEncoder`: 多相机 Encoder

### 5. Voxel Query (`voxel_query.py`)
- `VoxelQueries`: 完整 3D 体素查询
- `HierarchicalVoxelQueries`: 分层查询 + 上采样
- `BEVQueries`: BEV 查询 + 高度扩展

### 6. Decoder (`decoder.py`)
- `TransformerDecoder`: 标准 Transformer Decoder
- `VoxelDecoder`: 完整体素解码器
- `SimplifiedDecoder`: 简化版解码器

### 7. Main Network (`transformer_occ_net.py`)
- `TransformerOccNet`: 标准版 (~30M 参数)
- `TransformerOccNetLite`: 轻量版 (~15M 参数)

## 输入输出规格

**输入**:
```
images: [B, 8, 1, 960, 1280]  # 8相机 12-bit Bayer RAW
```

**输出**:
```
occ_logits: [B, 18, 200, 200, 16]  # 3D 占用网格 logits
```

## 使用方法

### 构建模型

```python
from models.transformer_occ import build_transformer_occ_net

# 标准版
model = build_transformer_occ_net(
    model_type='standard',
    num_classes=18,
    img_size=(960, 1280),
    output_grid_size=(200, 200, 16)
)

# 轻量版
model_lite = build_transformer_occ_net(
    model_type='lite',
    num_classes=18,
    img_size=(960, 1280),
    output_grid_size=(200, 200, 16)
)
```

### 训练

```bash
python train_transformer.py \
    --dataset /path/to/carla_dataset \
    --model-type lite \
    --batch-size 1 \
    --epochs 100 \
    --lr 1e-4
```

### 推理

```bash
python inference_transformer.py \
    --checkpoint checkpoints_transformer/best.pth \
    --dataset /path/to/carla_dataset \
    --model-type lite
```

### 验证网络

```bash
python verify_transformer_network.py
```

## 版本对比

| 配置 | Standard | Lite |
|------|----------|------|
| Patch Size | 8 | 16 |
| Encoder Layers | 6 | 4 |
| Decoder Layers | 6 | 2 |
| Query Type | 3D Voxel | BEV→3D |
| 参数量 | ~30M | ~15M |
| 显存 (BS=1) | ~8GB | ~4GB |

## 架构流程

```
输入: [B, 8, 1, 960, 1280] 8相机 Bayer
  ↓
┌─────────────────────────────────────┐
│ 1. Patch Embedding                  │
│    PixelUnshuffle → Conv            │
│    → [B, N_patches, D]              │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 2. Position Encoding                │
│    Spatial PE + Camera PE           │
│    (射线方向 + 相机位置)            │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 3. Transformer Encoder              │
│    窗口注意力 × L 层                │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 4. Voxel Queries                    │
│    可学习查询 + 3D 位置编码         │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 5. Transformer Decoder              │
│    Self-Attn + Cross-Attn × L 层    │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 6. 3D Upsample + Head               │
│    上采样 → 分类                    │
└─────────────────────────────────────┘
  ↓
输出: [B, 18, 200, 200, 16] 占用网格
```

## 依赖

- Python >= 3.8
- PyTorch >= 1.10
- NumPy
- tqdm

## 许可证

MIT License
