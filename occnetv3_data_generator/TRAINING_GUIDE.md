# OccNetV3 训练指南

本文档详细说明如何使用 `occnetv3_data_generator` 采集的数据训练 `occ_network` 3D 占用网络。

---

## 目录

1. [项目架构总览](#1-项目架构总览)
2. [数据格式说明](#2-数据格式说明)
3. [网络架构解析](#3-网络架构解析)
4. [环境配置](#4-环境配置)
5. [训练流程](#5-训练流程)
6. [推理与评估](#6-推理与评估)
7. [常见问题](#7-常见问题)

---

## 1. 项目架构总览

### 1.1 数据生成器 (`occnetv3_data_generator/`)

```
occnetv3_data_generator/
├── config/
│   ├── camera_config.py          # Tesla 8 相机布局配置
│   ├── occupancy_config.py       # 体素空间定义 (400×400×32, 0.2m)
│   └── actor_occupancy_mapping.py # CARLA Actor → 语义类别映射
├── sensors/
│   ├── camera_manager.py         # 8 相机同步采集管理
│   └── semantic_lidar_sensor.py  # 256 线语义激光雷达
├── processing/
│   ├── ground_truth_voxel_generator.py  # 体素真值生成
│   └── visibility_filter_simple.py      # 可见性过滤
├── data_utils/
│   └── data_saver.py             # DNG/NPY 数据保存
├── main_collection.py            # 主采集脚本 ⭐
├── train_demo.py                 # 简易训练示例
└── visualize_dataset.py          # 数据集可视化
```

### 1.2 网络代码 (`occ_network/`)

```
occ_network/
├── configs/
│   └── default.py                # 网络配置 (体素尺寸、学习率等)
├── models/
│   ├── patch_embed.py            # Patch 嵌入 (图像→Token)
│   ├── position_encoding.py      # 相机位置编码 (RoPE + FOV)
│   ├── attention.py              # Flash Attention / Deformable Attention
│   ├── encoder.py                # 多相机 Swin Transformer 编码器
│   ├── decoder.py                # BEV 解码器 + 高度扩展
│   ├── temporal.py               # 时序融合模块
│   ├── heads.py                  # Coarse-to-Fine 输出头
│   ├── sparse_modules.py         # 稀疏卷积支持 (spconv/torchsparse)
│   └── occ_net.py                # OccNetV3 完整网络 ⭐
├── losses/
│   └── losses.py                 # Focal Loss + Dice Loss + Flow Loss
├── data/
│   └── dataset.py                # 数据加载器 (支持 DNG/NPY)
├── utils/
│   └── camera.py                 # 相机投影工具
├── train.py                      # 训练脚本 ⭐
├── inference.py                  # 推理/基准测试脚本 ⭐
└── verify_network_shapes.py      # 网络结构验证
```

---

## 2. 数据格式说明

### 2.1 数据采集输出结构

运行 `main_collection.py` 后生成的数据目录结构：

```
dataset_10k/
├── cameras/                      # 8 相机 Bayer RAW 图像
│   ├── front_main/
│   │   ├── 000000.dng           # 12-bit Bayer RGGB (1280×960)
│   │   ├── 000001.dng
│   │   └── ...
│   ├── front_wide/
│   ├── front_narrow/
│   ├── left_pillar/
│   ├── right_pillar/
│   ├── left_repeater/
│   ├── right_repeater/
│   └── rear/
├── occupancy/                    # 3D 体素标签
│   ├── 000000.npy               # shape: (400, 400, 32), dtype: uint8
│   └── ...
├── flow/                         # 3D 运动流场 (可选)
│   ├── 000000.npy               # shape: (3, 400, 400, 32), dtype: float32
│   └── ...
├── ego_pose/                     # 车辆全局位姿
│   ├── 000000.npy               # shape: (4, 4), dtype: float32
│   └── ...
├── ego_motion/                   # 帧间相对运动
│   ├── 000000.npy               # shape: (4, 4), dtype: float32
│   └── ...
├── calibration/
│   ├── intrinsics.json          # 相机内参 {cam_id: 3×3 矩阵}
│   ├── extrinsics.json          # 相机外参 {cam_id: 4×4 矩阵}
│   └── camera_info.json         # 相机元信息 (FOV, 位置, 旋转)
├── train.txt                     # 训练集样本 ID 列表
└── val.txt                       # 验证集样本 ID 列表
```

### 2.2 体素空间配置

| 参数 | 值 | 说明 |
|------|-----|------|
| X 范围 | [-40, 40] m | 前后 80m |
| Y 范围 | [-40, 40] m | 左右 80m |
| Z 范围 | [-1.0, 5.4] m | 上下 6.4m |
| 分辨率 | 0.2 m | 每体素边长 |
| 网格尺寸 | (400, 400, 32) | 5,120,000 体素 |

### 2.3 语义类别 (18 类)

```python
SEMANTIC_CLASSES = {
    0:  'empty',                # 空气/无物体
    1:  'barrier',              # 护栏/路障
    2:  'bicycle',              # 自行车
    3:  'bus',                  # 公交车
    4:  'car',                  # 小汽车
    5:  'construction_vehicle', # 工程车辆
    6:  'motorcycle',           # 摩托车
    7:  'pedestrian',           # 行人
    8:  'traffic_cone',         # 交通锥
    9:  'trailer',              # 拖车/挂车
    10: 'truck',                # 卡车
    11: 'driveable_surface',    # 可行驶路面
    12: 'other_flat',           # 其他平坦表面
    13: 'sidewalk',             # 人行道
    14: 'terrain',              # 地形(草地等)
    15: 'manmade',              # 人造建筑
    16: 'vegetation',           # 植被
    17: 'general_object',       # 通用障碍物
}
```

---

## 3. 网络架构解析

### 3.1 OccNetV3 完整流程

```
输入: [B, 8, 1, 960, 1280]  # 8 相机 Bayer RAW 图像

  ↓ MultiCameraPatchEmbed (HybridPatchEmbed)
  → Stem CNN: Conv(1→32→64, stride=2×2)
  → Patch Projection: Conv(64→192, patch=4×4)
  → 输出: 8 × [B, 3600, 192]  (60×80 patches)

  ↓ MultiCameraEncoder (Swin Transformer)
  → 4 层 WindowTransformerBlock (window=8)
  → Flash Attention + RoPE + FOV 编码
  → 输出: 8 × [B, 3600, 192]

  ↓ Feature Fusion
  → Concatenate: [B, 3600, 192×8=1536]
  → Linear Projection: [B, 3600, 192]

  ↓ BEVDecoder (Deformable Attention)
  → BEV Queries: 128×128 可学习查询
  → 3 层 DecoderLayer (Self-Attn + Cross-Attn)
  → 输出: [B, 192, 128, 128]

  ↓ LightweightTemporalFusion
  → 运动补偿 (ego_motion warp)
  → 门控融合历史帧
  → 输出: [B, 192, 128, 128]

  ↓ CoarseHeightExpansion
  → Linear(192 → 192×8)
  → 输出: [B, 192, 128, 128, 8]

  ↓ LightweightUpsampler
  → Conv3D + Trilinear Interpolate
  → 输出: [B, 96, 400, 400, 32]

  ↓ CoarseToFineHead
  → Coarse: [B, 18, 100, 100, 8]
  → Fine:   [B, 18, 400, 400, 32]
  → Flow:   [B, 3, 400, 400, 32]

输出: {
  'semantic':        [B, 18, 400, 400, 32],  # 语义分割
  'coarse_semantic': [B, 18, 100, 100, 8],   # 粗糙预测
  'flow':            [B, 3, 400, 400, 32],   # 运动流场
  'coarse_flow':     [B, 3, 100, 100, 8]     # 粗糙流场
}
```

### 3.2 参数量统计

| 模块 | 参数量 |
|------|--------|
| MultiCameraPatchEmbed | ~1.5M |
| CameraPositionEncoding | ~0.1M |
| MultiCameraEncoder | ~15M |
| BEVDecoder | ~8M |
| TemporalFusion | ~0.5M |
| HeightExpansion | ~0.3M |
| CoarseToFineHead | ~2M |
| **总计** | **~27M** |

### 3.3 关键配置参数

```python
# occ_network/configs/default.py
class Config:
    # 图像输入
    image_size = (960, 1280)
    patch_size = 16
    num_cameras = 8
    in_channels = 1          # Bayer RAW 单通道

    # 体素输出
    voxel_size = (400, 400, 32)
    voxel_resolution = 0.2
    pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
    num_classes = 18

    # Transformer
    embed_dim = 192
    num_heads = 6
    num_encoder_layers = 4
    num_decoder_layers = 3
    window_size = 8

    # BEV
    bev_size = (128, 128)

    # Coarse-to-Fine
    coarse_voxel_size = (100, 100, 8)

    # 训练
    batch_size = 1
    lr = 1e-4
    max_epochs = 100
    use_amp = True           # 混合精度
    use_checkpoint = True    # 梯度检查点
```

---

## 4. 环境配置

### 4.1 硬件要求

| 配置项 | 最低要求 | 推荐配置 |
|--------|----------|----------|
| GPU | RTX 3070 (8GB) | RTX 4090 (24GB) |
| 显存 | 8GB (batch=1, AMP) | 24GB (batch=4) |
| 内存 | 32GB | 64GB |
| 存储 | 100GB SSD | 500GB NVMe |

### 4.2 软件依赖

```bash
# 激活 conda 环境
conda activate carla

# 核心依赖
pip install torch==2.7.1+cu118 torchvision==0.18.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# 数据处理
pip install numpy opencv-python rawpy pillow

# 可选: 稀疏卷积 (提升推理性能)
pip install spconv-cu118  # 或 torchsparse
```

### 4.3 验证安装

```bash
# 验证 PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 验证网络结构
cd d:\code\carla
python occ_network/verify_network_shapes.py
```

---

## 5. 训练流程

### 5.1 数据准备

#### Step 1: 启动 CARLA 服务器

```bash
# 在 x64 Native Tools Command Prompt for VS 2022 中运行
cd d:\code\carla
cmake --build Build --target launch
```

#### Step 2: 采集数据

```bash
# 在新的终端中运行 (需要激活 carla 环境)
conda activate carla
cd d:\code\carla

# 采集 1000 帧数据 (约需 30-60 分钟)
python occnetv3_data_generator/main_collection.py \
    --frames 1000 \
    --output dataset_1k \
    --num-vehicles 30 \
    --num-walkers 10
```

**采集参数说明:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--frames` | 10 | 采集帧数 |
| `--output` | dataset_10k_bak | 输出目录 |
| `--town` | Town10HD | CARLA 地图 |
| `--num-vehicles` | 30 | NPC 车辆数量 |
| `--num-walkers` | 10 | NPC 行人数量 |

#### Step 3: 生成训练/验证集划分

```bash
# 自动划分 (80% 训练, 20% 验证)
cd d:\code\carla
python -c "
import os
from pathlib import Path
import random

data_dir = Path('dataset_1k')
samples = [p.stem for p in (data_dir / 'occupancy').glob('*.npy')]
random.shuffle(samples)

split_idx = int(len(samples) * 0.8)
train_samples = samples[:split_idx]
val_samples = samples[split_idx:]

with open(data_dir / 'train.txt', 'w') as f:
    f.write('\n'.join(train_samples))

with open(data_dir / 'val.txt', 'w') as f:
    f.write('\n'.join(val_samples))

print(f'训练集: {len(train_samples)}, 验证集: {len(val_samples)}')
"
```

### 5.2 开始训练

```bash
conda activate carla
cd d:\code\carla

# 基础训练命令
python occ_network/train.py \
    --dataset dataset_1k \
    --batch-size 1 \
    --epochs 50 \
    --lr 1e-4 \
    --amp

# 完整训练命令 (带所有参数)
python occ_network/train.py \
    --dataset dataset_1k \
    --batch-size 1 \
    --epochs 100 \
    --lr 1e-4 \
    --amp \
    --grad-clip 1.0 \
    --save-dir checkpoints/occnetv3_exp1 \
    --log-interval 10
```

**训练参数说明:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | ./data | 数据集目录 |
| `--batch-size` | 1 | 批量大小 (显存 8GB 建议 1) |
| `--epochs` | 100 | 训练轮数 |
| `--lr` | 1e-4 | 学习率 |
| `--amp` | False | 启用混合精度训练 |
| `--grad-clip` | 1.0 | 梯度裁剪 |
| `--save-dir` | ./checkpoints | 模型保存目录 |
| `--resume` | None | 恢复训练的 checkpoint 路径 |

### 5.3 训练日志解读

```
Training on cuda
AMP: True, Checkpoint: True
Coarse-to-Fine: True, Sparse: Disabled
Total parameters: 27.35M
Trainable parameters: 27.35M

Epoch 0 Step 0 Loss: 3.2541 Focal: 1.8234 Dice: 0.9123 GPU Mem: 6.82GB Data: 0.125s Batch: 1.234s
Epoch 0 Step 10 Loss: 2.8765 Focal: 1.5432 Dice: 0.8765 GPU Mem: 7.12GB Data: 0.089s Batch: 0.876s
...
Epoch 0 Train Loss: 2.5432
Epoch 0 Val Loss: 2.3210
```

**关键指标:**

- **Loss**: 总损失 (Focal + Dice + Flow)
- **Focal**: Focal Loss (关注难分类样本)
- **Dice**: Dice Loss (类别平衡)
- **GPU Mem**: 显存占用
- **Data**: 数据加载时间
- **Batch**: 单批次训练时间

### 5.4 训练技巧

#### 显存不足时的优化

1. **启用混合精度**: `--amp` (节省约 40% 显存)
2. **减小 batch_size**: `--batch-size 1`
3. **启用梯度检查点**: 已默认开启 (`config.use_checkpoint = True`)
4. **减小图像分辨率**: 修改 `config.image_size`

#### 加速训练

1. **多 GPU 训练**: 使用 `torch.nn.DataParallel`
2. **增加 num_workers**: `--num-workers 4`
3. **使用 NVMe SSD**: 提升数据加载速度

#### 防止过拟合

1. **数据增强**: 在 Dataset 中添加随机翻转/旋转
2. **Dropout**: 调整 `config.drop_rate`
3. **Early Stopping**: 监控验证集 loss

---

## 6. 推理与评估

### 6.1 推理

```bash
cd d:\code\carla

# 基准测试 (推理速度 + 显存)
python occ_network/inference.py --benchmark

# 训练显存测试
python occ_network/inference.py --train_mem

# 加载 checkpoint 推理
python occ_network/inference.py \
    --checkpoint checkpoints/occnetv3_exp1/best.pth \
    --benchmark

# MC Dropout 不确定性估计
python occ_network/inference.py \
    --checkpoint checkpoints/occnetv3_exp1/best.pth \
    --uncertainty \
    --mc-samples 10
```

### 6.2 基准测试输出示例

```
==================================================
Benchmark Results (Batch Size = 1)
==================================================
Peak GPU Memory: 5.23 GB
Average Latency: 156.32 ms
FPS: 6.40
Output Shape: torch.Size([1, 18, 400, 400, 32])
==================================================
```

### 6.3 导出 ONNX

```bash
python occ_network/inference.py --export
# 输出: occ_net_v3.onnx
```

### 6.4 可视化预测结果

```python
import numpy as np
import torch
from occ_network.models import build_model
from occ_network.configs.default import config

# 加载模型
model = build_model(config).cuda()
ckpt = torch.load('checkpoints/best.pth')
model.load_state_dict(ckpt['model'])
model.eval()

# 推理
images = torch.randn(1, 8, 1, 960, 1280).cuda()
with torch.no_grad():
    outputs = model.inference(images)

# 获取预测
pred = outputs['pred'].cpu().numpy()  # (1, 400, 400, 32)

# 保存为 NPZ (供 occupancy_viewer 可视化)
np.savez_compressed('prediction.npz',
    occupancy=pred[0].astype(np.uint8),
    mask=np.ones_like(pred[0], dtype=bool),
    x_range=config.pc_range[:3:2],
    y_range=config.pc_range[1:4:2],
    z_range=config.pc_range[2:5:2],
    resolution=config.voxel_resolution,
    grid_size=config.voxel_size
)
```

---

## 7. 常见问题

### Q1: DNG 图像加载失败

**错误**: `OpenCV TIFF: Sorry, can not handle PhotometricInterpretation=32803`

**解决**:
```bash
pip install rawpy
```

`occ_network/data/dataset.py` 会优先使用 `rawpy` 加载 DNG。

### Q2: 显存不足 (OOM)

**解决方案**:
1. 添加 `--amp` 启用混合精度
2. 设置 `--batch-size 1`
3. 修改 `config.py` 降低 `embed_dim` (如 192 → 128)

### Q3: 训练 Loss 不下降

**检查项**:
1. 确认数据加载正确 (检查 `train.txt` 是否存在)
2. 检查 class_weights 是否合理 (默认配置中 empty=0.1 可能过低)
3. 降低学习率 (如 1e-4 → 5e-5)

### Q4: 预测结果全是 empty (类别 0)

**原因**: Class 0 权重过低导致网络忽略其他类别

**解决**: 修改 `occ_network/configs/default.py`:
```python
class_weights = [1.0, 3.0, 12.0, ...]  # 将 0.1 改为 1.0
```

### Q5: 数据采集卡死

**原因**: CARLA 同步模式下 Traffic Manager 端口被占用

**解决**:
1. 重启 CARLA 服务器
2. 运行 `occnetv3_data_generator/cleanup_world.py` 清理残留 Actor

### Q6: Flow 数据是什么?

**说明**: Flow 是 3D 运动场，表示每个体素在下一帧的位移向量 (vx, vy, vz)。
- 用于训练 Flow Head，支持运动预测
- 仅对动态物体 (车辆、行人) 有非零值
- 可选功能，设置 `use_flow=False` 禁用

---

## 附录

### A. 完整训练脚本示例

```bash
#!/bin/bash
# train_occnetv3.sh

# 环境设置
conda activate carla
cd d:\code\carla

# 数据采集 (约 2 小时采集 10000 帧)
echo "Step 1: 数据采集"
python occnetv3_data_generator/main_collection.py \
    --frames 10000 \
    --output dataset_10k \
    --town Town10HD \
    --num-vehicles 50 \
    --num-walkers 20

# 划分数据集
echo "Step 2: 划分数据集"
python -c "
import os
from pathlib import Path
import random
data_dir = Path('dataset_10k')
samples = [p.stem for p in (data_dir / 'occupancy').glob('*.npy')]
random.shuffle(samples)
split_idx = int(len(samples) * 0.8)
with open(data_dir / 'train.txt', 'w') as f:
    f.write('\n'.join(samples[:split_idx]))
with open(data_dir / 'val.txt', 'w') as f:
    f.write('\n'.join(samples[split_idx:]))
print(f'Train: {split_idx}, Val: {len(samples)-split_idx}')
"

# 训练
echo "Step 3: 开始训练"
python occ_network/train.py \
    --dataset dataset_10k \
    --batch-size 1 \
    --epochs 100 \
    --lr 1e-4 \
    --amp \
    --grad-clip 1.0 \
    --save-dir checkpoints/occnetv3_10k

echo "训练完成!"
```

### B. 网络配置快速参考

```python
# 轻量级配置 (适合 8GB 显存)
class LightConfig(Config):
    embed_dim = 128
    num_heads = 4
    num_encoder_layers = 2
    num_decoder_layers = 2
    bev_size = (64, 64)

# 高精度配置 (适合 24GB 显存)
class HighConfig(Config):
    embed_dim = 256
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 4
    bev_size = (256, 256)
```

### C. 损失函数权重调优

```python
# 平衡配置 (推荐)
class_weights = [
    1.0,   # 0: empty
    3.0,   # 1: barrier
    12.0,  # 2: bicycle (稀有)
    5.0,   # 3: bus
    3.0,   # 4: car
    8.0,   # 5: construction_vehicle
    12.0,  # 6: motorcycle (稀有)
    15.0,  # 7: pedestrian (重要+稀有)
    10.0,  # 8: traffic_cone
    5.0,   # 9: trailer
    5.0,   # 10: truck
    1.0,   # 11: driveable_surface (大面积)
    2.0,   # 12: other_flat
    2.0,   # 13: sidewalk
    2.0,   # 14: terrain
    2.0,   # 15: manmade
    2.0,   # 16: vegetation
    0.5,   # 17: general_object (兜底类)
]
```

### D. 数据加载器 (PyTorch 示例)

以下是一个更完整的 PyTorch Dataset 实现，用于加载该数据集：

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json

class OccNetDataset(Dataset):
    """
    OccNetV3 数据集加载器
    支持 DNG (Bayer RAW) 和 NPY 两种图像格式
    """
    def __init__(self, data_root, split='train', use_fp16=True):
        self.data_root = Path(data_root)
        self.split = split
        self.use_fp16 = use_fp16

        # 加载样本列表
        split_file = self.data_root / f'{split}.txt'
        if split_file.exists():
            with open(split_file, 'r') as f:
                self.samples = [line.strip() for line in f.readlines()]
        else:
            # 如果没有 split 文件，遍历 occupancy 目录
            self.samples = sorted([p.stem for p in (self.data_root / 'occupancy').glob('*.npy')])

        # 加载标定参数
        self.intrinsics = {}
        self.extrinsics = {}
        if (self.data_root / 'calibration/intrinsics.json').exists():
            with open(self.data_root / 'calibration/intrinsics.json') as f:
                self.intrinsics = json.load(f)
        if (self.data_root / 'calibration/extrinsics.json').exists():
            with open(self.data_root / 'calibration/extrinsics.json') as f:
                self.extrinsics = json.load(f)

        # 相机列表 (固定顺序)
        self.cameras = ['front_main', 'front_wide', 'front_narrow', 'left_pillar',
                       'right_pillar', 'left_repeater', 'right_repeater', 'rear']

    def __len__(self):
        return len(self.samples)

    def _load_dng_image(self, dng_path):
        """加载 DNG 格式图像 (12-bit Bayer RAW)"""
        try:
            import rawpy
            with rawpy.imread(str(dng_path)) as raw:
                img = raw.raw_image_visible.astype(np.float32)
                img = img / raw.white_level  # 归一化到 [0, 1]
                img = img[np.newaxis, :, :]  # 添加通道维度 (1, H, W)
                return img
        except ImportError:
            import cv2
            img = cv2.imread(str(dng_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"无法加载 DNG: {dng_path}")
            img = img.astype(np.float32) / 4095.0
            img = img[np.newaxis, :, :]
            return img

    def __getitem__(self, idx):
        sample_id = self.samples[idx]

        # 1. 加载图像 (8个相机)
        images = []
        for cam in self.cameras:
            # 尝试 DNG 格式
            dng_path = self.data_root / 'cameras' / cam / f'{sample_id}.dng'
            # 或 NPY 格式
            npy_path = self.data_root / 'images' / sample_id / f'cam_{cam}.npy'

            if dng_path.exists():
                img = self._load_dng_image(dng_path)
            elif npy_path.exists():
                img = np.load(npy_path).astype(np.float32)
            else:
                # 降级: 生成随机数据
                img = np.zeros((1, 960, 1280), dtype=np.float32)

            images.append(torch.from_numpy(img))

        # Stack images: (8, 1, H, W)
        image_tensor = torch.stack(images)

        # 2. 加载 Occupancy Label
        occ_path = self.data_root / 'occupancy' / f'{sample_id}.npy'
        occupancy = np.load(occ_path).astype(np.int64)

        # 3. 加载 Flow (可选)
        flow_path = self.data_root / 'flow' / f'{sample_id}.npy'
        if flow_path.exists():
            flow = np.load(flow_path).astype(np.float32)
        else:
            flow = np.zeros((3, 400, 400, 32), dtype=np.float32)

        # 4. 加载 Ego Motion/Pose (可选)
        motion_path = self.data_root / 'ego_motion' / f'{sample_id}.npy'
        pose_path = self.data_root / 'ego_pose' / f'{sample_id}.npy'

        ego_motion = np.load(motion_path) if motion_path.exists() else np.eye(4, dtype=np.float32)
        ego_pose = np.load(pose_path) if pose_path.exists() else np.eye(4, dtype=np.float32)

        # 转换为合适的数据类型
        dtype = torch.float16 if self.use_fp16 else torch.float32

        return {
            'images': image_tensor.to(dtype),
            'semantic': torch.from_numpy(occupancy),  # long tensor
            'flow': torch.from_numpy(flow).to(dtype),
            'flow_mask': torch.ones(occupancy.shape, dtype=torch.bool),
            'ego_motion': torch.from_numpy(ego_motion).to(dtype),
            'ego_pose': torch.from_numpy(ego_pose).to(dtype),
            'sample_id': sample_id
        }


def build_dataloader(data_root, split='train', batch_size=1, num_workers=0):
    """构建 DataLoader"""
    dataset = OccNetDataset(data_root, split)
    shuffle = (split == 'train')
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
```

---

**文档版本**: 1.0
**更新日期**: 2026-01-19
**适用代码**: occ_network @ ue5-dev branch
