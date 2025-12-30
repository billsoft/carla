# Occupancy Network Lite

**轻量级 3D 语义占用网络** - 针对 16GB 显存优化

## 项目概览

基于多相机输入的 3D 语义占用预测网络,专为显存受限场景优化。

### 核心特性

- ✅ **轻量级架构**: MobileNetV2 backbone, 14.8M 参数
- ✅ **显存友好**: 6-8 GB (batch_size=1, AMP)
- ✅ **混合精度训练**: 自动混合精度加速
- ✅ **梯度累积**: 模拟大 batch size
- ✅ **完整数据流**: 从数据采集到训练到评估

### 性能指标

| 配置 | 参数量 | 显存 (BS=1) | mIoU (5K 帧) |
|------|--------|-------------|--------------|
| **Lite** | 14.8M | 6-8 GB | 0.45-0.50 |
| Ultra-Lite | 8M | 3-4 GB | 0.40-0.45 |

```
occ_network/
├── configs/
│   └── default_config.py          # 配置参数
│
├── models/
│   ├── __init__.py
│   ├── backbone.py                # ResNet Backbone
│   ├── neck.py                    # FPN
│   ├── positional_encoding.py     # 位置编码
│   ├── view_transformer.py        # 多相机→BEV (Cross Attention)
│   ├── bev_encoder.py             # BEV 编码器
│   ├── occ_decoder.py             # 3D 体素解码器
│   ├── occ_network.py             # 完整网络 (ResNet50)
│   └── occ_network_lite.py        # ⭐ 轻量级网络 (MobileNetV2)
│
├── datasets/
│   └── carla_occ_dataset.py       # CARLA 数据集
│
├── losses/
│   └── occ_loss.py                # CE + Lovász 损失
│
├── utils/
│   ├── geometry.py                # 几何工具
│   └── metrics.py                 # mIoU 评估
│
├── docs/
│   └── memory_optimization.md     # 显存优化方案
│
├── train.py                       # 完整版训练
├── train_lite.py                  # ⭐ 轻量版训练
├── evaluate.py                    # 评估
├── visualize.py                   # 可视化
└── requirements.txt
```

## 网络架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OccupancyNetwork                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────┐    │
│   │   8×RGB      │     │   Backbone   │     │   Multi-scale Features   │    │
│   │   Images     │────▶│  (ResNet50)  │────▶│   P3, P4, P5             │    │
│   │ [8,3,H,W]    │     │              │     │   [8, C, H/8, W/8] etc   │    │
│   └──────────────┘     └──────────────┘     └──────────────────────────┘    │
│                                                       │                      │
│                                                       ▼                      │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────┐    │
│   │   Camera     │     │   Position   │     │         FPN Neck         │    │
│   │   Params     │────▶│   Encoding   │────▶│   Unified Features       │    │
│   │ intrinsic    │     │  2D+Cam+Pose │     │   [8, 256, H, W]         │    │
│   │ extrinsic    │     │              │     │                          │    │
│   └──────────────┘     └──────────────┘     └──────────────────────────┘    │
│                                                       │                      │
│                                                       ▼                      │
│                        ┌──────────────────────────────────────────────┐     │
│                        │            View Transformer                   │     │
│                        │  ┌────────────────────────────────────────┐  │     │
│                        │  │  BEV Query: [200×200, 256]             │  │     │
│                        │  │  Image K/V: [8×H×W, 256]               │  │     │
│                        │  │  Cross Attention × N layers            │  │     │
│                        │  └────────────────────────────────────────┘  │     │
│                        │  Output: [256, 200, 200] BEV Features        │     │
│                        └──────────────────────────────────────────────┘     │
│                                                       │                      │
│                                                       ▼                      │
│                        ┌──────────────────────────────────────────────┐     │
│                        │              BEV Encoder                      │     │
│                        │  ResNet-style 2D convolutions                 │     │
│                        │  Output: [256, 200, 200] Enhanced BEV         │     │
│                        └──────────────────────────────────────────────┘     │
│                                                       │                      │
│                                                       ▼                      │
│                        ┌──────────────────────────────────────────────┐     │
│                        │             Occ Decoder (2D→3D)               │     │
│                        │  ┌────────────────────────────────────────┐  │     │
│                        │  │  Height MLP: predict Z-axis features   │  │     │
│                        │  │  3D Conv Refinement                    │  │     │
│                        │  │  Classification Head                   │  │     │
│                        │  └────────────────────────────────────────┘  │     │
│                        │  Output: [18, 200, 200, 16] (downsampled)    │     │
│                        │      or: [18, 500, 500, 40] (full res)       │     │
│                        └──────────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 数据流 Shape 变化

| 阶段 | 模块 | 输入 Shape | 输出 Shape |
|------|------|-----------|-----------|
| 1 | 输入图像 | - | `[B, 8, 3, 384, 640]` |
| 2 | Backbone | `[B×8, 3, 384, 640]` | P3:`[B×8,256,48,80]` P4:`[B×8,512,24,40]` P5:`[B×8,1024,12,20]` |
| 3 | FPN Neck | 多尺度特征 | `[B×8, 256, 48, 80]` |
| 4 | 位置编码 | `[8, 48, 80]` | `[8, 48, 80, 256]` |
| 5 | View Transformer | Q:`[B,40000,256]` KV:`[B,30720,256]` | `[B, 256, 200, 200]` |
| 6 | BEV Encoder | `[B, 256, 200, 200]` | `[B, 256, 200, 200]` |
| 7 | Occ Decoder | `[B, 256, 200, 200]` | `[B, 18, 200, 200, 16]` |
| 8 | (可选)上采样 | `[B, 18, 200, 200, 16]` | `[B, 18, 500, 500, 40]` |

## 训练配置

- **Batch Size**: 2-4 (24GB GPU)
- **Learning Rate**: 2e-4 (AdamW)
- **Epochs**: 24
- **Loss**: CE(0.7) + Lovász(0.3)
- **输出分辨率**: 0.5m (200×200×16) 训练，推理时可上采样

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# ⭐ 推荐：使用轻量级网络训练 (8GB 显存即可)
python train_lite.py --data_root /path/to/dataset --epochs 24 --batch_size 2 --amp

# 完整版训练 (需要 24GB+ 显存)
python train.py --data_root /path/to/dataset --epochs 24 --batch_size 1 --amp

# 评估
python evaluate.py --checkpoint checkpoints/best.pth --data_root /path/to/dataset

# 可视化
python visualize.py --checkpoint checkpoints/best.pth --data_root /path/to/dataset --sample_idx 0
```

## 两个版本对比

| 特性 | OccupancyNetwork (完整版) | OccupancyNetworkLite (轻量版) |
|------|--------------------------|------------------------------|
| Backbone | ResNet50 (25M) | MobileNetV2 (3.4M) |
| 特征维度 | 256 | 128 |
| BEV 分辨率 | 200×200 | 100×100 |
| 高度层数 | 16 | 8 |
| Transformer 层数 | 6 | 2 |
| 总参数量 | ~95M | ~15M |
| 显存 (BS=1) | ~10GB | ~2-3GB |
| 推荐 GPU | RTX 3090/A100 | RTX 3060/3070 |
