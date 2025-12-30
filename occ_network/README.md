# Occupancy Network 项目结构

## 项目概览

基于多相机输入的 3D 语义占用网格预测网络。

```
occ_network/
├── configs/
│   └── default_config.py          # 配置参数（网络、数据、训练）
│
├── models/
│   ├── __init__.py
│   ├── backbone.py                # 2D 图像编码器 (EfficientNet/ResNet)
│   ├── neck.py                    # 特征金字塔网络 (FPN)
│   ├── positional_encoding.py     # 位置编码 (2D + 相机ID + 位姿)
│   ├── view_transformer.py        # 多相机→BEV 变换 (Cross Attention)
│   ├── bev_encoder.py             # BEV 特征编码器
│   ├── occ_decoder.py             # 3D 体素解码器 (2D→3D 提升)
│   └── occ_network.py             # 完整网络封装
│
├── datasets/
│   ├── __init__.py
│   └── carla_occ_dataset.py       # CARLA 数据集加载器
│
├── losses/
│   ├── __init__.py
│   └── occ_loss.py                # 损失函数 (CE + Lovász)
│
├── utils/
│   ├── __init__.py
│   ├── geometry.py                # 几何工具 (坐标变换)
│   └── metrics.py                 # 评估指标 (mIoU)
│
├── train.py                       # 训练入口
├── evaluate.py                    # 评估脚本
├── visualize.py                   # 可视化脚本
└── requirements.txt               # 依赖包
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

# 训练
python train.py --config configs/default_config.py --data_root /path/to/dataset

# 评估
python evaluate.py --checkpoint checkpoints/best.pth --data_root /path/to/dataset

# 可视化
python visualize.py --checkpoint checkpoints/best.pth --sample_idx 0
```
