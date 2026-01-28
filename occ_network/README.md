
## 十、训练指南 (Training Guide)

### 10.1 环境准备

确保你已经激活了正确的 Conda 环境，并且安装了必要的依赖。

```bash
conda activate carla
# 推荐安装 spconv 或 torchsparse 以加速 3D 处理
# pip install spconv-cu118  # 根据 CUDA 版本选择
```

### 10.2 数据准备

OccNetV3 需要特定格式的数据集（包含多视角图像、位姿、3D 占用标签）。

1.  **切换到数据生成器目录**:
    ```bash
    cd d:\code\carla\occnetv3_data_generator
    ```
2.  **运行采集脚本**:
    ```bash
    python main_collection_v2.py --frames 1000 --output d:/code/carla/dataset_10k_bak --town Town10HD --num-vehicles 30 --num-walkers 10
    ```
    *   `--frames`: 采集帧数
    *   `--output`: 输出目录
    *   `--town`: CARLA 地图

### 10.3 开始训练

使用 `train.py` 启动训练。

```bash
cd d:\code\carla\occ_network

# 标准训练命令 (推荐)
python train.py --dataset d:\code\carla\dataset_10k_bak --batch-size 1 --epochs 20 --amp
```

**关键参数说明**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | ./data | 数据集根目录 |
| `--batch-size` | 1 | 批次大小 (单卡建议设为 1，因为模型很大) |
| `--epochs` | 100 | 训练轮数 |
| `--amp` | False | **强烈建议开启**。自动混合精度训练，节省约 50% 显存并加速。 |
| `--grad-clip` | 1.0 | 梯度裁剪阈值，防止梯度爆炸。 |
| `--save-dir` | ./checkpoints | 模型保存路径。 |

### 10.4 训练策略详解

OccNetV3 采用了一套组合策略来保证收敛和性能。

#### A. 优化器与调度
*   **Optimizer**: `AdamW`
    *   Learning Rate: `1e-4`
    *   Weight Decay: `0.01` (防止过拟合)
*   **Scheduler**: `CosineAnnealingLR`
    *   余弦退火策略，学习率随时间平滑下降，有助于模型收敛到更优的极小值。
    *   含 `warmup` 热身阶段 (前 5 epochs)。

#### B. 复合损失函数 (Loss Function)
总损失由四部分组成：

1.  **Focal Loss** ($\alpha=0.25, \gamma=2.0$):
    *   用于语义分类。解决严重的类别不平衡问题（空气体素占绝大多数）。
2.  **Dice Loss** / **Lovasz-Softmax**:
    *   直接优化 IoU (Intersection over Union)。
3.  **Distance-Aware Loss** (距离感知损失):
    *   **原理**: 对近距离的体素赋予更高权重。
    *   **公式**: $W(d) = 1.0 + \exp(-d / 10.0)$
    *   **目的**: 提升近距离（安全关键区域）的检测精度。
4.  **Depth Supervision** (深度监督):
    *   **原理**: 利用 MultiCameraEncoder 输出的特征预测深度分布。
    *   **Loss**: 对数空间 L1 损失 (`L1(log(pred), log(gt))`)。
    *   **目的**: 强迫 2D 编码器学习正确的几何特征，加速收敛。

#### C. 训练稳定性技巧
*   **Gradient Clipping**: 裁剪梯度范数 > 1.0 的梯度，防止 Transformer 训练初期的梯度爆炸。
*   **NaN Check**: 自动检测 Loss 是否为 NaN/Inf，如果是则跳过该 Batch（常见于混合精度训练的数值溢出）。
*   **Data Prefetcher**: 使用 `FP16DataPrefetcher` 将数据预加载到 GPU 并转为 FP16，减少 GPU 等待 CPU 的时间。

### 10.5 硬件要求与性能

| 配置 | 显存占用 (Training) | 速度 |
|------|-------------------|------|
| **Batch=1, FP32** | ~22 GB | 慢 |
| **Batch=1, AMP** | ~14 GB (推荐) | 快 |
| **Batch=1, AMP + Gradient Checkpoint** | ~8 GB | 较慢 (计算换空间) |

> 💡 **提示**: 如果显存不足 (如只有 8GB 显存)，请在 `configs/default.py` 中开启 `use_checkpoint = True`。

---

## 十一、推理指南 (Inference Guide)

### 11.1 运行推理

使用 `inference.py` 脚本加载训练好的模型并对数据集进行推理，生成 3D 语义占用体素。

```bash
# 运行推理 (加载 dataset_10k_bak, 结果保存到 output_voxels)
python inference.py --run-inference --dataset d:\code\carla\dataset_10k_bak --output output_voxels --checkpoint checkpoints/best_model.pth
```

**参数说明**:
*   `--run-inference`: 启用推理模式。
*   `--dataset`: 数据集路径 (默认: `d:\code\carla\dataset_10k_bak`)。
*   `--output`: 结果输出目录 (默认: `output_voxels`)。
*   `--checkpoint`: 模型权重路径。如果不指定，将使用随机初始化权重（仅用于测试流程）。
*   `--split`: 数据集划分 (默认: `test`)。

### 11.2 输出格式

脚本会在输出目录下为每个样本生成一个 `.npy` 文件。

*   **文件名**: `{sample_id}.npy` (例如 `000001.npy`)
*   **数据格式**: NumPy 数组
*   **形状**: `(400, 400, 32)`
*   **数据类型**: `uint8`
*   **内容**: 每个体素的语义标签 ID (0-17)。

```python
import numpy as np

# 读取示例
voxel = np.load('output_voxels/000001.npy')
print(voxel.shape) # (400, 400, 32)
```

### 11.3 不确定性估计 (MC Dropout)

如果需要评估模型的不确定性（例如用于安全验证），可以启用 MC Dropout 模式。

```bash
python inference.py --uncertainty --mc-samples 10
```

这将通过多次前向传播计算预测的方差和熵。


# Memory Cell (推荐，默认)
python train_tbptt.py --dataset /path/to/data --mode memory_cell --amp

# Gradient Accumulation
python train_tbptt.py --dataset /path/to/data --mode grad_accum --window 3 --amp

# Classic TBPTT (高显存)
python train_tbptt.py --dataset /path/to/data --mode classic --window 3 --amp

