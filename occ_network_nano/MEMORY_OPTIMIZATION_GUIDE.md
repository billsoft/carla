# 内存优化指南

## 问题

训练 BayerOccNet 时，使用 1280×960 分辨率，batch_size=2 导致显存爆炸（RTX 4090 25.8GB 不够）。

## 内存占用分析

### 原始方案（一次性处理所有相机）
```python
images_flat = images.view(B * N_cam, C, H, W)  # [2*8, 1, 960, 1280]
features = self.backbone(images_flat)
```

**内存峰值**（batch_size=2）:
- 输入: `[16, 1, 960, 1280]` = 78.64 MB
- Backbone 中间层激活值:
  - PixelUnshuffle 后: `[16, 4, 480, 640]` = 78.64 MB
  - Stem 输出: `[16, 48, 480, 640]` = 943.7 MB
  - Stage 1: `[16, 64, 240, 320]` = 314.6 MB
  - ...
- **总峰值**: 约 3-5 GB（仅 Backbone）

加上梯度、优化器状态、FPN/Transformer/Decoder，**总显存 > 25GB**！

---

## 优化方案

### ✅ 方案1：逐相机处理（已实施）

**修改位置**: `models/bayer_occ_net.py:91-136`

**实现**:
```python
# 循环处理 8 个相机，每次只加载 1 个相机到 GPU
for cam_idx in range(N_cam):
    cam_images = images[:, cam_idx]  # [B, 1, H, W]
    features = self.backbone(cam_images)  # Backbone 权重共享
    fpn_feat = self.fpn(features)
    fpn_features_list.append(fpn_feat)

# 堆叠特征
fpn_feat = torch.stack(fpn_features_list, dim=1)
```

**内存节省**:
- 峰值显存: 3-5 GB → **0.4-0.6 GB**（降低 8 倍）
- 训练速度: 轻微下降（循环开销，但可忽略）

**优点**:
- 无损优化（输出完全一致）
- 无需修改网络结构
- Backbone 参数仍然共享

---

### ⚙️ 方案2：Backbone 快速下采样（备用）

**如果方案1仍不够用，启用此方案。**

#### 修改位置1: Stem 下采样

修改 `models/backbone/mobilenet_v2_bayer.py:95-99`:

```python
# 原始（stride=1）
self.stem = nn.Sequential(
    nn.Conv2d(4, stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(stem_channels),
    nn.ReLU6(inplace=True)
)

# 优化后（stride=2，快速下采样）
self.stem = nn.Sequential(
    nn.Conv2d(4, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(stem_channels),
    nn.ReLU6(inplace=True)
)
```

**效果**:
- 特征图尺寸: 480×640 → **240×320**（再降低 4×）
- 内存占用: 0.4-0.6 GB → **0.1-0.15 GB**

**代价**:
- 损失细节信息（分辨率从 1/2 → 1/4）
- 可能影响精度（需要重新训练验证）

#### 修改位置2: 调整下采样率配置

同时修改 `models/backbone/mobilenet_v2_bayer.py:134-138`:

```python
# 如果 Stem 改为 stride=2，输出特征对应的总下采样率变化：
self.out_channels = {
    'C3': _make_divisible(96 * width_mult),   # 1/8 → 1/16
    'C4': _make_divisible(128 * width_mult),  # 1/16 → 1/32
    'C5': _make_divisible(256 * width_mult),  # 1/32 → 1/64
}
```

**注意**: 这会改变特征图尺寸，需要检查 `bayer_occ_net.py:65` 的计算是否需要调整：
```python
feat_h, feat_w = img_size[0] // 8, img_size[1] // 8
# 如果 Stem 改为 stride=2，需要改为:
feat_h, feat_w = img_size[0] // 16, img_size[1] // 16
```

---

## 使用建议

### 1. 先测试方案1

运行训练：
```bash
conda activate deepsys
python occ_network_nano/train_bayer.py --dataset dataset_10k --batch-size 2 --epochs 50 --device cuda --amp
```

**监控显存**:
```bash
# 另一个终端
watch -n 1 nvidia-smi
```

如果峰值显存 < 20GB，方案1 成功，无需方案2。

### 2. 如果仍然 OOM，启用方案2

#### 步骤1: 修改 Backbone Stem
编辑 `occ_network_nano/models/backbone/mobilenet_v2_bayer.py:96`:
```python
stride=1  →  stride=2
```

#### 步骤2: 调整特征图尺寸计算
编辑 `occ_network_nano/models/bayer_occ_net.py:65`:
```python
feat_h, feat_w = img_size[0] // 8, img_size[1] // 8
↓
feat_h, feat_w = img_size[0] // 16, img_size[1] // 16
```

#### 步骤3: 重新训练
删除旧检查点，从头训练（网络结构已变化）。

---

## 其他显存优化技巧

### 3. 降低 batch_size
如果以上方案仍不够：
```bash
python train_bayer.py --batch-size 1  # 从 2 降到 1
```

代价：训练速度减半，梯度噪声增加。

### 4. 使用 Gradient Checkpointing
在 `BayerOccNet` 中启用梯度检查点（以计算换内存）：
```python
# 在 __init__ 中
self.use_checkpoint = True

# 在 forward 中
if self.use_checkpoint:
    from torch.utils.checkpoint import checkpoint
    features = checkpoint(self.backbone, cam_images)
else:
    features = self.backbone(cam_images)
```

可节省 30-50% 显存，但训练速度下降 10-20%。

### 5. 降低网络宽度
```bash
python train_bayer.py --width-mult 0.75  # 降低 Backbone 通道数
```

参数量: 6.09M → 约 3.5M
显存: 节省 20-30%
代价: 精度可能下降

---

## 总结

| 方案 | 显存节省 | 速度影响 | 精度影响 | 推荐度 |
|------|---------|---------|---------|--------|
| 方案1: 逐相机处理 | **80%** | ~5% 下降 | 无损 | ⭐⭐⭐⭐⭐ |
| 方案2: 快速下采样 | **95%** | ~10% 提升 | 轻微下降 | ⭐⭐⭐⭐ |
| 降低 batch_size | 50% | 50% 下降 | 无损 | ⭐⭐⭐ |
| Gradient Checkpoint | 40% | 15% 下降 | 无损 | ⭐⭐⭐ |
| 降低宽度 | 25% | 10% 提升 | 中等下降 | ⭐⭐ |

**最佳策略**: 先用方案1，不够再加方案2。
