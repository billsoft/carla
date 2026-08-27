# e2e_occ 网络架构

> 本文档对照 `e2e_occ/` 下的实际代码逐模块核对过（2026-08-27），不是从旧文档转写。
> 如果代码和本文档不一致，以代码为准，并请更新本文档。

## 1. 定位

`e2e_occ` 是基于 `occnetv3_data_generator` 采集数据训练的端到端 3D 占用网格（Occupancy
Grid）预测网络：输入 8 路 Bayer RAW 鱼眼图像，输出 `(400, 400, 32)`、18 类语义的体素网格。
"端到端"指 3D 查询点到 2D 图像特征的对应关系由网络通过 Deformable Cross-Attention
自主学习，不做显式深度估计或 BEV Pooling（LSS 类方法的做法）。

参数量（`E2EOccNet.get_num_params()`，2026-08-27 实测）：

| 模块 | 参数量 |
|---|---|
| `MultiCameraPatchEmbed` (raw_embed.py) | 0.96M |
| `ImageEncoder` (image_encoder.py) | 1.66M |
| `OccupancyDecoder` 合计 (occ_decoder.py) | 6.68M |
| ├─ coarse `DeformableDecoderLayer` × 2 | 2.24M |
| ├─ fine `DeformableDecoderLayer` × 2 | 1.71M |
| ├─ `TemporalFusionModule` | 1.18M |
| ├─ `coarse_to_fine` MLP | 0.26M |
| └─ `fine_spatial_conv` (Depthwise Conv3D) | 0.008M |
| `VoxelHead` (voxel_head.py) | 1.18M |
| **总计** | **10.49M** |

推理显存：`verify_network.py` 在 RTX 4090 上跑两步时序前向（batch=1, FP32）峰值 2.52GB。
**这是脚本自测数据，不是训练/推理吞吐量基准**——本仓库尚未在等距投影迁移后的新数据集
（`dataset_10k`）上跑过完整训练，速度/mIoU 等指标需要在那之后重新测量并补进本文档，
不要沿用旧版本文档里的数字（那些数字来自迁移前、相机模型和数据管线都不同的一次训练）。

## 2. 整体数据流

```
images [B,8,1,960,1280] (Bayer RAW, uint16/float32 归一化到 [0,1])
  │
  ▼ MultiCameraPatchEmbed (raw_embed.py)
feats [B,8,256,60,80]                              ← 8× downsample (rggb_conv) × 2×2×2 (stem)=16×
  │
  ▼ ImageEncoder (image_encoder.py)
feats [B,8,256,60,80]                              ← 射线方向编码 + Window Attention ×2
  │
  ▼ OccupancyDecoder — Coarse (occ_decoder.py)
coarse_feats [B,256,25,25,8]                       ← 5,000 个 3D 查询, Deformable Cross-Attn ×2
  │
  ▼ TemporalFusionModule (temporal_fusion.py)       ← 仅 config.use_temporal=True 时
fused_feats [B,256,25,25,8], new_memory
  │
  ▼ OccupancyDecoder — Fine (occ_decoder.py)
fine_feats [B,80,80,16,256]                        ← 三线性上采样+MLP, 102,400 个查询, Deformable Cross-Attn ×2(梯度检查点)
  │
  ▼ VoxelHead (voxel_head.py)
logits [B,18,400,400,32]                           ← 交织降维+上采样, 分类推迟到全分辨率
```

`E2EOccNet.forward`（`e2e_occ_net.py`）就是这条链路的直接串联，返回
`{'semantic': logits, 'memory': new_memory}`。

## 3. 各模块

### 3.1 MultiCameraPatchEmbed（`raw_embed.py`）

输入 `[B, N=8, C=1, H=960, W=1280]` 的单通道 Bayer RAW。

```python
x = rggb_conv(x)   # Conv2d(1→4, k=2, s=2)                 [B*N, 4,  480, 640]
x = stem(x)         # Conv2d(4→64,   k=3, s=2) BN GELU       [B*N, 64,  240, 320]
                     # Conv2d(64→128, k=3, s=2) BN GELU       [B*N, 128, 120, 160]
                     # Conv2d(128→256,k=3, s=2) BN GELU       [B*N, 256, 60,  80]
                     # Conv2d(256→256,k=3, s=1) BN            [B*N, 256, 60,  80]
```

`rggb_conv` 是可学习的 2×2 卷积（不是手工 RGGB 采样），让网络自适应学习颜色分离方式。
`stem` 共 4 层，**前 3 层 stride=2，最后 1 层 stride=1**（旧文档写的是"3 层，stride
2,2,1"，少数了一层——4 层里只有最后一层 stride=1）。总下采样倍数 = rggb_conv(2×) ×
stem 前 3 层(2×2×2) = 16×，对应 `config.feat_size = image_size // 16 = (60, 80)`。

`MultiCameraPatchEmbed` 在 `patch_embed` 之后接一个 `LayerNorm`（对 channel 维做归一化，
需要先 permute 到 channel-last 再 permute 回去）。

### 3.2 ImageEncoder（`image_encoder.py`）

输入特征图 `[B,8,256,60,80]` + 相机内外参。**逐相机串行处理**（`for i in range(N)`），
不是把 8 个相机拼 batch 一起算。

**射线方向编码**（`RayDirectionEncoding`，见 `position_encoding.py`，仅
`config.use_ray_encoding=True` 时启用）为每个特征图像素算出对应的入射光线方向并加到特征上，
详见第 4 节。

**Window Attention**（`WindowAttention`，`window_size=7`，2 层 `EncoderBlock`）：把
60×80 特征图 pad 到 7 的倍数后切成互不重叠的 7×7 窗口，仅在窗口内做自注意力，避免
`O((H·W)²)` 的全局注意力开销。每个 `EncoderBlock` 是标准的 Pre-Norm 结构：
`x = x + Attn(LN(x)); x = x + MLP(LN(x))`（MLP 为 `dim→4dim→dim`, GELU）。

输出仍是 `[B,8,256,60,80]`。

### 3.3 等距投影几何（`position_encoding.py` + `deformable_attention.py`）

这是本网络里**唯一必须自洽**的一对函数——一个做反投影（像素→射线方向），一个做正投影
（3D 点→像素），必须是同一个相机模型的逆变换，否则网络学到的 3D→2D 对应关系和射线方向
编码互相矛盾（这正是 2026-08-26/27 那次问题：网络里两处实现分别用了等距和针孔两种投影
模型）。相机本身用的是 CARLA `sensor.camera.rgb_fisheye` + `camera_model=equidistant`
（`Unreal/.../Util/CameraModelUtil.cpp::ComputeDistance` 的 Equidistant 分支：
`f = (Height/2) / (FOV/2)`，`r = f·θ`），网络这两处必须用完全相同的公式。

**焦距换算**（`rescale_focal_to_feature_map`，两处共用）：`intrinsics` 是按原始图像
分辨率（1280×960）标定的，但射线编码/参考点投影都在下采样后的特征图分辨率（如
60×80）上逐像素运算。直接用原图焦距会让特征图尺度的像素偏移（几十）除以原图尺度的焦距
（几百上千），把入射角压缩到几乎全部指向正前方。用 `intrinsics` 自带的主点
`(cx_orig, cy_orig)` 和目标 `(H, W)` 推出降采样比例，把焦距换算到特征图像素单位。

**反投影**（`RayDirectionEncoding.get_rays_from_params`，像素 → 世界系射线方向）：

```python
dx, dy = x - W/2, y - H/2                  # 特征图像素坐标，以主点为原点
f = rescale_focal_to_feature_map(intrinsics, H, W)
r = sqrt(dx**2 + dy**2); phi = atan2(dy, dx)
theta = r / f                               # 等距投影核心公式
cam_dir = [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]   # 相机系，已是单位向量
world_dir = R @ cam_dir                     # R = extrinsics[:3,:3]（Camera→World）
```

再做正弦位置编码（10 个频率）+ MLP 投影到 `embed_dim`，与图像特征相加。

**正投影**（`DeformableCrossAttention.get_reference_points`，3D 点 → 像素，是上式的精确逆变换）：

```python
cam_point = inv(extrinsics) @ world_point   # 世界系 → 相机系
theta = acos(cam_point.z / |cam_point|)     # 与光轴夹角
phi = atan2(cam_point.y, cam_point.x)
f = rescale_focal_to_feature_map(intrinsics, H, W)
r_img = f * theta
u, v = W/2 + r_img*cos(phi), H/2 + r_img*sin(phi)
```

两者的自洽性由 `verify_network.py::verify_equidistant_geometry` 做往返一致性检查
（3D 点正投影到像素，再从该像素反查射线方向，应基本指回原方向），round-trip 误差
< 2°（实测约 0.19°）。**修改任一处投影公式时都要同步改另一处，并重新跑这个检查。**

### 3.4 OccupancyDecoder — Coarse 阶段（`occ_decoder.py`）

3D 查询网格 `coarse_size=(25,25,8)=5,000` 个查询点，坐标归一化到 `[0,1]³`，对应体素空间
`voxel_range=(-40,-40,-1, 40,40,5.4)` 米（X/Y ±40m，Z −1~5.4m）。查询向量 = 可学习
`nn.Parameter` + `SineCosinePositionEncoding3D`，经过 2 层 `DeformableDecoderLayer`
（见 3.6 节），`config.use_self_attention=True` 时每层还有一次标准 `nn.MultiheadAttention`
自注意力（5,000 个查询互相看得起，量级可接受）。

输出 reshape 成 `[B,256,25,25,8]` 的体素特征。

### 3.5 TemporalFusionModule（`temporal_fusion.py`）

仅 `config.use_temporal=True` 时启用，输入上一帧的 `memory [B,5000,256]`（初始为
`None`，此时用全零初始化）和 `ego_motion [B,4,4]`（语义：`C_{t-1}→C_t`，上一帧体素坐标系
到当前帧体素坐标系的变换，由 `train.py` 算出
`ego_motion = inv(extrinsics_t[:,0]) @ extrinsics_{t-1}[:,0]`）。

**Ego-Motion Alignment**（`align_memory`）：把当前帧体素网格的每个点 `p_t` 反查其在
上一帧坐标系中的位置 `p_{t-1} = inv(ego_motion) @ p_t`，再用 3D `grid_sample`
（`align_corners=True`）从上一帧 memory 里三线性采样出对齐后的特征——如果不做这一步，
车辆前进后上一帧"前方 10m 处的车"会被错误地融合到当前帧"前方 9m"的位置。

对齐后：`EfficientTemporalAttention`（`F.scaled_dot_product_attention`，PyTorch 自动选
FlashAttention/MemEfficient kernel）做 Q=当前帧/K=V=对齐后的历史帧 的注意力，残差相加后
过一个 FFN（`dim→4dim→dim`），最后 `GRUGate`（标准 GRU 更新门 + 重置门）产出
`new_memory`。训练/推理都可选开 `use_checkpoint=True` 的梯度检查点。

### 3.6 DeformableCrossAttention / DeformableDecoderLayer（`deformable_attention.py`）

`DeformableDecoderLayer` = 可选 Self-Attention（`use_self_attn`，coarse 层默认开、fine
层默认关，102,400 个查询开自注意力会 OOM）→ `DeformableCrossAttention` → MLP，均为
Pre-Norm + 残差。

`DeformableCrossAttention.forward` 的核心是**逐相机串行采样**（显存优化，避免
`8×` 并行造成的显存峰值）：

```python
reference_points = get_reference_points(...)        # [B,N=8,Q,2]，等距正投影（见3.3）
offsets = tanh(Linear(query)) * 0.5                  # [B,Q,N,heads,points,2]，可学习偏移
attn_weights = softmax(Linear(query), dim=points)    # 每个相机、每个头内部对 points 做 softmax
output = 0
for cam in range(8):                                  # 串行
    for h in range(num_heads):                         # 每个相机内部再按头循环
        loc = reference_points[cam] + offsets[..., cam, h, :, :]
        sampled = grid_sample(value[cam, h], loc)      # 双线性采样
        output += (sampled * attn_weights[..., cam, h, :]).sum(points)
```

注意：`attn_weights` 的 softmax 是在 `num_points` 维度上做的（每个相机、每个头内部
`num_points=4` 个采样点之间归一化），**8 个相机之间没有 softmax 归一化，是直接累加**
（`output = output + output_cam`）——网络靠 `sampling_offsets`/`attention_weights` 两个
线性层的训练自己学会哪个相机该贡献多少、貌似不可见的相机贡献趋近于 0，而不是显式的跨
相机竞争机制。

### 3.7 OccupancyDecoder — Fine 阶段（`occ_decoder.py`）

Coarse 输出 `[B,256,25,25,8]` 三线性上采样到 `fine_size=(80,80,16)`，经过
`coarse_to_fine`（`LN→Linear(256→512)→GELU→Dropout→Linear(512→256)→LN`）变换成
102,400 个查询的初值，加 3D 位置编码后过 2 层 `DeformableDecoderLayer`
（`use_self_attn=False`，**训练时强制梯度检查点** `checkpoint_fine=True`）。

之后过一个 Depthwise Conv3D 残差块做空间一致性精化：
`Conv3d(256→256, k=3, groups=256) → BN3D → GELU`，参数量仅 `256×3³≈6.9K`（远小于标准
卷积的 `256²×3³≈1.8M`），残差相加。

输出 `[B,80,80,16,256]`。

### 3.8 VoxelHead（`voxel_head.py`）

**这是和早期版本差异最大的模块，务必对照代码而不是记忆/旧文档**——当前实现是"交织
降维+上采样"方案，把分类决策推迟到最高分辨率（400×400）才做，而不是在 80×80×16 就
先分类到 18 类再插值：

```
输入 [B,256,80,80,16]
  → Conv3d(256→128,k=3)+BN+GELU → Conv3d(128→64,k=3)+BN+GELU     # 仍在 80×80×16，降到 64 通道
  → 三线性插值到 200×200×32
  → Conv3d(64→32,k=3)+BN+GELU  加  Conv3d(64→32,k=1) 残差         # refine1 + skip1
  → 三线性插值到 400×400×32
  → Conv3d(32→18,k=3)+BN       加  Conv3d(32→18,k=1) 残差         # refine2 + skip2（无激活，直接是 logits）
输出 [B,18,400,400,32]
```

每次上采样后先用较高通道数（64→32→18，而不是直接 18）做 3×3 卷积精化+ 1×1 卷积残差
对齐，理由是分类边界在高分辨率下才做决策，能保留更多空间细节，而不是在低分辨率先分类
再靠插值"猜"边界。

### 3.9 语义类别（18 类，对齐 nuScenes）

权威定义见 `occnetv3_data_generator/config/occupancy_config.py`，不要在文档里维护
第二份拷贝（容易和代码不同步）。要点：`0: free` 在损失函数里权重必须 ≥1.0，
`11: driveable_surface` 与 `13: sidewalk` 在可见性过滤中强制保留。

## 4. 相关文档

- 训练流程/损失函数/显存优化：[`TRAINING.md`](./TRAINING.md)
- 数据格式与采集：[`../occnetv3_data_generator/README.md`](../occnetv3_data_generator/README.md)
- 快速开始/环境要求：[`README.md`](./README.md)
