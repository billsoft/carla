# e2e_occ 网络架构

> 2026-10 重构后版本，对照 `e2e_occ/` 实际代码编写。代码与本文不一致时以代码为准，并请更新本文。
> 原理性的逐模块讲解见 [`MODULES_EXPLAINED.md`](./MODULES_EXPLAINED.md)，每层输入输出形状的
> 自动生成报告见 [`shape_report.md`](./shape_report.md)（`verify_network.py` 产出）。

## 1. 定位

输入 8 路等距鱼眼 Bayer RAW，输出 `(400, 400, 32)`、18 类语义体素 + 3 维占用流（场景流）。
3D 查询点主动投影到图像取特征（BEVFormer / Tesla Occupancy 一脉），不做显式深度 lifting；
深度只作为训练时的辅助监督。

参数量（默认配置，`E2EOccNet.get_num_params()` 实测）：**13.45M**。
接近重构前结构的基线配置（关掉 P1~P3 开关）为 10.59M。

## 2. 坐标系约定（P0，无开关，始终生效）

| 对象 | 约定 |
|---|---|
| 车辆 / 体素坐标系 | CARLA 原生：X 前、Y 右、Z 上（左手系），米。体素按 `[X, Y, Z]` 索引，中心 = `vmin + (i+0.5)·cell` |
| 相机坐标系 | OpenCV：x 右、y 下、z 前（光轴）。网络内全部投影公式在此约定下 |
| 外参 `extrinsics` | **Camera(OpenCV)→Vehicle** 静态安装标定，不含车辆世界位姿 |
| `ego_motion` | 上一帧车辆系 → 当前帧车辆系 = `inv(pose_t) @ pose_{t-1}`，`pose` 来自 `ego_pose/*.npy` |
| 像素坐标 | 像素中心在整数坐标（与 CARLA `WideAngleLens.usf` 一致）；stride-s 特征像素 i ↔ 原图 `(i+0.5)s-0.5` |

所有正/反投影都调用 `geometry.py` 的同一份实现（`equidistant_project` / `equidistant_unproject`），
射线编码（反投影）与可变形注意力参考点（正投影）不可能再各写一套。

数据侧的转换在 `dataset.py` 完成：标定文件里是 UE 相机轴（x前 y右 z上），右乘
`M_CV_TO_UE` 转成 OpenCV 轴；旧版采集代码的旋转公式在 pitch/roll≠0 时符号与 UE 渲染相反
（只影响 rear 相机），按 `extrinsics.json` 的 `rotation_convention` 标记自动修正。
注意当前 LibCarla 的 `carla.Transform.get_matrix()` 经上游 #9751 改过符号，与 UE 渲染不一致，不能当标准。

**重构前的三处错误**（已修复，`verify_network.py` 第 2、3 节有对应的回归测试和反例测试）：
1. 网络把 UE 相机系的 z（向上）当光轴 → 投影整体错误；
2. 外参含车辆世界绝对位姿，3D 查询被当成世界原点附近的点 → 车离开原点后几乎全部投影到图像外；
3. 时序对齐用 0 号相机位姿算 ego_motion，且 `align_memory` 的 X/Y 轴对调。

## 3. 数据流

```
images [B,8,1,960,1280]
  │ MultiCameraPatchEmbed（raw_embed.py）
  ▼ {8: [B,8,256,120,160], 16: [B,8,256,60,80]}
  │ ImageEncoder（image_encoder.py）：各层级 + 射线编码；stride16 过 (移位)窗口注意力 ×2；
  │                                  stride32 = 下采样(stride16)；深度头接 stride8
  ▼ feats {8, 16, 32}，depth_logits [B,8,64,120,160]
  │ OccupancyDecoder 粗阶段：5,000 查询（25×25×8 体素中心）× 2 层
  │   [3D-RoPE 自注意力] → 多尺度可变形交叉注意力（可见相机归一化）→ SwiGLU FFN
  ▼ [B,5000,256]
  │ TemporalFusionModule：ego_motion 对齐记忆(+有效掩码) → 局部 3D 可变形时序注意力
  │                       (+帧间隔嵌入) → FFN → GRU → 记忆回注
  ▼ fused [B,5000,256]，memory [B,5000,256]；coarse_occ_logits [B,25,25,8]
  │ OccupancyDecoder 细阶段：三线性上采样到 80×80×16 + coarse_to_fine
  │   稀疏：按粗占用概率取 top-25%（25,600 个）细体素 × 2 层交叉注意力，scatter 回稠密
  │   Depthwise Conv3D 残差
  ▼ [B,80,80,16,256]
  │ VoxelHead（稠密，默认）或 ImplicitVoxelHead
  ▼ semantic [B,18,400,400,32]，flow [B,3,400,400,32]
```

`E2EOccNet.forward` 返回 `semantic, flow, memory, depth_logits, coarse_occ_logits, fine_selected`；
隐式头训练时（传了 `query_points`）返回 `semantic_points [B,P,18]、flow_points [B,P,3]`。

## 4. 模块与参数量

| 模块 | 文件 | 说明 |
|---|---|---|
| `MultiCameraPatchEmbed` | `raw_embed.py` | 2×2 可学习 RGGB 卷积 + 3 级 stride-2 卷积，输出 stride 8（经 1×1 投影到 256）与 stride 16 |
| `ImageEncoder` | `image_encoder.py` | 射线编码（共享 MLP）、2 个窗口注意力块（第 2 个循环移位 3 像素，Swin 掩码）、stride32 下采样、`DepthHead` |
| `RayDirectionEncoding` | `position_encoding.py` | 等距反投影到车辆系射线 → 10 频 NeRF 编码 → MLP |
| `DeformableCrossAttention` | `deformable_attention.py` | 8 相机 × 8 头 × 3 层级 × 4 点；偏移单位为该层级像素（±4px，`tanh`）；层级×点上 softmax；越界相机置零后按命中相机数平均 |
| `DeformableDecoderLayer` | `deformable_attention.py` | Pre-Norm(RMSNorm)：[`MultiHeadSelfAttention`+3D RoPE] → 交叉注意力 → SwiGLU |
| `OccupancyDecoder` | `occ_decoder.py` | 粗 2 层 + 时序 + 细 2 层；`coarse_occ_head`（稀疏选择依据 + BCE 辅助监督） |
| `TemporalFusionModule` | `temporal_fusion.py` | 对齐 / 局部可变形时序注意力（8 头 × 8 点，±2 格）/ 时间嵌入 / GRU / `mem_out` 回注 |
| `VoxelHead` | `voxel_head.py` | 256@80 → 64@80 → 32@200 → 18@400，`flow` 分支接 32@200；训练时三段梯度检查点 |
| `ImplicitVoxelHead` | `voxel_head.py` | 64 通道体素特征三线性插值 + 坐标傅里叶编码 → MLP(128) → 18+3 |
| 公共构件 | `layers.py` | `RMSNorm`/`LayerNorm` 工厂、`FeedForward`(SwiGLU/GELU)、`RoPE3D`、SDPA 自注意力 |
| 几何 | `geometry.py` | 坐标约定、CARLA 旋转公式、等距正/反投影、体素中心、grid_sample 归一化 |

## 5. 配置开关（`config.py`）

| 组 | 字段 | 默认 | 关掉时 |
|---|---|---|---|
| P1 | `use_multiscale` | True | 只用 stride 16 |
| P1 | `use_depth_aux` / `depth_bins` | True / 64 | 无深度头 |
| P1 | `use_flow_head` | True | 无流场输出 |
| P1 | `cam_visibility_norm` | True | 8 相机直接累加（重构前行为） |
| P2 | `use_sparse_fine` / `fine_keep_ratio` | True / 0.25 | 102,400 个细查询全部做交叉注意力 |
| P2 | `fine_select_mode` / `fine_select_quota` | `'topk'` / (0.5,0.25,0.25) | `'mixed'`：名额拆成 占用最高/最不确定/最近场，对照实验用 |
| P2 | `temporal_attn_type` | `'deformable'` | `'global'`：5000×5000 全局注意力 |
| P2 | `use_time_embedding` | True | — |
| P2 | `head_type` | `'dense'` | `'implicit'`：可查询隐式头 |
| P3 | `norm_type` / `ffn_type` | rmsnorm / swiglu | layernorm / gelu |
| P3 | `use_rope_3d` / `use_shifted_window` | True / True | — |
| 显存 | `checkpoint_head` / `lovasz_max_points` | True / 1,000,000 | — |

`train.py --set key=value ...` 可在命令行覆盖任意字段，方便做消融。

## 6. 实测资源占用

RTX 4090，batch=1，bf16 autocast，随机输入。测量时机器上有其他程序占用 GPU，耗时偏大，仅供量级参考：

| 配置 | 推理峰值显存（两帧） | 训练峰值显存（单帧前向+反向） |
|---|---|---|
| 默认 | 2.82 GB | 8.09 GB（`profile_network.py`），两帧流式训练 8.70 GB |
| 隐式头 | 1.71 GB | 7.97 GB |
| 接近重构前基线 | 2.31 GB | — |

单帧训练 forward+loss+backward 约 3~5 s（`profile_network.py` 分阶段：decoder 前向约 2 s、
backward 约 2.2 s）。这些是结构/资源数字，不是精度指标；mIoU 需要在真实数据集上完整训练后测量。

## 6.5 数值精度与推理开销（2026-10-15 评审后修复）

- **几何一律 FP32**：混合精度 (bf16 autocast) 会把投影里的 einsum 降成 BF16，实测细网格投影误差中位 0.97px、
  P95 3.45px、最大 7.6px 并有可见性翻转。现在 `geometry.no_autocast()` 包住相机投影、射线反投影、射线 Fourier
  编码、时序记忆对齐；所有几何采样走 `geometry.grid_sample_fp32()`（特征与坐标都 FP32）。卷积/注意力/FFN
  仍为混合精度。代价：训练峰值显存约 +1.9GB（采样用的 value 以 FP32 保存给反向），推理不变。
- **BN 与梯度检查点**：稠密头检查点反向重算时用 `voxel_head.frozen_bn_stats` 冻结 BN 统计，否则每个 batch
  被计入两次，训练态/推理态归一化不一致。
- **推理只算需要的**：`model(..., need_aux=None, need_flow=True)`，`need_aux` 默认等于 `self.training`——eval
  时不算深度辅助头（约占 14% 运算量）；`need_flow=False` 不算流场分支。`OccInference.predict(to_cpu=False)`
  结果留在设备上，接下游网络时不做每帧几十 MB 的回传。
- 回归检查见 `verify_network.py` 第 5 节。

## 7. 未实现 / 推迟的项

| 项 | 原因 |
|---|---|
| Mamba / SSM 记忆 | `mamba-ssm` 依赖 Triton/CUDA 自定义核，在 Windows 上基本不可用；当前 GRU + 局部时序注意力保留 |
| 世界模型式预训练（预测未来占用） | 需要独立的训练管线和评估方式；占用流头已覆盖部分"动态"监督 |
| `torch.compile` 默认开启 | Windows 上 Triton 不稳定，只提供 `train.py --compile` 选项 |
| 视觉大模型教师蒸馏、BEV/channel-to-height 轻量输出头、全稀疏表示 | 先用完整数据集训出基线，再用距离分段/小目标召回/长时序指标做同预算对照 (2026-10-15 评审建议) |
| 输出 latent 特征 + 观测有效性 + 不确定性给下游规划 | 端到端衔接阶段再定接口；`mask_camera` 已提供观测有效性的真值 |
