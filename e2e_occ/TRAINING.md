# e2e_occ 训练

> 2026-10 重构后版本，对照 `train.py` / `loss.py` / `dataset.py` / `pretrain_mae.py` 实际代码编写。
> 网络结构见 [`ARCHITECTURE.md`](./ARCHITECTURE.md)。**重构前的 checkpoint 与新结构不兼容，需要重新训练。**

## 1. 启动

```bash
conda activate deepsys

# 先跑结构/几何自检（几何与时序方向有回归测试）
python e2e_occ/verify_network.py

# 训练（默认：时序 T=4、流式训练 tbptt=1、bf16）
python e2e_occ/train.py --data_root <dataset_dir> --batch_size 1 --epochs 100 --amp --grad_accum 4

# 消融：任意配置字段都可以在命令行覆盖
python e2e_occ/train.py --data_root <dataset_dir> --amp --set use_sparse_fine=False temporal_attn_type=global

# 可选：先用无标注 RAW 做 MAE 自监督预训练骨干，再加载
python e2e_occ/pretrain_mae.py --data_root <dataset_dir> --epochs 20 --amp
python e2e_occ/train.py --data_root <dataset_dir> --amp --pretrained_backbone checkpoints_mae/backbone.pth

# 恢复（config 从 checkpoint 读取）
python e2e_occ/train.py --resume checkpoints/last_model.pth --data_root <dataset_dir> --amp
```

Windows 下 `--num_workers` 保持 0（`DataLoader` 多进程死锁的已知问题）。
**训练时不要同时运行占用 GPU 的程序**（游戏、OBS 录屏等）：显存不够时 Windows 会把显存
溢出到系统内存，实测速度会慢几十倍且没有任何报错。

## 2. 数据（`dataset.py`）

每个样本返回带时间维 `T` 的字典（非时序配置时 `T=1`）：

| 键 | 形状 | 说明 |
|---|---|---|
| `images` | `[T,8,1,960,1280]` | DNG 读出的 Bayer RAW / `(2^raw_bit_depth-1)`，位深从 `calibration/intrinsics.json` 读 |
| `voxels` | `[T,400,400,32]` uint8 | 语义真值（GPU 上转 long）；255 = 相机未观测，损失/mIoU 忽略 |
| `time_delta` | `[T]` | 与上一帧的间隔秒数（`scenes/<scene>.json` 的 `frame_dt`，抽帧采集后不再是 0.05） |
| `ego_motion` | `[T,4,4]` | `inv(pose_t) @ pose_{t-1}`，t=0 为单位阵 |
| `intrinsics` | `[8,3,3]` | 等距投影内参（原图分辨率） |
| `extrinsics` | `[8,4,4]` | Camera(OpenCV)→Vehicle 静态标定（见 ARCHITECTURE.md 第 2 节） |
| `depth` / `has_depth` | `[T,8,120,160]` / `[T]` | 深度监督目标（米，0=无效） |
| `flow` / `flow_mask` / `has_flow` | `[T,3,400,400,32]` f16 / `[T,400,400,32]` / `[T]` | 场景流目标 |

- **序列不跨场景**：按 `scene_XXXX_frame_YYYY` 解析，只有同一场景且帧号连续的 T 帧才组成一个样本。
  相邻序列起点间隔由 `sequence_stride` 控制，默认 0 = T（不重叠，每帧每 epoch 只处理一次）；
  设为 1 时序列全部重叠，样本数约 ×T，epoch 耗时也约 ×T。
- **相机可见性掩码**（`use_camera_mask`，默认开）：有 `mask_camera/<sid>.npy` 时，相机没观测到的
  体素置 255（Occ3D 约定），语义 CE/Lovász、粗占用 BCE（全 ignore 的粗格不算）、流场（`flow_mask & 观测`）、
  验证 mIoU 都跳过它们。`camera_mask_keep_ground_surface`（默认开）保留被车/人挡住的最上层地面体素。
  没有掩码的旧数据照常在全部体素上监督并告警；补算见 `occnetv3_data_generator/compute_camera_mask.py`。
- **深度目标的换算**：采集侧深度相机是**针孔** `sensor.camera.depth`（水平 FOV = 标定 `depth_fov`，旧数据为 `fov`，
  与 RGB 同位姿），值是平面 z 深度；RGB 是等距鱼眼。对鱼眼 stride-8 的每个特征像素：等距反投影
  得射线 → 投影到针孔深度图取值 → 除以射线 z 分量得到沿射线距离。针孔覆盖不到的鱼眼边角
  （广角相机的四角）标为无效。直接按像素下标读深度图是错的。
- **流场目标语义**：物体自身速度（不减自车速度）旋转到当前车辆系 × 帧间隔（米），只在动态物体体素内非零；停着的车与建筑一样为 0，自车运动由 ego_motion 表达（label_version 2 起；之前是相对自车速度）。流场损失只在 `flow_mask ∩ 相机观测` 内计算。
- 已弃用的 `camera_params/*.npz` 格式不再支持。缺 `ego_pose` 时 ego_motion 退化为单位阵并告警。

## 3. 训练循环（`train.py::run_sequence`）

逐帧前向一个序列，记忆跨帧携带：

```python
for t in range(T):
    if t % tbptt == 0: memory = memory.detach()
    out = model(images[:, t], K, E, memory, ego_motion[:, t] if t>0 else None, time_delta)
    loss_t = E2ELoss(out, targets_t)
    chunk += loss_t * (1 + t/(T-1))          # 越靠后的帧权重越高
    if 段结束: backward(chunk / Σ_全序列 w / grad_accum)
```

时间加权按**整个序列**的权重和归一化，任意 tbptt 下帧间相对权重都保留（若按每段归一化，
tbptt=1 时 `loss·w/w` 会把加权完全抵消）。

- **`--tbptt 1`（默认）= 流式训练**：每帧单独反向、记忆逐帧 detach，显存与单帧相同
  （StreamPETR 的做法）。为保证此时 GRU 仍能学习，时序模块把 GRU 新记忆经 `mem_out`
  回注到当前帧输出（否则 GRU 输出只在被 detach 的下一帧用到，永远没有梯度）。
  `--tbptt 2+` 让梯度穿过记忆，显存按帧数线性增长。
- **混合精度**：`--amp_dtype bf16`（默认）不需要 loss scaling；`fp16` 时启用 `GradScaler`。
- **学习率**：AdamW(lr=2e-4, wd=0.01；偏置/归一化/1 维嵌入参数不做 weight decay)，前 `--warmup_steps`(500)
  步线性 warmup（不超过总优化步数的 1/5，小数据集上自动缩短），之后按优化步余弦退火。
- 梯度裁剪 1.0；`--grad_accum` 梯度累积，epoch 末尾凑不满一组的 batch 按实际个数缩放后也会 step（不再丢弃）。
- `--seed` 固定样本顺序/隐式采样/初始化。
- **验证**：各损失项 + **mIoU**（不含 0:free，与 Occ3D 等基准一致）+ **几何 IoU**（占用/空二分类）+ 每类 IoU，
  都只在相机观测体素上统计。按 mIoU 保存 `best_model.pth`，每个 epoch 另存 `last_model.pth`。
- **断点恢复** `--resume last_model.pth`：恢复模型、优化器、学习率调度器、GradScaler 状态和历史最好 mIoU
  （2026-10-14 前只存了当前 epoch 的 mIoU，从 last 恢复后更差的模型可能覆盖 best）。
- 隐式头（`head_type=implicit`）训练时每帧采样 `implicit_train_points`(262,144) 个点：
  一半来自非空体素，一半均匀随机；验证时对整个网格分块查询。

## 4. 损失（`loss.py::E2ELoss`）

```
total = CE + 0.5·Lovász                       （语义）
      + depth_loss_weight · CE(深度 bin)       （64 个对数均匀 bin，1~60m，仅有效像素）
      + flow_loss_weight · SmoothL1(β=0.1)     （仅 flow_mask 内）
      + coarse_occ_loss_weight · BCE          （粗格是否含非空体素，max-pool 真值）
```

- Lovász 在超过 `lovasz_max_points`(100 万) 体素时子采样：保留全部非空体素，空体素随机补足。
  全量 512 万体素做逐类排序单帧要多占数 GB 显存。
- 输出层零附近初始化：初始语义 CE ≈ ln18 ≈ 2.89、深度 CE ≈ ln64 ≈ 4.16（重构前初始 CE ≈ 7）。
- `ignore_index=255` 的体素不参与语义损失与 mIoU。

## 5. 显存手段

| 手段 | 位置 |
|---|---|
| 稀疏细化（只对 25% 细体素做交叉注意力） | `occ_decoder.py`，`use_sparse_fine` |
| 细阶段 / 时序模块 / 稠密头梯度检查点 | `checkpoint_fine`、`use_checkpoint`、`checkpoint_head` |
| 可变形注意力逐相机循环 + 查询分块（32,768） | `deformable_attention.py` |
| Lovász 子采样 | `lovasz_max_points` |
| 流式训练（tbptt=1） | `train.py` |
| bf16 autocast | `train.py --amp` |

实测训练峰值（默认配置，单帧）约 8.1 GB。作为对照：重构过程中加稠密头梯度检查点和 Lovász
子采样之前，同一网络单帧约 10.6 GB，两帧共同反向（tbptt=2）超出了当时可用的约 15 GB 显存。

## 6. 指标

本文不写 mIoU / 延迟数字：重构后尚未在真实数据集上完整训练。完整训练后把实测结果（数据集、
GPU、epoch、batch、配置开关）写回这里。`verify_network.py` 全绿只说明结构、几何、梯度通路正确，
不说明网络学得好。
