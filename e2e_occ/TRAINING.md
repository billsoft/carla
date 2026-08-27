# e2e_occ 训练算法

> 对照 `train.py` / `loss.py` / `dataset.py` 实际代码核对过（2026-08-27）。网络结构见
> [`ARCHITECTURE.md`](./ARCHITECTURE.md)。

## 1. 环境与启动

```bash
conda activate deepsys

# 单帧训练（不用时序，config.use_temporal=False）
python e2e_occ/train.py --data_root <dataset_dir> --batch_size 1 --epochs 100 --lr 1e-4 --amp --grad_accum 4

# 时序训练（config.use_temporal=True，见 config.py，默认就是开的，temporal_frames=2）
python e2e_occ/train.py --data_root <dataset_dir> --batch_size 1 --epochs 100 --amp

# 从 checkpoint 恢复（config 也从 checkpoint 里读，避免结构不匹配）
python e2e_occ/train.py --resume checkpoints/best_model.pth --data_root <dataset_dir> --amp
```

`--num_workers` 默认 `0`——`dataset.py` 顶部注释明确写了这是为了避开 Windows 下
`DataLoader` 多进程的已知死锁问题，不要在 Windows 上不假思索地调大。

## 2. 数据加载与归一化（`dataset.py`）

- DNG 用 `rawpy.imread(...).raw_image_visible` 读取 Bayer RAW（未做去马赛克，网络自己在
  `raw_embed.py` 里学习 RGGB 分离），归一化除数是 `2^raw_bit_depth - 1`，`raw_bit_depth`
  从 `calibration/intrinsics.json` 的顶层 `raw_bit_depth` 字段读取（老数据集没有这个字段
  时退化为 12，历史上唯一实际用过的值）。**这个值必须和采集时 `main_collection.py
  --raw-bit-depth` 用的值一致**——DNG 的 EXIF `BitsPerSample` 标签靠不住（PIL 的
  TIFF writer 会按存储容器宽度把它覆盖成 16，不反映真实位深），所以不要指望从 DNG 文件本身
  反推位深，必须靠 calibration 里这份记录。
- 相机参数按优先级加载（`_get_frame_params`）：① `camera_params/{sample_id}.npz`
  （逐帧绝对外参，`dense_occupancy_collection` 格式）→ ② `ego_pose/{sample_id}.npy` +
  `calibration/extrinsics.json`（`occnetv3_data_generator` 当前格式，`Camera→World =
  Vehicle→World @ Camera→Vehicle`）→ ③ 纯静态标定退化（时序对齐会失效，仅调试用）。
- 时序样本：`config.use_temporal=True` 时 `sequence_length = config.temporal_frames`
  （默认 2），`__getitem__` 返回形状多一个 `T` 维：`images [T,N,1,H,W]`、
  `voxels [T,X,Y,Z]`、`extrinsics [T,N,4,4]`。`intrinsics` 不含 `T` 维（相机内参恒定）。

## 3. 训练循环（`train_epoch`，`train.py`）

### 3.1 单帧分支（`is_sequence=False`，`images.dim()==5`）

标准写法：`autocast → forward → loss → scaler.scale(loss/grad_accum).backward()`，每
`grad_accum` 步做一次 `unscale_ → clip_grad_norm_(1.0) → scaler.step → scaler.update →
zero_grad`。

### 3.2 时序分支 + TBPTT（`is_sequence=True`，`images.dim()==6`）

**TBPTT**（Truncated Backpropagation Through Time，`TBPTT_CHUNK_SIZE=2`，硬编码在
`train_epoch` 里，不是命令行参数）：把长度为 `T` 的序列切成每 2 帧一个 chunk，chunk 之间
`memory = memory.detach()` 截断计算图，chunk 内部才允许梯度贯穿多帧——这是显存和"时序
记忆需要跨帧梯度学习"之间的折中，序列越长这个折中越重要。

```python
for t_start in range(0, T, 2):
    if memory is not None:
        memory = memory.detach()          # 截断梯度历史

    chunk_loss, total_weight = 0, 0
    for t in range(t_start, min(t_start+2, T)):
        # ego_motion: C_{t-1}→C_t，语义是"上一帧体素坐标系→当前帧体素坐标系"
        ego_motion = inv(extrinsics[t][:,0]) @ extrinsics[t-1][:,0]   if t > 0 else None

        with autocast:
            outputs = model(images[t], intrinsics, extrinsics[t], memory=memory, ego_motion=ego_motion)
            loss_t = criterion(outputs['semantic'], voxels[t])['total']
            time_weight = 1.0 + t / max(1, T-1)     # 越靠后的帧权重越高
            chunk_loss += loss_t * time_weight
            total_weight += time_weight

        memory = outputs['memory']

    # 每个 chunk 结束才 backward 一次；除以 total_weight 消掉时间加权引入的量纲，
    # 再除以 grad_accum_steps 保持和单帧分支的梯度累积语义一致
    scaler.scale(chunk_loss / (total_weight * grad_accum_steps)).backward()
```

**时间加权**（`time_weight = 1.0 + t/(T-1)`，`T=2` 时 t=0 权重 1.0、t=1 权重 2.0）：
后期帧融合了更多历史信息，理应预测更准，用更高权重逼着模型把时序信息真正用起来，而不是
学会"忽略 memory 也能把每帧单独做对"这种退化解。

**为什么 chunk 内部 backward，而不是整个序列结束才 backward 一次**：`scaler`（AMP 的
`GradScaler`）状态在每次 `backward()` 后才明确；如果攒着整个序列的 loss 到最后一次性
backward，中间每个 chunk 该不该 detach 的边界会和 scaler 的 scale 因子搅在一起，
容易出细节错误。按 chunk 提交是这份实现里刻意保持的写法。

日志里有个专门的自检：每个 epoch 第 0 个 batch 的 `t=1`（也就是第一次算出
`ego_motion`）会把 4×4 矩阵原样打印出来——如果矩阵看起来完全不像一个刚性变换（比如平移
分量是几百几千米，或者旋转子块不是正交阵），基本可以断定是外参来源或 ego_motion 计算方式
出了问题，不用等一整个 epoch 训练完看 loss 才发现。

### 3.3 混合精度（AMP）

`torch.amp.autocast('cuda', enabled=args.amp)` + `torch.amp.GradScaler('cuda',
enabled=args.amp)`，`--amp` 不传就是纯 FP32（`GradScaler(enabled=False)` 时
`scaler.scale()`/`unscale_()`/`step()` 都是无操作直通，代码不用为开关 AMP 写两套分支）。

### 3.4 梯度裁剪与梯度累积

`nn.utils.clip_grad_norm_(model.parameters(), 1.0)`，固定阈值，在 `scaler.unscale_()`
之后、`scaler.step()` 之前做（AMP 下必须先 unscale 再裁剪，否则裁剪到的是放大过的梯度）。
梯度累积步数由 `--grad_accum` 控制，单帧/时序两个分支共用同一段"每
`grad_accum_steps` 步才真正 `step()`"的逻辑。

### 3.5 优化器与学习率

`AdamW(lr=1e-4 默认, weight_decay=0.01 默认)` + `CosineAnnealingLR(T_max=epochs)`，
每个 epoch 结束 `scheduler.step()` 一次（不是每个 batch）。

## 4. 损失函数（`loss.py`，`OccupancyLoss`）

```python
total = cross_entropy(pred, target) + lovasz_weight(默认0.5) * lovasz_softmax(pred, target)
```

- 两项都会先用 `target != ignore_index`（默认 `255`）过滤掉无效体素再算，valid 数量为 0
  时直接返回全 0 loss（不会报错中断训练，但要留意这种情况在日志里是否频繁出现——频繁出现
  说明某个 batch 的体素真值整体有问题）。
- Lovász-Softmax 是逐类别算的：对每个出现过的类别（`fg.sum()>0` 才参与），把预测误差
  降序排序后配合 Lovász 梯度（`_lovasz_grad`，标准的 Jaccard/IoU 次梯度公式）加权求和，
  再对所有出现过的类别取平均。它直接优化的是 IoU 而不是逐像素分类准确率，这也是为什么
  同时保留交叉熵——纯 Lovász 早期训练信号弱（几乎所有类别一开始都预测错，排序后的梯度
  结构建立较慢），交叉熵负责把训练"启动"起来。

`num_classes=18` 从 `config.num_classes` 传入，和 `voxel_head.py` 输出通道数一致，改
类别数时两处要一起改（权威定义见
`occnetv3_data_generator/config/occupancy_config.py`）。

## 5. 显存优化手段一览

| 手段 | 位置 | 作用 |
|---|---|---|
| 逐相机 + 逐头串行采样 | `deformable_attention.py` | 避免 8 相机并行造成的显存峰值（详见 ARCHITECTURE.md 3.6） |
| Fine 阶段禁用 Self-Attention | `config.use_fine_self_attention=False` | 102,400 个查询做 `Q×Q` 自注意力会 OOM |
| Fine 阶段梯度检查点 | `occ_decoder.py` `checkpoint_fine=True` | 用重算换显存，Coarse 阶段（仅 5,000 查询）不需要 |
| TemporalFusion 梯度检查点 | `temporal_fusion.py` `use_checkpoint=True` | 同上 |
| TBPTT（chunk=2） | `train.py` | 截断长序列的反传路径，见 3.2 |
| 混合精度 AMP | `train.py` `--amp` | FP16 前向+反向，FP32 权重更新（`GradScaler`） |
| 梯度累积 | `train.py` `--grad_accum` | 等效放大 batch size 而不增加单步显存 |
| Depthwise Conv3D | `occ_decoder.py` fine_spatial_conv | 参数量 `256×3³≈6.9K` vs 标准卷积 `256²×3³≈1.8M` |

没有一个是"可有可无"的可选项——这几个手段任意去掉一个，在这台机器的显存（RTX 4090,
~24GB）上大概率会在 Fine 阶段（102,400 查询）OOM，改代码前先想清楚要动的是哪一层。

## 6. 关于性能/精度指标

本文档不列具体的 mIoU / 推理延迟 / 显存占用数字。旧版本文档里有过一批具体数字
（如"mIoU 38.7%"、"60ms/帧"），那些是等距投影迁移**之前**、相机模型和数据管线都不同
的一次训练结果，迁移之后没有重新验证过，continuing 引用会误导判断。等
`dataset_10k` 正式采集完成、跑过一轮完整训练后，把新的实测数字（数据来源、GPU 型号、
epoch 数、batch size 一并注明）写回本节，不要脱离具体训练配置单独写一个数字。

`e2e_occ/verify_network.py` 提供的是**结构正确性**检查（形状、NaN/Inf、显存能否装得下、
等距投影几何是否自洽），不是性能基准，跑一遍绿灯只说明"网络能跑"，不说明"网络学得好"。
