# e2e_occ — 端到端 3D 占用网络

参考特斯拉 FSD 架构，输入 8 路等距投影（equidistant fisheye）Bayer RAW 图像，输出
`(400, 400, 32)`、18 类语义的 3D 占用网格。粗细两阶段 Deformable Cross-Attention 解码 +
GRU 时序融合，参数量 10.49M（2026-08-27 实测，见 [`ARCHITECTURE.md`](./ARCHITECTURE.md)）。

这是当前维护中的主力网络。`occ_network/`、`occ_transformer/` 是过时的早期实验，
`dense_occupancy_collection/` 已被 `occnetv3_data_generator/` 取代，均已弃用。

## 文档

- **[`ARCHITECTURE.md`](./ARCHITECTURE.md)** — 网络架构，逐模块对照实际代码
- **[`TRAINING.md`](./TRAINING.md)** — 训练流程、损失函数、显存优化
- **[`../occnetv3_data_generator/README.md`](../occnetv3_data_generator/README.md)** — 数据采集
- **[`../CLAUDE.md`](../CLAUDE.md)** — 仓库整体结构、CARLA 引擎构建

## 快速开始

```bash
conda activate deepsys

# 训练
python e2e_occ/train.py --data_root <dataset_dir> --batch_size 1 --epochs 100 --amp --grad_accum 4

# 推理
python e2e_occ/inference.py --checkpoint checkpoints/best_model.pth --data_root <dataset_dir> --output <out_dir>

# 结构自检（形状/NaN/显存/等距投影几何一致性，不是性能基准）
python e2e_occ/verify_network.py

# 可视化
python dataset_viewer_v2/server.py --dataset <out_dir>   # http://localhost:8085/
```

环境要求：Python 3.10+，PyTorch 2.0+（用到 `F.scaled_dot_product_attention` 自动选
FlashAttention kernel），CUDA 11.8+。Windows 下 `train.py --num_workers` 保持默认 `0`，
调大会撞上 `DataLoader` 多进程死锁的已知问题。

## 输入/输出

| | 形状 | 说明 |
|---|---|---|
| 输入 | `[B, 8, 1, 960, 1280]` | 8 相机 Bayer RAW，归一化到 `[0,1]`（位深见 `calibration/intrinsics.json` 的 `raw_bit_depth`） |
| 输出 | `[B, 18, 400, 400, 32]` | 语义 logits，X/Y ±40m、Z −1~5.4m，0.2m/体素 |

18 类语义定义（权威来源）：`occnetv3_data_generator/config/occupancy_config.py`。
