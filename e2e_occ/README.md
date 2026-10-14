# e2e_occ — 端到端 3D 占用网络

参考特斯拉 FSD 架构，输入 8 路等距投影（equidistant fisheye）Bayer RAW 图像，输出
`(400, 400, 32)`、18 类语义的 3D 占用网格（+ 3 维占用流）。多尺度可变形交叉注意力、
粗 → 时序 → 稀疏细两阶段解码，默认配置参数量 13.45M（2026-10 重构后实测，见
[`ARCHITECTURE.md`](./ARCHITECTURE.md)）。重构前的 checkpoint 不兼容。

这是当前维护中的主力网络。`occ_network/`、`occ_transformer/` 是过时的早期实验，
`dense_occupancy_collection/` 已被 `occnetv3_data_generator/` 取代，均已弃用。

## 文档

- **[`ARCHITECTURE.md`](./ARCHITECTURE.md)** — 网络架构、坐标约定、配置开关、实测资源
- **[`MODULES_EXPLAINED.md`](./MODULES_EXPLAINED.md)** — 逐模块原理讲解（作用/输入输出/算法/为什么）
- **[`shape_report.md`](./shape_report.md)** — 每层输入输出形状（`verify_network.py` 自动生成）
- **[`TRAINING.md`](./TRAINING.md)** — 训练流程、损失函数、显存优化
- **[`../occnetv3_data_generator/README.md`](../occnetv3_data_generator/README.md)** — 数据采集
- **[`../CLAUDE.md`](../CLAUDE.md)** — 仓库整体结构、CARLA 引擎构建

## 快速开始

```bash
conda activate deepsys

# 训练
python e2e_occ/train.py --data_root <dataset_dir> --batch_size 1 --epochs 100 --amp --grad_accum 4

# 推理（流式携带时序记忆，换场景重置；默认 --split val --num_samples 100，可加 --save_flow）
python e2e_occ/inference.py --checkpoint checkpoints/best_model.pth --data_root <dataset_dir> --output <out_dir>
# 推理速度基准（随机输入 + 真实标定）
python e2e_occ/inference.py --checkpoint checkpoints/best_model.pth --benchmark

# 结构自检（逐层形状、真实标定下的几何回归、时序方向、梯度通路，不是性能基准；
# 会重写 shape_report.md，--quick 只跑默认配置、报告也只剩默认配置一节）
python e2e_occ/verify_network.py
# 快速单帧前向/反向冒烟、逐阶段显存/耗时剖析（均支持随机输入，不需要数据集）
python e2e_occ/test.py
python e2e_occ/profile_network.py --set use_sparse_fine=False

# 可选：MAE 自监督预训练图像骨干
python e2e_occ/pretrain_mae.py --data_root <dataset_dir> --epochs 20 --amp

# 可视化
python dataset_viewer_v2/server.py --dataset <out_dir>   # http://localhost:8085/
```

环境要求：Python 3.10+，**PyTorch 2.4+**（默认 `norm_type='rmsnorm'` 用到 2.4 才加入的
`nn.RMSNorm`；`F.scaled_dot_product_attention` 自动选 FlashAttention kernel；`deepsys` 环境实测
2.6.0+cu124），CUDA 11.8+。Windows 下 `train.py --num_workers` 保持默认 `0`，
调大会撞上 `DataLoader` 多进程死锁的已知问题。

## 输入/输出

| | 形状 | 说明 |
|---|---|---|
| 输入 | `[B, 8, 1, 960, 1280]` | 8 相机 Bayer RAW，归一化到 `[0,1]`（位深见 `calibration/intrinsics.json` 的 `raw_bit_depth`） |
| 输出 | `[B, 18, 400, 400, 32]` | 语义 logits，X/Y ±40m、Z −1~5.4m，0.2m/体素 |
| 输出 | `[B, 3, 400, 400, 32]` | 占用流（物体自身位移，车辆系，米/帧） |

18 类语义定义（权威来源）：`occnetv3_data_generator/config/occupancy_config.py`。
