# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 重要提示 / Important Notes

**语言 / Language:** 请始终使用中文与用户沟通。Always communicate with users in Chinese.

**代码修复原则:** 有问题的文件在原文件上修复，不要创建 `_fix`、`_fixed`、`_new` 等后缀的新文件。

---

## ⚠️ 关键教训（踩过的坑）

### 1. 相机坐标系方向

在 CARLA 中，Y 轴正方向是"**右侧**"，负方向是"**左侧**"（驾驶员视角）：
- ✅ `left_pillar: position(0.0, -1.1, 1.7)` — Y 负值才是真正左侧
- ✅ `right_pillar: position(0.0, 1.1, 1.7)` — Y 正值是右侧
- 后视相机向车尾外延 (X=-2.7) 避免玻璃遮挡；侧后视相机 Y 绝对值要大避免车架遮挡

### 2. 地面层双重渲染问题（已修复）

**根因：** Map API 和 Static Mesh 同时生成地面 → "双层地面"、"浮空灰层"。

**修复方案**（在 `occnetv3_data_generator/processing/ground_truth_voxel_generator.py`）：
1. `_fill_static_environment()` 中跳过 `Roads, Sidewalks, Terrain, Ground, RoadLines` 的 EnvironmentObjects
2. 地下填充统一使用 Terrain(14)，不复制地表材质
3. Actor BBox 不能覆盖已存在的地面体素（防坑洞）

诊断工具：`occnetv3_data_generator/diagnose_ground_layer.py`

### 3. DNG 格式加载

- ❌ OpenCV 不支持 CFA PhotometricInterpretation=32803 格式的 DNG
- ✅ 必须安装 `rawpy`（或 `Pillow + piexif`）；降级方案自动保存 .npy

### 4. 导入方式

`e2e_occ/image_encoder.py` 和 `e2e_occ/occ_decoder.py` 中的模块导入应使用**相对导入**：
```python
from .position_encoding import ...   # ✅ 正确
from position_encoding import ...    # ❌ 作为包导入时报 ModuleNotFoundError
```

---

## Python 环境

| 环境 | 用途 | 完整路径 | 包含 |
|------|------|---------|------|
| **carla** | CARLA 数据采集 | `/c/ProgramData/anaconda3/envs/carla/python.exe` | PyTorch 2.7.1+cu118, CARLA 0.10.0, rawpy |
| **deepsys** | 占用网络训练 | `/c/ProgramData/anaconda3/envs/deepsys/python.exe` | PyTorch 2.6.0+cu124, 完整深度学习工具链 |

⚠️ 环境名 `carla` 的实际目录路径可能是 `car` 或其他名称，必须使用完整路径。

**网络代理:** `pip install --proxy http://192.168.100.182:7890 package_name`

---

## ⚠️ Windows 命令执行规范

**Claude Code 的 Bash tool 运行在 Git Bash 中，不继承 conda 激活状态。**

```bash
# ✅ 正确：使用完整 Unix 格式路径
/c/ProgramData/anaconda3/envs/deepsys/python.exe e2e_occ/train.py --amp

# ✅ 正确：通过 cmd.exe
cmd.exe /c "cd /d d:\code\carla && C:\ProgramData\anaconda3\envs\deepsys\python.exe script.py"

# ❌ 错误
python script.py          # bash 找不到
conda run -n carla ...    # conda 命令不存在
C:\path\python.exe        # 反斜杠被 bash 误解
```

后台运行长时间脚本：Bash tool 使用 `run_in_background=true`。

---

## CARLA 模拟器（核心框架）

### 项目结构

```
LibCarla/          → 核心 C++ 库（双编译：carla-server + carla-client）
Unreal/            → UE5.5 集成（Plugins/Carla/ 为主插件）
PythonAPI/         → Python 绑定（Boost.Python，carla/src/）
Ros2Native/        → ROS2 支持（仅 Linux，FastDDS）
```

### 构建命令

```bash
# 初始配置
cmake -G Ninja -S . -B Build --toolchain=$PWD/CMake/Toolchain.cmake -DCMAKE_BUILD_TYPE=Release
# Linux 加 ROS2: -DENABLE_ROS2=ON

cmake --build Build                                      # 构建
cmake --build Build --target carla-python-api-install   # 安装 Python API
cmake --build Build --target launch                      # 启动 UE 编辑器
cmake --build Build --target package                     # 打包发布版
```

初始安装（Windows 管理员权限）：`CarlaSetup.bat`（需 GitHub 凭据下载 UE5.5 fork）

**重要：** CARLA 需要 CARLA fork 的 UE5.5，不是官方 Epic UE5。构建需要 225GB+，3 小时+。

### 通信架构

- **RPC (rpclib)** — Port 2000：同步命令（spawn/control）→ `FCarlaServer` → `UCarlaEpisode`
- **Streaming (TCP)** — Port 2001：异步传感器数据流（低延迟）
- **Multi-GPU Router** — Port 2002（可选）

### 添加新传感器

1. UE 类 → [Unreal/CarlaUnreal/Plugins/Carla/Source/Carla/Sensor/](Unreal/CarlaUnreal/Plugins/Carla/Source/Carla/Sensor/)
2. 数据类型 → [LibCarla/source/carla/sensor/data/](LibCarla/source/carla/sensor/data/)
3. 序列化 → [LibCarla/source/carla/sensor/s11n/](LibCarla/source/carla/sensor/s11n/)
4. 注册 Actor 工厂；导出到 [PythonAPI/carla/src/](PythonAPI/carla/src/)

### 代码规范

**C++:** 空格非 Tab，80 列注释，`clang++ -Wall -Wextra -std=C++14` 无警告，使用 `carla::throw_exception`，服务端 try-catch 包 `#ifndef LIBCARLA_NO_EXCEPTIONS`。

**Python:** PEP8，Python 3.7+，120 列，通过 Pylint。

---

## Bayer RAW 底层支持（自编译 CARLA 增强）

本仓库对 CARLA UE5.5 进行了底层修改以支持 **原生 Bayer RAW 采集**，标准 CARLA 发行版不含此功能：

| 层次 | 修改内容 |
|------|---------|
| **UE5** `SceneCaptureCamera.h/cpp` | 新增 `RawType="bayer_rggb"` 属性；`FLinearColor[]` → RGGB uint16 采样算法 |
| **LibCarla** `sensor/data/Image.h` | 新增 `EPixelFormat::BAYER_RGGB_U16 = 3` 枚举 |
| **LibCarla** `s11n/ImageSerializer.h` | 新增 `pixel_format` 参数重载 |
| **Python API** `SensorData.cpp` | `image.raw_data` 零拷贝 `PyMemoryView` 访问 |

数据流：`UE5 Render → FLinearColor → Bayer采样 → uint16 → TCP → Python frombuffer → DNG保存`

---

## 数据采集系统（occnetv3_data_generator）

⚠️ **依赖自编译 CARLA UE5.5**（含上述底层修改）。`dense_occupancy_collection` 已搁置弃用。

### 核心功能

- 8 相机 Bayer RGGB 12-bit DNG（Tesla 环视布局，960×1280）
- 8 相机对应深度图（float32，米制）
- 256 线语义激光雷达点云
- 3D 占用网格真值（400×400×32，18 类语义）
- 场景流（Scene Flow）+ 可见性遮挡过滤

### Tesla 8 相机布局

| 相机 | FOV | 位置 (x, y, z) | 朝向 (pitch, yaw, roll) |
|------|-----|----------------|------------------------|
| front_main | 50° | (1.0, 0.0, 1.6) | (0, 0, 0) |
| front_wide | 120° | (1.0, 0.0, 1.6) | (0, 0, 0) |
| front_narrow | 35° | (1.0, 0.0, 1.6) | (0, 0, 0) |
| left_pillar | 80° | (0.0, **-1.1**, 1.7) | (0, -55, 0) |
| right_pillar | 80° | (0.0, **1.1**, 1.7) | (0, 55, 0) |
| left_repeater | 100° | (1.0, **-1.0**, 1.0) | (0, -130, 0) |
| right_repeater | 100° | (1.0, **1.0**, 1.0) | (0, 130, 0) |
| rear | 120° | (**-2.7**, 0.0, 1.2) | (-8, 180, 0) |

### 关键文件

- [occnetv3_data_generator/config/camera_config.py](occnetv3_data_generator/config/camera_config.py) — 相机参数
- [occnetv3_data_generator/config/occupancy_config.py](occnetv3_data_generator/config/occupancy_config.py) — 体素空间 + LiDAR 参数
- [occnetv3_data_generator/config/actor_occupancy_mapping.py](occnetv3_data_generator/config/actor_occupancy_mapping.py) — CARLA type_id → 18 类映射
- [occnetv3_data_generator/processing/ground_truth_voxel_generator.py](occnetv3_data_generator/processing/ground_truth_voxel_generator.py) — 体素生成核心
- [occnetv3_data_generator/processing/visibility_filter_simple.py](occnetv3_data_generator/processing/visibility_filter_simple.py) — LiDAR ID 聚类可见性过滤
- [occnetv3_data_generator/data_utils/data_saver.py](occnetv3_data_generator/data_utils/data_saver.py) — 异步 DNG+NPY 保存（4 线程）
- [occnetv3_data_generator/main_collection.py](occnetv3_data_generator/main_collection.py) — ⭐ 主采集脚本

### 运行

```bash
# 1. 启动 CARLA 服务器
cmake --build Build --target launch
# 或直接运行: CarlaUE5.exe -quality-level=Low -RenderOffScreen

# 2. 数据采集
/c/ProgramData/anaconda3/envs/carla/python.exe occnetv3_data_generator/main_collection.py \
    --frames 100 --output d:/code/carla/dataset_10k_bak \
    --town Town10HD --num-vehicles 30 --num-walkers 10

# 3. 可视化采集结果
/c/ProgramData/anaconda3/envs/carla/python.exe occnetv3_data_generator/visualize_dataset.py \
    --dataset d:/code/carla/dataset_10k_bak --sample 0
```

参数：`--frames`(帧数)，`--output`(输出目录)，`--town`(地图)，`--num-vehicles`，`--num-walkers`，`--clear-output`

### 体素生成流程

```
初始化空网格 → 填充静态环境（Map API 地面 + 静态物体 BBox 光栅化）
  → 填充动态 Actor（OBB 光栅化 + type_id 语义映射）
  → 生成场景流（速度差 × dt）
  → 可见性过滤（LiDAR 点云体素化 → ID 聚类保留 → 地面强制保留）
```

**性能：** 首帧 ~3-5s（Map API Cache 预热），后续帧 ~0.5-1s。瓶颈已通过 Cache + NumPy 向量化 + 异步 IO 优化。

---

## 数据集格式（dataset_10k）

```
dataset_10k/
├── calibration/
│   ├── intrinsics.json      # {cam_0: {fx, fy, cx, cy, fov}, ...}（静态，所有帧相同）
│   └── extrinsics.json      # {cam_0: {rotation_matrix, translation}, ...} Camera→Vehicle
├── images/
│   └── scene_0000_frame_0000/
│       ├── cam_0.dng ~ cam_7.dng   # Bayer RGGB 12-bit DNG (960×1280)
├── depth/
│   └── scene_0000_frame_0000/
│       ├── cam_0.npy ~ cam_7.npy   # (960, 1280) float32，单位：米
├── occupancy/
│   └── scene_0000_frame_0000.npy   # (400, 400, 32) uint8，语义类别 [0-17]
├── flow/
│   └── scene_0000_frame_0000.npy   # (3, 400, 400, 32) float16，(dx,dy,dz) 米/帧
├── flow_mask/
│   └── scene_0000_frame_0000.npy   # (400, 400, 32) uint8，动态区域掩码
├── ego_pose/
│   └── scene_0000_frame_0000.npy   # (4, 4) float32，Vehicle→World
├── ego_motion/
│   └── scene_0000_frame_0000.npy   # (4, 4) float32，相邻帧位姿差
├── train.txt / val.txt / test.txt   # 样本列表（格式：scene_XXXX_frame_XXXX）
```

**`e2e_occ/dataset.py` 相机参数加载优先级：**
1. `camera_params/{id}.npz`（逐帧绝对外参，dense_occupancy_collection 格式，已弃用）
2. `ego_pose/{id}.npy` + `calibration/`（当前格式，occnetv3_data_generator 生成）
3. `calibration/` 静态标定退化（时序对齐失效，仅调试用）

**外参约定：** `extrinsics = Camera→World`（含车辆绝对位姿）
**ego_motion 计算：** `inv(pose_t) @ pose_{t-1}`（上一帧坐标系 → 当前帧坐标系）

---

## 18 类语义标签（nuScenes 标准）

```
0:  free               ← 空气/无物体，⚠️ 损失函数权重必须 ≥ 1.0
1:  barrier            ← 护栏/路障
2:  bicycle            ← 自行车
3:  bus                ← 公交车
4:  car                ← 小汽车
5:  construction_vehicle
6:  motorcycle         ← 摩托车
7:  pedestrian         ← 行人
8:  traffic_cone       ← 交通锥
9:  trailer            ← 拖车
10: truck              ← 卡车
11: driveable_surface  ← 可行驶路面（强制保留）
12: other_flat         ← 其他平面
13: sidewalk           ← 人行道（强制保留）
14: terrain            ← 地形（草地/泥土），地下统一填充此类
15: manmade            ← 人造建筑
16: vegetation         ← 植被
17: general_object     ← 通用障碍物
```

**空间范围：** X=±40m，Y=±40m，Z=-1~5.4m；体素分辨率 0.2m；网格 400×400×32

---

## E2E 端到端占用网络 (e2e_occ) ⭐ 唯一维护中的网络

其他网络（`occ_network_nano`、`occ_network`、`occ_transformer`）均已过时，后续可能删除。

### 网络概述

参考特斯拉 FSD 架构。**约 8.9M 参数**，输入 8 路 Bayer RAW → 输出 (400,400,32) 18 类语义体素。

```
输入: [B,8,1,960,1280] RAW

① MultiCameraPatchEmbed (raw_embed.py)
   可学习 RGGB 2×2 Conv（代替传统 demosaic）+ 4层 Stem CNN（各÷2）
   → [B, 8, 256, 60, 80]（16× 下采样）

② ImageEncoder (image_encoder.py)
   射线方向编码（等距投影鱼眼模型，正弦频率编码 × 10频率，MLP→256）
   + 2层 WindowAttention（window=7×7，49 tokens，计算量仅原来 1%）
   → [B, 8, 256, 60, 80]（形状不变，特征含几何先验）

③ OccupancyDecoder (occ_decoder.py)
  ③-a 粗查询（Coarse）: 5000 可学习查询 (25×25×8) + 3D 正弦位置编码
       2层可变形交叉注意力（Deformable Cross-Attention）+ 自注意力
       → [B, 256, 25, 25, 8]

  ③-b 时序融合 (temporal_fusion.py)
       Ego-Motion 对齐（3D grid_sample warp，将历史记忆转换到当前帧坐标系）
       + 时序注意力（Q=当前，K/V=对齐记忆）+ GRU 门控（自适应融合新旧信息）
       → 融合特征 [B,5000,256] + 新记忆 [B,5000,256]

  ③-c 精细查询（Fine）: 三线性插值 25³→80×80×16，MLP 过渡
       102,400 查询点，2层可变形交叉注意力（无自注意力，强制梯度检查点）
       + 深度可分离 Conv3d（3×3×3）保证空间一致性
       → [B, 80, 80, 16, 256]

④ VoxelHead (voxel_head.py)
   先降维再分类再上采样（省 14× 显存）：
   Conv3d 256→128→64（逐步降维，平滑保留信息）+ 1×1×1 分类 64→18
   两步上采样：80→200（+精化卷积）→ 400（+精化卷积）
   → [B, 18, 400, 400, 32]
```

### 可变形交叉注意力核心逻辑

1. 3D 查询点坐标 → 投影到各相机 2D 像素（几何投影）
2. 线性层预测采样偏移 Δuv（每点 4 个采样点）
3. `grid_sample` 在特征图 (u+Δu, v+Δv) 处双线性插值
4. Softmax 权重加权求和 → 256 维输出

### 损失函数与训练

**损失：** `CE + 0.5 × Lovász-Softmax`（CE 逐点，Lovász 直接优化 IoU）

**训练技术：**
- **TBPTT：** chunk_size=2，每隔 2 帧截断梯度，`memory.detach()` 传递
- **时间步加权：** `weight = 1 + t/(T-1)`（后帧权重更高，鼓励利用时序）
- **AMP：** GradScaler 防梯度下溢
- **梯度裁剪：** `max_norm=1.0`
- **调度：** CosineAnnealing，保存 val_loss 最优

### 关键超参数

| 参数 | 值 | 含义 |
|------|----|------|
| `embed_dim` | 256 | 全局特征维度 |
| `num_heads` | 8 | 注意力头数（head_dim=32）|
| `image_size` | (960, 1280) | 输入分辨率 |
| `coarse_size` | (25, 25, 8) | 粗查询网格 |
| `fine_size` | (80, 80, 16) | 精细查询网格 |
| `voxel_size` | (400, 400, 32) | 输出体素 |
| `voxel_range` | (-40,-40,-1, 40,40,5.4) | 空间范围 |
| `encoder_layers` | 2 | 图像编码器层数 |
| `decoder_layers` | 2 | 粗/精解码器各 2 层 |
| `temporal_frames` | 2 | 时序帧数 |
| `num_sample_points` | 4 | 可变形注意力采样点数 |

**显存估算（BS=1，AMP）：** ~10-15 GB（目标 18-20 GB，Fine 阶段梯度检查点必须开启）

### 关键文件

| 文件 | 功能 |
|------|------|
| [e2e_occ/config.py](e2e_occ/config.py) | `E2EOccConfig` 数据类，所有超参数 |
| [e2e_occ/e2e_occ_net.py](e2e_occ/e2e_occ_net.py) | 主网络入口 `build_model()` |
| [e2e_occ/raw_embed.py](e2e_occ/raw_embed.py) | `MultiCameraPatchEmbed`：RGGB 解马赛克 + CNN Stem |
| [e2e_occ/image_encoder.py](e2e_occ/image_encoder.py) | `ImageEncoder`：射线编码 + 窗口注意力 |
| [e2e_occ/position_encoding.py](e2e_occ/position_encoding.py) | `RayDirectionEncoding`（等距投影）+ 3D 正弦编码 |
| [e2e_occ/occ_decoder.py](e2e_occ/occ_decoder.py) | `OccupancyDecoder`：粗细两阶段解码 |
| [e2e_occ/deformable_attention.py](e2e_occ/deformable_attention.py) | 可变形交叉注意力（3D→2D 投影 + grid_sample）|
| [e2e_occ/temporal_fusion.py](e2e_occ/temporal_fusion.py) | `TemporalFusion`：ego_motion warp + 时序注意力 + GRU |
| [e2e_occ/voxel_head.py](e2e_occ/voxel_head.py) | `VoxelHead`：降维(256→128→64) + 分类(→18) + 两步三线性上采样 |
| [e2e_occ/loss.py](e2e_occ/loss.py) | CE + Lovász-Softmax 组合损失 |
| [e2e_occ/dataset.py](e2e_occ/dataset.py) | `OccupancyDataset`：序列加载 [B,T,N,C,H,W]，多源相机参数 |
| [e2e_occ/train.py](e2e_occ/train.py) | ⭐ 训练脚本（TBPTT + AMP + CosineAnnealing）|
| [e2e_occ/inference.py](e2e_occ/inference.py) | 推理脚本 |
| [e2e_occ/verify_network.py](e2e_occ/verify_network.py) | 网络结构验证（无需数据集）|

### 运行命令

```bash
# 训练（在项目根目录 d:\code\carla 下执行）
/c/ProgramData/anaconda3/envs/deepsys/python.exe e2e_occ/train.py \
    --data_root d:/code/carla/dataset_10k_bak \
    --batch_size 1 --epochs 50 --amp

# 可选参数: --output_dir d:/code/carla/checkpoints --lr 1e-4 --resume path/to/ckpt.pth
# --grad_accum 2  （梯度累积，等效扩大 batch size）
# --num_workers 0 （Windows 下默认 0，避免死锁）

# 推理
/c/ProgramData/anaconda3/envs/deepsys/python.exe e2e_occ/inference.py \
    --checkpoint d:/code/carla/checkpoints/best.pth \
    --data_root d:/code/carla/dataset_10k_bak

# 验证网络结构（无需数据集）
/c/ProgramData/anaconda3/envs/deepsys/python.exe e2e_occ/verify_network.py
```

---

## 可视化工具（occupancy_viewer）

基于 **Three.js** 的交互式 3D 体素可视化，支持鼠标旋转/缩放、多视角切换、帧浏览、类别统计。

```bash
/c/ProgramData/anaconda3/envs/deepsys/python.exe occupancy_viewer/run_viewer.py
# 浏览器访问: http://localhost:8085/
```

修改 [occupancy_viewer/run_viewer.py](occupancy_viewer/run_viewer.py) 中的 `DATA_DIR` 切换数据源。

**NPZ 格式要求：** `occupancy`(uint8), `mask`(bool), `x_range`, `y_range`, `z_range`, `resolution`(float), `grid_size`

---

## 完整工作流

```bash
# 1. 启动 CARLA 服务器
cmake --build Build --target launch

# 2. 数据采集（carla 环境）
/c/ProgramData/anaconda3/envs/carla/python.exe occnetv3_data_generator/main_collection.py \
    --frames 100 --output d:/code/carla/dataset_10k_bak \
    --town Town10HD --num-vehicles 30 --num-walkers 10

# 3. 训练（deepsys 环境）
/c/ProgramData/anaconda3/envs/deepsys/python.exe e2e_occ/train.py \
    --data_root d:/code/carla/dataset_10k_bak \
    --batch_size 1 --epochs 50 --amp

# 4. 推理
/c/ProgramData/anaconda3/envs/deepsys/python.exe e2e_occ/inference.py \
    --checkpoint d:/code/carla/checkpoints/best.pth \
    --data_root d:/code/carla/dataset_10k_bak

# 5. 可视化
/c/ProgramData/anaconda3/envs/deepsys/python.exe occupancy_viewer/run_viewer.py
```

---

## 常见问题排查

### Q1: DNG 无法加载（`PhotometricInterpretation=32803`）
```bash
pip install --proxy http://192.168.100.182:7890 rawpy
# 或
pip install --proxy http://192.168.100.182:7890 Pillow piexif
```

### Q2: 推理结果全实心/单色
Class 0 (free) 损失权重设为 0 → 网络不学习预测空白。各网络 loss.py 中 Class 0 权重必须 ≥ 1.0。

### Q3: 显存不足
1. 减小 `--batch-size`（推荐 1）
2. 启用 `--amp`
3. 确认 Fine 阶段梯度检查点开启（e2e_occ 已强制开启）

### Q4: 训练梯度爆炸
添加 `--grad-clip 1.0` + 降低 `--lr 5e-5`

### Q5: 时序对齐失效（ego_motion 全零）
dataset.py 退化到静态标定。检查数据集是否有 `ego_pose/` 目录。

### Q6: 采集速度慢（>5s/帧）
Cache 首帧预热正常（~3-5s），后续应 <1s。NPC 数量建议 ≤50，确认开启混合物理模式。

### Q7: 地面出现灰色浮空层
Map API + Static Mesh 双重渲染问题。检查 `ground_truth_voxel_generator.py` 中 `SKIP_STATIC_TYPES` 和地下 Terrain(14) 填充是否正确。使用 `diagnose_ground_layer.py` 诊断。

### Q8: CARLA 服务器无法启动
确认环境变量 `CARLA_UNREAL_ENGINE_PATH` 已设置（初始安装后自动写入 `.bashrc`/系统环境变量）。

---

## Key Documentation

- CARLA docs: https://carla-ue5.readthedocs.io
- Python API: https://carla-ue5.readthedocs.io/en/latest/python_api/
- UE5 GitHub: https://www.unrealengine.com/en-US/ue-on-github（需关联 GitHub 账号）
- occnetv3_data_generator 详细技术文档：[occnetv3_data_generator/README.md](occnetv3_data_generator/README.md)
- e2e_occ 网络架构详解：[e2e_occ/网络说明.md](e2e_occ/网络说明.md)
