# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 重要提示 / Important Notes

**语言 / Language:** 请始终使用中文与用户沟通。Always communicate with users in Chinese.

**代码修复原则 / Code Fix Principles:**
- ⚠️ **重要：有问题的文件我们在问题文件上修复，不要创建新的 fix、fixed 等后缀的文件**
- Fix bugs directly in the original file, do NOT create new files with suffixes like `_fix`, `_fixed`, `_new`, etc.
- 直接修改源文件，保持代码库整洁

**⚠️ 关键教训：相机方向和地面渲染问题 / Critical Lessons Learned:**

**1. 相机坐标系方向 (Camera Coordinate System):**
- ❌ **错误**: 在CARLA中，Y轴正方向是"右侧"，负方向是"左侧" (从驾驶员视角)
- ⚠️ 之前的错误配置: `left_pillar: [0.5, 0.9, 1.3]` (实际在右侧)
- ✅ 正确配置: `left_pillar: [0.5, -0.9, 1.3]` (真正的左侧)
- 📌 口诀: **想象你坐在驾驶位，左手侧 = Y负值，右手侧 = Y正值**
- 📌 后视相机需要向后延伸 (X轴负值增大) 避免车内玻璃遮挡
- 📌 侧后视相机需要向外延伸 (Y轴绝对值增大) 避免车架遮挡

**2. 地面层双重渲染问题 (Double Ground Layer Issue):**
- ❌ **根本原因**: Map API 和 Static Mesh 同时生成地面，导致"双层地面"和"浮空灰层"
- 问题现象:
  - 地面上随机出现灰色/白色条纹层
  - 路沿石 (Barrier) 覆盖在泥土 (Terrain) 之上
  - 地面下方出现灰色墙体 (向下填充复制了地表颜色)
- ✅ **修复方案** (已在 ground_truth_voxel_generator.py 中实现):
  1. **排除地面类型的 Static Mesh 光栅化**: 在 `_fill_static_environment()` 中跳过 `Roads, Sidewalks, Terrain, Ground, RoadLines` 的 EnvironmentObjects
  2. **地下填充统一使用 Terrain (14)**: 不再复制地表材质，统一填充棕色泥土
  3. **地面保护机制**: Actor BBox 不能覆盖已存在的地面体素 (防止车辆 BBox 插入地下形成坑洞)
- 📌 诊断工具: `occnetv3_data_generator/diagnose_ground_layer.py` 可分析体素分布

**3. DNG 格式支持 (Bayer RAW Image Format):**
- ⚠️ 需要安装: `pip install Pillow piexif` (在 carla 环境中)
- ⚠️ OpenCV 不支持 CFA PhotometricInterpretation=32803 格式的 DNG
- ✅ 备用方案: 使用 `rawpy` 库 (`pip install rawpy`)
- ✅ 降级方案: 如果两者都不可用，自动保存为 .npy 格式

**Python 环境管理 / Python Environment:**
- 本项目使用 **conda** 管理 Python 环境
- **主环境名称：carla** (用于 CARLA 数据采集和 occ_network 训练)
  - ⚠️ **注意**: 环境名是 `carla`，但实际目录可能是 `car` 或其他名称
  - ✅ **正确路径**: `/c/ProgramData/anaconda3/envs/carla/python.exe`
  - 包含: PyTorch 2.7.1+cu118, CARLA 0.10.0, rawpy
- **备用环境：deepsys** (用于 occ_transformer 训练)
  - 路径: `/c/ProgramData/anaconda3/envs/deepsys/python.exe`
  - 包含: PyTorch 2.6.0+cu124, 完整的深度学习工具链
- Python 版本：**3.10.x**
- 激活命令：`conda activate carla` 或 `conda activate deepsys`
- **所有 Python 操作和包安装必须在对应环境中进行**

**网络代理 / Network Proxy:**
- 所有下载和包安装命令必须使用代理：`192.168.100.182:7890`
- pip 安装示例：`pip install --proxy http://192.168.100.182:7890 package_name`
- git 已配置代理：`http.proxy` 和 `https.proxy`

**⚠️ Windows 命令行执行规范 / Windows Command Line Rules:**

**CRITICAL: 在 Windows 系统上,Claude Code 必须遵守以下规则:**

1. **永远不要直接使用 bash 风格的命令**
   - ❌ 错误: `python script.py` (bash 会找不到命令)
   - ❌ 错误: `conda run -n carla python script.py` (conda 命令不存在)
   - ❌ 错误: `C:\path\to\python.exe` (路径中的反斜杠和冒号会被 bash 误解)

2. **所有命令必须通过 cmd.exe 执行**
   - ✅ 正确: `cmd.exe /c "python script.py"`
   - ✅ 正确: `cmd.exe /c "cd /d d:\code\carla && python script.py"`
   - ✅ 正确: `cmd.exe /c "C:\Users\bills\.conda\envs\carla\python.exe script.py"`

3. **路径处理规则**
   - 使用 `/d` 参数切换驱动器: `cmd.exe /c "cd /d d:\code\carla"`
   - 路径中使用反斜杠 `\` 或正斜杠 `/` (Windows 都支持)
   - 包含空格的路径必须用引号: `"C:\Program Files\..."`

4. **Python 脚本执行的标准格式**
   ```bash
   # Bash tool 中使用此格式:
   cmd.exe /c "python occ_network_nano\train_bayer.py --arg value"

   # 如果需要切换目录:
   cmd.exe /c "cd /d d:\code\carla && python script.py"

   # 使用特定环境的 Python:
   cmd.exe /c "C:\ProgramData\anaconda3\envs\deepsys\python.exe script.py"
   ```

5. **检查文件/目录是否存在**
   ```bash
   # 检查文件:
   cmd.exe /c "if exist path\to\file.txt (echo Found) else (echo Not found)"

   # 检查目录:
   cmd.exe /c "if exist path\to\dir\ (echo Found) else (echo Not found)"
   ```

6. **常见操作的正确格式**
   ```bash
   # 列出文件:
   cmd.exe /c "dir /b d:\code\carla\*.py"

   # 删除文件:
   cmd.exe /c "del /f /q file.txt"

   # 创建目录:
   cmd.exe /c "mkdir new_directory"

   # 查找文本:
   cmd.exe /c "findstr /s /i 'pattern' *.py"
   ```

7. **⚠️ Claude Code Bash 环境特性**
   - Claude Code 的 Bash tool 运行在 Git Bash 环境中，**不继承用户 PowerShell 的 conda 激活状态**
   - ✅ **必须使用完整的 Python 路径** (推荐):
     ```bash
     /c/ProgramData/anaconda3/envs/carla/python.exe d:/code/carla/script.py
     ```
   - Windows 路径必须转换为 Unix 格式: `C:\path` → `/c/path`
   - 后台运行长时间脚本: 使用 `run_in_background=true` 参数

## Project Overview

CARLA is an open-source simulator for autonomous driving research built on Unreal Engine 5.5. This UE5 branch (`ue5-dev`) runs parallel to the legacy UE4 version in `ue4-dev`.

**System Requirements:**
- Ubuntu 22.04 or Windows 11 (UE5 will not work on older OS versions)
- Unreal Engine 5.5 (CARLA fork required - needs GitHub linked to Epic Games)
- CMake 3.27.2+
- Visual Studio 2022 (Windows) or GCC/Clang (Linux)
- 32GB+ RAM, NVIDIA RTX 3070+ recommended

## Build Commands

### Initial Setup

**Linux:**
```bash
./CarlaSetup.sh --interactive
# Or unattended with credentials:
sudo -E env GIT_LOCAL_CREDENTIALS=github_username@github_token ./CarlaSetup.sh
```

**Windows:**

前置步骤（使用 conda 环境）：
```cmd
# 1. 激活 carla conda 环境
conda activate carla

# 2. 确认 Python 版本（应该是 3.10.x）
python --version

# 3. 安装必要的 Python 包
pip install --proxy http://192.168.100.182:7890 -r requirements.txt

# 4. 启用 Windows 开发者模式
# 打开：设置 -> 隐私和安全性 -> 开发者选项 -> 开启"开发人员模式"
# 或运行：powershell -Command "Start-Process ms-settings:developers"

# 5. 运行 CarlaSetup.bat（需要管理员权限）
CarlaSetup.bat
```

The setup script installs prerequisites, clones Unreal Engine 5.5, downloads content, and performs the initial build. This takes several hours and requires 225GB+ disk space.

**注意：** CarlaSetup.bat 需要你手动输入 GitHub 凭据以下载 Unreal Engine 5.5。

### Rebuilding After Changes

Use these commands in a terminal (Linux) or "x64 Native Tools Command Prompt for VS 2022" (Windows):

**Configure:**
```bash
# Linux (with ROS2 support)
cmake -G Ninja -S . -B Build --toolchain=$PWD/CMake/Toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_ROS2=ON

# Windows (ROS2 not available)
cmake -G Ninja -S . -B Build --toolchain=$PWD/CMake/Toolchain.cmake -DCMAKE_BUILD_TYPE=Release

# Target specific Python installation:
# Add -DPython_ROOT_DIR=PATH -DPython3_ROOT_DIR=PATH
```

**Build:**
```bash
cmake --build Build
```

**Build and install Python API:**
```bash
cmake --build Build --target carla-python-api-install
```

**Launch Unreal Editor:**
```bash
cmake --build Build --target launch
```

**Build package:**
```bash
# Shipping package
cmake --build Build --target package

# Development package (with debug logs)
cmake --build Build --target package-development
```

### Build Presets

For multiple configurations, use presets:
```bash
# Linux
cmake --preset Linux-Debug          # Maximum debug info
cmake --preset Linux-Development    # Moderate debug info
cmake --preset Linux-Release        # Minimal debug info

# Then build with:
cmake --build Build/Linux-Release/ --target launch
```

## Code Architecture

### High-Level Structure

```
LibCarla/          → Core C++ library (client + server)
Unreal/            → Unreal Engine 5.5 integration
PythonAPI/         → Python bindings via Boost.Python
Ros2Native/        → Native ROS2 support (Linux only)
Examples/          → C++ client examples
CMake/             → Build system configuration
```

### LibCarla - The Core Library

Location: [LibCarla/source/carla/](LibCarla/source/carla/)

LibCarla is compiled into two separate libraries from the same source:
- **carla-server**: Embedded in Unreal Engine, handles simulation
- **carla-client**: Used by Python API and C++ clients

**Key modules:**

- **client/** - Client API ([Actor](LibCarla/source/carla/client/Actor.h), [World](LibCarla/source/carla/client/World.h), [Sensor](LibCarla/source/carla/client/Sensor.h), [Vehicle](LibCarla/source/carla/client/Vehicle.h))
  - `client/detail/` - Internal implementation (Simulator, Client, Episode)

- **rpc/** - Remote Procedure Call protocol (MsgPack-based serialization)
  - Command structures, actor definitions, control messages

- **sensor/** - Sensor framework
  - `sensor/data/` - Data types (Image, LidarMeasurement, CollisionEvent)
  - `sensor/s11n/` - Serialization/deserialization

- **streaming/** - High-performance TCP streaming for sensor data
  - Separate from RPC for low-latency data transfer

- **trafficmanager/** - Autonomous traffic management
  - Multi-stage pipeline: Localization → Collision → TrafficLight → MotionPlan
  - [TrafficManagerLocal](LibCarla/source/carla/trafficmanager/TrafficManagerLocal.h) and [TrafficManagerRemote](LibCarla/source/carla/trafficmanager/TrafficManagerRemote.h)

- **geom/** - Geometric primitives (Transform, Location, Vector3D, BoundingBox)
- **road/** - OpenDRIVE road network representation
- **nav/** - Navigation mesh (Recast/Detour integration)
- **ros2/** - ROS2 integration (when `ENABLE_ROS2=ON`)

### Unreal Engine Integration

Location: [Unreal/CarlaUnreal/](Unreal/CarlaUnreal/)

**Structure:**
- [CarlaUnreal.uproject](Unreal/CarlaUnreal/CarlaUnreal.uproject) - Main project file
- **Plugins/Carla/** - Main CARLA plugin (most code here)
- **Plugins/CarlaTools/** - Development tools
- **Content/Carla/** - Game assets, maps, blueprints

**Carla Plugin** ([Plugins/Carla/Source/Carla/](Unreal/CarlaUnreal/Plugins/Carla/Source/Carla/)):

- **Server/** - [FCarlaServer](Unreal/CarlaUnreal/Plugins/Carla/Source/Carla/Server/CarlaServer.h)
  - RPC server receiving commands from clients
  - Integrates carla-server library

- **Game/** - Core simulation
  - [UCarlaEpisode](Unreal/CarlaUnreal/Plugins/Carla/Source/Carla/Game/CarlaEpisode.h) - Manages simulation episode
  - [ACarlaGameModeBase](Unreal/CarlaUnreal/Plugins/Carla/Source/Carla/Game/CarlaGameModeBase.h) - Main game mode
  - [FFrameData](Unreal/CarlaUnreal/Plugins/Carla/Source/Carla/Game/FrameData.h) - Per-frame state

- **Sensor/** - Unreal sensor implementations
  - Camera sensors (RGB, Depth, SemanticSegmentation, InstanceSegmentation)
  - Lidar, Collision, GNSS, IMU, Radar, ObstacleDetection
  - [FWorldObserver](Unreal/CarlaUnreal/Plugins/Carla/Source/Carla/Sensor/WorldObserver.h) - Captures world state

- **Actor/** - Actor spawning and management
- **Vehicle/** - Vehicle physics using Chaos Vehicle Plugin
- **Traffic/** - Traffic lights and signs
- **Weather/** - Weather system
- **Recorder/** - Episode recording/replay

### Python API

Location: [PythonAPI/carla/](PythonAPI/carla/)

- **carla/src/** - Boost.Python bindings (C++ code wrapping carla-client)
  - [PythonAPI.cpp](PythonAPI/carla/src/PythonAPI.cpp) - Module definition
  - Actor.cpp, World.cpp, Sensor.cpp, etc. - Individual class bindings

- **carla/agents/** - High-level autonomous agents (Python-only)
  - navigation/, tools/ - Navigation and utilities

The Python package is built as a wheel and installed via pip.

### Communication Architecture

**Two separate communication channels:**

1. **RPC (rpclib)** - Port 2000 (default)
   - Synchronous request-response (spawn actor, control commands)
   - Client → FCarlaServer → UCarlaEpisode

2. **Streaming (custom TCP)** - Port 2001 (default)
   - Asynchronous sensor data streaming
   - High-performance, low-latency
   - Unreal Sensor → Serialization → Streaming Server → Client

3. **Multi-GPU Router** - Port 2002 (optional)

### Traffic Manager Pipeline

Location: [LibCarla/source/carla/trafficmanager/](LibCarla/source/carla/trafficmanager/)

Sequential processing stages:
1. **ALSM** - Agent lifecycle management
2. **LocalizationStage** - Path computation
3. **CollisionStage** - Collision detection
4. **TrafficLightStage** - Traffic light handling
5. **MotionPlanStage** - Control command generation
6. **VehicleLightStage** - Light management

### ROS2 Integration

Location: [Ros2Native/](Ros2Native/)

- Only available on Linux with `-DENABLE_ROS2=ON`
- Uses FastDDS middleware
- Native publishers/subscribers for sensor data and control
- Launch server with `./CarlaUnreal.sh --ros2`

## Coding Standards

**General:**
- Use spaces, not tabs
- No trailing whitespace

**Python:**
- PEP8 style guide ([PythonAPI/](PythonAPI/))
- Compatible with Python 3.7+ (Python 2.7 compatibility no longer required for UE5)
- Max 80 columns for comments, 120 for code
- Must pass Pylint without errors

**C++:**
- Max 80 columns for comments
- Must compile without warnings: `clang++ -Wall -Wextra -std=C++14`
- Use `carla::throw_exception` instead of `throw`
- Unreal C++ follows [Unreal Engine Coding Standard](https://docs.unrealengine.com/latest/INT/Programming/Development/CodingStandard/) (but spaces, not tabs)
- LibCarla uses [Google's C++ Style Guide](https://google.github.io/styleguide/cppguide.html) variant
- Wrap `try-catch` blocks with `#ifndef LIBCARLA_NO_EXCEPTIONS` in server-side code

## Important Environment Variables

**CARLA_UNREAL_ENGINE_PATH** - Path to CARLA fork of Unreal Engine 5.5
- Must be set for rebuilds after initial setup
- Automatically set by setup scripts in `.bashrc` (Linux)

**GIT_LOCAL_CREDENTIALS** - For unattended builds
- Format: `github_username@github_token`

## Common Development Tasks

### Running a Python Example
```bash
# Install the Python API first
cmake --build Build --target carla-python-api-install

# Start the server
cmake --build Build --target launch

# In another terminal, run examples
cd PythonAPI/examples
python3 manual_control.py
python3 spawn_npc.py -n 50  # Spawn 50 vehicles
```

### Modifying C++ Code

1. Edit files in [LibCarla/source/carla/](LibCarla/source/carla/) (client/server) or [Unreal/CarlaUnreal/Plugins/Carla/Source/Carla/](Unreal/CarlaUnreal/Plugins/Carla/Source/Carla/) (Unreal)
2. Rebuild: `cmake --build Build`
3. Reinstall Python API if client code changed: `cmake --build Build --target carla-python-api-install`
4. Launch editor: `cmake --build Build --target launch`

### Adding a New Sensor

1. Create Unreal sensor class in [Unreal/CarlaUnreal/Plugins/Carla/Source/Carla/Sensor/](Unreal/CarlaUnreal/Plugins/Carla/Source/Carla/Sensor/)
2. Add sensor data type in [LibCarla/source/carla/sensor/data/](LibCarla/source/carla/sensor/data/)
3. Add serialization in [LibCarla/source/carla/sensor/s11n/](LibCarla/source/carla/sensor/s11n/)
4. Register in actor factory
5. Export to Python in [PythonAPI/carla/src/](PythonAPI/carla/src/)

### Git Workflow

- Development branch: `ue5-dev` (current)
- Create feature branches: `username/feature_name`
- Pull requests target `ue5-dev` (not `master`)
- Follow [Gitflow](https://nvie.com/posts/a-successful-git-branching-model/) branching model

## Testing

Run all tests (after building):
```bash
make check
```

Note: The UE5 version is currently experimental and some features may change significantly.

## Key Documentation

- Full docs: https://carla-ue5.readthedocs.io
- Build Linux: https://carla-ue5.readthedocs.io/en/latest/build_linux_ue5/
- Build Windows: https://carla-ue5.readthedocs.io/en/latest/build_windows_ue5/
- Python API: https://carla-ue5.readthedocs.io/en/latest/python_api/
- Blueprint library: https://carla-ue5.readthedocs.io/en/latest/bp_library/

## Dependencies

Major third-party libraries:
- **Boost** (asio, python, geometry)
- **rpclib** (RPC communication)
- **Recast/Detour** (navigation meshes)
- **libpng, zlib** (image handling)
- **Eigen3** (linear algebra)
- **FastDDS** (ROS2 DDS middleware, Linux only)
- **MsgPack** (serialization)

All dependencies are automatically downloaded and built by CMake.

## Important Notes

- CARLA requires the **CARLA fork** of Unreal Engine 5.5, not the standard Epic Games UE5
- Link GitHub to Epic Games account: https://www.unrealengine.com/en-US/ue-on-github
- Cannot build on external disks (permission issues)
- Windows requires Developer Mode enabled
- First build can take 3+ hours and use 225GB+ disk space
- Environment variable `CARLA_UNREAL_ENGINE_PATH` must be set after initial setup

---

## 占用网络架构总览

本项目包含三个迭代版本的占用网络，从轻量实验逐步发展到工业级端到端方案：

| 项目 | 参数量 | 输出分辨率 | 特点 | 状态 |
|------|--------|-----------|------|------|
| `occ_network_nano` | 6.08M | (200,200,16) | 轻量级 LSS 风格，无时序 | 早期实验 |
| `occ_network` (OccNetV3) | ~50M | (200,200,16) | Lift-Splat-Shoot + 时序 | 中间版本 |
| `occ_transformer` | ~20M | (200,200,16) | 纯 Transformer，BEV 查询 | 并行方案 |
| `e2e_occ` | ~9M | (400,400,32) | **端到端，粗细两阶段，GRU 时序** | **主力方案** ⭐ |

所有网络共享同一数据集格式 (`dataset_10k`) 和 viewer (`occupancy_viewer`)。

---

## E2E 端到端占用网络 (e2e_occ) ⭐ 主力方案

### 项目概述

`e2e_occ` 是**当前主力方案**，参考特斯拉 FSD 架构设计，使用 Window Attention 图像编码 + 粗细两阶段可变形 Cross-Attention 解码 + GRU 时序融合。

**关键特点:**
- 输入: 8 相机 Bayer RAW `[B,8,1,960,1280]`
- 输出: `(400,400,32)` 体素，18 语义类别
- 参数量: ~9M，峰值显存 ~3GB（FP16）
- 时序: GRU 记忆 + Ego-Motion 对齐 + TBPTT 训练

### 网络架构

```
RAW [B,8,1,960,1280]
  ↓ RGGB 解包 + Stem 卷积
  → 特征图 [B,8,256,60,80]
  ↓ 图像编码器 (Window Attention × 2层)
  → 含2D位置编码 + 等距投影射线编码
  ↓ 粗解码器 (Coarse, 25×25×8=5000 queries)
  → Self-Attention + Deformable Cross-Attention
  ↓ 时序融合 (GRU + Ego-Motion 对齐)
  ↓ 细解码器 (Fine, 80×80×16=102400 queries)
  → Deformable Cross-Attention + Depthwise Conv3d 空间一致性
  ↓ 体素头 (VoxelHead)
  → 分类 → 上采样 [B,18,400,400,32]
```

**关键设计**:
- 串行处理 8 相机（显存×1，不是×8）
- Fine 阶段强制使用梯度检查点（gradient checkpointing）
- TBPTT chunk_size=2，截断长序列梯度

### 快速开始

```bash
# 训练 (在项目根目录 d:\code\carla 下):
/c/ProgramData/anaconda3/envs/deepsys/python.exe e2e_occ/train.py \
    --dataset dataset_10k --batch-size 1 --epochs 50 --amp

# 推理:
/c/ProgramData/anaconda3/envs/deepsys/python.exe e2e_occ/inference.py \
    --checkpoint e2e_occ/checkpoints/best.pth \
    --dataset dataset_10k

# 验证网络结构:
/c/ProgramData/anaconda3/envs/deepsys/python.exe e2e_occ/verify_network.py
```

### 关键文件

- [e2e_occ/config.py](e2e_occ/config.py) — `E2EOccConfig` 数据类，所有超参数
- [e2e_occ/e2e_occ_net.py](e2e_occ/e2e_occ_net.py) — 主网络入口 `build_model()`
- [e2e_occ/image_encoder.py](e2e_occ/image_encoder.py) — Window Attention 图像编码器
- [e2e_occ/occ_decoder.py](e2e_occ/occ_decoder.py) — 粗细两阶段解码器
- [e2e_occ/temporal_fusion.py](e2e_occ/temporal_fusion.py) — GRU + Ego-Motion 对齐
- [e2e_occ/deformable_attention.py](e2e_occ/deformable_attention.py) — 可变形 Cross-Attention，含 3D→2D 投影
- [e2e_occ/voxel_head.py](e2e_occ/voxel_head.py) — 分类头 + 上采样
- [e2e_occ/loss.py](e2e_occ/loss.py) — `CrossEntropy + Lovász-Softmax` 组合损失
- [e2e_occ/dataset.py](e2e_occ/dataset.py) — 支持序列 `[B,T,N,C,H,W]` 的数据加载器

### 空间范围配置

```python
voxel_range = (-40.0, -40.0, -1.0, 40.0, 40.0, 5.4)  # x: ±40m, y: ±40m, z: -1~5.4m
voxel_resolution = 0.2  # 0.2m/体素
voxel_size = (400, 400, 32)  # 输出体素网格
```

---

## OccNetV3 网络 (occ_network)

**中间迭代版本**，基于 Lift-Splat-Shoot (LSS) 架构 + 深度监督 + 多任务损失。

```bash
# 训练:
/c/ProgramData/anaconda3/envs/deepsys/python.exe occ_network/train.py \
    --dataset dataset_10k_bak --batch-size 1 --epochs 20 --amp

# 数据采集 (需要先启动 CARLA 服务器):
/c/ProgramData/anaconda3/envs/carla/python.exe occnetv3_data_generator/main_collection.py \
    --frames 1000 --output dataset_10k_bak --town Town10HD
```

**损失函数**: Focal Loss + Distance-Aware Loss + 边缘感知深度监督 L1 Log Loss。

---

## Occupancy Network 项目说明 (occ_network_nano)

> **注意**: `occ_network_nano` 是早期轻量级实验版本，当前主力方案请使用 `e2e_occ`。

### 项目概述

`occ_network_nano` 是基于 CARLA 采集的数据训练的**轻量级 3D 占用网格预测网络**,使用 **Bayer RGGB 单通道 RAW 数据**作为输入。

**关键特点:**
- 输入: 8 个环视相机的 Bayer RAW 图像 (单通道 12-bit DNG)
- 输出: 3D 占用网格 `(200, 200, 16)` 体素, 18 个语义类别
- 网络: 轻量级设计,总参数 6.08M
- 数据集: `dataset_10k` (920 个样本,原始分辨率 500×500×40, 训练时下采样到 200×200×16)

### 目录结构

```
occ_network_nano/
├── models/                    # 网络模型
│   ├── backbone/             # BayerMobileNetV2 (4.90M 参数)
│   ├── neck/                 # LiteFPN (0.41M)
│   ├── transformer/          # LiteViewTransformer (0.24M)
│   ├── encoder/              # LiteBEVEncoder (0.24M)
│   ├── decoder/              # LiteOccDecoder (0.30M)
│   └── bayer_occ_net.py      # 完整网络
├── data/                     # 数据加载
│   └── carla_dataset_bayer.py  # Bayer 数据集加载器
├── utils/                    # 工具
│   ├── bayer_utils.py        # Bayer RAW 处理
│   └── loss.py               # 损失函数 (MaskedWeightedCELoss)
├── train_bayer.py            # 训练脚本 ⭐
├── inference_bayer.py        # 推理脚本 ⭐
└── verify_complete_network.py # 网络验证脚本
```

### 快速开始

#### 1. 训练模型

**环境要求**: `deepsys` (包含 PyTorch + CUDA)

```bash
# 用户在 PowerShell 中手动激活环境:
conda activate deepsys

# 开始训练:
python occ_network_nano/train_bayer.py \
    --dataset dataset_10k \
    --batch-size 2 \
    --epochs 50 \
    --device cuda \
    --amp
```

**训练参数说明**:
- `--dataset`: 数据集路径 (默认 `dataset_10k`)
- `--batch-size`: 批量大小 (推荐 2-4, 取决于显存)
- `--epochs`: 训练轮数 (推荐 50)
- `--amp`: 混合精度训练 (节省显存,加速训练)
- `--lr`: 学习率 (默认 1e-3)
- `--save-dir`: 模型保存目录 (默认 `outputs/bayer_raw`)

**输出**:
- Checkpoint: `outputs/bayer_raw/<timestamp>/epoch_XXX.pth`
- 日志: `outputs/bayer_raw/<timestamp>/train.log`

#### 2. 推理和可视化

**推理**:
```bash
python occ_network_nano/inference_bayer.py \
    --checkpoint outputs/bayer_raw/<timestamp>/epoch_049.pth \
    --dataset dataset_10k \
    --num-samples 10
```

**输出**:
- 推理结果: `inference_results/*.npz` (与 viewer 兼容格式)
- 指标文件: `inference_results/metrics.txt`

**可视化**:
```bash
# 启动 viewer (自动加载 inference_results)
python occupancy_viewer/run_viewer.py

# 浏览器访问: http://localhost:8085/
```

#### 3. 验证网络结构

```bash
python occ_network_nano/verify_complete_network.py
```

显示网络参数统计和前向传播测试。

### 数据格式

#### 输入数据 (dataset_10k)

```
dataset_10k/
├── cameras/              # Bayer RAW 图像
│   ├── cam_front_main/
│   │   ├── 000000.dng   # 12-bit Bayer RGGB, 单通道
│   │   └── ...
│   ├── cam_front_wide/
│   └── ... (8 个相机)
├── occupancy/            # 真值占用网格
│   ├── 000000.npz
│   │   ├── occupancy: (500, 500, 40) uint8  # 语义类别 [0-17]
│   │   ├── mask: (500, 500, 40) bool        # 有效区域
│   │   ├── actor_ids: (500, 500, 40) int32  # Actor ID
│   │   ├── x_range, y_range, z_range        # 空间范围
│   │   ├── resolution: 0.2m                 # 体素分辨率
│   │   └── grid_size: (500, 500, 40)
│   └── ...
└── camera_params/        # 相机参数
    ├── 000000.npz
    │   ├── intrinsics: (8, 3, 3)  # 内参矩阵
    │   └── extrinsics: (8, 4, 4)  # 外参矩阵
    └── ...
```

**重要**: 数据集加载器会自动将 `(500, 500, 40)` 下采样到 `(200, 200, 16)` 以适配网络输出。

#### 推理输出 (inference_results)

```
inference_results/
├── 000000.npz
│   ├── occupancy: (200, 200, 16) uint8   # 预测类别
│   ├── mask: (200, 200, 16) bool         # 有效区域
│   ├── x_range, y_range, z_range
│   ├── resolution: 0.5m                  # 更新的分辨率 (100m / 200)
│   └── grid_size: (200, 200, 16)
└── metrics.txt                           # 评估指标
```

### 语义类别 (18 类)

```
0:  Free/Unlabeled (空白空间) ⭐
1:  Building
2:  Fence
3:  Other
4:  Pedestrian
5:  Pole
6:  RoadLine
7:  Road
8:  Sidewalk
9:  Vegetation
10: Vehicle
11: Wall
12: TrafficSign
13: Sky
14: Ground
15: Bridge
16: RailTrack
17: GuardRail
```

**⚠️ Class 0 重要性**:
- Class 0 (空白空间) 是占用网格中最重要的类别之一
- 损失函数中 Class 0 权重必须 >= 1.0, 否则网络会完全忽略空白类
- 当前配置: Class 0 权重 = 1.0 (已修复,之前是 0.1)

### 损失函数

使用 `MaskedWeightedCELoss` (带掩码和类别权重的交叉熵):
- 支持 mask: 只计算有效体素 (mask=True)
- 支持 class_weights: 对类别不平衡进行加权
- Class 0 权重 = 1.0 (基准权重)
- 稀有类权重更高 (行人 5.0, 交通标志 5.0, 车辆 2.0)

### 网络架构

```
Input: [B, 8, 1, 384, 640]  # 8 相机 × Bayer RAW

    ↓ BayerMobileNetV2 (4.90M)
    → PixelUnshuffle(2): [B, 1, H, W] → [B, 4, H/2, W/2]
    → 分离 RGGB 通道,避免颜色混合
    → 输出多尺度特征: C3, C4, C5

    ↓ LiteFPN (0.41M)
    → 融合多尺度特征
    → 输出: [B×8, 128, H/8, W/8]

    ↓ LiteViewTransformer (0.24M)
    → 2D 多视图 → BEV 投影 (LSS 风格)
    → 输出: [B, 128, 100, 100]

    ↓ LiteBEVEncoder (0.24M)
    → BEV 特征增强 (残差块)
    → 输出: [B, 128, 100, 100]

    ↓ LiteOccDecoder (0.30M)
    → 高度扩展 + 3D 卷积
    → 输出: [B, 18, 200, 200, 16]
```

### 常见问题

#### Q1: 训练时 DNG 文件无法加载?

**错误**: `OpenCV TIFF: Sorry, can not handle PhotometricInterpretation=32803`

**原因**: OpenCV 不支持 CFA 格式的 DNG

**解决**: 安装 `rawpy` 库
```bash
pip install rawpy
```

`bayer_utils.py` 会优先使用 `rawpy` 加载 DNG, 失败后才降级到 OpenCV。

#### Q2: 推理结果全是实心/红色?

**原因**: 网络没有学会预测 Class 0 (空白)

**检查**:
1. 确认损失函数中 Class 0 权重 >= 1.0 (loss.py:95)
2. 检查数据集是否包含 Class 0
3. 重新训练模型

#### Q3: Viewer 无法显示推理结果?

**检查**:
1. `run_viewer.py` 中 `DATA_DIR` 是否指向 `inference_results`
2. 推理输出的 `resolution` 和 `grid_size` 是否正确
3. `mask` 数据类型是否为 `bool`

### 性能指标

**预期性能** (训练 50 epochs，Class 0 权重 = 1.0):
- Accuracy: 50-70%
- mIoU: 20-35%
- 推理速度: ~50-100 ms/sample (RTX 4090)
- 显存占用: ~4-6 GB (batch_size=2, AMP)

---

## Occupancy Transformer 项目说明 (occ_transformer)

### 项目概述

`occ_transformer` 是基于 **纯 Transformer 架构** 的 3D 占用网格预测网络,将多视角 2D 图像直接"翻译"为 3D 体素。

**核心思想**:
```
多视角 2D 像素序列 → Transformer → 3D 体素序列
     (源语言)                      (目标语言)
```

**关键创新**:
- 相机参数作为位置编码: 射线方向 + 相机位置作为几何先验
- Cross-Attention 学习像素-体素对应: 自动学习投影关系,无需显式几何投影

### 三个版本对比

| 版本 | 参数量 | BEV尺寸 | 序列长度 | Decoder层数 | 推荐场景 |
|------|--------|---------|----------|------------|---------|
| **Nano** | ~10M | 25×25 | 600 tokens | 2层 | 快速实验 |
| **Standard** | ~30M | 100×100 | 10,000 tokens | 6层 | 高精度 |
| **Balanced** ⭐ | ~20M | 50×50 | 2,500 tokens | 3层 | **推荐** |

### 快速开始 (Balanced 版本)

**环境要求**: `deepsys` (包含 PyTorch + CUDA)

```bash
# 用户在 PowerShell 中手动激活环境:
conda activate deepsys

# 训练:
python occ_transformer/train_balanced.py \
    --dataset dataset_10k \
    --batch-size 2 \
    --epochs 50 \
    --lr 1e-4 \
    --amp

# 推理:
python occ_transformer/inference_transformer.py \
    --checkpoint outputs/transformer_balanced/xxx/best.pth \
    --dataset dataset_10k \
    --model-type balanced

# 验证网络结构:
python occ_transformer/verify_balanced_net.py
```

### 网络架构 (Balanced 版)

```
Input: [B, 8, 1, 960, 1280]  # 8相机 Bayer RAW

    ↓ BayerPatchEmbed (patch_size=16)
    → PixelUnshuffle(2) + Conv
    → 输出: [B, N_patches, 512]

    ↓ CameraPositionEncoding
    → 射线方向 (dx, dy, dz)
    → 相机位置 (cx, cy, cz)
    → 像素坐标 (u, v)

    ↓ TransformerEncoder (4层)
    → Self-Attention (多头注意力)
    → FFN (前馈网络)
    → 输出: [B, N_patches, 512]

    ↓ BEVQueries (50×50)
    → 可学习查询 + 2D位置编码
    → 输出: [B, 2500, 512]

    ↓ TransformerDecoder (3层)
    → Self-Attention
    → Deformable Cross-Attention (像素→体素)
    → FFN
    → 输出: [B, 2500, 512]

    ↓ 高度扩展 + 上采样
    → Reshape: [B, 50, 50, 512]
    → 高度扩展: [B, 50, 50, 8, 256]
    → 上采样: [B, 200, 200, 16, 256]
    → 分类头: [B, 18, 200, 200, 16]
```

### 关键模块

**位置编码** ([position_encoding.py](occ_transformer/models/transformer_occ/position_encoding.py)):
- `CameraPositionEncoding`: 相机几何编码 (射线方向 + 相机位置)
- `Voxel3DPositionEncoding`: 3D 体素位置编码
- `Spatial2DPositionEncoding`: 2D BEV 位置编码

**注意力机制** ([attention.py](occ_transformer/models/transformer_occ/attention.py)):
- `MultiHeadAttention`: 标准多头注意力
- `DeformableAttention`: 可变形注意力 (节省显存)

**体素查询** ([voxel_query.py](occ_transformer/models/transformer_occ/voxel_query.py)):
- `BEVQueries`: 2D BEV 查询 + 高度扩展

### 训练参数说明

- `--model-type`: 模型版本 (nano/standard/balanced)
- `--dataset`: 数据集路径
- `--batch-size`: 批量大小 (推荐 1-2)
- `--epochs`: 训练轮数 (推荐 50-100)
- `--lr`: 学习率 (推荐 1e-4)
- `--amp`: 混合精度训练
- `--grad-clip`: 梯度裁剪 (推荐 1.0)

---

## 数据采集系统

### carla_data_collection (轻量级方案)

**目标**: 8 相机 RGB + 语义激光雷达 → 占用网格数据对

**快速开始**:
```bash
# 1. 启动 CARLA 服务器
cmake --build Build --target launch

# 2. 运行采集脚本 (采集 5 帧演示)
cd carla_data_collection
python scripts/collect_5_frames.py
```

**输出结构**:
```
dataset_output/town10_test/
├── cameras/              # 8 相机 RGB 图像 (PNG)
│   ├── cam_front_main/   # 前置主摄 (50° FOV)
│   ├── cam_front_wide/   # 前置广角 (120° 鱼眼)
│   ├── cam_front_narrow/ # 前置长焦 (35° FOV)
│   ├── cam_left_pillar/  # 左侧 B 柱
│   ├── cam_right_pillar/ # 右侧 B 柱
│   ├── cam_left_repeater/# 左侧翼子板
│   ├── cam_right_repeater/# 右侧翼子板
│   └── cam_rear/         # 后置相机 (120° 鱼眼)
├── occupancy/            # 3D 体素标签 (NPZ)
│   └── 000000.npz        # occupancy, mask, actor_ids
└── camera_params/        # 相机内外参 (NPZ)
    └── 000000.npz        # intrinsics, extrinsics
```

**关键配置文件**:
- [camera_config.py](carla_data_collection/config/camera_config.py) - Tesla 8 相机布局参数
- [occupancy_config.py](carla_data_collection/config/occupancy_config.py) - 体素空间参数

**数据验证**:
```bash
python carla_data_collection/scripts/verify_occupancy.py
```

### dense_occupancy_collection (高质量方案)

**目标**: CubeMap 深度图 + 保守光栅化 → nuScenes 17 类标准体素

**特性**:
- 6 深度相机 CubeMap (前/后/左/右/上/下)
- 保守光栅化 (Conservative Rasterization)
- 实例级补全 (Instance Completion)
- nuScenes 17 类语义标签

**运行**:
```bash
python dense_occupancy_collection/main_data_collection.py --frames 10
```

**语义映射** ([actor_occupancy_mapping.py](dense_occupancy_collection/config/actor_occupancy_mapping.py)):
- 详细的 CARLA Actor → nuScenes 类别映射表
- 支持 17 类标准标签 (free, car, pedestrian, truck, etc.)

---

## 数据可视化工具 (occupancy_viewer)

### 概述

基于 **Three.js** 的交互式 3D 体素可视化工具,支持推理结果和真值标签的实时渲染。

### 快速开始

```bash
# 启动 viewer (自动加载 inference_results)
python occupancy_viewer/run_viewer.py

# 浏览器访问:
http://localhost:8085/
```

### 功能特性

- ✅ 鼠标旋转/缩放/平移
- ✅ 多视角切换 (俯视/前视/侧视/自由)
- ✅ 帧浏览切换 (上一帧/下一帧)
- ✅ 实时体素统计 (总数/非空数/类别分布)
- ✅ 自动播放功能
- ✅ 18 类语义颜色映射

### 数据格式要求

NPZ 文件必须包含以下字段:
```python
{
    'occupancy': (X, Y, Z) uint8      # 体素类别 [0-17]
    'mask': (X, Y, Z) bool            # 可见性掩码
    'x_range': [xmin, xmax]           # X 轴范围
    'y_range': [ymin, ymax]           # Y 轴范围
    'z_range': [zmin, zmax]           # Z 轴范围
    'resolution': float               # 体素分辨率 (米)
    'grid_size': (int, int, int)      # 网格尺寸
}
```

### 配置

修改 [run_viewer.py](occupancy_viewer/run_viewer.py) 中的 `DATA_DIR` 变量:
```python
DATA_DIR = "inference_results"  # 推理结果
# 或
DATA_DIR = "dataset_10k/occupancy"  # 真值标签
```

---

## 完整工作流

### 1. 数据采集

```bash
# 启动 CARLA 服务器
cmake --build Build --target launch

# 方案 A: 轻量级采集 (8 相机 RGB + 激光雷达占用标签)
/c/ProgramData/anaconda3/envs/carla/python.exe carla_data_collection/scripts/collect_5_frames.py

# 方案 B: 高质量 CubeMap 深度采集 (nuScenes 17 类)
/c/ProgramData/anaconda3/envs/carla/python.exe dense_occupancy_collection/main_data_collection.py --frames 100

# 方案 C: OccNetV3 格式采集
/c/ProgramData/anaconda3/envs/carla/python.exe occnetv3_data_generator/main_collection.py \
    --frames 1000 --output dataset_10k_bak --town Town10HD
```

### 2. 网络训练 (推荐 e2e_occ)

```bash
# E2E 主力方案 (~9M 参数, 时序融合, 高分辨率输出):
/c/ProgramData/anaconda3/envs/deepsys/python.exe e2e_occ/train.py \
    --dataset dataset_10k --batch-size 1 --epochs 50 --amp

# Transformer 方案 (~20M 参数):
/c/ProgramData/anaconda3/envs/deepsys/python.exe occ_transformer/train_balanced.py \
    --dataset dataset_10k --batch-size 2 --epochs 50 --lr 1e-4 --amp

# 轻量级实验 (6.08M 参数, 无时序):
/c/ProgramData/anaconda3/envs/deepsys/python.exe occ_network_nano/train_bayer.py \
    --dataset dataset_10k --batch-size 2 --epochs 50 --amp
```

### 3. 推理与可视化

```bash
# E2E 推理:
/c/ProgramData/anaconda3/envs/deepsys/python.exe e2e_occ/inference.py \
    --checkpoint e2e_occ/checkpoints/best.pth --dataset dataset_10k

# 启动 viewer (浏览器访问 http://localhost:8085/):
/c/ProgramData/anaconda3/envs/deepsys/python.exe occupancy_viewer/run_viewer.py
```

---

## 语义类别定义

### carla_data_collection 和 occ_network_nano (18 类)

```
0:  Free/Unlabeled (空白空间) ⭐ 权重=1.0
1:  Building
2:  Fence
3:  Other
4:  Pedestrian (权重=5.0)
5:  Pole
6:  RoadLine
7:  Road
8:  Sidewalk
9:  Vegetation
10: Vehicle (权重=2.0)
11: Wall
12: TrafficSign (权重=5.0)
13: Sky
14: Ground
15: Bridge
16: RailTrack
17: GuardRail
```

### dense_occupancy_collection (17 类 nuScenes 标准)

```
0:  free (空白空间)
1:  barrier (护栏)
2:  bicycle
3:  bus
4:  car (最重要)
5:  construction_vehicle
6:  motorcycle
7:  pedestrian (重要)
8:  traffic_cone
9:  trailer
10: truck (重要)
11: driveable_surface (强制保留)
12: other_flat
13: sidewalk (强制保留)
14: terrain
15: manmade
16: vegetation
17: general_object
```

---

## 常见问题排查

### Q1: CARLA 服务器无法启动?

**检查**:
```bash
# 确认 CARLA_UNREAL_ENGINE_PATH 环境变量
echo %CARLA_UNREAL_ENGINE_PATH%

# 尝试手动启动
cmake --build Build --target launch
```

### Q2: DNG 文件无法加载?

**错误**: `OpenCV TIFF: Sorry, can not handle PhotometricInterpretation=32803`

**解决**:
```bash
pip install --proxy http://192.168.100.182:7890 rawpy
```

### Q3: 训练时显存不足?

**解决方案**:
1. 减小 batch_size (推荐 1-2)
2. 启用混合精度训练 (`--amp`)
3. 使用更小的模型 (nano 版本)
4. 降低图像分辨率

### Q4: 推理结果全是实心/红色?

**原因**: 网络没有学会预测 Class 0 (空白)

**检查**:
1. 确认损失函数中 Class 0 权重 >= 1.0
2. 检查数据集是否包含 Class 0
3. 重新训练模型

### Q5: Viewer 无法显示结果?

**检查**:
1. `run_viewer.py` 中 `DATA_DIR` 是否正确
2. NPZ 文件格式是否符合要求
3. `mask` 数据类型是否为 `bool`
4. `resolution` 和 `grid_size` 是否正确

### Q6: 训练过程中梯度爆炸?

**解决**:
```bash
# 添加梯度裁剪
python occ_transformer/train_balanced.py \
    --grad-clip 1.0 \
    --lr 5e-5  # 降低学习率
```
