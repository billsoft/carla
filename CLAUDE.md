# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 重要提示 / Important Notes

**语言 / Language:** 请始终使用中文与用户沟通。Always communicate with users in Chinese.

**代码修复原则 / Code Fix Principles:**
- ⚠️ **重要：有问题的文件我们在问题文件上修复，不要创建新的 fix、fixed 等后缀的文件**
- Fix bugs directly in the original file, do NOT create new files with suffixes like `_fix`, `_fixed`, `_new`, etc.
- 直接修改源文件，保持代码库整洁

**Python 环境管理 / Python Environment:**
- 本项目使用 **conda** 管理 Python 环境
- 环境名称：**carla**
- Python 版本：**3.10.19**
- 激活命令：`conda activate carla`
- **所有 Python 操作和包安装必须在 carla 环境中进行**
- 环境路径：`C:\Users\bills\.conda\envs\carla`

**网络代理 / Network Proxy:**
- 所有下载和包安装命令必须使用代理：`192.168.100.182:7890`
- pip 安装示例：`pip install --proxy http://192.168.100.182:7890 package_name`
- git 已配置代理：`http.proxy` 和 `https.proxy`

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
