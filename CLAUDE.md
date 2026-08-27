# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 重要提示 / Important Notes

**语言 / Language:** 请始终使用中文与用户沟通。Always communicate with the user in Chinese.

**代码修复原则:** 有问题的文件在原文件上修复，不要创建 `_fix`、`_fixed`、`_new` 等后缀的新文件。

---

## 仓库概况

这个仓库包含两个几乎独立的代码库：

1. **CARLA 仿真器本体**（`LibCarla/`、`Unreal/`、`PythonAPI/`、`Ros2Native/` 等）— upstream `carla-simulator/carla` 的 `ue5-dev` 分支（UE5.5 版本），本仓库在其底层做了少量增强（见下方 "Bayer RAW" 一节）。
2. **自动驾驶感知研究代码**（仓库根目录下的 `occnetv3_data_generator/`、`e2e_occ/`、`dataset_viewer_v2/` 等）— 基于上面的 CARLA 采集数据、训练占用网络（Occupancy Network）的个人研究项目，与 CARLA 官方代码无关，不会被 upstream 同步覆盖。

两者对 Claude Code 的操作要求不同：改 CARLA 引擎代码要走 CMake/Ninja 全量重编译（很慢）；改 `occnetv3_data_generator/` 或 `e2e_occ/` 下的 Python 代码可以直接跑脚本，不需要重新编译。

---

## 构建 CARLA（C++ / UE5 引擎部分）

### 已验证可用的 Windows 构建脚本

```cmd
cd D:\code\carla
BUILD_FINAL.bat
```

这个脚本比 README 里的通用 cmake 流程更可靠，因为它包含了几个针对本机环境的关键修复：
- 强制使用 VS 2026 (v18)：`C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat`
- `-DCARLA_UNREAL_ENGINE_PATH="D:\code\UnrealEngine5_carla"`（CARLA fork 的 UE5.5 源码路径，非官方 Epic UE5）
- `/wd4723` 编译选项，抑制 UE5 ChaosVehicles 代码里的 "potential divide by 0" 报错（否则编译失败）
- 编译目标固定为 `carla-unreal-editor`

构建产物验证：检查 `Unreal\CarlaUnreal\Plugins\Carla\Binaries\Win64\*.dll` 是否存在。

### 通用命令（Linux，以及 Windows 上手动排障时使用）

```sh
cmake -G Ninja -S . -B Build --toolchain=$PWD/CMake/Toolchain.cmake -DCMAKE_BUILD_TYPE=Release
# Linux 加 ROS2 支持: -DENABLE_ROS2=ON

cmake --build Build                                      # 构建
cmake --build Build --target carla-python-api-install    # 编译并安装 Python API
cmake --build Build --target launch                      # 构建并在 UE 编辑器中打开
cmake --build Build --target launch-only                 # 仅打开编辑器，不重新构建
cmake --build Build --target carla-help                  # 输出完整 target/option 列表到 Build/Help.md
```

主要 CMake target：`carla-server`、`carla-client`、`carla-python-api(-install)`、`carla-unreal`、`carla-unreal-editor`、`carla-unreal-package[-shipping|-debug|-debuggame|-development|-test]`、`launch`、`launch-only`。完整列表见 `Build/Help.md`（由 `carla-help` target 生成）。

初始环境搭建（首次，需要 GitHub 账号已关联 Epic Games 账号才能 clone UE5 fork，225GB+ 磁盘，3 小时+）：`CarlaSetup.bat`（Windows）/ `./CarlaSetup.sh --interactive`（Linux）。CARLA 仓库和 UE5 引擎 fork 是两个独立仓库、如何关联、编译时踩过的坑，完整记录见 `CARLA_BUILD_NOTES.md`。

### 启动服务器 / 排障

```cmd
start_carla_server.bat
# 或直接: CarlaUE5.exe -quality-level=Low -RenderOffScreen
```

编辑器打开后必须点击绿色 Play 按钮场景才会开始 tick，Python 客户端连接（port 2000）才不会超时；命令行加 `-CarlaAutoPlay` 可以让编辑器在资产/着色器编译完成后自动开始 Play，不需要手动点（`Carla.cpp::FCarlaModule::RegisterAutoPlayWatcher`，详见 `CARLA_BUILD_NOTES.md` 4.9 节）。常见问题（光照过暗、Traffic Manager port 8000 绑定冲突、`time-out while waiting for the simulator` 等）及修复方式见 `GUIDE.md` 的"问题排查"章节。

---

## 测试

```sh
# 单元测试：纯 unittest.TestCase，需要已构建并安装 carla Python 模块，不需要运行中的模拟器
python -m unittest discover -s PythonAPI/test/unit

# 集成测试（PythonAPI/test/API/）与 smoke 测试（PythonAPI/test/smoke/）：
# 需要 CARLA 服务器正在运行（编辑器已点击 Play，或 CarlaUE5.exe 已启动）
python PythonAPI/test/API/test_sync_mode.py
```

`PythonAPI/test/unit/unittest.cfg` 与 `PythonAPI/test/smoke/unittest.cfg` 配置了 nose2 的 junit-xml 插件（可选，用 `nose2` 替代 `python -m unittest` 以生成 `test-results.xml`）。`PythonAPI/test/API/Tests.md` 记录了已知在 UE5.5 上会崩溃/异常的传感器（例如 OpticalFlowCamera 会段错误），排查异常前先查一下这份文档。

---

## 代码规范

详见 `Docs/cont_coding_standard.md`。

- **通用：** 用空格不用 Tab；不要留尾随空白。
- **Python：** 遵循 PEP8（`.pep8` 里配置为 120 列），pylint 需无警告/错误（配置在 `PythonAPI/.pylintrc`）。
- **C++：** `clang++ -Wall -Wextra -std=C++14` 编译无警告；禁止直接用 `throw`，一律用 `carla::throw_exception`；server 端的 `try-catch` 需要包在 `#ifndef LIBCARLA_NO_EXCEPTIONS` 里；Unreal 插件代码遵循 UE 官方 Coding Standard（但用空格不用 Tab）；`LibCarla/` 遵循 Google C++ 风格的变体。

---

## Python 环境（Windows / Git Bash）

**Bash tool 运行在 Git Bash 中，不会继承 `conda activate` 状态**，必须用完整路径调用 python.exe：

```bash
/c/Users/bills/.conda/envs/carla/python.exe some_script.py
```

本机存在不止一套 conda 安装，同名环境路径不唯一，使用前建议先验证：

```bash
<候选python.exe路径> -c "import carla; print(carla.__version__)"
```

已知路径（用途不完全相同，不要假设它们等价）：
- `C:\Users\bills\.conda\envs\carla` — `BUILD_FINAL.bat` 用它设置 CMake 的 `Python_ROOT_DIR`（编译 Python API 用）
- `C:\ProgramData\anaconda3\envs\carla`、`C:\ProgramData\anaconda3\envs\deepsys` — 数据采集 / 网络训练脚本常用

反斜杠路径在 Git Bash 里会被转义，需要用 `/c/...` 格式，或整体交给 `cmd.exe /c "..."` 执行。长时间运行的脚本（训练、数据采集、编译）用 Bash tool 的 `run_in_background=true`，不要用固定 `sleep` 轮询。

---

## CARLA 引擎架构速览

```
LibCarla/          核心 C++ 库（双编译：carla-server + carla-client）
Unreal/CarlaUnreal/Plugins/Carla/   UE5.5 集成主插件
PythonAPI/carla/src/                Python 绑定（Boost.Python）
Ros2Native/                         ROS2 支持（仅 Linux，FastDDS）
```

**通信架构：**
- RPC (rpclib)，port 2000 — 同步命令（spawn/control）→ `FCarlaServer` → `UCarlaEpisode`
- Streaming (TCP)，port 2001 — 异步传感器数据流
- Multi-GPU Router，port 2002（可选）

**添加新传感器**需要同步改动 4 处：
1. UE 类 → `Unreal/CarlaUnreal/Plugins/Carla/Source/Carla/Sensor/`
2. 数据类型 → `LibCarla/source/carla/sensor/data/`
3. 序列化 → `LibCarla/source/carla/sensor/s11n/`
4. 注册 Actor 工厂，并导出到 `PythonAPI/carla/src/`

---

## 本仓库对 CARLA 的底层增强：等距投影鱼眼相机 + 原生 Bayer RAW 采集

标准 CARLA 发行版只支持 8-bit BGRA 相机输出，且没有等距投影（equidistant/鱼眼）相机的
Bayer RAW 采集能力。本仓库加了两块增强，供 `occnetv3_data_generator/` 使用：

1. **Bayer RAW HDR 采集**：`RawType`（`"uint8"`/`"uint16"`/`"float32"`/`"bayer_rggb"`）
   属性 + `EPixelFormat::BAYER_RGGB_U16` 等新像素格式，`LibCarla/sensor/data/Image.h`、
   `LibCarla/sensor/s11n/ImageSerializer.h`、Python API `SensorData.cpp` 的零拷贝
   `raw_data` 访问。原本只在普通针孔相机 `SceneCaptureCamera.h/.cpp` 上实现，
   2026-08 起等距鱼眼相机 `SceneCaptureCamera_WideAngleLens.h/.cpp` 也接上了同一套。
2. **等距投影鱼眼相机**（`sensor.camera.rgb_fisheye` + `camera_model=equidistant`）：
   复用 CARLA 已有的 `CameraModelUtil`/`ASceneCaptureCamera_WideAngleLens` 基础设施
   （cubemap 捕获 + distort compute shader），FOV ≥ 80° 的相机不用针孔投影模型，
   避免画面边缘严重畸变。

数据流：`UE5 等距 distort shader → FLinearColor → Bayer 采样 → uint16 → TCP →
Python np.frombuffer → DNG`。改这部分代码需要走上面"构建 CARLA"里的全量 C++
重编译。**这条数据流上有过 5 个互相独立的坑（属性没注册、鱼眼相机漏触发重新渲染、
GPU 读回没等完成、客户端反序列化按错误格式解析、16-bit 格式没启用），完整记录和
每一处的根因见 `CARLA_BUILD_NOTES.md` 4.8 节**，实现细节（数据采集侧怎么用）见
`occnetv3_data_generator/README.md`。

---

## ML 研究子项目（仓库根目录，与 CARLA 引擎代码分离）

只有 `e2e_occ/` 是当前维护中的网络。`occ_network/`、`occ_transformer/` 是过时的早期实验，`dense_occupancy_collection/` 已被 `occnetv3_data_generator/` 取代，均已弃用，改动前先确认用户确实需要碰它们。

### 数据采集：`occnetv3_data_generator/`

详细架构、坑点、性能数据见其自带的 `README.md`（内容很完整，不要重复造轮子）。要点：
- `main_collection.py` 是主入口，依赖本仓库自编译的 CARLA（上面的 Bayer RAW 增强），标准 CARLA 发行版跑不了。
- 输出：8 相机 Bayer RAW DNG（Tesla 环视布局）+ 深度图 + 256 线语义 LiDAR + 400×400×32（0.2m/体素，X/Y ±40m，Z -1~5.4m）体素真值（18 类语义）+ 场景流。
- 相机布局在 `config/camera_config.py`，体素空间定义在 `config/occupancy_config.py`，CARLA `type_id` → 18 类语义的映射在 `config/actor_occupancy_mapping.py`。

```bash
<python> occnetv3_data_generator/main_collection.py --frames 100 --output <dir> --town Town10HD --num-vehicles 30 --num-walkers 10
<python> occnetv3_data_generator/visualize_dataset.py --dataset <dir> --sample 0
```

### 网络：`e2e_occ/`

详细的逐模块架构见 `ARCHITECTURE.md`，训练流程/损失函数/显存优化见 `TRAINING.md`（均对照实际代码核对过，改网络结构前务必先读，不要凭记忆假设结构没变——`voxel_head.py` 等模块已经和早期版本有实质性差异）。要点：
- 参考 Tesla FSD 架构，10.49M 参数（实测），输入 8 路等距投影 Bayer RAW `[B,8,1,960,1280]`，输出 `(400,400,32)` 18 类语义体素。
- 粗细两阶段 Deformable Cross-Attention 解码 + GRU 时序融合（Ego-Motion 对齐）。
- 相机是等距投影（equidistant），不是针孔——`position_encoding.py`（反投影：像素→射线方向）和 `deformable_attention.py`（正投影：3D点→像素）必须用同一套等距投影公式，改一处要同步改另一处，见 `ARCHITECTURE.md` 第 3 节。

```bash
conda activate deepsys
python e2e_occ/train.py --data_root <dataset_dir> --batch_size 1 --epochs 100 --amp --grad_accum 4
python e2e_occ/inference.py --checkpoint <ckpt.pth> --data_root <dataset_dir> --output <dir>

# 可视化
python dataset_viewer_v2/server.py --dataset <dir>   # http://localhost:8085/
```

### 数据集格式

`dataset_10k*/` 下按 `scene_XXXX_frame_XXXX` 组织样本：`calibration/{intrinsics,extrinsics}.json`（静态，Camera→Vehicle）、`images/*.dng`（8 相机 Bayer）、`depth/*.npy`、`occupancy/*.npy`（`(400,400,32)` uint8）、`flow/*.npy`、`flow_mask/*.npy`、`ego_pose/*.npy`（`(4,4)` float32，Vehicle→World）、`ego_motion/*.npy`、`train.txt`/`val.txt`/`test.txt`。

- 外参约定：`extrinsics = Camera→World`（含车辆绝对位姿）。
- `ego_motion` 计算：`inv(pose_t) @ pose_{t-1}`（上一帧坐标系 → 当前帧坐标系）。
- `e2e_occ/dataset.py` 加载相机参数的优先级：`camera_params/{id}.npz`（逐帧绝对外参，`dense_occupancy_collection` 格式，已弃用）→ `ego_pose/{id}.npy` + `calibration/`（当前 `occnetv3_data_generator` 格式）→ `calibration/` 静态标定退化（时序对齐失效，仅调试用）。
- DNG 位深（`main_collection.py --raw-bit-depth`，默认 12）从 `calibration/intrinsics.json` 顶层 `raw_bit_depth` 字段读取，不要在 `dataset.py` 里硬编码某个位深的归一化除数——这类硬编码曾经导致过采集侧改了位深、加载侧没跟着改的静默数值错误。

18 类语义标签（对齐 nuScenes）的完整定义见 `occnetv3_data_generator/config/occupancy_config.py`（权威来源）；`0: free` 在损失函数里权重必须 ≥ 1.0，`11: driveable_surface` 与 `13: sidewalk` 在可见性过滤中强制保留。

---

## 关键教训 / 踩过的坑

### 相机 / 世界坐标系方向

CARLA 中 Y 轴正方向是"右侧"，负方向是"左侧"（驾驶员视角）：
- `left_pillar` 必须用 Y **负值**（如 `(0.0, -1.1, 1.7)`），不是正值。
- 后视相机沿 X 负方向外延（如 `X=-2.7`）避免拍到车玻璃；侧后视相机 Y 绝对值要大，避免拍到车架。

完整的 8 相机布局表见 `occnetv3_data_generator/README.md`。

### 体素地面双重渲染（已修复）

**根因：** Map API 和 Static Mesh 会同时生成地面，导致"双层地面"、"浮空灰层"。

**修复位置：** `occnetv3_data_generator/processing/ground_truth_voxel_generator.py`
1. 跳过 `Roads/Sidewalks/Terrain/Ground/RoadLines` 这几类 `EnvironmentObjects` 的 Static Mesh 光栅化。
2. 地下填充统一使用 `Terrain(14)`，不复制地表材质（否则会出现灰色墙体）。
3. Actor BBox 不能覆盖已存在的地面体素（防止出现坑洞）。

诊断工具：`occnetv3_data_generator/diagnose_ground_layer.py`。

### DNG 格式加载

OpenCV 不支持 CFA（`PhotometricInterpretation=32803`）格式的 DNG。必须安装 `rawpy`（或 `Pillow + piexif`）；缺失时数据采集会自动降级为保存 `.npy`。

---

## 已发现但未导入的外部 Agent 配置

本机存在 `~/.codex/config.toml`（Codex CLI 配置）和 `~/.gemini/settings.json`（Gemini CLI 配置），本次未读取或导入。如果需要把其中的 MCP servers / 自定义指令迁移到 Claude Code，回复 `/import` 扫描可导入项。
