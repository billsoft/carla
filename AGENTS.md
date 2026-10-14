# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## 重要提示 / Important Notes

**语言 / Language:** 请始终使用中文与用户沟通。Always communicate with the user in Chinese.

**代码修复原则:** 有问题的文件在原文件上修复，不要创建 `_fix`、`_fixed`、`_new` 等后缀的新文件。

**⚠️ 本目录是 UE5.8 移植 worktree（`D:\code\carla_ue58`，分支 `ue58-port`）**：引擎 `D:\code\UnrealEngine58_carla`、
MSVC 必须 14.44（`vcvars64.bat -vcvars_ver=14.44`）、Python 环境 `C:\Users\bills\.conda\envs\carla58`、Content 为
`ue58-dev-carla` 分支。与下文 5.5 内容冲突时以 `CLAUDE.md` 顶部的 5.5/5.8 对照表为准。

---

## 仓库概况

这个仓库包含两个几乎独立的代码库：

1. **CARLA 仿真器本体**（`LibCarla/`、`Unreal/`、`PythonAPI/`、`Ros2Native/` 等）— upstream `carla-simulator/carla` 的 `ue5-dev` 分支（UE5.5 版本），本仓库在其底层做了少量增强（见下方 "Bayer RAW" 一节）。
2. **自动驾驶感知研究代码**（仓库根目录下的 `occnetv3_data_generator/`、`e2e_occ/`、`dataset_viewer_v2/` 等）— 基于上面的 CARLA 采集数据、训练占用网络（Occupancy Network）的个人研究项目，与 CARLA 官方代码无关，不会被 upstream 同步覆盖。

两者对 Codex 的操作要求不同：改 CARLA 引擎代码要走 CMake/Ninja 全量重编译（很慢）；改 `occnetv3_data_generator/` 或 `e2e_occ/` 下的 Python 代码可以直接跑脚本，不需要重新编译。

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

构建产物验证：**必须对比重编译前后目标 DLL（如 `UnrealEditor-Carla.dll`）的
`LastWriteTime`，不要只看退出码 0**——`BUILD_FINAL.bat` 通过 Codex 的 Bash
工具（Git Bash/`cmd.exe`）调用时，已确认会静默空跑（打印一行 `cmd.exe` 交互式
banner 就退出，`CMakeCache.txt` 不变、没有任何 `cl.exe`/`ninja` 进程启动，退出码
却是 0），复现 3 次，具体分析和验证有效的 PowerShell 替代调用方式见
[`CARLA_BUILD_NOTES.md` §4.13](./CARLA_BUILD_NOTES.md#413-buildfinalbat-通过-Codex-的-bash-工具调用会静默空跑)。

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

初始环境搭建（首次，需要 GitHub 账号已关联 Epic Games 账号才能 clone UE5 fork，225GB+ 磁盘，3 小时+）：`CarlaSetup.bat`（Windows）/ `./CarlaSetup.sh --interactive`（Linux）。**从零开始的完整线性搭建/打包部署 SOP 见 `ENVIRONMENT_SETUP.md`（含引擎 Setup/编译、2 处引擎本地补丁、Python API 安装、打包 target，换机器/重新 clone 时优先读它）**。CARLA 仓库和 UE5 引擎 fork 是两个独立仓库、如何关联、编译时踩过的坑，完整记录见 `CARLA_BUILD_NOTES.md`。

### 启动服务器 / 排障

```cmd
start_carla_server.bat
# 或直接: CarlaUE5.exe -quality-level=Low -RenderOffScreen
```

编辑器打开后必须点击绿色 Play 按钮场景才会开始 tick，Python 客户端连接（port 2000）才不会超时；命令行加 `-CarlaAutoPlay` 可以让编辑器在资产/着色器编译完成后自动开始 Play，不需要手动点（`Carla.cpp::FCarlaModule::RegisterAutoPlayWatcher`，详见 `CARLA_BUILD_NOTES.md` 4.9 节）。常见问题（光照过暗、Traffic Manager port 8000 绑定冲突、`time-out while waiting for the simulator` 等）及修复方式见 `GUIDE.md` 的"问题排查"章节。

**`-quality-level=` 有 4 档：`Low`/`Medium`/`High`/`Epic`**（不传时默认 `Epic`），
是本仓库自己实现的选择器（`CarlaDeviceProfileSelectorModule.cpp`），不是 UE5 原版
Scalability 预设。**⚠️ `Low` 只适合快速冒烟测试，不要用它采集/对比任何要评估画质的
数据**：同位置同相机对照实测确认，Low 档位下画面明显糊成一片发灰发糊的"水墨画"
质感（尤其是带精细网格/文字的贴图，如脚手架、广告牌），Epic 档位下同一相机同一帧
清晰锐利，跟等距鱼眼相机、Bayer RAW 管线完全无关（同一现象在官方 `sensor.camera.rgb`
针孔相机上一模一样地出现）。**具体机制不是纹理 mip/流送精度**（`Low` 档的
`TextureQuality` 实际是满档，和 `Epic` 一样）——读源码确认更可能是
`ReflectionQuality=0`（Lumen 反射整个关掉）+ `r.Nanite.Streaming.PoolSize` 从
512MB 砍到 128MB（复杂几何体如脚手架被迫退化成低精度代理网格）共同导致，细节见
`CARLA_BUILD_NOTES.md` §4.14。正式无头生产启动脚本 `start_carla_server_headless.bat`
本身已经用的是 `-quality-level=Epic`，`main_collection.py`/`start_carla_server.bat` 都
不引用 `quality-level`——只有在手动直接起 `UnrealEditor.exe`/`CarlaUE5.exe` 做调试
时才可能被人为带上 `Low`。**任何用于评估图像清晰度/画质、或作为"最终验证"用途的
采集，启动前务必确认没有带 `-quality-level=Low`**。

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

**窄 FOV 相机（`front_main`/`front_narrow`）曾经天生比广角相机模糊，2026-08-29 已修复**：
根因是每个 cube face 的拍摄视场角写死 90°，与相机自己配置的 FOV 无关——FOV 越窄，
等于在这张固定分辨率的图里截取放大一小块，源纹素密度不变而输出像素密度需求越高，
必然模糊。修复不是放大纹理分辨率（那个方向试过 4 次，3 次卡死，已确认是错误方向），
而是让"仅需 1 张 face 就够"的单面相机直接按自己的真实 FOV 拍摄这张 face（复用已有
的 `CustomProjectionMatrix` 机制，只是角度从写死改成动态计算），纹理分配大小完全
不变。完整原理、代码改动点、验证过程见 `CARLA_BUILD_NOTES.md` 4.15 节。

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
- 参考 Tesla FSD 架构，默认配置 13.45M 参数（实测），输入 8 路等距投影 Bayer RAW `[B,8,1,960,1280]`，输出 `(400,400,32)` 18 类语义体素 + 3 维占用流。
- 多尺度（stride 8/16/32）可变形交叉注意力 + 粗 → 时序（局部可变形注意力 + GRU，Ego-Motion 对齐）→ 稀疏细两阶段解码；深度辅助监督。P1~P3 各项优化都有 `config.py` 开关，`train.py --set key=value` 可做消融。
- **坐标约定（2026-10 统一，见 `e2e_occ/geometry.py` 文件头）**：车辆/体素系 X前 Y右 Z上；相机系 OpenCV（x右 y下 z前）；网络吃的外参是 **Camera(OpenCV)→Vehicle 静态标定，不含车辆世界位姿**。所有等距正/反投影只在 `geometry.py` 实现一份，不要在别处另写投影公式。改几何相关代码后必须跑 `verify_network.py`（含真实标定下的几何回归测试和时序方向测试）。
- 2026-10 之前的 checkpoint 与当前结构不兼容（且旧版几何有 3 处错误，见 `MODULES_EXPLAINED.md` 第 11 节）。

```bash
conda activate deepsys
python e2e_occ/verify_network.py
python e2e_occ/train.py --data_root <dataset_dir> --batch_size 1 --epochs 100 --amp --grad_accum 4
python e2e_occ/inference.py --checkpoint <ckpt.pth> --data_root <dataset_dir> --output <dir>

# 可视化
python dataset_viewer_v2/server.py --dataset <dir>   # http://localhost:8085/
```

### 数据集格式

`dataset_10k*/` 下按 `scene_XXXX_frame_XXXX` 组织样本：`calibration/{intrinsics,extrinsics}.json`（静态，Camera→Vehicle）、`images/*.dng`（8 相机 Bayer）、`depth/*.npy`、`occupancy/*.npy`（`(400,400,32)` uint8）、`flow/*.npy`、`flow_mask/*.npy`、`ego_pose/*.npy`（`(4,4)` float32，Vehicle→World）、`ego_motion/*.npy`、`train.txt`/`val.txt`/`test.txt`。

- 外参约定：`calibration/extrinsics.json` 存 Camera(UE 轴 x前 y右 z上)→Vehicle；`e2e_occ/dataset.py` 转成 Camera(OpenCV 轴)→Vehicle 喂给网络。旋转矩阵必须按 UE `FRotationMatrix`（pitch 正值=抬头，即引擎实际渲染朝向）；旧版采集代码在 pitch/roll≠0 时符号与之相反，新版在 json 顶层写 `rotation_convention: ue_frotator` 标记，dataset 对没有标记的旧数据自动修正。
- **⚠️ 当前 LibCarla 的 `carla.Transform.get_matrix()` 与 UE 渲染不一致**：上游 #9751（2026-06）翻转了客户端 pitch/roll 符号，但 `Rotation→FRotator` 原样传递、UE 端未改。2026-10 起采集侧（真值生成器、`ego_pose`）改用 `occnetv3_data_generator/processing/transform_utils.py` 按 UE 公式自建矩阵和包围盒顶点，不再调用 `get_matrix()`/`get_world_vertices()`；此前采集的数据在坡道/倾斜路面上真值会歪（平地无影响）。上游 ue58-dev 已在 7c32b194a 撤销该翻转。
- `ego_motion` 计算：`inv(pose_t) @ pose_{t-1}`（上一帧车辆系 → 当前帧车辆系），`pose` 来自 `ego_pose/`。
- `depth/` 来自**针孔**深度相机（平面 z 深度），与鱼眼 RGB 像素不对应，dataset 里换算成沿鱼眼射线的距离后才作为监督目标。
- 时序样本不跨场景拼接；`camera_params/*.npz`（`dense_occupancy_collection` 格式）已不再支持。
- DNG 位深（`main_collection.py --raw-bit-depth`，默认 12）从 `calibration/intrinsics.json` 顶层 `raw_bit_depth` 字段读取，不要在 `dataset.py` 里硬编码某个位深的归一化除数——这类硬编码曾经导致过采集侧改了位深、加载侧没跟着改的静默数值错误。

18 类语义标签（对齐 nuScenes）的完整定义见 `occnetv3_data_generator/config/occupancy_config.py`（权威来源）；`0: free` 若将来在损失里加类别权重，其权重不得低于 1.0（当前 `e2e_occ/loss.py` 是无类别权重的 CE + Lovász）；`11: driveable_surface` 与 `13: sidewalk` 在可见性过滤中强制保留。

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

本机存在 `~/.codex/config.toml`（Codex CLI 配置）和 `~/.gemini/settings.json`（Gemini CLI 配置），本次未读取或导入。如果需要把其中的 MCP servers / 自定义指令迁移到 Codex，回复 `/import` 扫描可导入项。

---

<!-- 以下为上游 carla-simulator/carla ue58-dev 自带的 AGENTS.md，合并时原样保留 -->

# AGENTS.md

Guidance for any coding agent working in this repository.

## Skills

This repo ships a library of **60 vetted CARLA procedures** under
[`Skills/`](Skills/). Each is a directory containing a `SKILL.md` with YAML
frontmatter (`name`, `description`, `compatibility`), the step-by-step
procedure, and often `scripts/` and `references/` beside it.

**These skills are the source of truth** for building, packaging, cooking,
map/vehicle/asset import, running servers, scenarios and ROS 2. The raw
`Makefile` and `Util/BuildTools` scripts are a fallback for tasks no skill
covers.

### How to use them

1. Match the user's request against the catalogue below.
2. Read that skill's `SKILL.md` in full before acting — each encodes the
   failure modes that make the naive approach fail.
3. Run its `scripts/check_env.sh` (when `metadata.prerequisites` names one)
   before starting.
4. Follow the procedure rather than improvising; the gotchas sections exist
   because those mistakes have already cost someone a day.

Paths (`CARLA_ROOT` and friends) are configured on first need, not up front.
Never guess a path — several CARLA checkouts on one machine is normal and the
wrong one fails slowly. Ask the user, then record the answer.

### Optional: MCP server

[`Util/AgenticTools/`](Util/AgenticTools/) serves the same skills over MCP
(`list_skills`, `read_skill`, `check_prerequisites`, `get_config`,
`set_config`), which adds environment detection and persistent path config.
Project config is in [`.mcp.json`](.mcp.json); clients that read it get the
tools automatically. Agents without MCP should read `Skills/` directly — the
files are the source of truth either way.

## Skill catalogue

### `leaderboard`

- **`create-leaderboard-route`** — Creates and edits CARLA Leaderboard route files — placing waypoints and scenario trigger points interactively from the spectator camera....
- **`package-leaderboard-agent`** — Packages an agent for CARLA Leaderboard submission — builds the docker image with make_docker.sh (plain or ROS variant), checks the four required....
- **`read-leaderboard-results`** — Reads a CARLA Leaderboard results.json — per-route driving score, route completion, infraction penalty and the infraction lists — explains how the....
- **`run-leaderboard-evaluation`** — Runs the CARLA Leaderboard evaluator over a set of routes with your agent — routes/routes-subset/repetitions/track selection, checkpoint and....
- **`write-leaderboard-agent`** — Writes an autonomous agent for the CARLA Leaderboard — an AutonomousAgent subclass with setup/sensors/run_step/destroy, a sensor suite inside the....

### `python-api`

- **`bounding-boxes`** — Gets actor and level 3D bounding boxes and projects them into a camera image as 2D boxes — the dataset-annotation workflow.
- **`control-lights`** — Controls the map's light sources through the CARLA LightManager — turn street lamps, building/facade lights, vehicle lights or all of them on/off,....
- **`control-spectator`** — Moves the CARLA spectator camera, frames an actor, or follows one live — third-person/chase, first-person/driver, top-down, or front views,....
- **`control-traffic`** — Tunes the CARLA Traffic Manager that drives autopilot vehicles — global and per-vehicle speed, following distance, running lights/signs,....
- **`control-traffic-lights`** — Controls the traffic-light actors on a running CARLA server — set a light green/red/yellow/off, freeze all lights (optionally all-green), set....
- **`control-vehicle`** — Directly drives a specific vehicle on a running CARLA server — raw VehicleControl (throttle/steer/brake/reverse/hand-brake), Ackermann speed....
- **`create-sensor`** — Spawns and attaches sensors on a running CARLA server — cameras (rgb/depth/semantic/instance/optical-flow/normals), lidar, radar, IMU, GNSS,....
- **`debug-draw`** — Draws debug overlays in a running CARLA world via world.debug — points, lines, arrows, bounding boxes, and text strings, each with a colour and....
- **`debug-navmesh`** — Visualises and validates a map's pedestrian navigation mesh by sampling walkable locations (get_random_location_from_navigation) and drawing them....
- **`load-map`** — Loads, reloads, and reshapes the map on a running CARLA server via the Python API — get_available_maps, load_world and reload_world (with or....
- **`map-waypoints`** — Explores a CARLA map's road network through the Map/Waypoint API — a natural-language rundown of the whole map ("give me the topology"), drawing....
- **`navigate-to`** — Drives a vehicle to a destination autonomously using CARLA's navigation agents (BasicAgent/BehaviorAgent/ConstantVelocityAgent), or plans a route....
- **`query-recording`** — Inspects a recorded CARLA .log without replaying it — show_recorder_file_info for the header/actors/frames, collision queries between actor....
- **`read-sensor`** — Listens to a CARLA sensor and either saves its data to files, shows it live in a window, prints a one-shot summary, or (ros-info) reports the....
- **`record-simulation`** — Records a running CARLA simulation to a .log file via start_recorder/stop_recorder for later deterministic replay — capturing actor spawns,....
- **`replay-recording`** — Replays a CARLA .log recorded by the recorder — the whole run or a time window, at any speed, following a chosen actor, optionally regenerating....
- **`set-weather`** — Reads and sets the weather on a running CARLA server via get_weather/set_weather, and turns natural-language weather requests ("heavy rain at....
- **`set-world-settings`** — Reads and changes a running CARLA server's WorldSettings via apply_settings — synchronous vs asynchronous mode, fixed_delta_seconds (fps), physics....
- **`spawn-vehicles`** — Spawns vehicles on a running CARLA server and destroys them — scattered across the map's spawn points on Traffic Manager autopilot, a row queued....
- **`spawn-walkers`** — Spawns pedestrians that wander autonomously via WalkerAIController, and destroys them in the correct order.
- **`telemetry`** — Reads a specific actor's live telemetry from a running CARLA server — location and rotation, velocity and speed, acceleration, angular velocity,....
- **`toggle-env-objects`** — Lists and toggles a map's static environment objects via get_environment_objects / enable_environment_objects — hide or show buildings,....
- **`world-data`** — Queries the live CARLA world and resolves an entity description into a concrete actor id — find actors by type, role, or colour, or the single one....

### `ros2`

- **`add-ros-message-type`** — Adds a new ROS 2 message type to CARLA's native ROS 2 interface — the hand-written POD struct under LibCarla/source/carla/ros2/types/msg, its....
- **`add-ros-publisher`** — Adds a new topic to CARLA's native ROS 2 interface — a publisher (BasePublisher + PublisherImpl, topic suffix, QoS, registration in....
- **`visualize-ros-rviz`** — Inspects and visualises a ROS-2-enabled CARLA server from Docker containers, with no local ROS 2 install — list/echo/rate-check topics (the "are....

### `scenario-runner`

- **`analyze-scenario-results`** — Reads and summarises ScenarioRunner output — the criteria pass/fail tables from --output/--file, the machine-readable --json and --junit result....
- **`create-scenario`** — Writes a new ScenarioRunner scenario — a Python class deriving from BasicScenario with its behaviour tree assembled from srunner atomic behaviours....
- **`run-openscenario`** — Runs standard-format scenarios through ScenarioRunner — ASAM OpenSCENARIO 1.x (.xosc, via --openscenario, with catalogs and ParameterDeclaration....
- **`run-route-scenario`** — Runs a route through ScenarioRunner's RouteScenario mode (--route / --route-id) with an autonomous agent driving the ego — the same mechanism the....
- **`run-scenario`** — Runs a single ScenarioRunner Python scenario (or a whole group) against a running CARLA server — FollowLeadingVehicle, ControlLoss,....

### `scenic`

- **`create-scenic-scenario`** — Writes a new Scenic scenario (.scenic) from a natural-language description — "a car cuts in from the right lane", "a pedestrian crosses in front....
- **`run-scenic-scenario`** — Runs Scenic probabilistic scenarios against a running CARLA server with the `scenic FILE --simulate --2d` CLI — the carlaChallenge1-10 traffic....

### `setup`

- **`check-feature-support`** — Answers "can CARLA do X here, and is there a vetted procedure for it" before you promise anything, and detects client/server/tree version skew — a....
- **`download-carla`** — Downloads CARLA so nothing has to be fetched by hand — an official release package for any published version (resolved live from the GitHub....
- **`install-leaderboard`** — Installs the CARLA Autonomous Driving Leaderboard (leaderboard.carla.org) at a chosen version — 1.0, 2.0 or 2.1 — together with the exact....
- **`install-python-api`** — Installs the CARLA Python client (the `carla` package) into a chosen interpreter so the other python-api skills work — from a wheel bundled with....
- **`install-scenario-runner`** — Installs CARLA's ScenarioRunner (the official scenario engine) and picks the branch that matches the CARLA in use — master for 0.9.14-0.9.16....
- **`install-scenic`** — Installs Scenic, the probabilistic scenario description language, for the CARLA in use — pip installing it into the same interpreter that holds....

### `ue4`

- **`build-carla-ue4`** — Builds CARLA (branch ue4-dev, Unreal Engine 4.26) from source on Linux end-to-end — UE4 fork, the Carla server (editor C++ modules), the LibCarla....
- **`import-carla-map`** — Imports a custom RoadRunner map (FBX geometry + XODR OpenDRIVE) into a CARLA source build as a drivable map — reads the map from the directory the....
- **`import-carla-prop`** — Imports FBX meshes into a CARLA source build as spawnable static.prop.* blueprints — pass a directory whose subdirectories name the semantic tags,....
- **`import-carla-vehicle`** — Imports an FBX vehicle into a CARLA source build as a spawnable vehicle.<make>.<model> — checks the FBX carries CARLA's 4-wheel bone rig, imports....
- **`import-carla-walker`** — Imports an FBX pedestrian into a CARLA source build as a spawnable walker.pedestrian.NNNN — checks the FBX is skinned to CARLA's GEN3 rig, imports....
- **`package-carla-ue4`** — Cooks and packages CarlaUE4 into distributable tarballs under Dist/ — either the full simulator or a standalone asset package of selected maps and....
- **`run-carla-server`** — Launches a CARLA RPC server a carla.Client can connect to, working out for itself what you pointed it at — a downloaded/extracted release, a....

### `ue5`

- **`check-ue5-limitations`** — Tells you what CARLA on UE 5.5 (branch ue5-dev) cannot do that UE 5.8 (ue58-dev) can, so you do not attempt an unsupported workflow there — no....

### `ue58`

- **`build-carla-ue58`** — Builds CARLA on Unreal Engine 5.8 from the ue58-dev branch using its CMake build system — configure a preset, then build the carla-unreal /....
- **`carla-vehicle-authoring`** — Create, migrate, improve, or debug detailed vehicle assets for CARLA using Blender and Unreal, including reference-based geometry, materials,....
- **`import-carla-ue58-map`** — Gets a custom map into CARLA on UE 5.8 — the four distinct routes and which one to pick — via Util/Tools/Import.py for FBX+OpenDRIVE assets into a....
- **`import-carla-ue58-prop`** — Imports FBX meshes into CARLA on UE 5.8 as spawnable props — drives the editor headlessly to import the mesh, derives the prop size from its....
- **`import-carla-ue58-walker`** — Imports a pedestrian into CARLA on UE 5.8 as a spawnable, animating walker — imports the skinned FBX bound to CARLA's shared pedestrian skeleton,....
- **`package-carla-ue58`** — Cooks and packages a shippable CARLA server from a UE 5.8 ue58-dev checkout via the CMake `package` targets (shipping/development/debug/test),....
- **`run-autoware-ue58`** — Runs the Autoware autonomous-driving stack against CARLA on UE 5.8 over CARLA's native ROS 2 interface — no bridge process.
- **`run-carla-ue58-server`** — Starts a CARLA UE 5.8 server from a ue58-dev tree — the editor binary in -game mode (the working headless server), the packaged shipping build, or....

## Repository conventions

- `LibCarla/` — the C++ client/server library. `PythonAPI/` — bindings, docs
  (`PythonAPI/docs/*.yml`) and examples.
- `Docs/python_api.md` is **generated** from `PythonAPI/docs/*.yml` by
  `PythonAPI/docs/doc_gen.py`. Never edit it directly; edit the YAML and
  regenerate.
- Smoke tests live in `PythonAPI/test/smoke/` and need a running server on
  port 3654.
- Add a `CHANGELOG.md` entry under `## LATEST Changes` for user-visible changes.
