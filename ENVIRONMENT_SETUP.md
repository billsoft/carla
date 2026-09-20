# 从零搭建 CARLA 0.10 + UE5.5 开发环境（Windows）

> 本文是**线性操作手册（SOP）**：从一台空机器开始，按顺序走完即可得到一个
> 能编译、能调试、能打包发行的 CARLA + UE5 开发环境。
>
> 配套文档（不要重复维护，遇到问题按索引跳转）：
> - 踩坑/根因详解：[`CARLA_BUILD_NOTES.md`](./CARLA_BUILD_NOTES.md)
> - 编译完怎么跑、常见运行时报错：[`GUIDE.md`](./GUIDE.md)
> - 项目结构与 AI 协作约定：[`AGENTS.md`](./AGENTS.md)
> - 等距鱼眼 Bayer RAW 相机原理：[`EQUIDISTANT_BAYER_CAMERA_DEEP_DIVE.md`](./EQUIDISTANT_BAYER_CAMERA_DEEP_DIVE.md)
>
> 本文所有命令、路径、版本均对照 2026-09 本机实际环境核实过，不是官方文档的直译。

---

## 0. 最终形态与版本矩阵

这套环境由**两个独立 git 仓库**组成，缺一不可（关系详解见
[CARLA_BUILD_NOTES.md §1](./CARLA_BUILD_NOTES.md#1-两个仓库是什么关系)）：

```
D:\code\
├── carla\                         # CARLA 仓库（本仓库）：LibCarla + UE 插件 + PythonAPI + ML 研究代码
│   └── Unreal\CarlaUnreal\        #   一个空 UE 项目 + Carla 插件（不含引擎）
└── UnrealEngine5_carla\           # CARLA 专用 UE5 引擎 fork（引擎源码+编译产物，~211GB）
```

| 组件 | 版本 / 分支 | 本机实测 |
|---|---|---|
| CARLA | 0.10.0，fork `billsoft/carla` 的 `ue5-dev`，跟踪 `carla-simulator/carla` | `D:\code\carla` |
| UE5 引擎 | fork `CarlaUnreal/UnrealEngine` 的 `ue5-dev-carla`（UE 5.5）| `D:\code\UnrealEngine5_carla`，当前 commit `2ac0528831e0` |
| Visual Studio | 2026 (v18) Professional（官方脚本装的是 VS 2022，两者本机都在）| `C:\Program Files\Microsoft Visual Studio\18\Professional` |
| CMake | 4.2.0 | `C:\Program Files\CMake\bin` |
| Ninja | 1.12.1 | PATH |
| Python | 3.10.19，conda 环境 `carla`（编译/PythonAPI）、`deepsys`（训练）| `C:\ProgramData\anaconda3\envs\carla` |
| 环境变量 | `CARLA_UNREAL_ENGINE_PATH=D:\code\UnrealEngine5_carla` | 用户级持久变量 |

**磁盘/时间预算**：

| 项 | 占用 |
|---|---|
| UE5 引擎（含编译产物）| ~211GB |
| CARLA 工作区（编译后，不含打包产物）| ~100GB |
| 一次 Development 打包（含 zip）| 再加 ~45GB，可打包后清理 |
| 首次全程耗时（网络下载 + 引擎全量编译）| **3 小时以上** |

建议 D 盘预留 **300GB 以上**空闲再开始。

---

## 1. 前置准备

### 1.1 Epic EULA 授权（最容易被忽略的前置条件）

`https://github.com/CarlaUnreal/UnrealEngine.git` 是 Epic 按 CARLA 项目单独授权的
**私有 fork**。clone 前必须：

1. 注册/登录 GitHub 账号；
2. 用同一账号在 Epic Games 官网关联 GitHub 并接受 Unreal Engine EULA；
3. 浏览器登录 GitHub 后能打开上面的仓库页面（404 = 没权限，不是网络问题）。

这一步一次性生效，与 CARLA 仓库本身的权限无关。

### 1.2 安装工具链

- **Visual Studio**：安装「使用 C++ 的游戏开发」「使用 C++ 的桌面开发」工作负载，
  含 Windows 10/11 SDK、MSVC v143、Clang 工具集、CMake 组件。完整组件清单见
  [`Util/SetupUtils/InstallPrerequisites.bat`](./Util/SetupUtils/InstallPrerequisites.bat)
  里的 `visual_studio_components`。官方一键脚本默认装 VS 2022；本机用的是 VS 2026，
  构建脚本里已写死 v18 路径。
- **CMake ≥ 3.28（本机 4.2.0）**、**Ninja（本机 1.12.1）**、**Git**。
- **Python 3.10 + conda**：创建两个环境
  ```cmd
  conda create -n carla python=3.10 -y
  conda create -n deepsys python=3.10 -y
  pip install -r Util/SetupUtils/requirements.txt
  ```
  训练侧额外依赖（torch、rawpy、piexif 等）按 `occnetv3_data_generator/` 需要再装。

以上也可以直接让官方脚本代劳（它会装 VS2022/Ninja/Python 并 pip 装依赖）：
`CarlaSetup.bat --interactive`。但注意该脚本默认查找 VS 2022，用 VS 2026 时
后续编译走 [`BUILD_FINAL.bat`](./BUILD_FINAL.bat)（已写死 v18 路径）。

---

## 2. 获取 CARLA 仓库

```cmd
mkdir D:\code && cd /d D:\code
git clone -b ue5-dev https://github.com/<你的GitHub用户名>/carla.git
cd carla
git remote add upstream https://github.com/carla-simulator/carla.git
git fetch upstream ue5-dev
```

之后同步官方最新代码：`git fetch upstream ue5-dev` → `git merge upstream/ue5-dev`。
合并冲突极少（本地底层增强集中在鱼眼相机/Bayer RAW 几处），但合并后**必须重新走
第 7 步编译**才生效。

---

## 3. 下载游戏内容资源（carla-content）

CARLA 的地图/车辆资产不在主仓库，在 Bitbucket 独立仓库，clone 到项目内：

```cmd
cd D:\code\carla
mkdir Unreal\CarlaUnreal\Content
git -C Unreal\CarlaUnreal\Content clone -b ue5-dev ^
    https://bitbucket.org/carla-simulator/carla-content.git Carla
```

验证：`Unreal\CarlaUnreal\Content\Carla` 下应有 Maps、Static 等约 11 个分类目录
（`Content/Carla` 约 40GB，这是源资产，**永远不要删**）。

`CarlaSetup.bat` 会自动做这一步（目录已存在则跳过）。

---

## 4. 获取 UE5 引擎 fork

```cmd
cd D:\code
git clone -b ue5-dev-carla https://github.com/CarlaUnreal/UnrealEngine.git UnrealEngine5_carla
setx CARLA_UNREAL_ENGINE_PATH D:\code\UnrealEngine5_carla
```

`setx` 只对**新开**的终端生效；当前窗口再手动 `set CARLA_UNREAL_ENGINE_PATH=...`。
CMake 通过这个变量找引擎源码（`CMake/Options.cmake` 读取）。

---

## 5. 编译 UE5 引擎本体

CARLA 的 CMake 构建假设引擎的 Development Editor 二进制已经存在，首次必须先编引擎。
**在「x64 Native Tools Command Prompt」（或先 `call vcvars64.bat`）里执行：**

```cmd
cd /d D:\code\UnrealEngine5_carla

rem 5.1 下载引擎二进制依赖（GitDependencies，几十 GB，耗时最长的一步之一）
Setup.bat

rem 5.2 生成 UE5.sln / vcxproj
GenerateProjectFiles.bat

rem 5.3 编译 Development Editor 配置（等价于官方 CarlaSetup.bat 的做法）
msbuild Engine\Intermediate\ProjectFiles\UE5.vcxproj ^
    /property:Configuration="Development_Editor" /property:Platform="x64" /m
```

> 也可以用 UBT 直接编（二选一）：
> `Engine\Build\BatchFiles\Build.bat UE5Editor Win64 Development -WaitMutex -FromMsBuild`

**验证**：`Engine\Binaries\Win64\UnrealEditor.exe` 存在。
首次编译后引擎目录约 211GB；若只是更新引擎代码（`git pull`），增量重编只需
`Build.bat UE5Editor Win64 Development`，不用重跑 `Setup.bat`。

---

## 6. 应用引擎侧本地补丁（必做，不在 git 历史里）

本机对引擎有 **2 处手动补丁**（`git status` 可见的 `M` 文件，从未提交）。
重新 clone 引擎或 `git checkout .` 会丢失；**更新引擎前先 `git stash`，更新后
`git stash pop`**（实测无冲突）。这两处是 VS2026 下 CARLA 能编过的必要条件，
命令行 `/wd4723` 压不住头文件里的 `#pragma warning(error:...)`，只能改源码：

1. **`Engine/Source/Runtime/Core/Public/Windows/WindowsPlatformCompilerSetup.h`**
   把 4723 从 error 列表中去掉，在该 `#pragma` 下方另加一行：
   ```cpp
   // CARLA FIX: Removed 4723 from error list to allow VS 2026 compilation
   #pragma warning (disable: 4723)
   ```
   （不打的现象：UE5 ChaosVehicles 代码 C4723「potential divide by 0」被当错误，
   与你改不改 CARLA 代码无关。）

2. **`Engine/Plugins/Animation/RigLogic/Source/RigLogicLib/Private/dna/stream/FilteredBinaryInputArchive.cpp`**
   消除局部变量对外层作用域的 shadowing（严格警告级别下会报错）：
   `passingIndices` → `local_passingIndices`、`remappedIndices` →
   `local_remappedIndices`（文件内 lodNeuralNetworkMapping 和 lodSolverMapping
   两段各一处，全部引用一起改名）。

应用后 `git -C D:\code\UnrealEngine5_carla status --short` 应正好显示这两个 `M`。

---

## 7. 编译 CARLA（LibCarla + Carla UE 插件）

### 7.1 推荐：一键脚本（首次 / 改了 CMake 参数后）

**在真实 cmd 窗口里运行（双击或 cmd 中执行），不要通过 Git Bash 调 .bat**——
Git Bash 会把 `/c` 这类参数做路径转换导致脚本静默空跑（退出码 0 但什么都没编，
详见 [BUILD_NOTES §4.13](./CARLA_BUILD_NOTES.md#413-buildfinalbat-通过-claude-code-的-bash-工具调用会静默空跑)）：

```cmd
cd /d D:\code\carla
BUILD_FINAL.bat
```

脚本做的事（每条修复的来由见 [BUILD_NOTES §3](./CARLA_BUILD_NOTES.md#3-日常编译build_finalbat)）：
1. `call` VS2026 的 `vcvars64.bat`，验证 `cl.exe`；
2. `conda activate carla`（Python 3.10，路径 `C:\ProgramData\anaconda3\envs\carla`）；
3. 删除旧 `Build\CMakeCache.txt` 后重新 CMake configure，关键参数：
   `-DCARLA_UNREAL_ENGINE_PATH`、`/wd4723`、`/W1`（压下 V2X 等模块的既有警告）、
   静态库 `BUILD_SHARED_LIBS=OFF`；并自愈一个 zlib `zconf.h` 缺失问题；
4. 编译目标 **`carla-unreal-editor`**（UBT 依次构建 CarlaUnreal game + editor 两个
   target，产出插件 DLL）。

### 7.2 日常增量：PowerShell 手动命令（更快，不重新 configure）

只改了几个 `.cpp/.h` 时不必跑整个脚本（它每次删 CMakeCache）：

```powershell
& "C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat" | Out-Null
cd D:\code\carla
cmake --build Build --target carla-unreal-editor
```

实测纯增量约 **5–15 分钟**（ninja 对象缓存独立保留，删了 CMakeCache 也不影响）。

### 7.3 验证编译真的生效（重要）

**不要只看退出码 0**。对比编译前后目标 DLL 的修改时间，且要晚于你保存源码的时间：

```
Unreal\CarlaUnreal\Plugins\Carla\Binaries\Win64\UnrealEditor-Carla.dll
```

编译通过后**重启编辑器**（热重载对构造函数/头文件改动不可靠）。

### 7.4 编译卡住怎么办

接近完成时反复打印 `Delaying N processes ... memory pressure` 是 UBA 假死：
`taskkill /PID <ninja_pid> /F /T` 后重跑即可，增量缓存都在（[BUILD_NOTES §4.2](./CARLA_BUILD_NOTES.md#42-ubaunreal-build-accelerator内存压力假死)）。

---

## 8. 编译并安装 Python API

```powershell
& "C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat" | Out-Null
cd D:\code\carla
cmake --build Build --target carla-python-api-install
```

该 target（`PythonAPI/CMakeLists.txt`）先编 `carla-client` + Boost.Python 绑定，
再执行 **`pip install -e PythonAPI/carla`**（scikit-build-core 可编辑安装）。
安装结果在 conda 环境的 `Lib\site-packages\`：`carla.cp310-win_amd64.pyd` +
`carla-0.10.0.dist-info` + editable finder。

**只改了 Python 绑定（`PythonAPI/carla/src/*.cpp`）或 `LibCarla` 客户端代码时，
跑这个 target 即可，不需要重编 UE 插件。**

验证：

```cmd
conda activate carla
python -c "import carla; print(carla)"
```

---

## 9. 首次启动与连通验证

### 9.1 GUI 模式（开发调试默认）

```cmd
cd /d D:\code\carla
start_carla_server.bat
```

它用引擎目录的 `UnrealEditor.exe` 打开
`Unreal\CarlaUnreal\CarlaUnreal.uproject`。首次启动要编译着色器（5–15 分钟甚至更久）。

**编辑器打开后必须点绿色 Play（▶️）场景才会 tick**，Python 客户端才能连上。
不想手点可加 `-CarlaAutoPlay`（资产+着色器编完后自动进入 Play，30 分钟超时保护）：

```powershell
$exe = "D:\code\UnrealEngine5_carla\Engine\Binaries\Win64\UnrealEditor.exe"
$proj = "D:\code\carla\Unreal\CarlaUnreal\CarlaUnreal.uproject"
Start-Process -FilePath $exe -ArgumentList "`"$proj`" -CarlaAutoPlay -log -stdout -FullStdOutLogOutput" -PassThru
```

客户端轮询 `carla.Client('127.0.0.1', 2000).get_server_version()` 判断是否就绪，
不要用固定 sleep 猜（同机实测 2–20 分钟都出现过）。

### 9.2 端口

| 端口 | 用途 |
|---|---|
| 2000 | RPC（spawn/control 等同步命令）|
| 2001 | 传感器数据流（TCP streaming）|
| 8000 | Traffic Manager（同时只能有一个）|

### 9.3 画质档位警告

启动参数 `-quality-level=` 有 `Low/Medium/High/Epic` 四档（本仓库自实现的选择器，
默认 Epic）。**⚠️ 评估画质/清晰度的采集绝不能用 Low**——Low 档画面会糊成「水墨画」，
且与鱼眼/Bayer 管线无关（针孔相机一样糊），根因见
[BUILD_NOTES §4.14](./CARLA_BUILD_NOTES.md#414--quality-levellow-会把纹理糊成水墨画跟等距鱼眼bayer-raw-无关)。
生产无头脚本 [`start_carla_server_headless.bat`](./start_carla_server_headless.bat)
已固定 Epic。

### 9.4 验证清单

- [ ] `import carla` 成功；
- [ ] `test_connection.py` 输出 Connection Successful；
- [ ] `run_carla_examples.bat` 选手动驾驶（WASD，P 自动驾驶）能看到车和地图；
- [ ] 编辑器左上角有 FPS 计数。

常见连接问题（没点 Play、TM 端口占用、光照暗等）见 [`GUIDE.md`](./GUIDE.md)。

---

## 10. 打包独立发行版（部署到无编辑器的机器）

开发调试用编辑器即可；要给别的机器部署「只运行、不开发」的 CARLA 服务器，走打包：

```powershell
& "C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat" | Out-Null
cd D:\code\carla
cmake --build Build --target carla-unreal-package-development
```

- 配置可选：`carla-unreal-package`（默认随 `CMAKE_BUILD_TYPE`）、
  `-shipping`、`-debug`、`-debuggame`、`-development`、`-test`。
- 流程（`Unreal/CMakeLists.txt`）：先编 `carla-unreal-editor` + `carla-python-api`，
  再跑 UAT `BuildCookRun -cook -stage -archive -package`（烘培地图、暂存、归档），
  最后写 VERSION、清理多余文件、拷贝 PythonAPI/Tools，Development 默认**还会压一个 zip**。
- 只烘培部分地图可在 configure 时加 `-DCARLA_MAPS_TO_COOK=...`；Shipping 不压缩用
  `-DCARLA_UNREAL_PACKAGE_NO_COMPRESSION=ON`。

**产物**（本机 2025-12 打过一份 Development 包）：

```
Build\Package\Carla-0.10.0-Win64-Development\        (~30.5GB，zip 另 ~14.6GB)
├── Windows\
│   ├── CarlaUnreal.exe        ← 独立服务器可执行文件，目标机无需装 UE/编辑器
│   ├── CarlaUnreal\           （烘培后的 Content .pak）
│   └── Engine\
├── PythonAPI\                 （配套 whl/egg，客户端机器 pip 安装）
├── Tools\
├── VERSION / LICENSE / README / CHANGELOG
```

目标机部署：拷贝整个文件夹 → 安装 VC++ Redistributable（UAT `-prereqs` 会带上）→
`Windows\CarlaUnreal.exe -quality-level=Epic`（需要完全离屏渲染再加
`-RenderOffScreen`；注意 `-game -RenderOffScreen` 无头路线在本机实测崩溃，
见 [BUILD_NOTES §4.3](./CARLA_BUILD_NOTES.md#43-无头模式-game-renderoffscreen走不通)，
生产上以编辑器 + AutoPlay 或打 Development 包运行为准）。

**磁盘清理**：打包验证完若要回收空间，`Build/Package/StagedBuilds`、
`Unreal/CarlaUnreal/Saved/StagedBuilds`、`Saved/Cooked` 和 zip 都是可再生中间产物
（合计可达 ~90GB），删除不影响开发；源资产 `Content/Carla` 与
`Plugins/Carla/Binaries` 不能删。

---

## 11. 日常开发循环速查

| 改了什么 | 重编命令 | 之后要做什么 |
|---|---|---|
| UE 插件 C++（`Unreal/.../Sensor/` 等）| `cmake --build Build --target carla-unreal-editor` | 看 DLL 时间戳 → 重启编辑器 |
| `LibCarla/` 服务端 | 同上（UBT 会带上）| 重启编辑器 |
| Python 绑定 / LibCarla 客户端 | `cmake --build Build --target carla-python-api-install` | 重开 Python 进程 |
| Shader（`Plugins/Carla/Shaders/*.usf/.ush`）| 不涉及 C++，重启编辑器让其重编 shader | 重启编辑器 |
| ML 研究代码（`e2e_occ/`、`occnetv3_data_generator/`）| **不用编译** | 直接跑脚本（conda `deepsys`/`carla`）|

向引擎添加新传感器的 4 处联动改动见 [`AGENTS.md`](./AGENTS.md)「CARLA 引擎架构速览」。

---

## 12. 一页纸验收清单（从零搭完后逐项打勾）

- [ ] GitHub 账号已接受 Epic EULA，能打开 CarlaUnreal/UnrealEngine 私有仓库
- [ ] VS（C++ 游戏/桌面负载）、CMake、Ninja、Git、conda(python 3.10) 就绪
- [ ] `D:\code\carla` 在 `ue5-dev` 分支，`origin`/`upstream` 两个 remote 配置正确
- [ ] `Unreal\CarlaUnreal\Content\Carla` 资产已下载（~40GB）
- [ ] `D:\code\UnrealEngine5_carla` 在 `ue5-dev-carla` 分支
- [ ] `CARLA_UNREAL_ENGINE_PATH` 用户环境变量已设置
- [ ] 引擎 `Setup.bat` + `GenerateProjectFiles.bat` + Development Editor 编译完成，
      `UnrealEditor.exe` 存在
- [ ] 引擎 2 处本地补丁已应用（`git status` 可见两个 `M`）
- [ ] `BUILD_FINAL.bat` 全量编译成功，`UnrealEditor-Carla.dll` 时间戳更新
- [ ] `carla-python-api-install` 完成，`python -c "import carla"` 成功
- [ ] 编辑器启动 + Play（或 AutoPlay）后 `test_connection.py` 连接成功
- [ ] （如需部署）`carla-unreal-package-development` 打包成功，
      `CarlaUnreal.exe` 在目标机可运行
