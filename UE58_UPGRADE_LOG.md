# CARLA + UE 从 5.5 升级到 5.8：编译 / 打包 / 部署全过程

> 时间：2026-10-13 ～ 2026-10-14
> 结果：5.8 引擎、CARLA 插件、Python API 编译通过；打包成 Development 运行包部署到服务器 81 常驻运行；
> 修好鱼眼相机曝光（6 面共享自动曝光、无接缝）和 rt_lens 的 DLSS 降噪；从本机连 81 实采 200 帧数据验证通过。
> 5.8 代码在 `github.com/billsoft/carla` 的 `main` 分支（默认分支）。
> 渲染/真实感相关的问题清单见 [UE58_RENDERING_ISSUES.md](UE58_RENDERING_ISSUES.md)。

---

## 0. 现在的布局

```
GitHub: billsoft/carla
├── main      ← UE5.8（默认分支，今后在这里开发）
└── ue5-dev   ← UE5.5 存档

本机（192.168.100.182）
D:\code
├── carla\                   ← git 仓库本体 + 5.5 工作目录（分支 ue5-dev）
│   └── .git\                ← ⚠️ 5.8 目录的 git 数据也存在这里，不能整个删 carla\
├── carla_ue58\              ← 5.8 工作目录（git worktree，分支 main）
│   ├── Build\               ← CMake 构建目录（LibCarla、依赖库）
│   │   └── Package\         ← 目录链接 → C:\carla_build58\Package（打包输出）
│   ├── Unreal\CarlaUnreal\Content\Carla\   ← 5.8 Content（独立 git 仓库，约 54GB）
│   ├── Unreal\CarlaUnreal\Saved\Cooked\    ← 目录链接 → C:\carla_build58\Cooked（cook 中间结果）
│   └── Util\EnginePatches\  ← 我们对 UE 引擎源码的补丁（引擎目录不在 carla 仓库里，补丁存这里）
├── UnrealEngine5_carla\     ← 5.5 引擎（约 271GB，5.8 稳定后可删）
└── UnrealEngine58_carla\    ← 5.8 引擎（含本地补丁，见 2.10）
Python：C:\ProgramData\anaconda3\envs\carla（5.5 客户端） / C:\Users\bills\.conda\envs\carla58（5.8 客户端）

服务器 81（192.168.100.81，XTWY20240424，i9-13900K / 64GB / RTX 4090）
D:\carla-server\             ← 当前运行的 5.8 打包版（CarlaUnreal.exe）
D:\carla-server.bak\         ← 上一版（回滚用）
D:\carla-update\             ← 上传的 zip 暂存
D:\carla-launch\             ← start_carla.bat（服务端启动参数）、carla_viewer.py、carla 客户端 wheel
C:\ProgramData\miniconda3\envs\carla58   ← 81 上的 Python 3.10（跑预览窗口用）
计划任务：CARLA-Server（SYSTEM，开机自启，无窗口 GPU 渲染）、CARLA-Viewer（Administrator 登录后在桌面开预览窗口）
```

为什么是两个工作目录：升级要连续编译十几个小时，期间 5.5 必须一直可用，所以用 `git worktree` 从同一个仓库
拉出第二个目录专门做 5.8。

为什么服务端放 81：本机同时跑 UE 编辑器 + 8 路相机 + 路径追踪会把显存撑爆（本机还跑着 Docker 等）。
81 只跑服务端，本机只跑 Python 客户端（采集、训练、调试），连接时加 `--host 192.168.100.81`。

---

## 1. 整体流程

| # | 步骤 | 耗时（约） | 结果 |
|---|---|---|---|
| 1 | 克隆 UE5.8 引擎（`ue58-dev-carla` 分支） | 网络决定，有重试 | `D:\code\UnrealEngine58_carla` |
| 2 | 引擎 `Setup.bat`（下载引擎依赖） | 较长 | 完成 |
| 3 | 建 worktree，合并上游 `ue58-dev`，解决冲突 | 引擎下载期间并行完成 | 15 处冲突按 5.8 新 API 改写 |
| 4 | 编译 UE5.8 引擎 | 约 89 分钟（8552 个任务） | `UnrealEditor.exe` |
| 5 | 下载 5.8 Content | 约 9 小时 | 53.6GB，534,549 个文件 |
| 6 | 编译 CARLA 5.8（LibCarla + 插件） | 首次失败 1 次，修复后约 3 小时 | `UnrealEditor-Carla.dll` |
| 7 | 编译 5.8 Python API，建 `carla58` 环境 | 几分钟 | `carla-0.10.0-cp310` wheel |
| 8 | 同步 5.5 上未提交的修复 | 几分钟 | 12 个文件 |
| 9 | 本机编辑器冒烟测试 | 首次编着色器较久 | 2 帧数据正常 |
| 10 | 提交、推送，GitHub 默认分支改为 `main` | — | — |
| 11 | 打包 Development 运行包 | 首次约 2.5 小时，失败 2 次 | zip 约 13.5GB |
| 12 | 部署到 81、常驻运行 | 上传 + 解压约 15 分钟 | 81 真 GPU 渲染，本机连上 |
| 13 | 真实感调参（硬件光追 Lumen 等） | — | 启动参数生效，性能不降 |
| 14 | 修 rt_lens 降噪（DLSS-RR） | — | 降噪生效 |
| 15 | 修鱼眼曝光（引擎补丁 + CARLA） | 重新打包约 1 小时 | 6 面共享自动曝光、无接缝 |
| 16 | 81 / 本机预览窗口，200 帧实采 | 采集约 15 分钟 | 数据校验正常 |

---

## 2. 各步骤细节

### 2.1 获取 UE5.8 引擎

- 分支：CARLA 的 UE fork 的 `ue58-dev-carla`（需要 GitHub 账号已关联 Epic 账号）。
- **浅克隆**（`fetch --depth 1`）：完整历史几十 GB，浅克隆只拿最新一版。
- 直接 `git clone` 因网络中断失败，改成 `git init` + `fetch --depth 1` + 失败自动重试才成功。
- 再运行引擎根目录的 `Setup.bat` 下载二进制依赖（用完整路径调用，PowerShell 的 `Set-Location` 不会传给 cmd）。

### 2.2 合并上游 CARLA 代码（ue58-dev）

我们在 5.5 上的改动（Bayer RAW 采集、等距鱼眼、`-CarlaAutoPlay`、红绿灯模型等）和上游 5.8 撞了 **15 处冲突**。
原则：**以上游 5.8 新结构为主，把我们的功能嫁接上去**。主要几处：

| 文件 | 冲突原因 | 处理 |
|---|---|---|
| `SceneCaptureCamera.cpp/.h` | 5.8 改用 `FSensorCaptureContext` 打包帧号/时间戳/位姿 | 我们的 HDR/Bayer 路径改用新的 CaptureContext 发数据 |
| `SceneCaptureCamera_WideAngleLens.*` | 5.8 改为批量 GPU 回读 | 鱼眼 HDR 路径接到新的回读池 |
| `CameraModelUtil.*` | 5.8 新增 LUT1D / BrownConrady 镜头模型 | 用上游结构，保留"窄 FOV 单面按真实 FOV 拍"的修复 |
| `ActorBlueprintFunctionLibrary.cpp` | 5.8 自带归一化的 `fx/fy/cx/cy` 属性 | 删掉我们的像素 `cx/cy` 属性改用上游的；`camera_manager.py` 相应换算 |
| `TrafficLightManager.*` | 上游新增从配置采纳模型 | 我们的默认模型改为"只补缺失项" |
| `ImageUtil.cpp` | 上游重写回读 | 用上游版本，保留 Bayer 量化 `+0.5` 四舍五入 |
| `Carla.cpp` | 都加了注册钩子 | 两边都保留 |

### 2.3 编译 UE5.8 引擎

- VS2026 的 `vcvars64.bat` → `GenerateProjectFiles.bat` → `msbuild UE5.vcxproj`（Development_Editor）。
- **坑：内存不足被强杀**。UBA 在内存快满时会杀掉编译进程再重试，第一次被杀 1259 次。
  **解决**：`Engine\Saved\UnrealBuildTool\BuildConfiguration.xml` 限制并行：
  ```xml
  <MaxParallelActions>14</MaxParallelActions>
  <MemoryPerActionBytes>4294967296</MemoryPerActionBytes>
  ```
  之后约 89 分钟编完。

### 2.4 下载 5.8 Content

Content 在 Bitbucket `carla-content` 的 `ue58-dev-carla` 分支（Git LFS）。和 5.5 几乎不共享（524,617/534,513 个对象是新的），
共 53.6GB；正常 clone 要"LFS 缓存 + 检出"两份约 107GB，D 盘放不下。

**做法**：`GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1` 只拿占位文件 → 按目录分 77 批 `git lfs pull` →
每批检出后删 `.git\lfs\objects` 缓存（检出的文件就是完整资产），峰值只有"已检出 + 一批"，最终约 54GB。

| 现象 | 原因 | 解决 |
|---|---|---|
| clone 报 `active post-checkout hook found` | git 2.45.1 把 git-lfs 自带 hook 误判成攻击 | 只对这条命令设 `GIT_CLONE_PROTECTION_ACTIVE=false`（2.45.2 已修复） |
| 删不掉旧目录、clone 反复失败 | 停任务只停了外层，脚本还在后台跑，两个脚本抢一个目录 | 查清整个进程树全部结束再重来 |
| 缓存没删，D 盘越来越少 | Git Bash 下 `cmd /c rd` 的引号失效，静默失败 | 另起 PowerShell 清理器，每完成一批删之前的缓存 |
| 最后一批要 8 小时 | `__ExternalActors__` 有 48.9 万个小文件，瓶颈是请求延迟 | `lfs.concurrenttransfers` 从 8 调到 64，速度翻倍多 |

### 2.5 编译 CARLA 5.8

先 CMake 配置（下载 boost、eigen、zlib 等），再用 UBT 编 `CarlaUnreal`（游戏版）和 `CarlaUnrealEditor`（编辑器版）。

| 问题 | 原因 | 解决 |
|---|---|---|
| `BUILD_FINAL.bat` 一行都没执行 | 仓库里的 `.bat` 是 LF 换行，cmd 解析错乱 | 用 CRLF 副本运行 |
| eigen 下载失败 | 代理连 gitlab 断开 | 复制 5.5 已下载的 eigen 源码（同为 3.4.0） |
| `Missing CARLA content` | CMake 要求 `Content\Carla\.git` 存在 | 先下 Content |
| **链接 `LNK2019 __std_regex_transform_primary_char`** | CMake 侧默认 MSVC **14.50** 编 LibCarla，UE5.8 的 UBT 固定用 **14.44** 链接 | `vcvars64.bat -vcvars_ver=14.44`，已写进 `BUILD_FINAL.bat` 和 `build_package_development.bat` |

验证看产物 DLL 时间戳（`UnrealEditor-Carla.dll` 为本次新生成），不只看退出码。

### 2.6 Python API 与 conda 环境

- `cmake --build Build --target carla-python-api` → `Build\PythonAPI\dist\carla-0.10.0-cp310-cp310-win_amd64.whl`。
- 从 `carla` 环境克隆出 `carla58`，装 5.8 的 wheel；5.5 的 `carla` 环境没动。
- 5.8 客户端的 `get_matrix()` 已与 UE 渲染方向一致（上游 7c32b194a 撤销了 5.5 的符号翻转），与 `transform_utils.py` 只差 3e-8。
- **客户端 wheel 必须和服务端出自同一次编译**（自定义传感器数据格式）；81 上的 `carla58` 环境装的也是这个 wheel。

### 2.7 同步 5.5 上的未提交修复

12 个文件（坡道真值 `transform_utils.py`、rear 相机向下 8°、文档）。两边换行符不同（5.5 LF / worktree CRLF），
对 5.8 没改过的文件整文件复制；`AGENTS.md` 用"5.5 版本 + 上游追加段"重拼。

### 2.8 本机编辑器冒烟测试

| 问题 | 原因 | 解决 |
|---|---|---|
| 端口 2000 已开，但 `get_world()` 超时 | 首次启动 DDC 为空，游戏线程在编纹理/着色器 | 等它编完；以"连续 3 次能取到世界且帧号前进"判断就绪 |
| 编辑器莫名退出（`ConsoleCtrl RequestExit`） | 带 `-log` 启动时编辑器挂在工具控制台上，一条命令结束就把它带走 | `Invoke-CimMethod Win32_Process Create` 脱离启动，日志用 `-abslog` |

### 2.9 提交与 GitHub

- `e1bc2838b` 合并提交；`e99d7c9fa` 坡道真值 / rear −8° / 文档。
- `ue58-port` 改名为 `main` 推到 GitHub，网页设置里默认分支改为 `main`，删除远端 `ue58-port`。
- 5.5 那 12 个修复也在 `ue5-dev` 上提交了一份（`0bbc52823`，只在本地，未推送）。
- ⚠️ 2.10 之后的改动（打包修复、引擎补丁、曝光、部署脚本、预览窗口等）截至 2026-10-14 **都还没提交**。

### 2.10 引擎补丁（UE5.8 源码本地修改）

引擎目录 `D:\code\UnrealEngine58_carla` 是单独的 git 仓库（浅克隆），**不在 carla 仓库里**；对它的修改以补丁形式存在
`Util\EnginePatches\`，重新克隆引擎后要手动打上并重编：

```powershell
git -C D:\code\UnrealEngine58_carla apply D:\code\carla_ue58\Util\EnginePatches\ue58-scene-capture-shared-exposure.patch
```

| 补丁 | 内容 | 为什么 |
|---|---|---|
| `ue58-scene-capture-shared-exposure.patch` | 新增 `Engine/Public/SceneCaptureExposureSharing.h` + `Engine/Private/SceneCaptureExposureSharing.cpp`（"曝光来源"注册表）；`Renderer/Private/SceneCaptureRendering.cpp` 的 `SetupViewFamilyForSceneCapture` 里，有注册来源的 capture 把 `ViewInitOptions.ExposureSceneViewStateInterface` 指到来源 capture 的 view state | 让鱼眼的 6 个 cube 面共用面 0 的自动曝光（见 2.14）。引擎本身已有跨视图共享曝光的机制（Movie Render Queue 全景渲染在用），只是 scene capture 没接上。故意不改 `SceneCaptureComponent.h`（被大量引用，改了会触发引擎近乎全量重编），只新增文件，改动后增量编译很快 |

> 另：`Engine\Saved\UnrealBuildTool\BuildConfiguration.xml` 的并行限制（2.3）也是引擎目录下的本地文件。

### 2.11 打包 Development 运行包

服务器跑的是**打包版**（`CarlaUnreal.exe -RenderOffScreen`），不是编辑器。脚本：`build_package_development.bat`
（5.8 引擎 + MSVC 14.44 + 在脚本目录编译）。注意仓库里的 `.bat` 是 LF 换行，从 cmd 跑前转成 CRLF 副本。

**磁盘**：cook 中间结果 + 暂存 + 打包目录 + zip 要 60GB 以上，本机 D 盘不够。用**目录链接**把输出转到 C 盘，不改 CMake：
```powershell
New-Item -ItemType Junction -Path D:\code\carla_ue58\Build\Package -Target C:\carla_build58\Package
New-Item -ItemType Junction -Path D:\code\carla_ue58\Unreal\CarlaUnreal\Saved\Cooked -Target C:\carla_build58\Cooked
```
两个目录本来就被 `.gitignore` 忽略。打包完 `C:\carla_build58\Package\StagedBuilds`（与打包目录同内容，约 16GB）可删，
`Cooked`（约 28GB）留着能让下次打包快很多。

**打包阶段**：BUILD（编游戏版）→ COOK（转资源，首次约 1.5 小时，主要编着色器）→ STAGE → PACKAGE → ARCHIVE →
CARLA 后处理（VERSION、清理、附加文件、拷 DLSS DLL、压缩 zip）。

| 问题 | 原因 | 解决 |
|---|---|---|
| cook 失败 `Error_UnknownCookFailure`，唯一错误是 `BP_TLOpenDrive_v02` 蓝图编译错误（`ST_TrafficLightProps` 数组连到静态网格体数组） | 5.8 Content 里这个蓝图自身是坏的（结构体改了没重存）；我们 5.5 时期在 `DefaultGame.ini` 把整个 `Blueprints/TrafficLight` 目录设为常驻 cook，把它带进来了。之所以要常驻 cook：我们把 `TrafficLightManager` 的默认红绿灯/标志从构造函数硬引用改成了运行时懒加载（避开 5.5 启动时的重入编译崩溃），cooker 看不到这些引用 | 保留 `TrafficLight`、新增 `Static/TrafficSign` 两个常驻 cook 目录，再用 AssetManager 规则**单独排除** v02：`PrimaryAssetType="CarlaNeverCook"`、`bIsEditorOnly=True`、`SpecificAssets=(BP_TLOpenDrive_v02)`、`CookRule=NeverCook`（`DefaultGame.ini`） |
| 第二次尝试：用 AssetManager `AlwaysCook` 精确列出 20 个蓝图，报 16 个 `Registered PrimaryAssetId ... does not match object's real id` | 非 editor-only 类型保存时会校验对象自己的 `GetPrimaryAssetId()`，普通 Actor 蓝图返回空 | 放弃这条路（见上一行的最终方案）。依据 `AssetManager.cpp` 的 `ModifyCook`/`OnObjectPreSave` 与 `CookOnTheFlyServer.cpp` 的 `NeverCookPackageList` |
| `LogZenStore: Zen project creation FAILED` | Zen 存储服务没起（项目 `bUseZenStore=False`） | 只是警告，退回普通文件 cook；但 LegacyIterative 增量 cook 不可靠，改了 C++ 序列化后最好全量 cook |
| 新增的 DLSS 拷贝步骤失败（目标不是目录） | 包里的目录叫 `Windows`，不是 `${UE_SYSTEM_NAME}`（=`Win64`） | `Unreal/CMakeLists.txt` 写死 `Windows/CarlaUnreal/Binaries/Win64/`；那次手动补跑了拷 DLL 和 `Compress.cmake` |

打包产物：`Build\Package\Carla-0.10.0-Win64-Development\`（约 16GB）和同名 `.zip`（约 13.5GB）。

### 2.12 部署到服务器 81

**SSH 公钥免密**（不用密码登录）。公钥失效时用户自己输两次密码重装：
```powershell
scp $env:USERPROFILE\.ssh\id_ed25519.pub administrator@192.168.100.81:C:/ProgramData/ssh/new_key.pub
ssh administrator@192.168.100.81 "type C:\ProgramData\ssh\new_key.pub >> C:\ProgramData\ssh\administrators_authorized_keys && icacls C:\ProgramData\ssh\administrators_authorized_keys /inheritance:r /grant Administrators:F /grant SYSTEM:F && del C:\ProgramData\ssh\new_key.pub"
```
（不要写成一行嵌套引号的 PowerShell：会被本地和远端各解析一次而失败。）

**`remote\carla_remote.ps1`**：`status / start / stop / restart / logs / deploy [-SkipUpload] / test / firewall / viewer`。
`deploy` = scp 上传 zip → 停服务 → `tar.exe` 解压（比 `Expand-Archive` 快得多）→ 旧版改名 `.bak` → 启动。

| 问题 | 原因 | 解决 |
|---|---|---|
| 首次部署第 3 步中断 | Windows PowerShell 5.1 下原生命令输出任何 stderr（如 `schtasks /End` 报"任务不存在"）都会在 `ErrorActionPreference=Stop` 时变成终止错误 | `Invoke-RemotePs` 里临时改 `Continue`、只看退出码；`schtasks /End` 改 `cmd /c ... >nul 2>&1`；加 `-SkipUpload` 续跑不必重传 13GB |
| 改了启动参数却还是旧参数在跑 | `schtasks /TR` 最长 261 字符，长参数让 `/Create` 失败，`/Run` 跑的是旧注册的命令 | 计划任务改为运行 `D:\carla-launch\start_carla.bat`（参数写在 bat 里，放在部署目录外，重新部署不会被删） |
| 服务器重启后服务不在 | 任务原为 ONCE | 改为 `ONSTART` 开机自启 |
| 担心防火墙拦 2000/2001 | — | 实测本机能直连，暂不需要；需要时用户执行 `-Action firewall` |
| 服务端常驻，状态跨连接保留（中断的脚本留下同步模式、天气、车辆、传感器） | — | 脚本退出时恢复设置；另有 `remote\reset_carla_world.py --host 192.168.100.81` 一键复位 |

验证 81 真用 GPU：`nvidia-smi` 进程里有 `CarlaUnreal.exe`，日志 `Chosen D3D12 Adapter Id = 0`（RTX 4090）而不是 Microsoft Basic Render Driver。

### 2.13 真实感渲染设置（服务端启动参数，不用重新打包）

Development 包支持 `-ini:` 覆盖配置，参数在 `carla_remote.ps1` 的 `$CarlaArgs`（写入 `start_carla.bat`）：
```
-carla-server -quality-level=Epic -RenderOffScreen -nosound
-ini:Engine:[/Script/Engine.RendererSettings]:r.Lumen.HardwareRayTracing=True,[SystemSettings]:r.RayTracing.ResidentGeometryMemoryPoolSizeInMB=8000
```
- **硬件光追 Lumen**：玻璃幕墙/立面反射更准；16 路相机每帧耗时不变（约 1.25s，瓶颈在传输/同步，不在 GPU）。
- **光追几何内存 8000MB**：默认 2000MB 会超预算（画面左上角 `RAY TRACING GEOMETRY - REQUESTED MEMORY OVER BUDGET`）。
- **光追阴影 `r.RayTracing.Shadows` 不开**：光追不支持带 WPO（风摆）的植被，树/草阴影会错；虚拟阴影贴图无此问题。

### 2.14 修鱼眼相机曝光（6 面共享自动曝光）

**现象**：默认档位曝光固定、阴天/低太阳欠曝；换成带自动曝光的后处理档位，鱼眼图出现亮度接缝；`exposure_*` 属性改了没反应。

**根因**（源码多路分析）：
1. 鱼眼 = 6 个 `USceneCaptureComponent2D` cube 面分别渲染再拼接，每个面有自己的 view state，**各自测光** → 接缝。
2. 上游 #9110 把相机曝光改成 JSON 档位，#9741 移植鱼眼时只照抄了 `exposure_*` 属性列表、没有任何代码读取。
3. 空档位解析成 `Default`，而 `Default.json` 不存在 → 加载静默失败 → 每个面停在构造函数的固定手动曝光（约 EV 9.9）。

**修复**：
- 引擎补丁（2.10）+ `SceneCaptureSensor_WideAngleLens::BeginPlay`：面 1~5 注册使用面 0（前方）的 eye adaptation；
  面 0 总在渲染掩码里且最先拍，同一帧内其余面读到的就是它刚算出的曝光，零延迟。
- 同时关掉按面单独计算、仍会留接缝的效果：局部曝光（对比度系数设 1）、bloom、镜头光晕。
- `ActorBlueprintFunctionLibrary.cpp::SetCamera(..., WideAngleLens)`：档位没加载成功时，把 `exposure_mode / exposure_compensation /
  exposure_min/max_bright / exposure_speed_up/down`（手动模式另加 `shutter_speed / iso / fstop`）应用到 6 个面。
- 采集配置 `camera_config.py`：直方图自动曝光，范围 7~15 EV100（上游默认 10~12 太窄）。

**效果**：同一相机在低太阳和正午下亮度都约 120（修复前 70 / 109），200 帧实采目测无接缝。

### 2.15 rt_lens 降噪（DLSS-RR）

**现象**：日志 `DLSS Ray Reconstruction is not available on this system (driver/GPU/snippet)`，`rt_lens` 满屏噪点。
**根因**：插件只静态链接 NGX 加载器，DLSS 模型 DLL（`nvngx_dlssd.dll`、`nvngx_dlss.dll`）运行时才在 exe 目录或 `$DLSS_SDK` 下找；
`DLSS_SDK` 只在打包命令里设过，包里也没拷 DLL（`nvngx.log`：`NGXLoadFromPath failed 0xBAD00004`）。
**修复**：`Unreal/CMakeLists.txt` 打包后处理把两个 DLL（`C:\Users\bills\SDKs\DLSS\lib\Windows_x86_64\rel\`，310.9.1）拷到
`Windows\CarlaUnreal\Binaries\Win64\`。本机编辑器要用的话：设环境变量 `DLSS_SDK=C:\Users\bills\SDKs\DLSS` 再启动。
**注意**：`rt_lens` 不能和光栅化相机（含鱼眼 6 面、预览窗口）同一帧渲染，否则引擎断言崩溃。

### 2.16 预览窗口（pygame）

`remote\carla_viewer.py`：在服务端挂一个观察相机，画面传回窗口；只读，不 tick、不改设置。
- 操作：WASD 移动、Q/E 上升下降、鼠标左/右键拖动或方向键转视角、Shift 4 倍速、滚轮调速、F 切换自由漫游/跟随车辆、Tab 切视角、R 跳到 spectator、Esc 退出。
- 中文输入法会吞掉字母键：启动和获得焦点时 `pygame.key.stop_text_input()`，并按扫描码记录按键。
- 流畅度：移动用 `apply_batch`（不等回复）；本机默认服务端渲染 960×540 再放大（每帧约 2MB）；`get_settings` 3 秒才查一次。
- 观察相机带 `viewer@主机名` 标签，启动时清掉本机遗留的（强杀的窗口不会销毁自己的相机，会一直占服务端渲染）。
- 81 上：计划任务 `CARLA-Viewer`（Administrator 交互登录时启动，`pythonw`，失败自动重启），`carla_remote.ps1 -Action viewer` 部署/重启。
  在远程桌面会话里也能显示（pygame 只贴图不需要显卡）。
- 帧率：81 本机约 40 fps；本机经网络约 20~30 fps；服务端自身约 38 fps（瓶颈在游戏线程，GPU 只用 35%）。

---

## 3. 以后怎么用

**编译 5.8 编辑器版**（在 `D:\code\carla_ue58`）：
```powershell
& "C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44
cmake --build Build --target carla-unreal-editor
```

**打包并部署到 81**（改了 C++ 之后）：
```powershell
# CRLF 副本运行 build_package_development.bat（约 1 小时，首次 2.5 小时）
powershell -NoProfile -ExecutionPolicy Bypass -File remote\carla_remote.ps1 -Action deploy
```
只改 Python：不用动 81。只改渲染设置：改 `carla_remote.ps1` 的 `$CarlaArgs` 后 `-Action restart`。

**采集**（本机连 81）：
```powershell
C:\Users\bills\.conda\envs\carla58\python.exe occnetv3_data_generator\main_collection.py --host 192.168.100.81 --frames 200 --output <目录> --weather random --seed 1
```
不加 `--append` 会清空输出目录；`--weather` 默认 `random`，每个场景写 `scenes/<scene>.json`。
批量采集：`occnetv3_data_generator\batch_collection.py --host 192.168.100.81 --output <目录>`（多地图×多场景，断点续采）。

**看画面**：`python remote\carla_viewer.py`（本机，连 81）；81 桌面上的窗口登录即自动打开。
**看数据集**：`python dataset_viewer_v2\server.py --dataset <目录> --port 8085`（默认端口 8000 会和交通管理器冲突），
浏览器开 `http://127.0.0.1:8085/`（其他机器用 `http://192.168.100.182:8085/`）。

**同步官方更新**：`git fetch upstream && git merge upstream/ue58-dev`（在 `main` 分支）。上游若更新引擎分支，
重新拉引擎后要**重新打 `Util\EnginePatches\` 里的补丁**；Content 分支更新也要重新下载。

---

## 4. 待办

- [ ] 提交 2.10 之后的全部改动（引擎补丁文件、曝光修复、打包/部署脚本、预览窗口、数据集预览页面、文档）
- [ ] 随机天气恢复完整范围（太阳 8~70°、含阴天）：曝光已能自适应，临时限制可以去掉（`main_collection.py::choose_weather`）
- [ ] Bayer RAW 改为线性 HDR（鱼眼各面现在取色调映射后的 LDR，见 UE58_RENDERING_ISSUES.md 问题 3）
- [ ] 评估整套相机换成 `rt_lens`（单视图路径追踪，物理最准，需给它加 Bayer 输出并测性能）
- [ ] `ue5-dev` 上的 `0bbc52823` 是否推到 GitHub
- [ ] 验证没问题后删除 5.5 的大件（约 400GB）。**不要整个删 `D:\code\carla`**（5.8 的 git 数据在它的 `.git` 里）
- [ ] 本机 D 盘只剩约 10GB（游戏版编译中间文件 23GB）
- [ ] 旧的 rear +8° 数据是朝上拍的，不要和新数据混训
- [ ] git 升到 2.45.2+（避开 LFS clone 的 hook 误报）

---

## 5. 关键经验

1. **大升级先开 worktree**，保证旧版本一直可用。
2. **先算磁盘账**：Content LFS 缓存 + 检出要两倍空间；打包要 60GB+，用目录链接转到大盘。
3. **编译器版本要统一**：CMake 编的库和 UBT 链接用的 MSVC 必须同一版本（5.8 是 14.44）。
4. **不要只信退出码**：看产物时间戳和日志里的实际动作。
5. **后台长任务要干净停掉**：确认整个进程树都结束，否则留下"幽灵"进程抢资源。
6. **编辑器首次启动很慢是正常的**：以游戏线程能正常响应为准。
7. **引擎补丁要存进 carla 仓库**：引擎目录是独立仓库，换机器/重拉就丢。
8. **服务端常驻要能复位**：任何中断的脚本都可能把同步模式、天气、残留 actor 留在服务端。
9. **出问题先读源码找根因**：曝光接缝、DLSS 不可用、cook 失败都在源码里找到了确切原因，而不是靠调参碰运气。
