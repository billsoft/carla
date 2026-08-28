# CARLA + UE5 引擎构建笔记

> 记录本机（Windows，VS2026）上把这两个仓库从"能 clone"到"能编译、能跑、能改代码
> 重新编译"过程中踩过的坑。写这份笔记之前实际走过一次完整的：改引擎代码 → 编译失败
> → 定位 → 修复 → 再编译 → 崩溃 → 定位 → 修复 → ... 循环，前后跨了两个仓库、
> 大约十次增量重编译。目的是让下一次改动不用重新踩一遍。

## 1. 两个仓库是什么关系

这套开发环境由**两个独立的 git 仓库**组成，缺一不可：

| | 仓库 | 本机路径 | 远程 | 分支 |
|---|---|---|---|---|
| **CARLA** | 本仓库 | `D:\code\carla` | `origin`=`billsoft/carla`(fork)，`upstream`=`carla-simulator/carla` | `ue5-dev` |
| **UE5 引擎** | CARLA 专用 UE5 fork | `D:\code\UnrealEngine5_carla` | `CarlaUnreal/UnrealEngine` | `ue5-dev-carla` |

**关系**：CARLA 仓库不含 UE5 引擎源码，只含一个 UE5 **插件**
（`Unreal/CarlaUnreal/Plugins/Carla/`）和一个空的 UE5 **项目**
（`Unreal/CarlaUnreal/CarlaUnreal.uproject`）。编译 CARLA 的 UE5 部分本质上是"用
CARLA 专用 UE5 fork 的源码，把这个项目 + 插件编译进去"，所以 CMake 需要知道 UE5 引擎
源码在哪——通过环境变量 `CARLA_UNREAL_ENGINE_PATH` 指向 `D:\code\UnrealEngine5_carla`
（`CMake/Options.cmake`/`CMake/Toolchain.cmake` 里读取；`CarlaSetup.bat` 首次运行时
默认克隆到 CARLA 仓库同级目录并用 `setx` 持久化这个环境变量）。

**这两个仓库的版本必须互相匹配**——CARLA 插件代码用到的一些 UE5 API/行为可能只在
特定引擎 commit 之后才存在或修复过。本次会话里先单独更新了引擎仓库（拉了 1489 个
commit），过程中一度怀疑"是不是版本不匹配导致大量报错"，事后确认**不是**：真正的
崩溃/编译错误在引擎更新前后表现完全一致，根因在 CARLA 插件代码本身（见第 4 节），
和引擎版本无关。但这不代表版本对齐不重要——只是这次具体踩的坑恰好和它无关，遇到
诡异的引擎 API 报错时，版本不匹配仍然是需要排除的一个可能性。

**获取 UE5 引擎源码需要权限**：`https://github.com/CarlaUnreal/UnrealEngine.git`
是 Epic 官方按 CARLA 项目单独授权的私有 fork，`git clone` 前你的 GitHub 账号必须先
在 Epic Games 官网关联并接受 Unreal Engine EULA，否则 clone 会直接失败（看起来像是
仓库不存在/无权限，而不是网络问题）。这一步是一次性的，和 CARLA 仓库本身的访问权限
无关。

## 2. 首次搭建环境

```cmd
cd D:\code\carla
CarlaSetup.bat --interactive
```

这个脚本会依次：装 prerequisites（`Util/SetupUtils/InstallPrerequisites.bat`）→
clone 游戏内容资源（`carla-content`，从 Bitbucket，不是引擎源码）→
如果 `CARLA_UNREAL_ENGINE_PATH` 未设置且同级目录没有 `UnrealEngine5_carla`，
`git clone -b ue5-dev-carla https://github.com/CarlaUnreal/UnrealEngine.git
UnrealEngine5_carla` 并 `setx` 持久化路径 → 跑一次 CMake 配置。

**磁盘/时间预算**：完整走一遍（含 UE5 引擎源码 + 首次全量编译）需要 225GB+ 磁盘、
3 小时以上，纯网络下载 UE5 引擎源码这一步就不小。本机这次会话中途做过一次"仅更新
引擎仓库"（`git pull`，1489 个 commit，非首次 clone）耗时也有几十分钟。

**Windows 下已验证可用的编译方式**：不要照搬 `CarlaSetup.bat` 生成的 CMake 命令直接
用，本机走的是下面这条更可靠的路径。

## 3. 日常编译：`BUILD_FINAL.bat`

```cmd
cd D:\code\carla
BUILD_FINAL.bat
```

这个脚本比通用 CMake 流程更可靠，因为它包含了几个针对本机环境的关键修复，**每一条
都是真实踩过坑之后加上的，不是预防性代码**：

| 修复 | 现象（不加这条会怎样） | 原因 |
|---|---|---|
| 强制用 VS 2026 (v18) 的 `vcvars64.bat` | 用系统默认/PATH 里的编译器可能选到别的 VS 版本，UE5.5 对编译器版本敏感 | 本机装了不止一个 VS 版本 |
| `del Build\CMakeCache.txt` 后再 configure | 换了 CMake 参数（比如改了 `CARLA_UNREAL_ENGINE_PATH`）不生效，用的还是旧缓存 | CMakeCache 会缓存首次 configure 的路径参数 |
| `/wd4723`（抑制"potential divide by 0"） | UE5 ChaosVehicles 相关代码在 VS2026 下把这个警告当错误，**编译直接失败**，和你改没改代码无关 | 见下方"引擎侧本地补丁" |
| `CMAKE_CXX_FLAGS` 里显式 `/W1` | 不加的话某些编译单元会用 `/W4`，把 V2X 模块里本来就存在的 shadowing/窄化警告升级成错误，挡住全量编译 | 见第 4 节 V2X 修复 |
| `-DCARLA_UNREAL_ENGINE_PATH="D:\code\UnrealEngine5_carla"` | 不设置的话 CMake 找不到引擎源码，直接配置失败 | 见第 1 节 |
| 编译目标固定 `carla-unreal-editor` | 默认目标可能编不到 Carla 插件 DLL，或者编译范围过大耗时暴涨 | 只需要插件 DLL 时没必要编整个引擎/打包 |

**构建产物验证**：`Unreal\CarlaUnreal\Plugins\Carla\Binaries\Win64\UnrealEditor-Carla.dll`
的修改时间是不是刚编译完那一刻——`BUILD_FINAL.bat` 打印"BUILD COMPLETED
SUCCESSFULLY"不代表插件 DLL 真的更新了（比如目标选错、增量编译误判没有变化），
每次编译完先看这个文件的时间戳再去重启编辑器，比盯着终端输出可靠。

**增量编译有多快**：只改一两个 `.cpp`/`.h` 文件，哪怕 `CMakeCache.txt` 被删了重新
configure，ninja 的对象文件缓存是独立保留的，实测大约 5-15 分钟能编完（对比全量编译
的量级完全不同）。

### 引擎侧本地补丁（不在 git 历史里，clone 后需要手动重新应用）

`D:\code\UnrealEngine5_carla` 是普通 `git clone`，这两处修改是本机手动打的补丁，
`git status` 能看到但从未提交，**如果重新 clone 引擎仓库或者 `git checkout .` 会丢
（更新引擎版本前记得先 `git stash` 这两个文件，更新完再 `git stash pop`，本次会话
就是这么做的，无冲突）**：

1. `Engine/Source/Runtime/Core/Public/Windows/WindowsPlatformCompilerSetup.h`——在
   `4723` 的 `#pragma warning (error: ...)` 列表里去掉 `4723`，改成
   `#pragma warning (disable: 4723)`。不打这个补丁，`BUILD_FINAL.bat` 的
   `/wd4723` 命令行参数也压不住——引擎头文件里的 `#pragma warning(error:...)`
   优先级比命令行 `/wd` 高，只能在源码层面改。
2. `Engine/Plugins/Animation/RigLogic/Source/RigLogicLib/Private/dna/stream/FilteredBinaryInputArchive.cpp`——
   `passingIndices`/`remappedIndices` 两个局部变量改名为
   `local_passingIndices`/`local_remappedIndices`，消除和外层作用域的 shadowing
   警告（同样是被 `/W4` 或某个 warning-as-error 配置挡住编译）。

### 通用 CMake 命令（Linux，以及 Windows 手动排障时用）

见根目录 `CLAUDE.md`"构建 CARLA"一节，不重复列。

## 4. 本次会话踩过的坑（按发现顺序）

这些坑大多不是"改代码引入的新问题"，而是**引擎升级/首次真正跑通某条代码路径时才
暴露出来的既有问题**——教训是：不要看到一堆报错就怀疑是环境/版本问题，一个一个看
栈追出根因，大概率是具体某处代码的具体某行有问题。

### 4.1 `cmd.exe` 通过 Bash 工具调用会静默不执行

用 Bash 工具直接 `cmd.exe /c "BUILD_FINAL.bat > log 2>&1"` 只会打印 Windows 的
版权 banner，命令根本没跑（`log` 文件是空的），但也不报错，非常容易被误判为"编译
瞬间完成"。

**根因**：这是 MSYS/Git Bash 的路径转换机制——单斜杠开头的参数（`/c`）会被当成
Unix 风格路径自动转换成 Windows 路径，cmd.exe 收到的根本不是 `/c` 这个开关，于是
退化成不带参数的交互式 cmd.exe，在无 stdin 的后台环境下读到 EOF 立刻退出，只留下
启动时打印的版权 banner。**修复（两种都可行）**：要么用 PowerShell 工具执行
`& cmd.exe /c "..."`（PowerShell 不做这个转换）；要么留在 Bash 工具里，把 `/c` 换成
双斜杠 `cmd.exe //c "..."` 显式绕过转换——`taskkill //PID <pid> //F` 也是同一个
根因、同一个转义方式，这个坑不止影响 `cmd.exe`，任何单斜杠开头的原生 Windows 程序
参数（`/c`、`/PID`、`/F`……）在 Bash 工具里都要用双斜杠。

### 4.2 UBA（Unreal Build Accelerator）内存压力假死

编译在接近完成（比如 98%+）时卡住不动，反复打印一样的
`Delaying N processes from spawning due to memory pressure (Available: X.Xgb)`，
而且这个 `Available` 数字和 `Get-CimInstance Win32_OperatingSystem` 实际看到的
可用内存对不上（本机遇到过 UBA 报 ~9GB 可用，实际系统有 30GB+ 空闲）。**修复**：
`tasklist` 找到 `ninja.exe` 进程 `taskkill //PID <pid> //F //T`，重新跑
`BUILD_FINAL.bat`——增量缓存都在，几分钟内恢复到卡住之前的进度继续往下编。

### 4.3 无头模式（`-game -RenderOffScreen`）走不通

尝试过用 `-game -RenderOffScreen` 启动一个不带界面的服务器：第一次能连上 RPC 但
碰到 Episode 相关调用就出问题；引擎更新后的第二次尝试直接空 `Fatal error!` 崩溃，
没有可诊断的堆栈。**结论**：这条路线放弃，固定用正常交互式 GUI 编辑器 +
`-CarlaAutoPlay`（见 4.9）替代手动点 Play，而不是追求完全无头。

### 4.4 崩溃/卡死排查必须用非缓冲实时日志

UE5 默认写到 `Saved/Logs/*.log` 是有缓冲的，进程崩溃或长时间挂起时，最后那行真正
触发问题的日志经常没来得及刷盘，看到的日志"戛然而止"在崩溃点之前，看不出真正原因。
**必须用** `-log -stdout -FullStdOutLogOutput` 启动参数，配合 PowerShell 的
`Start-Process -RedirectStandardOutput ... -RedirectStandardError ...`
把 stdout/stderr 重定向到文件——这样才能看到崩溃前最后执行到的确切日志行。本次
会话里定位 4.6 提到的几处崩溃全靠这个。

### 4.5 V2X 模块预存在的编译错误挡住任何全量重编译

`BUILD_FINAL.bat` 显式设 `/W1` 之前，某些编译单元会用到更严格的警告级别，暴露出
`CustomV2XSensor.h/.cpp`、`PathLossModel.h/.cpp`、`CaService.cpp` 里几处和这次
任务完全无关的既有问题：`SetOwner(AActor *Owner)` 之类的参数名和成员/外层作用域
重名（C4458 shadowing）、`Vector3D{...}` 花括号初始化里的隐式窄化转换（C2398）。
修复是纯机械的重命名参数（加 `In` 前缀）和补 `static_cast<float>`，不涉及任何逻辑
改动——**这些错误不是这次任务引入的，是本来就有、只是从来没有在严格警告级别下真正
编译过**。

### 4.6 `ConstructorHelpers` 在这个引擎版本上不安全（4 处崩溃/挂起）

这是本次会话里最花时间的一类坑，统一的根因和结论值得单独说清楚：

**根因**：`ConstructorHelpers::FObjectFinder`/`FClassFinder` 官方文档写的是"只能在
构造函数里用"，但在这个引擎版本上，**在 CDO（Class Default Object）构造期间**
（也就是引擎/编辑器启动时，通过 `UObjectLoadAllCompiledInDefaultProperties`
无条件触发，和这个 Actor 会不会真的被用到没关系）加载资源，会有两种失败模式：

1. 加载的是**普通资源**（Material/Texture 等非 Blueprint 类）：概率性触发
   `EXCEPTION_ACCESS_VIOLATION`，出现在 `UClass::InternalCreateDefaultObjectWrapper`
   内部的重入式加载路径上，或加载链上某个还没挂载的插件内容包依赖导致的长时间
   （~60s）挂起（被 hang 检测器杀掉）。
2. 加载的是 **Blueprint 生成的 `_C` 类**（`ConstructorHelpers::FClassFinder`，
   比如交通灯/标志的蓝图模型）：即使换成"运行时安全"的
   `FSoftObjectPath(...).TryLoad()`，**照样崩溃**——因为加载一个还没编译过的
   Blueprint 类会触发**重入式 Blueprint 编译**，这个编译过程会回调进
   `FLiveCodingModule::StartupModule`，此时引擎自己都还在初始化过程中，直接
   `EXCEPTION_ACCESS_VIOLATION`。**这一点是本次排查走过弯路才确认的**：一开始以为
   只是 `ConstructorHelpers` 这个 API 本身的问题，换成 `FSoftObjectPath::TryLoad`
   就以为修好了，结果 `TrafficLightManager.cpp` 用 `TryLoad` 加载 Blueprint 类
   照样在完全相同的堆栈位置崩溃——**问题不是用了哪个加载 API，是"在 CDO 构造期间
   加载 Blueprint 类"这件事本身不安全**，普通资源（非 Blueprint）用 `TryLoad`
   才是真正安全的。

**修复原则**：
- 普通资源（Material/Texture）：直接把 `ConstructorHelpers::FObjectFinder<T>(path).Object`
  换成 `Cast<T>(FSoftObjectPath(path).TryLoad())`，可以留在构造函数里。
- Blueprint 类（`_C` 后缀）：**必须挪出构造函数**，改成惰性加载——加一个
  `bool bLoaded` 标志位 + `EnsureXxxLoaded()` 方法，在真正要用到这些类的入口函数
  最前面调用（不是构造函数，也不是 `BeginPlay` 意义上的"尽早"，是"第一次真正需要
  的时候"）。

**具体修了这 4 处**（详见对应文件的注释，都写了同样这段推理过程）：

| 文件 | 加载的是什么 | 修复方式 |
|---|---|---|
| `Commandlet/LoadAssetMaterialsCommandlet.cpp` | Blueprint (`RoadPainterPreset`) | 确认 `RoadPainterSubclass` 全插件没人读，**直接删掉这段死代码** |
| `Weather/Weather.cpp` | 普通 Material | 构造函数内换成 `TryLoad` |
| `OpenDrive/OpenDriveActor.cpp` | 普通 Texture2D | 构造函数内换成 `TryLoad` |
| `Commandlet/PrepareAssetsForCookingCommandlet.cpp/.h` | 7 个普通 Material | 挪到新方法 `LoadCarlaDefaultMaterials()`，在 `Main()` 开头调用一次 |
| `Traffic/TrafficLightManager.cpp/.h` | Blueprint（交通灯/标志/限速牌模型） | 挪到新方法 `EnsureDefaultModelsLoaded()`，在 `SpawnTrafficLights()`/`SpawnSignals()`/`GenerateSignalsAndTrafficLights()` 几个真正用到的入口最前面惰性调用，`bDefaultModelsLoaded` 标志位防重复 |

`TrafficLightManager` 这一处修了两轮：第一轮只是把 `FClassFinder` 换成
`TryLoad` 但留在构造函数里，**没修好**（同样的崩溃栈）；第二轮才是真正挪出构造
函数做惰性加载。中途还发现 `GenerateSignalsAndTrafficLights()`（`CallInEditor`
入口）有自己一份"如果模型没加载就报错返回"的检查，绕过了惰性加载，得单独在这个
入口前面也插一次 `EnsureDefaultModelsLoaded()` 调用——**改这类惰性加载修复时，
把所有可能绕过构造函数的入口都找一遍，不要假设只有一个入口**。

### 4.7 "Bad optional access"——一个存在了很久、和引擎版本无关的误导性 bug

`world.get_settings()` 之后打印/repr 这个对象会抛 `std::bad_optional_access`。
排查一度怀疑是本次引擎大版本更新（1489 个 commit）导致的 API 不兼容，实际验证
（更新引擎前后重现步骤完全一致）证明**和引擎版本无关**。真正根因在
`PythonAPI/carla/src/World.cpp` 的 `operator<<(std::ostream&, const
EpisodeSettings&)`——无条件写了
`out << settings.fixed_delta_seconds.value()`，而 `fixed_delta_seconds`
在同步模式关闭（变步长，新建 Episode 的默认状态）时本来就是空的
`std::optional`，`.value()` 直接抛异常。**`get_settings()` 这个 RPC 调用本身
从来没坏过，坏的是拿到结果之后"打印/repr 它"这一步**——这类"调用成功但
stringify 失败"的 bug 很容易被误判成"调用本身失败了"，排查时先确认异常抛出的
准确调用栈位置，不要假设异常发生在看起来"最可疑"的那次 RPC 调用里。修复很小：
判断 `has_value()` 再决定输出数值还是 `"None"`。这个文件属于 `carla-client`
库，只需要 `cmake --build Build --target carla-python-api-install` 重装
Python API，不需要重编引擎。

### 4.8 等距鱼眼相机 Bayer RAW 采集的 5 处坑

详见 `occnetv3_data_generator/README.md` 的"底层实现"一节，这里只列清单，避免
两处文档重复维护同一份细节：

1. `raw_type` 属性没注册成 `FActorVariation`，`set_attribute` 直接抛异常。
2. 鱼眼相机 HDR 分支漏调 `EnqueueRenderSceneImmediate()`，cubemap+distort 压根
   没重新渲染，读到的是旧/未初始化显存。
3. `ImageUtil.cpp::ReadImageDataAsync` 的 GPU 读回没等拷贝完成就 `Lock()` 读取
   （这是所有相机共用的底层代码）。
4. **`ImageSerializer.cpp::Deserialize`（客户端反序列化）无条件把 buffer 当
   4 字节 BGRA 处理，强制改第 4 个字节"修 alpha"，对非 BGRA 格式是灾难性破坏**——
   这是真正的根因，前三条各自也需要修，但都不是"数据传到 Python 端还是坏的"的
   直接原因，只有这一条命中了。
5. 渲染目标默认 8-bit，从未启用 16-bit HDR 格式（`Enable16BitFormat` 从没被
   调用过）。

这 5 条按顺序独立发现独立修复，中间每一条修完都重新编译+实测，表现"变了但还是
不对"——这是判断"修复方向对不对"的一个有用信号：如果同一个 bug 在两次不同修复
后表现出完全相同的具体数值（比如两次都是同样的浮点垃圾值），说明后一次修复根本
没触及问题所在的代码路径；如果表现变了（哪怕还是错的），说明触及到了，但可能
还没触及到根因本身，得继续往下查——本次 4.8.2→4.8.3 就是"表现变了但仍不对"，
4.8.3→4.8.4 才是"表现终于对了"。

### 4.9 `-CarlaAutoPlay`：省掉手动点 Play

`Unreal/CarlaUnreal/Plugins/Carla/Source/Carla/Carla.cpp` 的
`FCarlaModule::StartupModule()` 里加了 `RegisterAutoPlayWatcher()`：仅在命令行
带 `-CarlaAutoPlay` 时注册一个 `FTSTicker`，每秒检查一次
`FAssetCompilingManager::Get().GetNumRemainingAssets()` 和
`GShaderCompilingManager->GetNumRemainingJobs()` 是否都清零、`GEditor` 的世界
是否已加载，都满足后调用 `GEditor->RequestPlaySession(FRequestPlaySessionParams())`
自动开始 Play（30 分钟超时保护，避免异常情况下无限等待；不带这个参数完全不影响
正常手动使用编辑器）。

启动命令：

```powershell
$exe = "D:\code\UnrealEngine5_carla\Engine\Binaries\Win64\UnrealEditor.exe"
$proj = "D:\code\carla\Unreal\CarlaUnreal\CarlaUnreal.uproject"
Start-Process -FilePath $exe -ArgumentList "`"$proj`" -CarlaAutoPlay -log -stdout -FullStdOutLogOutput" `
    -PassThru -RedirectStandardOutput editor_stdout.log -RedirectStandardError editor_stderr.log
```

之后轮询 `carla.Client('127.0.0.1', 2000).get_server_version()` 直到连上即可
判断编辑器是否已经真正进入 Play——不要凭固定等待时间猜测，编辑器加载耗时波动很大
（本次会话里同样的重启操作，有时 2-3 分钟进 PIE，有时接近 20 分钟，原因未完全查明，
但 CPU 占用持续增长说明不是卡死，只是慢，继续轮询即可）。

### 4.10 光心偏移 (cx,cy) + 独立水平/垂直 FOV：物理镜头仿真基础设施

CARLA 的等距/Kannala-Brandt 鱼眼相机（`CameraModelUtil`/
`SceneCaptureSensor_WideAngleLens`）原本假设"完美"虚拟镜头：光心精确在图像几何中心
（`Center = Size/2` 硬编码），水平/垂直 FOV 靠一个共享的各向同性焦距按宽高比互相推导
（`XFOVAngle = YFOVAngle * Width/Height`）。Distort compute shader
（`WideAngleLens.usf`）其实早就把光心/焦距当 `float2` 运行时参数用了
（`CameraParams.xy`=焦距, `.zw`=光心），只是 C++ 端从来没把这两个分量分开传过。

加了 `FDistortCubemapToImageOptions::PrincipalPointOffset`（光心相对几何中心的像素
偏移）和 `XFocalLength`（独立于 `YFocalLength` 的水平焦距），C++ 端穿透到所有
6 个相机模型分支 + Kannala-Brandt 分支，**不用改 shader**。蓝图属性新增
`cx`/`cy`（绝对像素坐标）和 `fov_horizontal`（独立水平 FOV，度）。

修的时候顺带发现一个真实 bug：`SetFOVAngle()` 只更新了 `XFOVAngle`（推导值）却没有
同步重算 `XFocalLength`，导致后者一直停留在构造函数算出来的默认分辨率下的值——这个
bug 存在于加这次改动之前，只是原来没人会去读 `XFocalLength`（shader 一直只用同一个
标量），所以从来没暴露出来。

这一层默认不生效——`cx`/`cy`/`fov_horizontal`/`k0-k3` 都不设置时和加这些属性之前
行为完全一致（已实测验证零回归）。数据采集侧怎么用见
[`occnetv3_data_generator/README.md`](./occnetv3_data_generator/README.md)
的"物理镜头仿真层"一节。

**2026-08-27 补充**：评估后发现 `e2e_occ` 网络的射线编码/可变形注意力投影
（`position_encoding.py`/`deformable_attention.py`）里其实一直在悄悄忽略 `cx/cy`——
两处投影公式把光心硬编码成了特征图几何中心 `W/2, H/2`，`intrinsics` 里的真实
`cx/cy` 只在焦距降采样比例的（错误）反推里用了一下，从没真正参与过投影。这是个
潜伏 bug：当时所有相机的标定 `cx/cy` 都精确等于几何中心，所以看不出问题，一旦以后
真的接入非居中的物理标定数据就会静默出错。已经修好（两处投影公式改用真实 cx/cy，
顺带修了一处不相关的 `align_corners` 归一化不一致），`e2e_occ/ARCHITECTURE.md`
第 3.3 节和 `verify_network.py` 有完整记录。**k1-k4（Kannala-Brandt 畸变）依然
没有接入网络**——目前没有任何相机配置了真实畸变系数，没有数据驱动，留到真的需要
时再做，见 `occnetv3_data_generator/README.md` 对应小节。

### 4.11 `Variations.Contains(id)` 恒真陷阱：8 相机画面全部"看向错误方向"

4.10 加的 `cx`/`cy`/`fov_horizontal` 覆盖开关用 `if (Variations.Contains("fov_horizontal"))`
判断"用户是否显式设置了这个属性"，这个判断方式是错的，而且错得很隐蔽：**默认
不设置任何一个新属性时也会触发**，导致这次 4.10 加的功能一上线，全部 8 个鱼眼相机
的画面立刻全部错乱——表现是画面从图像中心硬切到不相关的场景内容，肉眼看像是
"相机装的是这个朝向，拍到的却是另一个朝向的画面"，且诡异的是 roll/pitch 看着都对，
只有水平方向不对，一度怀疑是外参 yaw 或坐标系公式的问题，排查了一圈无关目标
（`Rotation.h` 的 pitch/roll 符号、`ComputeCubemapRenderMask` 的面选择逻辑、
`WideAngleLens.usf`/`CameraModelCommon.ush` 跟已知良好的 commit `167cbff51`
逐字节 diff——shader 和数学公式完全没有回归）才定位到真正根因。

**根因**：`LibCarla/source/carla/client/ActorBlueprint.cpp::MakeActorDescription()`
会把蓝图**全部已注册属性**序列化进发给 server 的 `ActorDescription`，不管 Python
调用方有没有显式 `set_attribute()`——没显式设置的属性会带着它注册时的
`RecommendedValues[0]` 默认值一起发过去。于是服务端 `Variations.Contains(id)`
对任何已注册属性永远是 `true`，根本无法区分"用户显式设置"和"用的是默认值"。
`fov_horizontal` 注册的 `RecommendedValues` 是 `"90.0"`，`cx`/`cy` 是 `"0.0"`——
这两个值恰好都不是"什么都不做"的哨兵值（90° 不等于按宽高比推导的正确水平 FOV，
`(0,0)` 是图像左上角不是几何中心），所以**没有任何相机配置这些属性时，每一帧都被
无条件 `SetFOVAngleX(90.0)` + `SetPrincipalPoint(0,0)`，把 `SetFOVAngle()`/
`SetImageSize()` 刚算好的正确水平 FOV 和图像中心覆盖掉**，cubemap 因此按错误的
（偏大的）水平 FOV 采样，采到了 Front 以外的 Left/Right 面内容拼接进同一张图。

用诊断日志（临时加在 `SceneCaptureSensor_WideAngleLens.cpp::CaptureSceneExtended()`
里的 `UE_LOG`，验证完已移除）实锤：修复前任意相机配置下 `XFOV` 都固定打印
`90.00deg`、`PP` 都固定 `(0.0,0.0)`，和 Python 端传的 `fov`/`image_size` 完全无关。

**修复**：`ActorBlueprintFunctionLibrary.cpp` 里比照同一个函数中 `focal_length`
属性早就在用的正确写法——`focal_length` 的 `RecommendedValues` 是哨兵值 `"0.0"`，
判断用的是 `if (FocalLength != 0.0f)`（先取值，再判断是否等于哨兵），根本不用
`Contains()`。把 `fov_horizontal`/`cx`/`cy` 的判断改成同样的模式，并把
`fov_horizontal` 的 `RecommendedValues` 也从 `"90.0"` 改成语义正确的哨兵值
`"0.0"`（`cx`/`cy` 本来就是 `"0.0"`，不用改）。

**教训**：CARLA 的 `FActorVariation`/`Variations` 机制里，`Contains(id)` **不能**
用来判断"调用方是否显式设置了这个可选属性"——只要属性注册过，它就永远在
`Variations` 里，区别只是"用户的值"还是"注册时的默认值"。任何"不设置=保持某个
已有行为不变"语义的可选属性覆盖开关，必须选一个和"已有行为"不冲突的哨兵值，用
取值后比较哨兵的方式判断，不能用 `Contains()`——这是这次 4.10 新加功能自己引入的
回归，不是本来就有的坑，加新的可选属性覆盖开关时要按这个模式检查一遍。

### 4.12 等距鱼眼相机固定分辨率 cubemap 中间层：窄 FOV 相机天生模糊

2026-08-28 用户反馈 8 路相机图片（无论缩略图还是查看器里点开的原图）清晰度/质感
远不如 UE 编辑器里肉眼看到的效果，怀疑是渲染质量或采集流水线的问题。排查过程见
`occnetv3_data_generator/README.md`"物理镜头仿真层"一节，这里只记录 C++ 侧的根因
和修复。

**排除过程**（每一步都用对照实验验证，不是猜测）：连续 30 tick 固定机位测 TSR/Lumen
收敛——第 1 帧就已经很锐利，不是时域收敛问题；空场景 vs 20 车流场景对比——纹理流送/
Nanite 负载几乎不影响清晰度；真实驾驶 vs 冻结物理对比——不是运动模糊；`post_process_
profile`（`Town10HD_Opt.json`，带 `depthOfFieldFocalDistance=250` 即聚焦 2.5m 的
强景深）实测在没重编译前根本没注册进当前运行的二进制，说明这次的模糊和它无关。

**真正根因**：等距鱼眼相机（`ASceneCaptureSensor_WideAngleLens`）内部渲染 6 张
`Side x Side` 的 cubemap 面（`SceneCaptureSensor_WideAngleLens.cpp::BeginPlay()`
里 `Side = std::max(GetImageWidth(), GetImageHeight())`，固定等于输出分辨率，
和相机 FOV 无关），再用 `WideAngleLens.usf` 的 `SampleCubemap()` 重采样成最终
等距投影图像。cube face 天然覆盖约 90°/面，相机配置的 FOV 越窄，等于在这张固定
分辨率的图里截取放大一小块——FOV 越窄，输出像素密度需求越高，但源纹素密度不变，
必然更模糊，和渲染质量、光追、Lumen 无关。

用排除了"视场角本身改变可比较内容"这个混淆变量的对照实验坐实：同一位置、同一
角度范围(37.5°)，分别用 `sensor.camera.rgb` (pinhole，无 cubemap 中间层) 和
`sensor.camera.rgb_fisheye` (camera_model=equidistant) 采集：

| | Laplacian 方差(清晰度代理指标) |
|---|---|
| pinhole | 1399 |
| 等距鱼眼(cubemap重采样) | 217 |

同一 FOV 下 6.5 倍的差距，只能来自 cubemap 重采样这一步。8 个相机里 `front_main`
(37.5°) 和 `front_narrow`(26.25°，见 `occnetv3_data_generator/config/camera_config.py`)
FOV 最窄，受损最重；`front_wide`/`rear`(90°) 受损最小但也没有余量——cube face 在
90° 时原生密度和输出需求刚好打平，没有安全边际。

**尝试的修复（已回退，未进入生产）**：`Side` 按相机 FOV 反向缩放，`ScaledSide =
BaseSide * (90° / 相机最窄的 X/Y FOV)`，下限钳在原来的 `BaseSide`（宽 FOV 相机不
倒退）。孤立单相机测试确认有效：同一测试位置，`front_main` 实际配置 (37.5°) 清晰度
从 180 涨到 532（约 3 倍）。

**但在完整 8 相机生产阵列下，上限钳 2560（`front_narrow` 理论需求 ~4400px 的合理
折中）在全新启动的 editor 进程上、采集第一帧之前就直接硬崩溃**：
`Device->GetDevice()->CreateDescriptorHeap(...)` 报 `E_INVALIDARG`，崩溃时进程
WS 44-46GB（4090 只有 24GB 显存）。

随后把上限降到 1536（只让 8 台相机里的 2 台窄 FOV 相机线性边长提升约 20%）复测，
表面上"不再硬崩溃，但 `world.tick()` 卡死超过 60 秒触发客户端超时"。**但这个
1536 复测结果不可信，不能作为"两个不同 cap 都失败"的证据**：两次尝试之间的重
编译用 `BUILD_FINAL.bat`（通过 Bash 工具的 cmd.exe 调用）——事后确认这个调用
方式在本环境下会静默空跑：只打印一行 `cmd.exe` 交互式 banner 就在 1 秒内退出，
既没有删除重建 `CMakeCache.txt`，也没有任何 `cl.exe`/`ninja` 进程被启动过（复现
3 次，是可复现的工具问题，不是偶发）。也就是说 1536 这次复测大概率跑的还是那份
已经崩溃过的 2560 二进制，只是这次表现为超时而不是硬崩溃（当时后台还有一个占满
CPU 的进程在跑，这是第二个混淆变量）。**`BaseSide` 到 2560 之间没有任何一个 cap
值被真正验证过、也没有证据支撑"这不是简单的调低显存上限能解决的问题"这个结论
——已经从两篇文档和代码注释里删除了这个过度推断。**

**最终决定：完全回退**，`Side` 恢复为原始固定值（不随 FOV 缩放），见
`SceneCaptureSensor_WideAngleLens.cpp::BeginPlay()` 里保留的详细代码注释。回退
后用真正确认生效的重编译方式（PowerShell + `cmake --build Build --target
carla-unreal-editor`，用 DLL 时间戳而非退出码确认）验证：10 帧 × 8 相机生产采集
可以稳定跑完。**结论：`front_main`/`front_narrow` 这两台窄 FOV 相机目前仍然天生
比广角相机模糊，是已知但未解决的架构限制**——如果以后要重新尝试这个方向：①每次
重编译务必用 DLL 时间戳确认生效，不要信任 `BUILD_FINAL.bat` 通过 Bash 工具跑出来
的退出码（见下面"关联坑"之后新增的构建工具说明）；②先在隔离场景下单独验证一个
比较保守的 cap（如 1536），不要假设 2560 的失败模式就一定适用于更小的 cap。

**顺带处理的一个关联坑**：本节排查过程中确认 `post_process_profile` 这个属性
（会给 `Town10HD_Opt` 地图自动套用同名 JSON 档位）当时还没注册到鱼眼相机类
（`sensor.camera.rgb_fisheye`，生产 8 相机全部是这个类型）上——这是真实的 C++
缺口（只注册在普通针孔相机类上），已在 `ActorBlueprintFunctionLibrary.cpp` 里给
`SetCamera(..., ASceneCaptureSensor_WideAngleLens*)` 补上同款注册+加载逻辑，
重编译后 `bp.has_attribute('post_process_profile')` 从 `False` 变 `True`，缺口
本身已修好并保留。但补上后实测：`Town10HD_Opt.json` 那份档位（2.5m 强制景深 +
0.7 暗角 + 1.6 对比度调色 + `autoExposureBias=+1.2EV`）套用后，同点位对照反而比
不套用更糊更发灰——判断是白天场景下 `autoExposureBias` 过曝，不只是景深一处
问题。最终 `occnetv3_data_generator/sensors/camera_manager.py` 保持不设置这个
属性，训练相机维持不带暗角/浅景深/过曝的默认渲染。这个属性以后如果要重新启用，
先把 `autoExposureBias` 归零、重新做同点位对照，不要假设"官方同款档位"就一定
更好。

### 4.13 `BUILD_FINAL.bat` 通过 Claude Code 的 Bash 工具调用会静默空跑

2026-08-28 排查 4.12 的过程中撞见：`cmd.exe /c "BUILD_FINAL.bat" > log 2>&1` 这种
调用方式（Bash 工具，即 Git Bash 环境）看起来"成功"（exit code 0），但实际什么
都没编译——日志只有 `cmd.exe` 的交互式启动 banner（`Microsoft Windows [版本...]`
+ 版权行 + `D:\code\carla>` 提示符）三行，脚本本身 `@echo off` 之后的任何 `echo`
都没打印过。复现 3 次，规律一致：
- `Build\CMakeCache.txt` 时间戳完全不变（`BUILD_FINAL.bat` 第 44 行本该每次都
  `del /f` 后由 cmake 重新生成）。
- 空跑期间 `Get-Process`/`Get-CimInstance Win32_Process` 查不到任何 `cl.exe`、
  `cmake.exe`、`ninja.exe` 进程，说明连 CMake 配置阶段都没进入。
- 从调用到 "completed" 通知只有几秒钟，远不够真实构建（哪怕是增量构建也要几
  分钟起）。

**后果**：这次debug 期间，第一次"看起来成功"的重编译实际上是前一次会话（compact
之前）遗留的旧 DLL，被误判为"已经带上了本次改动"，导致后续基于它做的一次压力
测试结论完全不成立（见 4.12 里 cap=1536 复测那段的更正说明）——静默失败比报错
更危险，因为它会让人带着错误前提继续往下推理。

**验证有效的替代方式**（PowerShell 工具，不是 Bash 工具）：
```powershell
& "C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat" | Out-Null
cmake --build Build --target carla-unreal-editor
```
这条路径真实调用了 `UnrealBuildTool`，会打印 `[N/M] Compile ...`/`Link ...` 这些
实际的编译器动作，且会依次构建 `CarlaUnreal`（-game）和 `CarlaUnrealEditor`
（-editor）两个子 target（一次 `cmake --build --target carla-unreal-editor`
调用触发两次 UBT 调用，纯增量场景下大约 6 分钟，未出现空跑问题）。跳过了
`BUILD_FINAL.bat` 里每次都 `del /f Build\CMakeCache.txt` 强制全量重新 configure
的步骤（增量改动不需要重新 configure，`Build/` 下已有的 CMakeCache 直接可以用）。

**不管用哪种方式发起构建，都不要用退出码判断是否真的编译了** ——退出码 0 只说明
进程正常退出，不说明它做了什么。可靠的验证方式是重编译前后对比目标 DLL（例如
`Unreal\CarlaUnreal\Plugins\Carla\Binaries\Win64\UnrealEditor-Carla.dll`）的
`LastWriteTime` 是否发生了变化、且变化时间晚于源码编辑时间。

根因没有深挖（Git Bash/MSYS 的 pty 模拟和 `cmd.exe` 这类原生 Win32 控制台程序
交互本身有已知的兼容性问题，`BUILD_FINAL.bat` 内部 `call vcvars64.bat` /
`call conda activate` 这类嵌套 `call` 链可能触发了某种重新拉起控制台的路径），
没必要修——直接换成上面验证过的 PowerShell 调用方式即可。

### 4.14 `-quality-level=Low` 会把纹理糊成"水墨画"，跟等距鱼眼/Bayer RAW 无关

2026-08-28，4.12/4.13 都排查完、`final_verify2` 数据集也验证过"能稳定采集"之后，
用户看着 `dataset_viewer_v2` 里的图反馈"画质像水墨画一样发糊，怀疑是不是我们自己
封装的等距投影相机的问题"，并建议做同车同点位同 FOV 的鱼眼 vs 官方针孔对照实验。

**对照实验**（脚本见 `outputs/camera_comparison/compare_fisheye_vs_pinhole.py`）：
在同一辆车上，按 `occnetv3_data_generator/config/camera_config.py` 里 `TESLA_CAMERAS`
的 8 个真实点位，同时挂 8 个 `sensor.camera.rgb_fisheye`（等距投影，同 FOV/分辨率）
和 8 个官方 `sensor.camera.rgb`（针孔，`fov` 用同一份配置里历史上的水平 FOV 字段），
两组都用标准 `image.save_to_disk()` 输出 uint8 PNG（不经过我们自己的 Bayer/DNG raw
管线），排除 raw 管线这个变量。分别在 `-quality-level=Low` 和不带该参数（默认 Epic）
两种编辑器进程下各跑一次，四组图都在 `outputs/camera_comparison/`
（Epic）和 `outputs/camera_comparison/low_quality/`（Low）下留档。

**结果**：`front_narrow`（26.25° 垂直 FOV，受 4.12 提到的 cubemap 分辨率影响最大的
相机）在 Low 档位下，无论鱼眼还是官方针孔，同一栋脚手架建筑的立面纹理都是一片
发灰发糊的噪点状色块，边缘、文字广告牌完全糊成一团；同一位置 Epic 档位下两种相机
的纹理都变得清晰锐利、噪点消失、文字可辨。**鱼眼和针孔在同一档位下表现一致，说明
这个"水墨画"观感是纹理流送/mip 精度随 Scalability 档位整体下降导致的，和等距投影
实现、cubemap 重采样、Bayer RAW 管线都没有关系**——4.12 里"鱼眼比针孔更糊"的结论
（同 FOV Laplacian 差 6.5 倍）依然成立，但那是另一个独立问题，不要混为一谈。

**根因**：本次崩溃排查（4.12/4.13）期间，为了让编辑器重启更快，我在每次手动
`UnrealEditor.exe ... -CarlaAutoPlay` relaunch 时都加上了 `-quality-level=Low`——
包括生成 `final_verify2`（之前汇报"稳定性验证通过"的那份数据集）的那次编辑器进程。
也就是说 `final_verify2` 虽然证明了"回退 cube face FOV 缩放后能稳定采集完 10 帧"，
但画质本身是 Low 档位、不代表真实生产质量，已经用 Epic 档位重新采了一份
`outputs/final_verify_epic/` 替代它。

**这不是项目原有工作流的问题**：正式的无头生产启动脚本
`start_carla_server_headless.bat` 本身就写的是 `-quality-level=Epic`
（注释明确写着 "Sets the highest graphical quality for powerful GPUs"），
`start_carla_server.bat`、`main_collection.py` 都完全不引用 `quality-level`
这个参数——`Low` 只会在有人手动直接起 `UnrealEditor.exe`/`CarlaUE5.exe` 时
被人为带上（`CLAUDE.md` 的"直接启动"备选命令里就有这一条，本意是给快速冒烟
测试用，容易被误用到需要评估画质的场景）。**结论：以后任何要评估图像清晰度/
画质，或者作为"最终验证"用途的采集，启动编辑器前必须确认没有带
`-quality-level=Low`**，见 `CLAUDE.md` 对应位置补充的警告。

## 5. 相关文档

- 构建命令、目录结构速览：根目录 [`CLAUDE.md`](./CLAUDE.md)
- 等距鱼眼 Bayer RAW 相机的完整实现细节：[`occnetv3_data_generator/README.md`](./occnetv3_data_generator/README.md)
- 消费采集数据的网络：[`e2e_occ/README.md`](./e2e_occ/README.md)
