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
瞬间完成"。**必须用 PowerShell 工具，在里面套一层 `& cmd.exe /c "..."`。**

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

## 5. 相关文档

- 构建命令、目录结构速览：根目录 [`CLAUDE.md`](./CLAUDE.md)
- 等距鱼眼 Bayer RAW 相机的完整实现细节：[`occnetv3_data_generator/README.md`](./occnetv3_data_generator/README.md)
- 消费采集数据的网络：[`e2e_occ/README.md`](./e2e_occ/README.md)
