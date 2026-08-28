# OccNetV3 数据生成器

> **依赖自编译 CARLA UE5.5**：本仓库的 CARLA 在标准发行版之上做了两处底层增强——
> ① Bayer RAW HDR 采集（8-bit BGRA 之外的原生 16-bit 采集路径）② 等距投影
> （equidistant/鱼眼）Bayer 相机。标准 CARLA 发行版跑不了这份采集脚本。两处增强的
> 完整实现细节、坑点和修复记录见仓库根目录的
> [`CARLA_BUILD_NOTES.md`](../CARLA_BUILD_NOTES.md)（引擎获取/编译）和下面的
> "相机管线" 一节（数据采集侧怎么用）。

## 项目概述

`occnetv3_data_generator` 基于自编译 CARLA UE5.5 采集端到端占用网络（`e2e_occ`）的
训练数据：

- 8 相机环视等距投影（equidistant fisheye）Bayer RAW 图像采集（Tesla 相机布局）
- 深度图采集（与 RGB 相机重合，用于调试/校验，网络训练不直接用深度）
- 256 线语义激光雷达点云采集
- 3D 占用网格真值生成（400×400×32 体素，18 类语义，对齐 nuScenes）
- 场景流（Scene Flow）生成
- 基于 LiDAR 的可见性过滤（遮挡剔除）

## 相机管线

### 相机模型：等距投影，不是针孔

8 个相机 FOV 从 35° 到 120° 不等，广角相机用针孔投影模型在画面边缘失真严重（针孔的
`x=f·tan(θ)` 在 `θ→90°` 时发散），所以这套相机走的是**等距投影**（`r=f·θ`）：CARLA
蓝图是 `sensor.camera.rgb_fisheye`（不是 `sensor.camera.rgb`），核心属性
`camera_model=equidistant`（`perspective=False`、`equirectangular=False`、
`fov_mask=False`，否则引擎会在等距投影基础上再做一次转透视或转经纬图的 shader pass，
不是我们要的原生等距畸变输出）。这个相机在 CARLA 引擎里对应
`ASceneCaptureCamera_WideAngleLens`（`Unreal/.../Sensor/SceneCaptureCamera_WideAngleLens.cpp`），
和普通针孔相机 `ASceneCaptureCamera` 是两个不同的类，各自独立实现了 Bayer RAW 采集
（见下方"底层实现"）。

等距投影相机的 `fov` 属性对应的是**垂直** FOV（`YFOVAngle`），`camera_config.py` 里
每个相机字典的 `fov` 字段沿用的是历史上的水平 FOV 语义（仅作文档参考），真正传给
传感器、也用于算 intrinsics 的是 `fov_vertical` 字段。二者换算关系（各向同性等距投影
下精确成立）：`fov_vertical = fov_horizontal × (image_height / image_width) = fov × 0.75`
（960/1280）。对应地，`camera_manager.py::get_intrinsics()` 算焦距用的是等距公式
`focal = (height/2) / (fov_vertical弧度/2)`，不是针孔的 `width/(2·tan(fov/2))`。

**Tesla 8 相机布局：**

| 相机 ID | 水平 FOV | 垂直 FOV | 位置 (x,y,z) | 朝向 (pitch,yaw,roll) | 用途 |
|---------|---------|---------|--------------|------------------------|------|
| `front_main` | 50° | 37.5° | (1.0, 0.0, 1.6) | (0, 0, 0) | 前方主视 |
| `front_wide` | 120° | 90° | (1.0, 0.0, 1.6) | (0, 0, 0) | 前方广角 |
| `front_narrow` | 35° | 26.25° | (1.0, 0.0, 1.6) | (0, 0, 0) | 前方长焦 |
| `left_pillar` | 80° | 60° | (0.0, -1.1, 1.7) | (0, -55, 0) | 左 B 柱 |
| `right_pillar` | 80° | 60° | (0.0, 1.1, 1.7) | (0, 55, 0) | 右 B 柱 |
| `left_repeater` | 100° | 75° | (1.0, -1.0, 1.0) | (0, -130, 0) | 左后视 |
| `right_repeater` | 100° | 75° | (1.0, 1.0, 1.0) | (0, 130, 0) | 右后视 |
| `rear` | 120° | 90° | (-2.7, 0.0, 1.2) | (-8, 180, 0) | 后视 |

**坐标系/朝向踩坑记录：**
- CARLA 里 Y 轴正方向是"右侧"，负方向是"左侧"（驾驶员视角）——左侧相机必须用 Y **负值**
  （如 `left_pillar: Y=-1.1`），写反了相机会装反边。
- 后视相机沿 X 负方向外延（`X=-2.7`）避免拍到车玻璃；侧后视相机 Y 绝对值要够大，避免
  拍到车架。

### 底层实现：等距鱼眼相机的 Bayer RAW HDR 采集

标准 CARLA 只支持 8-bit BGRA（`EPixelFormat::BGRA_U8`）。本仓库在 C++ 层加了
`RawType` 属性（`"uint8"`/`"uint16"`/`"float32"`/`"bayer_rggb"`），非默认值时走
HDR/Raw 采集路径而不是默认的 8-bit `FColor` 路径。`camera_manager.py::_setup_cameras`
里通过 `camera_bp.set_attribute('raw_type', 'bayer_rggb')` 打开这条路径。

这套机制原本只在普通针孔相机 `ASceneCaptureCamera` 上实现过；2026-08 把等距鱼眼相机
`ASceneCaptureCamera_WideAngleLens` 也接上了同一套 HDR 采集，中间踩了几个只有鱼眼
相机才会遇到的坑（针孔相机的 `USceneCaptureComponent2D` 是引擎每帧自动捕获，鱼眼相机
的 6 面 cubemap 捕获 + 等距 distort compute shader 不是）：

1. **`raw_type` 属性没有注册成蓝图属性** — 两个相机类都需要在
   `ActorBlueprintFunctionLibrary.cpp` 的 `MakeCameraDefinition`/
   `MakeWideAngleLensCameraDefinition` 里显式注册成 `FActorVariation`，否则
   `set_attribute('raw_type', ...)` 会直接抛 `std::out_of_range`。
2. **鱼眼相机的 HDR 读取分支漏调 `EnqueueRenderSceneImmediate()`** — 这个调用内部就是
   `CaptureSceneExtended()`（6 面 cubemap 捕获 + distort compute shader 派发），针孔
   相机的 uint8 路径通过 `FPixelReader::SendPixelsInRenderThread` 内部间接调了它，鱼眼
   相机的 HDR 路径是自己发起读取，必须在读回纹理前手动调用一次，否则读到的是上一帧甚至
   未初始化的显存内容——表现为图像数据是均匀的两个值间隔重复（如 0/4080）。
3. **`ImageUtil.cpp::ReadImageDataAsync` 的 GPU 读回没等拷贝完成就读**（这是所有相机
   共用的底层代码，鱼眼相机才第一次真正走通这条路径把它暴露出来）——`EnqueueCopy` 提交
   GPU→CPU 拷贝后只 `ImmediateFlush` 到了 RHI 线程，不等 GPU 真正执行完，就直接
   `Lock()` 读取，拿到的是未就绪的暂存内存。表现为图像是 NaN / 巨大量级
   （`±1.7e38`~`3.4e38`）的浮点值——这类值恰好是把"随便的字节"重新解读成 float32
   会出现的典型模式（符号位+接近全 1 的指数位）。
4. **`LibCarla/source/carla/sensor/s11n/ImageSerializer.cpp::Deserialize`
   （客户端反序列化）无条件把整个 buffer 当 4 字节 BGRA `Color` 处理，强制把每组第 4
   个字节设成 `0xFF`（"修 alpha 通道"）**——这是真正的历史根因，且是所有 raw_type
   都会中招的通用 bug，不止鱼眼相机：对 `bayer_rggb`（2 字节/像素）这类非 BGRA 布局，
   这个操作会按错误的 4 字节步长踩坏数据；对 `float32`（3 通道各 4 字节）来说，
   第 4 个字节正好是某个 float 的符号位+高位指数位，几乎每个值都被写坏。修复：只在
   `pixel_format == BGRA_U8` 时才做这个 alpha 修正。前三个坑各自独立存在也需要修，
   但只有这一条才是"数据到了 Python 端还是坏的"的直接原因——之前三轮修复因为都没
   碰到这段代码，表现完全没变。
5. **渲染目标默认是 8-bit（`PF_B8G8R8A8`），从未启用 16-bit HDR 格式**
   （`bEnable16BitFormat` 一直是 `false`）——即使前面的坑都修好了，采到的也只是
   8-bit 数据量化后再线性放大到 0-65535（每通道约 256 级，间隔约 257），不是真正的
   16-bit 精度。修复：`raw_type` 非 `"uint8"` 时在 `SetCamera` 里调用
   `Camera->Enable16BitFormat(true)`，把渲染目标切到 `PF_FloatRGBA`。修复前后用
   `bayer_rggb` 实测：unique 值数量从 256（间隔约 257 的假 16-bit）跳到 13,815
   （真实高精度）。

**数据流：**

```
UE5 Render (equidistant distort compute shader → FLinearColor)
  → ImageUtil::ReadImageDataAsyncFLinearColor（等 GPU 拷贝真正完成）
  → ConvertRGBToBayerRGGB (Unreal/.../Sensor/ImageUtil.cpp，RGGB 2×2 采样)
  → ImageSerializer::Serialize(..., BAYER_RGGB_U16) → RPC Stream (TCP)
  → Python: image.raw_data → np.frombuffer(dtype=uint16) → (H,W) Bayer 数组
  → data_saver.py 量化到目标位深 → DNG
```

Python 侧解析（`camera_manager.py::convert_to_bayer`）现在就是读取引擎已经采好的单通道
`uint16` 数据，不再需要（也不应该）自己从 BGRA 图像里按相位挑通道模拟 Bayer——那是
`raw_type` 属性还没接入蓝图定义、`set_attribute` 静默走不到 HDR 分支时代的绕过写法，
拿到的从来不是真实传感器级数据。

### DNG 位深可配置

`main_collection.py --raw-bit-depth {8,10,12,14,16}`（默认 12，常见 CMOS 传感器精度）。
相机管线内部从 UE5 到 Python 客户端始终是 16-bit（0-65535）传输，这个参数只影响
`data_saver.py::_save_bayer_dng` 保存 DNG 前的量化位深（右移 `16 - raw_bit_depth`
位）。**不要指望从 DNG 文件本身反推位深**——DNG 的 EXIF `BitsPerSample` 标签靠不住
（PIL 的 TIFF writer 会按存储容器宽度把它覆盖成 16，不反映真实 ADC 位深）；实际使用
的位深记录在 `calibration/intrinsics.json` 的顶层 `raw_bit_depth` 字段里，
`e2e_occ/dataset.py` 加载 DNG 时从这里读取，不是硬编码。

### 物理镜头仿真层（默认关闭，基础设施）

采集用的等距投影相机默认是"完美"的虚拟镜头：各向同性焦距（水平/垂直 FOV 严格按宽高比
换算）、光心精确在几何中心、无径向畸变。真实物理镜头在光学实验室标定出来的参数从来不会
这么理想——这一层的目的是把实验室标定出的非理想参数加到虚拟相机的渲染画面上，让训练数据
物理上匹配真实传感器，而不是反过来"去畸变"虚拟画面（这个方向经常被搞反：虚拟镜头的完美
输出才是"原始信号"，真实镜头的畸变/光心偏移是要叠加上去的物理效应，不是要被校正掉的误差）。

CARLA 引擎侧（`Unreal/.../Util/CameraModelUtil.h/.cpp`、
`Unreal/.../Sensor/SceneCaptureSensor_WideAngleLens.h/.cpp`）已经支持：
- `camera_model=kannala-brandt` + `k0/k1/k2/k3`：径向畸变多项式
  `r = θ·(1 + k0·θ² + k1·θ⁴ + k2·θ⁶ + k3·θ⁸)`，和 OpenCV fisheye/Kannala-Brandt
  标定的 k1-k4 是同一套公式（0-indexed），实验室标定出来的系数可以原样填入。这部分
  蓝图属性一直就有，只是数据采集这边过去只用过 `equidistant`。
- `cx`/`cy`：像素单位的真实光心，不设置时精确等于 `(width/2, height/2)`。
- `fov_horizontal`：独立于 `fov`（垂直 FOV）的水平 FOV，不设置时保持"按宽高比从垂直
  FOV 线性推导"的默认行为（各向同性）。

`occnetv3_data_generator` 这边，每个相机在 `config/camera_config.py` 的
`TESLA_CAMERAS` 字典里可以选配 `lens_model`/`distortion_coeffs`/`principal_point`/
`fov_horizontal` 四个键（具体格式见该文件模块docstring），`sensors/camera_manager.py`
只在配置了对应键时才会设置这些蓝图属性、`get_intrinsics()` 也只在这种情况下才会返回
非理想值。**目前 8 个 Tesla 相机都没有配置任何一个键**——还没有真实实验室标定数据，
现在只是把接口打通。

**2026-08-27 修复的一处严重回归**：`cx`/`cy`/`fov_horizontal` 这一层刚加上时，
CARLA 引擎侧（`ActorBlueprintFunctionLibrary.cpp`）判断"该属性是否被显式设置"的
方式是错的，导致**不管上面 4 个键有没有配置，8 个相机的水平 FOV 和光心都被静默
覆盖成错误的默认值**，画面表现为从图像中心硬切到不相关场景内容（肉眼像是"相机
装的方向和实际拍到的方向不一致"）——即"现在只是把接口打通，实际采集行为完全一致"
这句话曾经并不成立，直到这处引擎 bug 被修好为止。根因、诊断过程、修复方式见
[`CARLA_BUILD_NOTES.md` §4.11](../CARLA_BUILD_NOTES.md#411-variationscontainsid-恒真陷阱8-相机画面全部看向错误方向)。
现在确认已恢复"不配置=行为不变"的正确语义（诊断日志 + 8 相机实拍图像双重验证）。

以后拿到某个物理相机模组的标定数据后，接入步骤：
1. 在对应相机的字典里加 `'lens_model': 'kannala-brandt'`、`'distortion_coeffs':
   (k0,k1,k2,k3)`、`'principal_point': (cx,cy)`（`fov_horizontal` 视标定报告是否
   给了独立的水平/垂直 FOV 再决定要不要加）。
2. 不需要改 `camera_manager.py`——这些键已经支持了。
3. **e2e_occ 网络消费进度（2026-08-27 更新）**：网络的射线编码/可变形注意力投影
   （`e2e_occ/position_encoding.py`、`e2e_occ/deformable_attention.py`）现在会
   真正读取 `intrinsics` 里的 `cx/cy` 参与投影计算（原来这两处都把光心硬编码成
   `W/2, H/2`，`cx/cy` 传了但从没被用过，是修完才发现的一处潜伏 bug），所以
   `principal_point` 这一项接入后网络端不需要额外改动，直接就能吃。**仍然不消费的
   只有 `distortion_coeffs`（Kannala-Brandt k0-k3 非等距畸变）**：网络的正/逆投影
   公式仍然是纯等距 `theta=r/f`、`r=f·theta`，没有畸变多项式项，这是刻意维持的
   职责边界——相机内外参只通过射线编码这一处表达给网络，接入 k1-k4 需要给正投影
   加畸变多项式、给反投影加牛顿迭代求逆（两处必须严格互逆，成本和风险都明显高于
   `cx/cy`），等真的有物理镜头标定出的 k1-k4 数据要接入时再做，不要在没有真实
   标定数据驱动的情况下现在就加。

### 图像清晰度问题（2026-08-28，部分修复；鱼眼分辨率问题因稳定性原因已回退）

用户反馈 8 路相机图片（缩略图和查看器原图都一样）清晰度/质感明显不如 UE 编辑器里
肉眼看到的效果。排查过程见 [`CARLA_BUILD_NOTES.md` §4.12](../CARLA_BUILD_NOTES.md#412-等距鱼眼相机固定分辨率-cubemap-中间层窄-fov-相机天生模糊)，
根因确认：等距鱼眼相机内部先渲染成固定分辨率的 cubemap 再重采样成最终图像，cube
face 分辨率不随相机 FOV 变化，FOV 越窄的相机（`front_main` 37.5°、`front_narrow`
26.25°）等于在固定分辨率的图里截取放大一小块，天然更模糊，和渲染质量/光追/Lumen
无关（同点位针孔相机对照测试证实 Lumen GI/反射/阴影渲染完全正常，问题只出在鱼眼
cubemap 重采样这一层）。

**cube face 分辨率随 FOV 反向缩放的修复已回退，未进入生产**：孤立单相机测试中该
方案确实有效（`front_main` 37.5° 配置 Laplacian 方差 180→532）。但在完整 8 相机
生产阵列下，cap=2560 于全新启动的编辑器进程上，在采集出第一帧之前就硬崩溃——
`CreateDescriptorHeap` 报 `E_INVALIDARG`，崩溃时 WS 44-46GB（4090 只有 24GB 显存）。
随后把 cap 降到 1536 复测时表面上"没崩溃、但 `world.tick()` 60 秒超时卡死"，**但
这次复测本身不可信**：两次尝试之间的重编译，用的 `BUILD_FINAL.bat`（通过本工具
的 Bash/cmd.exe 调用）静默空跑了——只打印一行 banner 就在 1 秒内退出，`CMakeCache.txt`
没被删除重建、也没有任何 `cl.exe`/`ninja` 进程启动过（复现 3 次，确认是可复现的
工具问题，不是偶发）。也就是说 cap=1536 的复测大概率跑的仍然是那份已经崩溃过的
2560 二进制，只是这次因为超时而不是硬崩溃收场（外加当时后台有一个占满 CPU 的进程
在跑，这也是个混淆变量）——`BaseSide` 和 2560 之间没有任何一个 cap 值被真正验证过。
最终**完全回退到原始固定 `Side`（不随 FOV 缩放）**，用真正确认生效的重编译方式
（PowerShell + `cmake --build Build --target carla-unreal-editor`，用 DLL 时间戳
而非退出码确认）验证：10 帧×8相机生产采集可以稳定跑完。**结论：`front_main`/
`front_narrow` 这两台窄 FOV 相机目前仍然天生比广角相机模糊，是已知但未解决的
架构限制**。如果以后要重新尝试这个方向：①每次重编译都要用 DLL 时间戳确认，不要
信任 `BUILD_FINAL.bat` 通过本工具跑出来的退出码；②先在隔离场景下单独验证一个
比较保守的 cap（如 1536）,不要假设 2560 的失败模式就一定适用于更小的 cap。具体
代码注释见 `SceneCaptureSensor_WideAngleLens.cpp::BeginPlay()`。

排查过程中顺带发现并处理了两个关联问题：
1. `post_process_profile` 属性会给 `Town10HD_Opt` 地图自动套用同名 JSON 后处理档位，
   这份档位其实是给人眼预览/电影感录屏调的（2.5m 强制景深 + 暗角 + 大幅调色 +
   `autoExposureBias=+1.2EV`），不适合训练传感器相机。过程中发现该属性在鱼眼相机
   类（`sensor.camera.rgb_fisheye`，生产 8 相机全部是这个类型）上其实根本没注册
   （C++ 缺口，已在 `ActorBlueprintFunctionLibrary.cpp` 补上并验证 `has_attribute`
   生效），但补上后同点位实测发现"套用档位"反而比"不套用"更糊更发灰（判断是
   `autoExposureBias` 在白天场景下过曝，不只是景深一处问题）——最终 `camera_manager.py`
   保持不设置这个属性，训练相机维持处处清晰的默认渲染,不带景深/暗角/调色。C++
   侧的属性注册缺口本身已经修好并保留，只是不默认启用。
2. DNG 的 `white_level` 元数据没有跟着 `raw_bit_depth` 走，rawpy 默认按 16-bit
   (65535) 解析白点，实际数据是 `raw_bit_depth`（默认12-bit，最大4095），会让
   自动曝光/色调映射算错。这个改动比较小，选择在读取侧修（`dataset_viewer_v2/
   server.py` 现在会读 `calibration/intrinsics.json` 的 `raw_bit_depth`，传给
   `rawpy.postprocess(..., user_sat=2**raw_bit_depth-1)`），没有去碰 DNG 写入侧
   的 TIFF tag 格式（piexif 不支持非标准 DNG tag，改起来成本明显更高）。**注意：
   直接用 rawpy/其他工具独立解码这批 DNG 时，如果不手动传 `user_sat`，色调/曝光
   会和 `dataset_viewer_v2` 里看到的不一致**，不是数据本身的问题。

调试专用：`main_collection.py --debug-actor-ids` 开关（默认关闭，训练不需要）会
额外保存每帧 `debug_actor_ids/*.npy`（每体素对应的 CARLA actor.id 或环境物体虚拟
ID），配合 `dataset_viewer_v2` 点击体素查看 `actor_id` 的功能，用于精确定位"某个
体素究竟是哪个 actor/环境物体生成的"这类问题（这次排查 `traffic.traffic_light`
bounding_box 偏移导致的白块就是靠这个直接定位到的，见下面"语义类别映射"一节）。

### 深度相机

与 RGB 相机完全重合（相同位置、FOV，标准 `sensor.camera.depth`，针孔投影，UE5
原生 24-bit 编码深度，`depth = (R+G·256+B·256²)/(256³-1)×1000` 米），主要用于调试
和标定校验，不是网络训练输入。

## 目录结构

```
occnetv3_data_generator/
├── config/
│   ├── camera_config.py            # 8 相机布局 (FOV/fov_vertical/位置)
│   ├── occupancy_config.py         # 体素空间定义、18 类语义标签（权威来源）
│   └── actor_occupancy_mapping.py  # CARLA Actor type_id → 18 类的完整映射
├── sensors/
│   ├── camera_manager.py           # RGB(等距鱼眼 Bayer) + 深度相机管理器
│   └── semantic_lidar_sensor.py    # 语义激光雷达
├── processing/
│   ├── ground_truth_voxel_generator.py  # 体素生成核心逻辑
│   └── visibility_filter_simple.py      # 基于 LiDAR 的可见性过滤
├── data_utils/
│   └── data_saver.py               # 数据集保存器 (DNG + NPY，异步 IO)
├── main_collection.py              # 主采集脚本
├── visualize_dataset.py            # 数据集可视化工具
└── diagnose_ground_layer.py        # 地面层诊断工具
```

## 主采集流程（`main_collection.py`）

```
连接CARLA → 设同步模式(20Hz) → 生成ego车辆 → 生成NPC → 附加传感器(8相机+深度+LiDAR)
  → 等传感器就绪 → [World Tick → 相机采集 → LiDAR采集 → Ego Pose → 体素生成+Flow+可见性过滤 → 保存] × N帧
```

`setup_carla` 里设 `synchronous_mode=True, fixed_delta_seconds=0.05`（20Hz）——这是
必须的：不设同步模式，`world.tick()` 会直接超时（`time-out ... while waiting for the
simulator`），因为服务端不会等客户端的 tick 请求，是按自己的渲染节奏自由跑的。

## 体素生成逻辑（`GroundTruthVoxelGenerator`）

输出：
- `occupancy`: `(400,400,32)` uint8，语义类别 `[0-17]`
- `flow`: `(3,400,400,32)` float16，场景流 `(dx,dy,dz)`
- `flow_mask`: `(400,400,32)` uint8，动态区域掩码

流程：初始化空网格 → 用 Map API 填充静态环境（地面/道路/建筑，向下填充地下层统一用
Terrain）→ 按距离过滤（60m）+ OBB 光栅化填充动态 Actor → 计算相对速度生成场景流 →
基于 LiDAR 点云的可见性过滤（ID 聚类保留 + 地面强制保留 `Z≤1.0m`）。

**地面双重渲染问题**（Map API 和 Static Mesh 同时生成地面导致"双层地面"/"浮空灰层"）：
跳过 `Roads/Sidewalks/Terrain/Ground/RoadLines` 这几类 `EnvironmentObjects` 的 Static
Mesh 光栅化，地下填充统一用 `Terrain(14)`（不复制地表材质，否则出现灰色墙体），Actor
BBox 不覆盖已存在的地面体素（防止出现坑洞）。诊断工具：`diagnose_ground_layer.py`。

**性能优化**：世界坐标 Grid Index 做 Map API 查询结果的 Cache Key（首帧慢，后续帧
Cache 命中率接近 100%）+ NumPy 向量化批量赋值替代逐体素 Python 循环。

## 语义类别映射

18 类语义标签（对齐 nuScenes）权威定义见 `config/occupancy_config.py`，不要在文档里
再维护一份拷贝。`0: free` 在损失函数里权重必须 ≥1.0，`11: driveable_surface` 与
`13: sidewalk` 在可见性过滤中强制保留。

`actor_occupancy_mapping.py` 映射优先级：① `type_id` 精确匹配（如
`vehicle.tesla.cybertruck`→truck）② `CityObjectLabel` 映射（静态物体）③ 兜底规则
（`vehicle.*`→car，`static.prop.*`→general_object，`walker.pedestrian.*`→pedestrian，
这条是函数开头的整体 `startswith` 判断，`WALKER_MAPPING` 字典本身只是文档性质、不参与
实际判断）。

**核对映射表覆盖度**：`survey_actor_types.py` 枚举蓝图库全量 `vehicle.*`/
`walker.pedestrian.*`/`static.prop.*` 类型 + 当前地图已生成的 `traffic.*` Actor +
`get_environment_objects(Any)` 的全部 `CityObjectLabel`，逐一核对是否被
`actor_occupancy_mapping.py` 显式覆盖，用法见脚本文件头注释。2026-08-27 用它核对出
两处问题并已修复：
1. `static.prop.plantpot01/02/03/05/06/07` 没有和 `plantpot04` 一起归进 Vegetation，
   掉进了 general_object 兜底——同一类道具被不一致分类，已统一到 Vegetation(16)。
2. `traffic.unknown`（`CarlaEpisode.cpp` 对无法识别的 `ETrafficSignState` 的兜底
   type_id，`semantic_tags` 为空、样本里有 14m×11.2m 的巨大扁平包围盒，形状特征是
   路口/触发区域逻辑体而非真实可见路牌）：已从 `ground_truth_voxel_generator.py` 的
   光栅化 Actor 列表里排除。这类 actor 之前会被光栅化成 general_object，还会被
   `Z≤1.0m` 的地面高度保护规则强制保留、完全绕开可见性过滤，在体素真值里凭空多出
   一大块不存在的物体。

**可见性→free 的强制保留规则本身也有一处相关 bug**：`occupancy_config.py` 的
`GROUND_LABELS` 曾经是 `[11, 12, 13, 14, 6]`，多出的 "6" 是旧版 CARLA 语义标签
(RoadLines=6) 的输入编号误当成 18 类输出编号写了进来——18 类里 6 是 motorcycle，
导致所有摩托车被地面保护逻辑无条件强制保留，绕开了"不可见就归 free"的可见性过滤。
已修复为 `[11, 12, 13, 14]`。

**`traffic.*` 动态 Actor 不应该用 `actor.bounding_box` 光栅化**：2026-08-28 用户报告
"车左后方路面出现一个图片里没有的白块"，给 `main_collection.py` 加了 `--debug-actor-ids`
调试开关（保存每帧 `debug_actor_ids/*.npy`，即 `generate()` 内部用的
`(400,400,32) int32` actor_id 网格：正数是真实 CARLA `actor.id`，负数是
`_fill_static_environment()` 给环境物体分配的虚拟ID `-(i+10000)`，`i` 是
`world.get_environment_objects(carla.CityObjectLabel.Any)` 返回列表的下标，可以
反查回具体是哪个环境物体。默认不生成，训练不需要。`dataset_viewer_v2` 配了对应的
`GET /api/actor_id/<frame_id>?x=&y=&z=` 端点，点体素时如果数据集有这个目录会在
tooltip 里附带显示 actor_id）直接定位到具体 actor：`traffic.traffic_light`。
实测 (`world.get_actors().filter('traffic.traffic_light')`) 它的 `bounding_box` 和
自己的 `transform.location` 能差 9~12 米（例如某个灯 `loc=(-34.4,-51.0,0.3)` 但
`bb.location=(-9.0,8.5,1.0)`，换算世界系中心在 `(-43.4,-42.5,1.3)`）——这是 CARLA
给红绿灯用的控制/触发体积，不是贴合灯杆网格的可见几何，光栅化出来就是一个和任何
可见物体都对不上的悬浮方块。而且 `traffic.*` 的 `type_id` 在
`actor_occupancy_mapping.py` 里没有精确匹配规则，会掉到 `semantic_tag` 兜底得到
`manmade(15)`，跟 `_fill_static_environment()` 那边 `TrafficLight`/`TrafficSigns`
环境物体统一映射到 `traffic_cone(8)` 还对不上，是两条互相矛盾的分类路径。真实可见的
红绿灯/路牌几何本来就已经由 `world.get_environment_objects()` 的
`TrafficLight`/`TrafficSigns`/`Poles` 类别在 `_fill_static_environment()` 里用贴合
mesh 的世界系 AABB 正确光栅化了，`traffic.*` 动态 Actor 这条路径纯属重复且不可靠。
修复：`ground_truth_voxel_generator.py` 的 `generate()` 不再把 `traffic.*` 塞进
`all_actors` 参与光栅化（之前 `traffic.unknown` 的单独排除写法已经不需要了，整个
`traffic.*` 都不再走这条路径）。用 `find_floating_manmade.py` 风格的连通分量扫描
(按 `debug_actor_ids` 反查虚拟/真实ID) 验证：修复前 15 帧里能扫到 151 个孤立小型
`manmade` 连通分量，修复后只剩 2 个，且反查到的是一个真实 `Buildings` 环境物体、
AABB 数值正常（没有离谱偏移），判定为真实建筑基座边缘，不是 bug。

## 使用方法

```bash
conda activate carla

# 采集
python occnetv3_data_generator/main_collection.py \
    --frames 1000 --output dataset_10k --town Town10HD \
    --num-vehicles 30 --num-walkers 10 --raw-bit-depth 12

# 可视化
python occnetv3_data_generator/visualize_dataset.py --dataset dataset_10k --sample 0
```

主要参数：`--frames`（默认10）、`--output`（默认 `dataset_10k_bak`）、`--town`
（默认 `Town10HD`）、`--num-vehicles`（默认30）、`--num-walkers`（默认10）、
`--raw-bit-depth`（`8/10/12/14/16`，默认12）、`--clear-output`（生成前清空输出目录，
默认开启）。

采集前需要 CARLA 编辑器已经进入 Play/PIE 状态（RPC 端口 2000 才会起来）。编辑器带
`-CarlaAutoPlay` 命令行参数启动可以省掉手动点 Play 这一步（见
[`CARLA_BUILD_NOTES.md`](../CARLA_BUILD_NOTES.md)）。

## 输出数据格式

```
dataset_10k/
├── calibration/
│   ├── intrinsics.json      # 相机内参(fx,fy,cx,cy,fov,fov_vertical,distortion.model=equidistant) + 顶层 raw_bit_depth
│   └── extrinsics.json      # 相机安装外参 (Camera→Vehicle，恒定)
├── images/scene_XXXX_frame_YYYY/cam_{0-7}.dng   # Bayer RGGB DNG，位深见 raw_bit_depth
├── depth/scene_XXXX_frame_YYYY/cam_{0-7}.npy    # (960,1280) float32，单位: 米
├── occupancy/scene_XXXX_frame_YYYY.npy          # (400,400,32) uint8
├── flow/scene_XXXX_frame_YYYY.npy               # (3,400,400,32) float16
├── flow_mask/scene_XXXX_frame_YYYY.npy          # (400,400,32) uint8
├── ego_pose/scene_XXXX_frame_YYYY.npy           # (4,4) float32, Vehicle→World
├── ego_motion/scene_XXXX_frame_YYYY.npy         # (4,4) float32, inv(pose_t)@pose_{t-1}
├── train.txt / val.txt / test.txt
```

DNG 保存走线程池异步 IO（4 workers），不阻塞主采集线程；`piexif`/`Pillow` 不可用时
自动降级为 `.npy`。

## 常见问题

**Q: 地面上出现灰色/白色浮空层？** Map API 和 Static Mesh 双重渲染地面，已在
`ground_truth_voxel_generator.py` 里跳过地面类型的 Static Mesh 光栅化。

**Q: 地下出现灰色墙体？** 向下填充复制了地表材质，已改为统一填充 `Terrain(14)`。

**Q: 相机拍到车内/车架？** 检查 `camera_config.py`：左侧相机 Y 必须是负值，后视相机
X 要向车尾外延，侧后视相机 Y 绝对值要够大。

**Q: DNG 文件无法加载？** OpenCV 不支持 CFA（`PhotometricInterpretation=32803`）
格式，需要 `pip install rawpy`（或 `Pillow + piexif`），否则数据采集会自动降级为
`.npy`。

**Q: 采到的图像是花屏/条纹/全是极端值？** 如果是自己改了 C++ 相机/序列化代码后出现
这种情况，先看"底层实现"一节列的 5 个坑是不是又踩中了某一个——尤其是 #4
（`ImageSerializer::Deserialize`），历史上表现最隐蔽：服务端渲染完全正常，问题只在
客户端反序列化这一步。

**Q: `world.tick()` 超时？** 确认 CARLA 编辑器已经点了 Play（或带
`-CarlaAutoPlay` 启动且已经等到自动触发）——没进入 PIE 状态时 RPC 端口没有真正在
tick，同步模式下的 `tick()` 请求会等不到响应直接超时。

## 性能参考

2026-08-27 实测（RTX 4090，Town10HD，20 NPC 车辆 + 10 行人，等距鱼眼相机迁移后）：
首帧（Map API Cache 冷启动）约 12s，后续帧约 2.5-3s/帧，其中体素生成+Flow+可见性
过滤占大头（约 1.4-10s，取决于当帧场景变化量），相机+LiDAR 采集本身只有几毫秒。这是
5 帧小批量冒烟测试的数字，不是大规模采集的稳态基准，正式跑 `dataset_10k` 之后应该
用实际日志重新统计并更新本节（不要沿用这几个数字当作长期性能保证）。

## 参考

- [`../CARLA_BUILD_NOTES.md`](../CARLA_BUILD_NOTES.md) — CARLA + UE5 引擎获取/编译/踩坑记录
- [`../CLAUDE.md`](../CLAUDE.md) — 仓库整体结构
- [`../e2e_occ/README.md`](../e2e_occ/README.md) — 消费这份数据的网络
- [Adobe DNG Specification](https://helpx.adobe.com/camera-raw/digital-negative.html)
