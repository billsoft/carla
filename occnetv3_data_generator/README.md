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
（`vehicle.*`→car，`static.prop.*`→general_object）。

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
