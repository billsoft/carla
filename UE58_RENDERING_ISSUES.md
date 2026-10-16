# UE5.8 渲染 / 真实感：待解决问题清单

> 建立：2026-10-14。测试环境：服务端 192.168.100.81（RTX 4090，UE5.8 Development 包），客户端本机 `carla58`。
> 对照图都在 `D:\code\carla_ue58\outputs\`（`lighting_preview\`、`exposure_sweep\`、`smoke58_weather\`）。
> 状态：🔴 未解决 / 🟡 有临时规避 / 🟢 已解决

## 当前"能跑通"的基线配置（2026-10-14）

- 服务端：`-quality-level=Epic`，Lumen 改用硬件光追（`r.Lumen.HardwareRayTracing=True`），光追几何内存 8000MB，
  阴影保持虚拟阴影贴图（VSM）。见 `remote/carla_remote.ps1` 的 `$CarlaArgs`。
- 鱼眼相机：直方图自动曝光（EV100 7~15），6 个 cube 面共享面 0 的曝光，**无接缝**（问题 1/2 已修）。
- 采集：`main_collection.py --weather random`，完整白天范围（太阳 8~70°、晴/少云/阴）。
- 16 路相机每帧约 1.25 s，瓶颈在传输/同步，不在 GPU。

---

## 1. 🟢 鱼眼相机曝光不能自适应（阴天/低太阳欠曝）—— 2026-10 已修

**现象**：默认档位下曝光基本固定（约 10–12 EV），低太阳、阴天场景整体偏暗（亮度均值 ~70/255，正午 ~110）。
随机天气后这类数据占比很大，真实车载相机会自动曝光把它们拉亮。

**已确认的原因**（2026-10-14 在 81 实测）：
- `sensor.camera.rgb_fisheye` 的 `exposure_mode / exposure_min_bright / exposure_max_bright / exposure_compensation /
  iso / shutter_speed` 在 5.8 上**不生效**：属性还注册着（`set_attribute` 不报错），但输出亮度怎么改都不变。
  5.8 的曝光只由 `post_process_profile`（`Content/Carla/Config/PostProcess/*.json`，打在 pak 里）+ 天气系统给的
  曝光范围决定。
- 换成带直方图自动曝光的档位（AutomotiveHDR / WideHDR8MP）亮度能自适应（低太阳与正午都 ~103），但出现问题 2 的接缝。

**修复**：`ActorBlueprintFunctionLibrary.cpp` 的 `SetCamera(..., ASceneCaptureSensor_WideAngleLens*)` 在没有加载
`post_process_profile` 时把 `exposure_mode/compensation/min/max/speed` 等属性写到 6 个面的后处理设置上
（之前 5.8 上游只注册不消费）；`camera_config.py` 设直方图模式、EV100 7~15。实测低太阳 Default 与 ClearNoon
画面均值都在 ~120（修复前 70/109）。`choose_weather` 的临时亮场景限制已撤销（2026-10-14）。

## 2. 🟢 自动曝光时鱼眼图出现亮度接缝 —— 2026-10 已修

**现象**：AutomotiveHDR 等档位下，鱼眼图顶部出现弧形亮带、中间出现竖直接缝（`outputs\exposure_sweep\compare_default_vs_autohdr.png`）。

**原因**：CARLA 的鱼眼（`ASceneCaptureSensor_WideAngleLens`）是 6 个 `USceneCaptureComponent2D` cube 面分别渲染再
畸变拼接；每个面有独立的 view state，直方图自动曝光**各面各自测光**，相邻面亮度不同。这是 cube-map 鱼眼方案的
固有问题（上游代码注释已把 cube-map 相机族标为 deprecated，推荐 `sensor.camera.rt_lens`），不是 Bayer RAW 改动引入的。

**修复**：引擎本身有共享曝光的机制（`FSceneViewInitOptions::ExposureSceneViewStateInterface`），只是 scene capture
没接上。引擎补丁 `Util/EnginePatches/ue58-scene-capture-shared-exposure.patch` 加了
`SceneCaptureExposureSharing::SetExposureSource()`，`SceneCaptureRendering.cpp` 建 view 时把面 1~5 的曝光状态指向
面 0；`SceneCaptureSensor_WideAngleLens::BeginPlay` 注册，并关掉各面独立的局部曝光/泛光/镜头光晕（同样逐 view
计算、会造成接缝）。已打包部署 81，200 帧数据抽查无接缝。重拉引擎后要重新 `git apply` 该补丁。

## 3. 🔴 Bayer RAW 不是线性 HDR

**现象**：鱼眼各面 `CaptureSource = SCS_FinalColorLDR`（`SceneCaptureSensor_WideAngleLens.cpp`），Bayer 数据是从
**色调映射后的 LDR 图**再马赛克得到的，不是物理线性的传感器 RAW；高光/暗部被色调曲线压缩。

**影响**：网络学到的是"ISP 之后的图"的统计，不是真实 RAW。真车部署若直接吃传感器 RAW，域差较大。

**根治方向**：面捕获改用线性 HDR（如 `SCS_SceneColorHDR` 或 FinalColorHDR 去掉 tonemapper），曝光仍需共享（与问题 2 一起做）。
会改变数据分布，旧数据不可混用。

## 4. 🟢 rt_lens（路径追踪相机）降噪不可用 —— 已找到根因并在 81 修复

**现象**：日志 `LogDLSSRR: Warning: DLSS Ray Reconstruction is not available on this system (driver/GPU/snippet).`，`rt_lens` 满屏噪点。

**根因**：DLSSRRDenoiser 插件只静态链接 NGX 加载器（`nvsdk_ngx_d.lib`），DLSS 模型 DLL（`nvngx_dlssd.dll` 光线重建、
`nvngx_dlss.dll` 超分）运行时才加载，只在 exe 目录或 `$DLSS_SDK/lib/Windows_x86_64/rel` 找；`DLSS_SDK` 只在打包命令里设过、
运行时没有，包里也没拷 DLL。本机 `Saved\nvngx.log`：`NGXLoadFromPath failed: -1160773628`（0xBAD00004 FeatureNotFound），
功能未被禁用、项目 ID 正常——纯粹找不到文件。依据 `Engine\Plugins\Experimental\DLSSRRDenoiser\...\DLSSRRNGX.cpp:144-160,224-228`。

**修复**：
- 81：已把 `C:\Users\bills\SDKs\DLSS\lib\Windows_x86_64\rel\nvngx_dlss{d,}.dll`（310.9.1）拷到
  `D:\carla-server\Windows\CarlaUnreal\Binaries\Win64\`，日志出现 `DLSS Ray Reconstruction initialized (D3D12 NGX)`，
  rt_lens 画面干净（`outputs\lighting_preview\rtlens_before_after_dlssrr.png`）。
- 永久：`Unreal/CMakeLists.txt` 打包后处理新增"拷贝 DLSS NGX snippet"步骤（下次打包起自动带上，重新部署不会丢）。
- 本机编辑器：用 `DLSS_SDK=C:\Users\bills\SDKs\DLSS` 环境变量启动，或把两个 DLL 拷到引擎 `Engine\Binaries\Win64\`。

**意义**：`rt_lens` 是**单视图**直接按等距投影路径追踪，整幅图只有一个自动曝光——不存在问题 2 的接缝，且物理最准；
配套 `rt_lens_distance` 直接给沿射线距离（正是数据集要的深度监督）。是长期的最优真实感路线，代价：需给 rt_lens 加
Bayer RAW 输出（C++），16 路相机要整体换成 rt_lens/rt_lens_distance（不能与光栅化相机同帧），并实测性能。

**另**：`rt_lens` 与光栅化相机（含鱼眼的 6 个面）在同一帧渲染会触发引擎断言
`IsCompatible(Renderer->ViewFamily.EngineShowFlags)` 导致服务端崩溃（2026-10-14 本机实测）。要用 rt_lens 采集，
需要整套相机都换 rt_lens，或分帧渲染。

## 5. 🟢 已解决（记录备查）

- 运行窗口光影比编辑器视口差：是测试脚本把天气改成 ClearNoon（顶光）且没恢复、同步模式残留。脚本已改为退出时恢复，另有 `remote/reset_carla_world.py`。
- `REQUESTED MEMORY OVER BUDGET 1.976 GiB / 1.953 GiB`：光追几何内存池（默认 2000MB）超预算，已提到 8000MB。
- 光追阴影 `r.RayTracing.Shadows`：不开。光追不支持带 WPO（风摆）的植被，树/草阴影会错；VSM 无此问题，性能无差别。
- DNG 在第三方看图软件里很暗：DNG 未写 WhiteLevel 标签（软件按 16 位解释），训练侧按 `intrinsics.json` 的 `raw_bit_depth=12` 归一化，不受影响。

---

## 6. 🟢 体素真值与照片不对应（2026-10-14，数据侧，label_version 2）

**现象**：预览里大量体素在照片里根本看不到（地下整列、建筑/车内部、被遮挡区域），以及车道线/路牌被标成锥桶。
**审计**：`occnetv3_data_generator/compute_camera_mask.py --stats`——只有 ~1/3 占用体素被相机观测到；
未观测的顶面体素里 >90% 是确实被挡住（深度图上视线先打到更近的表面），不是掩码算法漏标。
**修复**：相机可见性掩码 `mask_camera`（训练忽略未观测体素，Occ3D 约定）、语义映射修正、深度相机 `depth_fov`
覆盖整幅鱼眼、DNG 标签补全、按帧号对齐传感器数据、`--frame-skip`、批量采集按场景划分。详见
`occnetv3_data_generator/README.md` 的"相机可见性掩码"与"label_version 2"两节。旧数据需重采才能得到新标签。

---

## 7. 🟡 服务端在 RPC 线程崩溃退出（2026-10-15，有自动恢复规避）

**现象**：81 上的服务端无人操作时自行退出。日志 `D:\carla-server\Windows\CarlaUnreal\Saved\Logs\CarlaUnreal.log`：
`std::deque<clmdep_msgpack::v1::sbuffer>::_Xlen()`（rpclib 发送队列长度溢出异常）→ AbortHandler。
发生在一次 Town15 采集结束、客户端销毁全部 actor 约 1 分钟后，当时只剩 81 桌面上的 `CARLA-Viewer` 预览连着。
此前（2026-10-14）也出现过一次"预览窗口首次连接时数组越界崩溃"，疑似同一类与预览客户端相关的 RPC 并发问题。

**规避**：`batch_collection.py --auto-restart` 每个场景前检查服务端，连不上自动 `carla_remote.ps1 -Action restart`。

**待查**：用 Development 包的 PDB 符号还原完整调用栈；复现条件（切大地图 + 预览客户端在线 + 清空 actor）；
必要时批量采集期间先停掉 81 上的 `CARLA-Viewer` 任务对比。

---

## 8. 🟢 长焦相机源捕获面的余量 / 显式内参 / 纹理流送（2026-10-15）

背景：单面长焦（`front_main` 50°、`front_narrow` 35°）2026-08 起已按自身视场拍前向 cube face
（`CustomProjectionMatrix`），是真正的角分辨率提升，不是裁切放大。2026-10-15 评审指出三处可改进，已在
`SceneCaptureSensor_WideAngleLens.cpp` 修复：

- **捕获范围按射线边界推导**：新增 `PixelToRay()`/`ComputeLensCoverage()`，用与 `WideAngleLens.usf` 完全相同的
  像素→射线公式（显式内参 / 焦距+主点 / 各镜头模型）扫描输出图像全部边界像素和内部网格，前向面半视角 =
  atan(最大切线 × (1 + 3/面边长))（1.5 个源纹素的双线性余量）。之前是 `max(FOV)·0.5·√2` 的经验余量：
  35° 长焦源捕获面 49.5°、50° 主相机 70.7°；现在约 35.7° / 51.9°，同样 1280 的源纹理覆盖更窄的角度，
  中心源采样密度再提高约 1.4 倍。
- **显式内参不再关闭长焦增益**：之前 `!bExplicitIntrinsics` 才允许收窄；选面也只看 FOV 字段，显式内参视野
  比 FOV 字段宽时 shader 会采样没渲染的面。现在选面与收窄都来自同一份射线覆盖（LUT1D、FOV 圆形掩码
  仍走旧逻辑）。`SetPrincipalPoint()` 也会重新计算。
- **纹理流送按实际投影报告**：`PrePhysTick` 用面边长与实际捕获半角调用 `AddViewInformation`，前向面组件
  `FOVAngle` 也同步为实际捕获角（之前固定 90°，长焦放大了远处物体，流送仍按宽视野估算）。

Python 侧：`camera_manager.py` 8 台相机共用一个蓝图对象，逐相机的可选镜头参数（fx/fy/cx/cy、fov_horizontal、
k0~k3）每台先恢复默认再覆盖，避免串到下一台。

**仍未做**：曝光共享的测光区域是前向源面，不等于最终鱼眼画面（广角相机侧面明暗代表性不足）；
Bayer RAW 仍来自色调映射后的 LDR（问题 3）。

## 9. 🟢 鱼眼相机比同视场针孔糊一个量级 / 没有 Lumen（2026-10-15）

**现象**：用户反馈"仿真截图像低画质游戏"。81 同帧同位姿对照（中心区域拉普拉斯方差，鱼眼/同水平视场针孔）：
front_narrow 35° 只有 0.03、front_main 50° 0.15、90° 相机 0.3~0.4、120° 相机 0.5~0.66——视场越窄越糊，
与物理相机（同分辨率下长焦角分辨率更高）正好相反。逐步排除：Bayer 采样（采样前 RGB 与采样后几乎一样）、
cube 面 mip/采样器（`SampleLevel(…, 0)`）、渲染百分比（默认 100）、运动模糊（`r.MotionBlurQuality=0` 无变化）。

**根因与修复**（`SceneCaptureSensor_WideAngleLens.cpp`、`WideAngleLens.usf`、`CameraModelUtil`）：

1. **地图后处理体积的景深作用在鱼眼各面上（长焦发糊的主因）**。scene capture 会混入地图后处理体积
   （`FSceneView::StartFinalPostprocessSettings` → `UWorld::AddPostProcessingSettings`）；针孔对约 40 个相机字段设了
   `bOverride_`，鱼眼各面只有 3 个。景深模糊随焦距放大、与面分辨率无关：front_narrow 面加到 4096 也不变，
   启动参数加 `r.DepthOfFieldQuality=0` 后从 0.07 倍变为 1.30 倍（像素对齐、分块中位数）。各面现在关掉景深
   （车载定焦镜头近处到无穷远都清晰），色差/胶片颗粒覆盖为 0。
2. **鱼眼各面一直没有 Lumen**。引擎对 scene capture 默认把 GI/反射设为 None、表面缓存减半，只有组件自己的
   后处理设置能重新打开（`SceneCaptureRendering.cpp`）。针孔有，鱼眼没有。各面现在使用与针孔相同的 Lumen 设置。
3. **cube 面上的 TSR 历史累积失败**：同帧收敛曲线针孔 268→760→1948 持续上升，鱼眼 423→104→286 不升反降；
   TSR 对运动的车/人还会拖影。RGB 鱼眼改为确定性超采样：各面关掉时序 AA，面按约 2 倍角密度渲染，
   畸变 shader 对每个输出像素取 4 个旋转网格子样本求平均（像素面积积分，`WAL_SHADER_FLAGS_SUPERSAMPLE`，
   只对 RGB 开，深度/语义/实例分割不受影响）。代价：帧间逐像素时域噪声约 2/255（针孔 0.5），来自残余锯齿和 Lumen。
4. **面尺寸按镜头有效像素密度推算**（`ComputeRequiredFaceSide`）：逐点把相邻输出像素投到所在 cube 面上，
   要求面纹素间距 ≤ 输出像素间距 / `Carla.WideAngleLens.FaceOversample`（默认 2.0，上限
   `Carla.WideAngleLens.MaxFaceSide` 默认 2048）。之前固定 max(宽,高)=1280，90° 相机面中心每弧度只有 640 像素、
   输出要 815，等于放大采样。`face_size` 属性 > 0 时仍按它（`camera_config.py` 的 `face_size` 可不重编回退）。
5. `motion_blur_*` 属性以前注册了却从没应用（各面一直是引擎默认 0.5），现在按属性应用，采集配置为 0 即关闭。

**代价**（81，8 相机 + 深度 + 语义生产配置）：渲染部分（tick + 取相机数据）约 6.8 s/帧（面上限 2048），
上限 1536 时约 4.9 s/帧；显存采集时稳定在 23.3~23.9 GB（UE 按预算填满纹理/光追池，采集结束回落到约 8 GB），
两种配置都稳定跑完。要降负担可不重编，在 `carla_remote.ps1` 的 `$CarlaArgs` 里加
`[SystemSettings]:Carla.WideAngleLens.MaxFaceSide=1536,[SystemSettings]:Carla.WideAngleLens.FaceOversample=1.5`。
**81 的默认启动参数（`carla_remote.ps1 $CarlaArgs`）已采用这组较轻的设置**，纹理流送池 4000→3000 MB。
光追几何常驻池保持 8000 MB：试过 4000 MB，采集中出现光追 shader 显存页错误（`MMU Fault`, `ray_tracing_*`）
GPU 崩溃（同时有采集中途销毁/生成 NPC，两个因素都已撤掉）。

**显存真实占用（2026-10-16 更正）**：之前根据一次崩溃日志的 Local Used 15.4 GB 判断"nvidia-smi 的 23.5 GB 多是驻留
预留"，这是错的。批量采集默认配置（40 车 30 人）下进程计数器 `GPU Process Memory` 专用显存 23.5 GB 是真占满，
共享内存（溢出到系统内存）每分钟涨约 1.5 GB；10-15 18:58 那次崩溃日志是 Local Used 6.9 GB、**System Used 31.6 /
31.9 GB**——共享内存耗尽后 GPU PageFault。用服务端 `console_command` RPC 发 `r.DumpRenderTargetPoolMemory`：
渲染目标池 24.4 GB / 1605 张，大头是 **24 个鱼眼面各一份 per-view Lumen 场景**（表面缓存图集 4096²，约 360 MB/份，
合计约 9 GB）、Lumen 辐射缓存（每面 2 份，约 4.7 GB）、体积云重建历史（约 3.5 GB，其中 16 个深度/语义针孔相机约 1 GB）。
修复：同一辆车上所有鱼眼相机的各面共享同一个视点锚（同车最早生成的鱼眼相机的前向面；
`USceneCaptureComponent2D_CARLA::ShareViewOrigin`，即引擎 cube 相机各面用的 `FSceneViewStateReference::ShareOrigin`，
在每个相机首次捕获前设置——CARLA 先 spawn 再 attach，BeginPlay 时还找不到同车相机），Lumen 场景从 24 份降到 1 份；
不开后处理的相机（深度/语义）关掉体积云；光追几何池 8000→3000 MB（它是"未被 TLAS 引用的几何也留着"的缓存，
引擎默认 400 MB）。渲染目标池 24.4→15.2 GB，输出画面不变。锚相机先销毁时会先释放所有共享方，共享方随后各自
重新分配，不会触发 `~FSceneViewStateReference` 的断言。

**抽帧只在保存帧渲染（`capture_stride`，2026-10-16）**：新增所有传感器通用的 `capture_stride` 属性，服务端只在
`FCarlaEngine::GetFrameCounter() % capture_stride == 0` 的帧捕获/发数据（`ASensor::Tick`）。没用 `sensor_tick`：
它走 UE tick 间隔 + 超时补偿，浮点下偶尔 N±1 步才触发，图像会和客户端取的世界状态错帧。`main_collection.py`
采集前把保存帧对齐到 `frame_skip` 的倍数。深度/语义相机默认开（无时间累积，零画质影响）；RGB 鱼眼默认不开：
81 同场景 A/B，RGB 只在保存帧渲染时暗部平坦区噪声高 30~50%（Lumen 降噪少了中间几步的时间累积），
需要提速时加 `--rgb-saved-frames-only`。另外相机侧后处理/流场/保存移到后台线程与下一帧仿真并行。
81 实测每帧 约 22 s → 约 12 s（RGB 每步渲染）/ 约 7.6 s（`--rgb-saved-frames-only`）。

**`output_stride`（2026-10-16）**：`stat dumpframe` 显示一步 1.68 s 渲染线程时间里 1.2 s 是 `FlushBatchedReadbacksCmd`
（每 tick 一次 GPU 同步，等这一步全部渲染完再读回），24 个视图的场景渲染本身只有 0.35 s，GPU 利用率约 57%。新增
所有传感器通用的 `output_stride` 属性（`ASensor::ShouldOutputThisFrame`），鱼眼相机不输出的帧照常
`EnqueueRenderSceneImmediate()` 但不读回/发送：RGB 每步渲染保持 Lumen 时间累积，同时只有保存帧有 GPU 同步。
配合客户端改动后 Town10 约 12 s → 约 8 s/帧，Town15 约 29 s → 约 9~14 s/帧。

**附带修复**：`carla_remote.ps1 -Action restart` 之前不把 `-CarlaArgs` 传给子调用（自定义 cvar 被静默丢掉）；
`dataset_viewer_v2` 解码 DNG 不再按线性 RAW 叠伽马/自动亮度（数据本身是色调映射后的 sRGB 编码，问题 3），
之前预览发白发灰。`Carla.WideAngleLens.DumpAllFrames.Cubemap` 导出 8 位面渲染目标会触发 D3D12 格式断言崩溃，
不要在服务端打开。

**仍未做**：镜头光学 MTF/色差/渐晕建模；线性 RAW（问题 3）；地图体积的调色仍会混进鱼眼（针孔用
`ApplyPostProcessVolumeToSensor` 选择性吸收，两者调色可能略有差异）。
