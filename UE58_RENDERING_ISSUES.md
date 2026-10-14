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
