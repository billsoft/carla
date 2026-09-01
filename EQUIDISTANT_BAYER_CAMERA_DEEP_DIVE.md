# 从零实现等距投影 Bayer RAW 相机：算法、UE5 引擎机制与工程实践

> 本文记录本仓库在标准 CARLA 之上新增的一套相机能力——**等距投影(equidistant)/鱼眼相机 + 原生 Bayer RAW HDR 采集**——从数学模型到 UE5 引擎实现再到实际采集应用的完整链路。所有代码引用均对照本仓库当前源码，可直接按 `文件路径:行号` 定位到实现。

## 写在前面：我们在解决一个"反过来"的问题

计算机视觉里"相机模型"通常用来做一件事：**去畸变**。你有一台真实鱼眼镜头拍出来的照片，画面边缘挤压变形，用标定得到的内参 + 畸变系数(k1~k4)把它反解回一张"看起来像针孔相机拍的"矫正图，供后续算法（检测、SLAM）使用。OpenCV 的 `cv2.fisheye.undistort`、`cv2.fisheye.initUndistortRectifyMap` 做的都是这件事。

本仓库要解决的是**完全反过来的问题**：CARLA 引擎里的虚拟相机是"完美"的——它是纯几何光线投射的结果，没有任何镜头带来的非线性、没有色散、没有边缘畸变。但我们训练感知网络（`e2e_occ/`）要用来处理的，是车上装的**真实、不完美的物理鱼眼镜头**拍出来的画面。如果仿真数据是"完美"的，真实数据是"不完美"的，两者之间就存在一条系统性的域间隙(domain gap)。

于是我们做的事情是：拿一个真实鱼眼镜头标定出来的相机模型（等距投影，或者更精确的 Kannala-Brandt 多项式 + k1~k4），**正向**套用在 CARLA 这台完美的虚拟相机上，让它输出的画面带上和真实镜头一致的几何特征——FOV 压缩规律、边缘密度变化、（如果标定了具体的 k1~k4）连那台镜头独有的畸变"指纹"都能复现。这不是去畸变，是**造畸变**——用完美的射线追踪能力，去模拟一台本不完美的物理镜头。

本文分三部分：

- **第一部分**：广义相机模型的数学——r=f(θ) 家族公式、Kannala-Brandt 多项式里 k1~k4 到底是什么、内参外参各自在哪一层生效。
- **第二部分**：UE5 引擎机制——原生摄像机为什么做不到这件事、cubemap 分而治之的思路、Compute Shader 如何把第一部分的公式变成 GPU 上的实际渲染、以及一个真实的"算法驱动引擎改动"案例（窄 FOV 清晰度修复）。
- **第三部分**：工程应用——蓝图属性表、8 相机环视配置、数据落盘为 DNG 的完整链路、以及若干实践中踩过的坑。

---

# 第一部分：算法基础——广义相机模型与"反向畸变"

## 1.1 广义相机模型：r = f(θ) 家族

所有针孔以外的相机模型，本质上都在回答同一个问题：**一条从光心出发、与光轴夹角为 θ 的入射光线，最终会落在像平面上距离主点(principal point)多远的地方（半径 r）？**

不同镜头设计对应不同的 r(θ) 函数。本仓库在 shader 层（`Unreal/CarlaUnreal/Plugins/Carla/Shaders/CameraModelCommon.ush:23-28`）和 C++ 层（`Unreal/CarlaUnreal/Plugins/Carla/Source/Carla/Util/CameraModelUtil.cpp:477-547`）各实现了一份完全对应的六种模型：

| 模型 | 公式 r(θ) | 物理含义 |
|---|---|---|
| Perspective（针孔） | `r = f·tan(θ)` | UE5 原生投影用的模型，θ→90° 时 r→∞，FOV 理论上限 <180° |
| Stereographic（体视投影） | `r = 2f·tan(θ/2)` | 保角（conformal），常见于星图/鱼眼艺术效果 |
| **Equidistant（等距）** | `r = f·θ` | 像面半径与入射角**线性**成正比——"真鱼眼"的经典定义，本仓库默认模型 |
| Equisolid（等立体角） | `r = 2f·sin(θ/2)` | 保持单位像素对应的立体角不变，常见于真实测光鱼眼镜头 |
| Orthogonal（正交） | `r = f·sin(θ)` | θ=90° 时 r 打满，超过 90° 无法投影 |
| Custom / Kannala-Brandt | 见 1.2 | 多项式修正的等距模型，可拟合任意真实标定结果 |

这六个分支在两处代码里逐字对应：

```hlsl
// CameraModelCommon.ush:104-132（正向：给定角度求半径，f 已归一，θ 为弧度）
float ComputeDistancePerspective(float Angle)      { return tan(Angle); }
float ComputeDistanceStereographic(float Angle)    { return tan(Angle * 0.5F) * 2; }
float ComputeDistanceEquidistance(float Angle)     { return Angle; }
float ComputeDistanceEquisolid(float Angle)        { return sin(Angle * 0.5F) * 2; }
float ComputeDistanceOrthogonal(float Angle)       { return sin(Angle); }
```

```hlsl
// CameraModelCommon.ush:174-197（反向：给定半径求角度，即上面公式的反函数）
float ComputeAnglePerspective(float Distance)      { return atan(Distance); }
float ComputeAngleStereographic(float Distance)    { return atan(Distance * 0.5F) * 2; }
float ComputeAngleEquidistance(float Distance)     { return Distance; }
float ComputeAngleEquisolid(float Distance)        { return asin(clamp(Distance*0.5F,-1,1)) * 2; }
float ComputeAngleOrthogonal(float Distance)       { return asin(clamp(Distance,-1,1)); }
```

**关键点**：`ComputeDistance` 和 `ComputeAngle` 互为反函数。渲染管线里两个方向都要用——反向（半径→角度）用来把"某个输出像素在哪"翻译成"应该往哪个方向发一条射线去采样"；正向（角度→半径）用来在给定 FOV 的前提下反推焦距 f（1.4 节展开）。

## 1.2 Kannala-Brandt 多项式：k1~k4 到底在拟合什么

上面五种模型都是解析形式的理想曲线，现实中一支真实镜头的 r(θ) 曲线几乎不会精确落在任何一条上——光学设计的取舍、装配公差都会让它偏离理想曲线。工业界标定这种偏离的标准做法是给等距模型加一个多项式修正项，这就是 Kannala-Brandt 模型（也是 OpenCV `cv2.fisheye` 标定模块用的模型）：

```
θ_d(θ) = θ · (1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)
```

本仓库的实现（0-indexed，`k0,k1,k2,k3` 对应上式的 `k1,k2,k3,k4`）：

```hlsl
// CameraModelCommon.ush:57-74
float ComputeCameraPolynomial(float Theta)
{
    float Result = 1.0F;
    float Theta2 = Theta * Theta;
    float ThetaN = 1.0f;
    for (uint i = 0; i < CoefficientCount; ++i)
    {
        ThetaN *= Theta2;              // Theta^2, Theta^4, Theta^6, Theta^8 ...
        Result += Coefficients[i] * ThetaN;
    }
    return Result * Theta;             // θ·(1 + k0θ² + k1θ⁴ + k2θ⁶ + k3θ⁸)
}
```

展开正好是 `θ + k0θ³ + k1θ⁵ + k2θ⁷ + k3θ⁹`——和 OpenCV fisheye 标定输出的 `k1,k2,k3,k4` 是**同一套公式、同一个符号约定**（`occnetv3_data_generator/config/camera_config.py:14-16` 的注释也明确写了这一点）。这意味着：如果未来有真实车载鱼眼相机的标定数据，标定工具（OpenCV `calibrateCamera`、Kalibr 等）解出来的 k1~k4 可以**原样**填进 `camera_model=kannala-brandt` 的相机蓝图属性，不需要任何换算——引擎里跑的正是标定工具假设的同一条曲线，只是用途从"去畸变"换成了"造畸变"。

### 反函数没有解析解：牛顿迭代

正向公式（角度→半径）是纯多项式，好算；但反向（半径→角度，采样时真正要用的方向）没有解析反函数，只能数值求解。代码用牛顿迭代法，从 θ₀=r 开始迭代 32 次（`KANNALA_BRANDT_SOLVER_ITERATIONS`）：

```hlsl
// CameraModelCommon.ush:201-214
float Theta = Distance;
UNROLL
for (uint i = 0; i != KANNALA_BRANDT_SOLVER_ITERATIONS; ++i)
{
    float N = Distance - ComputeCameraPolynomial(Theta);
    float D = -ComputeCameraPolynomialDerivative(Theta);
    Theta -= N / D;   // 等价于 Theta += (Distance - Poly(Theta)) / PolyDeriv(Theta)
}
return Theta;
```

其中导数解析可推（`Result*Theta` 求导）：

```
d/dθ [θ + k0θ³ + k1θ⁵ + k2θ⁷ + k3θ⁹] = 1 + 3k0θ² + 5k1θ⁴ + 7k2θ⁶ + 9k3θ⁸
```

对应代码（`CameraModelCommon.ush:76-96`）逐项验证过与上式完全一致。这个 `UNROLL` 属性会让 HLSL 编译器把 32 次迭代**展开成 32 条连续指令**而不是一个循环——因为是逐像素在 GPU 上跑，固定次数展开比条件跳转更利于并行吞吐，是 shader 编程的常见优化手法。

## 1.3 内参：焦距 f 与主点偏移 (cx, cy)

r(θ) 公式里的 `f` 不是自由参数，而是由"配置的 FOV 必须精确映射到半个图像高度"这条约束反推出来的——这正是相机标定里"焦距"的定义方式：

```cpp
// CameraModelUtil.cpp:511-547（C++ 侧，构造相机/切换 FOV 时调用一次）
float ComputeDistance(ECameraModel CameraModel, float Angle, int32 ImageHeight, ...)
{
    const auto R = ImageHeight * 0.5F;   // 半个图像高度，单位：像素
    Angle *= 0.5F;                        // 传入的是全 FOV，这里取半角
    switch (CameraModel) {
      case ECameraModel::Equidistant: F = R / Angle; break;   // f = R / (FOV/2)
      ...
    }
    return F;
}
```

即：**焦距 = 半图高 / 半 FOV 弧度**（等距模型下）。这与真实相机标定完全一致——你标定出的 fx/fy 本质上也是"传感器物理尺寸"和"视场角"共同决定的一个派生量，不是独立可调的自由度。

主点偏移 `PrincipalPointOffset`（对应蓝图属性 `cx`/`cy`）模拟的是另一种真实镜头的不完美：**光心从来不会精确落在像素网格的几何中心**，装配公差、传感器切割误差都会带来几像素到几十像素的偏移。默认值 (0,0) 复现"完美"行为（光心=几何中心），只有显式设置才会偏移采样：

```hlsl
// WideAngleLens.usf:137-148（Center 就是 (ImageWidth/2+cx, ImageHeight/2+cy)）
const float2 FocalDistance = CameraParams.xy;
const float2 Center = CameraParams.zw;
const float2 UV = (float2(PixelPosition) - Center) / FocalDistance;
```

## 1.4 外参：为什么这一层的数学"不在"这里

需要澄清一个容易混淆的边界：本文讨论的所有公式，都发生在**相机局部坐标系**内——回答的是"给定一条相对于光轴夹角为 θ 的光线，它落在像面哪里"，或反过来"给定像面上一点，对应哪条局部射线方向"。至于这个相机装在车上什么位置、朝向哪个角度、车本身在世界坐标系哪里——这些是标准的刚体变换合成，UE5 的 Actor/Component 变换层级已经处理得很好，完全不需要在 Part 1/2 讨论的这套自定义数学里重新实现。

具体到代码：六个 `FaceCaptures[i]` 是 `USceneCaptureComponent2D_CARLA`，`SetupAttachment(RootComponent)` 挂在传感器 Actor 上，`SetRelativeRotation(...)` 只设置六个立方体面各自相对朝向（前后左右上下）。整个鱼眼相机 Actor 再作为普通 Actor 挂在车辆上、拥有世界坐标系变换——这条链路和标准针孔相机没有任何区别，UE5 的场景图会自动把"外参"（安装位置+朝向）应用在渲染前，Part 1/2 的射线数学全程只处理"局部空间里射线往哪个方向"这一个问题。

在应用层（第三部分），`occnetv3_data_generator/sensors/camera_manager.py::get_extrinsics()` 把每个相机在车辆坐标系下的安装 position/rotation 转成 4×4 的 Camera→Vehicle 矩阵，正是"外参"在这套系统里唯一存在的地方——它是标定/采集层的产物，不是渲染算法的一部分。

## 1.5 正向采样公式完整推导：逐行对照 shader

最终把以上所有内参、畸变模型串起来的，是 `WideAngleLens.usf::MainCS` 里对每个输出像素执行的这段代码（`WideAngleLens.usf:143-168`）：

```hlsl
const float2 UV = (float2(PixelPosition) - Center) / FocalDistance;  // 归一化半径坐标

Alpha = ComputeAngle(length(UV));   // ① 半径→角度：这是 r(θ) 的反函数，Alpha = θ

const float Phi   = PiHalf - Alpha;          // ② 转成"余纬度"
const float Theta = atan2(UV.y, UV.x);       // ③ 图像平面内的方位角

Direction = normalize(float3(
    sin(Phi),
    cos(Phi) * cos(Theta),
    -cos(Phi) * sin(Theta)));               // ④ 球坐标→局部空间 3D 单位射线方向
```

逐步展开这四步在做什么：

1. **① 半径→入射角**：`UV` 是像素到主点的偏移量除以焦距，量纲上就是 r(θ)/f。`ComputeAngle` 对当前选定的相机模型做反函数运算，还原出这个像素"应该"对应的入射角 θ（对等距模型这一步是恒等映射；对 Kannala-Brandt 模型这一步就是上面的牛顿迭代）。
2. **②③ 转球坐标**：θ 只是"离光轴多远"，还需要方位角 Theta（`atan2(UV.y,UV.x)`，即这个像素在圆周上的哪个角度方向）才能确定唯一射线。
3. **④ 求出实际 3D 方向**：把 (θ, azimuth) 转成局部空间的单位向量。这个方向就是"一台带有 θ(r) 畸变特性的真实镜头，会把哪个方向的光线画在这个像素上"。

拿到 `Direction` 之后，只需要向"完美"的立方体贴图（下一部分讲怎么来的）**采样一次**，把结果原样画在这个像素上（`WideAngleLens.usf:197`：`OutColor = SampleCubemap(Direction)`）。

这就是"反向应用畸变模型"的核心洞察：**不需要对渲染好的图像做像素级的形变/重采样（warp），而是对每一个输出像素反推它对应的真实入射光线方向，再直接从一个几何完美、没有任何畸变的球面全景捕获里取那个方向的真值颜色**。相比传统的图像 warp（先渲染一张图，再按畸变场对像素做插值搬运，容易在强畸变区域出现拉伸模糊或信息丢失），这种"每个输出像素精确对应一次真实采样"的方式，畸变越强、精度反而越有保证——因为它永远是在向着"应该采样哪个方向"这个问题要一个精确解，而不是在一张已经渲染完、信息已经离散化的图像上做二次处理。

> `Alpha` 在算出 `Direction` 之后还会 `*= 2.0F`（`WideAngleLens.usf:168`），这一步只用于随后的 FOV 渐隐遮罩（`fov_mask`/`fov_fade_size`）——因为 `YFOVAngle` 存的是**全** FOV，而这里的 Alpha 此时是**半**角，乘 2 只是为了让两者在同一量纲下比较，与射线方向计算本身无关。

---

# 第二部分：UE5 引擎机制——从原生摄像机到自定义等距相机

## 2.1 为什么不能直接给 UE5 摄像机设一个"等距投影"

UE5 的 `USceneCaptureComponent2D` 配合 `FOVAngle`，底层用的是 `FPerspectiveMatrix`（或 Reversed-Z 版本 `FReversedZPerspectiveMatrix`）——一个标准的**齐次线性投影矩阵**，数学上等价于第一部分表格里的 Perspective 模型（`r=f·tan(θ)`）。这类线性投影矩阵有一个硬性的数学限制：θ→90° 时 `tan(θ)→∞`，也就是说**理论上不可能表达 ≥180° 的 FOV**，实践中 UE5 也就到 170° 左右就已经严重失真。更不用说等距、等立体角这些非 tan 的非线性映射——UE5 原生投影管线根本没有"换一种 r(θ) 曲线"这个自由度，`FOVAngle` 只能控制 tan 曲线的陡峭程度，换不了曲线形状。

要做到任意 FOV（含全景 360°）、任意 r(θ) 曲线，必须完全跳出 UE5 原生投影矩阵这条路。

## 2.2 立方体贴图分而治之：6 张矩形图覆盖整个球面

思路和 VR/全景视频管线的经典做法一致：**没有一个线性投影能覆盖整个球面，但 6 个 90° 的线性投影拼起来可以**——这就是 cubemap。`ASceneCaptureSensor_WideAngleLens` 的构造函数（`SceneCaptureSensor_WideAngleLens.cpp:169-202`）建了 6 个 `USceneCaptureComponent2D_CARLA`，各自朝向前后左右上下：

```cpp
const FVector Forward[] = {
    FVector::ForwardVector, -FVector::ForwardVector,
    FVector::RightVector,   -FVector::RightVector,
    FVector::UpVector,      -FVector::UpVector,
};
...
FaceCapture->SetRelativeRotation(FRotationMatrix::MakeFromXY(Forward[i], Right[i]).ToQuat());
FaceCapture->bUseCustomProjectionMatrix = true;
FaceCapture->CustomProjectionMatrix = ProjectionMatrix;   // 90°(半角45°) 的标准透视投影
```

每一张面单独看仍然是普通的线性透视投影（UE5 原生能力完全够用，`FOV=90°` 远小于 170° 的极限），六张拼起来就覆盖了完整 4π 立体角。第一部分推导出的 `Direction`（真实入射光线方向），最终就是拿去这 6 张面里挑一张、算出对应 UV，取一次样：

```hlsl
// WideAngleLens.usf:46-108 SampleCubemap()：按 Direction 分量绝对值最大的轴判断该采样哪个面
if (DirectionAbs.x >= DirectionAbs.y && DirectionAbs.x >= DirectionAbs.z)
    FaceIndex = Direction.x < 0 ? 1 : 0;   // ±X → Back/Front
...
const float2 Tangent = Key.xy / Key.z;      // 切平面坐标（标准 cubemap 采样数学）
const float2 UV = (Tangent + float2(1,1)) * 0.5F;
```

### 智能跳过：不是每次都渲染全部 6 张面

全量渲染 6 张面成本是单张面的 6 倍——但大多数相机根本用不到全部 6 张面（比如一台 90° FOV 的相机，理论上只可能碰到前面这一张）。`ComputeCubemapRenderMask()`（`SceneCaptureSensor_WideAngleLens.cpp:532-567`）用一个安全余量（`fov_mask=false` 时用 √2，`fov_mask=true` 时用 1.0，对应是否允许边缘出现轻微暗角）判断真正需要渲染哪几张面：

```cpp
auto Mask = 1U << CubeFace_PosX;  // 默认只渲染前面
const auto FOV = FVector2D(GetFOVAngleX(), GetFOVAngleY()) * (GetFOVMaskEnable() ? 1 : Sqrt2);
if (FOV.Y > HalfPi) { Mask |= (1U<<CubeFace_PosZ) | (1U<<CubeFace_NegZ); }   // 需要上下
if (FOV.X > HalfPi) { Mask |= (1U<<CubeFace_PosY) | (1U<<CubeFace_NegY); }   // 需要左右
if (FOV.X > Pi || FOV.Y > Pi) { Mask |= 1U << CubeFace_NegX; }              // 需要正后方
```

`occnetv3_data_generator` 8 相机环视里，`front_narrow`（垂直 FOV 26.25°）和 `front_main`（37.5°）两台窄角相机的 FOV 乘以 √2 之后仍然远小于 90°，`ComputeCubemapRenderMask()` 结果只有前面这一张——**这两台相机实际只渲染 1/6 的面数**，是一次真实、可测量的 GPU 成本节省，而不只是"GPU 并行数学"这么泛泛而谈。

## 2.3 一次 Compute Shader Dispatch 完成全部畸变映射

六张面渲染完之后，第一部分的公式要在 GPU 上对每一个输出像素跑一遍——这是一个典型的 embarrassingly-parallel 问题（每个像素独立计算，互不依赖），UE5 里最适合的工具是 **Compute Shader + RDG（Render Dependency Graph）**。

`CameraModelUtil.cpp` 用宏 `DECLARE_WIDE_ANGLE_LENS_SHADER` 为六种相机模型各自声明了一个 shader **permutation**（`CameraModelUtil.cpp:125-191`），本质是同一份 `WideAngleLens.usf`，用不同的 `CAMERA_TYPE` 预处理宏编译出 6 份独立的 GPU 程序：

```cpp
#define DECLARE_WIDE_ANGLE_LENS_SHADER(NAME, CAMERA_MODEL) \
    class NAME : public FWideAngleLensShaderBase<CAMERA_MODEL>, public FGlobalShader { \
      ... \
      static void ModifyCompilationEnvironment(...) { \
          OutEnvironment.SetDefine(TEXT("CAMERA_TYPE"), static_cast<uint32>(CAMERA_MODEL)); \
      } \
    }
DECLARE_WIDE_ANGLE_LENS_SHADER(FWideAngleLensShader_Equidistance, ECameraModel::Equidistant);
```

这是 shader 工程里常见的性能取舍：与其在一份 shader 里用 `switch(CameraModel)` 做运行时分支（GPU 上分支发散代价高，且编译器无法针对具体模型常量折叠），不如为每种模型各编译一份专用程序，运行时按 `Options.CameraModel` 选择调用哪一个（`CameraModelUtil.cpp:569-676` 的大 `switch`）。

实际调度（`CameraModelUtil.cpp:268-284`）：

```cpp
GraphBuilder.AddPass(RDG_EVENT_NAME("WideAngleLens-Dispatch"), Parameters, ERDGPassFlags::Compute,
    [Parameters, Size](FRHICommandListImmediate& RHICmdList) {
        TShaderMapRef<FShaderType> ComputeShader(GetGlobalShaderMap(GMaxRHIFeatureLevel));
        FComputeShaderUtils::Dispatch(RHICmdList, ComputeShader, *Parameters,
            FComputeShaderUtils::GetGroupCount(FIntVector(Size.X, Size.Y, 1), FIntVector(SubgroupSize, 1, 1)));
    });
```

`SubgroupSize=32`（`CameraModelUtil.cpp:32`，对应主流 GPU 的 warp/wavefront 宽度），线程网格按输出图像尺寸 `ceil(Width/32) × Height × 1` 划分，**一个 GPU 线程负责一个输出像素**：读取 6 张 cubemap 面纹理作为 SRV、写最终失真图像作为 UAV，中间不经过 CPU，全程留在显存里。相对于场景本身的光栅化/光线追踪渲染成本，这一步重投影几乎是"免费"的。

## 2.4 从算法到机制的真实案例：窄 FOV 清晰度修复

这是一个把第一部分的公式和第二部分的引擎机制直接接起来的真实工程案例（完整验证过程见 `CARLA_BUILD_NOTES.md` §4.15）。

**症状**：8 相机环视里 `front_main`（37.5° 垂直 FOV）、`front_narrow`（26.25°）两台长焦相机，画面明显比广角相机模糊，即便所有相机都用同样的 1280×960 输出分辨率。

**根因**：构造函数把每一张 cubemap 面的捕获角度写死成 90°（`SceneCaptureSensor_WideAngleLens.cpp:130`：`constexpr auto FOV = 90.0F * Deg2Rad;`），跟相机实际配置的 FOV 无关。一台 26.25° FOV 的相机，第一部分推导出的射线方向 `Direction` 采样范围其实只覆盖这张 90° 面纹理中间一小块——固定分辨率的纹理，实际"有效"覆盖的立体角越窄，等效于把一小块像素数字裁剪+放大，纹素密度不够，必然模糊。这跟渲染质量、纹理 mip 精度都无关，是纯粹的几何采样密度问题。

**修复**：既然 2.2 节的 `ComputeCubemapRenderMask()` 已经能判断出"这台相机只需要前面这一张面"，那么对这类相机，前面这张面完全没必要固定捕获 90°——可以直接按相机自己的真实 FOV 捕获，把同样数量的像素铺满这个更窄的立体角，纹素密度随之提高。新增的 `UpdateFrontFaceProjection()`（`SceneCaptureSensor_WideAngleLens.cpp:569-618`）复用**已有**的 `CustomProjectionMatrix` 机制，只是把半角从写死的 45° 换成动态计算：

```cpp
void ASceneCaptureSensor_WideAngleLens::UpdateFrontFaceProjection()
{
    const bool bSingleFace = CubemapRenderMask == (1U << CubeFace_PosX);
    if (bSingleFace)
    {
        // 与 ComputeCubemapRenderMask() 用的是同一个安全余量，保证不会比该函数
        // 判定"只需要一张面"时假设的覆盖范围更窄。
        const float Margin = GetFOVMaskEnable() ? 1.0F : Sqrt2;
        const float CandidateHalfFOV = FMath::Max(XFOVAngle, YFOVAngle) * 0.5F * Margin;
        if (CandidateHalfFOV < DefaultHalfFOV * 0.95F)
            NewHalfFOV = FMath::Max(CandidateHalfFOV, MinHalfFOV);
    }
    FrontFaceHalfFOV = NewHalfFOV;
    FaceCaptures[CubeFace_PosX]->CustomProjectionMatrix =
        ProjectionMatrixType(FrontFaceHalfFOV, 1.0F, 1.0F, GNearClippingPlane);
}
```

这一步只改了"捕获这张面时用多大视场角"，**完全没有碰纹理分配尺寸**——这个区分很重要：早期几次尝试改"放大纹理分辨率"来解决同一个问题，连续 3 次卡死引擎（详见 `CARLA_BUILD_NOTES.md` §4.12），根因是 RDG/描述符管线对"6 张面尺寸不一致"这件事有某种未知的隐藏假设。而"捕获角度"和"纹理分配尺寸"是两个完全独立的变量——本次修复只动前者，`InitCustomFormat`/`Side` 等纹理分配代码一行没改，因此不会触发同一类问题。

对应地，`WideAngleLens.usf::SampleCubemap()` 的 UV 换算公式假设每张面固定覆盖 ±45°（`tan(45°)=1`），面 0 被动态收窄之后，需要用新增的 `FrontFaceTanHalfFOV` 参数把切平面坐标重新缩放到实际覆盖范围，且**只对面 0 生效**：

```hlsl
// WideAngleLens.usf:73-81
const float2 Tangent = FaceIndex == 0
    ? Key.xy / Key.z / FrontFaceTanHalfFOV
    : Key.xy / Key.z;
```

这一个改动完整体现了本文标题里"算法驱动引擎"这句话的意思：**第一部分的 r(θ)/FOV 数学告诉我们一台窄角相机真正需要多大立体角，第二部分的引擎机制（`CustomProjectionMatrix` + shader UV 缩放）把这个数字变成实际的像素密度分配**——两边改动量都很小，但缺一不可，改一处必须同步改另一处。

修复效果通过日志验证了实际生效数值：`front_narrow → 24.75°`半角，`front_main → 35.36°`半角（均从固定 45° 收窄而来），并通过同 tick、同外参的"等距鱼眼 vs 官方针孔"A/B 清晰度对照、以及完整 8 相机 10 帧生产管线跑通两种方式做了独立验证。

## 2.5 从 HDR 渲染到 Bayer RAW 比特流：GPU→CPU 最后一跳

前面几节讲的都是"如何算出正确的颜色"，最后还要走完"颜色 → 传给 Python 客户端的字节流"这一段。

**为什么需要 16-bit HDR 渲染目标**：普通相机的最终画面是 8-bit BGRA（`PF_B8G8R8A8`），已经过 tone mapping、gamma 校正，压缩进 [0,255]。要模拟真实图像传感器的原始输出（RAW），必须在 tone mapping 之前拿到线性、未截断的辐照度值——这就是为什么 `raw_type != uint8` 时会强制打开 16-bit 浮点渲染目标（`SceneCaptureSensor_WideAngleLens.cpp:769`：`Format = bEnable16BitFormat ? PF_FloatRGBA : PF_B8G8R8A8`）。

**GPU→CPU 异步回读**：`ImageUtil::ReadImageDataAsync`（`ImageUtil.cpp:341-372`）用 `FRHIGPUTextureReadback` 发起异步拷贝，用后台线程（而非渲染线程）忙等 `IsReady()`——注释里特别强调了这一点：如果把等待放在渲染管线自己的线程上会造成死锁，因为 readback 是否 ready 本身要靠 RHI 线程推进。

**Bayer 马赛克采样（CPU 侧）**：读回的仍然是完整 RGB(A) HDR 数据，真实 Bayer 传感器每个物理像素只有一个颜色通道（被一层 RGGB 滤光阵列覆盖）。`ImageUtil::ConvertRGBToBayerRGGB`（`ImageUtil.cpp:36-72`）按标准 RGGB 排列，每个像素只保留一个通道：

```cpp
if (y % 2 == 0)  BayerValue = (x % 2 == 0) ? Pixel.R : Pixel.G;   // 偶数行: R G R G ...
else             BayerValue = (x % 2 == 0) ? Pixel.G : Pixel.B;   // 奇数行: G B G B ...
```

再线性映射到 uint16 全量程 [0,65535]。这一步目前是在 CPU 上做的（GPU 回读完成之后），是一个明确的架构取舍：8 相机 × 10fps 的规模下 CPU 单通道采样完全跟得上，没有必要再加一个 GPU pass；如果未来相机数/帧率大幅提高，这里是一个可以搬回 GPU 的优化点。

最终数据打上 `EPixelFormat::BAYER_RGGB_U16` 格式标签（`LibCarla/source/carla/sensor/data/Image.h:23`），走 `SerializeAndSend` 发送到 TCP 流（`SceneCaptureCamera_WideAngleLens.cpp:121-137`），Python 客户端通过 `carla.Image.raw_data` 零拷贝拿到原始字节。完整链路：

```
6× SceneCaptureComponent2D (矩形透视捕获，可选光追)
        │  CaptureScene()（仅渲染 ComputeCubemapRenderMask 选中的面）
        ▼
6× FaceRenderTargets (PF_FloatRGBA，HDR 线性)
        │  WideAngleLens.usf::MainCS（第一部分公式，GPU compute shader）
        ▼
CaptureRenderTarget (单张 HDR 输出，仍是完整 RGB)
        │  FRHIGPUTextureReadback（异步 GPU→CPU）
        ▼
ImageUtil::ConvertRGBToBayerRGGB（CPU，RGGB 马赛克 + 量化到 uint16）
        │
        ▼
EPixelFormat::BAYER_RGGB_U16 → TCP → Python raw_data（零拷贝）
```

---

# 第三部分：工程应用——参数配置、数据采集与产物格式

## 3.1 蓝图属性一览

相机通过 `sensor.camera.rgb_fisheye` 蓝图生成（`ActorBlueprintFunctionLibrary.cpp:461-656` 注册全部属性），核心属性与代码位置对照：

| 属性 id | 类型 | 默认值 | 含义与代码位置 |
|---|---|---|---|
| `camera_model` | string | `perspective` | 六选一模型名（`equidistant`/`kannala-brandt`/…），解析见 `ActorBlueprintFunctionLibrary.cpp:2012-2039` |
| `fov` | float | 90.0 | **注意：这是垂直 FOV**，映射到 `SetFOVAngle`→`YFOVAngle`（`ActorBlueprintFunctionLibrary.cpp:2061-2065`），不是字面意义上笼统的"FOV" |
| `fov_horizontal` | float | 0.0(哨兵值="未设置") | 独立于纵向 FOV 的水平 FOV，默认按宽高比从 `fov` 派生（各向同性），非 0 才生效（`:2079-2081`） |
| `k0`,`k1`,`k2`,`k3` | float | 参考镜头标定值 | 仅 `camera_model=kannala-brandt` 时生效，见 1.2 节 |
| `cx`,`cy` | float | 0.0(哨兵值) | 主点偏移，非 0 才覆盖几何中心（`:2089-2094`） |
| `focal_length` | float | 0.0(哨兵值) | 显式指定焦距，覆盖由 FOV 派生的默认值 |
| `perspective`/`equirectangular`/`fov_mask`/`fov_fade_size`/`longitude_offset` | bool/float | false/0 | 渲染模式开关：转回透视输出 / 输出等距经纬图 / 是否收紧安全余量 / FOV 边缘渐隐宽度 / 经纬图水平偏移 |
| `raw_type` | string | `uint8` | `uint8`/`uint16`/`float32`/`bayer_rggb`，非 `uint8` 会自动联动打开 16-bit 渲染目标（`ActorBlueprintFunctionLibrary.cpp:2159-2172`） |
| `image_size_x`/`image_size_y` | int | 800/600 | 输出分辨率 |

## 3.2 真实案例：8 相机 Tesla 式环视配置

`occnetv3_data_generator/config/camera_config.py` 里的 `TESLA_CAMERAS` 是这套相机能力在实际项目里的应用现场。有一个历史遗留但容易踩坑的命名问题需要特别说明：

```python
# camera_config.py:5-9
# 等距投影的 fov 属性是垂直 FOV（YFOVAngle），而这里每个相机的 'fov' 字段沿用的是
# 历史上的水平 FOV 语义，'fov_vertical' 才是实际传给传感器的值。
# fov_vertical = fov * (image_size_y / image_size_x) = fov * 0.75  (960/1280)
```

也就是说 `camera_config.py` 字典里的 `fov` 字段只是给人看的历史参考值，`camera_manager.py::_setup_cameras()` 实际调用的是：

```python
# camera_manager.py:105
camera_bp.set_attribute('fov', str(cam_config['fov_vertical']))
```

8 台相机的实际配置（节选自 `camera_config.py:28-132`）：

| id | fov(水平参考) | fov_vertical(实际生效) | 安装位置 (x,y,z) | 朝向 (pitch,yaw,roll) |
|---|---|---|---|---|
| front_main | 50° | 37.5° | (1.0, 0.0, 1.6) | (0,0,0) |
| front_wide | 120° | 90° | (1.0, 0.0, 1.6) | (0,0,0) |
| front_narrow | 35° | 26.25° | (1.0, 0.0, 1.6) | (0,0,0) |
| left_pillar | 90° | 67.5° | (0.0,-1.1,1.7) | (0,-45,0) |
| right_pillar | 90° | 67.5° | (0.0,1.1,1.7) | (0,45,0) |
| left_repeater | 90° | 67.5° | (1.0,-1.0,1.0) | (0,-135,0) |
| right_repeater | 90° | 67.5° | (1.0,1.0,1.0) | (0,135,0) |
| rear | 120° | 90° | (-2.7,0.0,1.2) | (8,180,0) |

按 2.2 节的判定规则，`front_main`（37.5°）和 `front_narrow`（26.25°）两台的 FOV×√2 仍小于 90°，是 `CubemapRenderMask` 只含前面这一张的相机——正是 2.4 节修复直接受益、也是修复前肉眼可见最模糊的两台相机；其余 6 台 FOV 都在 90° 以上，天然需要渲染多张面，未受该修复影响也不需要收窄。

`camera_config.py` 顶部还预留了一组"物理镜头仿真"字段（`lens_model`/`distortion_coeffs`/`principal_point`/`fov_horizontal`），对应第一部分 1.2/1.3 节的 Kannala-Brandt 系数与主点偏移——**目前 8 台相机都没有配置这些键**，即当前数据集用的仍是"完美"等距模型。这是留好的基础设施：一旦有真实标定的车载镜头参数，只需要在这个字典里加上对应键，不需要改任何引擎代码。（`e2e_occ` 网络当前的射线编码假设 fx=fy 各向同性，真正启用 `fov_horizontal` 之前需要先确认/升级那部分算法。）

## 3.3 从像素到 DNG：数据落盘全链路

`raw_type='bayer_rggb'` 时，`camera.listen()` 回调收到的 `image.raw_data` 直接是引擎侧算好的 (H,W) uint16 Bayer 马赛克（不再是 BGRA）：

```python
# camera_manager.py:316-334
@staticmethod
def convert_to_bayer(image: carla.Image) -> np.ndarray:
    array = np.frombuffer(image.raw_data, dtype=np.uint16)
    expected = image.height * image.width
    bayer = array[-expected:].reshape((image.height, image.width))
    return bayer.copy()
```

落盘时（`data_utils/data_saver.py:302-346`），会把引擎侧始终以 16-bit 全量程（0~65535）传输的数据**右移**到配置的目标位深（默认 12-bit），模拟真实传感器 ADC 的实际精度，再写成带 CFA（Color Filter Array）EXIF 标签的 16-bit 单通道 TIFF（本质上就是 DNG）：

```python
shift = 16 - self.raw_bit_depth   # 默认 raw_bit_depth=12 → shift=4
bayer_quantized = (bayer_u16 >> shift).astype(np.uint16) if shift > 0 else bayer_u16
...
exif_dict = {"0th": {
    piexif.ImageIFD.PhotometricInterpretation: 32803,       # CFA 标签
    piexif.ImageIFD.BitsPerSample: (self.raw_bit_depth,),
}}
img_pil.save(str(output_path), format='TIFF', compression='none', exif=exif_bytes)  # 另存为 .dng
```

**读取侧对应要求**：OpenCV 不支持 `PhotometricInterpretation=32803`（CFA）格式，必须用 `rawpy` 或 `Pillow+piexif` 读取（`CLAUDE.md` 已记录的已知限制）。位深必须以 `calibration/intrinsics.json` 顶层的 `raw_bit_depth` 字段为准（采集时写入，`data_saver.py:174`），不能在加载侧硬编码某个位深的归一化除数——这类硬编码曾经导致过采集侧改了位深、加载侧没跟着改的静默数值错误。

## 3.4 内参重建：K 矩阵与渲染公式的一致性

标定文件 `calibration/intrinsics.json` 里保存的 K 矩阵，是 `camera_manager.py::get_intrinsics()`（`:446-489`）在 Python 侧**独立重新实现**了一遍 1.3 节的公式：

```python
fy = (height / 2.0) / (np.radians(fov_vertical) / 2.0)      # 对应 CameraModelUtil::ComputeDistance
fx = (width / 2.0) / (np.radians(fov_horizontal) / 2.0)     # 等距分支: F = R / (FOV/2)
cx, cy = cam_config.get('principal_point', (width/2.0, height/2.0))
K = [[fx,0,cx],[0,fy,cy],[0,0,1]]
```

这一行 `fy = (H/2)/(FOV_v/2)` 与 `CameraModelUtil.cpp:530-531` 的 `F = R / Angle`（等距分支）**必须永远保持数学等价**——这是整条数据管线里最容易被忽视、也最重要的一致性约束：如果 Python 侧计算 K 矩阵的公式和 C++ 侧引擎实际渲染用的公式有一天出现分歧（比如谁改了模型/公式却没同步改另一边），训练出来的网络会拿着一个和输入图像**不匹配**的 K 矩阵去做反投影/正投影，且不会有任何报错——是一类典型的静默数值错误。这也是为什么第一部分把公式讲得如此具体：任何改动这套相机的人，都需要同时知道公式在 shader/C++/Python 三处分别长什么样。

## 3.5 常见坑一览

| 坑 | 说明 |
|---|---|
| `fov` 属性实际是垂直 FOV | 蓝图属性名叫 `fov` 但对应 `YFOVAngle`，`camera_config.py` 里字面的 `fov` 字段只是历史参考值，真正生效的是 `fov_vertical`，见 3.2 |
| k0~k3 只在 `camera_model=kannala-brandt` 下生效 | `equidistant`（默认）模式完全不读这几个系数，是常见误解 |
| `raw_type != uint8` 必须联动 16-bit 渲染目标 | C++ 侧已自动处理（`ActorBlueprintFunctionLibrary.cpp:2168-2171`），但了解原因（1.3/2.5 节）有助于排查"RAW 数据看起来被截断成 8-bit 精度"这类问题 |
| DNG 位深必须读 `calibration/intrinsics.json::raw_bit_depth` | 不要在加载侧硬编码某个位深的归一化除数，见 3.3 |
| `post_process_profile` 曾经在鱼眼相机上完全不生效 | 历史 bug（`sensor.camera.rgb_fisheye` 蓝图从未注册该属性），已在 `ActorBlueprintFunctionLibrary.cpp` 修复，若用的是旧编译产物需要重新全量编译才能生效 |
| 窄 FOV 相机变模糊≠渲染质量问题 | 2.4 节的清晰度修复之前，`front_main`/`front_narrow` 模糊的根因是采样密度几何问题，与 `-quality-level` 画质档位、光追开关等无关；修复后二者也不再互相干扰，仍应分别验证 |

---

## 结语

这套相机能力最有意思的地方，其实是把"相机标定"这个通常单向使用的工具**反过来用**：标定公式本来是从"真实照片"求"理想世界"的映射，这里则是拿同一套公式，从"理想世界"（UE5 完美射线追踪）反向合成出"如果这台完美相机装了一支有着某种真实光学特性的镜头，它会拍出什么样"。第一部分的每一条公式，都同时活在三个地方——HLSL shader（运行时逐像素采样）、C++（焦距/内参预计算）、Python（标定文件导出）——第二部分讲的是这些公式如何变成 UE5 里可运行、可加速的渲染管线，第三部分则是这套机制在真实 8 相机数据采集项目里如何被参数化配置、落盘、消费。

延伸阅读：`CARLA_BUILD_NOTES.md` §4.8（Bayer RAW HDR 采集的 5 个独立坑）、§4.15（窄 FOV 清晰度修复完整验证记录）；`occnetv3_data_generator/README.md`（数据采集侧的完整架构与性能数据）。
