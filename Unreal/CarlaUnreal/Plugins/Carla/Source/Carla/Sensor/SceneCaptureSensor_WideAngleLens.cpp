// Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include "SceneCaptureSensor_WideAngleLens.h"
#include "Carla/Game/CarlaStatics.h"
#include "Carla/Actor/ActorBlueprintFunctionLibrary.h"
#include "Carla/Sensor/PostProcessConfig.h"
#include "Carla/Sensor/SceneCaptureSensor.h"

#include <util/ue-header-guard-begin.h>
#include "Engine/Engine.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphUtils.h"
#include "HAL/IConsoleManager.h"
#include "Kismet/KismetSystemLibrary.h"
#include "SceneCaptureExposureSharing.h"  // CARLA engine patch: shared eye adaptation between scene captures
#include <util/ue-header-guard-end.h>

#include <cmath>

// extern bool CARLA_API GCARLASelectiveRendering;
// extern bool CARLA_API GCARLALightingOnly;

static TAutoConsoleVariable<int32> CVarWideAngleSensorDumpAllFrames(
    TEXT("Carla.WideAngleLens.DumpAllFrames"),
    0,
    TEXT("If enabled, saves all frames of all *_WideAngleLens sensors to disk.\n")
    TEXT("0: Disabled\n")
    TEXT("1: Enabled\n"));

static TAutoConsoleVariable<FString> CVarWideAngleSensorDumpAllFramesPath(
    TEXT("Carla.WideAngleLens.DumpAllFramesPath"),
    TEXT(""),
    TEXT("Sets the destination path when saving frames via \"Carla.WideAngleLens.DumpAllFrames\".\n"));

static TAutoConsoleVariable<int32> CVarWideAngleSensorDumpAllFramesCubemap(
    TEXT("Carla.WideAngleLens.DumpAllFrames.Cubemap"),
    0,
    TEXT("If enabled, saves each individual face of each *_WideAngleLens sensor.\n")
    TEXT("0: Disabled\n")
    TEXT("1: Enabled\n"));

static TAutoConsoleVariable<float> CVarWideAngleFaceOversample(
    TEXT("Carla.WideAngleLens.FaceOversample"),
    2.0F,
    TEXT("RGB wide-angle cameras: cube-face texel density relative to the output image's angular pixel density,\n")
    TEXT("used when face_size is 0 (label/depth cameras always use 1.0). 2.0 + the 4-sample pixel-area\n")
    TEXT("integration in WideAngleLens.usf replaces TSR on the faces. Clamped by Carla.WideAngleLens.MaxFaceSide.\n"),
    ECVF_ReadOnly);

static TAutoConsoleVariable<int32> CVarWideAngleMaxFaceSide(
    TEXT("Carla.WideAngleLens.MaxFaceSide"),
    2048,
    TEXT("Upper bound for the lens-derived cube-face side (pixels).\n"),
    ECVF_ReadOnly);

static TAutoConsoleVariable<int32> CVarWideAngleSensorSkipVFTR(
    TEXT("Carla.WideAngleLens.SkipVFTR"),
    0,
    TEXT("If enabled, *_WideAngleLens sensors do not toggle r.VolumetricFog.TemporalReprojection when rendering.\n")
    TEXT("0: Disabled\n")
    TEXT("1: Enabled\n"));

static auto WIDE_ANGLE_LENS_SENSOR_COUNTER = 0u;

// =============================================================================
// -- Local static methods -----------------------------------------------------
// =============================================================================

// Local namespace to avoid name collisions on unit builds.
namespace SceneCaptureSensorWideAngleLens_local_ns {

    static void SetCameraDefaultOverrides(USceneCaptureComponent2D_CARLA& CaptureComponent);

} // namespace SceneCaptureSensorWideAngleLens_local_ns

// =============================================================================
// -- ASceneCaptureSensor_WideAngleLens ----------------------------------------
// =============================================================================

ASceneCaptureSensor_WideAngleLens::ASceneCaptureSensor_WideAngleLens(const FObjectInitializer& ObjectInitializer) :
    Super(ObjectInitializer),
    FaceCaptures(),
    FaceRenderTargets(),
    CaptureRenderTarget(),
    TargetGamma(0.0F),
    ImageWidth(1280U),
    ImageHeight(1280U),
    CameraModel(ECameraModel::Default),
    KannalaBrandtCameraCoefficients
    {
        0.08309221636708493F,
        0.01112126630599195F,
        0.008587261043925865F,
        0.0008542188930970716F
    },
    LensLUT(),
    LensThetaMax(PI * 0.5F),
    LensIntrinsics(0.0F, 0.0F, 0.5F, 0.5F),
    YFOVAngle(PI * 0.5F),
    XFOVAngle(VerticalToHorizontal(YFOVAngle)),
    YFocalLength(
        CameraModelUtil::ComputeDistance(
            CameraModel,
            YFOVAngle,
            ImageHeight,
            KannalaBrandtCameraCoefficients)),
    XFocalLength(
        CameraModelUtil::ComputeDistance(
            CameraModel,
            XFOVAngle,
            ImageWidth,
            KannalaBrandtCameraCoefficients)),
    PrincipalPointX(ImageWidth * 0.5F),
    PrincipalPointY(ImageHeight * 0.5F),
    LongitudeOffset(),
    FOVFadeSize(),
    CubemapRenderMask(0),
    FrontFaceHalfFOV(PI * 0.25F),
    CubemapSampler(CameraModelUtil::GetSampler(SF_AnisotropicLinear)),
    bUseRayTracing(true),
    bEnablePostProcessingEffects(true),
    bEnable16BitFormat(false),
    bRenderPerspective(false),
    bRenderEquirectangular(false),
    bFOVMaskEnable(false),
    bExplicitIntrinsics(false)
{
    FaceCaptures.SetNum(6);
    FaceRenderTargets.SetNum(6);

    // Computed in the constructor body, not the initializer list:
    // ComputeCubemapRenderMask() reads bFOVMaskEnable / bRenderEquirectangular,
    // which are declared after CubemapRenderMask and would still be
    // uninitialized during initializer-list evaluation.
    CubemapRenderMask = ComputeCubemapRenderMask();

    PrimaryActorTick.bCanEverTick = true;
    PrimaryActorTick.TickGroup = TG_PrePhysics;

    using ProjectionMatrixType = std::conditional_t<
        (bool)ERHIZBuffer::IsInverted,
        FReversedZPerspectiveMatrix,
        FPerspectiveMatrix>;

    constexpr auto Deg2Rad = (PI / 180.0F);
    constexpr auto FOV = 90.0F * Deg2Rad;
    constexpr auto HalfFOV = FOV * 0.5F;

    const auto SensorIndex = WIDE_ANGLE_LENS_SENSOR_COUNTER++;
    const auto FaceIndexBase = SensorIndex * 6;
    const auto ProjectionMatrix = ProjectionMatrixType(HalfFOV, 1.0F, 1.0F, GNearClippingPlane);

    const FVector Forward[] =
    {
        FVector::ForwardVector,
        -FVector::ForwardVector,
        FVector::RightVector,
        -FVector::RightVector,
        FVector::UpVector,
        -FVector::UpVector,
    };

    const FVector Right[] =
    {
        FVector::RightVector,
        -FVector::RightVector,
        -FVector::ForwardVector,
        FVector::ForwardVector,
        FVector::RightVector,
        -FVector::RightVector,
    };

    CaptureRenderTarget = CreateDefaultSubobject<UTextureRenderTarget2D>(
        FName(*FString::Printf(
            TEXT("CaptureRenderTarget2D-WideLens-Final-d%d"),
            SensorIndex)));

    CaptureRenderTarget->CompressionSettings = TextureCompressionSettings::TC_Default;
    CaptureRenderTarget->SRGB = false;
    CaptureRenderTarget->bAutoGenerateMips = false;
    CaptureRenderTarget->bGPUSharedFlag = true;
    CaptureRenderTarget->AddressX = TextureAddress::TA_Clamp;
    CaptureRenderTarget->AddressY = TextureAddress::TA_Clamp;

    for (uint8 i = 0; i != 6; ++i)
    {
        const auto AbsIndex = FaceIndexBase + i;

        auto& RenderTarget = FaceRenderTargets[i];
        auto& FaceCapture = FaceCaptures[i];

        RenderTarget = CreateDefaultSubobject<UTextureRenderTarget2D>(
            FName(*FString::Printf(TEXT("CaptureRenderTarget2D-WideLens-Face-d%d"), AbsIndex)));
        check(RenderTarget != nullptr);
        RenderTarget->CompressionSettings = TextureCompressionSettings::TC_Default;
        RenderTarget->SRGB = false;
        RenderTarget->bAutoGenerateMips = false;
        RenderTarget->bGPUSharedFlag = true;
        RenderTarget->AddressX = TextureAddress::TA_Clamp;
        RenderTarget->AddressY = TextureAddress::TA_Clamp;

        FaceCapture = CreateDefaultSubobject<USceneCaptureComponent2D_CARLA>(
            FName(*FString::Printf(TEXT("USceneCaptureComponent2D_CARLA-%d"), AbsIndex)));
        check(FaceCapture != nullptr);
        FaceCapture->SetupAttachment(RootComponent);
        FaceCapture->SetRelativeRotation(FRotationMatrix::MakeFromXY(Forward[i], Right[i]).ToQuat());
        FaceCapture->ViewActor = this;
        FaceCapture->ProjectionType = ECameraProjectionMode::Perspective;
        FaceCapture->PrimitiveRenderMode = ESceneCapturePrimitiveRenderMode::PRM_RenderScenePrimitives;
        FaceCapture->bCaptureOnMovement = false;
        FaceCapture->bCaptureEveryFrame = false;
        FaceCapture->bAlwaysPersistRenderingState = true;
        FaceCapture->bUseCustomProjectionMatrix = true;
        FaceCapture->CustomProjectionMatrix = ProjectionMatrix;
        // Propagate ray-tracing flag to this face capture.
        FaceCapture->bUseRayTracingIfEnabled = ASceneCaptureSensor::ResolveCameraRayTracing(bUseRayTracing, false);
        SceneCaptureSensorWideAngleLens_local_ns::SetCameraDefaultOverrides(*FaceCapture);
    }

    // FaceCaptures now exist (loop above); safe to narrow the front face's
    // capture angle if this camera's default FOV (90 deg) already qualifies
    // as single-face. Real narrowing normally happens later once SetFOVAngle
    // is called via Set()/UActorBlueprintFunctionLibrary::SetCamera, which
    // re-invokes this same update.
    UpdateFrontFaceProjection();
}

void ASceneCaptureSensor_WideAngleLens::Set(const FActorDescription& Description)
{
    Super::Set(Description);
    UActorBlueprintFunctionLibrary::SetCamera(Description, this);
}

void ASceneCaptureSensor_WideAngleLens::SetImageSize(uint32 InWidth, uint32 InHeight)
{
    bool UpdateRenderMask = InWidth != ImageWidth || InHeight != ImageHeight;

    ImageWidth = InWidth;
    ImageHeight = InHeight;

    // Reset the principal point to the new size's exact geometric center.
    // Mirrors how YFocalLength/XFocalLength only get refreshed for the new
    // size once SetFOVAngle/SetFOVAngleX runs again — callers that need a
    // real calibrated (non-centered) principal point must call
    // SetPrincipalPoint() after SetImageSize(), same ordering requirement.
    PrincipalPointX = ImageWidth * 0.5F;
    PrincipalPointY = ImageHeight * 0.5F;

    if (UpdateRenderMask)
    {
        CubemapRenderMask = ComputeCubemapRenderMask();
        UpdateFrontFaceProjection();
    }
}

void ASceneCaptureSensor_WideAngleLens::SetImageSize(int32 Width, int32 Height)
{
    // Clamp to a sane minimum: negative values would wrap to enormous
    // dimensions when cast to uint32.
    SetImageSize(
        static_cast<uint32>(FMath::Max(Width, 1)),
        static_cast<uint32>(FMath::Max(Height, 1)));
}

ECameraModel ASceneCaptureSensor_WideAngleLens::GetCameraModel() const
{
    return CameraModel;
}

void ASceneCaptureSensor_WideAngleLens::SetCameraModel(ECameraModel NewCameraModel)
{
    bool UpdateRenderMask = NewCameraModel != CameraModel;

    CameraModel = NewCameraModel;

    if (UpdateRenderMask)
    {
        CubemapRenderMask = ComputeCubemapRenderMask();
        UpdateFrontFaceProjection();
    }
}

float ASceneCaptureSensor_WideAngleLens::GetFOVAngle() const
{
    return YFOVAngle;
}

float ASceneCaptureSensor_WideAngleLens::GetFOVAngleY() const
{
    return YFOVAngle;
}

float ASceneCaptureSensor_WideAngleLens::GetFOVAngleX() const
{
    return XFOVAngle;
}

constexpr auto DegToRad = PI / 180.0F;
constexpr auto RadToDeg = 180.0F / PI;

void ASceneCaptureSensor_WideAngleLens::SetFOVAngle(float NewFOV)
{
    NewFOV *= DegToRad;

    // XFOVAngle is derived from the aspect ratio, which SetImageSize() may have
    // changed since the last call. Compare it too: otherwise a vertical FOV that
    // happens to equal the previous one (e.g. the 90 deg default used by
    // front_wide/rear) skips the mask / front-face refresh even though the
    // horizontal FOV just changed.
    const float NewXFOV = VerticalToHorizontal(NewFOV);
    bool UpdateRenderMask = NewFOV != YFOVAngle || NewXFOV != XFOVAngle;

    YFOVAngle = NewFOV;
    XFOVAngle = NewXFOV;

    // LUT1D and BrownConrady carry their own scale (the LUT samples / the
    // explicit intrinsics); ComputeDistance has no closed form for them.
    if (CameraModel != ECameraModel::LUT1D && CameraModel != ECameraModel::BrownConrady)
    {
        YFocalLength = CameraModelUtil::ComputeDistance(
            CameraModel,
            NewFOV,
            ImageHeight,
            KannalaBrandtCameraCoefficients);
    }

    // Keep XFocalLength tracking the aspect-ratio-derived XFOVAngle by
    // default (isotropic fx=fy, the pre-existing behavior). A later explicit
    // SetFOVAngleX() call overrides both again — see its own comment.
    // 与上面 YFocalLength 同样的防护：ComputeDistance 对 LUT1D/BrownConrady 会 check(false)。
    if (CameraModel != ECameraModel::LUT1D && CameraModel != ECameraModel::BrownConrady)
    {
        XFocalLength = CameraModelUtil::ComputeDistance(
            CameraModel,
            XFOVAngle,
            ImageWidth,
            KannalaBrandtCameraCoefficients);
    }

    if (UpdateRenderMask)
    {
        CubemapRenderMask = ComputeCubemapRenderMask();
        UpdateFrontFaceProjection();
    }
}

void ASceneCaptureSensor_WideAngleLens::SetFOVAngleX(float NewFOV)
{
    NewFOV *= DegToRad;

    bool UpdateRenderMask = NewFOV != XFOVAngle;

    XFOVAngle = NewFOV;

    if (CameraModel != ECameraModel::LUT1D && CameraModel != ECameraModel::BrownConrady)
    {
        XFocalLength = CameraModelUtil::ComputeDistance(
            CameraModel,
            NewFOV,
            ImageWidth,
            KannalaBrandtCameraCoefficients);
    }

    if (UpdateRenderMask)
    {
        CubemapRenderMask = ComputeCubemapRenderMask();
        UpdateFrontFaceProjection();
    }
}

void ASceneCaptureSensor_WideAngleLens::SetPrincipalPoint(float Cx, float Cy)
{
    PrincipalPointX = Cx;
    PrincipalPointY = Cy;
    // 主点偏移会改变射线覆盖 (选面 / 前向面收窄)
    CubemapRenderMask = ComputeCubemapRenderMask();
    UpdateFrontFaceProjection();
}

float ASceneCaptureSensor_WideAngleLens::GetPrincipalPointX() const
{
    return PrincipalPointX;
}

float ASceneCaptureSensor_WideAngleLens::GetPrincipalPointY() const
{
    return PrincipalPointY;
}

void ASceneCaptureSensor_WideAngleLens::SetLensLUT(const TArray<float>& Samples, float ThetaMaxRadians)
{
    LensLUT = Samples;
    LensThetaMax = ThetaMaxRadians > 0.0F ? ThetaMaxRadians : PI * 0.5F;
    CubemapRenderMask = ComputeCubemapRenderMask();
    UpdateFrontFaceProjection();
}

void ASceneCaptureSensor_WideAngleLens::SetLensIntrinsics(float Fx, float Fy, float Cx, float Cy)
{
    bExplicitIntrinsics = Fx > 0.0F;
    LensIntrinsics = FVector4f(Fx, Fy > 0.0F ? Fy : Fx, Cx, Cy);
    CubemapRenderMask = ComputeCubemapRenderMask();
    UpdateFrontFaceProjection();
}

void ASceneCaptureSensor_WideAngleLens::SetTargetGamma(float Gamma)
{
    TargetGamma = Gamma;
}

float ASceneCaptureSensor_WideAngleLens::GetFocalLength() const
{
    return YFocalLength;
}

void ASceneCaptureSensor_WideAngleLens::SetFocalLength(float NewFocalLength)
{
    bool UpdateRenderMask = NewFocalLength != YFocalLength;

    YFocalLength = NewFocalLength;

    if (UpdateRenderMask)
    {
        CubemapRenderMask = ComputeCubemapRenderMask();
        UpdateFrontFaceProjection();
    }
}

void ASceneCaptureSensor_WideAngleLens::SetCameraCoefficients(TArrayView<const float> Coefficients)
{
    bool UpdateRenderMask = KannalaBrandtCameraCoefficients.Num() != Coefficients.Num();

    for (uint32 i = 0; i != (uint32)Coefficients.Num() && !UpdateRenderMask; ++i)
        UpdateRenderMask = UpdateRenderMask || KannalaBrandtCameraCoefficients[i] != Coefficients[i];

    KannalaBrandtCameraCoefficients = TArray<float>(Coefficients);

    if (UpdateRenderMask)
    {
        CubemapRenderMask = ComputeCubemapRenderMask();
        UpdateFrontFaceProjection();
    }
}

void ASceneCaptureSensor_WideAngleLens::SetCameraCoefficients(const TArray<float>& Coefficients)
{
    SetCameraCoefficients(TArrayView<const float>(Coefficients));
}

const TArray<float>& ASceneCaptureSensor_WideAngleLens::GetCameraCoefficients() const
{
    return KannalaBrandtCameraCoefficients;
}

UTextureRenderTarget2D* ASceneCaptureSensor_WideAngleLens::GetCaptureRenderTarget()
{
    return CaptureRenderTarget;
}

float ASceneCaptureSensor_WideAngleLens::GetTargetGamma() const
{
    return TargetGamma;
}

bool ASceneCaptureSensor_WideAngleLens::GetRenderPerspective() const
{
    return bRenderPerspective;
}

void ASceneCaptureSensor_WideAngleLens::SetRenderPerspective(bool bEnable)
{
    bool UpdateRenderMask = bRenderPerspective != bEnable;

    bRenderPerspective = bEnable;

    if (UpdateRenderMask)
    {
        CubemapRenderMask = ComputeCubemapRenderMask();
        UpdateFrontFaceProjection();
    }
}

bool ASceneCaptureSensor_WideAngleLens::GetRenderEquirectangular() const
{
    return bRenderEquirectangular;
}

void ASceneCaptureSensor_WideAngleLens::SetRenderEquirectangular(bool bEnable)
{
    bool UpdateRenderMask = bRenderEquirectangular != bEnable;

    bRenderEquirectangular = bEnable;

    if (UpdateRenderMask)
    {
        CubemapRenderMask = ComputeCubemapRenderMask();
        UpdateFrontFaceProjection();
    }
}

bool ASceneCaptureSensor_WideAngleLens::GetFOVMaskEnable() const
{
    return bFOVMaskEnable;
}

void ASceneCaptureSensor_WideAngleLens::SetFOVMaskEnable(bool bEnable)
{
    bool UpdateRenderMask = bFOVMaskEnable != bEnable;

    bFOVMaskEnable = bEnable;

    if (UpdateRenderMask)
    {
        CubemapRenderMask = ComputeCubemapRenderMask();
        UpdateFrontFaceProjection();
    }
}

float ASceneCaptureSensor_WideAngleLens::GetFOVFadeSize() const
{
    return FOVFadeSize;
}

void ASceneCaptureSensor_WideAngleLens::SetFOVFadeSize(float NewFOVFadeSize)
{
    FOVFadeSize = NewFOVFadeSize;
}

float ASceneCaptureSensor_WideAngleLens::GetRenderEquirectangularLongitudeOffset() const
{
    return LongitudeOffset * RadToDeg;
}

void ASceneCaptureSensor_WideAngleLens::SetRenderEquirectangularLongitudeOffset(
    float NewLatitudeOffset)
{
    LongitudeOffset = NewLatitudeOffset * DegToRad;
}

void ASceneCaptureSensor_WideAngleLens::SetUseRayTracing(bool Enable)
{
    bUseRayTracing = Enable;
    for (auto FaceCapture : FaceCaptures)
    {
        if (FaceCapture != nullptr)
            FaceCapture->bUseRayTracingIfEnabled = ASceneCaptureSensor::ResolveCameraRayTracing(Enable, false);
    }
}

void ASceneCaptureSensor_WideAngleLens::EnqueueRenderSceneImmediate()
{
    TRACE_CPUPROFILER_EVENT_SCOPE(ASceneCaptureSensor_WideAngleLens::EnqueueRenderSceneImmediate);
    CaptureSceneExtended();
}

uint8 ASceneCaptureSensor_WideAngleLens::FindFaceIndex(FVector2D UV) const
{
    const float R = hypotf(UV.X, UV.Y);
    float Theta;
    switch (CameraModel)
    {
    case ECameraModel::LUT1D:
        Theta = CameraModelUtil::LUT1D::SampleInverse(R, LensLUT, LensThetaMax);
        break;
    case ECameraModel::BrownConrady:
        Theta = CameraModelUtil::ComputeAngle(ECameraModel::Perspective, R, KannalaBrandtCameraCoefficients);
        break;
    default:
        Theta = CameraModelUtil::ComputeAngle(CameraModel, R, KannalaBrandtCameraCoefficients);
        break;
    }
    const float HalfPi = PI / 2.0f;
    const float Phi = HalfPi - Theta;
    const float Rho = atan2f(UV.Y, UV.X);

    float PhiSin = 0.0F;
    float PhiCos = 0.0F;
    float RhoSin = 0.0F;
    float RhoCos = 0.0F;

    FMath::SinCos(&PhiSin, &PhiCos, Phi);
    FMath::SinCos(&RhoSin, &RhoCos, Rho);

    auto Direction = FVector(PhiSin, PhiCos * RhoCos, -PhiCos * RhoSin);
    Direction.Normalize();

    auto DirectionAbs = Direction.GetAbs();

    if (DirectionAbs.X >= DirectionAbs.Y && DirectionAbs.X >= DirectionAbs.Z)
        return Direction.X < 0 ? 1U : 0U;
    else if (DirectionAbs.Y >= DirectionAbs.X && DirectionAbs.Y >= DirectionAbs.Z)
        return Direction.Y < 0 ? 3U : 2U;
    else
        return Direction.Z < 0 ? 5U : 4U;
}

uint8 ASceneCaptureSensor_WideAngleLens::ComputeCubemapRenderMask() const
{
    // Equirectangular projection samples the full sphere, so every cube face
    // must be rendered regardless of the configured FOV.
    if (bRenderEquirectangular)
        return (1U << CubeFace_PosX) | (1U << CubeFace_NegX) |
               (1U << CubeFace_PosY) | (1U << CubeFace_NegY) |
               (1U << CubeFace_PosZ) | (1U << CubeFace_NegZ);

    static const float Pi = PI;
    static const float HalfPi = Pi / 2.0f;
    static const float Sqrt2 = sqrtf(2.0f);

    // 2026-10-15：一般情况（非 LUT1D、不开 FOV 圆形掩码）按镜头实际射线覆盖选面，与 shader 同一套
    // 像素→射线公式，显式内参 / 偏心主点 / 独立水平焦距都自动生效。之前只看 FOV 字段 ×√2 估算：
    // 显式内参对应的视野比 FOV 字段宽时，shader 会去采样没渲染的面。
    if (CameraModel != ECameraModel::LUT1D && !GetFOVMaskEnable())
    {
        uint8 CoverageMask = 0;
        float MaxFrontTan = 0.0F;
        ComputeLensCoverage(CoverageMask, MaxFrontTan);
        return CoverageMask != 0 ? CoverageMask : (uint8)(1U << CubeFace_PosX);
    }

    auto Mask = 1U << CubeFace_PosX; // Render front face by default.

    // LUT1D: the table's acceptance angle bounds the field of view on both
    // axes (the principal point may sit anywhere, so assume the full circle).
    // Otherwise the fov attribute does, widened to the frame corners unless
    // the FOV mask crops them.
    const auto FOV = CameraModel == ECameraModel::LUT1D
        ? FVector2D(2.0F * LensThetaMax, 2.0F * LensThetaMax)
        : FVector2D(GetFOVAngleX(), GetFOVAngleY()) * (GetFOVMaskEnable() ? 1 : Sqrt2);

    if (FOV.Y > HalfPi)
    {
        Mask |= 1U << CubeFace_PosZ;
        Mask |= 1U << CubeFace_NegZ;
    }

    if (FOV.X > HalfPi)
    {
        Mask |= 1U << CubeFace_PosY;
        Mask |= 1U << CubeFace_NegY;
    }

    if (FOV.X > Pi || FOV.Y > Pi)
    {
        Mask |= 1U << CubeFace_NegX;
    }

    return (uint8)Mask;
}

void ASceneCaptureSensor_WideAngleLens::UpdateFrontFaceProjection()
{
    using ProjectionMatrixType = std::conditional_t<
        (bool)ERHIZBuffer::IsInverted,
        FReversedZPerspectiveMatrix,
        FPerspectiveMatrix>;

    constexpr float DefaultHalfFOV = PI * 0.25F; // 45 deg: the original fixed 90 deg face.
    constexpr float MinHalfFOV = 1.0F * (PI / 180.0F); // 1 deg floor, avoid a degenerate projection.

    float NewHalfFOV = DefaultHalfFOV;

    // Narrowing the face is only valid when SampleCubemap() (WideAngleLens.usf)
    // is guaranteed to only ever look up FaceIndex 0 for this camera — every
    // other face's UV math still assumes the untouched +/-45 deg convention.
    // LUT1D 镜头保持 90 度原行为（射线范围由 LUT 决定，这里不做覆盖分析）。
    // 2026-10-15 起显式内参（fx/fy/cx/cy）也允许收窄：收窄范围由实际射线覆盖推导，不再依赖 FOV 字段，
    // 之前显式内参一律退回 90 度，长焦的角分辨率增益静默消失。
    const bool bSingleFace = CubemapRenderMask == (1U << CubeFace_PosX) &&
        CameraModel != ECameraModel::LUT1D && !bRenderEquirectangular;

    if (bSingleFace)
    {
        float CandidateHalfFOV;
        if (GetFOVMaskEnable())
        {
            // 圆形 FOV 掩码：画面只保留 YFOV 圆内，按圆半径收窄（与选面逻辑一致）
            CandidateHalfFOV = FMath::Max(XFOVAngle, YFOVAngle) * 0.5F;
        }
        else
        {
            // 按输出图像全部边界/内部射线在前向面上的最大切线收窄，外加 1.5 个源纹素的双线性采样余量。
            // 之前用 max(FOV)·0.5·√2 的经验余量：35° 长焦的源捕获面 49.5°，最终画面中间一行只用到
            // 1280 宽源纹理里约 875 个像素；按射线边界收紧后约 36°，同样 1280 的纹理覆盖更窄的角度。
            uint8 CoverageMask = 0;
            float MaxFrontTan = 0.0F;
            ComputeLensCoverage(CoverageMask, MaxFrontTan);
            const float Side = static_cast<float>(FMath::Max<uint32>(GetFaceSide(), 1U));
            CandidateHalfFOV = FMath::Atan(MaxFrontTan * (1.0F + 3.0F / Side));
        }

        // Only narrow when it buys a meaningful density gain; otherwise keep
        // the original behavior untouched rather than churn the projection
        // matrix for a negligible difference.
        if (CandidateHalfFOV < DefaultHalfFOV * 0.95F)
            NewHalfFOV = FMath::Max(CandidateHalfFOV, MinHalfFOV);
    }

    if (FMath::IsNearlyEqual(NewHalfFOV, FrontFaceHalfFOV, 1e-5F))
        return;

    FrontFaceHalfFOV = NewHalfFOV;

    auto* FrontFaceCapture = FaceCaptures[CubeFace_PosX];
    if (FrontFaceCapture != nullptr)
    {
        FrontFaceCapture->CustomProjectionMatrix =
            ProjectionMatrixType(FrontFaceHalfFOV, 1.0F, 1.0F, GNearClippingPlane);
        // 组件 FOVAngle 与实际投影保持一致（度）：引擎里按 FOVAngle 估算视图需求的地方（纹理流送等）
        // 不会再按 90 度理解一个其实只有几十度的长焦捕获
        FrontFaceCapture->FOVAngle = FMath::RadiansToDegrees(2.0F * FrontFaceHalfFOV);
    }

    UE_LOG(LogCarla, Log,
        TEXT("[WideAngleLens] %s: front face capture half-FOV -> %.2f deg (bSingleFace=%d, XFOV=%.2f, YFOV=%.2f)"),
        *GetName(), FMath::RadiansToDegrees(FrontFaceHalfFOV), bSingleFace,
        FMath::RadiansToDegrees(XFOVAngle), FMath::RadiansToDegrees(YFOVAngle));
}

uint32 ASceneCaptureSensor_WideAngleLens::GetFaceSide() const
{
    if (FaceSize > 0)
        return static_cast<uint32>(FaceSize);
    return AutoFaceSide > 0 ? AutoFaceSide : std::max(GetImageWidth(), GetImageHeight());
}

uint32 ASceneCaptureSensor_WideAngleLens::ComputeRequiredFaceSide() const
{
    const int32 W = static_cast<int32>(ImageWidth);
    const int32 H = static_cast<int32>(ImageHeight);
    const uint32 Fallback = std::max(GetImageWidth(), GetImageHeight());
    // LUT1D / 经纬图输出不走 PixelToRay，保持原行为
    if (W < 2 || H < 2 || CameraModel == ECameraModel::LUT1D || bRenderEquirectangular)
        return Fallback;

    // 超采样 (RGB) 相机按 CVar 倍数加密；标签/深度相机只要求不放大 (1 倍)
    const float Oversample = bSupersampleDistortion ?
        FMath::Max(CVarWideAngleFaceOversample.GetValueOnGameThread(), 0.25F) : 1.0F;
    const float FrontTan = FMath::Tan(FrontFaceHalfFOV);
    // 射线投到主轴对应 cube 面的平面上 (切线坐标)；面边长 Side 对应切线跨度 2*HalfTan
    auto FaceUV = [](const FVector& D, int32 Axis)
    {
        const FVector A = D.GetAbs();
        switch (Axis)
        {
        case 0: return FVector2D(D.Y / A.X, D.Z / A.X);
        case 1: return FVector2D(D.X / A.Y, D.Z / A.Y);
        default: return FVector2D(D.X / A.Z, D.Y / A.Z);
        }
    };
    float Need = 0.0F;
    for (int32 Y = 0; Y + 1 < H; Y += 8)
    {
        for (int32 X = 0; X + 1 < W; X += 8)
        {
            const FVector D = PixelToRay(static_cast<float>(X), static_cast<float>(Y));
            const FVector A = D.GetAbs();
            const int32 Axis = (A.X >= A.Y && A.X >= A.Z) ? 0 : (A.Y >= A.Z ? 1 : 2);
            const float HalfTan = (Axis == 0 && D.X > 0.0) ? FrontTan : 1.0F;
            const FVector2D P = FaceUV(D, Axis);
            const float StepX = (FaceUV(PixelToRay(X + 1.0F, static_cast<float>(Y)), Axis) - P).Size();
            const float StepY = (FaceUV(PixelToRay(static_cast<float>(X), Y + 1.0F), Axis) - P).Size();
            const float Step = FMath::Min(StepX, StepY);
            if (Step > 1e-9F)
                Need = FMath::Max(Need, 2.0F * HalfTan * Oversample / Step);
        }
    }
    if (Need <= 0.0F)
        return Fallback;
    const int32 MaxSide = FMath::Max(CVarWideAngleMaxFaceSide.GetValueOnGameThread(), 256);
    const int32 Side = FMath::Clamp(FMath::CeilToInt(Need), 256, MaxSide);
    return static_cast<uint32>(Align(Side, 16));
}

FVector ASceneCaptureSensor_WideAngleLens::PixelToRay(float Px, float Py) const
{
    // 与 CameraModelUtil::DistortCubemapToImage 构造 CameraParams、WideAngleLens.usf MainCS 构造射线完全一致
    const float W = static_cast<float>(ImageWidth);
    const float H = static_cast<float>(ImageHeight);
    float Fx, Fy, Cx, Cy;
    if (bExplicitIntrinsics)
    {
        Fx = LensIntrinsics.X * W;
        Fy = LensIntrinsics.Y * H;
        Cx = LensIntrinsics.Z * W;
        Cy = LensIntrinsics.W * H;
    }
    else
    {
        Fx = XFocalLength > 0.0F ? XFocalLength : YFocalLength;
        Fy = YFocalLength;
        Cx = PrincipalPointX;
        Cy = PrincipalPointY;
    }
    const float Ux = (Px - Cx) / FMath::Max(Fx, 1e-6F);
    const float Uy = (Py - Cy) / FMath::Max(Fy, 1e-6F);
    const float R = FMath::Sqrt(Ux * Ux + Uy * Uy);
    const ECameraModel Model = CameraModel == ECameraModel::BrownConrady ? ECameraModel::Perspective : CameraModel;
    const float Alpha = CameraModelUtil::ComputeAngle(Model, R, KannalaBrandtCameraCoefficients);
    if (R < 1e-9F)
        return FVector(1.0, 0.0, 0.0);
    const float SinA = FMath::Sin(Alpha);
    return FVector(FMath::Cos(Alpha), SinA * Ux / R, -SinA * Uy / R);
}

void ASceneCaptureSensor_WideAngleLens::ComputeLensCoverage(uint8& OutMask, float& OutMaxFrontTan) const
{
    OutMask = 0;
    OutMaxFrontTan = 0.0F;
    const int32 W = static_cast<int32>(ImageWidth);
    const int32 H = static_cast<int32>(ImageHeight);
    if (W <= 0 || H <= 0)
        return;

    auto Accumulate = [&](int32 X, int32 Y)
    {
        const FVector D = PixelToRay(static_cast<float>(X), static_cast<float>(Y));
        const FVector A = D.GetAbs();
        uint8 Face;
        if (A.X >= A.Y && A.X >= A.Z)
            Face = D.X < 0 ? CubeFace_NegX : CubeFace_PosX;
        else if (A.Y >= A.X && A.Y >= A.Z)
            Face = D.Y < 0 ? CubeFace_NegY : CubeFace_PosY;
        else
            Face = D.Z < 0 ? CubeFace_NegZ : CubeFace_PosZ;
        OutMask |= (uint8)(1U << Face);
        if (Face == CubeFace_PosX)
            OutMaxFrontTan = FMath::Max(OutMaxFrontTan, (float)FMath::Max(A.Y, A.Z) / (float)A.X);
    };

    // 全部边界像素（单调镜头模型的最大视角都在边界上）+ 内部 16 像素间隔网格（选面：
    // 超广角时某个面可能整个落在画面内部而不碰边界）
    for (int32 X = 0; X < W; ++X)
    {
        Accumulate(X, 0);
        Accumulate(X, H - 1);
    }
    for (int32 Y = 0; Y < H; ++Y)
    {
        Accumulate(0, Y);
        Accumulate(W - 1, Y);
    }
    for (int32 Y = 0; Y < H; Y += 16)
        for (int32 X = 0; X < W; X += 16)
            Accumulate(X, Y);
}

void ASceneCaptureSensor_WideAngleLens::CaptureSceneExtended()
{
    TRACE_CPUPROFILER_EVENT_SCOPE(ASceneCaptureSensor_WideAngleLens::CaptureSceneExtended);

    bool SkipVFTR = CVarWideAngleSensorSkipVFTR.GetValueOnAnyThread() != 0;

    // Capture the previous value so it can be restored afterwards, instead of
    // unconditionally forcing the cvar back on and clobbering the project /
    // scalability setting for every other view in the scene.
    IConsoleVariable* VolumetricFogTRVar = nullptr;
    int32 PreviousVolumetricFogTR = 1;

    if (!SkipVFTR)
    {
        VolumetricFogTRVar = IConsoleManager::Get().FindConsoleVariable(
            TEXT("r.VolumetricFog.TemporalReprojection"));
        if (VolumetricFogTRVar != nullptr)
        {
            PreviousVolumetricFogTR = VolumetricFogTRVar->GetInt();
            FlushRenderingCommands();
            VolumetricFogTRVar->Set(TEXT("0"), ECVF_SetByCode);
        }
    }

    // Show-only mode: refresh the actor list (rescans only when the world's
    // actor count changed) and push it into every face capture.
    if (ShowOnlyFilter.IsEnabled())
    {
        ShowOnlyFilter.Refresh(GetWorld(), &GetEpisode());
        for (auto FaceCapture : FaceCaptures)
            if (FaceCapture != nullptr)
                ShowOnlyFilter.ApplyTo(*FaceCapture);
    }

    for (uint8 i = 0; i < 6; ++i)
        if (CubemapRenderMask & (1U << i))
            FaceCaptures[i]->CaptureScene();

    // Snapshot the data the render-thread pass needs so the lambda does not
    // dereference `this` after the actor may have been destroyed. The render
    // targets and sampler are still resolved through a TWeakObjectPtr so we
    // skip the dispatch cleanly if the owning actor is gone.
    TWeakObjectPtr<ASceneCaptureSensor_WideAngleLens> WeakSelf(this);
    UTextureRenderTarget2D* RenderTargetsSnapshot[] =
    {
        FaceRenderTargets[0],
        FaceRenderTargets[1],
        FaceRenderTargets[2],
        FaceRenderTargets[3],
        FaceRenderTargets[4],
        FaceRenderTargets[5]
    };
    UTextureRenderTarget2D* CaptureRenderTargetSnapshot = CaptureRenderTarget;
    FRHISamplerState* CubemapSamplerSnapshot = CubemapSampler;

    CameraModelUtil::FDistortCubemapToImageOptions DistortedOptions = { };
    DistortedOptions.KannalaBrandtCoefficients = KannalaBrandtCameraCoefficients;
    DistortedOptions.LUT = LensLUT;
    DistortedOptions.ThetaMax = LensThetaMax;
    DistortedOptions.bExplicitIntrinsics = bExplicitIntrinsics;
    // Normalized viewport units -> pixels (x by width, y by height).
    DistortedOptions.Intrinsics = FVector4f(
        LensIntrinsics.X * static_cast<float>(ImageWidth),
        LensIntrinsics.Y * static_cast<float>(ImageHeight),
        LensIntrinsics.Z * static_cast<float>(ImageWidth),
        LensIntrinsics.W * static_cast<float>(ImageHeight));
    {
        const FLinearColor Invalid = GetInvalidPixelColor();
        DistortedOptions.InvalidColor = FVector4f(Invalid.R, Invalid.G, Invalid.B, Invalid.A);
    }
    DistortedOptions.YFOVAngle = YFOVAngle;
    DistortedOptions.YFocalLength = YFocalLength;
    DistortedOptions.XFocalLength = XFocalLength;
    DistortedOptions.PrincipalPointOffset = FVector2D(
        PrincipalPointX - ImageWidth * 0.5F,
        PrincipalPointY - ImageHeight * 0.5F);
    DistortedOptions.LongitudeOffset = LongitudeOffset;
    DistortedOptions.FOVFadeSize = FOVFadeSize;
    DistortedOptions.CameraModel = CameraModel;
    DistortedOptions.bRenderEquirectangular = bRenderEquirectangular;
    DistortedOptions.bFOVMaskEnable = bFOVMaskEnable;
    DistortedOptions.bRenderPerspective = bRenderPerspective;
    DistortedOptions.FrontFaceTanHalfFOV = FMath::Tan(FrontFaceHalfFOV);
    DistortedOptions.bSupersample = bSupersampleDistortion;

    ENQUEUE_RENDER_COMMAND(WideAngleLensCommand)(
        [WeakSelf,
         RenderTargetsSnapshot,
         CaptureRenderTargetSnapshot,
         CubemapSamplerSnapshot,
         DistortedOptions](FRHICommandListImmediate& RHICmdList)
    {
        TRACE_CPUPROFILER_EVENT_SCOPE(WideAngleLensCommand);

        if (!WeakSelf.IsValid())
            return;

        FRDGBuilder GraphBuilder(RHICmdList);

        UTextureRenderTarget2D* RenderTargets[] =
        {
            RenderTargetsSnapshot[0],
            RenderTargetsSnapshot[1],
            RenderTargetsSnapshot[2],
            RenderTargetsSnapshot[3],
            RenderTargetsSnapshot[4],
            RenderTargetsSnapshot[5]
        };

        CameraModelUtil::DistortCubemapToImage(
            GraphBuilder,
            CaptureRenderTargetSnapshot,
            RenderTargets,
            CubemapSamplerSnapshot,
            DistortedOptions);

        GraphBuilder.Execute();
    });

    if (CVarWideAngleSensorDumpAllFrames.GetValueOnAnyThread() == 1)
    {
        static thread_local auto FrameCounter = 0U;

        const TCHAR* Names[(size_t)ECameraModel::MaxEnum] =
        {
            TEXT("Perspective"),
            TEXT("Stereographic"),
            TEXT("Equidistant"),
            TEXT("Equisolid"),
            TEXT("Orthographic"),
            TEXT("KannalaBrandt"),
            TEXT("BrownConrady"),
            TEXT("LUT1D"),
            TEXT("FTheta"),
        };

        auto CameraTypeName = Names[(uint8)CameraModel];
        auto Path = CVarWideAngleSensorDumpAllFramesPath.GetValueOnAnyThread();

        if (CVarWideAngleSensorDumpAllFramesCubemap.GetValueOnAnyThread())
        {
            for (uint8 FaceIndex = 0; FaceIndex != 6; ++FaceIndex)
            {
                FPixelReader::SavePixelsToDisk(
                    *FaceRenderTargets[FaceIndex],
                    FString::Printf(
                        TEXT("%s/Frame-%s-%u-Face-%u.png"),
                        *Path,
                        CameraTypeName,
                        FrameCounter,
                        FaceIndex)).Wait();
            }
        }

        FPixelReader::SavePixelsToDisk(
            *CaptureRenderTarget,
            FString::Printf(
                TEXT("%s/Frame-%s-%u-Final.png"),
                *Path,
                CameraTypeName,
                FrameCounter)).Wait();

        ++FrameCounter;
    }

    if (VolumetricFogTRVar != nullptr)
    {
        FlushRenderingCommands();
        VolumetricFogTRVar->Set(*FString::FromInt(PreviousVolumetricFogTR), ECVF_SetByCode);
    }
}

void ASceneCaptureSensor_WideAngleLens::BeginPlay()
{
    const bool bInForceLinearGamma = !bEnablePostProcessingEffects;

    // 面边长按镜头有效像素密度推算 (2026-10-15)：之前固定 max(宽, 高)，90° 视场的 B 柱/翼子板相机输出每弧度
    // ~815 像素，而 90° 面中心只有 640，等于放大采样。前向面先按射线覆盖收窄，再按收窄后的角度推算边长，
    // 最后用新边长重算一次收窄余量 (余量只依赖 1.5 个纹素，第二次几乎不变)
    if (FaceSize <= 0)
    {
        UpdateFrontFaceProjection();
        AutoFaceSide = ComputeRequiredFaceSide();
        UpdateFrontFaceProjection();
        UE_LOG(LogCarla, Log, TEXT("[WideAngleLens] %s: %ux%u output -> cube face side %u (front half-FOV %.2f deg)"),
            *GetName(), GetImageWidth(), GetImageHeight(), AutoFaceSide, FMath::RadiansToDegrees(FrontFaceHalfFOV));
    }

    const auto Format = bEnable16BitFormat ? PF_FloatRGBA : PF_B8G8R8A8;

    // Cube faces are captured at a fixed "Side" resolution and then resampled
    // into the final image by WideAngleLens.usf. A cube face covers ~90° of
    // view natively, so a camera configured with a narrower FOV than that is
    // effectively cropping-and-magnifying a small slice of a fixed-resolution
    // face — the narrower the FOV, the softer the output, independent of
    // scene detail or render quality. Confirmed empirically (2026-08-28,
    // matched-FOV pinhole-vs-fisheye A/B): a 37.5°-vertical fisheye camera
    // scored ~217 vs. ~1399 (Laplacian variance, a sharpness proxy) for an
    // otherwise-identical pinhole camera at the same angular extent — this
    // is why the two narrowest cameras in the 8-camera rig (front_main 37.5°,
    // front_narrow 26.25°, see occnetv3_data_generator/config/camera_config.py)
    // come out visibly blurrier than the wide ones (front_wide/rear 90°) even
    // though all 8 share the same 1280x960 output resolution.
    //
    // 2026-08-28, first attempt (reverted): scaled Side for ALL 6
    // FaceRenderTargets uniformly, regardless of which faces
    // ComputeCubemapRenderMask() would actually render into. Under the full
    // 8-camera rig this hard-crashed at cap=2560 (CreateDescriptorHeap
    // E_INVALIDARG, WS 44-46 GB) and a cap=1536 retest was inconclusive (a
    // silently no-op'd rebuild meant it likely re-tested the already-crashed
    // 2560 binary; that run also had an unrelated CPU-pinning background
    // process running, a second confound now removed). Reverted to a single
    // fixed Side for a while — see git history on this file for that
    // intermediate state.
    //
    // 2026-08-28, second attempt (reverted): re-examined
    // ComputeCubemapRenderMask() and found front_main/front_narrow's FOV
    // stays under 90 degrees even after the sqrt(2) safety margin, so their
    // mask is CubeFace_PosX only (1 of 6 faces) — WideAngleLens.usf's
    // SampleCubemap() never samples the other 5 for these cameras. Scaling
    // only the mask-selected face (instead of all 6, as the first attempt
    // did) should have cost under 250 MB total across both narrow cameras
    // even at a 4096 cap — but memory-budget math turned out not to predict
    // this pipeline's actual behavior at all. Tested twice:
    //   - First test: GPU VRAM was independently discovered afterward to
    //     have been at 23.1/24.6 GB from ~20 unrelated background desktop
    //     apps. Hung on the first world.tick(); WS climbed 1.5->55 GB with
    //     near-zero CPU and never recovered (had to force-kill).
    //   - Retested the fully-reverted (no scaling) code under that same
    //     near-full VRAM and got a similar WS-ballooning symptom (52 GB) —
    //     so that first failure alone wasn't clean evidence against this
    //     code specifically.
    //   - Third test, after closing the background apps (nvidia-smi
    //     confirmed 1.9-2.2/24.6 GB used both before spawning and again
    //     right after the process recovered): spawning the 16-camera A/B
    //     rig still failed the first world.tick() (30s timeout), WS
    //     spiked to ~42 GB then dropped back to ~1.5 GB — the Windows
    //     process itself recovered and looked idle/healthy (low WS,
    //     "Responding: true"), but the CARLA RPC server inside it stayed
    //     dead: even a trivial client.get_world() call timed out
    //     repeatedly afterward (20s+), requiring another force-kill.
    // Two independent hangs on the exact same operation (first tick after
    // spawning these two per-face-enlarged cameras), the second with VRAM
    // headroom confirmed both before and after, rules out VRAM contention
    // as the sole explanation. This is real evidence of a defect in the
    // per-face approach itself, not a memory-budget problem — most likely
    // something about a camera's cube faces having non-uniform sizes
    // breaks an assumption in the RDG/descriptor pipeline (a shared pool
    // keyed without size, a synchronization primitive one of the newly
    // large faces never signals, etc.). Reverted to a single fixed Side
    // for all 6 faces after that.
    //
    // 2026-08-28, fourth attempt (also reverted — STOP retrying without a
    // GPU/RDG profiler): both prior hangs were observed under a 16-camera
    // A/B rig (8 fisheye + 8 pinhole test cameras) at an aggressive cap
    // (4096, ~3.4x BaseSide for front_narrow). This attempt controlled for
    // both variables at once: tested against the REAL production 8-camera
    // pipeline (main_collection.py, no extra pinhole cameras) with the cap
    // cut to 2048 (~1.6x BaseSide). Result: hung again, at the identical
    // checkpoint (first world.tick() after spawning, 60s timeout this
    // time since production uses a longer client timeout than the A/B
    // script), WS climbed to ~48 GB, and this time the Windows process
    // itself reported "Responding: False" (not just the CARLA RPC layer —
    // worse than the third attempt's symptom). Three independent hangs now,
    // each with a different risk factor removed (CPU contention, VRAM
    // contention, extra test cameras, aggressive cap) and the failure
    // still reproduces identically every time. This rules out "too much
    // load" or "too aggressive a scale factor" as the explanation — it is
    // the non-uniform face size itself, at any scale, under any camera
    // count. Do not attempt a fifth variant of this idea without actually
    // attaching RenderDoc or PIX to see what the first tick's GPU timeline
    // looks like; further guessing at cap values or camera counts is not
    // going to find this bug.
    // 5.8 上游新增 FaceSize 属性：>0 时所有 6 个面统一用该尺寸（统一尺寸，不触发上面记录的
    // 非均匀面尺寸卡死问题）。默认 0 时用本函数开头按镜头有效像素密度推算的 AutoFaceSide——同样是 6 个面
    // 统一一个边长，只是不同相机之间不同。2026-10-15 在 81 上只验证过 1280/2560 两种面尺寸的相机同帧共存，
    // 8 相机生产配置要以实际采集验收为准；出问题时 camera_manager 可显式传 face_size 回退到固定边长。
    const auto Side = GetFaceSide();

    CaptureRenderTarget->InitCustomFormat(
        GetImageWidth(),
        GetImageHeight(),
        Format,
        bInForceLinearGamma);

    for (auto FaceRenderTarget : FaceRenderTargets)
    {
        FaceRenderTarget->InitCustomFormat(
            Side, Side,
            Format,
            bInForceLinearGamma);
    }

    if (bEnablePostProcessingEffects)
    {
        for (auto Face : FaceRenderTargets)
            Face->TargetGamma = TargetGamma;
        CaptureRenderTarget->TargetGamma = TargetGamma;
    }

    for (uint8 i = 0; i != 6; ++i)
    {
        FaceCaptures[i]->Deactivate();
        FaceCaptures[i]->TextureTarget = FaceRenderTargets[i];
    }

    SetUpSceneCaptureComponents(FaceCaptures);

    for (auto FaceCapture : FaceCaptures)
    {
        // Use PostProcessConfig for each face capture to honour the
        // ue5-dev convention (mirrors ASceneCaptureSensor::BeginPlay).
        auto PostProcessConfig = FPostProcessConfig(
            FaceCapture->PostProcessSettings,
            FaceCapture->ShowFlags);
        PostProcessConfig.UpdateFromSceneCaptureComponent2D(*FaceCapture);
        PostProcessConfig.EnablePostProcessingEffects(bEnablePostProcessingEffects);
        FaceCapture->ShowFlags = PostProcessConfig.EngineShowFlags;
        FaceCapture->PostProcessSettings = PostProcessConfig.PostProcessSettings;

        if (bEnablePostProcessingEffects)
        {
            // Effects computed per view (per face) leave seams in the stitched image even when all faces share
            // one exposure (see below): local exposure (bilateral grid / fusion), bloom and lens flares.
            // A real camera has one sensor; keep only the global, shared exposure + tone curve.
            auto &PP = FaceCapture->PostProcessSettings;
            PP.bOverride_LocalExposureHighlightContrastScale = true;
            PP.LocalExposureHighlightContrastScale = 1.0f;
            PP.bOverride_LocalExposureShadowContrastScale = true;
            PP.LocalExposureShadowContrastScale = 1.0f;
            PP.bOverride_LocalExposureDetailStrength = true;
            PP.LocalExposureDetailStrength = 1.0f;
            FaceCapture->ShowFlags.SetBloom(false);
            FaceCapture->ShowFlags.SetLensFlares(false);

            // 运动模糊 (2026-10-15)：motion_blur_* 属性以前注册了却从没应用，各面一直是引擎默认强度 0.5，
            // 运动模糊显示开关又被 EnablePostProcessingEffects 打开 (针孔相机宽 <1920 时会强制关掉，鱼眼没有)。
            // 这里按属性应用 (采集配置为 0 → 关闭)。注意：静止场景的鱼眼发软不是它造成的 (r.MotionBlurQuality=0 对照无变化)
            PP.bOverride_MotionBlurAmount = true;
            PP.MotionBlurAmount = FMath::Max(MotionBlurIntensity, 0.0F);
            PP.bOverride_MotionBlurMax = true;
            PP.MotionBlurMax = MotionBlurMaxDistortion;
            PP.bOverride_MotionBlurPerObjectSize = true;
            PP.MotionBlurPerObjectSize = MotionBlurMinObjectScreenSize;
            if (MotionBlurIntensity <= 0.0F)
                FaceCapture->ShowFlags.SetMotionBlur(false);

            // cube 面不跑 TSR (2026-10-15)：81 同帧收敛曲线——针孔清晰度 (中心拉普拉斯方差) 随帧数
            // 268→760→1948 持续上升，鱼眼面 423→104→286 不升反降，即面上的时序历史累积失败、越累积越糊；
            // 同时 TSR 对运动的车/人会留拖影，不适合训练数据。RGB 相机改为面按 ~2 倍角密度渲染 +
            // 畸变 shader 像素面积积分 (确定性超采样)
            if (bSupersampleDistortion)
            {
                FaceCapture->ShowFlags.SetTemporalAA(false);
                FaceCapture->ShowFlags.SetAntiAliasing(false);
            }

            // 景深与镜头瑕疵 (2026-10-15，鱼眼长焦发糊的根因)：scene capture 会混入地图的后处理体积
            // (FSceneView::StartFinalPostprocessSettings → UWorld::AddPostProcessingSettings)。针孔相机对几乎所有
            // 相机字段都设了 bOverride_ (SceneCaptureSensor.cpp SetCameraDefaultOverrides)，地图体积的景深进不来；
            // 鱼眼各面只覆盖了 3 个字段，Town10 体积里的景深直接作用在各面上。景深模糊随焦距放大、与面分辨率无关：
            // 81 像素对齐对照，front_narrow (35°) 清晰度只有同视场针孔的 0.07 倍，面加到 4096 也不变；
            // 启动参数 r.DepthOfFieldQuality=0 后变为 1.30 倍。车载相机是定焦超焦距镜头 (近处到无穷远都清晰)，
            // 这里直接关掉景深，并把色差、胶片颗粒覆盖为 0 (与针孔的默认覆盖一致)
            FaceCapture->ShowFlags.SetDepthOfField(false);
            PP.bOverride_DepthOfFieldFocalDistance = true;
            PP.DepthOfFieldFocalDistance = 0.0f;
            PP.bOverride_DepthOfFieldDepthBlurRadius = true;
            PP.DepthOfFieldDepthBlurRadius = 0.0f;
            PP.bOverride_SceneFringeIntensity = true;
            PP.SceneFringeIntensity = 0.0f;
            PP.bOverride_FilmGrainIntensity = true;
            PP.FilmGrainIntensity = 0.0f;
        }

        FaceCapture->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
        FaceCapture->UpdateContent();
        FaceCapture->Activate();
    }

    // One auto exposure for the whole camera: faces 1..5 use face 0's (front, +X) eye adaptation through the
    // engine's ExposureSceneViewStateInterface (engine patch SceneCaptureExposureSharing.h). Before this every
    // face metered on its own, so histogram exposure (any post_process_profile) left brightness seams.
    // Face 0 is always in the render mask (ComputeCubemapRenderMask) and is captured first in
    // CaptureSceneExtended, so the other faces read the exposure it adapted in the same frame.
    for (int32 i = 1; i < FaceCaptures.Num(); ++i)
        SceneCaptureExposureSharing::SetExposureSource(FaceCaptures[i], FaceCaptures[0]);

    // Raise the render-fence block timeout: a single wide-angle capture
    // enqueues up to 6 face scene-captures plus the RDG composite pass, which
    // can exceed the default timeout on heavy scenes and trip a false-positive
    // render-fence fatal assert. This is a process-wide setting; CARLA already
    // runs the render pipeline outside interactive-latency budgets, so a wider
    // watchdog is harmless for other views.
    UKismetSystemLibrary::ExecuteConsoleCommand(
        GetWorld(),
        FString("g.TimeoutForBlockOnRenderFence 300000"));

    // This ensures the camera is always spawning the raindrops in case the
    // weather was previously set to have rain.
    auto Weather = GetEpisode().GetWeather();
    if (Weather != nullptr)
        Weather->NotifyWeather(this);

    ReadbackPool = MakeShared<FRHIGPUReadbackPool, ESPMode::ThreadSafe>(
        TEXT("WideAngleLensReadback"));

    Super::BeginPlay();
}

void ASceneCaptureSensor_WideAngleLens::PrePhysTick(float DeltaSeconds)
{
    Super::PrePhysTick(DeltaSeconds);

    auto CaptureComponents = GetCaptureComponents2D();

    // Add the view information every tick. It's only used for one tick and then
    // removed by the streamer.
    // 按前向面的实际捕获向流送系统报告视图需求：屏幕尺寸 = 面渲染目标边长，视角 = 实际捕获半角
    // (单面长焦时已收窄)。之前用组件 FOVAngle (固定 90 度) 和输出宽度报告，长焦相机实际放大了远处物体，
    // 流送却按宽视野计算，远处纹理的高精度 mip 可能不加载。
    const float Side = static_cast<float>(GetFaceSide());
    const float HalfFOVTan = FMath::Tan(FrontFaceHalfFOV);
    IStreamingManager::Get().AddViewInformation(
        CaptureComponents[0]->GetComponentLocation(),
        Side,
        HalfFOVTan > 0.0f ? Side / HalfFOVTan : Side);
}

void ASceneCaptureSensor_WideAngleLens::PostPhysTick(UWorld* World, ELevelTick TickType, float DeltaTime)
{
    Super::PostPhysTick(World, TickType, DeltaTime);
}

void ASceneCaptureSensor_WideAngleLens::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    Super::EndPlay(EndPlayReason);
    // The sensor counter is process-wide and only used to derive unique
    // subobject names at construction. Resetting it here would let a
    // surviving sibling sensor collide with the indices of the next spawned
    // sensor.
    FlushRenderingCommands();

    // Release the GPU memory now rather than at the next garbage collection:
    // six persistent face view states (Lumen scene, TSR history, shadow
    // caches; several GB at 2048^2) plus the seven render targets. The
    // UObjects stay alive until GC, but without their resources.
    for (auto FaceCapture : FaceCaptures)
    {
        if (FaceCapture == nullptr)
            continue;
        SceneCaptureExposureSharing::SetExposureSource(FaceCapture, nullptr);
        FaceCapture->Deactivate();
        FaceCapture->TextureTarget = nullptr;
        FaceCapture->ReleaseViewStates();
    }
    for (auto FaceRenderTarget : FaceRenderTargets)
    {
        if (FaceRenderTarget != nullptr)
            FaceRenderTarget->ReleaseResource();
    }
    if (CaptureRenderTarget != nullptr)
        CaptureRenderTarget->ReleaseResource();
    FlushRenderingCommands();
    // In-flight readback tasks keep their own reference; the pool dies with the
    // last of them (see ASceneCaptureSensor::EndPlay).
    ReadbackPool.Reset();
}

TArrayView<USceneCaptureComponent2D_CARLA*> ASceneCaptureSensor_WideAngleLens::GetCaptureComponents2D()
{
    return FaceCaptures;
}

float ASceneCaptureSensor_WideAngleLens::VerticalToHorizontal(
    float Value) const
{
    Value *= GetImageWidth();
    Value /= GetImageHeight();
    return Value;
}



// =============================================================================
// -- Local static functions implementations -----------------------------------
// =============================================================================

namespace SceneCaptureSensorWideAngleLens_local_ns {

    static void SetCameraDefaultOverrides(USceneCaptureComponent2D_CARLA& CaptureComponent)
    {
        FPostProcessSettings& PostProcessSettings = CaptureComponent.PostProcessSettings;
        PostProcessSettings.bOverride_VignetteIntensity = true;
        PostProcessSettings.VignetteIntensity = 0;
        PostProcessSettings.bOverride_DepthOfFieldVignetteSize = true;
        PostProcessSettings.DepthOfFieldVignetteSize = 0;
        PostProcessSettings.bOverride_AutoExposureMethod = true;
        PostProcessSettings.AutoExposureMethod = EAutoExposureMethod::AEM_Manual;

        // 与针孔相机 (SceneCaptureSensor.cpp 同名函数) 一致的 Lumen 质量设置 (2026-10-15)：之前鱼眼各面一项都没设，
        // 全靠地图后处理体积和引擎默认值，同一场景两种相机的 GI/反射质量不一致
        PostProcessSettings.bOverride_DynamicGlobalIlluminationMethod = true;
        PostProcessSettings.DynamicGlobalIlluminationMethod = EDynamicGlobalIlluminationMethod::Lumen;
        PostProcessSettings.bOverride_LumenSceneLightingQuality = true;
        PostProcessSettings.LumenSceneLightingQuality = 1.0f;
        PostProcessSettings.bOverride_LumenSceneDetail = true;
        PostProcessSettings.LumenSceneDetail = 1.0f;
        PostProcessSettings.bOverride_LumenSceneViewDistance = true;
        PostProcessSettings.LumenSceneViewDistance = 20000.0f;
        PostProcessSettings.bOverride_LumenFinalGatherQuality = true;
        PostProcessSettings.LumenFinalGatherQuality = 1.0f;
        PostProcessSettings.bOverride_LumenMaxTraceDistance = true;
        PostProcessSettings.LumenMaxTraceDistance = 20000.0f;
        PostProcessSettings.bOverride_LumenSurfaceCacheResolution = true;
        PostProcessSettings.LumenSurfaceCacheResolution = 1.0f;
        PostProcessSettings.bOverride_LumenSceneLightingUpdateSpeed = true;
        PostProcessSettings.LumenSceneLightingUpdateSpeed = 1.0f;
        PostProcessSettings.bOverride_LumenFinalGatherLightingUpdateSpeed = true;
        PostProcessSettings.LumenFinalGatherLightingUpdateSpeed = 1.0f;
        PostProcessSettings.bOverride_LumenDiffuseColorBoost = true;
        PostProcessSettings.LumenDiffuseColorBoost = 1.0f;
        PostProcessSettings.bOverride_LumenSkylightLeaking = true;
        PostProcessSettings.LumenSkylightLeaking = 0.1f;
        PostProcessSettings.bOverride_LumenFullSkylightLeakingDistance = true;
        PostProcessSettings.LumenFullSkylightLeakingDistance = 1000.0f;
        PostProcessSettings.bOverride_ReflectionMethod = true;
        PostProcessSettings.ReflectionMethod = EReflectionMethod::Lumen;
        PostProcessSettings.bOverride_LumenReflectionQuality = true;
        PostProcessSettings.LumenReflectionQuality = 1.0f;
        PostProcessSettings.bOverride_LumenRayLightingMode = true;
        PostProcessSettings.LumenRayLightingMode = ELumenRayLightingModeOverride::Default;
        PostProcessSettings.bOverride_LumenFrontLayerTranslucencyReflections = true;
        PostProcessSettings.LumenFrontLayerTranslucencyReflections = true;
        PostProcessSettings.bOverride_LumenMaxReflectionBounces = true;
        PostProcessSettings.LumenMaxReflectionBounces = 3;
    }


} // namespace SceneCaptureSensorWideAngleLens_local_ns
