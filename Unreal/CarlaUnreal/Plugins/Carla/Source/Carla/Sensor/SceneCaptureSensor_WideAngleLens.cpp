// Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include "SceneCaptureSensor_WideAngleLens.h"
#include "Carla/Game/CarlaStatics.h"
#include "Carla/Actor/ActorBlueprintFunctionLibrary.h"
#include "Carla/Sensor/PostProcessConfig.h"

#include <util/ue-header-guard-begin.h>
#include "Engine/Engine.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphUtils.h"
#include "HAL/IConsoleManager.h"
#include "Kismet/KismetSystemLibrary.h"
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
    bFOVMaskEnable(false)
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
        FaceCapture->bUseRayTracingIfEnabled = bUseRayTracing;
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

    bool UpdateRenderMask = NewFOV != YFOVAngle;

    YFOVAngle = NewFOV;
    XFOVAngle = VerticalToHorizontal(NewFOV);

    YFocalLength = CameraModelUtil::ComputeDistance(
        CameraModel,
        NewFOV,
        ImageHeight,
        KannalaBrandtCameraCoefficients);

    // Keep XFocalLength tracking the aspect-ratio-derived XFOVAngle by
    // default (isotropic fx=fy, the pre-existing behavior). A later explicit
    // SetFOVAngleX() call overrides both again — see its own comment.
    XFocalLength = CameraModelUtil::ComputeDistance(
        CameraModel,
        XFOVAngle,
        ImageWidth,
        KannalaBrandtCameraCoefficients);

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

    XFocalLength = CameraModelUtil::ComputeDistance(
        CameraModel,
        NewFOV,
        ImageWidth,
        KannalaBrandtCameraCoefficients);

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
}

float ASceneCaptureSensor_WideAngleLens::GetPrincipalPointX() const
{
    return PrincipalPointX;
}

float ASceneCaptureSensor_WideAngleLens::GetPrincipalPointY() const
{
    return PrincipalPointY;
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
            FaceCapture->bUseRayTracingIfEnabled = Enable;
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
    const float Theta = CameraModelUtil::ComputeAngle(CameraModel, R, KannalaBrandtCameraCoefficients);
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

    auto Mask = 1U << CubeFace_PosX; // Render front face by default.

    const auto FOV = FVector2D(GetFOVAngleX(), GetFOVAngleY()) * (GetFOVMaskEnable() ? 1 : Sqrt2);

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
    static const float Sqrt2 = sqrtf(2.0f);

    float NewHalfFOV = DefaultHalfFOV;

    // Narrowing the face is only valid when SampleCubemap() (WideAngleLens.usf)
    // is guaranteed to only ever look up FaceIndex 0 for this camera — every
    // other face's UV math still assumes the untouched +/-45 deg convention.
    const bool bSingleFace = CubemapRenderMask == (1U << CubeFace_PosX);

    if (bSingleFace)
    {
        // Mirrors ComputeCubemapRenderMask()'s own safety margin exactly, so
        // the captured cone is never narrower than what that mask decision
        // already assumed was sufficient to cover.
        const float Margin = GetFOVMaskEnable() ? 1.0F : Sqrt2;
        const float CandidateHalfFOV = FMath::Max(XFOVAngle, YFOVAngle) * 0.5F * Margin;

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
    }

    UE_LOG(LogCarla, Log,
        TEXT("[WideAngleLens] %s: front face capture half-FOV -> %.2f deg (bSingleFace=%d, XFOV=%.2f, YFOV=%.2f)"),
        *GetName(), FMath::RadiansToDegrees(FrontFaceHalfFOV), bSingleFace,
        FMath::RadiansToDegrees(XFOVAngle), FMath::RadiansToDegrees(YFOVAngle));
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
    const int32 Side = static_cast<int32>(std::max(GetImageWidth(), GetImageHeight()));

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

        FaceCapture->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
        FaceCapture->UpdateContent();
        FaceCapture->Activate();
    }

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

    Super::BeginPlay();
}

void ASceneCaptureSensor_WideAngleLens::PrePhysTick(float DeltaSeconds)
{
    Super::PrePhysTick(DeltaSeconds);

    auto CaptureComponents = GetCaptureComponents2D();

    // Add the view information every tick. It's only used for one tick and then
    // removed by the streamer.
    // FOVAngle is stored in degrees; convert to radians and use the
    // half-angle tangent, guarding against a non-positive denominator.
    const float HalfFOVTan = FMath::Tan(
        0.5f * FMath::DegreesToRadians(CaptureComponents[0]->FOVAngle));
    IStreamingManager::Get().AddViewInformation(
        CaptureComponents[0]->GetComponentLocation(),
        ImageWidth,
        HalfFOVTan > 0.0f ? ImageWidth / HalfFOVTan : ImageWidth);
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
    }


} // namespace SceneCaptureSensorWideAngleLens_local_ns
