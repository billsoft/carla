// Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "Carla/Actor/ActorDefinition.h"
#include "Carla/Sensor/PixelReader.h"
#include "Carla/Sensor/ShaderBasedSensor.h"
#include "Carla/Util/CameraModelUtil.h"

#include <util/ue-header-guard-begin.h>
#if __has_include("GBufferView.h")
#define CARLA_HAS_GBUFFER_API
#include "GBufferView.h"
#endif
#include <util/ue-header-guard-end.h>

#include "SceneCaptureSensor_WideAngleLens.generated.h"



UCLASS()
class CARLA_API ASceneCaptureSensor_WideAngleLens : public ASensor
{
  GENERATED_BODY()

  friend class ACarlaGameModeBase;
  friend class FPixelReader;

public:

  ASceneCaptureSensor_WideAngleLens(const FObjectInitializer &ObjectInitializer);

  void Set(const FActorDescription &ActorDescription) override;

  UFUNCTION(BlueprintCallable)
  void SetImageSize(int32 Width, int32 Height);

  void SetImageSize(uint32 Width, uint32 Height);

  uint32 GetImageWidth() const
  {
    return ImageWidth;
  }

  uint32 GetImageHeight() const
  {
    return ImageHeight;
  }

  UFUNCTION(BlueprintCallable, BlueprintPure)
  FIntPoint GetImageSize() const
  {
    return FIntPoint(GetImageWidth(), GetImageHeight());
  }

  UFUNCTION(BlueprintCallable)
  void EnablePostProcessingEffects(bool Enable = true)
  {
    bEnablePostProcessingEffects = Enable;
  }

  UFUNCTION(BlueprintCallable)
  void Enable16BitFormat(bool Enable = true)
  {
    bEnable16BitFormat = Enable;
  }

  UFUNCTION(BlueprintCallable, BlueprintPure)
  bool Is16BitFormatEnabled() const
  {
    return bEnable16BitFormat;
  }

  UFUNCTION(BlueprintCallable, BlueprintPure)
  ECameraModel GetCameraModel() const;

  UFUNCTION(BlueprintCallable)
  void SetCameraModel(ECameraModel NewCameraModel);

  UFUNCTION(BlueprintCallable, BlueprintPure)
  float GetFOVAngle() const;

  UFUNCTION(BlueprintCallable, BlueprintPure)
  float GetFOVAngleY() const;

  UFUNCTION(BlueprintCallable, BlueprintPure)
  float GetFOVAngleX() const;

  UFUNCTION(BlueprintCallable)
  void SetFOVAngle(float NewFOV);

  /// Independently overrides the horizontal FOV (and derived horizontal focal
  /// length), which otherwise tracks SetFOVAngle's vertical value scaled by
  /// the image aspect ratio (see VerticalToHorizontal). Call after
  /// SetFOVAngle to simulate a physical lens whose horizontal/vertical FOV
  /// don't share a single isotropic focal length.
  UFUNCTION(BlueprintCallable)
  void SetFOVAngleX(float NewFOV);

  UFUNCTION(BlueprintCallable, BlueprintPure)
  float GetFocalLength() const;

  UFUNCTION(BlueprintCallable)
  void SetFocalLength(float NewFocalLength);

  /// Sets the lens's calibrated principal point (optical center), in pixels,
  /// relative to the image's top-left corner. Defaults to the exact
  /// geometric center (ImageWidth/2, ImageHeight/2) — a real physical lens's
  /// calibrated optical center is rarely exactly there.
  UFUNCTION(BlueprintCallable)
  void SetPrincipalPoint(float Cx, float Cy);

  UFUNCTION(BlueprintCallable, BlueprintPure)
  float GetPrincipalPointX() const;

  UFUNCTION(BlueprintCallable, BlueprintPure)
  float GetPrincipalPointY() const;

  void SetCameraCoefficients(TArrayView<const float> Coefficients);

  UFUNCTION(BlueprintCallable)
  void SetCameraCoefficients(const TArray<float>& Coefficients);

  UFUNCTION(BlueprintCallable, BlueprintPure)
  const TArray<float>& GetCameraCoefficients() const;

  UFUNCTION(BlueprintCallable, BlueprintPure)
  UTextureRenderTarget2D* GetCaptureRenderTarget();

  UFUNCTION(BlueprintCallable, BlueprintPure)
  float GetTargetGamma() const;

  UFUNCTION(BlueprintCallable)
  void SetTargetGamma(float Gamma);

  UFUNCTION(BlueprintCallable)
  void SetRenderPerspective(bool bEnable);

  UFUNCTION(BlueprintCallable, BlueprintPure)
  bool GetRenderPerspective() const;

  UFUNCTION(BlueprintCallable)
  void SetRenderEquirectangular(bool bEnable);

  UFUNCTION(BlueprintCallable, BlueprintPure)
  bool GetRenderEquirectangular() const;

  UFUNCTION(BlueprintCallable)
  void SetFOVMaskEnable(bool bEnable);

  UFUNCTION(BlueprintCallable, BlueprintPure)
  bool GetFOVMaskEnable() const;

  UFUNCTION(BlueprintCallable)
  void SetFOVFadeSize(float NewFOVFadeSize);

  UFUNCTION(BlueprintCallable, BlueprintPure)
  float GetFOVFadeSize() const;

  UFUNCTION(BlueprintCallable)
  void SetRenderEquirectangularLongitudeOffset(float Shift);

  UFUNCTION(BlueprintCallable, BlueprintPure)
  float GetRenderEquirectangularLongitudeOffset() const;

  void SetCubemapSampler(FRHISamplerState* NewCubemapSampler)
  {
    CubemapSampler = NewCubemapSampler;
  }

  const FRHISamplerState* GetCubemapSampler() const
  {
    return CubemapSampler;
  }

  /// Returns null to opt the wide-angle sensor pipeline into FPixelReader's
  /// per-call readback fallback. The wide-angle compute-shader path does not
  /// own a persistent FRHIGPUReadbackPool.
  FRHIGPUReadbackPoolPtr GetReadbackPool() const { return nullptr; }

  UFUNCTION(BlueprintCallable)
  void SetUseRayTracing(bool Enable);

  UFUNCTION(BlueprintCallable, BlueprintPure)
  bool GetUseRayTracing() const
  {
    return bUseRayTracing;
  }

  /// Immediate enqueues render commands of the scene at the current time.
  void EnqueueRenderSceneImmediate();

  /// Blocks until the render thread has finished all its tasks.
  void WaitForRenderThreadToFinish()
  {
    TRACE_CPUPROFILER_EVENT_SCOPE(ASceneCaptureSensor_WideAngleLens::WaitForRenderThreadToFinish);
    FlushRenderingCommands();
  }

  TArrayView<USceneCaptureComponent2D_CARLA*> GetCaptureComponents2D();

protected:

  UFUNCTION(BlueprintCallable, BlueprintPure)
  float VerticalToHorizontal(float Value) const;

  UFUNCTION(BlueprintCallable, BlueprintPure)
  uint8 FindFaceIndex(FVector2D UV) const;

  UFUNCTION(BlueprintCallable, BlueprintPure)
  uint8 ComputeCubemapRenderMask() const;

  void CaptureSceneExtended();

  virtual void BeginPlay() override;

  virtual void PrePhysTick(float DeltaSeconds) override;
  virtual void PostPhysTick(UWorld *World, ELevelTick TickType, float DeltaTime) override;

  virtual void SetUpSceneCaptureComponents(TArrayView<USceneCaptureComponent2D_CARLA*> SceneCaptures) {}

  virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

  UPROPERTY(EditAnywhere)
  TArray<USceneCaptureComponent2D_CARLA*> FaceCaptures;

  UPROPERTY(EditAnywhere)
  TArray<UTextureRenderTarget2D*> FaceRenderTargets;

  UPROPERTY(EditAnywhere)
  UTextureRenderTarget2D* CaptureRenderTarget;

  UPROPERTY(EditAnywhere)
  float TargetGamma;

  UPROPERTY(EditAnywhere)
  uint32 ImageWidth;

  UPROPERTY(EditAnywhere)
  uint32 ImageHeight;

  UPROPERTY(EditAnywhere)
  ECameraModel CameraModel;

  UPROPERTY(EditAnywhere)
  TArray<float> KannalaBrandtCameraCoefficients;

  UPROPERTY(EditAnywhere)
  float YFOVAngle;

  UPROPERTY(EditAnywhere)
  float XFOVAngle;

  UPROPERTY(EditAnywhere)
  float YFocalLength;

  /// Horizontal focal length, independent from YFocalLength. Defaults to
  /// tracking YFocalLength (via XFOVAngle's aspect-ratio derivation) until
  /// SetFOVAngleX is called explicitly.
  UPROPERTY(EditAnywhere)
  float XFocalLength;

  /// Principal point (optical center), in pixels from the image's top-left
  /// corner. Defaults to the exact geometric center.
  UPROPERTY(EditAnywhere)
  float PrincipalPointX;

  UPROPERTY(EditAnywhere)
  float PrincipalPointY;

  UPROPERTY(EditAnywhere)
  float LongitudeOffset;

  UPROPERTY(EditAnywhere)
  float FOVFadeSize;

  uint8 CubemapRenderMask;

  FRHISamplerState* CubemapSampler;

  bool bUseRayTracing : 1;
  bool bEnablePostProcessingEffects : 1;
  bool bEnable16BitFormat : 1;
  bool bRenderPerspective : 1;
  bool bRenderEquirectangular : 1;
  bool bFOVMaskEnable : 1;
};
