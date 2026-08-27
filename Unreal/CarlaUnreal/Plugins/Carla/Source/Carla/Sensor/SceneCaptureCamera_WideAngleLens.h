// Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "Carla/Actor/ActorDefinition.h"
#include "Carla/Sensor/ShaderBasedSensor_WideAngleLens.h"

#include <util/ue-header-guard-begin.h>
#include "Actor/ActorBlueprintFunctionLibrary.h"
#include <util/ue-header-guard-end.h>

#include "SceneCaptureCamera_WideAngleLens.generated.h"

/// A sensor that captures images from the scene.
UCLASS()
class CARLA_API ASceneCaptureCamera_WideAngleLens : public AShaderBasedSensor_WideAngleLens
{
  GENERATED_BODY()

public:

  static FActorDefinition GetSensorDefinition();

  ASceneCaptureCamera_WideAngleLens(const FObjectInitializer& ObjectInitializer);

  /// 像素格式（"uint8"/"uint16"/"float32"/"bayer_rggb"），和 ASceneCaptureCamera::RawType 语义一致
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera Settings")
  FString RawType = TEXT("uint8");

protected:

  void BeginPlay() override;
  void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
  void PostPhysTick(UWorld* World, ELevelTick TickType, float DeltaSeconds) override;

  virtual void OnFirstClientConnected() override;
  virtual void OnLastClientDisconnected() override;

  /// HDR（RawType != "uint8"）数据发送函数，逻辑与 ASceneCaptureCamera::SendHDRDataToClient 一致。
  /// Size 是读回回调报告的实际纹理尺寸（不是 GetImageWidth()/GetImageHeight()），
  /// 避免两者不一致时越界读取 Pixels。
  void SendHDRDataToClient(
    const TArrayView<const FLinearColor>& Pixels,
    FIntPoint Size,
    uint64 FrameIndex);

private:
};
