// Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include "Carla/Sensor/SceneCaptureCamera_WideAngleLens.h"
#include "Carla.h"
#include "Carla/Game/CarlaEngine.h"
#include "Carla/Sensor/ImageUtil.h"

#include "Actor/ActorBlueprintFunctionLibrary.h"

#include "Carla/Sensor/PixelReader.h"

// ========== Bayer RAW / HDR 支持（镜像 SceneCaptureCamera.cpp） ==========
#include <carla/sensor/s11n/ImageSerializer.h>  // HDR 序列化支持
#include <carla/sensor/data/Image.h>            // 像素格式枚举
#include <carla/Buffer.h>                       // Buffer

FActorDefinition ASceneCaptureCamera_WideAngleLens::GetSensorDefinition()
{
  constexpr bool bEnableModifyingPostProcessEffects = true;
  return UActorBlueprintFunctionLibrary::MakeWideAngleLensCameraDefinition(
      TEXT("rgb"),
      bEnableModifyingPostProcessEffects);
}

ASceneCaptureCamera_WideAngleLens::ASceneCaptureCamera_WideAngleLens(const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer)
{
  Super::SetCubemapSampler(CameraModelUtil::GetSampler(ESamplerFilter::SF_AnisotropicLinear));

  EnablePostProcessingEffects(true);
}

void ASceneCaptureCamera_WideAngleLens::BeginPlay()
{
  Super::BeginPlay();
}

void ASceneCaptureCamera_WideAngleLens::OnFirstClientConnected()
{
}

void ASceneCaptureCamera_WideAngleLens::OnLastClientDisconnected()
{
}

void ASceneCaptureCamera_WideAngleLens::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
  Super::EndPlay(EndPlayReason);
}

void ASceneCaptureCamera_WideAngleLens::PostPhysTick(UWorld* World, ELevelTick TickType, float DeltaSeconds)
{
  TRACE_CPUPROFILER_EVENT_SCOPE(ASceneCaptureCamera_WideAngleLens::PostPhysTick);
  // Skip the whole 6-face capture + compute + readback pipeline when no
  // client is subscribed.
  if (!AreClientsListening())
    return;

  auto FrameIndex = FCarlaEngine::GetFrameCounter();

  // ========== 根据 RawType 选择数据捕获方式（镜像 ASceneCaptureCamera::PostPhysTick） ==========
  if (RawType == TEXT("float32") || RawType == TEXT("uint16") || RawType == TEXT("bayer_rggb"))
  {
    // HDR 模式：直接从 distort 后的最终纹理（等距/鱼眼投影结果）读取 FLinearColor。
    // 注意：不能用 ImageUtil::ReadSensorImageDataAsyncFLinearColor(AShaderBasedSensor&, ...)，
    // 因为本类继承自 AShaderBasedSensor_WideAngleLens，不在 AShaderBasedSensor 那条继承链上。
    //
    // 关键：WideAngleLens 的 6 面 cubemap 捕获 + distort compute shader 不是每帧自动触发的
    // （不像普通 USceneCaptureComponent2D 的 bCaptureEveryFrame），而是要显式调用
    // EnqueueRenderSceneImmediate()（内部即 CaptureSceneExtended()）来触发。uint8 分支走的
    // FPixelReader::SendPixelsInRenderThread 内部会自动调用它，但这里是自己发起的读取，必须
    // 在读回纹理之前手动调用一次，否则读到的是上一帧甚至未初始化的显存内容。
    EnqueueRenderSceneImmediate();

    ImageUtil::ReadImageDataAsyncFLinearColor(*GetCaptureRenderTarget(), [this, FrameIndex](
      TArrayView<const FLinearColor> Pixels,
      FIntPoint Size) -> bool
    {
      static bool bLoggedOnce = false;
      if (!bLoggedOnce)
      {
        bLoggedOnce = true;
        const FLinearColor& P0 = Pixels[0];
        const FLinearColor& PCenter = Pixels[(Size.Y / 2) * Size.X + Size.X / 2];
        UE_LOG(LogCarla, Warning,
          TEXT("[WideAngleLens HDR diag] ImageSize=(%d,%d) ReadbackSize=(%d,%d) Pixels.Num()=%d ")
          TEXT("Pixel[0]=(%f,%f,%f,%f) PixelCenter=(%f,%f,%f,%f)"),
          GetImageWidth(), GetImageHeight(), Size.X, Size.Y, Pixels.Num(),
          P0.R, P0.G, P0.B, P0.A,
          PCenter.R, PCenter.G, PCenter.B, PCenter.A);
      }
      SendHDRDataToClient(Pixels, Size, FrameIndex);
      return true;
    });
  }
  else
  {
    // 默认 uint8 模式：沿用原有 FColor 路径
    FPixelReader::SendPixelsInRenderThread<ASceneCaptureCamera_WideAngleLens, FColor>(*this);
  }
}

// ========== Bayer RAW / uint16 / float32 HDR 数据发送 ==========
// 逻辑与 ASceneCaptureCamera::SendHDRDataToClient 完全一致，仅数据来源换成
// distort 后的最终纹理（equidistant/等距投影已经在 compute shader 里完成）。
void ASceneCaptureCamera_WideAngleLens::SendHDRDataToClient(
  const TArrayView<const FLinearColor>& Pixels,
  FIntPoint Size,
  uint64 FrameIndex)
{
  TRACE_CPUPROFILER_EVENT_SCOPE(ASceneCaptureCamera_WideAngleLens::SendHDRDataToClient);

  const int32 Width = Size.X;
  const int32 Height = Size.Y;
  const int32 TotalPixels = Width * Height;

  if (RawType == TEXT("bayer_rggb"))
  {
    // 单通道 Bayer RGGB (0-65535)
    TArray<uint16> BayerData;
    ImageUtil::ConvertRGBToBayerRGGB(Pixels, Width, Height, BayerData);

    auto DataStream = GetDataStream(*this);
    DataStream.SerializeAndSend(
        *this,
        FrameIndex,
        DataStream.PopBufferFromPool(),
        reinterpret_cast<const uint8*>(BayerData.GetData()),
        BayerData.Num() * sizeof(uint16),
        Width,
        Height,
        carla::sensor::data::EPixelFormat::BAYER_RGGB_U16
    );
  }
  else if (RawType == TEXT("uint16"))
  {
    // 16bit RGB (0-65535)
    TArray<uint8> OutputData;
    OutputData.SetNumUninitialized(TotalPixels * 3 * sizeof(uint16));

    uint16* OutPtr = reinterpret_cast<uint16*>(OutputData.GetData());

    for (int32 i = 0; i < TotalPixels; ++i)
    {
      const FLinearColor& Pixel = Pixels[i];
      OutPtr[i * 3 + 0] = static_cast<uint16>(FMath::Clamp(Pixel.R, 0.0f, 1.0f) * 65535.0f);
      OutPtr[i * 3 + 1] = static_cast<uint16>(FMath::Clamp(Pixel.G, 0.0f, 1.0f) * 65535.0f);
      OutPtr[i * 3 + 2] = static_cast<uint16>(FMath::Clamp(Pixel.B, 0.0f, 1.0f) * 65535.0f);
    }

    auto DataStream = GetDataStream(*this);
    DataStream.SerializeAndSend(
        *this,
        FrameIndex,
        DataStream.PopBufferFromPool(),
        reinterpret_cast<const uint8*>(OutputData.GetData()),
        OutputData.Num(),
        Width,
        Height,
        carla::sensor::data::EPixelFormat::RGB_U16
    );
  }
  else if (RawType == TEXT("float32"))
  {
    // float32 RGB (HDR, 0.0-inf)
    TArray<uint8> OutputData;
    OutputData.SetNumUninitialized(TotalPixels * 3 * sizeof(float));

    float* OutPtr = reinterpret_cast<float*>(OutputData.GetData());

    for (int32 i = 0; i < TotalPixels; ++i)
    {
      const FLinearColor& Pixel = Pixels[i];
      OutPtr[i * 3 + 0] = Pixel.R;
      OutPtr[i * 3 + 1] = Pixel.G;
      OutPtr[i * 3 + 2] = Pixel.B;
    }

    auto DataStream = GetDataStream(*this);
    auto Buffer = carla::sensor::s11n::ImageSerializer::Serialize(
        *this,
        FrameIndex,
        DataStream.PopBufferFromPool(),
        reinterpret_cast<const uint8*>(OutputData.GetData()),
        OutputData.Num(),
        Width,
        Height,
        carla::sensor::data::EPixelFormat::RGB_F32
    );

    auto View = carla::BufferView::CreateFrom(std::move(Buffer));
    DataStream.Send(*this, View);
  }
  else
  {
    UE_LOG(LogCarla, Warning, TEXT("Invalid RawType '%s', falling back to uint8"), *RawType);

    TArray<FColor> FColorPixels;
    FColorPixels.SetNumUninitialized(TotalPixels);

    for (int32 i = 0; i < TotalPixels; ++i)
    {
      FColorPixels[i] = Pixels[i].ToFColor(true);
    }

    SendDataToClient(*this, TArrayView<const FColor>(FColorPixels), FrameIndex);
  }
}
