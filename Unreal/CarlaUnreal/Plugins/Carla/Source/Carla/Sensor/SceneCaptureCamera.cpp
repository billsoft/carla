// Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include "Carla/Sensor/SceneCaptureCamera.h"
#include "Carla.h"
#include "Carla/Game/CarlaEngine.h"

// ========== 新增头文件 ==========
#include <carla/sensor/s11n/ImageSerializer.h>  // HDR 序列化支持
#include <carla/sensor/data/Image.h>        // 像素格式枚举
#include <carla/Buffer.h>                   // Buffer

#include <util/ue-header-guard-begin.h>
#include "Actor/ActorBlueprintFunctionLibrary.h"
#include <util/ue-header-guard-end.h>

FActorDefinition ASceneCaptureCamera::GetSensorDefinition()
{
    constexpr bool bEnableModifyingPostProcessEffects = true;
    return UActorBlueprintFunctionLibrary::MakeCameraDefinition(
        TEXT("rgb"),
        bEnableModifyingPostProcessEffects);
}

ASceneCaptureCamera::ASceneCaptureCamera(const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer)
{
    AddPostProcessingMaterial(
        TEXT("Material'/Carla/PostProcessingMaterials/PhysicLensDistortion.PhysicLensDistortion'"));
}

void ASceneCaptureCamera::BeginPlay()
{
  Super::BeginPlay();
}

void ASceneCaptureCamera::OnFirstClientConnected()
{
}

void ASceneCaptureCamera::OnLastClientDisconnected()
{
}

void ASceneCaptureCamera::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
  Super::EndPlay(EndPlayReason);
}

void ASceneCaptureCamera::PostPhysTick(UWorld *World, ELevelTick TickType, float DeltaSeconds)
{
  TRACE_CPUPROFILER_EVENT_SCOPE(ASceneCaptureCamera::PostPhysTick);
  Super::PostPhysTick(World, TickType, DeltaSeconds);

  if (!AreClientsListening())
      return;

  auto FrameIndex = FCarlaEngine::GetFrameCounter();

  // ========== 新增: 根据 RawType 选择数据捕获方式 ==========

  // 检查用户设置的输出格式
  if (RawType == TEXT("float32") || RawType == TEXT("uint16") || RawType == TEXT("bayer_rggb"))
  {
    // HDR 模式: 使用 FLinearColor (float32 × 4)
    ImageUtil::ReadSensorImageDataAsyncFLinearColor(*this, [this, FrameIndex](
      TArrayView<const FLinearColor> Pixels,
      FIntPoint Size) -> bool
    {
      SendHDRDataToClient(Pixels, FrameIndex);
      return true;
    });
  }
  else
  {
    // 默认 uint8 模式: 使用 FColor (uint8 × 4)
    ImageUtil::ReadSensorImageDataAsyncFColor(*this, [this, FrameIndex](
      TArrayView<const FColor> Pixels,
      FIntPoint Size) -> bool
    {
      SendDataToClient(*this, Pixels, FrameIndex);
      return true;
    });
  }
}

#ifdef CARLA_HAS_GBUFFER_API
void ASceneCaptureCamera::SendGBufferTextures(FGBufferRequest& GBuffer)
{
    SendGBufferTexturesInternal(*this, GBuffer);
}
#endif

// ========== 新增: RGB → Bayer RGGB 转换 ==========
/**
 * 将 RGB 图像转换为单通道 Bayer RGGB 模式
 *
 * Bayer Pattern (RGGB):
 *   R G R G ...
 *   G B G B ...
 *   R G R G ...
 *   G B G B ...
 *
 * @param Pixels RGB 像素数组 (FLinearColor)
 * @param Width 图像宽度
 * @param Height 图像高度
 * @param OutBayerData 输出 Bayer 数据 (uint16, 单通道)
 */
static void ConvertRGBToBayerRGGB(
  const TArrayView<const FLinearColor>& Pixels,
  int32 Width,
  int32 Height,
  TArray<uint16>& OutBayerData)
{
  OutBayerData.SetNumUninitialized(Width * Height);

  for (int32 y = 0; y < Height; ++y)
  {
    for (int32 x = 0; x < Width; ++x)
    {
      const int32 idx = y * Width + x;
      const FLinearColor& Pixel = Pixels[idx];

      uint16 BayerValue = 0;

      // RGGB 模式判断
      if (y % 2 == 0)  // 偶数行
      {
        if (x % 2 == 0)
          BayerValue = static_cast<uint16>(FMath::Clamp(Pixel.R, 0.0f, 1.0f) * 65535.0f);  // R
        else
          BayerValue = static_cast<uint16>(FMath::Clamp(Pixel.G, 0.0f, 1.0f) * 65535.0f);  // G
      }
      else  // 奇数行
      {
        if (x % 2 == 0)
          BayerValue = static_cast<uint16>(FMath::Clamp(Pixel.G, 0.0f, 1.0f) * 65535.0f);  // G
        else
          BayerValue = static_cast<uint16>(FMath::Clamp(Pixel.B, 0.0f, 1.0f) * 65535.0f);  // B
      }

      OutBayerData[idx] = BayerValue;
    }
  }
}

void ASceneCaptureCamera::SendHDRDataToClient(
  const TArrayView<const FLinearColor>& Pixels,
  uint64 FrameIndex)
{
  TRACE_CPUPROFILER_EVENT_SCOPE(ASceneCaptureCamera::SendHDRDataToClient);

  // 获取图像尺寸 (GetImageSize 不存在，改用 GetImageWidth/Height)
  const int32 Width = GetImageWidth();
  const int32 Height = GetImageHeight();
  const int32 TotalPixels = Width * Height;

  // ========== 根据 RawType 选择输出格式 ==========

  if (RawType == TEXT("bayer_rggb"))
  {
    // 模式 3: 转换为单通道 Bayer RGGB (0-65535)
    TArray<uint16> BayerData;
    ConvertRGBToBayerRGGB(Pixels, Width, Height, BayerData);

    // 发送数据
    auto DataStream = GetDataStream(*this);
    DataStream.SerializeAndSend(
        *this,
        FrameIndex,
        DataStream.PopBufferFromPool(),
        reinterpret_cast<const uint8*>(BayerData.GetData()),
        BayerData.Num() * sizeof(uint16),  // 单通道 × 2 bytes
        Width,
        Height,
        carla::sensor::data::EPixelFormat::BAYER_RGGB_U16
    );
  }
  else if (RawType == TEXT("uint16"))
  {
    // 模式 1: 转换为 16bit RGB (0-65535)
    TArray<uint8> OutputData;
    OutputData.SetNumUninitialized(TotalPixels * 3 * sizeof(uint16));  // RGB × 2 bytes

    uint16* OutPtr = reinterpret_cast<uint16*>(OutputData.GetData());

    for (int32 i = 0; i < TotalPixels; ++i)
    {
      const FLinearColor& Pixel = Pixels[i];

      // FLinearColor [0.0, inf] → uint16 [0, 65535]
      // Clamp 到 [0, 1] 然后缩放
      OutPtr[i * 3 + 0] = static_cast<uint16>(FMath::Clamp(Pixel.R, 0.0f, 1.0f) * 65535.0f);  // R
      OutPtr[i * 3 + 1] = static_cast<uint16>(FMath::Clamp(Pixel.G, 0.0f, 1.0f) * 65535.0f);  // G
      OutPtr[i * 3 + 2] = static_cast<uint16>(FMath::Clamp(Pixel.B, 0.0f, 1.0f) * 65535.0f);  // B
    }

    // 发送数据 (rename Stream -> DataStream to avoid shadowing)
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
    // 模式 2: 保持 float32 RGB (HDR, 0.0-inf)
    TArray<uint8> OutputData;
    OutputData.SetNumUninitialized(TotalPixels * 3 * sizeof(float));  // RGB × 4 bytes

    float* OutPtr = reinterpret_cast<float*>(OutputData.GetData());

    for (int32 i = 0; i < TotalPixels; ++i)
    {
      const FLinearColor& Pixel = Pixels[i];

      OutPtr[i * 3 + 0] = Pixel.R;  // R (保持 HDR 范围)
      OutPtr[i * 3 + 1] = Pixel.G;  // G
      OutPtr[i * 3 + 2] = Pixel.B;  // B
    }

    // 发送数据
    auto DataStream = GetDataStream(*this);
    auto Buffer = carla::sensor::s11n::ImageSerializer::Serialize(
        *this,
        FrameIndex,
        DataStream.PopBufferFromPool(),
        reinterpret_cast<const uint8*>(OutputData.GetData()),
        OutputData.Num(),
        Width,
        Height,
        carla::sensor::data::EPixelFormat::RGB_F32  // 新增的像素格式
    );

    // Create BufferView and Send
    auto View = carla::BufferView::CreateFrom(std::move(Buffer));
    DataStream.Send(*this, View);
  }
  else
  {
    UE_LOG(LogCarla, Warning, TEXT("Invalid RawType '%s', falling back to uint8"), *RawType);

    // 降级为 uint8 (不应该到这里，但作为安全措施)
    TArray<FColor> FColorPixels;
    FColorPixels.SetNumUninitialized(TotalPixels);

    for (int32 i = 0; i < TotalPixels; ++i)
    {
      FColorPixels[i] = Pixels[i].ToFColor(true);  // sRGB = true
    }

    // Cast TArray to TArrayView
    SendDataToClient(*this, TArrayView<const FColor>(FColorPixels), FrameIndex);
  }
}
