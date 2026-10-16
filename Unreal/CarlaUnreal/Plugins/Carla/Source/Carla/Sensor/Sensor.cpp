// Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include <Carla/Sensor/Sensor.h>
#include <Carla.h>
#include <Carla/Sensor/SensorManager.h>
#include <Carla/Actor/ActorDescription.h>
#include <Carla/Actor/ActorBlueprintFunctionLibrary.h>
#include <Carla/Game/CarlaStatics.h>
#include <Carla/Sensor/ImageUtil.h>
#include <Carla/Sensor/ShaderBasedSensor.h>

#include <util/ue-header-guard-begin.h>
#include <Engine/CollisionProfile.h>
#include <util/ue-header-guard-end.h>

ASensor::ASensor(const FObjectInitializer &ObjectInitializer)
  : Super(ObjectInitializer)
{
  auto *Mesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("CamMesh"));
  Mesh->SetCollisionProfileName(UCollisionProfile::NoCollision_ProfileName);
  Mesh->bHiddenInGame = true;
  Mesh->CastShadow = false;
  RootComponent = Mesh;
}

void ASensor::BeginPlay()
{
  Super::BeginPlay();
  UCarlaEpisode* CurrentEpisode = UCarlaStatics::GetCurrentEpisode(GetWorld());
  FSensorManager& SensorManager = CurrentEpisode->GetSensorManager();
  SensorManager.RegisterSensor(this);
}

void ASensor::Set(const FActorDescription &Description)
{
  // make a copy
  SensorDescription = Description;

  // set the tick interval of the sensor
  if (Description.Variations.Contains("sensor_tick"))
  {
    SetActorTickInterval(
        UActorBlueprintFunctionLibrary::ActorAttributeToFloat(Description.Variations["sensor_tick"],
        0.0f));
  }
  if (Description.Variations.Contains("capture_stride"))
  {
    CaptureStride = FMath::Max(1, UActorBlueprintFunctionLibrary::ActorAttributeToInt(
        Description.Variations["capture_stride"], 1));
  }
  if (Description.Variations.Contains("output_stride"))
  {
    OutputStride = FMath::Max(1, UActorBlueprintFunctionLibrary::ActorAttributeToInt(
        Description.Variations["output_stride"], 1));
  }
}

std::optional<FActorAttribute> ASensor::GetAttribute(const FString Name)
{
  if (SensorDescription.Variations.Contains(Name))
  {
    return SensorDescription.Variations[Name];
  }
  else
    return {};
}

void ASensor::Tick(const float DeltaTime)
{
  TRACE_CPUPROFILER_EVENT_SCOPE(ASensor::Tick);
  Super::Tick(DeltaTime);
  if (bClientsListening)
  {
    if(!Stream.AreClientsListening())
    {
      OnLastClientDisconnected();
      bClientsListening = false;
    }
  }
  else
  {
    if(Stream.AreClientsListening())
    {
      OnFirstClientConnected();
      bClientsListening = true;
    }
  }
  
  // capture_stride (2026-10-16)：同步模式抽帧采集时，客户端只要每 N 步里的一步数据，其余步相机照样完整渲染
  // (81 上占 tick 时间的大半)。sensor_tick 走 UE 的 tick 间隔 + 超时补偿，浮点下偶尔 N-1/N+1 步才触发，
  // 图像就和客户端在该步取的世界状态对不上；这里按引擎帧号 (= 客户端 world.tick() 的返回值、image.frame)
  // 确定性判断，客户端只需让保存帧的帧号是 N 的倍数
  ReadyToTick = CaptureStride <= 1 || FCarlaEngine::GetFrameCounter() % static_cast<uint64_t>(CaptureStride) == 0;
  ReadyToTickDeltaSeconds = DeltaTime;
  PrePhysTick(DeltaTime);
}

void ASensor::SetSeed(const int32 InSeed)
{
  check(RandomEngine != nullptr);
  Seed = InSeed;
  RandomEngine->Seed(InSeed);
}

void ASensor::PostActorCreated()
{
  Super::PostActorCreated();

#if WITH_EDITOR
  auto *StaticMeshComponent = Cast<UStaticMeshComponent>(RootComponent);
  if (StaticMeshComponent && !IsRunningCommandlet() && !StaticMeshComponent->GetStaticMesh())
  {
    UStaticMesh *CamMesh = LoadObject<UStaticMesh>(
        NULL,
        TEXT("/Engine/EditorMeshes/MatineeCam_SM.MatineeCam_SM"),
        NULL,
        LOAD_None,
        NULL);
    StaticMeshComponent->SetStaticMesh(CamMesh);
  }
#endif // WITH_EDITOR
}

void ASensor::EndPlay(EEndPlayReason::Type EndPlayReason)
{
  Super::EndPlay(EndPlayReason);

  // close all sessions associated to the sensor stream
  auto *GameInstance = UCarlaStatics::GetGameInstance(GetEpisode().GetWorld());
  auto &StreamingServer = GameInstance->GetServer().GetStreamingServer();
  auto StreamId = carla::streaming::detail::token_type(Stream.GetToken()).get_stream_id();
  StreamingServer.CloseStream(StreamId);

  UCarlaEpisode* CurrentEpisode = UCarlaStatics::GetCurrentEpisode(GetWorld());
  if(CurrentEpisode)
  {
    FSensorManager& SensorManager = CurrentEpisode->GetSensorManager();
    SensorManager.DeRegisterSensor(this);
  }
}

void ASensor::PostPhysTickInternal(UWorld *World, ELevelTick TickType, [[maybe_unused]] float DeltaSeconds)
{
  TRACE_CPUPROFILER_EVENT_SCOPE(ASensor::PostPhysTickInternal);
  if(ReadyToTick)
  {
    // DeltaSeconds is only the current physics substep's delta; this sensor's
    // own tick interval may span several substeps, so the correct elapsed
    // time is ReadyToTickDeltaSeconds (captured in Tick(), see Sensor.h).
    PostPhysTick(World, TickType, ReadyToTickDeltaSeconds);
    ReadyToTick = false;
  }
}
