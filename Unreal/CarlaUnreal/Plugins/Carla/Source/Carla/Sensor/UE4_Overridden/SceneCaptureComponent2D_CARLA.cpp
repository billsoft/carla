// Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include "SceneCaptureComponent2D_CARLA.h"
#include "Carla.h"



USceneCaptureComponent2D_CARLA::USceneCaptureComponent2D_CARLA(const FObjectInitializer& ObjectInitializer) :
    Super(ObjectInitializer)
{
}

const AActor* USceneCaptureComponent2D_CARLA::GetViewOwner() const
{
  return ViewActor;
}

void USceneCaptureComponent2D_CARLA::ReleaseViewStates()
{
  check(IsInGameThread());
  // Sharers first: each drops its reference to our view state (and removes
  // itself from the list already moved out here).
  auto Sharers = MoveTemp(OriginSharers);
  for (auto &Sharer : Sharers)
  {
    if (Sharer.IsValid())
      Sharer->ReleaseViewStates();
  }
  if (auto *Anchor = OriginAnchor.Get())
    Anchor->OriginSharers.Remove(this);
  OriginAnchor.Reset();
  ViewStates.Empty();
}

bool USceneCaptureComponent2D_CARLA::ShareViewOrigin(USceneCaptureComponent2D_CARLA* Anchor)
{
  check(IsInGameThread());
  // ShareOrigin cannot nest, cannot be set twice and must precede Allocate.
  if (Anchor == nullptr || Anchor == this || OriginAnchor.IsValid() || OriginSharers.Num() != 0 ||
      ViewStates.Num() != 0 || Anchor->OriginAnchor.IsValid() ||
      !bAlwaysPersistRenderingState || !Anchor->bAlwaysPersistRenderingState)
  {
    UE_LOG(LogCarla, Warning, TEXT("%s: cannot share the view origin of %s, keeping a separate view state"),
        *GetName(), Anchor != nullptr ? *Anchor->GetName() : TEXT("null"));
    return false;
  }
  // The anchor's state must exist before a sharer allocates
  // (FSceneViewStateReference::AllocateInternal checks it).
  if (Anchor->GetViewState(0) == nullptr)
    return false;
  ViewStates.Add(new FSceneViewStateReference());
  ViewStates[0].ShareOrigin(&Anchor->ViewStates[0]);
  Anchor->OriginSharers.Add(this);
  OriginAnchor = Anchor;
  return GetViewState(0) != nullptr;
}

void USceneCaptureComponent2D_CARLA::OnUnregister()
{
  ReleaseViewStates();
  Super::OnUnregister();
}
