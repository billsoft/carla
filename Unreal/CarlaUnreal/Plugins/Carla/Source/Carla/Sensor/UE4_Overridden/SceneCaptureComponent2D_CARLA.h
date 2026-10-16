// Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include <util/ue-header-guard-begin.h>
#include "CoreMinimal.h"
#include "Components/SceneCaptureComponent2D.h"
#include <util/ue-header-guard-end.h>

#include "SceneCaptureComponent2D_CARLA.generated.h"



UCLASS(hidecategories=(Collision, Object, Physics, SceneComponent, Mobility))
class CARLA_API USceneCaptureComponent2D_CARLA : public USceneCaptureComponent2D
{
	GENERATED_BODY()
public:

  USceneCaptureComponent2D_CARLA(const FObjectInitializer& = FObjectInitializer::Get());

  UPROPERTY()
  TObjectPtr<AActor> ViewActor;

	virtual const AActor* GetViewOwner() const override;

  /// Free the persistent per-view render state (Lumen scene, TSR/TAA
  /// history, shadow caches...) right away. USceneCaptureComponent only
  /// releases ViewStates when the UObject is garbage collected, which in
  /// -game can be a minute after the sensor is destroyed; with
  /// bAlwaysPersistRenderingState that state is hundreds of MB per view.
  /// Game thread only; call after the render thread is done with the view
  /// (FlushRenderingCommands). An origin anchor releases its sharers first
  /// (see ShareViewOrigin).
  void ReleaseViewStates();

  /// Make this capture's view state share Anchor's view origin, exactly like
  /// the faces of an engine cube capture (USceneCaptureComponent::GetViewState,
  /// FSceneViewStateReference::ShareOrigin): the renderer then uses Anchor's
  /// per-view Lumen scene (surface cache atlases, ~400 MB at the default
  /// 4096^2 atlas) and global distance field instead of keeping one per
  /// capture. Only valid for captures at the same location (the faces of one
  /// fisheye camera). Allocates both view states now, so call it after
  /// bAlwaysPersistRenderingState is set and before the first capture.
  /// Returns false (and keeps a separate view state) when sharing is not
  /// possible.
  bool ShareViewOrigin(USceneCaptureComponent2D_CARLA* Anchor);

  /// True when this capture shares another capture's view origin (it can
  /// then not be an anchor itself: ShareOrigin does not nest).
  bool IsViewOriginSharer() const { return OriginAnchor.IsValid(); }

  bool HasOriginSharers() const { return OriginSharers.Num() != 0; }

  bool HasViewState() const { return ViewStates.Num() != 0; }

  virtual void OnUnregister() override;

private:

  /// ~FSceneViewStateReference asserts that every sharer is destroyed before
  /// its anchor, so the anchor tracks its sharers and releases them first,
  /// whatever order the components unregister in.
  TArray<TWeakObjectPtr<USceneCaptureComponent2D_CARLA>> OriginSharers;

  TWeakObjectPtr<USceneCaptureComponent2D_CARLA> OriginAnchor;
};
