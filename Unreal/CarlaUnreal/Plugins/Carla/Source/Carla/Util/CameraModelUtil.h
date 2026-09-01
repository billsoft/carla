// Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include <util/ue-header-guard-begin.h>
#include "CoreMinimal.h"
#include "RHIDefinitions.h"
#include "Containers/ArrayView.h"
#include <util/ue-header-guard-end.h>

#include "CameraModelUtil.generated.h"



class FRDGBuilder;
class FRDGTexture;
class FRHISamplerState;
class UTextureRenderTarget2D;



UENUM(BlueprintType)
enum class ECameraModel : uint8
{
    Perspective,
    Stereographic,
    Equidistant,
    Equisolid,
    Orthographic,
    KannalaBrandt,
    MaxEnum UMETA(Hidden),
    Default = Perspective UMETA(Hidden)
};



namespace CameraModelUtil
{
    struct FDistortCubemapToImageOptions
    {
        TArrayView<const float> KannalaBrandtCoefficients;
        float YFOVAngle;
        float YFocalLength;
        // Horizontal focal length. Defaults to YFocalLength (isotropic, matches the
        // behavior before this field existed) — set independently to support a
        // physical lens with different horizontal/vertical FOV rather than one
        // derived from the other via the image aspect ratio.
        float XFocalLength;
        // Principal point offset from the exact geometric image center, in pixels.
        // (0,0) (the default) reproduces the pre-existing behavior of always
        // sampling around the image center; a real lens's calibrated optical
        // center rarely lands exactly there.
        FVector2D PrincipalPointOffset;
        float LongitudeOffset;
        float FOVFadeSize;
        // tan() of the front cube face's actual captured half-FOV. 1.0 (i.e.
        // tan(45 deg)) reproduces the original fixed-90-degree face; a caller
        // that narrows the front face's CustomProjectionMatrix (see
        // ASceneCaptureSensor_WideAngleLens::UpdateFrontFaceProjection) must
        // pass the matching tan(HalfFOV) here so SampleCubemap() samples the
        // narrowed face's UV range correctly. Must always be set explicitly —
        // this struct is aggregate-initialized (`= { }`) elsewhere, which
        // zero-inits this field rather than defaulting it to 1.0.
        float FrontFaceTanHalfFOV;
        ECameraModel CameraModel;
        bool bRenderEquirectangular : 1;
        bool bFOVMaskEnable : 1;
        bool bRenderPerspective : 1;
    };



    float ComputeAngle(
        ECameraModel CameraModel,
        float Distance,
        TArrayView<const float> Coefficients);

    float ComputeDistance(
        ECameraModel CameraModel,
        float Angle,
        int32 ImageHeight,
        TArrayView<const float> Coefficients);



    namespace KannalaBrandt
    {
        float ComputeCameraPolynomial(
            float Theta,
            TArrayView<const float> Coefficients);

        float ComputeCameraPolynomialDerivative(
            float Theta,
            TArrayView<const float> Coefficients);
    }



	void DistortCubemapToImage(
        FRDGBuilder& GraphBuilder,
        FRDGTexture* Destination,
        FRDGTexture** CubeTextures, // CubeTextures[6]
        FRHISamplerState* Sampler,
        const FDistortCubemapToImageOptions& Options);

    void DistortCubemapToImage(
        FRDGBuilder& GraphBuilder,
        UTextureRenderTarget2D* Destination,
        UTextureRenderTarget2D** CubeRenderTargets, // CubeRenderTargets[6]
        FRHISamplerState* Sampler,
        const FDistortCubemapToImageOptions& Options);

    FRHISamplerState* GetSampler(ESamplerFilter Filter);

} // CameraModelUtil
