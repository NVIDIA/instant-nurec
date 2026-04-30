// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
module cameras;

namespace cameras
{

[BackwardDifferentiable]
bool perspectiveGaussianCovarianceProjection(
    no_diff in float2 resolution,
    no_diff in float2 nearFarClipDistances,
    no_diff in float2 focalLength,
    no_diff in float2 principalPoint,
    in float3 worldCenter,
    in float3x3 worldCovariance,
    no_diff in float4x3 viewT,
    out float2 projectedCenter,
    out float2x2 projectedCovariance
)
{
    const float3 center = mul(float4(worldCenter, 1.0), viewT);
    if ((nearFarClipDistances.x > center.z) || (nearFarClipDistances.y < center.z)) {
        return false;
    }

    // FIXME : bounds computation should take the principal point into account
    const float2 halfResolution = 0.5f * resolution;
    const float2 bounds = 1.3 * focalLength;
    const float invCenterZ = 1.0f / center.z;
    const float2 clampedCenterXY = min(bounds, max(-bounds, center.xy * invCenterZ)) * center.z;

    // Jacobian transposed (J^T)
    const float3x3 Jt = float3x3(focalLength.x * invCenterZ, 0.0, -(focalLength.x * clampedCenterXY.x) * invCenterZ,
                                 0.0, focalLength.y * invCenterZ, -(focalLength.y * clampedCenterXY.y) * invCenterZ,
                                 0.0, 0.0, 0.0);
    
    // World to sensor transposed (W^T) time Jacobian transposed (J^T)
    const float3x3 WtJt = mul(float3x3(viewT), Jt);

    // Σ' = JWΣW^TJ^T
    projectedCovariance = float2x2(mul(transpose(WtJt), mul(worldCovariance, WtJt)));

    const float eps = 1e-07f;
    projectedCenter = (center.xy * focalLength ) / (center.z + eps) + principalPoint;

    return true;
}

}