// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include <nrend/kernels/cuda/common/rayPayload.cuh>
#include <nrend/kernels/cuda/models/ngpOccupancyGridMarching.cuh>

__global__ void render(nrend::RenderParameters params,
                       const tcnn::vec3* __restrict__ wordlRayOriginPtr,
                       const tcnn::vec3* __restrict__ worldRayDirectionPtr,
                       const nrend::TTimestamp* __restrict__ worldRayTimestampCudaPtr,
                       uint32_t* __restrict__ instanceIdPtr,
                       float* __restrict__ worldHitDistancePtr,
                       tcnn::vec3* __restrict__ worldHitNormalPtr,
                       tcnn::vec4* __restrict__ radianceDensityPtr,
                       const uint8_t* __restrict__ densityGridPtr,
                       const tcnn::network_precision_t* __restrict__ nerfParamsPtr,
                       const tcnn::network_precision_t* __restrict__ backParamsPtr) {
    constexpr int RadianceDim = 3;

    auto ray =
        initializeRay<RayPayload<RadianceDim>>(params, wordlRayOriginPtr, worldRayDirectionPtr, worldRayTimestampCudaPtr, instanceIdPtr, worldHitDistancePtr);

    ray.tMinMax.x = advance_n_steps(ray.tMinMax.x, rnd(ray.rndSeed));

    while (true) {
        tcnn::vec3 pos = ray.origin;

        if (ray.isAlive()) {
            ray.tMinMax.x = advanceToNextOccupiedVoxel(
                ray.origin, ray.direction, densityGridPtr, 0, DensityGridMaxCascade, ray.tMinMax);

            if (ray.tMinMax.x >= ray.tMinMax.y) {
                ray.kill();
            } else {
                pos += ray.direction * ray.tMinMax.x;
            }
        }

        if (__all_sync(0xFFFFFFFF, !ray.isAlive())) {
            break;
        }

        // Evaluate NeRF model
        const float dt = calc_dt(ray.tMinMax.x);

        tcnn::vec<6 + N_EXTRA_DIMS> nerf_in;
        nerf_in.slice<0, 3>() = params.objectAABB.relative_pos(pos);
        nerf_in.slice<3, 3>() = (ray.direction + 1.0f) * 0.5f;
#if N_EXTRA_DIMS > 0
        nerf_in.slice<6, N_EXTRA_DIMS>() = 0.f;
#endif

        const tcnn::vec4 nerf_out = eval_nerf(nerf_in, nerfParamsPtr);

        // All threads in the warp must execute the above MLPs for coherence reasons.
        // Starting from here, it's fine to skip computation.
        if (!ray.isAlive()) {
            continue;
        }

        const float trn    = __expf(-__expf(nerf_out.w) * dt * aabbRelativeScale);
        const float weight = (1.0f - trn) * ray.transmittance;

        ray.features.vec += weight * (ApplyRGBSigmoid ? tcnn::vec3{tcnn::logistic(nerf_out[0]), tcnn::logistic(nerf_out[1]),
                                                                   tcnn::logistic(nerf_out[2])}
                                                      : nerf_out.xyz());
        ray.hitT += weight * ray.tMinMax.x;
        ray.transmittance *= trn;

        ray.tMinMax.x += dt;

        if (ray.transmittance < params.defaultMinTransmittance) {
            ray.kill();
        }
    }

    if (ray.isValid()) {
        // mark the ray as front hit if the traversed volume is sufficiently opaque
        if (ray.transmittance < params.hitTransmittance) {
            ray.hitFront();
        }

        // apply the background model only if there is currently no background hit
        // FIXME :
        // - take into account the background alpha
        // - how to comp several volumes with different background models ?
        if (ApplyBackgroundModel && !ray.hasBackHit()) {
            const tcnn::vec3 back_out = eval_background((ray.direction + 1.0f) * 0.5f, backParamsPtr);
            ray.features.vec += ray.transmittance * back_out;
            ray.transmittance = 0.0f; ///< background has been hit, set the transmittance to 0
        }

        finalizeRay<SRGBModel, SRGBOutput, false>(
            ray, params, wordlRayOriginPtr, instanceIdPtr, worldHitDistancePtr, worldHitNormalPtr, radianceDensityPtr);
    }
}
