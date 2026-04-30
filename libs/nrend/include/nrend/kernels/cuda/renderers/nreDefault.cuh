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
#include <nrend/kernels/cuda/models/nreOccupancyGridMarching.cuh>

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

    ray.tMinMax.x += M_STEP_SIZE * rnd(ray.rndSeed);

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
        const float dt = calc_dt(ray.tMinMax.x, M_CONE_ANGLE, M_STEP_SIZE, 1e10f);

        tcnn::vec<6 + N_EXTRA_DIMS> nerf_in;
        nerf_in.slice<0, 3>() = params.objectAABB.relative_pos(pos);
#if N_EXTRA_DIMS > 0
        nerf_in.slice<3, N_EXTRA_DIMS>() = 0.f;
#endif
        nerf_in.slice<3 + N_EXTRA_DIMS, 3>() = (ray.direction + 1.0f) * 0.5f;

        const tcnn::vec4 nerf_out = eval_nerf(nerf_in, nerfParamsPtr);

        // All threads in the warp must execute the above MLPs for coherence reasons.
        // Starting from here, it's fine to skip computation.
        if (!ray.isAlive()) {
            continue;
        }

        const float trn    = __expf(-__expf(nerf_out.w) * dt * densityScale);
        const float weight = (1.0f - trn) * ray.transmittance;

        ray.features.vec += weight * nerf_out.xyz();
        ray.hitT += weight * ray.tMinMax.x;
        ray.transmittance *= trn;

        ray.tMinMax.x += dt;

        if (ray.transmittance < params.defaultMinTransmittance) {
            ray.kill();
        }
    }

    // mark the ray as front hit if the traversed volume is sufficiently opaque
    if (ray.isValid() && (ray.transmittance < params.hitTransmittance)) {
        ray.hitFront();
    }

    // process the ray background hits (need all threads to be active for potential coopvec operations)
    if (ApplyBackgroundModel) {
        const bool needBackEvaluation = ray.isValid() && !ray.hasBackHit();
        if (__any_sync(0xFFFFFFFF, needBackEvaluation)) {
            const tcnn::vec3 backgroundRadiance = eval_background((ray.direction + 1.0f) * 0.5f, backParamsPtr);
            if (needBackEvaluation) {
                ray.features.vec += ray.transmittance * backgroundRadiance;
                ray.transmittance = 0.0f; ///< background has been hit, set the transmittance to 0
            }
        }
    }

    if (ray.isValid()) {
        finalizeRay<SRGBModel, SRGBOutput, false>(
            ray, params, wordlRayOriginPtr, instanceIdPtr, worldHitDistancePtr, worldHitNormalPtr, radianceDensityPtr);
    }
}
