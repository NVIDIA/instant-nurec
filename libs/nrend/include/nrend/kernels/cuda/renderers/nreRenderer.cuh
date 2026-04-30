// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#pragma once

#include <nrend/kernels/cuda/common/rayPayload.cuh>

__global__ void render(nrend::RenderParameters params,
                       const tcnn::vec3* __restrict__ wordlRayOriginPtr,
                       const tcnn::vec3* __restrict__ worldRayDirectionPtr,
                       const nrend::TTimestamp* __restrict__ worldRayTimestampCudaPtr,
                       const tcnn::ivec2* __restrict__ sensorsIdsPtr,
                       uint32_t* __restrict__ instanceIdPtr,
                       float* __restrict__ worldHitDistancePtr,
                       tcnn::vec3* __restrict__ worldHitNormalPtr,
                       tcnn::vec4* __restrict__ radianceDensityPtr,
                       const tcnn::ivec2* trackInstancesIds,
                       const nrend::TTrackInstancePose* __restrict__ trackInstancesStartPoseCudaPtr,
                       const nrend::TTrackInstancePose* __restrict__ trackInstancesEndPoseCudaPtr,
                       const uint64_t* __restrict__ parameterMemoryHandles) {

    auto ray = initializeRay<RayPayload<TNREModel::TRayPayload::FeatDim>>(
        params, wordlRayOriginPtr, worldRayDirectionPtr, worldRayTimestampCudaPtr, instanceIdPtr, worldHitDistancePtr);

    TNREModel model;
    model.march(params, ray, {parameterMemoryHandles}, trackInstancesIds, trackInstancesStartPoseCudaPtr, trackInstancesEndPoseCudaPtr, sensorsIdsPtr);

    if (ray.isValid()) {
        finalizeRay<SRGBModel, SRGBOutput, false>(
            ray, params, wordlRayOriginPtr, instanceIdPtr, worldHitDistancePtr, worldHitNormalPtr, radianceDensityPtr);
    }
}
