// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#pragma once

#include <nrend/utils/nreVec.h>

struct NREInstancesExtentDefaultParams {
    static constexpr int NumInstances = 1;
    static const float instanceExpandedExtents[3 * NumInstances];
};
const float NREInstancesExtentDefaultParams::instanceExpandedExtents[3 * NumInstances] = {1.0f, 1.0f, 1.0f};

template <typename Params = NREInstancesExtentDefaultParams>
struct NREInstancesExtent final : Params {

    static inline __device__ const tcnn::vec3 contract_position(const tcnn::vec3& position, uint16_t instanceId) {
        const tcnn::vec3 instanceExtent = fetchExpandedExtent(instanceId);
        const float instanceMaxExtent   = fmaxf(instanceExtent.x, fmaxf(instanceExtent.y, instanceExtent.z));
        return position / instanceMaxExtent + 0.5f;
    }

    static inline __device__ tcnn::vec3 fetchExpandedExtent(uint16_t instanceId) {
        const uint16_t instanceExtentIdx = tcnn::clamp<uint16_t>(instanceId, 0u, static_cast<uint16_t>(Params::NumInstances) - 1u) * 3;
        return tcnn::vec3{Params::instanceExpandedExtents[instanceExtentIdx],
                          Params::instanceExpandedExtents[instanceExtentIdx + 1],
                          Params::instanceExpandedExtents[instanceExtentIdx + 2]};
    }
};
