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

#include <nrend/sensors/sensors.h>

#include <nrend/utils/nreVec.h>

#include <tiny-cuda-nn/bounding_box.h>

namespace nrend {

using TTrackInstancePose = tcnn::vec<7>;

struct MemoryHandles {

    const uint64_t* handles;

    template <typename T>
    inline TCNN_HOST_DEVICE T* bufferPtr(int index) {
        return reinterpret_cast<T*>(handles[index]);
    }
};

struct RenderParameters {
    uint32_t id;
    tcnn::vec2 frameResolution;
    tcnn::vec2 frameTileOffset;
    tcnn::ivec2 frameTileResolution;
    float hitTransmittance;
    tcnn::BoundingBox objectAABB;
    tcnn::mat4x3 worldToObjectTransform;
    tcnn::mat4x3 objectToWorldTransform;
    TSensorModel sensorModel;
    TSensorState sensorState;
    tcnn::mat4x3 colorCorrectionMatrix;
    tcnn::uvec4 objectInstanceIds;
    int32_t numActiveTrackInstances;

    static constexpr float defaultMinTransmittance = 0.0001f;
};

// RenderParameters is not POD
struct RenderParametersArray {
    char data[sizeof(RenderParameters)];
};

} // namespace nrend
