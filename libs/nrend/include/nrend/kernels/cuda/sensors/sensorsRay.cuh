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

#include <nrend/sensors/sensors.h>

namespace nrend {

template <uint32_t InvalidRayIdx = -1U>
static inline __device__ uint32_t threadSensorRayIdx(
    const RowOffsetStructuredSpinningLidarProjectionParameters& sensorParams,
    const tcnn::ivec2& /*resolution*/,
    const tcnn::ivec2& /*position*/,
    const tcnn::vec2& offset) {

    // FIXME : LIDAR does not support tile-frame rendering
    if ((fabsf(offset.x) > 0) || (fabsf(offset.y) > 0)) {
        return InvalidRayIdx;
    }

    const int tileId            = blockIdx.y * sensorParams.elevationNBins + blockIdx.x;
    const tcnn::ivec2 tileRange = sensorParams.tilesPackInfo[tileId];
    const int threadId          = threadIdx.y * blockDim.x + threadIdx.x;
    if (threadId < tileRange.y) {
        const tcnn::ivec2 rayUV = sensorParams.tilesToElementsMap[tileRange.x + threadId];
        return rayUV.x + rayUV.y * sensorParams.nRows;
    }
    return InvalidRayIdx;
}

template <uint32_t InvalidRayIdx = -1U>
static inline __device__ uint32_t threadSensorRayIdx(
    const SensorProjectionModel& sensorModel,
    const tcnn::ivec2& resolution,
    const tcnn::ivec2& position,
    const tcnn::vec2& offset) {
    if (sensorModel.modelType == SensorProjectionModel::RowOffsetStructuredSpinningLidarModel) {
        return threadSensorRayIdx<InvalidRayIdx>(sensorModel.nreHesaiP128LidarParams, resolution, position, offset);
    }
    if ((position.x < resolution.x) && (position.y < resolution.y)) {
        return position.x + resolution.x * position.y;
    }
    return InvalidRayIdx;
}

static inline __device__ uint32_t threadSensorRayTileIdx(const SensorProjectionModel& sensorModel) {
    if (sensorModel.modelType == SensorProjectionModel::RowOffsetStructuredSpinningLidarModel) {
        return blockIdx.y * sensorModel.nreHesaiP128LidarParams.elevationNBins + blockIdx.x;
    }
    return blockIdx.y * gridDim.x + blockIdx.x;
}

} // namespace nrend
