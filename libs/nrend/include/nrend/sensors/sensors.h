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

// clang-format off
#include <nrend/utils/nreVec.h>
#include <nrend/utils/nrePose.h>
#include <nrend/sensors/sensorsModel.h>
// clang-format on
namespace nrend {

/// TODO : switch to uint32_t
using TTimestamp = int64_t;

// 3D position and 3D quaternion (x,y,z,w)
using TSensorPose = tcnn::vec<7>;

using TSensorModel = SensorProjectionModel;
struct TSensorState {
    TTimestamp startTimestamp;
    TSensorPose startPose;
    TTimestamp endTimestamp;
    TSensorPose endPose;
};

enum SensorType {
    Camera,
    Lidar
};

static inline TCNN_HOST_DEVICE bool sensorIsCamera(const SensorProjectionModel& sensorModel) {
    return (sensorModel.modelType == SensorProjectionModel::OrthographicModel) ||
           (sensorModel.modelType == SensorProjectionModel::PerspectiveModel) ||
           (sensorModel.modelType == SensorProjectionModel::OpenCVPinholeModel) ||
           (sensorModel.modelType == SensorProjectionModel::OpenCVFisheyeModel) ||
           (sensorModel.modelType == SensorProjectionModel::FThetaModel);
}

static inline TCNN_HOST_DEVICE bool sensorIsLidar(const SensorProjectionModel& sensorModel) {
    return sensorModel.modelType == SensorProjectionModel::RowOffsetStructuredSpinningLidarModel;
}

} // namespace nrend