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

#include <tiny-cuda-nn/common.h>

namespace nrend {

using TPose = tcnn::vec<7>; //< [position, quaternion xyzw]

static inline TCNN_HOST_DEVICE TPose poseInverse(const TPose& pose) {
    const tcnn::mat3 invRotation = tcnn::transpose(
        tcnn::to_mat3(tcnn::tquat{pose[6], pose[3], pose[4], pose[5]}));
    const tcnn::quat invQuaternion = tcnn::quat{invRotation};

    TPose invPose;
    invPose.slice<0, 3>() = -1.0f * invRotation * pose.slice<0, 3>();
    invPose.slice<3, 4>() = tcnn::vec4{invQuaternion.x, invQuaternion.y, invQuaternion.z, invQuaternion.w};

    return invPose;
}

static inline TCNN_HOST_DEVICE TPose interpolatedPose(const TPose& startPose,
                                                      const TPose& endPose,
                                                      float relativeTime) {
    using namespace tcnn;

    const quat interpolatedQuat = slerp(quat{startPose[6], startPose[3], startPose[4], startPose[5]},
                                        quat{endPose[6], endPose[3], endPose[4], endPose[5]},
                                        relativeTime);
    TPose interpolated;
    interpolated.slice<0, 3>() = mix(startPose.slice<0, 3>(), endPose.slice<0, 3>(), relativeTime);
    interpolated.slice<3, 4>() = vec4{interpolatedQuat.x, interpolatedQuat.y, interpolatedQuat.z, interpolatedQuat.w};

    return interpolated;
}

static inline TCNN_HOST_DEVICE tcnn::mat4x3 poseToMat(const TPose& pose) {
    using namespace tcnn;

    const mat3 rotation = to_mat3(quat{pose[6], pose[3], pose[4], pose[5]});
    return mat4x3{rotation[0],
                  rotation[1],
                  rotation[2],
                  pose.slice<0, 3>()};
}

} // namespace nrend