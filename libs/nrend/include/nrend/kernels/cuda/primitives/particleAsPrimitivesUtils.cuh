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

#include <optix.h>

namespace nrend {

#ifdef __CUDACC__

// encode an opacity in [0,1] as a uint16_t to be stored in the instanceId
inline __device__ uint32_t opacityAsInstanceId(float opacity) {
    return __float2uint_rd(opacity * static_cast<float>(OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID));
};

// decode an opacity in [0,1] from a uint16_t stored in the instanceId
inline __device__ float instanceIdAsOpacity(uint32_t instanceId) {
    return __uint2float_rd(instanceId) / static_cast<float>(OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID);
};

#endif

} // namespace nrend
