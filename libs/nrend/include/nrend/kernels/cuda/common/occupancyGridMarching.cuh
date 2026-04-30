// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include <tiny-cuda-nn/bounding_box.h>

namespace {
using namespace tcnn;

inline constexpr __host__ __device__ uint32_t NERF_GRIDSIZE() {
    return 128;
}
inline constexpr __host__ __device__ uint32_t NERF_GRID_N_CELLS() {
    return NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE();
}
inline constexpr __host__ __device__ uint32_t NERF_STEPS() {
    return 1024;
} // finest number of steps per unit length
inline constexpr __host__ __device__ uint32_t NERF_CASCADES() {
    return 8;
}
inline constexpr __host__ __device__ float SQRT3() {
    return 1.73205080757f;
}
inline constexpr __host__ __device__ float STEPSIZE() {
    return (SQRT3() / NERF_STEPS());
} // for nerf raymarch
inline constexpr __host__ __device__ float MIN_CONE_STEPSIZE() {
    return STEPSIZE();
}
// Maximum step size is the width of the coarsest gridsize cell.
inline constexpr __host__ __device__ float MAX_CONE_STEPSIZE() {
    return STEPSIZE() * (1 << (NERF_CASCADES() - 1)) * NERF_STEPS() / NERF_GRIDSIZE();
}

inline __host__ __device__ uint32_t cascaded_grid_idx_at(vec3 pos, uint32_t mip) {
    float mip_scale = scalbnf(1.0f, -mip);
    pos -= vec3(0.5f);
    pos *= mip_scale;
    pos += vec3(0.5f);

    ivec3 i = pos * (float)NERF_GRIDSIZE();
    if (i.x < 0 || i.x >= NERF_GRIDSIZE() || i.y < 0 || i.y >= NERF_GRIDSIZE() || i.z < 0 || i.z >= NERF_GRIDSIZE()) {
        return 0xFFFFFFFF;
    }

    return morton3D(i.x, i.y, i.z);
}

inline __host__ __device__ uint32_t grid_mip_offset(uint32_t mip) {
    return NERF_GRID_N_CELLS() * mip;
}

inline __host__ __device__ bool density_grid_occupied_at(const vec3& pos, const uint8_t* density_grid_bitfield, uint32_t mip) {
    uint32_t idx = cascaded_grid_idx_at(pos, mip);
    if (idx == 0xFFFFFFFF) {
        return false;
    }
    return density_grid_bitfield[idx / 8 + grid_mip_offset(mip) / 8] & (1 << (idx % 8));
}

} // namespace
