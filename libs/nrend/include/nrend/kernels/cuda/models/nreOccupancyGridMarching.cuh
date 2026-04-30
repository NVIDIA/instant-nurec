// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include <nrend/kernels/cuda/common/occupancyGridMarching.cuh>

namespace {
inline __device__ float distance_to_next_voxel(const vec3& pos, const vec3& dir, float res) { // dda like step
    vec3 p   = res * (pos - 0.5f);
    float tx = (floorf(p.x + 0.5f + 0.5f * sign(dir.x)) - p.x) / dir.x;
    float ty = (floorf(p.y + 0.5f + 0.5f * sign(dir.y)) - p.y) / dir.y;
    float tz = (floorf(p.z + 0.5f + 0.5f * sign(dir.z)) - p.z) / dir.z;
    float t  = min(min(tx, ty), tz);

    return fmaxf(t / res, 0.0f);
}

inline __device__ float calc_dt(float t, float cone_angle, float dt_min, float dt_max) {
    return clamp(t * cone_angle, dt_min, dt_max);
}

inline __device__ float advance_to_next_voxel(float t, const vec3& pos, const vec3& dir, uint32_t mip) {
    const float res = scalbnf(NERF_GRIDSIZE(), -(int)mip);
    return t + distance_to_next_voxel(pos, dir, res) + 1e-06f;
}

inline __device__ uint32_t mip_from_pos(const vec3& pos, uint32_t max_cascade = NERF_CASCADES() - 1) {
    int exponent;
    float maxval = max(abs(pos - 0.5f));
    frexpf(maxval, &exponent);
    return (uint32_t)clamp(exponent + 1, 0, (int)max_cascade);
}

__device__ float advanceToNextOccupiedVoxel(const vec3& ray_origin,
                                            const vec3& ray_dir,
                                            const uint8_t* __restrict__ density_grid,
                                            uint32_t min_mip,
                                            uint32_t max_mip,
                                            const vec2& tminmax) {
    float t = tminmax.x;
    while (true) {
        const vec3 pos = ray_origin + t * ray_dir;
        if (t >= tminmax.y) {
            return tminmax.y;
        }

        const uint32_t mip = tcnn::clamp(mip_from_pos(pos), min_mip, max_mip);

        if (!density_grid || density_grid_occupied_at(pos, density_grid, mip)) {
            return t;
        }

        const float t_min = advance_to_next_voxel(t, pos, ray_dir, mip);
        const float dt    = calc_dt(t, M_CONE_ANGLE, M_STEP_SIZE, 1e10f);
        t += dt * max(static_cast<int>(((t_min - t) / dt) + 0.5f), 1);
    }
}
} // namespace
