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
inline __host__ __device__ float distance_to_next_voxel(const vec3& pos, const vec3& dir, float res) { // dda like step
    vec3 p   = res * (pos - 0.5f);
    float tx = (floorf(p.x + 0.5f + 0.5f * sign(dir.x)) - p.x) / dir.x;
    float ty = (floorf(p.y + 0.5f + 0.5f * sign(dir.y)) - p.y) / dir.y;
    float tz = (floorf(p.z + 0.5f + 0.5f * sign(dir.z)) - p.z) / dir.z;
    float t  = min(min(tx, ty), tz);

    return fmaxf(t / res, 0.0f);
}

inline __host__ __device__ float to_stepping_space(float t) {
    if (M_CONE_ANGLE <= 1e-5f) {
        return t / MIN_CONE_STEPSIZE();
    }

    if (t <= stepping_space_at) {
        return (t - stepping_space_at) / MIN_CONE_STEPSIZE() + stepping_space_a;
    } else if (t <= stepping_space_bt) {
        return logf(t) / stepping_space_log1p_c;
    } else {
        return (t - stepping_space_bt) / MAX_CONE_STEPSIZE() + stepping_space_b;
    }
}

inline __host__ __device__ float from_stepping_space(float n) {
    if (M_CONE_ANGLE <= 1e-5f) {
        return n * MIN_CONE_STEPSIZE();
    }
    if (n <= stepping_space_a) {
        return (n - stepping_space_a) * MIN_CONE_STEPSIZE() + stepping_space_at;
    } else if (n <= stepping_space_b) {
        return expf(n * stepping_space_log1p_c);
    } else {
        return (n - stepping_space_b) * MAX_CONE_STEPSIZE() + stepping_space_bt;
    }
}

inline __host__ __device__ float advance_n_steps(float t, float n) {
    return from_stepping_space(to_stepping_space(t) + n);
}

inline __host__ __device__ float calc_dt(float t) {
    return advance_n_steps(t, 1.f) - t;
}

inline __host__ __device__ float advance_to_next_voxel(
    float t, const vec3& pos, const vec3& dir, uint32_t mip, bool exponentialStepping = true) {
    float res = scalbnf(NERF_GRIDSIZE(), -(int)mip);
    if (exponentialStepping) {
        float t_target = t + distance_to_next_voxel(pos, dir, res) + 1e-06f;
        // Analytic stepping in multiples of 1 in the "log-space" of our exponential stepping routine
        t        = to_stepping_space(t);
        t_target = to_stepping_space(t_target);
        return from_stepping_space(t + ceilf(fmaxf(t_target - t, 0.5f)));
    } else {
        return t + distance_to_next_voxel(pos, dir, res) + 1e-06f;
    }
}

inline __host__ __device__ uint32_t mip_from_pos(const vec3& pos, uint32_t max_cascade = NERF_CASCADES() - 1) {
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
        vec3 pos = ray_origin + t * ray_dir;
        if (t >= tminmax.y) {
            return tminmax.y;
        }

        uint32_t mip = tcnn::clamp(mip_from_pos(pos), min_mip, max_mip);

        if (!density_grid || density_grid_occupied_at(pos, density_grid, mip)) {
            return t;
        }

        // Find largest empty voxel surrounding us, such that we can advance as far as possible in the next step.
        // Other places that do voxel stepping don't need this, because they don't rely on thread coherence as
        // much as this one here.
        while (mip < max_mip && !density_grid_occupied_at(pos, density_grid, mip + 1)) {
            ++mip;
        }

        t = advance_to_next_voxel(t, pos, ray_dir, mip);
    }
}
} // namespace
