// SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#pragma once

#include <ku/helper_math.cuh>

__device__ __forceinline__ float2 ray_aabb_intersect(
    const float3 ray_o,
    const float3 inv_d,
    const float3 aabb_min,
    const float3 aabb_max) {
    const float3 t_min = (aabb_min - ray_o) * inv_d;
    const float3 t_max = (aabb_max - ray_o) * inv_d;

    const float3 _t1 = fminf(t_min, t_max);
    const float3 _t2 = fmaxf(t_min, t_max);
    const float t1   = fmaxf(fmaxf(_t1.x, _t1.y), _t1.z);
    const float t2   = fminf(fminf(_t2.x, _t2.y), _t2.z);

    if (t1 > t2)
        return make_float2(-1.0f); // no intersection
    return make_float2(t1, t2);
}

__device__ __forceinline__ float2 ray_sphere_intersect(
    const float3 ray_o,
    const float3 ray_d,
    const float3 center,
    const float radius) {
    const float3 co = ray_o - center;

    const float a      = dot(ray_d, ray_d);
    const float half_b = dot(ray_d, co);
    const float c      = dot(co, co) - radius * radius;

    const float discriminant = half_b * half_b - a * c;

    if (discriminant < 0)
        return make_float2(-1.0f); // no intersection

    const float disc_sqrt = sqrtf(discriminant);
    return make_float2(-half_b - disc_sqrt, -half_b + disc_sqrt) / a;
}
