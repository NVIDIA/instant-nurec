// SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include "utils.h"

#include "intersection.cuh"

#include <ku/common.cuh>

#include <c10/cuda/CUDAStream.h>

template <typename scalar_t>
__global__ void ray_aabb_intersect_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> rays_o,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> rays_d,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> aabbs_min,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> aabbs_max,
    torch::PackedTensorAccessor64<bool, 2, torch::RestrictPtrTraits> hits_flag,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> hits_t) {

    auto const N_rays = rays_o.size(0), N_aabbs = aabbs_min.size(0);
    auto const N_intersections = N_rays * N_aabbs;

    auto const thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_idx >= N_intersections)
        // Ray or aabb bounding box do not exist
        return;

    auto const ray_idx  = thread_idx / N_aabbs;
    auto const aabb_idx = thread_idx % N_aabbs;

    auto const ray_o = make_float3(rays_o[ray_idx][0], rays_o[ray_idx][1], rays_o[ray_idx][2]);
    auto const ray_d = make_float3(rays_d[ray_idx][0], rays_d[ray_idx][1], rays_d[ray_idx][2]);
    auto const inv_d = 1.0f / ray_d;

    auto const aabb_min = make_float3(aabbs_min[aabb_idx][0], aabbs_min[aabb_idx][1], aabbs_min[aabb_idx][2]);
    auto const aabb_max = make_float3(aabbs_max[aabb_idx][0], aabbs_max[aabb_idx][1], aabbs_max[aabb_idx][2]);

    auto const t1t2 = ray_aabb_intersect(ray_o, inv_d, aabb_min, aabb_max);

    if (t1t2.y > 0) { // if ray hits the voxel store output values if output is requested
        if (hits_flag.size(0))
            hits_flag[ray_idx][aabb_idx] = true;
        if (hits_t.size(0)) {
            hits_t[ray_idx][aabb_idx][0] = t1t2.x;
            hits_t[ray_idx][aabb_idx][1] = t1t2.y;
        }
    }
}

std::vector<std::optional<torch::Tensor>> ray_aabb_intersect_cu(
    const torch::Tensor rays_o,    // N_rays x 3 (3d world positions)
    const torch::Tensor rays_d,    // N_rays x 3 (normalized 3d world directions)
    const torch::Tensor aabbs_min, // (N_AABBs x 3) coordinates of the bottom back left corner of AABB (min_x, min_y, min_z)
    const torch::Tensor aabbs_max, // (N_AABBs x 3) coordinates of the top front right corner of AABB (max_x, max_y, max_z)
    const bool compute_hits_flag,  // Whether to allocate + compute per-intersection flags
    const bool compute_hits_t      // Whether to allocate + compute per-intersection ts
) {
    auto rays_o_arg = torch::TensorArg{rays_o, "rays_o", 1};
    auto rays_d_arg = torch::TensorArg{rays_d, "rays_d", 2};

    auto aabbs_min_arg = torch::TensorArg{aabbs_min, "aabbs_min", 3};
    auto aabbs_max_arg = torch::TensorArg{aabbs_max, "aabbs_max", 4};

    torch::checkAllSameType(__func__, {rays_o_arg, rays_d_arg, aabbs_min_arg, aabbs_max_arg});
    torch::checkAllSameGPU(__func__, {rays_o_arg, rays_d_arg, aabbs_min_arg, aabbs_max_arg});
    torch::checkAllContiguous(__func__, {rays_o_arg, rays_d_arg, aabbs_min_arg, aabbs_max_arg});

    auto const N_rays = rays_o.size(0), N_aabbs = aabbs_min.size(0);

    torch::checkSize(__func__, rays_o_arg, {N_rays, 3});
    torch::checkSize(__func__, rays_d_arg, {N_rays, 3});
    torch::checkSize(__func__, aabbs_min_arg, {N_aabbs, 3});
    torch::checkSize(__func__, aabbs_max_arg, {N_aabbs, 3});

    // initialize output buffers if outputs are enabled
    auto hits_flag = std::optional<torch::Tensor>{torch::empty({0, 0}, rays_o.options().dtype(torch::kBool))};
    if (compute_hits_flag)
        hits_flag = torch::zeros({N_rays, N_aabbs}, rays_o.options().dtype(torch::kBool));

    auto hits_t = std::optional<torch::Tensor>{torch::empty({0, 0, 0}, rays_o.options())};
    if (compute_hits_t)
        // initialize all hits_t to -1 (if there is no intersection this stays -1 otherwise updated with the t value)
        hits_t = torch::full({N_rays, N_aabbs, 2}, -1.0f, rays_o.options());

    auto const N_intersections = N_rays * N_aabbs;
    auto const threads         = dim3(nextPowerOfTwoCapped(N_intersections, 1024));
    auto const blocks          = dim3((N_intersections + threads.x - 1) / threads.x);
    auto const stream          = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES(rays_o.scalar_type(), "ray_aabb_intersect_cu",
                               ([&] {
                                   ray_aabb_intersect_kernel<<<blocks, threads, 0, stream>>>(
                                       rays_o.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       rays_d.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       aabbs_min.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       aabbs_max.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       hits_flag->packed_accessor64<bool, 2, torch::RestrictPtrTraits>(),
                                       hits_t->packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());
                               }));

    // Drop surrogate buffers if not supposed to be computed
    if (!compute_hits_flag)
        hits_flag.reset();

    if (!compute_hits_t)
        hits_t.reset();

    // NOTE: that the entering depth `hits_t[:, :, 0]` could be negative as they are the results of extended intersection compute.
    return {hits_flag, hits_t};
}

__global__ void ray_sphere_intersect_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_o,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rays_d,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> centers,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> radii,
    const int max_hits,
    int* __restrict__ hit_cnt,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> hits_t,
    torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> hits_sphere_idx) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int s = blockIdx.y * blockDim.y + threadIdx.y;

    if (s >= centers.size(0) || r >= rays_o.size(0))
        return;

    const float3 ray_o  = make_float3(rays_o[r][0], rays_o[r][1], rays_o[r][2]);
    const float3 ray_d  = make_float3(rays_d[r][0], rays_d[r][1], rays_d[r][2]);
    const float3 center = make_float3(centers[s][0], centers[s][1], centers[s][2]);

    const float2 t1t2 = ray_sphere_intersect(ray_o, ray_d, center, radii[s]);

    if (t1t2.y > 0) { // if ray hits the sphere
        const int cnt = atomicAdd(&hit_cnt[r], 1);
        if (cnt < max_hits) {
            hits_t[r][cnt][0]       = t1t2.x;
            hits_t[r][cnt][1]       = t1t2.y;
            hits_sphere_idx[r][cnt] = s;
        }
    }
}

std::vector<torch::Tensor> ray_sphere_intersect_cu(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor centers,
    const torch::Tensor radii,
    const int max_hits) {

    const int N_rays = rays_o.size(0), N_spheres = centers.size(0);
    auto hits_t = torch::zeros({N_rays, max_hits, 2}, rays_o.options()) - 1;
    auto hits_sphere_idx =
        torch::zeros({N_rays, max_hits},
                     torch::dtype(torch::kLong).device(rays_o.device())) -
        1;
    auto hit_cnt =
        torch::zeros({N_rays},
                     torch::dtype(torch::kInt32).device(rays_o.device()));

    const dim3 threads(256, 1);
    const dim3 blocks((N_rays + threads.x - 1) / threads.x,
                      (N_spheres + threads.y - 1) / threads.y);

    AT_DISPATCH_FLOATING_TYPES(rays_o.scalar_type(), "ray_sphere_intersect_cu",
                               ([&] {
                                   ray_sphere_intersect_kernel<<<blocks, threads>>>(
                                       rays_o.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                                       rays_d.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                                       centers.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                                       radii.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                       max_hits,
                                       hit_cnt.data_ptr<int>(),
                                       hits_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                                       hits_sphere_idx.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>());
                               }));

    // sort intersections from near to far based on t1
    auto hits_order = std::get<1>(torch::sort(hits_t.index({"...", 0})));
    hits_sphere_idx = torch::gather(hits_sphere_idx, 1, hits_order);
    hits_t          = torch::gather(hits_t, 1, hits_order.unsqueeze(-1).tile({1, 1, 2}));

    // NOTE: that the entering depth `hits_t[:, :, 0]` could be negative as they are the results of extended intersection compute.
    return {hit_cnt, hits_t, hits_sphere_idx};
}
