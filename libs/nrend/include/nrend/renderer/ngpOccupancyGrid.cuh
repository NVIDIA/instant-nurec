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

#include <tiny-cuda-nn/reduce_sum.h>

#include <neural-graphics-primitives/bounding_box.cuh>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/nerf_device.cuh>

namespace nrend {

struct NGPDensityGrid {
    // density grid structure for accelerated rendering
    std::vector<uint8_t> density_grid_host;
    uint32_t density_grid_ema_step = 0;
    uint32_t max_cascade           = 0;
};

struct NGPDensityGridDevice {
    tcnn::GPUMemory<__half> density_grid_device;
    tcnn::GPUMemory<uint8_t> density_grid_bitfield;
    inline uint8_t* get_density_grid_bitfield_mip(uint32_t mip) const {
        return density_grid_bitfield.data() + ngp::grid_mip_offset(mip) / 8;
    }
    tcnn::GPUMemory<float> density_grid_mean;
};

void update_density_grid_mean_and_bitfield(
    const NGPDensityGrid& ngpDensityGrid, NGPDensityGridDevice& ngpDensityGridDevice, cudaStream_t stream);

inline __device__ __half cascaded_grid_at(tcnn::vec3 pos, const __half* cascaded_grid, uint32_t mip) {
    uint32_t idx = ngp::cascaded_grid_idx_at(pos, mip);
    if (idx == 0xFFFFFFFF) {
        return __float2half(0.0f);
    }
    return cascaded_grid[idx + ngp::grid_mip_offset(mip)];
}

__global__ void generate_grid_samples_nerf_uniform(
    tcnn::ivec3 res_3d,
    const uint32_t step,
    ngp::BoundingBox render_aabb,
    tcnn::mat3 render_aabb_to_local,
    ngp::BoundingBox train_aabb,
    ngp::NerfPosition* __restrict__ out);

void grid_samples_half_to_float(
    const uint32_t n_elements,
    ngp::BoundingBox aabb,
    float* dst,
    const tcnn::network_precision_t* network_output,
    ngp::ENerfActivation density_activation,
    const ngp::NerfPosition* __restrict__ coords_in,
    const tcnn::network_precision_t* __restrict__ grid_in,
    uint32_t max_cascade,
    cudaStream_t stream);

tcnn::vec3 compute_world_scale(uint32_t max_cascade, const ngp::BoundingBox& aabb);

void convert_grid_to_voxels(
    uint32_t max_cascade,
    const uint8_t* bitOccupancyFieldPtr,
    const ngp::BoundingBox& aabb,
    std::vector<ngp::BoundingBox>& voxels,
    std::vector<uint32_t>& voxelGridIdx);

void convert_voxels_to_triangles(const std::vector<ngp::BoundingBox>& aabb, std::vector<float3>& vrt, std::vector<int3>& tri);

} // namespace nrend
