// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include <nrend/renderer/ngpOccupancyGrid.cuh>

namespace {

using namespace nrend;

__global__ void grid_to_bitfield(
    const uint32_t n_elements,
    const __half* __restrict__ grid,
    uint8_t* __restrict__ grid_bitfield,
    const float* __restrict__ mean_density_ptr) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) {
        return;
    }

    uint8_t bits = 0;

    const __half thresh = __float2half(std::min(ngp::NERF_MIN_OPTICAL_THICKNESS(), *mean_density_ptr));

#pragma unroll
    for (uint8_t j = 0; j < 8; ++j) {
        bits |= grid[i * 8 + j] > thresh ? ((uint8_t)1 << j) : 0;
    }

    grid_bitfield[i] = bits;
}

__global__ void bitfield_max_pool(
    const uint32_t n_elements, const uint8_t* __restrict__ prev_level, uint8_t* __restrict__ next_level) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) {
        return;
    }

    uint8_t bits = 0;

#pragma unroll
    for (uint8_t j = 0; j < 8; ++j) {
        // If any bit is set in the previous level, set this
        // level's bit. (Max pooling.)
        bits |= prev_level[i * 8 + j] > 0 ? ((uint8_t)1 << j) : 0;
    }

    uint32_t x = tcnn::morton3D_invert(i >> 0) + ngp::NERF_GRIDSIZE() / 8;
    uint32_t y = tcnn::morton3D_invert(i >> 1) + ngp::NERF_GRIDSIZE() / 8;
    uint32_t z = tcnn::morton3D_invert(i >> 2) + ngp::NERF_GRIDSIZE() / 8;

    next_level[tcnn::morton3D(x, y, z)] |= bits;
}

__global__ void grid_samples_half_to_float_k(
    const uint32_t n_elements,
    ngp::BoundingBox aabb,
    float* dst,
    const tcnn::network_precision_t* network_output,
    ngp::ENerfActivation density_activation,
    const ngp::NerfPosition* __restrict__ coords_in,
    const tcnn::network_precision_t* __restrict__ grid_in,
    uint32_t max_cascade) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements) {
        return;
    }

    // let's interpolate for marching cubes based on the raw MLP output, not the density (exponentiated) version
    // float mlp = network_to_density(float(network_output[i * padded_output_width]), density_activation);
    float mlp = float(network_output[i]);

    if (grid_in) {
        tcnn::vec3 pos     = ngp::unwarp_position(coords_in[i].p, aabb);
        float grid_density = __half2float(cascaded_grid_at(pos, grid_in, ngp::mip_from_pos(pos, max_cascade)));
        if (grid_density < ngp::NERF_MIN_OPTICAL_THICKNESS()) {
            mlp = -10000.f;
        }
    }
    dst[i] = mlp;
}

inline bool voxel_cover_previous_cascade(uint32_t c, uint32_t i, uint32_t j, uint32_t k) {
    constexpr uint32_t min_prev = ngp::NERF_GRIDSIZE() >> 2;
    constexpr uint32_t max_prev = 3 * (ngp::NERF_GRIDSIZE() >> 2) - 1;

    return !(
        (c == 0) || (i < min_prev) || (i > max_prev) || (j < min_prev) || (j > max_prev) || (k < min_prev) ||
        (k > max_prev));
}

} // namespace

namespace nrend {

void update_density_grid_mean_and_bitfield(
    const NGPDensityGrid& ngpDensityGrid, NGPDensityGridDevice& ngpDensityGridDevice, cudaStream_t stream) {
    const uint32_t n_elements        = ngp::NERF_GRIDSIZE() * ngp::NERF_GRIDSIZE() * ngp::NERF_GRIDSIZE();
    const size_t size_including_mips = ngp::grid_mip_offset(ngp::NERF_CASCADES()) / 8;
    ngpDensityGridDevice.density_grid_bitfield.enlarge(size_including_mips);
    ngpDensityGridDevice.density_grid_mean.enlarge(tcnn::reduce_sum_workspace_size(n_elements));

    CUDA_CHECK_THROW(cudaMemsetAsync(ngpDensityGridDevice.density_grid_mean.data(), 0, sizeof(float), stream));

    const bool hasDensityGridDevice = ngpDensityGridDevice.density_grid_device.size() > 0;
    __half* density_grid            = hasDensityGridDevice ? ngpDensityGridDevice.density_grid_device.data() : nullptr;
    if (!hasDensityGridDevice) {
        CUDA_CHECK_THROW(cudaMallocAsync(&density_grid, ngpDensityGrid.density_grid_host.size(), stream));
        CUDA_CHECK_THROW(cudaMemcpyAsync(
            density_grid,
            ngpDensityGrid.density_grid_host.data(),
            ngpDensityGrid.density_grid_host.size(),
            cudaMemcpyHostToDevice,
            stream));
    }

    tcnn::reduce_sum(
        density_grid,
        [n_elements] __device__(float val) { return fmaxf(val, 0.f) / (n_elements); },
        ngpDensityGridDevice.density_grid_mean.data(),
        n_elements,
        stream);

    tcnn::linear_kernel(
        grid_to_bitfield,
        0,
        stream,
        n_elements / 8 * ngp::NERF_CASCADES(),
        density_grid,
        ngpDensityGridDevice.density_grid_bitfield.data(),
        ngpDensityGridDevice.density_grid_mean.data());

    if (!hasDensityGridDevice) {
        CUDA_CHECK_THROW(cudaFreeAsync(density_grid, stream));
    }

    for (uint32_t level = 1; level < ngp::NERF_CASCADES(); ++level) {
        tcnn::linear_kernel(
            bitfield_max_pool,
            0,
            stream,
            n_elements / 64,
            ngpDensityGridDevice.get_density_grid_bitfield_mip(level - 1),
            ngpDensityGridDevice.get_density_grid_bitfield_mip(level));
    }
}

__global__ void generate_grid_samples_nerf_uniform(
    tcnn::ivec3 res_3d,
    const uint32_t step,
    ngp::BoundingBox render_aabb,
    tcnn::mat3 render_aabb_to_local,
    ngp::BoundingBox train_aabb,
    ngp::NerfPosition* __restrict__ out) {
    // check grid_in for negative values -> must be negative on output
    uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t z = threadIdx.z + blockIdx.z * blockDim.z;
    if (x >= res_3d.x || y >= res_3d.y || z >= res_3d.z) {
        return;
    }

    uint32_t i     = x + y * res_3d.x + z * res_3d.x * res_3d.y;
    tcnn::vec3 pos = tcnn::vec3{(float)x, (float)y, (float)z} / tcnn::vec3(res_3d - 1);
    pos            = transpose(render_aabb_to_local) * (pos * (render_aabb.max - render_aabb.min) + render_aabb.min);
    out[i]         = {ngp::warp_position(pos, train_aabb), ngp::warp_dt(ngp::MIN_CONE_STEPSIZE())};
}

void grid_samples_half_to_float(
    const uint32_t n_elements,
    ngp::BoundingBox aabb,
    float* dst,
    const tcnn::network_precision_t* network_output,
    ngp::ENerfActivation density_activation,
    const ngp::NerfPosition* __restrict__ coords_in,
    const tcnn::network_precision_t* __restrict__ grid_in,
    uint32_t max_cascade,
    cudaStream_t stream) {
    tcnn::linear_kernel(
        grid_samples_half_to_float_k, 0, stream, n_elements, aabb, dst, network_output, density_activation, coords_in, grid_in, max_cascade);
}

tcnn::vec3 compute_world_scale(uint32_t max_cascade, const ngp::BoundingBox& aabb) {
    return (aabb.max - aabb.min) / (float)(ngp::NERF_GRIDSIZE() * (1 << (max_cascade)));
}

void convert_grid_to_voxels(
    uint32_t max_cascade,
    const uint8_t* bitOccupancyFieldPtr,
    const ngp::BoundingBox& aabb,
    std::vector<ngp::BoundingBox>& voxels,
    std::vector<uint32_t>& voxelGridIdx) {
    if (ngp::NERF_GRIDSIZE() < 4) {
        return;
    }

    const __half* occupancyFieldPtr  = reinterpret_cast<const __half*>(bitOccupancyFieldPtr);
    const int32_t maxCascadeMinBound = -1 * static_cast<int32_t>((ngp::NERF_GRIDSIZE() >> 1) * (1 << max_cascade));

    const size_t initialNumAABB = static_cast<size_t>(
        ngp::NERF_GRIDSIZE() * ngp::NERF_GRIDSIZE() * ngp::NERF_GRIDSIZE() * (max_cascade + 1) * 0.2f + 0.5f);
    voxels.reserve(initialNumAABB);
    voxelGridIdx.reserve(initialNumAABB);

    const vec3 worldScale = compute_world_scale(max_cascade, aabb);

    for (uint32_t c = 0; c <= max_cascade; ++c) {
        const int32_t maxBound    = (ngp::NERF_GRIDSIZE() >> 1) * (1 << c);
        const int32_t minBound    = -1 * maxBound;
        const int32_t voxelSize   = (maxBound - minBound) / ngp::NERF_GRIDSIZE();
        const int32_t voxelOffset = minBound - maxCascadeMinBound;

        for (int32_t i = 0; i < ngp::NERF_GRIDSIZE(); ++i) {
            for (int32_t j = 0; j < ngp::NERF_GRIDSIZE(); ++j) {
                for (int32_t k = 0; k < ngp::NERF_GRIDSIZE(); ++k) {
                    // property : when NERF_GRIDSIZE >= 4, higher mip voxel is either completely filled with with
                    //            lower mip voxels or not at all.
                    if (!voxel_cover_previous_cascade(c, i, j, k)) {
                        float occupancy    = .0f;
                        const uint32_t idx = tcnn::morton3D(i, j, k);
                        if (idx != 0xFFFFFFFF) {
                            occupancy = static_cast<float>(occupancyFieldPtr[idx + ngp::grid_mip_offset(c)]);
                        }
                        const bool isCellActive = occupancy > ngp::NERF_MIN_OPTICAL_THICKNESS();
                        if (isCellActive) {
                            voxels.emplace_back(
                                tcnn::vec3{
                                    static_cast<float>(voxelOffset + i * voxelSize),
                                    static_cast<float>(voxelOffset + j * voxelSize),
                                    static_cast<float>(voxelOffset + k * voxelSize)} *
                                        worldScale +
                                    aabb.min,
                                tcnn::vec3{
                                    static_cast<float>(voxelOffset + (i + 1) * voxelSize),
                                    static_cast<float>(voxelOffset + (j + 1) * voxelSize),
                                    static_cast<float>(voxelOffset + (k + 1) * voxelSize)} *
                                        worldScale +
                                    aabb.min);
                            voxelGridIdx.emplace_back(
                                static_cast<uint32_t>(i) | (static_cast<uint32_t>(j) << 8) |
                                (static_cast<uint32_t>(k) << 16) | (static_cast<uint32_t>(c) << 24));
                        }
                    }
                }
            }
        }
    }
}

void convert_voxels_to_triangles(const std::vector<ngp::BoundingBox>& aabb, std::vector<float3>& vrt, std::vector<int3>& tri) {
    const size_t numVoxels = aabb.size();

    constexpr int numVrt = 8;
    constexpr int numTri = 12;

    int3 voxelTri[numTri] = {
        make_int3(0, 2, 3),
        make_int3(0, 1, 2),
        make_int3(3, 2, 6),
        make_int3(3, 6, 7),
        make_int3(2, 1, 5),
        make_int3(2, 5, 6),
        make_int3(4, 1, 0),
        make_int3(4, 5, 1),
        make_int3(4, 0, 3),
        make_int3(4, 3, 7),
        make_int3(7, 5, 4),
        make_int3(7, 6, 5)};

    tri.resize(numVoxels * numTri);
    for (int i = 0; i < numVoxels; ++i) {
        for (int j = 0; j < numTri; ++j) {
            tri[i * numTri + j] = int3{voxelTri[j].x + i * numVrt, voxelTri[j].y + i * numVrt, voxelTri[j].z + i * numVrt};
        }
    }

    vrt.resize(numVoxels * numVrt);
    for (size_t i = 0; i < numVoxels; ++i) {
        vrt[i * numVrt + 0] = float3{aabb[i].min.x, aabb[i].min.y, aabb[i].min.z};
        vrt[i * numVrt + 1] = float3{aabb[i].min.x, aabb[i].max.y, aabb[i].min.z};
        vrt[i * numVrt + 2] = float3{aabb[i].max.x, aabb[i].max.y, aabb[i].min.z};
        vrt[i * numVrt + 3] = float3{aabb[i].max.x, aabb[i].min.y, aabb[i].min.z};
        vrt[i * numVrt + 4] = float3{aabb[i].min.x, aabb[i].min.y, aabb[i].max.z};
        vrt[i * numVrt + 5] = float3{aabb[i].min.x, aabb[i].max.y, aabb[i].max.z};
        vrt[i * numVrt + 6] = float3{aabb[i].max.x, aabb[i].max.y, aabb[i].max.z};
        vrt[i * numVrt + 7] = float3{aabb[i].max.x, aabb[i].min.y, aabb[i].max.z};
    }
}
} // namespace nrend
