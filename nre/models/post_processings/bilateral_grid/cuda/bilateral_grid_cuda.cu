// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

// #define BILATERALGRID_CHRONO

#include <array>
#ifdef BILATERALGRID_CHRONO
#include <chrono>
#endif
#include <functional>

#include "utils.h"

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

__device__ static inline int bool_to_bitmask(bool value, int bit_position) {
    return value ? 1 << bit_position : 0;
}

__device__ static inline bool bitmask_to_bool(int bitmask, int bit_position) {
    return (bitmask & (1 << bit_position)) != 0;
}

static constexpr int NOVELVIEW_GRID_INDEX = -1;

// Class for bilateral grid operations that shares common calculations between
// forward and backward
template <typename scalar_t>
class BilateralGridOp {
    // Input data with safe accessors
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> grid_;
    int32_t g_idx_;
    scalar_t x_;
    scalar_t y_;
    scalar_t z_;
    scalar_t r_;
    scalar_t g_;
    scalar_t b_;
    int depth_;
    int height_;
    int width_;
    int max_x_;
    int max_y_;
    int max_z_;
    int idx_;
    int rgb_dim_;

    // Computed indices and weights - Initialize ALL variables
    int x0_{0};
    int y0_{0};
    int z0_{0};
    int x1_{0};
    int y1_{0};
    int z1_{0};
    int x0_clamped_{0};
    int y0_clamped_{0};
    int z0_clamped_{0};
    int x1_clamped_{0};
    int y1_clamped_{0};
    int z1_clamped_{0};
    scalar_t wx_{0.0};
    scalar_t wy_{0.0};
    scalar_t wz_{0.0};
    scalar_t nx_{0.0};
    scalar_t ny_{0.0};
    scalar_t nz_{0.0};

    int idx000_{0};
    int idx001_{0};
    int idx010_{0};
    int idx011_{0};
    int idx100_{0};
    int idx101_{0};
    int idx110_{0};
    int idx111_{0};

    std::array<scalar_t, 3> transformed_rgb_;
    std::array<scalar_t, 3> affine_diag_;

    int mask_;
    bool enable_gridsize1_optimization_;

    static constexpr auto R_WEIGHT = 0.299F;
    static constexpr auto G_WEIGHT = 0.587F;
    static constexpr auto B_WEIGHT = 0.114F;

public:
    __device__ BilateralGridOp(
        torch::PackedTensorAccessor32<scalar_t, 5,
                                      torch::RestrictPtrTraits> const& grid,
        int grid_idx, int idx,
        torch::PackedTensorAccessor32<scalar_t, 2,
                                      torch::RestrictPtrTraits> const& coords_xy,
        torch::PackedTensorAccessor32<scalar_t, 2,
                                      torch::RestrictPtrTraits> const& rgb,
        int depth, int height, int width, bool enable_gridsize1_optimization)
        : grid_(grid), g_idx_(grid_idx), idx_(idx), rgb_dim_(0) // Will be set per channel
        , depth_(depth)
        , height_(height)
        , width_(width)
        , max_x_(width - 1)
        , max_y_(height - 1)
        , max_z_(depth - 1)
        , enable_gridsize1_optimization_(enable_gridsize1_optimization && (depth_ * height_ * width_ == 1)) {

        x_ = coords_xy[idx][0] * 2.0F - 1.0F;
        y_ = coords_xy[idx][1] * 2.0F - 1.0F;
        r_ = rgb[idx][0];
        g_ = rgb[idx][1];
        b_ = rgb[idx][2];

        // Skip pixel processing for invalid grid index
        if (g_idx_ == NOVELVIEW_GRID_INDEX) {
            transformed_rgb_[0] = r_;
            transformed_rgb_[1] = g_;
            transformed_rgb_[2] = b_;
            return;
        }

        // Calculate luma and rescale to [-1, 1] for grid_sample
        z_ = 2.F * (r_ * R_WEIGHT + g_ * G_WEIGHT + b_ * B_WEIGHT) - 1.F;

        // Setup coordinates and indices
        // Normalize coordinates to [0, max_xyz]
        nx_ = (x_ + 1.F) * 0.5F * max_x_;
        ny_ = (y_ + 1.F) * 0.5F * max_y_;
        nz_ = (z_ + 1.F) * 0.5F * max_z_;

        // ALWAYS compute coordinate variables to avoid uninitialized access
        // Get integer and fractional parts for trilinear interpolation
        x0_ = static_cast<int>(floorf(nx_));
        y0_ = static_cast<int>(floorf(ny_));
        z0_ = static_cast<int>(floorf(nz_));
        x1_ = x0_ + 1;
        y1_ = y0_ + 1;
        z1_ = z0_ + 1;

        // Clamp to grid bounds
        x0_clamped_ = max(0, min(x0_, max_x_));
        y0_clamped_ = max(0, min(y0_, max_y_));
        z0_clamped_ = max(0, min(z0_, max_z_));
        x1_clamped_ = max(0, min(x1_, max_x_));
        y1_clamped_ = max(0, min(y1_, max_y_));
        z1_clamped_ = max(0, min(z1_, max_z_));

        // Calculate interpolation weights
        wx_ = nx_ - static_cast<scalar_t>(x0_);
        wy_ = ny_ - static_cast<scalar_t>(y0_);
        wz_ = nz_ - static_cast<scalar_t>(z0_);

        // Calculate base indices for the 8 grid vertices
        idx000_ = (z0_clamped_ * height_ + y0_clamped_) * width_ + x0_clamped_;
        idx001_ = (z0_clamped_ * height_ + y0_clamped_) * width_ + x1_clamped_;
        idx010_ = (z0_clamped_ * height_ + y1_clamped_) * width_ + x0_clamped_;
        idx011_ = (z0_clamped_ * height_ + y1_clamped_) * width_ + x1_clamped_;
        idx100_ = (z1_clamped_ * height_ + y0_clamped_) * width_ + x0_clamped_;
        idx101_ = (z1_clamped_ * height_ + y0_clamped_) * width_ + x1_clamped_;
        idx110_ = (z1_clamped_ * height_ + y1_clamped_) * width_ + x0_clamped_;
        idx111_ = (z1_clamped_ * height_ + y1_clamped_) * width_ + x1_clamped_;

        // Set mask based on optimization mode
        if (enable_gridsize1_optimization_) {
            // For 1x1x1 grid, all coordinates are valid and point to the same cell
            mask_ = nx_ >= 0.F && nx_ <= 1.F && ny_ >= 0.F && ny_ <= 1.F &&
                    nz_ >= 0.F && nz_ <= 1.F;
        } else {
            // Padding mode is zero padding so we need to set the bits to 1 for the
            // valid grid points
            mask_ = 0;
            mask_ |= bool_to_bitmask(
                x0_ == x0_clamped_ && y0_ == y0_clamped_ && z0_ == z0_clamped_, 0);
            mask_ |= bool_to_bitmask(
                x1_ == x1_clamped_ && y0_ == y0_clamped_ && z0_ == z0_clamped_, 1);
            mask_ |= bool_to_bitmask(
                x0_ == x0_clamped_ && y1_ == y1_clamped_ && z0_ == z0_clamped_, 2);
            mask_ |= bool_to_bitmask(
                x1_ == x1_clamped_ && y1_ == y1_clamped_ && z0_ == z0_clamped_, 3);
            mask_ |= bool_to_bitmask(
                x0_ == x0_clamped_ && y0_ == y0_clamped_ && z1_ == z1_clamped_, 4);
            mask_ |= bool_to_bitmask(
                x1_ == x1_clamped_ && y0_ == y0_clamped_ && z1_ == z1_clamped_, 5);
            mask_ |= bool_to_bitmask(
                x0_ == x0_clamped_ && y1_ == y1_clamped_ && z1_ == z1_clamped_, 6);
            mask_ |= bool_to_bitmask(
                x1_ == x1_clamped_ && y1_ == y1_clamped_ && z1_ == z1_clamped_, 7);
        }

        // Perform trilinear interpolation for each of the 12 coefficients
        // Must be called once per coefficient !
        auto const computeAffine = [&](int c) -> scalar_t {
            scalar_t result{};

            if (enable_gridsize1_optimization_) {
                // For 1x1x1 grid, all coordinates are valid and point to the same cell
                result = mask_ != 0 ? grid_[g_idx_][c][0][0][0] : 0.F;
            } else {
                // Safe bounds-checked grid access
                auto const c000{
                    bitmask_to_bool(mask_, 0)
                        ? grid_[g_idx_][c][z0_clamped_][y0_clamped_][x0_clamped_]
                        : scalar_t{0}};
                auto const c001{
                    bitmask_to_bool(mask_, 1)
                        ? grid_[g_idx_][c][z0_clamped_][y0_clamped_][x1_clamped_]
                        : scalar_t{0}};
                auto const c010{
                    bitmask_to_bool(mask_, 2)
                        ? grid_[g_idx_][c][z0_clamped_][y1_clamped_][x0_clamped_]
                        : scalar_t{0}};
                auto const c011{
                    bitmask_to_bool(mask_, 3)
                        ? grid_[g_idx_][c][z0_clamped_][y1_clamped_][x1_clamped_]
                        : scalar_t{0}};
                auto const c100{
                    bitmask_to_bool(mask_, 4)
                        ? grid_[g_idx_][c][z1_clamped_][y0_clamped_][x0_clamped_]
                        : scalar_t{0}};
                auto const c101{
                    bitmask_to_bool(mask_, 5)
                        ? grid_[g_idx_][c][z1_clamped_][y0_clamped_][x1_clamped_]
                        : scalar_t{0}};
                auto const c110{
                    bitmask_to_bool(mask_, 6)
                        ? grid_[g_idx_][c][z1_clamped_][y1_clamped_][x0_clamped_]
                        : scalar_t{0}};
                auto const c111{
                    bitmask_to_bool(mask_, 7)
                        ? grid_[g_idx_][c][z1_clamped_][y1_clamped_][x1_clamped_]
                        : scalar_t{0}};

                // Trilinear interpolation
                auto const c00{c000 + wx_ * (c001 - c000)};
                auto const c01{c010 + wx_ * (c011 - c010)};
                auto const c10{c100 + wx_ * (c101 - c100)};
                auto const c11{c110 + wx_ * (c111 - c110)};

                auto const c0{c00 + wy_ * (c01 - c00)};
                auto const c1{c10 + wy_ * (c11 - c10)};

                result = c0 + wz_ * (c1 - c0);
            }

            if (c == 0 || c == 5 || c == 10) {
                // c >> 2 for c in [0;5;10] is equivalent to the map c == 0 => 0; c == 5
                // => 1; c == 10 => 2
                affine_diag_[c >> 2] = result;
            }

            return result;
        };

        // Apply affine transformation
        transformed_rgb_[0] = computeAffine(0) * r_ + computeAffine(1) * g_ +
                              computeAffine(2) * b_ + computeAffine(3);
        transformed_rgb_[1] = computeAffine(4) * r_ + computeAffine(5) * g_ +
                              computeAffine(6) * b_ + computeAffine(7);
        transformed_rgb_[2] = computeAffine(8) * r_ + computeAffine(9) * g_ +
                              computeAffine(10) * b_ + computeAffine(11);
    }

    __device__ void forward(
        torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>&
            output_rgb) {
        // Clamp to [0, 1] and write with bounds-checked access
#pragma unroll
        for (int i = 0; i < 3; i++) {
            output_rgb[idx_][i] = max(0.F, min(1.F, transformed_rgb_[i]));
        }
    }

    template <size_t threads_per_block = 256>
    __device__ void backward(
        torch::PackedTensorAccessor32<
            scalar_t, 2, torch::RestrictPtrTraits> const& grad_output,
        torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits>&
            grad_grid,
        torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>&
            grad_rgb,
        int first_grid_idx) {

        if (g_idx_ == NOVELVIEW_GRID_INDEX) {
            return;
        }

        /**
         * Step 1: Gradient of clamp(transformed_rgb, 0, 1)
         */
        std::array<scalar_t, 3> d_output_rgb_d_transformed_rgb;
#pragma unroll
        for (int i = 0; i < 3; i++) {
            auto val = transformed_rgb_[i];
            d_output_rgb_d_transformed_rgb[i] =
                val > 1.F || val < 0.F ? 0.F : grad_output[idx_][i];
        }

        /**
         * Step 2: Gradient of trilinear interpolation
         */

        auto const d_output_rgb_d_grid = [&](int c) -> scalar_t {
            switch (c) {
            case 0:
                return r_ * d_output_rgb_d_transformed_rgb[0];
            case 1:
                return g_ * d_output_rgb_d_transformed_rgb[0];
            case 2:
                return b_ * d_output_rgb_d_transformed_rgb[0];
            case 3:
                return d_output_rgb_d_transformed_rgb[0];
            case 4:
                return r_ * d_output_rgb_d_transformed_rgb[1];
            case 5:
                return g_ * d_output_rgb_d_transformed_rgb[1];
            case 6:
                return b_ * d_output_rgb_d_transformed_rgb[1];
            case 7:
                return d_output_rgb_d_transformed_rgb[1];
            case 8:
                return r_ * d_output_rgb_d_transformed_rgb[2];
            case 9:
                return g_ * d_output_rgb_d_transformed_rgb[2];
            case 10:
                return b_ * d_output_rgb_d_transformed_rgb[2];
            case 11:
                return d_output_rgb_d_transformed_rgb[2];
            default:
                return {};
            }
        };

        if (enable_gridsize1_optimization_) {
#pragma unroll
            for (int c = 0; c < 12; c++) {
                if (mask_) {
                    auto const dval = d_output_rgb_d_grid(c);
                    atomicAdd(&grad_grid[g_idx_][c][0][0][0], dval);
                }
            }
#pragma unroll
            for (int i = 0; i < 3; i++) {
                grad_rgb[idx_][i] = d_output_rgb_d_transformed_rgb[i] * affine_diag_[i];
            }
        } else {
            // Calculate interpolation weights for gradient accumulation
            auto const wx_c{1.F - wx_};
            auto const wy_c{1.F - wy_};
            auto const wz_c{1.F - wz_};

            auto const w000{wx_c * wy_c * wz_c};
            auto const w001{wx_ * wy_c * wz_c};
            auto const w010{wx_c * wy_ * wz_c};
            auto const w011{wx_ * wy_ * wz_c};
            auto const w100{wx_c * wy_c * wz_};
            auto const w101{wx_ * wy_c * wz_};
            auto const w110{wx_c * wy_ * wz_};
            auto const w111{wx_ * wy_ * wz_};

#pragma unroll
            for (int c = 0; c < 12; c++) {
                auto const d_output_rgb_d_grid_c = d_output_rgb_d_grid(c);

                auto const accum_grad_grid_at_corner =
                    [&](int z, int y, int x, int corner_idx, scalar_t value) -> void {
                    if (bitmask_to_bool(mask_, corner_idx) && fabs(value) > 0) {
                        atomicAdd(&grad_grid[g_idx_][c][z][y][x], value);
                    }
                };

                accum_grad_grid_at_corner(z0_clamped_, y0_clamped_, x0_clamped_, 0,
                                          d_output_rgb_d_grid_c * w000);
                accum_grad_grid_at_corner(z0_clamped_, y0_clamped_, x1_clamped_, 1,
                                          d_output_rgb_d_grid_c * w001);
                accum_grad_grid_at_corner(z0_clamped_, y1_clamped_, x0_clamped_, 2,
                                          d_output_rgb_d_grid_c * w010);
                accum_grad_grid_at_corner(z0_clamped_, y1_clamped_, x1_clamped_, 3,
                                          d_output_rgb_d_grid_c * w011);
                accum_grad_grid_at_corner(z1_clamped_, y0_clamped_, x0_clamped_, 4,
                                          d_output_rgb_d_grid_c * w100);
                accum_grad_grid_at_corner(z1_clamped_, y0_clamped_, x1_clamped_, 5,
                                          d_output_rgb_d_grid_c * w101);
                accum_grad_grid_at_corner(z1_clamped_, y1_clamped_, x0_clamped_, 6,
                                          d_output_rgb_d_grid_c * w110);
                accum_grad_grid_at_corner(z1_clamped_, y1_clamped_, x1_clamped_, 7,
                                          d_output_rgb_d_grid_c * w111);
            }

            /**
             * Step 3: Gradient for input RGB and coordinates
             */
            // Calculate derivatives of the affine transformation with respect to
            // inputs
            auto const d_affine_d_rgb = [&](int c, int dim) -> scalar_t {
                // Get the 8 grid vertices with safe access (now variables are always
                // initialized)
                auto const c000{
                    bitmask_to_bool(mask_, 0)
                        ? grid_[g_idx_][c][z0_clamped_][y0_clamped_][x0_clamped_]
                        : scalar_t{0}};
                auto const c001{
                    bitmask_to_bool(mask_, 1)
                        ? grid_[g_idx_][c][z0_clamped_][y0_clamped_][x1_clamped_]
                        : scalar_t{0}};
                auto const c010{
                    bitmask_to_bool(mask_, 2)
                        ? grid_[g_idx_][c][z0_clamped_][y1_clamped_][x0_clamped_]
                        : scalar_t{0}};
                auto const c011{
                    bitmask_to_bool(mask_, 3)
                        ? grid_[g_idx_][c][z0_clamped_][y1_clamped_][x1_clamped_]
                        : scalar_t{0}};
                auto const c100{
                    bitmask_to_bool(mask_, 4)
                        ? grid_[g_idx_][c][z1_clamped_][y0_clamped_][x0_clamped_]
                        : scalar_t{0}};
                auto const c101{
                    bitmask_to_bool(mask_, 5)
                        ? grid_[g_idx_][c][z1_clamped_][y0_clamped_][x1_clamped_]
                        : scalar_t{0}};
                auto const c110{
                    bitmask_to_bool(mask_, 6)
                        ? grid_[g_idx_][c][z1_clamped_][y1_clamped_][x0_clamped_]
                        : scalar_t{0}};
                auto const c111{
                    bitmask_to_bool(mask_, 7)
                        ? grid_[g_idx_][c][z1_clamped_][y1_clamped_][x1_clamped_]
                        : scalar_t{0}};

                auto const d_affine_d_nz{
                    ((c100 - c000) * wx_c * wy_c + (c101 - c001) * wx_ * wy_c +
                     (c110 - c010) * wx_c * wy_ + (c111 - c011) * wx_ * wy_)};

                // Calculate partial derivatives with respect to grid coordinates
                switch (dim) {
                case 0:
                    return d_affine_d_nz * R_WEIGHT * max_z_;
                case 1:
                    return d_affine_d_nz * G_WEIGHT * max_z_;
                case 2:
                    return d_affine_d_nz * B_WEIGHT * max_z_;
                default:
                    return {};
                }
            };

            // Calculate gradients for input RGB
            auto const d_transformed_rgb_d_rgb_c = [&](int dim) -> scalar_t {
                return affine_diag_[dim] + d_affine_d_rgb(4 * dim, dim) * r_ +
                       d_affine_d_rgb(4 * dim + 1, dim) * g_ +
                       d_affine_d_rgb(4 * dim + 2, dim) * b_ +
                       d_affine_d_rgb(4 * dim + 3, dim);
            };

            // Write RGB gradients
#pragma unroll
            for (int i = 0; i < 3; i++) {
                grad_rgb[idx_][i] =
                    d_output_rgb_d_transformed_rgb[i] * d_transformed_rgb_d_rgb_c(i);
            }
        }
    }
};

// CUDA kernel for the forward pass - template for both int16 and int32 grid
// indices
template <typename scalar_t, typename grid_idcs_t>
__global__ void bilateral_grid_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> grid,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        coords_xy,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> rgb,
    torch::PackedTensorAccessor32<grid_idcs_t, 1, torch::RestrictPtrTraits>
        grid_idcs,
    int batch_size, int depth, int height, int width,
    bool enable_gridsize1_optimization,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        transformed_rgb) {

    auto const idx{threadIdx.x + blockIdx.x * blockDim.x};

    if (idx < batch_size) {
        // Create bilateral grid operator and run forward pass
        assert(grid_idcs[idx] >= -1 && grid_idcs[idx] < grid.size(0));
        BilateralGridOp<scalar_t> op(grid, static_cast<int>(grid_idcs[idx]), idx,
                                     coords_xy, rgb, depth, height, width,
                                     enable_gridsize1_optimization);
        op.forward(transformed_rgb);
    }
}

// CUDA kernel for the backward pass - template for both int16 and int32 grid
// indices
template <size_t threads_per_block = 256, typename scalar_t,
          typename grid_idcs_t>
__global__ void bilateral_grid_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> grid,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        coords_xy,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> rgb,
    torch::PackedTensorAccessor32<grid_idcs_t, 1, torch::RestrictPtrTraits>
        grid_idcs,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        grad_output,
    int batch_size, int depth, int height, int width,
    bool enable_gridsize1_optimization,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits>
        grad_grid,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        grad_rgb) {

    auto const first_thread_idx{blockIdx.x * blockDim.x};
    auto const idx{threadIdx.x + first_thread_idx};

    if (idx < batch_size) {
        int const grid_idx{static_cast<int>(grid_idcs[idx])};
        int const first_grid_idx{static_cast<int>(grid_idcs[first_thread_idx])};

        // Create bilateral grid operator and run backward pass
        BilateralGridOp<scalar_t> op(grid, grid_idx, idx, coords_xy, rgb, depth,
                                     height, width, enable_gridsize1_optimization);
        op.template backward<threads_per_block>(grad_output, grad_grid, grad_rgb,
                                                first_grid_idx);
    }
}

void bilateral_grid_forward_cuda(torch::Tensor const& grid,
                                 torch::Tensor const& coords_xy,
                                 torch::Tensor const& rgb,
                                 torch::Tensor const& grid_idcs,
                                 torch::Tensor const& output,
                                 bool enable_gridsize1_optimization) {

    // grid indices can be camera id (int16) or frame id (int32)
    CHECK_INPUT_CONTIG(grid);
    CHECK_INPUT_CONTIG(coords_xy);
    CHECK_INPUT(rgb);
    CHECK_INPUT_CONTIG(grid_idcs);
    CHECK_INPUT(output);

    auto const batch_size{static_cast<int>(coords_xy.size(0))};
    auto const depth{static_cast<int>(grid.size(2))};
    auto const height{static_cast<int>(grid.size(3))};
    auto const width{static_cast<int>(grid.size(4))};

    if (rgb.strides() != output.strides()) {
        throw std::runtime_error("rgb and output must have the same strides");
    }

    if (rgb.size(0) != output.size(0) || rgb.size(1) != 3) {
        throw std::runtime_error(
            "rgb and output must have the same size and an inner dimension of 3");
    }

    // Validate grid_idcs bounds
    if (grid_idcs.scalar_type() != torch::kInt32 &&
        grid_idcs.scalar_type() != torch::kInt16) {
        throw std::runtime_error("grid_idcs must be int32 or int16");
    }

    // Launch the kernel
    auto const threads{256};
    auto const blocks{static_cast<int>(div_round_up(batch_size, threads))};

    c10::cuda::OptionalCUDAGuard device_guard(torch::device_of(grid));
    auto const stream{c10::cuda::getCurrentCUDAStream()};
    auto const is_grid_idc_int16{grid_idcs.scalar_type() == torch::kInt16};

    AT_DISPATCH_FLOATING_TYPES(
        grid.scalar_type(), "bilateral_grid_forward_kernel", ([&] {
            // Create packed accessors for safe tensor access

            auto const invoke_forward = [&](auto grid_idcs_accessor) {
                bilateral_grid_forward_kernel<<<blocks, threads, 0, stream>>>(
                    grid.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                    coords_xy
                        .packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    rgb.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    grid_idcs_accessor, batch_size, depth, height, width,
                    enable_gridsize1_optimization,
                    output
                        .packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
            };

            if (is_grid_idc_int16) {
                invoke_forward(
                    grid_idcs
                        .packed_accessor32<int16_t, 1, torch::RestrictPtrTraits>());
            } else {
                invoke_forward(
                    grid_idcs
                        .packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>());
            }
        }));
}

void bilateral_grid_backward_cuda(
    torch::Tensor const& grid, torch::Tensor const& coords_xy,
    torch::Tensor const& rgb, torch::Tensor const& grid_idcs,
    torch::Tensor const& grad_output, torch::Tensor const& grad_grid,
    torch::Tensor const& grad_rgb, bool enable_gridsize1_optimization) {

#ifdef BILATERALGRID_CHRONO
    auto const start_time{std::chrono::high_resolution_clock::now()};
#endif

    CHECK_INPUT_CONTIG(grid);
    CHECK_INPUT_CONTIG(coords_xy);
    CHECK_INPUT(rgb);
    CHECK_INPUT_CONTIG(grid_idcs);
    CHECK_INPUT(grad_output);
    CHECK_INPUT_CONTIG(grad_grid);
    CHECK_INPUT(grad_rgb);

    auto const batch_size{static_cast<int>(coords_xy.size(0))};
    auto const depth{static_cast<int>(grid.size(2))};
    auto const height{static_cast<int>(grid.size(3))};
    auto const width{static_cast<int>(grid.size(4))};

    if (rgb.size(0) != grad_output.size(0) || rgb.size(0) != grad_rgb.size(0) ||
        rgb.size(1) != 3 || rgb.size(1) != grad_output.size(1) ||
        rgb.size(1) != grad_rgb.size(1)) {
        throw std::runtime_error(
            "rgb, grad_rgb and grad_output must have the same size and an inner "
            "dimension of 3");
    }

    if (rgb.strides() != grad_output.strides() ||
        rgb.strides() != grad_rgb.strides()) {
        throw std::runtime_error(
            "rgb, grad_rgb and grad_output must have the same strides");
    }

    // Validate grid_idcs bounds
    if (grid_idcs.scalar_type() != torch::kInt32 &&
        grid_idcs.scalar_type() != torch::kInt16) {
        throw std::runtime_error("grid_idcs must be int32 or int16");
    }

    // Launch the kernel
    constexpr auto threads_per_block = 256;
    auto const blocks{
        static_cast<int>(div_round_up(batch_size, threads_per_block))};
    c10::cuda::OptionalCUDAGuard device_guard(torch::device_of(grid));
    auto const stream{c10::cuda::getCurrentCUDAStream()};
    auto const is_grid_idc_int16{grid_idcs.scalar_type() == torch::kInt16};

    AT_DISPATCH_FLOATING_TYPES(
        grid.scalar_type(), "bilateral_grid_backward_kernel", ([&] {
            // Create packed accessors for safe tensor access

            auto const invoke_backward = [&](auto grid_idcs_accessor) {
                bilateral_grid_backward_kernel<
                    threads_per_block><<<blocks, threads_per_block, 0, stream>>>(
                    grid.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                    coords_xy
                        .packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    rgb.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    grid_idcs_accessor,
                    grad_output
                        .packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    batch_size, depth, height, width, enable_gridsize1_optimization,
                    grad_grid
                        .packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                    grad_rgb
                        .packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
            };
            if (is_grid_idc_int16) {
                invoke_backward(
                    grid_idcs
                        .packed_accessor32<int16_t, 1, torch::RestrictPtrTraits>());
            } else {
                invoke_backward(
                    grid_idcs
                        .packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>());
            }
        }));

#ifdef BILATERALGRID_CHRONO
    cudaDeviceSynchronize();

    auto const end_time{std::chrono::high_resolution_clock::now()};
    auto const duration{std::chrono::duration_cast<std::chrono::microseconds>(
                            end_time - start_time)
                            .count()};
    std::cout << "Bilateral grid backward kernel time: "
              << static_cast<float>(duration) / 1000.F << "ms" << std::endl;
#endif
}
