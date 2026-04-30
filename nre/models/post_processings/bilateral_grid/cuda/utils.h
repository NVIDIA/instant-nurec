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

// Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

#include <torch/extension.h>

#define CHECK_INPUT(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor");

#define CHECK_INPUT_CONTIG(x) \
    CHECK_INPUT(x);           \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous");

// Forward declarations for CUDA kernels
void bilateral_grid_forward_cuda(torch::Tensor const& grid,
                                 torch::Tensor const& coords_xy,
                                 torch::Tensor const& rgb,
                                 torch::Tensor const& grid_idx,
                                 torch::Tensor const& output,
                                 bool const enable_gridsize1_optimization);

void bilateral_grid_backward_cuda(
    torch::Tensor const& grid, torch::Tensor const& coords_xy,
    torch::Tensor const& rgb, torch::Tensor const& grid_idx,
    torch::Tensor const& grad_output, torch::Tensor const& grad_grid,
    torch::Tensor const& grad_rgb, bool const enable_gridsize1_optimization);

inline uint32_t div_round_up(uint32_t val, uint32_t divisor) {
    return (val + divisor - 1) / divisor;
}