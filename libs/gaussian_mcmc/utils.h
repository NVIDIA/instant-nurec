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

#include <ku/common_host.h>
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> compute_relocation_tensor_cu(
    const torch::Tensor opacities,
    const torch::Tensor scales,
    const torch::Tensor ratios,
    const torch::Tensor binoms,
    const int n_max,
    const float min_opacity);

torch::Tensor quat_scale_to_covariance_cu(
    const torch::Tensor quats,
    const torch::Tensor scales,
    const std::string quaternion_format);
