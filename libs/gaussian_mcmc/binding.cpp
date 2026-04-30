// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include "utils.h"
#include <vector>

std::tuple<torch::Tensor, torch::Tensor> compute_relocation_tensor(
    const torch::Tensor opacities,
    const torch::Tensor scales,
    const torch::Tensor ratios,
    const torch::Tensor binoms,
    const int n_max,
    const float min_opacity) {
    CHECK_INPUT(opacities);
    CHECK_INPUT(scales);
    CHECK_INPUT(ratios);
    CHECK_INPUT(binoms);

    return compute_relocation_tensor_cu(opacities, scales, ratios, binoms, n_max, min_opacity);
}

torch::Tensor quat_scale_to_covariance(
    const torch::Tensor quats,
    const torch::Tensor scales,
    const std::string quaternion_format) {
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);

    // Check that the format is valid
    TORCH_CHECK(
        quaternion_format == "xyzw" || quaternion_format == "wxyz",
        "quaternion_format must be either 'xyzw' or 'wxyz', but got '",
        quaternion_format, "'");

    return quat_scale_to_covariance_cu(quats, scales, quaternion_format);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_relocation_tensor", &compute_relocation_tensor);
    m.def("quat_scale_to_covariance",
          &quat_scale_to_covariance_cu,
          py::arg("quats"),
          py::arg("scales"),
          py::arg("quaternion_format") = "xyzw");
}
