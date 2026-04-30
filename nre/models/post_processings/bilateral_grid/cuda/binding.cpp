// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include "utils.h"

#include <ku/common_host.h>
#include <vector>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bilateral_grid_forward", &bilateral_grid_forward_cuda,
          "Apply bilateral grid transformation (CUDA)");
    m.def("bilateral_grid_backward", &bilateral_grid_backward_cuda,
          "Backward pass for bilateral_grid_forward (CUDA)");
}