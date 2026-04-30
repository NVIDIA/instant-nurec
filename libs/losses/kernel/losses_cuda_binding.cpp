// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include "losses_cuda.h"

#include <ku/common_host.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Dispatch 0: Road Gaussians loss
    m.def("road_gaussians_forward", &road_gaussians_forward_cuda, "Road gaussians forward pass (CUDA)");
    m.def("road_gaussians_backward", &road_gaussians_backward_cuda, "Road gaussians backward pass (CUDA)");
    // Dispatch 1: Camera losses
    m.def("camera_losses_forward", &camera_losses_forward_cuda, "Camera losses forward pass (CUDA)");
    m.def("camera_losses_backward", &camera_losses_backward_cuda, "Camera losses backward pass (CUDA)");
    // Dispatch 2: LiDAR losses
    m.def("lidar_losses_forward", &lidar_losses_forward_cuda, "LiDAR losses forward pass (CUDA)");
    m.def("lidar_losses_backward", &lidar_losses_backward_cuda, "LiDAR losses backward pass (CUDA)");
    // Dispatch 3: Gaussian regularization losses
    m.def("gaussian_losses_forward", &gaussian_losses_forward_cuda, "Gaussian losses forward pass (CUDA)");
    m.def("gaussian_losses_backward", &gaussian_losses_backward_cuda, "Gaussian losses backward pass (CUDA)");
    // Dispatch 4: Background texture & grid losses
    m.def("background_grid_losses_forward", &background_grid_losses_forward_cuda, "Background/grid losses forward pass (CUDA)");
    m.def("background_grid_losses_backward", &background_grid_losses_backward_cuda, "Background/grid losses backward pass (CUDA)");
}
