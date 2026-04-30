// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include "pose_calib_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &pose_calib_forward_cuda, "Pose calibration forward (CUDA)");
    m.def("backward", &pose_calib_backward_cuda, "Pose calibration backward (CUDA)");
}
