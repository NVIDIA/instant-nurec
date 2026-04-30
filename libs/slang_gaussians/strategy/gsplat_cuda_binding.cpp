// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include "gsplat_cuda.h"

#include <ku/common_host.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("update_gradient_buffers", &update_gradient_buffers_cuda, "Update gradient accumulation buffers for GSplat densification (CUDA)");
}
