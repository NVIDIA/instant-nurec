// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include <torch/extension.h>

#include "knn.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knn_check_version", &KnnCheckVersion);
    m.def("knn_points_idx", &KNearestNeighborIdx);
    m.def("knn_points_backward", &KNearestNeighborBackward);
}
