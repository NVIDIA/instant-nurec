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

#include <tiny-cuda-nn/vec.h>

template <int InputDim>
struct NRESkipGeometry {
    static inline __device__ float eval(const tcnn::vec<InputDim>& inputs, float alphaInvarianceOffset) {
        return __expf(inputs[0] + alphaInvarianceOffset);
    }
};
