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

#include <nrend/renderer/renderParameters.h>
#include <tiny-cuda-nn/vec.h>

template <typename... TPostProcessings>
struct NREPostProcessings {
    template <typename TRay>
    static inline __device__ void eval(TRay& ray,
                                       const nrend::RenderParameters& renderParams,
                                       nrend::MemoryHandles trainParams,
                                       const tcnn::ivec2* sensorsIdsPtr) {
        (TPostProcessings::eval(ray, renderParams, trainParams, sensorsIdsPtr), ...);
    }
};

struct NRESkipPostProcessing {
    template <typename TRay>
    static inline __device__ void eval(TRay&,
                                       const nrend::RenderParameters&,
                                       nrend::MemoryHandles,
                                       const tcnn::ivec2*) {
    }
};