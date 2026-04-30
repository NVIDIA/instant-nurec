// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <optix.h>

#include <nrend/renderer/renderParameters.h>

namespace nrend {
struct GRTOptixRenderParameters {
    OptixTraversableHandle traversableHandle;
    RenderParametersArray renderParametersArray;
    const tcnn::vec3* __restrict__ wordlRayOriginPtr;
    const tcnn::vec3* __restrict__ worldRayDirectionPtr;
    const nrend::TTimestamp* __restrict__ worldRayTimestampCudaPtr;
    const tcnn::ivec2* __restrict__ sensorsIdsPtr;
    uint32_t* __restrict__ instanceIdPtr;
    float* __restrict__ worldHitDistancePtr;
    tcnn::vec2* __restrict__ worldHitDistanceBoundsPtr;
    tcnn::vec3* __restrict__ worldHitNormalPtr;
    void* __restrict__ radianceDensityPtr;
    void* __restrict__ extendedFeaturesPtr;
    void* __restrict__ sceneDataPtr;
    const uint64_t* __restrict__ parameterMemoryHandles;
};

struct GRTOptixRenderBackwardParameters {
    OptixTraversableHandle traversableHandle;
    RenderParametersArray renderParametersArray;
    const tcnn::vec3* __restrict__ wordlRayOriginPtr;
    const tcnn::vec3* __restrict__ worldRayDirectionPtr;
    const nrend::TTimestamp* __restrict__ worldRayTimestampCudaPtr;
    const tcnn::ivec2* __restrict__ sensorsIdsPtr;
    const uint32_t* __restrict__ instanceIdPtr;
    const float* __restrict__ worldHitDistancePtr;
    const tcnn::vec3* __restrict__ worldHitNormalPtr;
    const tcnn::vec2* __restrict__ worldHitDistanceBoundsPtr;
    const void* __restrict__ radianceDensityPtr;
    const void* __restrict__ extendedFeaturesPtr;
    const uint64_t* __restrict__ parameterMemoryHandles;
    const float* __restrict__ worldHitDistanceGradientPtr;
    const tcnn::vec3* __restrict__ worldHitNormalGradientPtr;
    const void* __restrict__ radianceDensityGradientPtr;
    const void* __restrict__ extendedFeaturesGradientPtr;
    tcnn::vec3* __restrict__ wordlRayOriginGradientPtr;
    tcnn::vec3* __restrict__ worldRayDirectionGradientPtr;
    const uint64_t* __restrict__ parameterGradientMemoryHandles;
};

} // namespace nrend