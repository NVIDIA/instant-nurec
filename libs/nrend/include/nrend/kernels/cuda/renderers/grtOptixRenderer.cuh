// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

__global__ void preProcessParticles(uint32_t numParticles,
                                    uint32_t numActiveTrackInstances,
                                    const tcnn::ivec2* __restrict__ activeTrackInstancesIdsCudaPtr,
                                    nrend::TTimestamp timestamp,
                                    const nrend::TTrackInstancePose* __restrict__ activeTrackInstancesStartPoseCudaPtr,
                                    const nrend::TTrackInstancePose* __restrict__ activeTrackInstancesEndPoseCudaPtr,
                                    const uint64_t* __restrict__ parameterMemoryHandles) {
    TGRTModel::preprocess(numParticles,
                          numActiveTrackInstances,
                          activeTrackInstancesIdsCudaPtr,
                          timestamp,
                          activeTrackInstancesStartPoseCudaPtr,
                          activeTrackInstancesEndPoseCudaPtr,
                          {parameterMemoryHandles});
}

__global__ void buildParticlePrimitives(uint32_t numParticles,
                                        void* __restrict__ particlesPrimitiveData,
                                        void* __restrict__ particlesPrimitiveExtendedData,
                                        OptixTraversableHandle asHandle,
                                        const uint64_t* __restrict__ parameterMemoryHandles) {
    TGRTParticlePrimitive::eval<TGRTModel::Particles>(numParticles,
                                                      particlesPrimitiveData,
                                                      particlesPrimitiveExtendedData,
                                                      asHandle,
                                                      {parameterMemoryHandles});
}
