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

#include <nrend/models/nreModel.h>

namespace nrend {

class INREParticlesModel {

public:
    INREParticlesModel()          = default;
    virtual ~INREParticlesModel() = default;

    virtual bool isDynamic() const        = 0;
    virtual uint32_t numParticles() const = 0;

    virtual Status prepareParticlesParameters(uint32_t numActiveTrackInstances,
                                              const tcnn::ivec2* activeTrackInstancesIdsCudaPtr,
                                              const KernelMemoryPtrVec& parameterMemoryPtrVec,
                                              std::vector<KernelBindedTransientMemory>& transientParameters,
                                              uint32_t& numParticles,
                                              uint32_t& numParticlesToPreProcess,
                                              cudaStream_t cudaStream,
                                              const Logger& logger) const = 0;
};

} // namespace nrend