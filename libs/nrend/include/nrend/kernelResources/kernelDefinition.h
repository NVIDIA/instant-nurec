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

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace nrend {

struct CudaKernelOptions {
    std::vector<const char*> entryPointNames;
};

struct OptixPipelineOptions {
    const char* raygenEntryPointName       = "raygen";
    const char* missEntryPointName         = nullptr;
    const char* anyHitEntryPointName       = nullptr;
    const char* closestHitEntryPointName   = nullptr;
    const char* intersectionEntryPointName = nullptr;
    const char* parametersVariableName     = "params";
    uint32_t numPayloadValues              = 0;
    uint32_t numAttributeValues            = 0;
    enum Flags {
        None                       = 0,
        EnableTrianglePrimitives   = 1 << 0,
        EnableSpherePrimitives     = 1 << 1,
        EnableCustomPrimitives     = 1 << 2,
        EnableMotionBlur           = 1 << 3,
        AllowSingleGas             = 1 << 4,
        AllowSingleLevelInstancing = 1 << 5,
    };
    uint32_t flags         = Flags::None;
    uint32_t maxTraceDepth = 1;
};

struct KernelDefinition {
    enum Type {
        CudaKernel,
        OptixPipeline,
        Num
    } type;
    std::variant<CudaKernelOptions,
                 OptixPipelineOptions>
        options;
    std::string sourceCode;
};

struct KernelDefinitionsTable {

    uint32_t registerKernel(const KernelDefinition& kernelDefinition) const {
        m_kernelDefinitions.push_back(kernelDefinition);
        return m_kernelDefinitions.size() - 1;
    }

    inline const KernelDefinition& operator[](uint32_t index) const {
        return m_kernelDefinitions[index];
    }

    inline size_t size() const {
        return m_kernelDefinitions.size();
    }

private:
    mutable std::vector<KernelDefinition> m_kernelDefinitions;
};

} // namespace nrend
