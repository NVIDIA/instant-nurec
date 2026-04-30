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

#include <nrend/kernelResources/kernelDefinition.h>
#include <nrend/kernelResources/kernelMemory.h>
#include <nrend/kernelResources/kernelResources.h>

#include <memory>
#include <vector>

namespace nrend {

class RtcKernel;

class KernelResourcesProvider {

public:
    enum KernelOpts {
        None                          = 0,
        Differentiable                = 1 << 0,
        LinearRGB                     = 1 << 1,
        DisableFeatures               = 1 << 2,
        DisableExtendedFeatures       = 1 << 3,
        DisableSensorExtendedFeatures = 1 << 4,
        DisableNormals                = 1 << 5,
        DisableBackground             = 1 << 6,
        DisablePostProcessings        = 1 << 7,
        DisableRayGradients           = 1 << 8,
        EnableCumulatedWeights        = 1 << 9,
        EnableVisibility              = 1 << 10,
    };

    virtual Status registerKernelResources(
        const KernelMemoryBindings& memoryBindings,
        const KernelSourceCodeTable& sourceCodeTable,
        KernelOpts kernelOpts,
        const Logger& logger) const { return Status(); }

    enum ProcessMemoryFlag {
        Initialization,
        Update
    };

    virtual Status processKernelMemory(
        const KernelMemoryBindings& memoryBindings,
        KernelMemoryBindings::BindingsFlag bindingsFlag,
        const std::vector<std::unique_ptr<KernelMemory>>& memory,
        ProcessMemoryFlag processFlag,
        uint64_t processQueueHandle,
        const Logger& logger) const { return Status(); }
};

class KernelDefinitionsProvider : public KernelResourcesProvider {

public:
    virtual Status registerKernelDefinitions(
        const KernelMemoryBindings& memoryBindings,
        const KernelSourceCodeTable& sourceCodeTable,
        const KernelDefinitionsTable& kernelDefinitionsTable,
        KernelOpts kernelOpts,
        const Logger& logger) const { return Status(); }

    virtual Status configureCompiledKernels(
        const std::vector<std::unique_ptr<RtcKernel>>& compiledKernels,
        KernelOpts kernelOpts,
        const Logger& logger) const { return Status(); }
};

// Kernel binded transient memory : kernel memory to be modified per render call
struct KernelBindedTransientMemory {
    KernelMemoryBindings::BindingsFlag bindingsFlag;
    std::unique_ptr<IScopedKernelMemory> memory;
    int memoryBindingIndex;
};

} // namespace nrend
