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

#include <nrend/renderer/rendererImplementation.h>
#include <nrend/utils/cuda/cudaKernelResources.h>

namespace nrend {

class NREModel;

class NRERenderer final : public NRendererImplementation, public KernelDefinitionsProvider {

    std::unique_ptr<NREModel> m_modelPtr;              ///< model definition
    mutable CudaKernelResources m_cudaKernelResources; ///< cached (mutable) cuda kernel resources
    mutable uint32_t m_renderKernelIndex = 0;          ///< index of the kernel in the kernel definitions table

    KernelOpts m_optFlags = KernelOpts::None;

public:
    static constexpr char name[]                           = "nre-nerf";
    static constexpr ModelVersion::Number minVersionNumber = {0, 1, 249};
    static constexpr ModelVersion::Number maxVersionNumber = {999, 999, 999};

    NRERenderer(const nlohmann::json& rendererState, const Logger& logger);
    virtual ~NRERenderer();

    bool supportVersion(const ModelVersion& version,
                        RenderingParameters::RendererHints /*rendererHint*/,
                        RenderingParameters::OptFlags renderFlags) const override;

    Status initialize(const ModelVersion& version,
                      const nlohmann::json& modelState,
                      const RenderingParameters& renderParams) override;

    /// march the scene according to the given camera and composite the result into the given cuda arrays
    Status renderForward(const RenderParameters& params,
                         const tcnn::vec3* wordlRayOriginCudaPtr,
                         const tcnn::vec3* worldRayDirectionCudaPtr,
                         const TTimestamp* worldRayTimestampCudaPtr,
                         const tcnn::ivec2* sensorsIdsPtr,
                         const tcnn::ivec2* activeTrackInstancesIdsCudaPtr,
                         const TTrackInstancePose* activeTrackInstancesPoseCudaPtr,
                         const TTrackInstancePose* activeTrackInstancesEndPoseCudaPtr,
                         uint32_t* instanceIdCudaPtr,
                         float* worldHitDistanceCudaPtr,
                         tcnn::vec3* worldHitNormalCudaPtr,
                         tcnn::vec4* radianceDensityCudaPtr,
                         void* extendedFeaturesCudaPtr,
                         void* sceneDataCudaPtr,
                         ForwardContext** forwardContext,
                         int cudaDeviceIndex,
                         cudaStream_t cudaStream) const override;

public:
    Status registerKernelDefinitions(
        const KernelMemoryBindings& memoryBindings,
        const KernelSourceCodeTable& sourceCodeTable,
        const KernelDefinitionsTable& kernelDefinitionsTable,
        KernelOpts kernelOpts,
        const Logger& logger) const override;

    Status processKernelMemory(
        const KernelMemoryBindings& memoryBindings,
        KernelMemoryBindings::BindingsFlag bindingsFlag,
        const std::vector<std::unique_ptr<KernelMemory>>& memory,
        ProcessMemoryFlag processFlag,
        uint64_t processQueueHandle,
        const Logger& logger) const override;
};

} // namespace nrend
