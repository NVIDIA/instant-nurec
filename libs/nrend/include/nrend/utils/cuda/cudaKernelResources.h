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

#include <nrend/kernelResources/kernelResourcesProvider.h>
#include <nrend/modelParameters.h>
#include <nrend/renderer/renderParameters.h>
#include <nrend/utils/cuda/cudaBuffer.h>
#include <nrend/utils/cuda/cudaRtcKernel.h>
#include <nrend/utils/cuda/cudaTexture.h>
#include <nrend/utils/optix/optixRtcPipeline.h>
#include <nrend/utils/spinMutex.h>

#include <memory>

namespace nrend {

class CudaKernelResources final {

public:
    CudaKernelResources();
    virtual ~CudaKernelResources() = default;

    struct Resources final {
        inline KernelMemory* memory(
            size_t bindingIndex,
            KernelMemoryBindings::BindingsFlag bindingFlag = KernelMemoryBindings::Parameters) const {
            return bindingIndex < bindedMemory[bindingFlag].size() ? bindedMemory[bindingFlag][bindingIndex].get() : nullptr;
        }

        std::array<KernelMemoryPtrVec, KernelMemoryBindings::BindingsFlag::Num> bindedMemory;
        std::vector<std::unique_ptr<nrend::RtcKernel>> compiledKernels;
        std::array<CudaBuffer, KernelMemoryBindings::BindingsFlag::Num> bindedMemoryHandles;
    };

    using ResourcesPtr = std::shared_ptr<Resources>;

    class StreamLock final {
        cudaStream_t m_stream;
        ResourcesPtr m_resourcesPtr;

    public:
        StreamLock(cudaStream_t, ResourcesPtr);
        ~StreamLock();

        inline const uint64_t* memoryHandlesPtr(KernelMemoryBindings::BindingsFlag bindingFlag) const {
            return m_resourcesPtr ? reinterpret_cast<const uint64_t*>(m_resourcesPtr->bindedMemoryHandles[static_cast<int>(bindingFlag)].data()) : nullptr;
        }

        inline const KernelMemoryPtrVec* memoryPtrVec(KernelMemoryBindings::BindingsFlag bindingFlag) const {
            if (!m_resourcesPtr) {
                return nullptr;
            }
            const size_t bindingFlagIndex = static_cast<size_t>(bindingFlag);
            return &m_resourcesPtr->bindedMemory[bindingFlagIndex];
        }

        template <typename... Types>
        inline Status launchCudaKernel(uint32_t kernelIndex, uint32_t entryPointIndex, dim3 blocks, dim3 threads, uint32_t shmemSize, cudaStream_t stream, const Logger& logger, Types&&... args) {
            RETURN_ERROR_IF(!m_resourcesPtr || kernelIndex >= m_resourcesPtr->compiledKernels.size(), logger, ErrorCode::InvalidResource, "CudaKernelResources : invalid kernel resource");
            CudaRtcKernel* cudaRtcKernelPtr = dynamic_cast<CudaRtcKernel*>(m_resourcesPtr->compiledKernels[kernelIndex].get());
            RETURN_ERROR_IF(!cudaRtcKernelPtr, logger, ErrorCode::InvalidResource, "CudaKernelResources : invalid cuda kernel");
            return cudaRtcKernelPtr->launch(
                entryPointIndex,
                blocks,
                threads,
                shmemSize,
                stream,
                logger,
                std::forward<Types>(args)...);
        }

        template <typename ParametersType>
        inline Status launchOptixPipeline(uint32_t kernelIndex, dim3 blocks, cudaStream_t stream, const Logger& logger, ParametersType* parametersDevicePtr) {
            RETURN_ERROR_IF(!m_resourcesPtr || kernelIndex >= m_resourcesPtr->compiledKernels.size(), logger, ErrorCode::InvalidResource, "CudaKernelResources : invalid kernel resource");
            OptixRtcPipeline* optixPipelinePtr = dynamic_cast<OptixRtcPipeline*>(m_resourcesPtr->compiledKernels[kernelIndex].get());
            RETURN_ERROR_IF(!optixPipelinePtr, logger, ErrorCode::InvalidResource, "CudaKernelResources : invalid optix pipeline");
            return optixPipelinePtr->launch(
                blocks,
                stream,
                logger,
                parametersDevicePtr);
        }

        inline ResourcesPtr resourcesPtr() const {
            return m_resourcesPtr;
        }

        inline int resourcesUseCount() const {
            return m_resourcesPtr.use_count();
        }
    };

    std::unique_ptr<StreamLock> prepare(const KernelDefinitionsProvider* kernelProviderPtr,
                                        KernelResourcesProvider::KernelOpts kernelOpts,
                                        cudaStream_t stream,
                                        int cudaDeviceId,
                                        Logger logger);

    std::unique_ptr<StreamLock> update(
        const KernelDefinitionsProvider* kernelProviderPtr,
        KernelResourcesProvider::KernelOpts kernelOpts,
        cudaStream_t stream,
        int cudaDeviceId,
        const NamedParameterDefinitionsSpan& namedParametersDefinitions,
        KernelMemoryBindings::BindingsFlag memoryFlag,
        bool attachBuffer,
        Logger logger);

    Status detach(
        const KernelDefinitionsProvider* kernelProviderPtr,
        cudaStream_t stream,
        int cudaDeviceId,
        KernelMemoryBindings::BindingsFlag memoryFlag,
        bool copy,
        Logger logger);

private:
    Status checkCudaDeviceId(int cudaDeviceId, Logger logger);

    Status initializeKernelResources(const KernelDefinitionsProvider* kernelProviderPtr,
                                     KernelResourcesProvider::KernelOpts kernelOpts,
                                     bool skipUpdateBindedMemoryBuffers,
                                     cudaStream_t stream,
                                     int cudaDeviceId,
                                     const Logger& logger);

    Status updateKernelResources(const KernelDefinitionsProvider* kernelProviderPtr,
                                 KernelResourcesProvider::KernelOpts kernelOpts,
                                 cudaStream_t stream,
                                 int cudaDeviceId,
                                 const NamedParameterDefinitionsSpan& namedParametersDefinitions,
                                 KernelMemoryBindings::BindingsFlag memoryFlag,
                                 bool attachBuffer,
                                 const Logger& logger);

    Status updateBindedMemoryBuffers(ResourcesPtr resourcesPtr,
                                     const KernelDefinitionsProvider* kernelProviderPtr,
                                     KernelResourcesProvider::ProcessMemoryFlag updateFlag,
                                     KernelMemoryBindings::BindingsFlag memoryFlag,
                                     cudaStream_t stream,
                                     int cudaDeviceId,
                                     const Logger& logger);

private:
    SpinMutex m_mutex;

    bool m_kernelInitialized = false;
    KernelSourceCodeTable m_sourceCodeTable;
    KernelDefinitionsTable m_kernelDefinitionsTable;
    KernelMemoryBindings m_memoryBindings;

    std::vector<ResourcesPtr> m_perDeviceResources;
};

} // namespace nrend
