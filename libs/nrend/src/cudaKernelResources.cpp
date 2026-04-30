// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include <nrend/kernelResources/rtcKernelConfig.h>
#include <nrend/utils/cuda/cudaKernelResources.h>
#include <nrend/utils/slang/slangRtcKernel.h>

namespace {

void freeSharedPtr(void* sharedPtr) {
    if (std::shared_ptr<nrend::CudaKernelResources::Resources>* ptr =
            reinterpret_cast<std::shared_ptr<nrend::CudaKernelResources::Resources>*>(sharedPtr)) {
        delete ptr; ///< release the ref-counted resource
    }
}

inline nrend::KernelMemory* createCudaMemory(nrend::KernelMemoryBindings::MemoryType memoryType, cudaStream_t stream) {
    if (memoryType == nrend::KernelMemoryType::Buffer) {
        return static_cast<nrend::KernelMemory*>(new nrend::CudaBuffer());
    }
    if ((memoryType == nrend::KernelMemoryType::Texture2D_RED_32F) ||
        (memoryType == nrend::KernelMemoryType::Texture2D_RG_32F) ||
        (memoryType == nrend::KernelMemoryType::Texture2D_RGBA_32F)) {
        return static_cast<nrend::KernelMemory*>(new nrend::CudaTexture2D());
    }
    if ((memoryType == nrend::KernelMemoryType::TextureCubeMap_RED_32F) ||
        (memoryType == nrend::KernelMemoryType::TextureCubeMap_RG_32F) ||
        (memoryType == nrend::KernelMemoryType::TextureCubeMap_RGBA_32F)) {
        return static_cast<nrend::KernelMemory*>(new nrend::CudaTextureCubeMap());
    }
    return nullptr;
}

// Remove part of slang generated cuda to prevent its ministd to be redefined
// TODO : log a feature request to Slang team
inline void cleanupSlangGeneratedCudaCode(std::string& code, const nrend::Logger& logger) {
    const std::string slangMiniStd = R"(
#if SLANG_CUDA_RTC

typedef signed char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long long int64_t;
typedef ptrdiff_t intptr_t;

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef size_t uintptr_t;

typedef long long longlong;
typedef unsigned long long ulonglong;

#else

// When not using NVRTC, match the platform's int64_t definition for signed type
// On Linux: int64_t is 'long', on Windows: int64_t is 'long long'
typedef int64_t longlong;
// ulonglong must remain 'unsigned long long' to match CUDA's atomic operations
typedef unsigned long long ulonglong;

#endif
)";

    const std::string slangCleanedMiniStd = R"(
#if SLANG_CUDA_RTC
typedef ptrdiff_t intptr_t;
typedef size_t uintptr_t;

typedef long long longlong;
typedef unsigned long long ulonglong;

#else

typedef int64_t longlong;
typedef unsigned long long ulonglong;

#endif
)";

    const std::tuple<std::string, std::string, std::string> replacements[] = {
        {"mini std", slangMiniStd, slangCleanedMiniStd},
        {"wave min definition", "SLANG_WAVE_MIN_SPEC(int8_t, (int8_t)0x7F)", ""},
        {"wave max definition", "SLANG_WAVE_MAX_SPEC(int8_t, (int8_t)0x80)", ""},
    };

    for (const auto& [name, original, replacement] : replacements) {
        const auto pos = code.find(original);
        if (pos != std::string::npos) {
            code.replace(pos, original.length(), replacement);
        } else {
            LOG_WARN(logger, "CudaKernelResources: generated cuda code does not contains expected slang %s.", name.c_str());
        }
    }
}

} // namespace

nrend::CudaKernelResources::CudaKernelResources() {
    int numCudaDevices = 0;
    cudaGetDeviceCount(&numCudaDevices);
    m_perDeviceResources.resize(numCudaDevices);
}

nrend::CudaKernelResources::StreamLock::StreamLock(cudaStream_t stream, std::shared_ptr<Resources> resources)
    : m_stream(stream), m_resourcesPtr(resources) {
}

nrend::CudaKernelResources::StreamLock::~StreamLock() {
    std::shared_ptr<Resources>* ptr = new std::shared_ptr<Resources>(m_resourcesPtr);
    cudaError_t err                 = cudaLaunchHostFunc(m_stream, freeSharedPtr, ptr);
    if (err != cudaSuccess) {
        delete ptr; /// NB : this may leads to dangling GPU resources
    }
}

nrend::Status nrend::CudaKernelResources::checkCudaDeviceId(int cudaDeviceId, Logger logger) {
    if ((cudaDeviceId < 0) || (cudaDeviceId >= static_cast<int>(m_perDeviceResources.size()))) {
        RETURN_ERROR(logger, ErrorCode::BadInput, "CudaKernelResources: invalid device id [%d/%d]", cudaDeviceId, static_cast<int>(m_perDeviceResources.size()));
    }
    return Status();
}

nrend::Status nrend::CudaKernelResources::initializeKernelResources(
    const KernelDefinitionsProvider* kernelProviderPtr,
    KernelResourcesProvider::KernelOpts kernelOpts,
    bool skipUpdateBindedMemoryBuffers,
    cudaStream_t stream,
    int cudaDeviceId,
    const Logger& logger) {

    CHECK_STATUS_RETURN(checkCudaDeviceId(cudaDeviceId, logger));

    if (m_perDeviceResources[cudaDeviceId].get()) {
        return Status();
    }

    if (!m_kernelInitialized) {
        CHECK_STATUS_RETURN(kernelProviderPtr->registerKernelDefinitions(
            m_memoryBindings,
            m_sourceCodeTable,
            m_kernelDefinitionsTable,
            kernelOpts,
            logger));
        m_kernelInitialized = true;
    }

    // concatenate code into a single cuda string
    std::string kernelCudaSourceCode;

    // compile slang code as cuda
    const std::string kernelSlangSourceCode = m_sourceCodeTable.sourceCode(KernelSourceCodeTable::Idiom::Slang);
    if (!kernelSlangSourceCode.empty()) {
        CHECK_STATUS_RETURN(SlangRtcKernel::generateIntermediateTarget(
            SlangRtcKernel::IntermediateTarget::Cuda,
            kernelSlangSourceCode,
            RtcKernelConfig::includeDirectories(),
            RtcKernelConfig::cacheDirectory(),
            RtcKernelConfig::extraIncludes(),
            kernelCudaSourceCode,
            logger));
        cleanupSlangGeneratedCudaCode(kernelCudaSourceCode, logger);
    }
    kernelCudaSourceCode += m_sourceCodeTable.sourceCode(KernelSourceCodeTable::Idiom::Cuda);

    // create the resource
    std::shared_ptr<Resources> resourcesPtr = std::make_shared<Resources>();
    // compile the kernels
    Status status;
    for (size_t i = 0; i < m_kernelDefinitionsTable.size(); ++i) {
        const auto& kernelDefinition = m_kernelDefinitionsTable[i];
        if (kernelDefinition.type == KernelDefinition::Type::CudaKernel) {
            resourcesPtr->compiledKernels.push_back(std::make_unique<nrend::CudaRtcKernel>(
                std::get<CudaKernelOptions>(kernelDefinition.options),
                kernelCudaSourceCode + kernelDefinition.sourceCode,
                RtcKernelConfig::includeDirectories(),
                RtcKernelConfig::cacheDirectory(),
                RtcKernelConfig::extraIncludes(),
                logger,
                status));
        } else if (kernelDefinition.type == KernelDefinition::Type::OptixPipeline) {
            resourcesPtr->compiledKernels.push_back(std::make_unique<nrend::OptixRtcPipeline>(
                cudaDeviceId,
                std::get<OptixPipelineOptions>(kernelDefinition.options),
                kernelCudaSourceCode + kernelDefinition.sourceCode,
                RtcKernelConfig::includeDirectories(),
                RtcKernelConfig::cacheDirectory(),
                RtcKernelConfig::extraIncludes(),
                stream,
                logger,
                status));
        } else {
            RETURN_ERROR(logger, ErrorCode::BadInput, "CudaKernelResources : invalid kernel type %d.", static_cast<int>(kernelDefinition.type));
        }
        if (!status) {
            return status;
        }
    }

    // Custom post-compile configuration
    status = kernelProviderPtr->configureCompiledKernels(resourcesPtr->compiledKernels, kernelOpts, logger);
    if (!status) {
        return status;
    }

    // create all memory resources
    for (int memoryFlag = 0; memoryFlag < KernelMemoryBindings::BindingsFlag::Num; ++memoryFlag) {

        const int numBindedMemory = m_memoryBindings.numRegisteredMemory(static_cast<KernelMemoryBindings::BindingsFlag>(memoryFlag));
        resourcesPtr->bindedMemory[memoryFlag].resize(numBindedMemory);

        for (int bindedMemoryIndex = 0; bindedMemoryIndex < numBindedMemory; ++bindedMemoryIndex) {
            resourcesPtr->bindedMemory[memoryFlag][bindedMemoryIndex].reset(
                createCudaMemory(m_memoryBindings.registeredMemoryType(static_cast<KernelMemoryBindings::BindingsFlag>(memoryFlag), bindedMemoryIndex),
                                 stream));

            if (!resourcesPtr->bindedMemory[memoryFlag][bindedMemoryIndex]) {
                RETURN_ERROR(logger, ErrorCode::InvalidResource,
                             "CudaKernelResources : cannot create cuda memory %s on device %d.",
                             m_memoryBindings.registeredMemoryName(static_cast<KernelMemoryBindings::BindingsFlag>(memoryFlag), bindedMemoryIndex).c_str(),
                             cudaDeviceId);
            }
        }
    }

    // update the parameters memory
    if (!skipUpdateBindedMemoryBuffers) {
        CHECK_STATUS_RETURN(updateBindedMemoryBuffers(resourcesPtr,
                                                      kernelProviderPtr,
                                                      KernelResourcesProvider::Initialization,
                                                      KernelMemoryBindings::Parameters,
                                                      stream,
                                                      cudaDeviceId,
                                                      logger));
    }

    // set the resource
    m_perDeviceResources[cudaDeviceId] = resourcesPtr;
    LOG_INFO(logger, "CudaKernelResources : cuda resources created on device %d.", cudaDeviceId);

    return Status();
}

nrend::Status nrend::CudaKernelResources::updateKernelResources(
    const KernelDefinitionsProvider* kernelProviderPtr,
    KernelResourcesProvider::KernelOpts kernelOpts,
    cudaStream_t stream,
    int cudaDeviceId,
    const NamedParameterDefinitionsSpan& namedParametersDefinitions,
    KernelMemoryBindings::BindingsFlag memoryFlag,
    bool attachMemory,
    const Logger& logger) {

    CHECK_STATUS_RETURN(initializeKernelResources(kernelProviderPtr, kernelOpts, true, stream, cudaDeviceId, logger));
    ResourcesPtr resourcesPtr = m_perDeviceResources[cudaDeviceId];

    for (size_t i = 0; i < namedParametersDefinitions.size; ++i) {

        const NamedParameterDefinition& namedParameterDefinition = namedParametersDefinitions.data[i];

        if (namedParameterDefinition.definition.type == ParameterDefinition::Value) {
            if (memoryFlag != KernelMemoryBindings::Parameters) {
                RETURN_ERROR(logger, ErrorCode::BadInput, "CudaKernelResources : cannot update model parameter value %s, unsupported memory flag %d.",
                             namedParameterDefinition.name, static_cast<int>(memoryFlag));
            }
            const int parameterValueIndex = m_memoryBindings.registeredValueIndex(namedParameterDefinition.name);
            if (parameterValueIndex == KernelMemoryBindings::InvalidMemoryIndex) {
                RETURN_ERROR(logger, ErrorCode::BadInput, "CudaKernelResources : cannot update model parameter value, parameter %s not registered.",
                             namedParameterDefinition.name);
            }
            const auto valueBinding = m_memoryBindings.registeredValueBinding(parameterValueIndex);
            if (valueBinding.size != namedParameterDefinition.definition.size) {
                RETURN_ERROR(logger, ErrorCode::BadInput, "CudaKernelResources : cannot update model parameter value %s, incorrect size [%d/%d].",
                             namedParameterDefinition.name, static_cast<int>(namedParameterDefinition.definition.size), static_cast<int>(valueBinding.size));
            }
            CHECK_STATUS_RETURN(m_memoryBindings.setRegisteredValue(parameterValueIndex, static_cast<const char*>(namedParameterDefinition.definition.dataPtr), logger));
        }

        else {
            const int parameterMemoryIndex = m_memoryBindings.registeredMemoryIndex(memoryFlag, namedParameterDefinition.name);
            if (parameterMemoryIndex == KernelMemoryBindings::InvalidMemoryIndex) {
                RETURN_ERROR(logger, ErrorCode::BadInput, "CudaKernelResources : cannot update model parameter memory, parameter %s not registered.",
                             namedParameterDefinition.name);
            }

            auto memoryType = m_memoryBindings.registeredMemoryType(memoryFlag, parameterMemoryIndex);
            if (memoryType != KernelMemoryBindings::MemoryType::Buffer) {
                RETURN_ERROR(logger, ErrorCode::NotImplemented,
                             "CudaKernelResources : cannot update model parameter %s, only MemoryType::Buffer is supported, got %d.",
                             namedParameterDefinition.name, static_cast<int>(memoryType));
            }

            auto* memoryPtr = resourcesPtr->bindedMemory[static_cast<int>(memoryFlag)][parameterMemoryIndex]->as<CudaBuffer>();
            if (memoryPtr) {
                CHECK_STATUS_RETURN(memoryPtr->setFromDevice(
                    namedParameterDefinition.definition.dataPtr,
                    namedParameterDefinition.definition.size,
                    attachMemory,
                    reinterpret_cast<uint64_t>(stream),
                    logger));
            } else {
                RETURN_ERROR(logger, ErrorCode::InvalidResource,
                             "CudaKernelResources : cannot update model parameter %s, memory is not a cuda buffer.",
                             namedParameterDefinition.name);
            }
        }
    }

    return updateBindedMemoryBuffers(resourcesPtr,
                                     kernelProviderPtr,
                                     KernelResourcesProvider::Update,
                                     memoryFlag,
                                     stream,
                                     cudaDeviceId,
                                     logger);
}

nrend::Status nrend::CudaKernelResources::updateBindedMemoryBuffers(
    ResourcesPtr resourcesPtr,
    const KernelDefinitionsProvider* kernelProviderPtr,
    KernelResourcesProvider::ProcessMemoryFlag updateFlag,
    KernelMemoryBindings::BindingsFlag memoryFlag,
    cudaStream_t stream,
    int cudaDeviceId,
    const Logger& logger) {

    if (!resourcesPtr) {
        RETURN_ERROR(
            logger,
            ErrorCode::InvalidResource,
            "CudaKernelResources : cannot update binded buffer, resources have not been successfully initialized on device %d.",
            cudaDeviceId);
    }

    // update kernel memory
    const int parametersValueIndex = m_memoryBindings.registeredValuesMemoryIndex(memoryFlag);
    if (parametersValueIndex != KernelMemoryBindings::InvalidMemoryIndex) {
        CHECK_STATUS_RETURN(resourcesPtr->bindedMemory[static_cast<int>(memoryFlag)][parametersValueIndex]->setFromHost(
            m_memoryBindings.parametersValueBuffer().data(),
            m_memoryBindings.parametersValueBuffer().size(),
            reinterpret_cast<uint64_t>(stream),
            logger));
    }

    // parameters buffer
    CHECK_STATUS_RETURN(kernelProviderPtr->processKernelMemory(
        m_memoryBindings,
        memoryFlag,
        resourcesPtr->bindedMemory[static_cast<int>(memoryFlag)],
        updateFlag,
        reinterpret_cast<uint64_t>(stream),
        logger));

    // setup the handles buffer
    const int numBindedMemory = m_memoryBindings.numRegisteredMemory(memoryFlag);
    std::vector<uint64_t> bindedMemoryHostHandles(numBindedMemory);
    for (int bindedMemoryIndex = 0; bindedMemoryIndex < numBindedMemory; ++bindedMemoryIndex) {
        auto memoryPtr = resourcesPtr->bindedMemory[static_cast<int>(memoryFlag)][bindedMemoryIndex].get();
        if (memoryPtr) {
            bindedMemoryHostHandles[bindedMemoryIndex] = resourcesPtr->bindedMemory[static_cast<int>(memoryFlag)][bindedMemoryIndex]->handle();
        } else {
            RETURN_ERROR(logger, ErrorCode::InvalidResource,
                         "CudaKernelResources : cannot update binded buffer %d@%d (%s), resources have not been successfully initialized on device %d.",
                         bindedMemoryIndex, memoryFlag,
                         m_memoryBindings.registeredMemoryName(memoryFlag, bindedMemoryIndex).c_str(),
                         cudaDeviceId);
        }
    }
    CHECK_STATUS_RETURN(resourcesPtr->bindedMemoryHandles[static_cast<int>(memoryFlag)].setFromHost(
        bindedMemoryHostHandles.data(),
        bindedMemoryHostHandles.size() * sizeof(uint64_t),
        reinterpret_cast<uint64_t>(stream),
        logger));

    return Status();
}

std::unique_ptr<nrend::CudaKernelResources::StreamLock> nrend::CudaKernelResources::prepare(
    const KernelDefinitionsProvider* kernelProviderPtr,
    KernelResourcesProvider::KernelOpts kernelOpts,
    cudaStream_t stream,
    int cudaDeviceId,
    Logger logger) {

    std::unique_lock<SpinMutex> lock(m_mutex);
    Status status = initializeKernelResources(kernelProviderPtr, kernelOpts, false, stream, cudaDeviceId, logger);
    if (!status) {
        return std::unique_ptr<nrend::CudaKernelResources::StreamLock>();
    }

    return std::unique_ptr<CudaKernelResources::StreamLock>(
        m_perDeviceResources[cudaDeviceId] ? new StreamLock(stream, m_perDeviceResources[cudaDeviceId]) : nullptr);
}

std::unique_ptr<nrend::CudaKernelResources::StreamLock> nrend::CudaKernelResources::update(
    const KernelDefinitionsProvider* kernelProviderPtr,
    KernelResourcesProvider::KernelOpts kernelOpts,
    cudaStream_t stream,
    int cudaDeviceId,
    const NamedParameterDefinitionsSpan& namedParametersDefinitions,
    KernelMemoryBindings::BindingsFlag memoryFlag,
    bool attachMemory,
    Logger logger) {

    std::unique_lock<SpinMutex> lock(m_mutex);
    Status status = updateKernelResources(kernelProviderPtr, kernelOpts, stream, cudaDeviceId, namedParametersDefinitions, memoryFlag, attachMemory, logger);
    if (!status) {
        return std::unique_ptr<nrend::CudaKernelResources::StreamLock>();
    }

    return std::unique_ptr<CudaKernelResources::StreamLock>(
        m_perDeviceResources[cudaDeviceId] ? new StreamLock(stream, m_perDeviceResources[cudaDeviceId]) : nullptr);
}

nrend::Status nrend::CudaKernelResources::detach(
    const KernelDefinitionsProvider* kernelProviderPtr,
    cudaStream_t stream,
    int cudaDeviceId,
    KernelMemoryBindings::BindingsFlag memoryFlag,
    bool copy,
    Logger logger) {
    CHECK_STATUS_RETURN(checkCudaDeviceId(cudaDeviceId, logger));
    ResourcesPtr resourcesPtr = m_perDeviceResources[cudaDeviceId];
    if (!resourcesPtr) {
        RETURN_ERROR(
            logger,
            ErrorCode::InvalidResource,
            "CudaKernelResources : cannot detach resources, resources have not been successfully initialized on device %d.",
            cudaDeviceId);
    }

    const int numBindedMemory = m_memoryBindings.numRegisteredMemory(memoryFlag);
    for (int bindedMemoryIndex = 0; bindedMemoryIndex < numBindedMemory; ++bindedMemoryIndex) {
        auto& memoryPtr = resourcesPtr->bindedMemory[static_cast<int>(memoryFlag)][bindedMemoryIndex];
        if (memoryPtr) {
            // NOTE(qi): memory detach is only supported for cuda buffer currently
            auto* bufferPtr = memoryPtr->as<CudaBuffer>();
            if (!bufferPtr) {
                auto memoryName = m_memoryBindings.registeredMemoryName(memoryFlag, bindedMemoryIndex);
                auto memoryType = m_memoryBindings.registeredMemoryType(memoryFlag, bindedMemoryIndex);
                RETURN_ERROR(logger, ErrorCode::InvalidResource,
                             "CudaKernelResources : cannot detach binded buffer %s, only MemoryType::Buffer is supported, got %d.",
                             memoryName.c_str(), static_cast<int>(memoryType));
            }
            if (bufferPtr->attached()) {
                CHECK_STATUS_RETURN(bufferPtr->detach(copy, reinterpret_cast<uint64_t>(stream), logger));
            }
        }
    }

    return updateBindedMemoryBuffers(resourcesPtr,
                                     kernelProviderPtr,
                                     KernelResourcesProvider::Update,
                                     memoryFlag,
                                     stream,
                                     cudaDeviceId,
                                     logger);
}
