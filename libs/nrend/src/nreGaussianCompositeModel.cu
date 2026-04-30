// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include <nrend/models/nreGaussiansCompositeModel.h>

#include <tiny-cuda-nn/common.h>

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

namespace {

// FIXME : update this for future HW
constexpr uint32_t kWarpSize = 32;

inline bool invalidIndex(int32_t index, int32_t size) {
    return (index < 0) || (index >= size);
}

#define RETURN_ERROR_IF_INVALID_INDEX(index, size, logger)                                \
    RETURN_ERROR_IF(invalidIndex(index, size), logger, nrend::ErrorCode::InvalidResource, \
                    "NREGaussiansCompositeModel : invalid index. [%d / %d]", static_cast<int>(index), static_cast<int>(size));

#define RETURN_ERROR_IF_INVALID_INDEX_PTR(index, array, logger)                 \
    RETURN_ERROR_IF(invalidIndex(index, array.size()) || !array[index], logger, \
                    nrend::ErrorCode::InvalidResource,                          \
                    "NREGaussiansCompositeModel : invalid index. [%d / %zu]", static_cast<int>(index), array.size());

#define RETURN_ERROR_IF_INVALID_CAST_PTR(ptr, logger)                \
    RETURN_ERROR_IF(!ptr, logger, nrend::ErrorCode::InvalidResource, \
                    "NREGaussiansCompositeModel : invalid memory cast type.");

template <typename T = float>
inline nrend::Status copyFromKernelMemory(
    nrend::CudaBuffer* dstBuffer,
    const nrend::KernelMemoryPtrVec& srcMemoryVec,
    int32_t srcMemoryIndex,
    size_t elementSz,
    size_t numElements,
    size_t elementsOffset,
    uint64_t processQueueHandle,
    const nrend::Logger& logger) {
    RETURN_ERROR_IF_INVALID_INDEX_PTR(srcMemoryIndex, srcMemoryVec, logger);
    const nrend::CudaBuffer* srcBuffer = srcMemoryVec[srcMemoryIndex]->as<const nrend::CudaBuffer>();
    RETURN_ERROR_IF(!srcBuffer || srcBuffer->size() < numElements * elementSz * sizeof(T),
                    logger,
                    nrend::ErrorCode::InvalidResource,
                    "NREGaussiansCompositeModel : invalid source memory size. [%d / %zu]",
                    srcBuffer ? static_cast<int>(srcBuffer->size() * sizeof(T)) : -1,
                    numElements * elementSz * sizeof(T));
    return dstBuffer->data() ? dstBuffer->copyFromDevice(srcBuffer->data(),
                                                         numElements * elementSz * sizeof(T),
                                                         elementsOffset * elementSz * sizeof(T),
                                                         processQueueHandle,
                                                         logger)
                             : nrend::Status();
}

}; // namespace

nrend::Status nrend::NREGaussiansCompositeModel::prepareParticlesParameters(
    uint32_t numActiveTrackInstances,
    const tcnn::ivec2* activeTrackInstancesIdsCudaPtr,
    const KernelMemoryPtrVec& parameterMemoryPtrVec,
    std::vector<KernelBindedTransientMemory>& transientParameters,
    uint32_t& numParticles,
    uint32_t& numParticlesToPreProcess,
    cudaStream_t cudaStream,
    const Logger& logger) const {

    RETURN_ERROR_IF(!m_compositeModelPtr, logger, ErrorCode::InvalidResource, "NREGaussiansCompositeModel : no valid composite model.");

    const uint64_t queueHandle = reinterpret_cast<uint64_t>(cudaStream);

    const auto pushBackNewTransitientParameters = [&](int parameterBindingIndex) -> ScopedCudaBuffer* {
        auto* buffer = new ScopedCudaBuffer(queueHandle);
        transientParameters.push_back({
            KernelMemoryBindings::BindingsFlag::Parameters,
            std::unique_ptr<IScopedKernelMemory>(buffer),
            parameterBindingIndex,
        });
        return buffer;
    };

    numParticles             = m_numStaticParticles;
    numParticlesToPreProcess = 0;

    if (numActiveTrackInstances > 0) {
        // download the active track instances ids
        std::vector<tcnn::ivec2> activeTrackInstancesIdsHost(numActiveTrackInstances);
        CUDA_CHECK_RETURN(cudaMemcpyAsync(activeTrackInstancesIdsHost.data(),
                                          activeTrackInstancesIdsCudaPtr,
                                          numActiveTrackInstances * sizeof(tcnn::ivec2),
                                          cudaMemcpyDeviceToHost,
                                          cudaStream),
                          logger);
        // FIXME : remove this synchronization by providing the data on host memory (?)
        CUDA_CHECK_RETURN(cudaStreamSynchronize(cudaStream), logger);
        // create the mapping from active instances to their primitive and primitive local instance id
        std::vector<RenderingActiveInstance> renderingActiveInstances(numActiveTrackInstances);
        std::vector<uint32_t> renderingCumulativeNumInstances(numActiveTrackInstances);
        renderingCumulativeNumInstances[0] = 0;
        for (uint32_t i = 0; i < numActiveTrackInstances; i++) {
            RETURN_ERROR_IF_INVALID_INDEX(activeTrackInstancesIdsHost[i].x, m_activeInstances.size(), logger);
            const ActiveInstance& activeInstances = m_activeInstances[activeTrackInstancesIdsHost[i].x];

            // number of particles for this instance, ceil to the next multiple of kWarpSize
            const uint32_t activeInstanceNumParticlesToPreProcess = tcnn::div_round_up(activeInstances.numParticles, kWarpSize) * kWarpSize;

            if (i < numActiveTrackInstances - 1) {
                renderingCumulativeNumInstances[i + 1] = renderingCumulativeNumInstances[i] + activeInstanceNumParticlesToPreProcess;
            }

            renderingActiveInstances[i] = {
                activeInstances.numParticles,
                numParticles,
                activeInstances.primitiveId,
                activeInstances.primitiveInstanceId,
                activeInstances.particlesOffset,
            };

            numParticles += activeInstances.numParticles;
            numParticlesToPreProcess += activeInstanceNumParticlesToPreProcess;
        }

        // prepare the transient parameters
        transientParameters.reserve(transientParameters.size() + NumParameters);

        ScopedCudaBuffer* renderingCumulativeNumInstancesBuffer = pushBackNewTransitientParameters(m_parametersBindingIndex[RenderingCumulativeNumInstances]);
        CHECK_STATUS_RETURN(renderingCumulativeNumInstancesBuffer->setFromHostVector(renderingCumulativeNumInstances, logger));

        ScopedCudaBuffer* renderingActiveInstancesBuffer = pushBackNewTransitientParameters(m_parametersBindingIndex[RenderingActiveInstances]);
        CHECK_STATUS_RETURN(renderingActiveInstancesBuffer->setFromHostVector(renderingActiveInstances, logger));
    }

    const int densityParameterIndex = m_parametersBindingIndex[ParticleDensity];
    RETURN_ERROR_IF_INVALID_INDEX_PTR(densityParameterIndex, parameterMemoryPtrVec, logger);
    const CudaBuffer* densityParameterPtr = parameterMemoryPtrVec[densityParameterIndex]->as<CudaBuffer>();
    RETURN_ERROR_IF_INVALID_CAST_PTR(densityParameterPtr, logger);
    ScopedCudaBuffer* densityParameterBuffer = pushBackNewTransitientParameters(densityParameterIndex);
    CHECK_STATUS_RETURN(densityParameterBuffer->resize(numParticles * sizeof(float) * m_compositeModelPtr->densityParametersDim(), logger));
    CHECK_STATUS_RETURN(densityParameterBuffer->copyFromDevice(*densityParameterPtr, logger));

    const int featuresParameterIndex = m_parametersBindingIndex[ParticleFeatures];
    RETURN_ERROR_IF_INVALID_INDEX_PTR(featuresParameterIndex, parameterMemoryPtrVec, logger);
    const CudaBuffer* featuresParameterPtr = parameterMemoryPtrVec[featuresParameterIndex]->as<CudaBuffer>();
    RETURN_ERROR_IF_INVALID_CAST_PTR(featuresParameterPtr, logger);
    ScopedCudaBuffer* featuresParameterBuffer = pushBackNewTransitientParameters(featuresParameterIndex);
    CHECK_STATUS_RETURN(featuresParameterBuffer->resize(numParticles * m_compositeModelPtr->radianceParametersTypeSize() * m_compositeModelPtr->radianceParametersDim(), logger));
    if (featuresParameterBuffer->size() > 0) {
        CHECK_STATUS_RETURN(featuresParameterBuffer->copyFromDevice(*featuresParameterPtr, logger));
    }
    if (m_extendedFeaturesEnabled) {
        const int extendedFeaturesParameterIndex = m_parametersBindingIndex[ParticleExtendedFeatures];
        RETURN_ERROR_IF_INVALID_INDEX_PTR(extendedFeaturesParameterIndex, parameterMemoryPtrVec, logger);
        const CudaBuffer* extendedFeaturesParameterPtr = parameterMemoryPtrVec[extendedFeaturesParameterIndex]->as<CudaBuffer>();
        RETURN_ERROR_IF_INVALID_CAST_PTR(extendedFeaturesParameterPtr, logger);
        ScopedCudaBuffer* extendedFeaturesParameterBuffer = pushBackNewTransitientParameters(extendedFeaturesParameterIndex);
        CHECK_STATUS_RETURN(extendedFeaturesParameterBuffer->resize(numParticles * m_compositeModelPtr->extendedFeaturesParametersTypeSize() * m_compositeModelPtr->extendedFeaturesParametersDim(), logger));
        if (extendedFeaturesParameterBuffer->size() > 0) {
            CHECK_STATUS_RETURN(extendedFeaturesParameterBuffer->copyFromDevice(*extendedFeaturesParameterPtr, logger));
        }
    }

    if (m_sensorExtendedFeaturesEnabled) {
        const int cameraExtendedFeaturesParameterIndex = m_parametersBindingIndex[ParticleCameraExtendedFeatures];
        RETURN_ERROR_IF_INVALID_INDEX_PTR(cameraExtendedFeaturesParameterIndex, parameterMemoryPtrVec, logger);
        const CudaBuffer* cameraExtendedFeaturesParameterPtr = parameterMemoryPtrVec[cameraExtendedFeaturesParameterIndex]->as<CudaBuffer>();
        RETURN_ERROR_IF_INVALID_CAST_PTR(cameraExtendedFeaturesParameterPtr, logger);
        ScopedCudaBuffer* cameraExtendedFeaturesParameterBuffer = pushBackNewTransitientParameters(cameraExtendedFeaturesParameterIndex);
        CHECK_STATUS_RETURN(cameraExtendedFeaturesParameterBuffer->resize(numParticles * m_compositeModelPtr->cameraExtendedFeaturesParametersTypeSize() * m_compositeModelPtr->cameraExtendedFeaturesParametersDim(), logger));
        if (cameraExtendedFeaturesParameterBuffer->size() > 0) {
            CHECK_STATUS_RETURN(cameraExtendedFeaturesParameterBuffer->copyFromDevice(*cameraExtendedFeaturesParameterPtr, logger));
        }
        const int lidarExtendedFeaturesParameterIndex = m_parametersBindingIndex[ParticleLidarExtendedFeatures];
        RETURN_ERROR_IF_INVALID_INDEX_PTR(lidarExtendedFeaturesParameterIndex, parameterMemoryPtrVec, logger);
        const CudaBuffer* lidarExtendedFeaturesParameterPtr = parameterMemoryPtrVec[lidarExtendedFeaturesParameterIndex]->as<CudaBuffer>();
        RETURN_ERROR_IF_INVALID_CAST_PTR(lidarExtendedFeaturesParameterPtr, logger);
        ScopedCudaBuffer* lidarExtendedFeaturesParameterBuffer = pushBackNewTransitientParameters(lidarExtendedFeaturesParameterIndex);
        CHECK_STATUS_RETURN(lidarExtendedFeaturesParameterBuffer->resize(numParticles * m_compositeModelPtr->lidarExtendedFeaturesParametersTypeSize() * m_compositeModelPtr->lidarExtendedFeaturesParametersDim(), logger));
        if (lidarExtendedFeaturesParameterBuffer->size() > 0) {
            CHECK_STATUS_RETURN(lidarExtendedFeaturesParameterBuffer->copyFromDevice(*lidarExtendedFeaturesParameterPtr, logger));
        }
    }

    return Status();
}

nrend::Status nrend::NREGaussiansCompositeModel::registerKernelResources_(
    const KernelMemoryBindings& memoryBindings,
    const KernelSourceCodeTable& sourceCodeTable,
    KernelResourcesProvider::KernelOpts kernelOpts,
    const Logger& logger) const {

    RETURN_ERROR_IF(kernelOpts & KernelResourcesProvider::Differentiable,
                    logger, ErrorCode::NotImplemented, "NREGaussiansCompositeModel : not differentiable.");

    RETURN_ERROR_IF(!m_compositeModelPtr, logger, ErrorCode::InvalidResource, "NREGaussiansCompositeModel : no valid composite model.");

    m_extendedFeaturesEnabled       = !(kernelOpts & KernelResourcesProvider::DisableExtendedFeatures) && (featuresLayout().extendedFeaturesDim > 0);
    m_sensorExtendedFeaturesEnabled = !(kernelOpts & KernelResourcesProvider::DisableSensorExtendedFeatures) &&
                                      (featuresLayout().cameraExtendedFeaturesDim + featuresLayout().lidarExtendedFeaturesDim > 0);

    CHECK_STATUS_RETURN(m_compositeModelPtr->registerKernelResources(memoryBindings, sourceCodeTable, kernelOpts, logger));

    m_parametersBindingIndex[ParticleDensity] =
        memoryBindings.registeredMemoryIndex(KernelMemoryBindings::Parameters, m_compositeModelPtr->densityParametersKey());
    m_parametersBindingIndex[ParticleFeatures] =
        memoryBindings.registeredMemoryIndex(KernelMemoryBindings::Parameters, m_compositeModelPtr->radianceParametersKey());
    m_parametersBindingIndex[ParticleExtendedFeatures] =
        memoryBindings.registeredMemoryIndex(KernelMemoryBindings::Parameters, m_compositeModelPtr->extraSignalParametersKey());
    m_parametersBindingIndex[ParticleCameraExtendedFeatures] =
        memoryBindings.registeredMemoryIndex(KernelMemoryBindings::Parameters, m_compositeModelPtr->cameraExtendedFeaturesParametersKey());
    m_parametersBindingIndex[ParticleLidarExtendedFeatures] =
        memoryBindings.registeredMemoryIndex(KernelMemoryBindings::Parameters, m_compositeModelPtr->lidarExtendedFeaturesParametersKey());

    CHECK_STATUS_RETURN(memoryBindings.registerMemory(KernelMemoryBindings::Parameters,
                                                      renderingCumulativeNumInstancesParameterKey(),
                                                      KernelMemoryType::Buffer,
                                                      logger));
    m_parametersBindingIndex[RenderingCumulativeNumInstances] =
        memoryBindings.registeredMemoryIndex(KernelMemoryBindings::Parameters, renderingCumulativeNumInstancesParameterKey());

    CHECK_STATUS_RETURN(memoryBindings.registerMemory(KernelMemoryBindings::Parameters,
                                                      renderingActiveInstanceParametersKey(),
                                                      KernelMemoryType::Buffer,
                                                      logger));
    m_parametersBindingIndex[RenderingActiveInstances] =
        memoryBindings.registeredMemoryIndex(KernelMemoryBindings::Parameters, renderingActiveInstanceParametersKey());

    // compute the number of static particles (from background primitives)
    const uint32_t backgroundPrimitivesOffset = m_subModelPtr.size() - m_primitives.size() - m_backgroundPrimitives.size();
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_backgroundPrimitives.size()); ++i) {
        const NRESHGaussianModel* particlesModelPtr = dynamic_cast<const NRESHGaussianModel*>(m_subModelPtr[backgroundPrimitivesOffset + i].get());
        if (particlesModelPtr) {
            m_numStaticParticles += particlesModelPtr->numParticles();
        } else {
            LOG_ERROR(logger, "NREGaussiansCompositeModel : unsupported background static gaussian node %s", m_backgroundPrimitives[i].c_str());
        }
    }

    const std::string cudaSourceCodeTemplate = R"(
            #include <nrend/kernels/cuda/models/nreGaussiansCompositeModel.cuh>

            struct {NREGaussianCompositeAlias}InternalParams 
            {{
                static constexpr int RenderingCumulativeNumInstancesBufferIndex = {RenderingCumulativeNumInstancesBufferIndex};
                static constexpr int RenderingActiveInstancesBufferIndex        = {RenderingActiveInstancesBufferIndex};
                static constexpr int ParticleDensityBufferIndex                 = {ParticleDensityBufferIndex};
                static constexpr int ParticleFeaturesBufferIndex                = {ParticleFeaturesBufferIndex};
                static constexpr int ExtendedFeaturesDim                        = {ExtendedFeaturesDim};
                static constexpr int ParticleExtendedFeaturesBufferIndex        = {ParticleExtendedFeaturesBufferIndex};
                static constexpr int CameraExtendedFeaturesDim                  = {CameraExtendedFeaturesDim};
                static constexpr int ParticleCameraExtendedFeaturesBufferIndex  = {ParticleCameraExtendedFeaturesBufferIndex};
                static constexpr int LidarExtendedFeaturesDim                   = {LidarExtendedFeaturesDim};
                static constexpr int ParticleLidarExtendedFeaturesBufferIndex   = {ParticleLidarExtendedFeaturesBufferIndex};
                static constexpr int NumStaticParticles                         = {NumStaticParticles};
                static constexpr int PrimitiveIdOffset                          = {PrimitiveIdOffset};
                static constexpr bool WarpParallelSearch                        = {WarpParallelSearch};
                static constexpr bool EnableBackground                          = {EnableBackground};
                static constexpr bool EnablePostProcessings                     = {EnablePostProcessings};
                static constexpr bool SaturateRadiance                          = {SaturateRadiance};
            }};

            using {NREGaussianCompositeAlias} = NREGaussiansComposite<{NREGaussianNodeAlias}::Particles,
                                                                      {NREAppearanceEmbeddingClassAlias}, 
                                                                      {NREBackgroundClassAlias},
                                                                      {NREPostProcessingsClassAlias},
                                                                      {NREGaussianCompositeAlias}InternalParams,
                                                                      {NREVariadicPrimitiveTypes}>;
        )";
    sourceCodeTable.registerKernel(
        KernelSourceCodeTable::Cuda,
        fmt::format(cudaSourceCodeTemplate,
                    fmt::arg("NREGaussianCompositeAlias", cudaCallPrefix()),
                    fmt::arg("RenderingCumulativeNumInstancesBufferIndex", m_parametersBindingIndex[RenderingCumulativeNumInstances]),
                    fmt::arg("RenderingActiveInstancesBufferIndex", m_parametersBindingIndex[RenderingActiveInstances]),
                    fmt::arg("ParticleDensityBufferIndex", m_parametersBindingIndex[ParticleDensity]),
                    fmt::arg("ParticleFeaturesBufferIndex", m_parametersBindingIndex[ParticleFeatures]),
                    fmt::arg("ExtendedFeaturesDim", m_extendedFeaturesEnabled ? m_compositeModelPtr->featuresLayout().extendedFeaturesDim : 0),
                    fmt::arg("ParticleExtendedFeaturesBufferIndex", m_parametersBindingIndex[ParticleExtendedFeatures]),
                    fmt::arg("CameraExtendedFeaturesDim", m_sensorExtendedFeaturesEnabled ? m_compositeModelPtr->featuresLayout().cameraExtendedFeaturesDim : 0),
                    fmt::arg("ParticleCameraExtendedFeaturesBufferIndex", m_parametersBindingIndex[ParticleCameraExtendedFeatures]),
                    fmt::arg("LidarExtendedFeaturesDim", m_sensorExtendedFeaturesEnabled ? m_compositeModelPtr->featuresLayout().lidarExtendedFeaturesDim : 0),
                    fmt::arg("ParticleLidarExtendedFeaturesBufferIndex", m_parametersBindingIndex[ParticleLidarExtendedFeatures]),
                    fmt::arg("NumStaticParticles", m_numStaticParticles),
                    fmt::arg("PrimitiveIdOffset", m_backgroundPrimitives.size()),
                    fmt::arg("WarpParallelSearch", true),
                    fmt::arg("EnableBackground", !(kernelOpts & KernelResourcesProvider::DisableBackground)),
                    fmt::arg("EnablePostProcessings", !(kernelOpts & KernelResourcesProvider::DisablePostProcessings)),
                    fmt::arg("SaturateRadiance", m_saturateRadiance),
                    fmt::arg("NREGaussianNodeAlias", m_compositeModelPtr->cudaCallPrefix()),
                    fmt::arg("NREAppearanceEmbeddingClassAlias", cudaCallPrefix("appearance_embedding")),
                    fmt::arg("NREBackgroundClassAlias", cudaCallPrefix("background")),
                    fmt::arg("NREPostProcessingsClassAlias", cudaCallPrefix("post_processings")),
                    fmt::arg("NREVariadicPrimitiveTypes", getVariadicActivePrimitiveTypesStr("gaussians_nodes"))));

    return Status();
}

nrend::Status nrend::NREGaussiansCompositeModel::processKernelMemory_(
    const KernelMemoryBindings& memoryBindings,
    KernelMemoryBindings::BindingsFlag bindingsFlag,
    const KernelMemoryPtrVec& memory,
    ProcessMemoryFlag processFlag,
    uint64_t processQueueHandle,
    const Logger& logger) const {

    if (processFlag != ProcessMemoryFlag::Initialization) {
        RETURN_ERROR(logger, ErrorCode::BadInput, "NREGaussiansCompositeModel does not support parameter update.");
        return Status();
    }

    if (bindingsFlag != KernelMemoryBindings::Parameters) {
        return Status();
    }

    // compute the active instances data
    uint32_t numActivePrimitivesInstances = 0;
    for (size_t i = 0; i < m_primitives.size(); ++i) {
        numActivePrimitivesInstances += m_primitives[i].numActiveInstances;
    }
    m_activeInstances.resize(numActivePrimitivesInstances);

    const uint32_t primitiveOffset = m_subModelPtr.size() - m_primitives.size();
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_primitivesInstancesMap.size()); ++i) {
        const uint32_t primitiveId = m_primitivesInstancesMap[i].primitiveId;
        const uint32_t instanceId  = m_primitivesInstancesMap[i].instanceId;

        const NRESHGaussianModel* particlesModelPtr = dynamic_cast<const NRESHGaussianModel*>(m_subModelPtr[primitiveOffset + primitiveId].get());
        const uint32_t numParticles                 = particlesModelPtr ? particlesModelPtr->numInstanceParticles(instanceId) : 0;
        const uint32_t particlesOffset              = particlesModelPtr ? particlesModelPtr->instanceParticlesOffset(instanceId) : 0;
        for (uint16_t mappingIndex : m_primitivesInstancesMap[i].trackMappingIndex) {
            if (mappingIndex >= numActivePrimitivesInstances) {
                RETURN_ERROR(logger, ErrorCode::InvalidResource, "NREGaussiansCompositeModel : Invalid mapping index: %hu / %u ",
                             mappingIndex, numActivePrimitivesInstances);
            }
            // Offset the primitive id by the number of background primitives
            m_activeInstances[mappingIndex] = {primitiveId, instanceId, numParticles, particlesOffset};
        }
    }

    const int32_t particleDensityIndex                = m_parametersBindingIndex[ParticleDensity];
    const int32_t particleFeaturesIndex               = m_parametersBindingIndex[ParticleFeatures];
    const int32_t particleExtendedFeaturesIndex       = m_parametersBindingIndex[ParticleExtendedFeatures];
    const int32_t particleCameraExtendedFeaturesIndex = m_parametersBindingIndex[ParticleCameraExtendedFeatures];
    const int32_t particleLidarExtendedFeaturesIndex  = m_parametersBindingIndex[ParticleLidarExtendedFeatures];

    RETURN_ERROR_IF_INVALID_INDEX(particleDensityIndex, memory.size(), logger);
    RETURN_ERROR_IF_INVALID_INDEX(particleFeaturesIndex, memory.size(), logger);
    RETURN_ERROR_IF_INVALID_INDEX(particleExtendedFeaturesIndex, memory.size(), logger);
    RETURN_ERROR_IF_INVALID_INDEX(particleCameraExtendedFeaturesIndex, memory.size(), logger);
    RETURN_ERROR_IF_INVALID_INDEX(particleLidarExtendedFeaturesIndex, memory.size(), logger);

    CudaBuffer* particleDensityBuffer                = memory[particleDensityIndex]->as<CudaBuffer>();
    CudaBuffer* particleFeaturesBuffer               = memory[particleFeaturesIndex]->as<CudaBuffer>();
    CudaBuffer* particleExtendedFeaturesBuffer       = memory[particleExtendedFeaturesIndex]->as<CudaBuffer>();
    CudaBuffer* particleCameraExtendedFeaturesBuffer = memory[particleCameraExtendedFeaturesIndex]->as<CudaBuffer>();
    CudaBuffer* particleLidarExtendedFeaturesBuffer  = memory[particleLidarExtendedFeaturesIndex]->as<CudaBuffer>();

    RETURN_ERROR_IF_INVALID_CAST_PTR(particleDensityBuffer, logger);
    RETURN_ERROR_IF_INVALID_CAST_PTR(particleFeaturesBuffer, logger);
    RETURN_ERROR_IF_INVALID_CAST_PTR(particleExtendedFeaturesBuffer, logger);
    RETURN_ERROR_IF_INVALID_CAST_PTR(particleCameraExtendedFeaturesBuffer, logger);
    RETURN_ERROR_IF_INVALID_CAST_PTR(particleLidarExtendedFeaturesBuffer, logger);

    CHECK_STATUS_RETURN(particleDensityBuffer->resize(
        sizeof(float) * m_compositeModelPtr->densityParametersDim() * m_numStaticParticles, processQueueHandle, logger));
    CHECK_STATUS_RETURN(particleFeaturesBuffer->resize(
        m_compositeModelPtr->radianceParametersTypeSize() * m_compositeModelPtr->radianceParametersDim() * m_numStaticParticles, processQueueHandle, logger));
    if (m_extendedFeaturesEnabled) {
        CHECK_STATUS_RETURN(particleExtendedFeaturesBuffer->resize(
            m_compositeModelPtr->extendedFeaturesParametersTypeSize() * m_compositeModelPtr->extendedFeaturesParametersDim() * m_numStaticParticles, processQueueHandle, logger));
    }
    if (m_sensorExtendedFeaturesEnabled) {
        CHECK_STATUS_RETURN(particleCameraExtendedFeaturesBuffer->resize(
            m_compositeModelPtr->cameraExtendedFeaturesParametersTypeSize() * m_compositeModelPtr->cameraExtendedFeaturesParametersDim() * m_numStaticParticles, processQueueHandle, logger));
        CHECK_STATUS_RETURN(particleLidarExtendedFeaturesBuffer->resize(
            m_compositeModelPtr->lidarExtendedFeaturesParametersTypeSize() * m_compositeModelPtr->lidarExtendedFeaturesParametersDim() * m_numStaticParticles, processQueueHandle, logger));
    }

    const uint32_t backgroundPrimitivesOffset = m_subModelPtr.size() - m_primitives.size() - m_backgroundPrimitives.size();
    uint32_t numStaticParticlesInitialized    = 0;
    for (uint32_t i = 0; i < m_backgroundPrimitives.size(); ++i) {
        const NRESHGaussianModel* particlesModelPtr = dynamic_cast<const NRESHGaussianModel*>(m_subModelPtr[backgroundPrimitivesOffset + i].get());
        if (!particlesModelPtr) {
            continue;
        }
        const uint32_t numParticles = particlesModelPtr->numParticles();

        CHECK_STATUS_RETURN(copyFromKernelMemory(
            particleDensityBuffer,
            memory,
            memoryBindings.registeredMemoryIndex(bindingsFlag, particlesModelPtr->densityParametersKey()),
            particlesModelPtr->densityParametersDim(),
            numParticles,
            numStaticParticlesInitialized,
            processQueueHandle,
            logger));
        CHECK_STATUS_RETURN(copyFromKernelMemory<uint8_t>(
            particleFeaturesBuffer,
            memory,
            memoryBindings.registeredMemoryIndex(bindingsFlag, particlesModelPtr->radianceParametersKey()),
            particlesModelPtr->radianceParametersDim() * particlesModelPtr->radianceParametersTypeSize(),
            numParticles,
            numStaticParticlesInitialized,
            processQueueHandle,
            logger));
        if (m_extendedFeaturesEnabled) {
            CHECK_STATUS_RETURN(copyFromKernelMemory<uint8_t>(
                particleExtendedFeaturesBuffer,
                memory,
                memoryBindings.registeredMemoryIndex(bindingsFlag, particlesModelPtr->extraSignalParametersKey()),
                particlesModelPtr->extendedFeaturesParametersDim() * particlesModelPtr->extendedFeaturesParametersTypeSize(),
                numParticles,
                numStaticParticlesInitialized,
                processQueueHandle,
                logger));
        }
        if (m_sensorExtendedFeaturesEnabled) {
            CHECK_STATUS_RETURN(copyFromKernelMemory<uint8_t>(
                particleCameraExtendedFeaturesBuffer,
                memory,
                memoryBindings.registeredMemoryIndex(bindingsFlag, particlesModelPtr->cameraExtendedFeaturesParametersKey()),
                particlesModelPtr->cameraExtendedFeaturesParametersDim() * particlesModelPtr->cameraExtendedFeaturesParametersTypeSize(),
                numParticles,
                numStaticParticlesInitialized,
                processQueueHandle,
                logger));

            CHECK_STATUS_RETURN(copyFromKernelMemory<uint8_t>(
                particleLidarExtendedFeaturesBuffer,
                memory,
                memoryBindings.registeredMemoryIndex(bindingsFlag, particlesModelPtr->lidarExtendedFeaturesParametersKey()),
                particlesModelPtr->lidarExtendedFeaturesParametersDim() * particlesModelPtr->lidarExtendedFeaturesParametersTypeSize(),
                numParticles,
                numStaticParticlesInitialized,
                processQueueHandle,
                logger));
        }
        numStaticParticlesInitialized += numParticles;
    }

    RETURN_ERROR_IF(numStaticParticlesInitialized != m_numStaticParticles, logger, ErrorCode::InvalidResource,
                    "NREGaussiansCompositeModel : Invalid number of static particles initialized. [%u / %u]",
                    numStaticParticlesInitialized, m_numStaticParticles);

    return Status();
}
