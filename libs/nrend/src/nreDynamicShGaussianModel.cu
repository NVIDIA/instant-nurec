// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include <nrend/models/nreDynamicShGaussianModel.h>

#include <tiny-cuda-nn/common.h>

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

namespace {

inline uint32_t getMSBExclusive(uint32_t x) {
#if defined(_MSC_VER)
    unsigned long index;
    if (_BitScanReverse(&index, x)) {
        return index + 1; // Make it exclusive by adding 1
    }
    return 0; // Handle x == 0 case
#else
    if (x == 0)
        return 0;
    return 32 - __builtin_clz(x); // Already exclusive
#endif
}

inline bool invalidIndex(int32_t index, int32_t size) {
    return (index < 0) || (index >= size);
}

#define RETURN_ERROR_IF_INVALID_INDEX(index, size, logger)                                \
    RETURN_ERROR_IF(invalidIndex(index, size), logger, nrend::ErrorCode::InvalidResource, \
                    "NREDynamicSHGaussianModel : invalid index. [%d / %d]", index, static_cast<int>(size));

#define RETURN_ERROR_IF_INVALID_INDEX_PTR(index, array, logger)                 \
    RETURN_ERROR_IF(invalidIndex(index, array.size()) || !array[index], logger, \
                    nrend::ErrorCode::InvalidResource,                          \
                    "NREDynamicSHGaussianModel : invalid index. [%d / %zu]", index, array.size());

#define RETURN_ERROR_IF_INVALID_CAST_PTR(ptr, logger, key)           \
    RETURN_ERROR_IF(!ptr, logger, nrend::ErrorCode::InvalidResource, \
                    "NREDynamicSHGaussianModel : invalid memory cast type for %s.", key.c_str());

__global__ void fillSequence(uint32_t* __restrict__ buffer, uint32_t num_elements) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        buffer[idx] = idx;
    }
}

}; // namespace

// TODO : check if sorting by both instance id and particles id is better (to preserve potential existing spatial coherence)
nrend::Status nrend::NRERigidSHGaussianModel::packParametersFromHostTensorsWithInstanceIdSort(CudaBuffer* densityParamsPtr,
                                                                                              CudaBuffer* radianceParamsPtr,
                                                                                              CudaBuffer* extraSignalParamsPtr,
                                                                                              CudaBuffer* cameraExtendedFeaturesParamsPtr,
                                                                                              CudaBuffer* lidarExtendedFeaturesParamsPtr,
                                                                                              bool halfPrecisionFeatures,
                                                                                              uint64_t processQueueHandle,
                                                                                              const Logger& logger) const {

    RETURN_ERROR_IF(!densityParamsPtr || !radianceParamsPtr || !extraSignalParamsPtr, logger, ErrorCode::InvalidResource,
                    "NREDynamicSHGaussianModel : invalid cuda buffers.");

    const int threads       = 1024;
    const int blocks        = tcnn::div_round_up<int>(m_particlesNumber, threads);
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(processQueueHandle);

    ScopedCudaBuffer particlesIdx(processQueueHandle);
    // Memory allocation scope
    {
        ScopedCudaBuffer unsortedParticlesIdx(processQueueHandle);
        CHECK_STATUS_RETURN(unsortedParticlesIdx.resize(m_particlesNumber * sizeof(uint32_t), logger));
        fillSequence<<<blocks, threads, 0, cudaStream>>>(unsortedParticlesIdx.ptr<uint32_t>(), m_particlesNumber);
        CUDA_CHECK_STREAM_RETURN(cudaStream, logger);

        ScopedCudaBuffer instancesIdx(processQueueHandle);
        CHECK_STATUS_RETURN(instancesIdx.resize(m_particlesNumber * sizeof(uint32_t), logger));

        ScopedCudaBuffer unsortedInstancesIdx(processQueueHandle);
        CHECK_STATUS_RETURN(unsortedInstancesIdx.setFromHost(m_particlesInstanceIdxTensor.buffer.data(),
                                                             m_particlesInstanceIdxTensor.buffer.size(),
                                                             logger));
        if (unsortedInstancesIdx.size() != m_particlesNumber * sizeof(uint32_t)) {
            RETURN_ERROR(logger, ErrorCode::BadInput, "NREDynamicSHGaussianModel : invalid instance index buffer size. [%zu / %u]",
                         unsortedInstancesIdx.size() / sizeof(uint32_t), m_particlesNumber);
        }
        CHECK_STATUS_RETURN(particlesIdx.resize(m_particlesNumber * sizeof(uint32_t), logger));

        size_t sortingWorkingBufferSize = 0;
        CUDA_CHECK_RETURN(cub::DeviceRadixSort::SortPairs(nullptr,
                                                          sortingWorkingBufferSize,
                                                          unsortedInstancesIdx.ptr<uint32_t>(),
                                                          instancesIdx.ptr<uint32_t>(),
                                                          unsortedParticlesIdx.ptr<uint32_t>(),
                                                          particlesIdx.ptr<uint32_t>(),
                                                          m_particlesNumber,
                                                          0, getMSBExclusive(m_numInstances), cudaStream),
                          logger);
        ScopedCudaBuffer sortingWorkingBuffer(processQueueHandle);
        CHECK_STATUS_RETURN(sortingWorkingBuffer.resize(sortingWorkingBufferSize, logger));

        CUDA_CHECK_RETURN(cub::DeviceRadixSort::SortPairs(sortingWorkingBuffer.ptr<void>(),
                                                          sortingWorkingBufferSize,
                                                          unsortedInstancesIdx.ptr<uint32_t>(),
                                                          instancesIdx.ptr<uint32_t>(),
                                                          unsortedParticlesIdx.ptr<uint32_t>(),
                                                          particlesIdx.ptr<uint32_t>(),
                                                          m_particlesNumber,
                                                          0, getMSBExclusive(m_numInstances), cudaStream),
                          logger);

        // Compute the number of particles per instance using run-length encoding
        ScopedCudaBuffer numInstances(processQueueHandle);
        CHECK_STATUS_RETURN(numInstances.resize(sizeof(uint32_t), logger));
        // aliasing buffers to avoid extra memory allocation
        auto& uniqueInstanceIdx      = unsortedParticlesIdx;
        auto& instanceParticlesCount = unsortedInstancesIdx;
        CUDA_CHECK_RETURN(cub::DeviceRunLengthEncode::Encode(nullptr,
                                                             sortingWorkingBufferSize,
                                                             instancesIdx.ptr<uint32_t>(),
                                                             uniqueInstanceIdx.ptr<uint32_t>(),
                                                             instanceParticlesCount.ptr<uint32_t>(),
                                                             numInstances.ptr<uint32_t>(),
                                                             m_particlesNumber,
                                                             cudaStream),
                          logger);
        CHECK_STATUS_RETURN(sortingWorkingBuffer.enlarge(sortingWorkingBufferSize, logger));
        CUDA_CHECK_RETURN(cub::DeviceRunLengthEncode::Encode(sortingWorkingBuffer.ptr<void>(),
                                                             sortingWorkingBufferSize,
                                                             instancesIdx.ptr<uint32_t>(),
                                                             uniqueInstanceIdx.ptr<uint32_t>(),
                                                             instanceParticlesCount.ptr<uint32_t>(),
                                                             numInstances.ptr<uint32_t>(),
                                                             m_particlesNumber,
                                                             cudaStream),
                          logger);

        uint32_t numInstancesHost;
        CUDA_CHECK_RETURN(cudaMemcpyAsync(&numInstancesHost,
                                          numInstances.ptr<uint32_t>(),
                                          sizeof(uint32_t),
                                          cudaMemcpyDeviceToHost,
                                          cudaStream),
                          logger);
        CUDA_CHECK_RETURN(cudaStreamSynchronize(cudaStream), logger);
        RETURN_ERROR_IF(numInstancesHost > m_numInstances, logger, ErrorCode::BadInput,
                        "NREDynamicSHGaussianModel : number of instances mismatch. [%d / %d (%u)]", numInstancesHost, m_numInstances, m_particlesNumber);

        std::vector<uint32_t> instancesNumParticles(numInstancesHost);
        CUDA_CHECK_RETURN(cudaMemcpyAsync(instancesNumParticles.data(),
                                          instanceParticlesCount.ptr<uint32_t>(),
                                          sizeof(uint32_t) * instancesNumParticles.size(),
                                          cudaMemcpyDeviceToHost),
                          logger);
        std::vector<uint32_t> uniqueInstanceIdxHost(numInstancesHost);
        CUDA_CHECK_RETURN(cudaMemcpyAsync(uniqueInstanceIdxHost.data(),
                                          uniqueInstanceIdx.ptr<uint32_t>(),
                                          uniqueInstanceIdxHost.size() * sizeof(uint32_t),
                                          cudaMemcpyDeviceToHost),
                          logger);
        CUDA_CHECK_RETURN(cudaStreamSynchronize(cudaStream), logger);

        RETURN_ERROR_IF(m_numInstances > m_instancesNumParticles.size() || m_numInstances > m_instancesParticlesOffset.size(), logger, ErrorCode::BadInput,
                        "NREDynamicSHGaussianModel : number of instances mismatch. [%zu %zu / %u]",
                        m_instancesNumParticles.size(), m_instancesParticlesOffset.size(), m_numInstances);

        for (uint32_t i = 0; i < numInstancesHost; i++) {
            RETURN_ERROR_IF_INVALID_INDEX(uniqueInstanceIdxHost[i], m_numInstances, logger);
            m_instancesNumParticles[uniqueInstanceIdxHost[i]] = instancesNumParticles[i];
        }
        for (uint32_t i = 1; i < m_instancesNumParticles.size(); i++) {
            m_instancesParticlesOffset[i] = m_instancesParticlesOffset[i - 1] + m_instancesNumParticles[i - 1];
        }
    }

    ScopedCudaBuffer positions(processQueueHandle);
    const int positionsBufferSize = sizeof(__half) * 3 * m_particlesNumber;
    if (m_paramsTensor[Positions].buffer.size() != positionsBufferSize) {
        RETURN_ERROR(logger, ErrorCode::BadInput, "NREDynamicSHGaussianModel : input positions data has wrong size [%d/%d].",
                     static_cast<int>(m_paramsTensor[Positions].buffer.size()), static_cast<int>(positionsBufferSize));
    }
    CHECK_STATUS_RETURN(positions.setFromHost(m_paramsTensor[Positions].buffer.data(),
                                              m_paramsTensor[Positions].buffer.size(),
                                              logger));

    return packParametersFromHostTensors(*densityParamsPtr,
                                         *radianceParamsPtr,
                                         *extraSignalParamsPtr,
                                         *cameraExtendedFeaturesParamsPtr,
                                         *lidarExtendedFeaturesParamsPtr,
                                         positions.ptr<tcnn::tvec<__half, 3>>(),
                                         particlesIdx.ptr<uint32_t>(),
                                         halfPrecisionFeatures,
                                         processQueueHandle,
                                         logger);
}

nrend::Status nrend::NRERigidSHGaussianModel::processKernelMemory_(
    const KernelMemoryBindings& memoryBindings,
    KernelMemoryBindings::BindingsFlag bindingsFlag,
    const std::vector<std::unique_ptr<KernelMemory>>& memory,
    ProcessMemoryFlag processFlag,
    uint64_t processQueueHandle,
    const Logger& logger) const {

    if (processFlag != ProcessMemoryFlag::Initialization) {
        RETURN_ERROR(logger, ErrorCode::BadInput, "NREDynamicSHGaussianModel does not support parameter update.");
    }

    if (bindingsFlag != KernelMemoryBindings::Parameters) {
        return Status();
    }

    CUDA_CHECK_STREAM_RETURN(reinterpret_cast<cudaStream_t>(processQueueHandle), logger);

    const int densityParametersIndex = memoryBindings.registeredMemoryIndex(bindingsFlag, densityParametersKey());
    RETURN_ERROR_IF_INVALID_INDEX_PTR(densityParametersIndex, memory, logger);
    const int radianceParametersIndex = memoryBindings.registeredMemoryIndex(bindingsFlag, radianceParametersKey());
    RETURN_ERROR_IF_INVALID_INDEX_PTR(radianceParametersIndex, memory, logger);

    CudaBuffer* densityParametersBuffer = memory[densityParametersIndex]->as<CudaBuffer>();
    RETURN_ERROR_IF_INVALID_CAST_PTR(densityParametersBuffer, logger, densityParametersKey());
    CudaBuffer* radianceParametersBuffer = memory[radianceParametersIndex]->as<CudaBuffer>();
    RETURN_ERROR_IF_INVALID_CAST_PTR(radianceParametersBuffer, logger, radianceParametersKey());

    if (densityParametersBuffer->attached() != radianceParametersBuffer->attached()) {
        RETURN_ERROR(logger, ErrorCode::BadInput, "NREDynamicSHGaussianModel : resource %s and %s have a different attachment.",
                     densityParametersKey().c_str(), radianceParametersKey().c_str());
    }

    const int extraSignalParametersIndex = memoryBindings.registeredMemoryIndex(bindingsFlag, extraSignalParametersKey());
    RETURN_ERROR_IF_INVALID_INDEX_PTR(extraSignalParametersIndex, memory, logger);
    CudaBuffer* extraSignalParametersBuffer = memory[extraSignalParametersIndex]->as<CudaBuffer>();
    RETURN_ERROR_IF_INVALID_CAST_PTR(extraSignalParametersBuffer, logger, extraSignalParametersKey());
    const int cameraExtendedFeaturesParametersIndex = memoryBindings.registeredMemoryIndex(bindingsFlag, cameraExtendedFeaturesParametersKey());
    RETURN_ERROR_IF_INVALID_INDEX_PTR(cameraExtendedFeaturesParametersIndex, memory, logger);
    CudaBuffer* cameraExtendedFeaturesParametersBuffer = memory[cameraExtendedFeaturesParametersIndex]->as<CudaBuffer>();
    RETURN_ERROR_IF_INVALID_CAST_PTR(cameraExtendedFeaturesParametersBuffer, logger, cameraExtendedFeaturesParametersKey());
    const int lidarExtendedFeaturesParametersIndex = memoryBindings.registeredMemoryIndex(bindingsFlag, lidarExtendedFeaturesParametersKey());
    RETURN_ERROR_IF_INVALID_INDEX_PTR(lidarExtendedFeaturesParametersIndex, memory, logger);
    CudaBuffer* lidarExtendedFeaturesParametersBuffer = memory[lidarExtendedFeaturesParametersIndex]->as<CudaBuffer>();
    RETURN_ERROR_IF_INVALID_CAST_PTR(lidarExtendedFeaturesParametersBuffer, logger, lidarExtendedFeaturesParametersKey());

    if (m_extendedFeaturesEnabled) {
        if (densityParametersBuffer->attached() != extraSignalParametersBuffer->attached()) {
            RETURN_ERROR(logger, ErrorCode::BadInput, "NREDynamicSHGaussianModel : resource %s and %s have a different attachment.",
                         densityParametersKey().c_str(), extraSignalParametersKey().c_str());
        }
    }
    if (m_sensorExtendedFeaturesEnabled) {
        if (densityParametersBuffer->attached() != cameraExtendedFeaturesParametersBuffer->attached()) {
            RETURN_ERROR(logger, ErrorCode::BadInput, "NREDynamicSHGaussianModel : resource %s and %s have a different attachment.",
                         densityParametersKey().c_str(), cameraExtendedFeaturesParametersKey().c_str());
        }
        if (densityParametersBuffer->attached() != lidarExtendedFeaturesParametersBuffer->attached()) {
            RETURN_ERROR(logger, ErrorCode::BadInput, "NREDynamicSHGaussianModel : resource %s and %s have a different attachment.",
                         densityParametersKey().c_str(), lidarExtendedFeaturesParametersKey().c_str());
        }
    }

    CHECK_STATUS_RETURN(memoryBindings.getRegisteredValue(particlesNumberParameterKey(), m_particlesNumber, logger));

    if ((m_particlesNumber > 0) && (processFlag == ProcessMemoryFlag::Initialization) && !densityParametersBuffer->attached()) {
        if (!m_validInitialParameters) {
            RETURN_ERROR(logger, ErrorCode::InvalidResource, "NREDynamicSHGaussianModel [%s] : cannot initialize resource, invalid initial parameters.", m_callPrefix.c_str());
        }
        CHECK_STATUS_RETURN(densityParametersBuffer->resize(sizeof(float) * densityParametersDim() * m_particlesNumber, processQueueHandle, logger));
        CHECK_STATUS_RETURN(radianceParametersBuffer->resize(radianceParametersTypeSize() * radianceParametersDim() * m_particlesNumber, processQueueHandle, logger));
        if (m_extendedFeaturesEnabled) {
            CHECK_STATUS_RETURN(extraSignalParametersBuffer->resize(extendedFeaturesParametersTypeSize() * extendedFeaturesParametersDim() * m_particlesNumber, processQueueHandle, logger));
        }
        if (m_sensorExtendedFeaturesEnabled) {
            CHECK_STATUS_RETURN(cameraExtendedFeaturesParametersBuffer->resize(cameraExtendedFeaturesParametersTypeSize() * cameraExtendedFeaturesParametersDim() * m_particlesNumber, processQueueHandle, logger));
            CHECK_STATUS_RETURN(lidarExtendedFeaturesParametersBuffer->resize(lidarExtendedFeaturesParametersTypeSize() * lidarExtendedFeaturesParametersDim() * m_particlesNumber, processQueueHandle, logger));
        }
        CHECK_STATUS_RETURN(packParametersFromHostTensorsWithInstanceIdSort(densityParametersBuffer,
                                                                            radianceParametersBuffer,
                                                                            extraSignalParametersBuffer,
                                                                            cameraExtendedFeaturesParametersBuffer,
                                                                            lidarExtendedFeaturesParametersBuffer,
                                                                            m_halfPrecisionFeatures,
                                                                            processQueueHandle,
                                                                            logger));
    }

    if (densityParametersBuffer->size() != sizeof(float) * densityParametersDim() * m_particlesNumber) {
        RETURN_ERROR(logger, ErrorCode::BadInput, "NREDynamicSHGaussianModel : resource %s has a wrong size [%zu /%zu].", densityParametersKey().c_str(),
                     densityParametersBuffer->size(), sizeof(float) * densityParametersDim() * m_particlesNumber);
    }

    if (radianceParametersBuffer->size() != radianceParametersTypeSize() * radianceParametersDim() * m_particlesNumber) {
        RETURN_ERROR(logger, ErrorCode::BadInput, "NREDynamicSHGaussianModel : resource %s has a wrong size [%zu /%zu].", radianceParametersKey().c_str(),
                     radianceParametersBuffer->size(), radianceParametersTypeSize() * radianceParametersDim() * m_particlesNumber);
    }

    if (m_extendedFeaturesEnabled) {
        if (extraSignalParametersBuffer->size() != extendedFeaturesParametersTypeSize() * extendedFeaturesParametersDim() * m_particlesNumber) {
            RETURN_ERROR(logger, ErrorCode::BadInput, "NREDynamicSHGaussianModel : resource %s has a wrong size [%zu /%zu].", extraSignalParametersKey().c_str(),
                         extraSignalParametersBuffer->size(), extendedFeaturesParametersTypeSize() * extendedFeaturesParametersDim() * m_particlesNumber);
        }
    }

    if (m_sensorExtendedFeaturesEnabled) {
        if (cameraExtendedFeaturesParametersBuffer->size() != cameraExtendedFeaturesParametersTypeSize() * cameraExtendedFeaturesParametersDim() * m_particlesNumber) {
            RETURN_ERROR(logger, ErrorCode::BadInput, "NREDynamicSHGaussianModel : resource %s has a wrong size [%zu /%zu].", cameraExtendedFeaturesParametersKey().c_str(),
                         cameraExtendedFeaturesParametersBuffer->size(), cameraExtendedFeaturesParametersTypeSize() * cameraExtendedFeaturesParametersDim() * m_particlesNumber);
        }
        if (lidarExtendedFeaturesParametersBuffer->size() != lidarExtendedFeaturesParametersTypeSize() * lidarExtendedFeaturesParametersDim() * m_particlesNumber) {
            RETURN_ERROR(logger, ErrorCode::BadInput, "NREDynamicSHGaussianModel : resource %s has a wrong size [%zu /%zu].", lidarExtendedFeaturesParametersKey().c_str(),
                         lidarExtendedFeaturesParametersBuffer->size(), lidarExtendedFeaturesParametersTypeSize() * lidarExtendedFeaturesParametersDim() * m_particlesNumber);
        }
    }

    return Status();
}

nrend::Status nrend::NREDeformableSHGaussianModel::registerModelKernelResources_(
    const KernelMemoryBindings& memoryBindings,
    const KernelSourceCodeTable& sourceCodeTable,
    KernelResourcesProvider::KernelOpts kernelOpts,
    const Logger& logger) const {

    const std::string cudaSourceCodeTemplate = R"(
        #include <nrend/kernels/cuda/models/nreDynamicShGaussianModel.cuh>

        {NREInstancesExtentClassDefinition}

        struct {NREDeformableShGaussianAlias}Params {{
            static constexpr bool UseDeformNetwork            = {UseDeformNetwork};
            static constexpr bool DeformPositions             = {DeformPositions};
            static constexpr bool DeformRotations             = {DeformRotations};
            static constexpr bool DeformRotationsFromIdentity = {DeformRotationsFromIdentity};
            static constexpr bool DeformScales                = {DeformScales};
        }};

        using {NREDeformableShGaussianAlias} = NREDeformableShGaussian<{NREDeformableShGaussianAlias}Particles,
                                                                       {NREInstancesExtentClassAlias},
                                                                       {DeformNetworkAlias},
                                                                       {NREDeformableShGaussianAlias}Params>;
    )";
    sourceCodeTable.registerKernel(
        KernelSourceCodeTable::Cuda,
        fmt::format(cudaSourceCodeTemplate,
                    fmt::arg("NREInstancesExtentClassDefinition", m_instancesExtent.sourceDefinition(cudaCallPrefix() + "_instances__extent")),
                    fmt::arg("NREInstancesExtentClassAlias", cudaCallPrefix() + "_instances__extent"),
                    fmt::arg("NREDeformableShGaussianAlias", cudaCallPrefix()),
                    fmt::arg("DeformNetworkAlias", cudaCallPrefix("deform_network.feature_volume")),
                    fmt::arg("UseDeformNetwork", m_useDeformNetwork),
                    fmt::arg("DeformPositions", m_deformNetworkSettings.deformPositions),
                    fmt::arg("DeformRotations", m_deformNetworkSettings.deformRotations),
                    fmt::arg("DeformRotationsFromIdentity", m_deformNetworkSettings.deformRotationsFromIdentity),
                    fmt::arg("DeformScales", m_deformNetworkSettings.deformScales)));

    return Status();
}
