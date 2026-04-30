// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include <nrend/models/nreShGaussianModel.h>

#include <nrend/kernels/cuda/common/cudaMath.cuh>

#include <tiny-cuda-nn/common.h>

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <limits>

namespace {

inline bool invalidIndex(int32_t index, int32_t size) {
    return (index < 0) || (index >= size);
}

#define RETURN_ERROR_IF_INVALID_INDEX(index, size, logger)                                \
    RETURN_ERROR_IF(invalidIndex(index, size), logger, nrend::ErrorCode::InvalidResource, \
                    "NRESHGaussianModel : invalid index. [%d / %d]", index, static_cast<int>(size));

#define RETURN_ERROR_IF_INVALID_INDEX_PTR(index, array, logger)                 \
    RETURN_ERROR_IF(invalidIndex(index, array.size()) || !array[index], logger, \
                    nrend::ErrorCode::InvalidResource,                          \
                    "NRESHGaussianModel : invalid index. [%d / %zu]", index, array.size());

#define RETURN_ERROR_IF_INVALID_CAST_PTR(ptr, logger)                \
    RETURN_ERROR_IF(!ptr, logger, nrend::ErrorCode::InvalidResource, \
                    "NRESHGaussianModel : invalid memory cast type.");

inline std::string getExtendedFeaturesType(int dim, int degree, bool enabled) {
    if (enabled && dim > 0) {
        if (degree > 0) {
            return fmt::format("ShVector<{Dim},{Degree},{NumCoefficients}>", fmt::arg("Dim", dim), fmt::arg("Degree", degree), fmt::arg("NumCoefficients", (degree + 1) * (degree + 1)));
        } else {
            return fmt::format("Vector<{Dim}>", fmt::arg("Dim", dim));
        }
    }
    return "Vector<1>";
}

inline std::string getExtendedFeaturesTypeMacro(int dim, int degree, bool enabled) {
    return (enabled && degree > 0) ? "ParticleFeaturesSphericalHarmonicsEntryPoints" : "ParticleFeaturesVectorEntryPoints";
}

struct ParticleDensity {
    tcnn::vec3 position;
    float density;
    tcnn::vec4 quaternion;
    tcnn::vec3 scale;
    float padding;
};

__global__ void packParticleDensityFromParameters(
    uint32_t numParticles,
    const tcnn::tvec<__half, 3>* __restrict__ particlesPosition,
    const tcnn::tvec<__half, 3>* __restrict__ particlesScale,
    const tcnn::tvec<__half, 4>* __restrict__ particlesRotation,
    const __half* __restrict__ particlesFactor,
    ParticleDensity* __restrict__ particlesDensity,
    const uint32_t* __restrict__ resortedParticlesIdx,
    const uint8_t bakeActivation) {
    const uint32_t particleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (particleIdx < numParticles) {
        const uint32_t unsortedParticleIdx = resortedParticlesIdx ? resortedParticlesIdx[particleIdx] : particleIdx;
        // clang-format off
        particlesDensity[particleIdx] = ParticleDensity{
            particlesPosition[unsortedParticleIdx],
            // TODO : generic density activation
            bakeActivation & nrend::NRESHGaussianModel::BakeActivationDensity ? 
                1.0f / (1.0f + tcnn::exp(__half2float(-particlesFactor[unsortedParticleIdx]))) : 
                __half2float(particlesFactor[unsortedParticleIdx]), ///< logistic activation
            bakeActivation & nrend::NRESHGaussianModel::BakeActivationRotation ? 
                tcnn::normalize(tcnn::vec4(particlesRotation[unsortedParticleIdx])) : 
                tcnn::vec4(particlesRotation[unsortedParticleIdx]), ///< quaternion normalization activation
            // TODO : generic scale activation
            bakeActivation & nrend::NRESHGaussianModel::BakeActivationScale ? 
                tcnn::exp(tcnn::vec3(particlesScale[unsortedParticleIdx])) : 
                tcnn::vec3(particlesScale[unsortedParticleIdx]), ///< exponential activation
            0.f};
        // clang-format on
    }
}

template <typename TIn, typename TOut>
__global__ void packParticleRadianceFromParameters(
    uint32_t numParticles,
    uint32_t radianceDim,
    const TIn* __restrict__ particlesAlbedoCoefficients,
    uint32_t numSpecularCoefficients,
    const TIn* __restrict__ particlesSpecularCoefficients,
    TOut* __restrict__ particlesSphCoefficients,
    const uint32_t* __restrict__ resortedParticlesIdx) {
    const uint32_t particleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (particleIdx < numParticles) {
        const uint32_t unsortedParticleIdx      = resortedParticlesIdx ? resortedParticlesIdx[particleIdx] : particleIdx;
        const uint32_t coefficientsOffset       = particleIdx * (1 + numSpecularCoefficients) * radianceDim;
        const uint32_t albedoCoefficientsOffset = unsortedParticleIdx * radianceDim;
        for (uint32_t i = 0; i < radianceDim; ++i) {
            particlesSphCoefficients[coefficientsOffset + i] = static_cast<TOut>(particlesAlbedoCoefficients[albedoCoefficientsOffset + i]);
        }
        const uint32_t specularCoefficientsOffset = unsortedParticleIdx * numSpecularCoefficients * radianceDim;
        for (uint32_t i = 0; i < numSpecularCoefficients * radianceDim; ++i) {
            particlesSphCoefficients[coefficientsOffset + radianceDim + i] = static_cast<TOut>(particlesSpecularCoefficients[specularCoefficientsOffset + i]);
        }
    }
}

template <typename TIn, typename TOut>
__global__ void packParticleExtraSignalsFromParameters(
    uint32_t numParticles,
    uint32_t extraSignalDim,
    const TIn* __restrict__ particlesExtraSignals,
    TOut* __restrict__ particlesExtraSignalsPacked,
    const uint32_t* __restrict__ resortedParticlesIdx) {
    const uint32_t particleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (particleIdx < numParticles) {
        const uint32_t unsortedParticleIdx       = resortedParticlesIdx ? resortedParticlesIdx[particleIdx] : particleIdx;
        const uint32_t extraSignalOffset         = particleIdx * extraSignalDim;
        const uint32_t unsortedExtraSignalOffset = unsortedParticleIdx * extraSignalDim;
        for (uint32_t i = 0; i < extraSignalDim; ++i) {
            particlesExtraSignalsPacked[extraSignalOffset + i] = static_cast<TOut>(particlesExtraSignals[unsortedExtraSignalOffset + i]);
        }
    }
}

__device__ inline uint64_t expandUInt21(uint32_t b) {
    uint64_t eb = b & 0x1fffff;
    eb          = (eb | (eb << 32)) & 0x1f00000000ffff;
    eb          = (eb | (eb << 16)) & 0x1f0000ff0000ff;
    eb          = (eb | (eb << 8)) & 0x100f00f00f00f00f;
    eb          = (eb | (eb << 4)) & 0x10c30c30c30c30c3;
    eb          = (eb | (eb << 2)) & 0x1249249249249249;
    return eb;
}

__forceinline__ __device__ uint64_t morton3d63(tcnn::uvec3 position) {
    return expandUInt21(position.x) | (expandUInt21(position.y) << 1) | (expandUInt21(position.z) << 2);
}

template <int BLOCK_SIZE>
__global__ void morton3d63Preprocess(uint32_t num,
                                     const tcnn::tvec<__half, 3>* __restrict__ positions,
                                     tcnn::vec3* __restrict__ aabb,
                                     uint32_t* __restrict__ index) {
    // Allocate shared memory for CUB block reduce
    __shared__ typename cub::BlockReduce<tcnn::vec3, BLOCK_SIZE>::TempStorage min_temp_storage;
    __shared__ typename cub::BlockReduce<tcnn::vec3, BLOCK_SIZE>::TempStorage max_temp_storage;

    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize thread data for min/max reduction with appropriate extremes
    tcnn::vec3 thread_min = tcnn::vec3(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    tcnn::vec3 thread_max = tcnn::vec3(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());

    // Only update min/max for valid indices
    if (idx < num) {
        const tcnn::vec3 position = tcnn::vec3(positions[idx]);
        thread_min                = position;
        thread_max                = position;
    }

    // Perform block-wide reduction for min and max
    tcnn::vec3 block_min = cub::BlockReduce<tcnn::vec3, BLOCK_SIZE>(min_temp_storage).Reduce(thread_min, [] __device__(const tcnn::vec3& a, const tcnn::vec3& b) {
        return tcnn::min(a, b);
    });

    tcnn::vec3 block_max = cub::BlockReduce<tcnn::vec3, BLOCK_SIZE>(max_temp_storage).Reduce(thread_max, [] __device__(const tcnn::vec3& a, const tcnn::vec3& b) {
        return tcnn::max(a, b);
    });

    // First thread in block performs atomic min/max to global memory
    if (threadIdx.x == 0) {
        atomicMinFloat(&aabb[0].x, block_min.x);
        atomicMinFloat(&aabb[0].y, block_min.y);
        atomicMinFloat(&aabb[0].z, block_min.z);
        atomicMaxFloat(&aabb[1].x, block_max.x);
        atomicMaxFloat(&aabb[1].y, block_max.y);
        atomicMaxFloat(&aabb[1].z, block_max.z);
    }

    // Store index if within bounds
    if (idx < num) {
        index[idx] = idx;
    }
}

__global__ void morton3d63Compute(const tcnn::vec3* __restrict__ aabb,
                                  uint32_t num,
                                  const tcnn::tvec<__half, 3>* __restrict__ positions,
                                  uint64_t* __restrict__ m3d63s) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num) {
        constexpr uint32_t resolution       = 0x001FFFFF; //< 21 bits
        const tcnn::vec3 positionNormalized = (tcnn::vec3(positions[idx]) - aabb[0]) / (aabb[1] - aabb[0]);
        m3d63s[idx]                         = morton3d63(tcnn::uvec3(positionNormalized.x * resolution,
                                                                     positionNormalized.y * resolution,
                                                                     positionNormalized.z * resolution));
    }
}
}; // namespace

nrend::Status nrend::NRESHGaussianModel::packParametersFromHostTensors(CudaBuffer& densityParamsBuffer,
                                                                       CudaBuffer& radianceParamsBuffer,
                                                                       CudaBuffer& extraSignalParamsBuffer,
                                                                       CudaBuffer& cameraExtendedFeaturesParamsBuffer,
                                                                       CudaBuffer& lidarExtendedFeaturesParamsBuffer,
                                                                       const tcnn::tvec<__half, 3>* positionsPtr,
                                                                       const uint32_t* resortingIndicesPtr,
                                                                       bool halfPrecisionFeatures,
                                                                       uint64_t processQueueHandle,
                                                                       const Logger& logger) const {
    const int threads       = 1024;
    const int blocks        = tcnn::div_round_up<int>(m_particlesNumber, threads);
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(processQueueHandle);
    {
        ScopedCudaBuffer scales(processQueueHandle);
        scales.setFromHost(m_paramsTensor[Scales].buffer.data(), m_paramsTensor[Scales].buffer.size(), logger);
        ScopedCudaBuffer rotations(processQueueHandle);
        rotations.setFromHost(m_paramsTensor[Rotations].buffer.data(), m_paramsTensor[Rotations].buffer.size(), logger);
        ScopedCudaBuffer densities(processQueueHandle);
        densities.setFromHost(m_paramsTensor[Densities].buffer.data(), m_paramsTensor[Densities].buffer.size(), logger);

        packParticleDensityFromParameters<<<blocks, threads, 0, cudaStream>>>(
            m_particlesNumber,
            positionsPtr,
            scales.ptr<const tcnn::tvec<__half, 3>>(),
            rotations.ptr<const tcnn::tvec<__half, 4>>(),
            densities.ptr<const __half>(),
            reinterpret_cast<ParticleDensity*>(densityParamsBuffer.data()),
            resortingIndicesPtr,
            m_bakeActivation);
        CUDA_CHECK_STREAM_RETURN(cudaStream, logger);
    }

    {
        const size_t albedoBufferSize = sizeof(__half) * m_radianceDim * m_particlesNumber;
        if (m_paramsTensor[SphAlbedos].buffer.size() != albedoBufferSize) {
            RETURN_ERROR(logger, ErrorCode::BadInput, "NRESHGaussianModel : input albedo data has wrong size [%d/%d].",
                         static_cast<int>(m_paramsTensor[SphAlbedos].buffer.size()), static_cast<int>(albedoBufferSize));
        }
        const size_t specularBufferSize = sizeof(__half) * (m_radianceMaxNumCoefficients - 1) * m_radianceDim * m_particlesNumber;
        if (m_paramsTensor[SphSpeculars].buffer.size() != specularBufferSize) {
            RETURN_ERROR(logger, ErrorCode::BadInput, "NRESHGaussianModel : input specular data has wrong size [%d/%d].",
                         static_cast<int>(m_paramsTensor[SphSpeculars].buffer.size()), static_cast<int>(specularBufferSize));
        }

        ScopedCudaBuffer albedoBuffer(processQueueHandle);
        albedoBuffer.setFromHost(m_paramsTensor[SphAlbedos].buffer.data(), m_paramsTensor[SphAlbedos].buffer.size(), logger);
        ScopedCudaBuffer specularBuffer(processQueueHandle);
        specularBuffer.setFromHost(m_paramsTensor[SphSpeculars].buffer.data(), m_paramsTensor[SphSpeculars].buffer.size(), logger);

        if (halfPrecisionFeatures) {
            packParticleRadianceFromParameters<__half, __half><<<blocks, threads, 0, cudaStream>>>(
                m_particlesNumber,
                m_radianceDim,
                albedoBuffer.ptr<__half>(),
                m_radianceMaxNumCoefficients - 1,
                specularBuffer.ptr<__half>(),
                radianceParamsBuffer.ptr<__half>(),
                resortingIndicesPtr);
        } else {
            packParticleRadianceFromParameters<__half, float><<<blocks, threads, 0, cudaStream>>>(
                m_particlesNumber,
                m_radianceDim,
                albedoBuffer.ptr<__half>(),
                m_radianceMaxNumCoefficients - 1,
                specularBuffer.ptr<__half>(),
                radianceParamsBuffer.ptr<float>(),
                resortingIndicesPtr);
        }
        CUDA_CHECK_STREAM_RETURN(cudaStream, logger);
    }

    if (m_extendedFeaturesEnabled) {
        ScopedCudaBuffer extraSignalsBuffer(processQueueHandle);
        extraSignalsBuffer.setFromHost(m_paramsTensor[ExtraSignals].buffer.data(), m_paramsTensor[ExtraSignals].buffer.size(), logger);
        if (halfPrecisionFeatures) {
            packParticleExtraSignalsFromParameters<__half, __half><<<blocks, threads, 0, cudaStream>>>(
                m_particlesNumber,
                extendedFeaturesParametersDim(),
                extraSignalsBuffer.ptr<__half>(),
                extraSignalParamsBuffer.ptr<__half>(),
                resortingIndicesPtr);
        } else {
            packParticleExtraSignalsFromParameters<__half, float><<<blocks, threads, 0, cudaStream>>>(
                m_particlesNumber,
                extendedFeaturesParametersDim(),
                extraSignalsBuffer.ptr<__half>(),
                extraSignalParamsBuffer.ptr<float>(),
                resortingIndicesPtr);
        }
        CUDA_CHECK_STREAM_RETURN(cudaStream, logger);
    }
    if (m_sensorExtendedFeaturesEnabled) {
        ScopedCudaBuffer cameraExtendedFeaturesBuffer(processQueueHandle);
        cameraExtendedFeaturesBuffer.setFromHost(m_paramsTensor[CameraExtraSignals].buffer.data(), m_paramsTensor[CameraExtraSignals].buffer.size(), logger);
        ScopedCudaBuffer lidarExtendedFeaturesBuffer(processQueueHandle);
        lidarExtendedFeaturesBuffer.setFromHost(m_paramsTensor[LidarExtraSignals].buffer.data(), m_paramsTensor[LidarExtraSignals].buffer.size(), logger);
        if (halfPrecisionFeatures) {
            packParticleExtraSignalsFromParameters<__half, __half><<<blocks, threads, 0, cudaStream>>>(
                m_particlesNumber,
                cameraExtendedFeaturesParametersDim(),
                cameraExtendedFeaturesBuffer.ptr<__half>(),
                cameraExtendedFeaturesParamsBuffer.ptr<__half>(),
                resortingIndicesPtr);
            packParticleExtraSignalsFromParameters<__half, __half><<<blocks, threads, 0, cudaStream>>>(
                m_particlesNumber,
                lidarExtendedFeaturesParametersDim(),
                lidarExtendedFeaturesBuffer.ptr<__half>(),
                lidarExtendedFeaturesParamsBuffer.ptr<__half>(),
                resortingIndicesPtr);
        } else {
            packParticleExtraSignalsFromParameters<__half, float><<<blocks, threads, 0, cudaStream>>>(
                m_particlesNumber,
                cameraExtendedFeaturesParametersDim(),
                cameraExtendedFeaturesBuffer.ptr<__half>(),
                cameraExtendedFeaturesParamsBuffer.ptr<float>(),
                resortingIndicesPtr);
            packParticleExtraSignalsFromParameters<__half, float><<<blocks, threads, 0, cudaStream>>>(
                m_particlesNumber,
                lidarExtendedFeaturesParametersDim(),
                lidarExtendedFeaturesBuffer.ptr<__half>(),
                lidarExtendedFeaturesParamsBuffer.ptr<float>(),
                resortingIndicesPtr);
        }
    }

    return Status();
}

nrend::Status nrend::NRESHGaussianModel::packParametersFromHostTensorsWithMortonSort(CudaBuffer* densityParamsPtr,
                                                                                     CudaBuffer* radianceParamsPtr,
                                                                                     CudaBuffer* extraSignalParamsPtr,
                                                                                     CudaBuffer* cameraExtendedFeaturesParamsPtr,
                                                                                     CudaBuffer* lidarExtendedFeaturesParamsPtr,
                                                                                     bool halfPrecisionFeatures,
                                                                                     uint64_t processQueueHandle,
                                                                                     const Logger& logger) const {

    RETURN_ERROR_IF(!densityParamsPtr || !radianceParamsPtr || !extraSignalParamsPtr, logger, ErrorCode::InvalidResource,
                    "NRESHGaussianModel : invalid cuda buffers.");

    const int threads       = 1024;
    const int blocks        = tcnn::div_round_up<int>(m_particlesNumber, threads);
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(processQueueHandle);

    ScopedCudaBuffer positions(processQueueHandle);
    const size_t positionsBufferSize = sizeof(__half) * 3 * m_particlesNumber;
    if (m_paramsTensor[Positions].buffer.size() != positionsBufferSize) {
        RETURN_ERROR(logger, ErrorCode::BadInput, "NRESHGaussianModel : input positions data has wrong size [%d/%d].",
                     static_cast<int>(m_paramsTensor[Positions].buffer.size()), static_cast<int>(positionsBufferSize));
    }
    positions.setFromHost(m_paramsTensor[Positions].buffer.data(), m_paramsTensor[Positions].buffer.size(), logger);
    ScopedCudaBuffer sortedParticlesIdx(processQueueHandle);

    if (m_morton3DParticleSort) {
        ScopedCudaBuffer particlesIdx(processQueueHandle);
        CHECK_STATUS_RETURN(particlesIdx.resize(m_particlesNumber * sizeof(uint32_t), logger));

        ScopedCudaBuffer aabb(processQueueHandle);
        CHECK_STATUS_RETURN(aabb.resize(2 * sizeof(tcnn::vec3), logger));
        const tcnn::vec3 aabbHost[2] = {{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()},
                                        {-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()}};
        CUDA_CHECK_RETURN(cudaMemcpyAsync(aabb.ptr<void>(), aabbHost, sizeof(aabbHost), cudaMemcpyHostToDevice, cudaStream), logger);

        morton3d63Preprocess<threads><<<blocks, threads, 0, cudaStream>>>(m_particlesNumber,
                                                                          reinterpret_cast<const tcnn::tvec<__half, 3>*>(positions.data()),
                                                                          aabb.ptr<tcnn::vec3>(),
                                                                          particlesIdx.ptr<uint32_t>());
        CUDA_CHECK_STREAM_RETURN(cudaStream, logger);

        ScopedCudaBuffer mortonIndex(processQueueHandle);
        CHECK_STATUS_RETURN(mortonIndex.resize(m_particlesNumber * sizeof(uint64_t), logger));

        morton3d63Compute<<<blocks, threads, 0, cudaStream>>>(aabb.ptr<tcnn::vec3>(),
                                                              m_particlesNumber,
                                                              positions.ptr<tcnn::tvec<__half, 3>>(),
                                                              mortonIndex.ptr<uint64_t>());
        CUDA_CHECK_STREAM_RETURN(cudaStream, logger);

        ScopedCudaBuffer sortedMortonIndex(processQueueHandle);
        CHECK_STATUS_RETURN(sortedMortonIndex.resize(m_particlesNumber * sizeof(uint64_t), logger));
        CHECK_STATUS_RETURN(sortedParticlesIdx.resize(m_particlesNumber * sizeof(uint32_t), logger));

        size_t sortingWorkingBufferSize = 0;
        CUDA_CHECK_RETURN(cub::DeviceRadixSort::SortPairs(nullptr,
                                                          sortingWorkingBufferSize,
                                                          mortonIndex.ptr<uint64_t>(),
                                                          sortedMortonIndex.ptr<uint64_t>(),
                                                          particlesIdx.ptr<uint32_t>(),
                                                          sortedParticlesIdx.ptr<uint32_t>(),
                                                          m_particlesNumber,
                                                          0, sizeof(uint64_t) * 8,
                                                          cudaStream),
                          logger);
        ScopedCudaBuffer sortingWorkingBuffer(processQueueHandle);
        CHECK_STATUS_RETURN(sortingWorkingBuffer.resize(sortingWorkingBufferSize, logger));

        CUDA_CHECK_RETURN(cub::DeviceRadixSort::SortPairs(sortingWorkingBuffer.ptr<void>(),
                                                          sortingWorkingBufferSize,
                                                          mortonIndex.ptr<uint64_t>(),
                                                          sortedMortonIndex.ptr<uint64_t>(),
                                                          particlesIdx.ptr<uint32_t>(),
                                                          sortedParticlesIdx.ptr<uint32_t>(),
                                                          m_particlesNumber,
                                                          0, sizeof(uint64_t) * 8, cudaStream),
                          logger);
    }

    return packParametersFromHostTensors(*densityParamsPtr,
                                         *radianceParamsPtr,
                                         *extraSignalParamsPtr,
                                         *cameraExtendedFeaturesParamsPtr,
                                         *lidarExtendedFeaturesParamsPtr,
                                         positions.ptr<tcnn::tvec<__half, 3>>(),
                                         sortedParticlesIdx.ptr<uint32_t>(),
                                         halfPrecisionFeatures,
                                         processQueueHandle,
                                         logger);
}

nrend::NRESHGaussianModel::NRESHGaussianModel(const nlohmann::json& config,
                                              const Logger& logger,
                                              const nlohmann::json& stateDict,
                                              const std::string& prefix,
                                              const std::vector<const char*>& submodelCStr)
    : NREModel(config, logger, stateDict, prefix, submodelCStr) {

    if (config.contains("particle")) {
        const auto particleConfig  = config["particle"];
        m_densityKernelPlanar      = particleConfig.value("density_kernel_planar", m_densityKernelPlanar);
        m_densityKernelDegree      = particleConfig.value("density_kernel_degree", m_densityKernelDegree);
        m_densityKernelMinResponse = particleConfig.value("density_kernel_min_response", m_densityKernelMinResponse);
        m_raySpreadFilterEnabled   = particleConfig.value("ray_spread_filter_enabled", m_raySpreadFilterEnabled);
        m_radianceSphO0            = particleConfig.value("radiance_sph_O0", m_radianceSphO0);
        m_radianceSphDegree        = particleConfig.value("radiance_sph_degree", m_radianceSphDegree);
        m_radianceDim              = particleConfig.value("radiance_dimension", m_radianceDim);
        m_radianceFourierDim       = particleConfig.value("fourier_features_dim", m_radianceFourierDim);
        if (m_radianceFourierDim != 1) {
            LOG_ERROR(logger, "NRESHGaussianModel : unsupported fourier_features_dim %d.", m_radianceFourierDim);
        }
        m_radianceMaxNumCoefficients      = (m_radianceSphDegree + 1) * (m_radianceSphDegree + 1);
        m_extendedFeaturesSphDegree       = particleConfig.value("extra_signal_sph_degree", m_extendedFeaturesSphDegree);
        m_extendedFeaturesDim             = particleConfig.value("extra_signal_dim", m_extendedFeaturesDim);
        m_cameraExtendedFeaturesSphDegree = particleConfig.value("camera_extra_signal_sph_degree", m_cameraExtendedFeaturesSphDegree);
        m_cameraExtendedFeaturesDim       = particleConfig.value("camera_extra_signal_dim", m_cameraExtendedFeaturesDim);
        m_lidarExtendedFeaturesSphDegree  = particleConfig.value("lidar_extra_signal_sph_degree", m_lidarExtendedFeaturesSphDegree);
        m_lidarExtendedFeaturesDim        = particleConfig.value("lidar_extra_signal_dim", m_lidarExtendedFeaturesDim);
    }
    m_transmittanceThreshold = config.value("transmittance_threshold", m_transmittanceThreshold);
    m_morton3DParticleSort   = config.value("morton3d_particle_sort", m_morton3DParticleSort);
    // TODO : generic density activation
    if (config.value("density_activation", "sigmoid") != "sigmoid") {
        LOG_ERROR(logger, "NRESHGaussianModel : unsupported density_activation %s.", static_cast<std::string>(config["density_activation"]).c_str());
    }
    // TODO : generic scale activation
    if (config.value("scale_activation", "exp") != "exp") {
        LOG_ERROR(logger, "NRESHGaussianModel : unsupported scale_activation %s.", static_cast<std::string>(config["scale_activation"]).c_str());
    }
    if (config.value("rotation_activation", "normalize") != "normalize") {
        LOG_ERROR(logger, "NRESHGaussianModel : unsupported rotation_activation %s.", static_cast<std::string>(config["rotation_activation"]).c_str());
    }

    auto isExpectedMissingTensor = [&](int p) {
        return (p == ExtraSignals) && (m_extendedFeaturesDim == 0) ||
               (p == CameraExtraSignals) && (m_cameraExtendedFeaturesDim == 0) ||
               (p == LidarExtraSignals) && (m_lidarExtendedFeaturesDim == 0);
    };

    m_particlesNumber = 0;
    for (int p = 0; p < NumParameters; ++p) {
        m_paramsTensor[p].key = prefix + s_parametersKey[p];
        if (!readStateDictTensor(stateDict, m_paramsTensor[p])) {
            LOG_WARN(logger,
                     "NRESHGaussianModel : missing parameters tensor <%s> in the state_dict.",
                     m_paramsTensor[p].key.c_str());
            m_validInitialParameters = m_validInitialParameters && isExpectedMissingTensor(p);
        } else if (m_validInitialParameters) {
            if (p == Positions) {
                m_particlesNumber        = m_paramsTensor[p].shape[0];
                m_validInitialParameters = m_particlesNumber > 0;
            } else if (m_particlesNumber != m_paramsTensor[p].shape[0]) {
                LOG_WARN(logger,
                         "NRESHGaussianModel : inconsistent parameters tensor <%s> in the state_dict : number of particles [%d/%d]",
                         m_paramsTensor[p].key.c_str(),
                         m_paramsTensor[p].shape[0],
                         m_particlesNumber);
                m_validInitialParameters = false;
            }
            if ((p == ExtraSignals) && (m_paramsTensor[p].shape[1] != extendedFeaturesParametersDim())) {
                LOG_WARN(logger,
                         "NRESHGaussianModel : inconsistent parameters tensor <%s> in the state_dict : number of extra signals [%d/%d]",
                         m_paramsTensor[p].key.c_str(),
                         m_paramsTensor[p].shape[0],
                         extendedFeaturesParametersDim());
                m_validInitialParameters = false;
            }
            LOG_DEBUG(logger, "Parsed model tensor parameters %s with shape (%d, %d, %d, %d) [%zu]",
                      m_paramsTensor[p].key.c_str(),
                      m_paramsTensor[p].shape.size() > 0 ? m_paramsTensor[p].shape[0] : 0,
                      m_paramsTensor[p].shape.size() > 1 ? m_paramsTensor[p].shape[1] : 0,
                      m_paramsTensor[p].shape.size() > 2 ? m_paramsTensor[p].shape[2] : 0,
                      m_paramsTensor[p].shape.size() > 3 ? m_paramsTensor[p].shape[3] : 0,
                      m_paramsTensor[p].buffer.size());
        }
    }
    LOG_INFO(logger, "Parsed model %s with %d particles", m_callPrefix.c_str(), m_particlesNumber);
}

nrend::Status nrend::NRESHGaussianModel::registerParticleKernelResources_(
    const KernelMemoryBindings& memoryBindings,
    const KernelSourceCodeTable& sourceCodeTable,
    KernelResourcesProvider::KernelOpts kernelOpts,
    const Logger& logger) const {

    m_halfPrecisionFeatures         = m_halfPrecisionFeatures && !(kernelOpts & KernelResourcesProvider::Differentiable);
    m_extendedFeaturesEnabled       = !(kernelOpts & KernelResourcesProvider::DisableExtendedFeatures) && (m_extendedFeaturesDim > 0);
    m_sensorExtendedFeaturesEnabled = !(kernelOpts & KernelResourcesProvider::DisableSensorExtendedFeatures) &&
                                      (m_cameraExtendedFeaturesDim + m_lidarExtendedFeaturesDim > 0);

    CHECK_STATUS_RETURN(memoryBindings.registerValue(particlesNumberParameterKey(), m_particlesNumber, logger));
    CHECK_STATUS_RETURN(memoryBindings.registerValue(radianceActiveShDegreesParameterKey(), m_radianceSphDegree, logger));
    CHECK_STATUS_RETURN(memoryBindings.registerMemory(KernelMemoryBindings::Parameters,
                                                      densityParametersKey(),
                                                      KernelMemoryType::Buffer,
                                                      logger));
    CHECK_STATUS_RETURN(memoryBindings.registerMemory(KernelMemoryBindings::Parameters,
                                                      radianceParametersKey(),
                                                      KernelMemoryType::Buffer,
                                                      logger));

    // Need to register the extra signal parameters even if they are not enabled (for the update model to work)
    CHECK_STATUS_RETURN(
        memoryBindings.registerMemory(KernelMemoryBindings::Parameters,
                                      extraSignalParametersKey(),
                                      KernelMemoryType::Buffer,
                                      logger));
    CHECK_STATUS_RETURN(
        memoryBindings.registerMemory(KernelMemoryBindings::Parameters,
                                      cameraExtendedFeaturesParametersKey(),
                                      KernelMemoryType::Buffer,
                                      logger));
    CHECK_STATUS_RETURN(
        memoryBindings.registerMemory(KernelMemoryBindings::Parameters,
                                      lidarExtendedFeaturesParametersKey(),
                                      KernelMemoryType::Buffer,
                                      logger));

    if (kernelOpts & KernelResourcesProvider::Differentiable) {
        CHECK_STATUS_RETURN(
            memoryBindings.registerMemory(KernelMemoryBindings::BindingsFlag::ParameterGradients,
                                          densityParametersKey(),
                                          KernelMemoryType::Buffer,
                                          logger));
        CHECK_STATUS_RETURN(
            memoryBindings.registerMemory(KernelMemoryBindings::ParameterGradients,
                                          radianceParametersKey(),
                                          KernelMemoryType::Buffer,
                                          logger));

        CHECK_STATUS_RETURN(
            memoryBindings.registerMemory(KernelMemoryBindings::ParameterGradients,
                                          extraSignalParametersKey(),
                                          KernelMemoryType::Buffer,
                                          logger));
        CHECK_STATUS_RETURN(
            memoryBindings.registerMemory(KernelMemoryBindings::ParameterGradients,
                                          cameraExtendedFeaturesParametersKey(),
                                          KernelMemoryType::Buffer,
                                          logger));
        CHECK_STATUS_RETURN(
            memoryBindings.registerMemory(KernelMemoryBindings::ParameterGradients,
                                          lidarExtendedFeaturesParametersKey(),
                                          KernelMemoryType::Buffer,
                                          logger));
    }

    // FIXME : Slang integration does not permits defining different gaussian particles with different
    //         parameters (e.g. different density kernel degree)
    // TODO : Find a way to use generics with cudaDeviceExport functions...

    const std::string gaussianParticlesSlangTemplate = R"(
        #ifndef __gaussianParticles_slang
        #define __gaussianParticles_slang

        namespace gaussianParticle 
        {{
            static const int KernelDegree               = {KernelDegree};
            static const float KernelScale              = {KernelScale};
            static const float MinParticleKernelDensity = {MinKernelDensity};
            static const float MaxParticleAlpha         = {MaxAlpha};
            static const float MinParticleAlpha         = {MinAlpha};
            static const bool Surfel                    = {Surfel};
            static const bool RaySpreadFilterEnabled    = {RaySpreadFilterEnabled};
        }};

        #include "nrend/kernels/slang/models/gaussianParticles.slang"

        #endif
    )";
    const std::string gaussianParticlesSlangDefinition =
        fmt::format(gaussianParticlesSlangTemplate,
                    fmt::arg("KernelDegree", m_densityKernelDegree),
                    fmt::arg("KernelScale", m_densityKernelScale),
                    fmt::arg("MinKernelDensity", m_densityKernelMinResponse),
                    fmt::arg("MaxAlpha", m_alphaMaxValue),
                    fmt::arg("MinAlpha", m_alphaMinValue),
                    fmt::arg("Surfel", m_densityKernelPlanar),
                    fmt::arg("RaySpreadFilterEnabled", m_raySpreadFilterEnabled));

    const std::string shRadiativeParticleSlangTemplate = R"(
        #ifndef __shRadiativeParticles_slang
        #define __shRadiativeParticles_slang

        namespace shRadiativeParticle 
        {{
            static const bool Differentiable = {Differentiable};
            static const bool RadianceSphO0 = {RadianceSphO0};
        
            typedef {ShRadianceCoefficientsType} ShRadianceCoefficientsType;
            
            static const int RadianceMaxNumShCoefficients = {RadianceMaxNumShCoefficients};
            static const int Dim = {FeaturesDim};
        }};

        #include "nrend/kernels/slang/models/shRadiativeParticles.slang"

        #endif
    )";
    const std::string shRadiativeParticlesSlangDefinition =
        fmt::format(shRadiativeParticleSlangTemplate,
                    fmt::arg("Differentiable", static_cast<bool>(kernelOpts & KernelResourcesProvider::Differentiable)),
                    fmt::arg("RadianceSphO0", m_radianceSphO0),
                    fmt::arg("ShRadianceCoefficientsType", m_halfPrecisionFeatures ? "half" : "float"),
                    fmt::arg("RadianceMaxNumShCoefficients", m_radianceMaxNumCoefficients),
                    fmt::arg("FeaturesDim", m_radianceDim));

    const std::string extendedFeatureParticleSlangTemplate = R"(
        #ifndef __particleFeatures_slang
        #define __particleFeatures_slang
    
        namespace particleFeatures
        {{
            static const bool Differentiable = {Differentiable};
            typedef {ParametersType} ParametersType;
        }}
        
        #include "nrend/kernels/slang/models/particleFeatures.slang"

        namespace particleFeatures
        {{
            typealias ExtendedFeatures = {ExtendedFeaturesType};
            typealias CameraExtendedFeatures = {CameraExtendedFeaturesType};
            typealias LidarExtendedFeatures = {LidarExtendedFeaturesType};
        }};

        #include "nrend/kernels/slang/models/particleFeaturesCudaEntryPoints.slang"

        {ExtendedFeaturesTypeMacro}(particleExtendedFeatures, ExtendedFeatures);
        {CameraExtendedFeaturesTypeMacro}(particleCameraExtendedFeatures, CameraExtendedFeatures);
        {LidarExtendedFeaturesTypeMacro}(particleLidarExtendedFeatures, LidarExtendedFeatures);
    
        #endif
    )";
    const std::string extendedFeatureParticlesSlangDefinition =
        fmt::format(extendedFeatureParticleSlangTemplate,
                    fmt::arg("Differentiable", static_cast<bool>(kernelOpts & KernelResourcesProvider::Differentiable)),
                    fmt::arg("ParametersType", m_halfPrecisionFeatures ? "half" : "float"),
                    fmt::arg("ExtendedFeaturesType", getExtendedFeaturesType(m_extendedFeaturesDim, m_extendedFeaturesSphDegree, m_extendedFeaturesEnabled)),
                    fmt::arg("CameraExtendedFeaturesType", getExtendedFeaturesType(m_cameraExtendedFeaturesDim, m_cameraExtendedFeaturesSphDegree, m_sensorExtendedFeaturesEnabled)),
                    fmt::arg("LidarExtendedFeaturesType", getExtendedFeaturesType(m_lidarExtendedFeaturesDim, m_lidarExtendedFeaturesSphDegree, m_sensorExtendedFeaturesEnabled)),
                    fmt::arg("ExtendedFeaturesTypeMacro", getExtendedFeaturesTypeMacro(m_extendedFeaturesDim, m_extendedFeaturesSphDegree, m_extendedFeaturesEnabled)),
                    fmt::arg("CameraExtendedFeaturesTypeMacro", getExtendedFeaturesTypeMacro(m_cameraExtendedFeaturesDim, m_cameraExtendedFeaturesSphDegree, m_sensorExtendedFeaturesEnabled)),
                    fmt::arg("LidarExtendedFeaturesTypeMacro", getExtendedFeaturesTypeMacro(m_lidarExtendedFeaturesDim, m_lidarExtendedFeaturesSphDegree, m_sensorExtendedFeaturesEnabled)));

    const std::string slangSourceCodeTemplate = R"(
        {GaussianParticleDefinition}

        {ShRadiativeParticleDefinition}

        {ExtendedFeatureParticleDefinition}
    )";
    sourceCodeTable.registerKernel(
        KernelSourceCodeTable::Slang,
        fmt::format(slangSourceCodeTemplate,
                    fmt::arg("GaussianParticleDefinition", gaussianParticlesSlangDefinition),
                    fmt::arg("ShRadiativeParticleDefinition", shRadiativeParticlesSlangDefinition),
                    fmt::arg("ExtendedFeatureParticleDefinition", extendedFeatureParticlesSlangDefinition)));

    const std::string cudaSourceCodeTemplate = R"(
        struct {NREShGaussianAlias}InternalParams 
        {{
            static constexpr int DensityRawParametersBufferIndex                        = {DensityRawParametersBufferIndex};
            static constexpr int DensityRawParametersGradientBufferIndex                = {DensityRawParametersGradientBufferIndex};
            static constexpr int FeaturesRawParametersBufferIndex                       = {FeaturesRawParametersBufferIndex};
            static constexpr int FeaturesRawParametersGradientBufferIndex               = {FeaturesRawParametersGradientBufferIndex};
            static constexpr int ExtendedFeaturesRawParametersBufferIndex               = {ExtendedFeaturesRawParametersBufferIndex};
            static constexpr int ExtendedFeaturesRawParametersGradientBufferIndex       = {ExtendedFeaturesRawParametersGradientBufferIndex};
            static constexpr int CameraExtendedFeaturesRawParametersBufferIndex         = {CameraExtendedFeaturesRawParametersBufferIndex};
            static constexpr int CameraExtendedFeaturesRawParametersGradientBufferIndex = {CameraExtendedFeaturesRawParametersGradientBufferIndex};
            static constexpr int LidarExtendedFeaturesRawParametersBufferIndex          = {LidarExtendedFeaturesRawParametersBufferIndex};
            static constexpr int LidarExtendedFeaturesRawParametersGradientBufferIndex  = {LidarExtendedFeaturesRawParametersGradientBufferIndex};
            static constexpr int GlobalParametersValueBufferIndex                       = {GlobalParametersValueBufferIndex};
            static constexpr int FeatureShDegreeValueOffset                             = {FeatureShDegreeValueOffset};
        }};

        struct {NREShGaussianAlias}ExternalParams 
        {{
            static constexpr int FeaturesDim                         = {FeaturesDim};
            static constexpr int FeaturesParametersDim               = {FeaturesParametersDim};
            static constexpr int ExtendedFeaturesDim                 = {ExtendedFeaturesDim};
            static constexpr int ExtendedFeaturesParametersDim       = {ExtendedFeaturesParametersDim};
            static constexpr int CameraExtendedFeaturesDim           = {CameraExtendedFeaturesDim};
            static constexpr int CameraExtendedFeaturesParametersDim = {CameraExtendedFeaturesParametersDim};
            static constexpr int LidarExtendedFeaturesDim            = {LidarExtendedFeaturesDim};
            static constexpr int LidarExtendedFeaturesParametersDim  = {LidarExtendedFeaturesParametersDim};
            static constexpr float AlphaThreshold                    = {AlphaThreshold};
            static constexpr float MinTransmittanceThreshold         = {MinTransmittanceThreshold};
        }};

        #include <nrend/kernels/cuda/models/nreShRadiativeGaussianParticles.cuh>

        using {NREShGaussianAlias}Particles = NREShRadiativeGaussianVolumetricFeaturesParticles<{TDensityRawParameters},
                                                                                                {TDensityParameters},
                                                                                                {TFeaturesParameters},
                                                                                                {TFeaturesType},
                                                                                                {NREShGaussianAlias}InternalParams,
                                                                                                {NREShGaussianAlias}ExternalParams,
                                                                                                {TDifferentiable}>;
    )";
    sourceCodeTable.registerKernel(
        KernelSourceCodeTable::Cuda,
        fmt::format(cudaSourceCodeTemplate,
                    fmt::arg("NREShGaussianAlias", cudaCallPrefix()),
                    fmt::arg("DensityRawParametersBufferIndex", memoryBindings.registeredMemoryIndex(KernelMemoryBindings::Parameters, densityParametersKey())),
                    fmt::arg("DensityRawParametersGradientBufferIndex", memoryBindings.registeredMemoryIndex(KernelMemoryBindings::ParameterGradients, densityParametersKey())),
                    fmt::arg("FeaturesRawParametersBufferIndex", memoryBindings.registeredMemoryIndex(KernelMemoryBindings::Parameters, radianceParametersKey())),
                    fmt::arg("FeaturesRawParametersGradientBufferIndex", memoryBindings.registeredMemoryIndex(KernelMemoryBindings::ParameterGradients, radianceParametersKey())),
                    fmt::arg("ExtendedFeaturesRawParametersBufferIndex", memoryBindings.registeredMemoryIndex(KernelMemoryBindings::Parameters, extraSignalParametersKey())),
                    fmt::arg("ExtendedFeaturesRawParametersGradientBufferIndex", memoryBindings.registeredMemoryIndex(KernelMemoryBindings::ParameterGradients, extraSignalParametersKey())),
                    fmt::arg("CameraExtendedFeaturesRawParametersBufferIndex", memoryBindings.registeredMemoryIndex(KernelMemoryBindings::Parameters, cameraExtendedFeaturesParametersKey())),
                    fmt::arg("CameraExtendedFeaturesRawParametersGradientBufferIndex", memoryBindings.registeredMemoryIndex(KernelMemoryBindings::ParameterGradients, cameraExtendedFeaturesParametersKey())),
                    fmt::arg("LidarExtendedFeaturesRawParametersBufferIndex", memoryBindings.registeredMemoryIndex(KernelMemoryBindings::Parameters, lidarExtendedFeaturesParametersKey())),
                    fmt::arg("LidarExtendedFeaturesRawParametersGradientBufferIndex", memoryBindings.registeredMemoryIndex(KernelMemoryBindings::ParameterGradients, lidarExtendedFeaturesParametersKey())),
                    fmt::arg("GlobalParametersValueBufferIndex", memoryBindings.registeredValuesMemoryIndex()),
                    fmt::arg("FeatureShDegreeValueOffset", memoryBindings.registeredValueBinding(memoryBindings.registeredValueIndex(radianceActiveShDegreesParameterKey())).offset),
                    fmt::arg("FeaturesDim", m_radianceDim),
                    fmt::arg("FeaturesParametersDim", radianceParametersDim()),
                    fmt::arg("ExtendedFeaturesDim", m_extendedFeaturesEnabled ? m_extendedFeaturesDim : 0),
                    fmt::arg("ExtendedFeaturesParametersDim", m_extendedFeaturesEnabled ? extendedFeaturesParametersDim() : 0),
                    fmt::arg("CameraExtendedFeaturesDim", m_sensorExtendedFeaturesEnabled ? m_cameraExtendedFeaturesDim : 0),
                    fmt::arg("CameraExtendedFeaturesParametersDim", m_sensorExtendedFeaturesEnabled ? cameraExtendedFeaturesParametersDim() : 0),
                    fmt::arg("LidarExtendedFeaturesDim", m_sensorExtendedFeaturesEnabled ? m_lidarExtendedFeaturesDim : 0),
                    fmt::arg("LidarExtendedFeaturesParametersDim", m_sensorExtendedFeaturesEnabled ? lidarExtendedFeaturesParametersDim() : 0),
                    fmt::arg("AlphaThreshold", m_alphaMinValue),
                    fmt::arg("MinTransmittanceThreshold", m_transmittanceThreshold),
                    fmt::arg("TDensityRawParameters", "gaussianParticle_RawParameters_0"),
                    fmt::arg("TDensityParameters", "gaussianParticle_Parameters_0"),
                    fmt::arg("TFeaturesParameters", "shRadiativeParticle_Parameters_0"),
                    fmt::arg("TFeaturesType", m_halfPrecisionFeatures ? "half" : "float"),
                    fmt::arg("TDifferentiable", kernelOpts & KernelResourcesProvider::Differentiable)));

    return Status();
}

nrend::Status nrend::NRESHGaussianModel::registerModelKernelResources_(
    const KernelMemoryBindings& memoryBindings,
    const KernelSourceCodeTable& sourceCodeTable,
    KernelResourcesProvider::KernelOpts kernelOpts,
    const Logger& logger) const {

    const std::string cudaSourceCodeTemplate = R"(
        #include <nrend/kernels/cuda/models/nreShGaussianModel.cuh>

        using {NREShGaussianAlias} = NREShGaussian<{NREShGaussianAlias}Particles>;
    )";
    sourceCodeTable.registerKernel(
        KernelSourceCodeTable::Cuda,
        fmt::format(cudaSourceCodeTemplate, fmt::arg("NREShGaussianAlias", cudaCallPrefix())));

    return Status();
}

nrend::Status nrend::NRESHGaussianModel::processKernelMemory_(
    const KernelMemoryBindings& memoryBindings,
    KernelMemoryBindings::BindingsFlag bindingsFlag,
    const std::vector<std::unique_ptr<KernelMemory>>& memory,
    ProcessMemoryFlag processFlag,
    uint64_t processQueueHandle,
    const Logger& logger) const {
    if (bindingsFlag != KernelMemoryBindings::Parameters) {
        return Status();
    }

    const int densityParametersIndex = memoryBindings.registeredMemoryIndex(bindingsFlag, densityParametersKey());
    RETURN_ERROR_IF_INVALID_INDEX_PTR(densityParametersIndex, memory, logger);
    const int radianceParametersIndex = memoryBindings.registeredMemoryIndex(bindingsFlag, radianceParametersKey());
    RETURN_ERROR_IF_INVALID_INDEX_PTR(radianceParametersIndex, memory, logger);
    const int extraSignalParametersIndex = memoryBindings.registeredMemoryIndex(bindingsFlag, extraSignalParametersKey());
    RETURN_ERROR_IF_INVALID_INDEX_PTR(extraSignalParametersIndex, memory, logger);
    const int cameraExtendedFeaturesParametersIndex = memoryBindings.registeredMemoryIndex(bindingsFlag, cameraExtendedFeaturesParametersKey());
    RETURN_ERROR_IF_INVALID_INDEX_PTR(cameraExtendedFeaturesParametersIndex, memory, logger);
    const int lidarExtendedFeaturesParametersIndex = memoryBindings.registeredMemoryIndex(bindingsFlag, lidarExtendedFeaturesParametersKey());
    RETURN_ERROR_IF_INVALID_INDEX_PTR(lidarExtendedFeaturesParametersIndex, memory, logger);

    CudaBuffer* densityParametersBuffer = memory[densityParametersIndex]->as<CudaBuffer>();
    RETURN_ERROR_IF_INVALID_CAST_PTR(densityParametersBuffer, logger);

    CudaBuffer* radianceParametersBuffer = memory[radianceParametersIndex]->as<CudaBuffer>();
    RETURN_ERROR_IF_INVALID_CAST_PTR(radianceParametersBuffer, logger);

    CudaBuffer* extraSignalParametersBuffer = memory[extraSignalParametersIndex]->as<CudaBuffer>();
    RETURN_ERROR_IF_INVALID_CAST_PTR(extraSignalParametersBuffer, logger);

    CudaBuffer* cameraExtendedFeaturesParametersBuffer = memory[cameraExtendedFeaturesParametersIndex]->as<CudaBuffer>();
    RETURN_ERROR_IF_INVALID_CAST_PTR(cameraExtendedFeaturesParametersBuffer, logger);

    CudaBuffer* lidarExtendedFeaturesParametersBuffer = memory[lidarExtendedFeaturesParametersIndex]->as<CudaBuffer>();
    RETURN_ERROR_IF_INVALID_CAST_PTR(lidarExtendedFeaturesParametersBuffer, logger);

    if (densityParametersBuffer->attached() != radianceParametersBuffer->attached()) {
        RETURN_ERROR(logger, ErrorCode::BadInput, "NRESHGaussianModel : resource %s and %s have a different attachment.",
                     densityParametersKey().c_str(), radianceParametersKey().c_str());
    }
    if (m_extendedFeaturesEnabled) {
        if (densityParametersBuffer->attached() != extraSignalParametersBuffer->attached()) {
            RETURN_ERROR(logger, ErrorCode::BadInput, "NRESHGaussianModel : resource %s and %s have a different attachment.",
                         densityParametersKey().c_str(), extraSignalParametersKey().c_str());
        }
    }
    if (m_sensorExtendedFeaturesEnabled) {
        if (densityParametersBuffer->attached() != cameraExtendedFeaturesParametersBuffer->attached()) {
            RETURN_ERROR(logger, ErrorCode::BadInput, "NRESHGaussianModel : resource %s and %s have a different attachment.",
                         densityParametersKey().c_str(), cameraExtendedFeaturesParametersKey().c_str());
        }
        if (densityParametersBuffer->attached() != lidarExtendedFeaturesParametersBuffer->attached()) {
            RETURN_ERROR(logger, ErrorCode::BadInput, "NRESHGaussianModel : resource %s and %s have a different attachment.",
                         densityParametersKey().c_str(), lidarExtendedFeaturesParametersKey().c_str());
        }
    }

    CHECK_STATUS_RETURN(memoryBindings.getRegisteredValue(particlesNumberParameterKey(), m_particlesNumber, logger));

    if ((processFlag == ProcessMemoryFlag::Initialization) && !densityParametersBuffer->attached()) {
        if (!m_validInitialParameters) {
            RETURN_ERROR(logger, ErrorCode::InvalidResource, "NRESHGaussianModel : cannot initialize resource, invalid initial parameters.");
        }
        CHECK_STATUS_RETURN(densityParametersBuffer->resize(sizeof(float) * densityParametersDim() * m_particlesNumber, processQueueHandle, logger));
        CHECK_STATUS_RETURN(radianceParametersBuffer->resize(radianceParametersTypeSize() * radianceParametersDim() * m_particlesNumber, processQueueHandle, logger));
        // FIXME: extra signal parameters are always in float
        if (m_extendedFeaturesEnabled) {
            CHECK_STATUS_RETURN(extraSignalParametersBuffer->resize(extendedFeaturesParametersTypeSize() * extendedFeaturesParametersDim() * m_particlesNumber, processQueueHandle, logger));
        }
        if (m_sensorExtendedFeaturesEnabled) {
            CHECK_STATUS_RETURN(cameraExtendedFeaturesParametersBuffer->resize(cameraExtendedFeaturesParametersTypeSize() * cameraExtendedFeaturesParametersDim() * m_particlesNumber, processQueueHandle, logger));
            CHECK_STATUS_RETURN(lidarExtendedFeaturesParametersBuffer->resize(lidarExtendedFeaturesParametersTypeSize() * lidarExtendedFeaturesParametersDim() * m_particlesNumber, processQueueHandle, logger));
        }
        CHECK_STATUS_RETURN(packParametersFromHostTensorsWithMortonSort(densityParametersBuffer,
                                                                        radianceParametersBuffer,
                                                                        extraSignalParametersBuffer,
                                                                        cameraExtendedFeaturesParametersBuffer,
                                                                        lidarExtendedFeaturesParametersBuffer,
                                                                        m_halfPrecisionFeatures,
                                                                        processQueueHandle,
                                                                        logger));
    }

    if (densityParametersBuffer->size() != sizeof(float) * densityParametersDim() * m_particlesNumber) {
        RETURN_ERROR(logger, ErrorCode::BadInput, "NRESHGaussianModel : resource %s has a wrong size [%zu /%zu].", densityParametersKey().c_str(),
                     densityParametersBuffer->size(), sizeof(float) * densityParametersDim() * m_particlesNumber);
    }

    // radiance is optional (lidar rendering may ignore radiance)
    if (radianceParametersBuffer->size() != 0 &&
        radianceParametersBuffer->size() != radianceParametersTypeSize() * radianceParametersDim() * m_particlesNumber) {
        RETURN_ERROR(logger, ErrorCode::BadInput, "NRESHGaussianModel : resource %s has a wrong size [%zu /%zu]. %s , %zu, %s", radianceParametersKey().c_str(),
                     radianceParametersBuffer->size(), radianceParametersTypeSize() * radianceParametersDim() * m_particlesNumber,
                     radianceParametersBuffer->attached() ? "attached" : "not attached", densityParametersBuffer->size());
    }

    if (m_extendedFeaturesEnabled) {
        // extra signal features are optional
        if (extraSignalParametersBuffer->size() != 0 &&
            extraSignalParametersBuffer->size() != extendedFeaturesParametersTypeSize() * extendedFeaturesParametersDim() * m_particlesNumber) {
            RETURN_ERROR(logger, ErrorCode::BadInput, "NRESHGaussianModel : resource %s has a wrong size [%zu /%zu].", extraSignalParametersKey().c_str(),
                         extraSignalParametersBuffer->size(), extendedFeaturesParametersTypeSize() * extendedFeaturesParametersDim() * m_particlesNumber);
        }
    }
    if (m_sensorExtendedFeaturesEnabled) {
        // camera extended features are optional
        if (cameraExtendedFeaturesParametersBuffer->size() != 0 &&
            cameraExtendedFeaturesParametersBuffer->size() != cameraExtendedFeaturesParametersTypeSize() * cameraExtendedFeaturesParametersDim() * m_particlesNumber) {
            RETURN_ERROR(logger, ErrorCode::BadInput, "NRESHGaussianModel : resource %s has a wrong size [%zu /%zu].", cameraExtendedFeaturesParametersKey().c_str(),
                         cameraExtendedFeaturesParametersBuffer->size(), cameraExtendedFeaturesParametersTypeSize() * cameraExtendedFeaturesParametersDim() * m_particlesNumber);
        }
        // lidar extended features are optional
        if (lidarExtendedFeaturesParametersBuffer->size() != 0 &&
            lidarExtendedFeaturesParametersBuffer->size() != lidarExtendedFeaturesParametersTypeSize() * lidarExtendedFeaturesParametersDim() * m_particlesNumber) {
            RETURN_ERROR(logger, ErrorCode::BadInput, "NRESHGaussianModel : resource %s has a wrong size [%zu /%zu].", lidarExtendedFeaturesParametersKey().c_str(),
                         lidarExtendedFeaturesParametersBuffer->size(), lidarExtendedFeaturesParametersTypeSize() * lidarExtendedFeaturesParametersDim() * m_particlesNumber);
        }
    }
    return Status();
}
