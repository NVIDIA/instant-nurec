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
#include <nrend/models/nreParticlesModel.h>
#include <nrend/utils/cuda/cudaBuffer.h>

namespace nrend {

class NRESHGaussianModel : public INREParticlesModel, public NREModel {

public:
    enum BakeActivation {
        BakeActivationNone     = 0,
        BakeActivationDensity  = 1 << 0,
        BakeActivationRotation = 1 << 1,
        BakeActivationScale    = 1 << 2,
        BakeActivationAll      = BakeActivationDensity | BakeActivationRotation | BakeActivationScale
    };

protected:
    enum Parameters {
        Positions,
        Scales,
        Rotations,
        Densities,
        SphAlbedos,
        SphSpeculars,
        ExtraSignals,
        CameraExtraSignals,
        LidarExtraSignals,
        NumParameters
    };
    static constexpr char const* s_parametersKey[NumParameters] = {
        "positions",
        "scales",
        "rotations",
        "densities",
        "features_albedo",
        "features_specular",
        "extra_signal",
        "camera_extra_signal",
        "lidar_extra_signal"};

    std::array<TStateDictTensor, NumParameters> m_paramsTensor;

    bool m_validInitialParameters         = true;
    int m_densityKernelDegree             = 2;
    bool m_densityKernelPlanar            = false;
    float m_densityKernelScale            = 3.0f;
    float m_densityKernelMinResponse      = 0.0113f;
    float m_alphaMinValue                 = 1.0f / 255.0f;
    float m_alphaMaxValue                 = 0.99f;
    bool m_raySpreadFilterEnabled         = false;
    int m_radianceSphDegree               = 3;
    bool m_radianceSphO0                  = false; ///< SH degree 0 coefficients directly encode radiance (no SH basis scaling)
    int m_radianceDim                     = 3;
    int m_radianceFourierDim              = 1;
    int m_radianceMaxNumCoefficients      = 16;
    float m_transmittanceThreshold        = 0.0001f;
    int m_extendedFeaturesDim             = 0;
    int m_extendedFeaturesSphDegree       = 0;
    int m_cameraExtendedFeaturesDim       = 0;
    int m_cameraExtendedFeaturesSphDegree = 0;
    int m_lidarExtendedFeaturesDim        = 0;
    int m_lidarExtendedFeaturesSphDegree  = 0;

    mutable bool m_halfPrecisionFeatures         = true;
    mutable bool m_extendedFeaturesEnabled       = false;
    mutable bool m_sensorExtendedFeaturesEnabled = false;
    uint8_t m_bakeActivation                     = BakeActivationAll;
    bool m_morton3DParticleSort                  = false;
    mutable uint32_t m_particlesNumber           = 0;

public:
    static constexpr char name[] = "sh-gaussians";

    NRESHGaussianModel(const nlohmann::json& config,
                       const Logger& logger,
                       const nlohmann::json& stateDict,
                       const std::string& prefix,
                       const std::vector<const char*>& submodelCStr = {});
    virtual ~NRESHGaussianModel() = default;

    virtual FeaturesLayout featuresLayout() const override {
        return FeaturesLayout{m_radianceDim,
                              m_extendedFeaturesDim,
                              m_cameraExtendedFeaturesDim,
                              m_lidarExtendedFeaturesDim};
    }

    virtual bool isDynamic() const override { return false; }
    virtual uint32_t numParticles() const override { return m_particlesNumber; }
    // static model does not have instances
    virtual uint32_t numInstanceParticles(uint32_t /*instanceId*/) const { return 0; }
    virtual uint32_t instanceParticlesOffset(uint32_t /*instanceId*/) const { return 0; }

    virtual Status prepareParticlesParameters(uint32_t /*numActiveTrackInstances*/,
                                              const tcnn::ivec2* /*activeTrackInstancesIdsCudaPtr*/,
                                              const KernelMemoryPtrVec& /*parameterMemoryPtrVec*/,
                                              std::vector<KernelBindedTransientMemory>& /*transientParameters*/,
                                              uint32_t& numParticles,
                                              uint32_t& numParticlesToPreProcess,
                                              cudaStream_t /*cudaStream*/,
                                              const Logger& /*logger*/) const override {
        numParticles             = m_particlesNumber;
        numParticlesToPreProcess = 0;
        return Status();
    }

    inline std::string densityParametersKey() const { return m_callPrefix + "particle_density"; }
    static constexpr inline int densityParametersDim() { return 12; } ///< position[3], opacity[1], quaternion[4], scales[3], padding[1]

    inline std::string particlesNumberParameterKey() const { return m_callPrefix + "particles_number"; }
    inline std::string radianceActiveShDegreesParameterKey() const { return m_callPrefix + "active_sh_degree"; }
    inline std::string radianceParametersKey() const { return m_callPrefix + "particle_radiance"; }
    inline std::string extraSignalParametersKey() const { return m_callPrefix + "particle_extra_signal"; }
    inline std::string cameraExtendedFeaturesParametersKey() const { return m_callPrefix + "particle_camera_extra_signal"; }
    inline std::string lidarExtendedFeaturesParametersKey() const { return m_callPrefix + "particle_lidar_extra_signal"; }
    inline int radianceParametersDim() const { return m_radianceMaxNumCoefficients * m_radianceDim; }
    inline size_t radianceParametersTypeSize() const { return m_halfPrecisionFeatures ? sizeof(__half) : sizeof(float); }
    inline int extendedFeaturesParametersDim() const { return m_extendedFeaturesDim * (m_extendedFeaturesSphDegree + 1) * (m_extendedFeaturesSphDegree + 1); }
    inline size_t extendedFeaturesParametersTypeSize() const { return m_halfPrecisionFeatures ? sizeof(__half) : sizeof(float); }
    inline int cameraExtendedFeaturesParametersDim() const { return m_cameraExtendedFeaturesDim * (m_cameraExtendedFeaturesSphDegree + 1) * (m_cameraExtendedFeaturesSphDegree + 1); }
    inline size_t cameraExtendedFeaturesParametersTypeSize() const { return m_halfPrecisionFeatures ? sizeof(__half) : sizeof(float); }
    inline int lidarExtendedFeaturesParametersDim() const { return m_lidarExtendedFeaturesDim * (m_lidarExtendedFeaturesSphDegree + 1) * (m_lidarExtendedFeaturesSphDegree + 1); }
    inline size_t lidarExtendedFeaturesParametersTypeSize() const { return m_halfPrecisionFeatures ? sizeof(__half) : sizeof(float); }

protected:
    Status packParametersFromHostTensors(CudaBuffer& densityParamsBuffer,
                                         CudaBuffer& radianceParamsBuffer,
                                         CudaBuffer& extraSignalParamsBuffer,
                                         CudaBuffer& cameraExtendedFeaturesParamsBuffer,
                                         CudaBuffer& lidarExtendedFeaturesParamsBuffer,
                                         const tcnn::tvec<__half, 3>* positionsPtr,
                                         const uint32_t* resortingIndicesPtr,
                                         bool halfPrecisionFeatures,
                                         uint64_t processQueueHandle,
                                         const Logger& logger) const;

    Status packParametersFromHostTensorsWithMortonSort(CudaBuffer* densityParamsPtr,
                                                       CudaBuffer* radianceParamsPtr,
                                                       CudaBuffer* extraSignalParamsPtr,
                                                       CudaBuffer* cameraExtendedFeaturesParamsPtr,
                                                       CudaBuffer* lidarExtendedFeaturesParamsPtr,
                                                       bool halfPrecisionFeatures,
                                                       uint64_t processQueueHandle,
                                                       const Logger& logger) const;

private:
    virtual Status registerParticleKernelResources_(const KernelMemoryBindings& memoryBindings,
                                                    const KernelSourceCodeTable& sourceCodeTable,
                                                    KernelResourcesProvider::KernelOpts kernelOpts,
                                                    const Logger& logger) const;

    virtual Status registerModelKernelResources_(const KernelMemoryBindings& memoryBindings,
                                                 const KernelSourceCodeTable& sourceCodeTable,
                                                 KernelResourcesProvider::KernelOpts kernelOpts,
                                                 const Logger& logger) const;

    virtual Status registerKernelResources_(const KernelMemoryBindings& memoryBindings,
                                            const KernelSourceCodeTable& sourceCodeTable,
                                            KernelResourcesProvider::KernelOpts kernelOpts,
                                            const Logger& logger) const override {
        CHECK_STATUS_RETURN(registerParticleKernelResources_(memoryBindings,
                                                             sourceCodeTable,
                                                             kernelOpts,
                                                             logger));
        CHECK_STATUS_RETURN(registerModelKernelResources_(memoryBindings,
                                                          sourceCodeTable,
                                                          kernelOpts,
                                                          logger));
        return Status();
    }

protected:
    virtual Status processKernelMemory_(const KernelMemoryBindings& memoryBindings,
                                        KernelMemoryBindings::BindingsFlag bindingsFlag,
                                        const std::vector<std::unique_ptr<KernelMemory>>& memory,
                                        ProcessMemoryFlag processFlag,
                                        uint64_t processQueueHandle,
                                        const Logger& logger) const override;
};

} // namespace nrend