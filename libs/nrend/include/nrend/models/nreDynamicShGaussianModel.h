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

#include <nrend/models/nreInstancesExtent.h>
#include <nrend/models/nreShGaussianModel.h>

namespace nrend {

class NRERigidSHGaussianModel : public NRESHGaussianModel {

protected:
    struct Settings {
        bool optimizeTrackAlbedo = false;
        bool optimizeTrackScale  = false;
    } m_settings;

    uint32_t m_numInstances = 0;
    mutable std::vector<uint32_t> m_instancesNumParticles;
    mutable std::vector<uint32_t> m_instancesParticlesOffset;

    TStateDictTensor m_particlesInstanceIdxTensor;

public:
    static constexpr char name[] = "rigid-gaussians";

    NRERigidSHGaussianModel(const nlohmann::json& config,
                            const Logger& logger,
                            const nlohmann::json& stateDict,
                            const std::string& prefix,
                            const std::vector<const char*>& submodelCStr = {})
        : NRESHGaussianModel(config, logger, stateDict, prefix, submodelCStr) {

        m_settings.optimizeTrackAlbedo = config.value("optimize_track_albedo", m_settings.optimizeTrackAlbedo);
        m_settings.optimizeTrackScale  = config.value("optimize_track_scale", m_settings.optimizeTrackScale);
        if (m_settings.optimizeTrackAlbedo || m_settings.optimizeTrackScale) {
            LOG_ERROR(logger, "NRERigidShGaussianModel current implementation does not support track-level albedo or scale optimization");
        }

        const std::string extraStateKey    = prefix + "_extra_state";
        const bool stateDictWithExtraState = stateDict.contains(extraStateKey);
        if (!stateDictWithExtraState) {
            LOG_ERROR(logger, "NRERigidSHGaussianModel : missing extra state <%s> in the state_dict.",
                      extraStateKey.c_str());
        }

        if (stateDictWithExtraState) {
            m_numInstances = stateDict[extraStateKey].value("n_tracks", 0);
            m_instancesNumParticles.resize(m_numInstances, 0);
            m_instancesParticlesOffset.resize(m_numInstances, 0);
        }

        m_particlesInstanceIdxTensor.key = prefix + "gaussian_cuboid_ids";
        if (!readStateDictTensor(stateDict, m_particlesInstanceIdxTensor, m_particlesNumber * sizeof(uint32_t))) {
            LOG_WARN(logger,
                     "NRERigidSHGaussianModel : missing particles to instances mapping <%s> [%zu/%u] in the state_dict.",
                     m_particlesInstanceIdxTensor.key.c_str(),
                     m_particlesInstanceIdxTensor.buffer.size() / sizeof(uint32_t),
                     m_particlesNumber);
        }
    }
    virtual ~NRERigidSHGaussianModel() = default;

    virtual uint32_t numInstanceParticles(uint32_t instanceId) const override {
        return instanceId < m_instancesNumParticles.size() ? m_instancesNumParticles[instanceId] : 0;
    }
    virtual uint32_t instanceParticlesOffset(uint32_t instanceId) const override {
        return instanceId < m_instancesParticlesOffset.size() ? m_instancesParticlesOffset[instanceId] : 0;
    }

protected:
    Status packParametersFromHostTensorsWithInstanceIdSort(CudaBuffer* densityParamsPtr,
                                                           CudaBuffer* radianceParamsPtr,
                                                           CudaBuffer* extraSignalParamsPtr,
                                                           CudaBuffer* cameraExtendedFeaturesParamsPtr,
                                                           CudaBuffer* lidarExtendedFeaturesParamsPtr,
                                                           bool halfPrecisionFeatures,
                                                           uint64_t processQueueHandle,
                                                           const Logger& logger) const;

    virtual Status processKernelMemory_(const KernelMemoryBindings& memoryBindings,
                                        KernelMemoryBindings::BindingsFlag bindingsFlag,
                                        const std::vector<std::unique_ptr<KernelMemory>>& memory,
                                        ProcessMemoryFlag processFlag,
                                        uint64_t processQueueHandle,
                                        const Logger& logger) const override;
};

class NREDeformableSHGaussianModel : public NRERigidSHGaussianModel {

    bool m_useDeformNetwork = false;

    struct DeformNetworkSettings {
        bool deformPositions             = true;
        bool deformRotations             = true;
        bool deformRotationsFromIdentity = false;
        bool deformScales                = false;
    } m_deformNetworkSettings;

    NREInstancesExtent m_instancesExtent;

public:
    static constexpr char name[] = "deformable-gaussians";

    NREDeformableSHGaussianModel(const nlohmann::json& config,
                                 const Logger& logger,
                                 const nlohmann::json& stateDict,
                                 const std::string& prefix)
        : NRERigidSHGaussianModel(config, logger, stateDict, prefix, {}) {

        const std::string extraStateKey    = prefix + "_extra_state";
        const bool stateDictWithExtraState = stateDict.contains(extraStateKey);
        if (!stateDictWithExtraState) {
            LOG_ERROR(logger, "NREDeformableSHGaussianModel : missing extra state <%s> in the state_dict.",
                      extraStateKey.c_str());
        }
        if (stateDictWithExtraState) {
            m_useDeformNetwork = stateDict[extraStateKey].value("use_deform_network", false);
        }

        if (!config.contains("deform_network")) {
            LOG_ERROR(logger, "NREDeformableSHGaussianModel : missing deform_network in the config.");
            m_useDeformNetwork = false;
            return;
        }

        // manually add deform network submodel (not part of the configuration filesystem)
        initializeSubModels(
            config["deform_network"],
            stateDict,
            prefix + "deform_network.",
            {"feature_volume"},
            logger);

        // read deform network settings
        m_deformNetworkSettings.deformPositions = config["deform_network"].value(
            "deform_positions", m_deformNetworkSettings.deformPositions);
        m_deformNetworkSettings.deformRotations = config["deform_network"].value(
            "deform_rotations", m_deformNetworkSettings.deformRotations);
        m_deformNetworkSettings.deformRotationsFromIdentity = config["deform_network"].value(
            "rotations_from_identity", m_deformNetworkSettings.deformRotationsFromIdentity);
        m_deformNetworkSettings.deformScales = config["deform_network"].value(
            "deform_scales", m_deformNetworkSettings.deformScales);
        m_useDeformNetwork = m_useDeformNetwork && (m_deformNetworkSettings.deformPositions ||
                                                    m_deformNetworkSettings.deformRotations ||
                                                    m_deformNetworkSettings.deformScales);

        m_instancesExtent = NREInstancesExtent(logger, stateDict, prefix);
        if (m_instancesExtent.numInstances() != m_numInstances) {
            LOG_ERROR(logger, "NREDeformableSHGaussianModel : number of instances mismatch [%u/%u].",
                      m_instancesExtent.numInstances(),
                      m_numInstances);
        }
    }
    virtual ~NREDeformableSHGaussianModel() = default;

private:
    virtual Status registerModelKernelResources_(const KernelMemoryBindings& memoryBindings,
                                                 const KernelSourceCodeTable& sourceCodeTable,
                                                 KernelResourcesProvider::KernelOpts kernelOpts,
                                                 const Logger& logger) const override;
};

} // namespace nrend
