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

#include <nrend/models/nreBaseCompositeModel.h>
#include <nrend/models/nreShGaussianModel.h>

namespace nrend {

class NREGaussiansCompositeModel : public INREParticlesModel, public NREBaseCompositeModel {

    enum Parameters {
        RenderingCumulativeNumInstances, //< modulo warp size
        RenderingActiveInstances,
        ParticleDensity,
        ParticleFeatures,
        ParticleExtendedFeatures,
        ParticleCameraExtendedFeatures,
        ParticleLidarExtendedFeatures,
        NumParameters
    };
    mutable std::array<int32_t, NumParameters> m_parametersBindingIndex{-1};

    mutable uint32_t m_numStaticParticles = 0;

    struct ActiveInstance {
        static constexpr uint32_t InvalidPrimitiveId         = std::numeric_limits<uint32_t>::max();
        static constexpr uint32_t InvalidPrimitiveInstanceId = std::numeric_limits<uint32_t>::max();

        uint32_t primitiveId         = InvalidPrimitiveId;
        uint32_t primitiveInstanceId = InvalidPrimitiveInstanceId;
        uint32_t numParticles        = 0;
        uint32_t particlesOffset     = 0;
    };
    mutable std::vector<ActiveInstance> m_activeInstances;

    struct RenderingActiveInstance {
        uint32_t numParticles        = 0;
        uint32_t cumNumParticles     = 0;
        uint32_t primitiveId         = ActiveInstance::InvalidPrimitiveId;
        uint32_t primitiveInstanceId = ActiveInstance::InvalidPrimitiveInstanceId;
        uint32_t particlesOffset     = 0;
    };

    std::unique_ptr<NRESHGaussianModel> m_compositeModelPtr;
    mutable bool m_extendedFeaturesEnabled       = false;
    mutable bool m_sensorExtendedFeaturesEnabled = false;
    bool m_saturateRadiance                      = true;

public:
    static constexpr char name[] = "gaussians-composite";

    NREGaussiansCompositeModel(const nlohmann::json& config,
                               const Logger& logger,
                               const nlohmann::json& stateDict,
                               const std::string& prefix)
        : NREBaseCompositeModel(config,
                                logger,
                                stateDict,
                                prefix,
                                {"appearance_embedding", "background", "post_processing"},
                                {"appearance_embedding", "background", "post_processings"},
                                "layers", "gaussians_nodes") {

        m_saturateRadiance = config.value("saturate_radiance", m_saturateRadiance);

        // use background mode configuration as composite model
        // TODO : rework this to either support no background model or to directly use the background model as composite model (avoiding duplication)
        if (config.contains("layers") && config["layers"].contains("background")) {
            NREModel* nreModelPtr = createFromJSON(config["layers"]["background"], logger, stateDict, prefix + "particle_model.");
            m_compositeModelPtr.reset(dynamic_cast<NRESHGaussianModel*>(nreModelPtr));
            if (!m_compositeModelPtr) {
                delete nreModelPtr;
            }
        }
        if (!m_compositeModelPtr) {
            LOG_ERROR(logger, "NREGaussiansCompositeModel : no valid composite model found");
        }
    }
    virtual ~NREGaussiansCompositeModel() = default;

    virtual FeaturesLayout featuresLayout() const override {
        return m_compositeModelPtr ? m_compositeModelPtr->featuresLayout() : FeaturesLayout{0, 0, 0, 0};
    }

    virtual bool isDynamic() const override { return !m_primitives.empty(); }
    // FIXME: gaussian composite models may have more particles than the static particles
    virtual uint32_t numParticles() const override { return m_numStaticParticles; }

    virtual Status prepareParticlesParameters(uint32_t numActiveTrackInstances,
                                              const tcnn::ivec2* activeTrackInstancesIdsCudaPtr,
                                              const KernelMemoryPtrVec& parameterMemoryPtrVec,
                                              std::vector<KernelBindedTransientMemory>& transientParameters,
                                              uint32_t& numParticles,
                                              uint32_t& numParticlesToPreProcess,
                                              cudaStream_t cudaStream,
                                              const Logger& logger) const override;

    inline std::string renderingCumulativeNumInstancesParameterKey() const { return m_callPrefix + "rendering_cumulative_num_instances"; }
    inline std::string renderingActiveInstanceParametersKey() const { return m_callPrefix + "rendering_active_instances"; }

private:
    virtual Status registerKernelResources_(const KernelMemoryBindings& memoryBindings,
                                            const KernelSourceCodeTable& sourceCodeTable,
                                            KernelResourcesProvider::KernelOpts kernelOpts,
                                            const Logger& logger) const override;

    virtual Status processKernelMemory_(const KernelMemoryBindings&,
                                        KernelMemoryBindings::BindingsFlag,
                                        const std::vector<std::unique_ptr<KernelMemory>>&,
                                        ProcessMemoryFlag,
                                        uint64_t /*processQueueHandle*/,
                                        const Logger&) const override;
};

} // namespace nrend