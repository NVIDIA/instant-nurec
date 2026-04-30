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

#include <nrend/models/nreInstancesExtent.h>
#include <nrend/models/nreModel.h>

#include <tiny-cuda-nn/bounding_box.h>

namespace nrend {

class NREBaseAABBAccStructure : public NREModel {

protected:
    bool m_stratified                  = false;
    std::string m_contractionDegreeStr = "null";

    inline std::string contractionTypeStr() const {
        if (m_contractionDegreeStr == "null") {
            return "NoContraction";
        } else if (m_contractionDegreeStr == "inf") {
            return "DegreeInf";
        } else if (m_contractionDegreeStr == "2" || m_contractionDegreeStr == "2.0") {
            return "Degree2";
        } else if (m_contractionDegreeStr == "merf") {
            return "Merf";
        }
        return "NoContraction";
    }

public:
    NREBaseAABBAccStructure(const nlohmann::json& config,
                            const Logger& logger,
                            const nlohmann::json& stateDict,
                            const std::string& prefix)
        : NREModel(config, logger, stateDict, prefix, {}) {

        // read configuration parameters

        // ignore statified sampling which is used only during training
        // m_stratified = config.value("stratified", true);

        // contraction_degree may be "null" (None)
        try {
            m_contractionDegreeStr = config.value("contraction_degree", "null");
        } catch (const std::exception& e) {
            m_contractionDegreeStr = "null";
        }
    }

    virtual ~NREBaseAABBAccStructure() = default;
};

class NREDenseObjectAccStructure : public NREBaseAABBAccStructure {

    float m_uniformStepSize;
    float m_cuboidTracksExpand;
    bool m_singleJitter;

    NREInstancesExtent m_instancesExtent;

public:
    static constexpr char name[] = "dense-object-acc-structure";

    NREDenseObjectAccStructure(const nlohmann::json& config,
                               const Logger& logger,
                               const nlohmann::json& stateDict,
                               const std::string& prefix)
        : NREBaseAABBAccStructure(config, logger, stateDict, prefix) {

        // read configuration parameters
        m_uniformStepSize = config.value("uniform_step_size", 0.05f);
        // currently not used since we read the already expanded extent from the statedict
        m_cuboidTracksExpand = config.value("cuboid_tracks_expand", 0.05f);
        m_singleJitter       = config.value("single_jitter", false);

        // extent of every instances
        m_instancesExtent = NREInstancesExtent(logger, stateDict, prefix);
    }

    virtual ~NREDenseObjectAccStructure() = default;

private:
    inline float globalAlphaInvarianceOffset() const {
        return -std::log(m_instancesExtent.maxExtent());
    }

    virtual Status registerKernelResources_(
        const KernelMemoryBindings& memoryBindings,
        const KernelSourceCodeTable& sourceCodeTable,
        KernelResourcesProvider::KernelOpts,
        const Logger&) const override {

        const std::string sourceCodeTemplate = R"(
            #include <nrend/kernels/cuda/models/nreAccelerationStructure.cuh>

            {NREInstancesExtentClassDefinition}

            struct {NREAccStructureClassAlias}Params
            {{
                static constexpr bool stratifiedSampling = {stratifiedSampling};
                static constexpr bool singleJitter       = {singleJitter};
                static constexpr float uniformStepSize   = {uniformStepSize};
                static constexpr float alphaInvarianceOffset = {alphaInvarianceOffset};
            }};
            using {NREAccStructureClassAlias} = NREDenseObjectAccStructure<{NREInstancesExtentClassAlias}, {NREAccStructureClassAlias}Params>;
        )";

        sourceCodeTable.registerKernel(
            KernelSourceCodeTable::Cuda,
            fmt::format(sourceCodeTemplate,
                        fmt::arg("NREAccStructureClassAlias", cudaCallPrefix()),
                        fmt::arg("NREInstancesExtentClassDefinition", m_instancesExtent.sourceDefinition(cudaCallPrefix() + "_instances__extent")),
                        fmt::arg("NREInstancesExtentClassAlias", cudaCallPrefix() + "_instances__extent"),
                        fmt::arg("stratifiedSampling", m_stratified),
                        fmt::arg("singleJitter", m_singleJitter),
                        fmt::arg("uniformStepSize", m_uniformStepSize),
                        fmt::arg("alphaInvarianceOffset", globalAlphaInvarianceOffset())));

        return Status();
    }
};

class NRENeRFAccelerationStructure : public NREBaseAABBAccStructure {
    float m_occThreshold;
    int m_gridResolution;
    float m_expStepFactor;
    float m_minStepSize;
    int m_cascades;

    tcnn::BoundingBox m_sceneContractorAABB;

    TStateDictTensor m_occupancyBitfieldTensor;

public:
    static constexpr char name[] = "nerfacc-acc-structure";

    NRENeRFAccelerationStructure(const nlohmann::json& config,
                                 const Logger& logger,
                                 const nlohmann::json& stateDict,
                                 const std::string& prefix)
        : NREBaseAABBAccStructure(config, logger, stateDict, prefix) {

        // read configuration parameters
        m_occThreshold   = config.value("occ_threshold", 0.01f);
        m_gridResolution = config.value("grid_resolution", 128);
        m_expStepFactor  = config.value("exp_step_factor", 0.00390625f);
        m_minStepSize    = config.value("min_step_size", 0.05f);
        m_cascades       = config.value("cascades", 5);

        const std::string aabbblbKey = prefix + "_scene_contractor.aabb.blb";
        const std::string aabbtrfKey = prefix + "_scene_contractor.aabb.trf";
        if (stateDict.contains(aabbblbKey) && stateDict.contains(aabbtrfKey)) {
            const nlohmann::json::binary_t& aabbblb = stateDict[aabbblbKey];
            const nlohmann::json::binary_t& aabbtrf = stateDict[aabbtrfKey];

            assert((aabbblb.size() == sizeof(tcnn::tvec<__half, 3>)) && (aabbblb.size() == aabbtrf.size()));

            m_sceneContractorAABB.min = tcnn::tvec<__half, 3>(reinterpret_cast<const __half*>(aabbblb.data()));
            m_sceneContractorAABB.max = tcnn::tvec<__half, 3>(reinterpret_cast<const __half*>(aabbtrf.data()));

            if (m_sceneContractorAABB.is_empty()) {
                LOG_WARN(logger, "NRENeRFAccelerationStructure : AABB is empty.");
            }
        } else {
            LOG_ERROR(logger, "NRENeRFAccelerationStructure : missing aabb information (<%s> , <%s>) in the state_dict.",
                      aabbblbKey.c_str(), aabbtrfKey.c_str());
        }

        // read data parameters
        m_occupancyBitfieldTensor.key = prefix + "occupancy_bitfield";
        if (!readStateDictTensor(stateDict, m_occupancyBitfieldTensor)) {
            LOG_INFO(
                logger, "NRENeRFAccelerationStructure : missing tensor <%s> in the state_dict.", m_occupancyBitfieldTensor.key.c_str());
        }
    }

    virtual ~NRENeRFAccelerationStructure() = default;

private:
    virtual Status registerKernelResources_(
        const KernelMemoryBindings& memoryBindings,
        const KernelSourceCodeTable& sourceCodeTable,
        KernelResourcesProvider::KernelOpts,
        const Logger& logger) const override {

        Status status = memoryBindings.registerMemory(KernelMemoryBindings::BindingsFlag::Parameters,
                                                      m_occupancyBitfieldTensor.key,
                                                      KernelMemoryType::Buffer,
                                                      logger);
        if (!status) {
            return status;
        }

        const std::string sourceCodeTemplate = R"(
            #include <nrend/kernels/cuda/models/nreAccelerationStructure.cuh>
            
            struct {NREAccStructureClassAlias}Params
            {{
               static constexpr NREContractionType contractionType = NREContractionType::{contractionType};
               static constexpr float alphaInvarianceOffset = {alphaInvarianceOffset};
               static constexpr float aabbScale = {aabbScale};
               const bool stratifiedSampling = {stratifiedSampling};
               static constexpr int occupancyBitfieldBufferIdx = {occupancyBitfieldBufferIdx};
               static constexpr uint32_t gridResolution = {gridResolution};
               static constexpr uint32_t gridNumCells = gridResolution * gridResolution * gridResolution;
               static constexpr uint32_t minCascade = 0;
               static constexpr uint32_t maxCascade = {maxCascade};
               static constexpr float coneAngle = {coneAngle};
               static constexpr float dtMin = {dtMin};
               static constexpr float dtMax = {dtMax};
            
               static const float aabb[6];
            }};
            const float {NREAccStructureClassAlias}Params::aabb[6] = {{ {aabbMinX}, {aabbMinY}, {aabbMinZ}, {aabbMaxX}, {aabbMaxY}, {aabbMaxZ} }};
            using {NREAccStructureClassAlias} = NREAccNeRFAccAccStructure<{NREAccStructureClassAlias}Params>;
        )";

        // alphaInvarianceOffset : used to normalized the density scale wrt the aabb scale
        const float alphaInvarianceOffset = m_sceneContractorAABB.is_empty() ? 1.0f : -std::log(tcnn::max(m_sceneContractorAABB.diag()));
        // aabbScale : used to normalize the aabb to [-2^(maxCascade-1), 2^(maxCascade -1)]
        const float aabbScale = scalbnf(tcnn::max(m_sceneContractorAABB.diag()), -(m_cascades - 1));
        // parameters binding index
        const int occupancyBitfieldBindingIndex = memoryBindings.registeredMemoryIndex(KernelMemoryBindings::BindingsFlag::Parameters, m_occupancyBitfieldTensor.key);

        sourceCodeTable.registerKernel(
            KernelSourceCodeTable::Cuda,
            fmt::format(sourceCodeTemplate,
                        fmt::arg("NREAccStructureClassAlias", cudaCallPrefix()), fmt::arg("aabbMinX", m_sceneContractorAABB.min.x),
                        fmt::arg("aabbMinY", m_sceneContractorAABB.min.y), fmt::arg("aabbMinZ", m_sceneContractorAABB.min.z),
                        fmt::arg("aabbMaxX", m_sceneContractorAABB.max.x), fmt::arg("aabbMaxY", m_sceneContractorAABB.max.y),
                        fmt::arg("aabbMaxZ", m_sceneContractorAABB.max.z),
                        fmt::arg("contractionType", contractionTypeStr()),
                        fmt::arg("alphaInvarianceOffset", alphaInvarianceOffset),
                        fmt::arg("aabbScale", aabbScale),
                        fmt::arg("stratifiedSampling", m_stratified),
                        fmt::arg("occupancyBitfieldBufferIdx", occupancyBitfieldBindingIndex),
                        fmt::arg("gridResolution", m_gridResolution),
                        fmt::arg("maxCascade", m_cascades - 1),
                        fmt::arg("coneAngle", m_expStepFactor),
                        fmt::arg("dtMin", m_minStepSize),
                        fmt::arg("dtMax", 1e10f)));

        return status;
    }

    virtual Status processKernelMemory_(
        const KernelMemoryBindings& memoryBindings,
        KernelMemoryBindings::BindingsFlag bindingsFlag,
        const std::vector<std::unique_ptr<KernelMemory>>& memory,
        ProcessMemoryFlag processFlag,
        uint64_t processQueueHandle,
        const Logger& logger) const override {

        if ((processFlag != ProcessMemoryFlag::Initialization) || (bindingsFlag != KernelMemoryBindings::Parameters)) {
            return Status();
        }

        const int occupancyBitfieldIndex = memoryBindings.registeredMemoryIndex(bindingsFlag, m_occupancyBitfieldTensor.key);
        if ((occupancyBitfieldIndex < 0) || (occupancyBitfieldIndex >= memory.size()) || !memory[occupancyBitfieldIndex]) {
            RETURN_ERROR(logger, ErrorCode::BadInput, "NRENeRFAccelerationStructure : %s parameters not correctly binded.", m_occupancyBitfieldTensor.key.c_str());
        }

        return memory[occupancyBitfieldIndex]->setFromHost(m_occupancyBitfieldTensor.buffer.data(), m_occupancyBitfieldTensor.buffer.size(), processQueueHandle, logger);
    }
};

} // namespace nrend