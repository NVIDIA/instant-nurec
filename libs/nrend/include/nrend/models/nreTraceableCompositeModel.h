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

namespace nrend {

class NRETraceableCompositeModel : public NREBaseCompositeModel {

public:
    static constexpr char name[] = "general-composite";

    NRETraceableCompositeModel(const nlohmann::json& config,
                               const Logger& logger,
                               const nlohmann::json& stateDict,
                               const std::string& prefix)
        // TODO : difference between submodel creation and model creation (may use different prefix / dict)
        : NREBaseCompositeModel(config,
                                logger,
                                stateDict,
                                prefix,
                                {"background", "appearance_embedding", "post_processing"},
                                {"background", "appearance_embedding", "post_processings"},
                                "primitives",
                                "traceable_primitives") {
    }

    virtual ~NRETraceableCompositeModel() = default;

private:
    virtual Status registerKernelResources_(
        const KernelMemoryBindings&,
        const KernelSourceCodeTable& sourceCodeTable,
        KernelResourcesProvider::KernelOpts,
        const Logger&) const override {

        int16_t numActivePrimitivesInstances = 0;
        for (size_t i = 0; i < m_primitives.size(); ++i) {
            numActivePrimitivesInstances += m_primitives[i].numActiveInstances;
        }

        // compute the packedIdx array of active primitive instances
        std::vector<uint16_t> primitiveInstancesPackedIdx(numActivePrimitivesInstances, PrimitiveInstance::InvalidPackedIdx);
        for (size_t i = 0; i < m_primitivesInstancesMap.size(); ++i) {
            for (uint16_t mappingIndex : m_primitivesInstancesMap[i].trackMappingIndex) {
                // Offset the primitive id by the number of background primitives
                primitiveInstancesPackedIdx[mappingIndex] = m_primitivesInstancesMap[i].packedIdx(m_backgroundPrimitives.size());
            }
        }

        // generate the const array of sorted packedIdx array string
        std::string primitivesInstancesMapStr;
        for (size_t i = 0; i < primitiveInstancesPackedIdx.size(); ++i) {
            primitivesInstancesMapStr += std::to_string(primitiveInstancesPackedIdx[i]) + ",";
        }
        if (!primitivesInstancesMapStr.empty()) {
            primitivesInstancesMapStr.pop_back();
        }

        // compute the number of active primitives
        uint8_t numActivePrimitives = m_backgroundPrimitives.size();
        for (const auto& primitive : m_primitives) {
            numActivePrimitives += primitive.numActiveInstances ? 1 : 0;
        }

        const std::string NRETraceableCompositeParamsDefinitionStr = fmt::format(R"(
            struct {NRETraceableCompositeClassAlias}Params {{
               static constexpr uint8_t NumPrimitives = {NumPrimitives};
            
               static constexpr uint8_t NumBackgroundPrimitives = {NumBackgroundPrimitives};
               static constexpr uint16_t NumPrimitiveInstances = {NumPrimitiveInstances};
               static constexpr uint16_t MaxPrimitiveInstancesBatchSize = {MaxPrimitiveInstancesBatchSize};
            
               static const uint16_t primitiveInstancesPackedIdx[{PackedIdxArraySize}];
            
               static constexpr float minTransmittance = {MinTransmittance};
            }};
            const uint16_t {NRETraceableCompositeClassAlias}Params::primitiveInstancesPackedIdx[{PackedIdxArraySize}] = {{{PrimitivesInstancesMap}}};
        )",
                                                                                 fmt::arg("NRETraceableCompositeClassAlias", cudaCallPrefix()),
                                                                                 fmt::arg("NumPrimitives", numActivePrimitives),
                                                                                 fmt::arg("NumBackgroundPrimitives", m_backgroundPrimitives.size()),
                                                                                 fmt::arg("NumPrimitiveInstances", numActivePrimitivesInstances),
                                                                                 fmt::arg("MaxPrimitiveInstancesBatchSize", m_maxPrimitiveInstancesBatchSize),
                                                                                 fmt::arg("PackedIdxArraySize", std::max<uint16_t>(1, numActivePrimitivesInstances)), //< empty array not allowed
                                                                                 fmt::arg("MinTransmittance", m_transmittanceThreshold),
                                                                                 fmt::arg("PrimitivesInstancesMap", primitivesInstancesMapStr));

        const std::string sourceCodeTemplate = R"(
            #include <nrend/kernels/cuda/models/nreTraceableComposite.cuh>

            {NRETraceableCompositeParamsDefinition}

            using {NRETraceableCompositeClassAlias} = NRETraceableComposite<{NRETraceableCompositeClassAlias}Params, 
                                                                            {NREAppearanceEmbeddingClassAlias},
                                                                            {NREBackgroundClassAlias},
                                                                            {NREPostProcessingsClassAlias},
                                                                            {NREVariadicPrimitiveTypes}>;
        )";

        sourceCodeTable.registerKernel(
            KernelSourceCodeTable::Cuda,
            fmt::format(sourceCodeTemplate,
                        fmt::arg("NRETraceableCompositeParamsDefinition", NRETraceableCompositeParamsDefinitionStr),
                        fmt::arg("NRETraceableCompositeClassAlias", cudaCallPrefix()),
                        fmt::arg("NREAppearanceEmbeddingClassAlias", cudaCallPrefix("appearance_embedding")),
                        fmt::arg("NREBackgroundClassAlias", cudaCallPrefix("background")),
                        fmt::arg("NREPostProcessingsClassAlias", cudaCallPrefix("post_processings")),
                        fmt::arg("NREVariadicPrimitiveTypes", getVariadicActivePrimitiveTypesStr("traceable_primitives"))));

        return Status();
    }
};
} // namespace nrend