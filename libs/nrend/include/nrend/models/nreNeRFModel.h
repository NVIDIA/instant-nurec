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

#include <nrend/models/nreModel.h>

namespace nrend {

class NRENeRFModel : public NREModel {

    float m_transmittanceThreshold = 0.0001f;

public:
    static constexpr char name[] = "nerf";

    NRENeRFModel(const nlohmann::json& config,
                 const Logger& logger,
                 const nlohmann::json& stateDict,
                 const std::string& prefix)
        : NREModel(config,
                   logger,
                   stateDict,
                   prefix,
                   {"acc_structure", "feature_volume", "geometry", "texture", "appearance_embedding", "background"}) {

        // discrepancy between module and config name
        initializeSubModels(config, stateDict, prefix, {"post_processing"}, {"post_processings"}, logger);

        m_transmittanceThreshold = config.value("transmittance_threshold", 0.0001f);
    }

    virtual ~NRENeRFModel() = default;

protected:
    virtual Status registerKernelResources_(
        const KernelMemoryBindings&,
        const KernelSourceCodeTable& sourceCodeTable,
        KernelResourcesProvider::KernelOpts,
        const Logger&) const override {

        const std::string sourceCodeTemplate = R"(
            #include <nrend/kernels/cuda/models/nreNeRF.cuh>

            struct {NRENeRFClassAlias}Params {{
               static constexpr float minTransmittance = {MinTransmittance};
            }};
            using {NRENeRFClassAlias} = NRENeRF<{NREAccelerationStructureClassAlias}, 
                                                {NREFeatureVolumeClassAlias}, 
                                                {NREGeometryClassAlias},
                                                {NRETextureClassAlias}, 
                                                {NREAppearanceEmbeddingClassAlias}, 
                                                {NREBackgroundClassAlias}, 
                                                {NREPostProcessingsClassAlias}, 
                                                {NRENeRFClassAlias}Params>;
        )";

        sourceCodeTable.registerKernel(
            KernelSourceCodeTable::Cuda,
            fmt::format(sourceCodeTemplate,
                        fmt::arg("NRENeRFClassAlias", cudaCallPrefix()),
                        fmt::arg("MinTransmittance", m_transmittanceThreshold),
                        fmt::arg("NREAccelerationStructureClassAlias", cudaCallPrefix("acc_structure")),
                        fmt::arg("NREFeatureVolumeClassAlias", cudaCallPrefix("feature_volume")),
                        fmt::arg("NREGeometryClassAlias", cudaCallPrefix("geometry")),
                        fmt::arg("NRETextureClassAlias", cudaCallPrefix("texture")),
                        fmt::arg("NREAppearanceEmbeddingClassAlias", cudaCallPrefix("appearance_embedding")),
                        fmt::arg("NREBackgroundClassAlias", cudaCallPrefix("background")),
                        fmt::arg("NREPostProcessingsClassAlias", cudaCallPrefix("post_processings"))));

        return Status();
    }
};
} // namespace nrend