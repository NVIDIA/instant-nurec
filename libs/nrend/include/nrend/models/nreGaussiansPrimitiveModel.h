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

#include <nrend/models/nreShGaussianModel.h>

// TODO : deprecate this model (only used for compatibility with regression tests)

namespace nrend {

class NREGaussiansPrimitiveModel : public NRESHGaussianModel {

    // return the name of the layers node used as a model for the primitives
    static const char* baseLayerNodeModelName(const nlohmann::json& config) {
        // backward comp : support legacy "gaussians"
        return config["layers"].contains("gaussians") ? "gaussians" : "background";
    }

    bool m_saturateRadiance = true;

public:
    static constexpr char name[] = "gaussians_primitive";

    NREGaussiansPrimitiveModel(const nlohmann::json& config,
                               const Logger& logger,
                               const nlohmann::json& stateDict,
                               const std::string& prefix)
        : NRESHGaussianModel(config["layers"][baseLayerNodeModelName(config)],
                             logger, stateDict,
                             prefix + "gaussians_nodes." + baseLayerNodeModelName(config) + ".") {
        initializeSubModels(config,
                            stateDict,
                            prefix,
                            {"appearance_embedding", "background", "post_processing"},
                            {"appearance_embedding", "background", "post_processings"},
                            logger);
        m_saturateRadiance = config.value("saturate_radiance", m_saturateRadiance);
    }

    virtual ~NREGaussiansPrimitiveModel() = default;

private:
    virtual Status registerModelKernelResources_(const KernelMemoryBindings&,
                                                 const KernelSourceCodeTable& sourceCodeTable,
                                                 KernelResourcesProvider::KernelOpts kernelOpts,
                                                 const Logger& logger) const override {
        if (kernelOpts & KernelResourcesProvider::Differentiable) {
            RETURN_ERROR(logger, ErrorCode::NotImplemented, "NREGaussiansPrimitiveModel : not differentiable.");
        }

        const std::string cudaSourceCodeTemplate = R"(
            #include <nrend/kernels/cuda/models/nreGaussiansPrimitiveModel.cuh>

            using {NREGaussiansPrimitiveAlias} = NREGaussiansPrimitive<{NREGaussianNodeAlias}Particles,
                                                                       {NREAppearanceEmbeddingClassAlias}, 
                                                                       {NREBackgroundClassAlias},
                                                                       {EnableBackground},
                                                                       {NREPostProcessingsClassAlias},
                                                                       {EnablePostProcessings},
                                                                       {SaturateRadiance}>;
        )";
        sourceCodeTable.registerKernel(
            KernelSourceCodeTable::Cuda,
            fmt::format(cudaSourceCodeTemplate,
                        fmt::arg("NREGaussiansPrimitiveAlias", cudaCallPrefix()),
                        fmt::arg("NREGaussianNodeAlias", cudaCallPrefix()),
                        // FIXME : hack to workaround the inherited prefix
                        fmt::arg("NREAppearanceEmbeddingClassAlias", "model_appearance__embedding_"),
                        fmt::arg("NREBackgroundClassAlias", "model_background_"),
                        fmt::arg("NREPostProcessingsClassAlias", "model_post__processings_"),
                        fmt::arg("EnableBackground", !(kernelOpts & KernelResourcesProvider::DisableBackground)),
                        fmt::arg("EnablePostProcessings", !(kernelOpts & KernelResourcesProvider::DisablePostProcessings)),
                        fmt::arg("SaturateRadiance", m_saturateRadiance)));

        return Status();
    }
};

} // namespace nrend