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

class NRESkipGeometry : public NREModel {
    int m_inputFeatureDim;

public:
    static constexpr char name[] = "skip-density";

    NRESkipGeometry(const nlohmann::json& config,
                    const Logger& logger,
                    const nlohmann::json& stateDict,
                    const std::string& prefix)
        : NREModel(config, logger, stateDict, prefix, {}) {
        if (config.contains("input_feature_dim")) {
            m_inputFeatureDim = config["input_feature_dim"];
        }
    }

    virtual ~NRESkipGeometry() = default;

protected:
    virtual Status registerKernelResources_(
        const KernelMemoryBindings&,
        const KernelSourceCodeTable& sourceCodeTable,
        KernelResourcesProvider::KernelOpts,
        const Logger&) const override {

        sourceCodeTable.registerKernel(
            KernelSourceCodeTable::Cuda,
            fmt::format(R"(
                #include <nrend/kernels/cuda/models/nreGeometry.cuh>
                
                using {NREGeometryClassAlias} = NRESkipGeometry<{InputFeatureDim}>;
            )",
                        fmt::arg("NREGeometryClassAlias", cudaCallPrefix()),
                        fmt::arg("InputFeatureDim", m_inputFeatureDim)));
        return Status();
    }
};

} // namespace nrend