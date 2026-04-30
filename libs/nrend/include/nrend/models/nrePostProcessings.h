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
namespace nrend {

class NREPostProcessings : public NREModel {
public:
    static constexpr char name[] = "post_processing";

    std::vector<std::string> m_operatorsName;
    std::vector<std::string> m_operatorsPrefix;

    NREPostProcessings(const nlohmann::json& config,
                       const Logger& logger,
                       const nlohmann::json& stateDict,
                       const std::string& prefix)
        : NREModel(config, logger, stateDict, prefix, {}) {

        // collect the operators keys from the config
        for (const auto& [key, _] : config.items()) {
            m_operatorsName.push_back(key);
        }
        // sort the operators keys lexicographically
        std::sort(m_operatorsName.begin(), m_operatorsName.end());
        // create the operators prefixes
        m_operatorsPrefix.resize(m_operatorsName.size());
        for (size_t i = 0; i < m_operatorsName.size(); ++i) {
            m_operatorsPrefix[i] = std::to_string(i);
        }

        // initialize the operators
        initializeSubModels(config, stateDict, prefix, m_operatorsName, m_operatorsPrefix, logger);
    }

    virtual ~NREPostProcessings() = default;

protected:
    virtual Status registerKernelResources_(
        const KernelMemoryBindings&,
        const KernelSourceCodeTable& sourceCodeTable,
        KernelResourcesProvider::KernelOpts,
        const Logger&) const override {

        // create the variadic template signature for the operators
        std::string operatorsTemplate;
        for (const std::string& operatorPrefix : m_operatorsPrefix) {
            operatorsTemplate += cudaCallPrefix(operatorPrefix.c_str()) + ",";
        }
        if (!operatorsTemplate.empty()) {
            operatorsTemplate.pop_back();
        }

        sourceCodeTable.registerKernel(
            KernelSourceCodeTable::Cuda,
            fmt::format(R"(
                #include <nrend/kernels/cuda/models/nrePostProcessings.cuh>
                
                using {NREPostProcessingsClassAlias} = NREPostProcessings<{OperatorsTemplate}>;
            )",
                        fmt::arg("NREPostProcessingsClassAlias", cudaCallPrefix()),
                        fmt::arg("OperatorsTemplate", operatorsTemplate)));
        return Status();
    }
};

class NRESkipPostProcessing : public NREModel {
public:
    static constexpr char name[] = "skip-post-processing";

    NRESkipPostProcessing(const nlohmann::json& config,
                          const Logger& logger,
                          const nlohmann::json& stateDict,
                          const std::string& prefix)
        : NREModel(config, logger, stateDict, prefix, {}) {
    }

    virtual ~NRESkipPostProcessing() = default;

protected:
    virtual Status registerKernelResources_(
        const KernelMemoryBindings&,
        const KernelSourceCodeTable& sourceCodeTable,
        KernelResourcesProvider::KernelOpts,
        const Logger&) const override {

        sourceCodeTable.registerKernel(
            KernelSourceCodeTable::Cuda,
            fmt::format(R"(
                #include <nrend/kernels/cuda/models/nrePostProcessings.cuh>
                
                using {NREPostProcessingsClassAlias} = NRESkipPostProcessing;
            )",
                        fmt::arg("NREPostProcessingsClassAlias", cudaCallPrefix())));
        return Status();
    }
};

class NRECameraBilateralGridPostProcessing : public NRESkipPostProcessing {
public:
    static constexpr char name[] = "bilateral-grid-per-camera";

    NRECameraBilateralGridPostProcessing(const nlohmann::json& config,
                                         const Logger& logger,
                                         const nlohmann::json& stateDict,
                                         const std::string& prefix)
        : NRESkipPostProcessing(config, logger, stateDict, prefix) {
        LOG_WARN(logger, "NRECameraBilateralGridPostProcessing is not implemented : skipping");
    }

    virtual ~NRECameraBilateralGridPostProcessing() = default;
};

class NREFrameBilateralGridPostProcessing : public NRESkipPostProcessing {
public:
    static constexpr char name[] = "bilateral-grid-per-frame";

    NREFrameBilateralGridPostProcessing(const nlohmann::json& config,
                                        const Logger& logger,
                                        const nlohmann::json& stateDict,
                                        const std::string& prefix)
        : NRESkipPostProcessing(config, logger, stateDict, prefix) {
        LOG_WARN(logger, "NREFrameBilateralGridPostProcessing is not implemented : skipping");
    }

    virtual ~NREFrameBilateralGridPostProcessing() = default;
};

} // namespace nrend