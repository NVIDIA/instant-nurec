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

class NREWeightedInstanceInputEmbedding : public NREModel {
    int m_numEmbeddings = 0;
    int m_embeddingDims;

    TStateDictTensor m_moduleParamsTensor;

public:
    static constexpr char name[] = "weighted-instance-input-embedding";

    NREWeightedInstanceInputEmbedding(const nlohmann::json& config,
                                      const Logger& logger,
                                      const nlohmann::json& stateDict,
                                      const std::string& prefix)
        : NREModel(config, logger, stateDict, prefix, {}) {

        m_embeddingDims = config.value("embedding_dim", 1);

        m_moduleParamsTensor.key = prefix + "embedding.weight";
        if (readStateDictTensor(stateDict, m_moduleParamsTensor) && checkStateDictTensorSize<half>(m_moduleParamsTensor)) {
            m_numEmbeddings = m_moduleParamsTensor.shape[0];
        } else {
            LOG_INFO(logger, "NREWeightedInstanceInputEmbedding : missing valid tensor params {%s} in the state_dict.", m_moduleParamsTensor.key.c_str());
        }
    }

    virtual ~NREWeightedInstanceInputEmbedding() = default;

protected:
    virtual Status registerKernelResources_(
        const KernelMemoryBindings& memoryBindings,
        const KernelSourceCodeTable& sourceCodeTable,
        KernelResourcesProvider::KernelOpts,
        const Logger& logger) const override {

        Status status = memoryBindings.registerMemory(KernelMemoryBindings::BindingsFlag::Parameters,
                                                      m_moduleParamsTensor.key,
                                                      KernelMemoryType::Buffer,
                                                      logger);
        if (!status) {
            return status;
        }

        const std::string sourceCodeTemplate = R"(
            #include <nrend/kernels/cuda/models/nreInputEmbedding.cuh>
            using {NREWeightedInstanceInputEmbeddingClassAlias} =
               NREWeightedInstanceInputEmbedding<{NumEmbedding}, {EmbeddingDims}, {FirstBufferIdx}>;
        )";

        const int modelTensorIndex = memoryBindings.registeredMemoryIndex(KernelMemoryBindings::Parameters, m_moduleParamsTensor.key);

        sourceCodeTable.registerKernel(
            KernelSourceCodeTable::Cuda,
            fmt::format(sourceCodeTemplate,
                        fmt::arg("NREWeightedInstanceInputEmbeddingClassAlias", cudaCallPrefix()),
                        fmt::arg("NumEmbedding", m_numEmbeddings),
                        fmt::arg("EmbeddingDims", m_embeddingDims),
                        fmt::arg("FirstBufferIdx", modelTensorIndex)));

        return status;
    }

    virtual Status processKernelMemory_(
        const KernelMemoryBindings& memoryBindings,
        KernelMemoryBindings::BindingsFlag bindingsFlag,
        const std::vector<std::unique_ptr<KernelMemory>>& memory,
        ProcessMemoryFlag processFlag,
        uint64_t processQueueHandle,
        const Logger& logger) const override {

        Status status;

        if ((processFlag != ProcessMemoryFlag::Initialization) || (bindingsFlag != KernelMemoryBindings::Parameters)) {
            return status;
        }

        const int modelTensorIndex = memoryBindings.registeredMemoryIndex(bindingsFlag, m_moduleParamsTensor.key);
        if ((modelTensorIndex < 0) || (modelTensorIndex >= static_cast<int>(memory.size())) || !memory[modelTensorIndex]) {
            RETURN_ERROR(logger, ErrorCode::BadInput, "NREWeightedInstanceInputEmbedding : resource %s not correctly binded.", m_moduleParamsTensor.key.c_str());
        }
        return memory[modelTensorIndex]->setFromHost(m_moduleParamsTensor.buffer.data(), m_moduleParamsTensor.buffer.size(), processQueueHandle, logger);
    }
};

class NREIndividualRemapTimeInputEmbedding : public NREModel {
    int m_numEmbeddings = 0;
    float m_remapMin;
    float m_remapMax;

    TStateDictTensor m_moduleParamsTensor;

public:
    static constexpr char name[] = "individual-remap-time-input-embedding";

    NREIndividualRemapTimeInputEmbedding(const nlohmann::json& config,
                                         const Logger& logger,
                                         const nlohmann::json& stateDict,
                                         const std::string& prefix)
        : NREModel(config, logger, stateDict, prefix, {}) {

        m_remapMin = config.value("remap_min", 0.0f);
        m_remapMax = config.value("remap_max", 1.0f);

        // FIXME : individual-remap-embedding data comes from its parent
        m_moduleParamsTensor.key = prefix + "timestamps_us_ranges";
        if (readStateDictTensor(stateDict, m_moduleParamsTensor) && checkStateDictTensorSize<int64_t>(m_moduleParamsTensor)) {
            m_numEmbeddings = m_moduleParamsTensor.shape[0];
        } else {
            LOG_INFO(logger, "NREIndividualRemapTimeInputEmbedding : missing tensor params {%s} in the state_dict.", m_moduleParamsTensor.key.c_str());
        }
    }

    virtual ~NREIndividualRemapTimeInputEmbedding() = default;

protected:
    virtual Status registerKernelResources_(
        const KernelMemoryBindings& memoryBindings,
        const KernelSourceCodeTable& sourceCodeTable,
        KernelResourcesProvider::KernelOpts,
        const Logger& logger) const override {

        Status status = memoryBindings.registerMemory(KernelMemoryBindings::BindingsFlag::Parameters,
                                                      m_moduleParamsTensor.key,
                                                      KernelMemoryType::Buffer,
                                                      logger);
        if (!status) {
            return status;
        }

        const std::string sourceCodeTemplate = R"(
            #include <nrend/kernels/cuda/models/nreInputEmbedding.cuh>

            struct {NREIndividualRemapTimeInputEmbeddingClassAlias}Params
            {{
               static constexpr float remapMin     = {RemapMin};
               static constexpr float remapRange   = {RemapRange};
            }};
            using {NREIndividualRemapTimeInputEmbeddingClassAlias} = 
               NREIndividualRemapTimeInputEmbedding<{NumEmbedding}, {FirstBufferIdx}, {NREIndividualRemapTimeInputEmbeddingClassAlias}Params>;
        )";

        const int modelTensorIndex = memoryBindings.registeredMemoryIndex(KernelMemoryBindings::Parameters, m_moduleParamsTensor.key);

        sourceCodeTable.registerKernel(
            KernelSourceCodeTable::Cuda,
            fmt::format(sourceCodeTemplate,
                        fmt::arg("NREIndividualRemapTimeInputEmbeddingClassAlias", cudaCallPrefix()),
                        fmt::arg("RemapMin", m_remapMin),
                        fmt::arg("RemapRange", m_remapMax - m_remapMin),
                        fmt::arg("NumEmbedding", m_numEmbeddings),
                        fmt::arg("FirstBufferIdx", modelTensorIndex)));

        return status;
    }

    virtual Status processKernelMemory_(
        const KernelMemoryBindings& memoryBindings,
        KernelMemoryBindings::BindingsFlag bindingsFlag,
        const std::vector<std::unique_ptr<KernelMemory>>& memory,
        ProcessMemoryFlag processFlag,
        uint64_t processQueueHandle,
        const Logger& logger) const override {

        Status status;

        if ((processFlag != ProcessMemoryFlag::Initialization) || (bindingsFlag != KernelMemoryBindings::Parameters)) {
            return status;
        }

        const int modelTensorIndex = memoryBindings.registeredMemoryIndex(KernelMemoryBindings::Parameters, m_moduleParamsTensor.key);
        if ((modelTensorIndex < 0) || (modelTensorIndex >= static_cast<int>(memory.size())) || !memory[modelTensorIndex]) {
            RETURN_ERROR(logger, ErrorCode::BadInput, "NREIndividualRemapTimeInputEmbedding : resource %s not correctly binded.", m_moduleParamsTensor.key.c_str());
        }
        return memory[modelTensorIndex]->setFromHost(m_moduleParamsTensor.buffer.data(), m_moduleParamsTensor.buffer.size(), processQueueHandle, logger);
    }
};

} // namespace nrend