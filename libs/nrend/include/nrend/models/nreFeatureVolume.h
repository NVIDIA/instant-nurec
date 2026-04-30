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

#include <tiny-cuda-nn/network_with_input_encoding.h>

namespace nrend {

class NREHashGridFeatureVolume : public NREModel {

protected:
    bool m_includeXYZ;
    int m_inputDims;
    int m_posDims;
    int m_outputDims;

    bool m_encodingHasLevelInput = false;
    int m_nEncodingLevels        = 1;
    int m_nEncodingActiveLevels  = 0;

    using TNetworkParams = __half;
    using TNetwork       = tcnn::NetworkWithInputEncoding<TNetworkParams>;
    std::unique_ptr<TNetwork> m_networkPtr;

    mutable bool m_moduleParamsHasJitLayout = false;
    mutable TStateDictTensor m_moduleParamsTensor;

public:
    static constexpr char name[] = "hash-grid";

    NREHashGridFeatureVolume(const nlohmann::json& config,
                             const Logger& logger,
                             const nlohmann::json& stateDict,
                             const std::string& prefix)
        : NREModel(config, logger, stateDict, prefix, {}) {

        m_includeXYZ = config.value("include_xyz", false);
        m_inputDims  = config.value("n_input_dims", 3);
        m_posDims    = config.value("n_pos_dims", 3);
        m_outputDims = config.value("n_output_dims", 16);

        const std::string extraStateKey    = prefix + "encoding._extra_state";
        const bool stateDictWithExtraState = stateDict.contains(extraStateKey);
        if (!stateDictWithExtraState) {
            LOG_WARN(logger, "NREHashGridFeatureVolume : missing extra state <%s> in the state_dict.",
                     extraStateKey.c_str());
        }

        const std::string encodingConfigKey   = "encoding_config";
        const std::string mlpNetworkConfigKey = "mlp_network_config";
        if (stateDictWithExtraState && stateDict[extraStateKey].contains(encodingConfigKey) && config.contains(mlpNetworkConfigKey)) {
            const nlohmann::json& encodingConfig   = stateDict[extraStateKey][encodingConfigKey];
            const nlohmann::json& mlpNetworkConfig = config[mlpNetworkConfigKey];

            m_nEncodingLevels       = stateDict[extraStateKey].value("n_levels", 0);
            m_encodingHasLevelInput = m_nEncodingLevels >= 0;
            m_nEncodingActiveLevels = stateDict[extraStateKey].value("n_active_levels", m_nEncodingLevels);
            // encoding input dims includes all embeddings (model input dims does not)
            const int networkInputDims = stateDict[extraStateKey].value("n_input_dims", 3) + (m_nEncodingActiveLevels ? 1 : 0);

            m_networkPtr = std::make_unique<TNetwork>(networkInputDims, m_outputDims, encodingConfig, mlpNetworkConfig);
            m_networkPtr->set_params(nullptr, nullptr, nullptr);
        } else {
            if (stateDictWithExtraState && !stateDict.contains(encodingConfigKey)) {
                LOG_WARN(logger, "NREHashGridFeatureVolume : missing encoding configurations <%s> in the state_dict.",
                         encodingConfigKey.c_str());
            }
            if (!config.contains(mlpNetworkConfigKey)) {
                LOG_WARN(logger, "NREHashGridFeatureVolume : missing mlp configurations <%s> in the config.",
                         mlpNetworkConfigKey.c_str());
            }
        }

        m_moduleParamsTensor.key = prefix + "encoding.tcnn_module.params";
        if (!readStateDictTensor(stateDict, m_moduleParamsTensor, m_networkPtr ? m_networkPtr->n_params() * sizeof(TNetworkParams) : 0)) {
            LOG_WARN(logger,
                     "NREHashGridFeatureVolume : missing valid encoding parameters tensor <%s> [%d/%d] in the state_dict.",
                     m_moduleParamsTensor.key.c_str(),
                     static_cast<int>(m_moduleParamsTensor.buffer.size() / sizeof(TNetworkParams)),
                     static_cast<int>(m_networkPtr ? m_networkPtr->n_params() : 0));
            m_networkPtr.reset();
        }
    }

    virtual ~NREHashGridFeatureVolume() = default;

protected:
    virtual std::string levelEncodingKernelCode() const {
        if (m_encodingHasLevelInput) {
            return fmt::format(R"(
                struct {NREHashGridFeatureVolumeClassAlias}LevelEncoding
                {{
                   static constexpr int Dim = 1;
                   static inline __device__ tcnn::vec<1> eval() {{return tcnn::vec<1>{{encoding}};}}

                protected:
                   static constexpr float encoding = {LevelEncoding};
                }};
            )",
                               fmt::arg("NREHashGridFeatureVolumeClassAlias", cudaCallPrefix()),
                               fmt::arg("LevelEncoding", m_nEncodingLevels > 0 ? static_cast<float>(m_nEncodingActiveLevels) / m_nEncodingLevels : 0.f));
        } else {
            return fmt::format(R"(
                struct {NREHashGridFeatureVolumeClassAlias}LevelEncoding
                {{
                   static constexpr int Dim = 0;
                   static inline __device__ tcnn::vec<1> eval() {{
                       __builtin_unreachable();
                       return tcnn::vec<1>::zero();
                   }}
                }};
            )",
                               fmt::arg("NREHashGridFeatureVolumeClassAlias", cudaCallPrefix()));
        }
    }

    std::string skipKernelCode() const {
        return fmt::format(R"(
            #include <nrend/kernels/cuda/models/nreFeatureVolume.cuh>

            using {NREHashGridFeatureVolumeClassAlias} = NRESkipFeatureVolume<{OutputDim}>;
        )",
                           fmt::arg("NREHashGridFeatureVolumeClassAlias", cudaCallPrefix()),
                           fmt::arg("OutputDim", m_outputDims));
    }

    virtual Status registerKernelResources_(
        const KernelMemoryBindings& memoryBindings,
        const KernelSourceCodeTable& sourceCodeTable,
        KernelResourcesProvider::KernelOpts,
        const Logger& logger) const override {

        Status status;

        if (!m_networkPtr) {
            sourceCodeTable.registerKernel(
                KernelSourceCodeTable::Cuda,
                skipKernelCode());
            return status;
        }

        status = memoryBindings.registerMemory(KernelMemoryBindings::BindingsFlag::Parameters,
                                               m_moduleParamsTensor.key,
                                               KernelMemoryType::Buffer,
                                               logger);
        if (!status) {
            return status;
        }

        const std::string sourceCodeTemplate = R"(
            #include <nrend/kernels/cuda/models/nreFeatureVolume.cuh>

            {LevelEncodingBody}
            
            struct {NREHashGridFeatureVolumeClassAlias}Evaluator
            {{
            protected:
               {NetworkEvalBody}
            }};
            using {NREHashGridFeatureVolumeClassAlias} = 
               NREHashGridFeatureVolume<{NREHashGridFeatureVolumeClassAlias}LevelEncoding, {NREHashGridFeatureVolumeClassAlias}Evaluator, {FirstBufferIdx}>;
        )";

        const std::string networkEvalBodyStr = m_networkPtr->generate_device_function("_eval");
        const int modelTensorIndex           = memoryBindings.registeredMemoryIndex(KernelMemoryBindings::Parameters, m_moduleParamsTensor.key);

        sourceCodeTable.registerKernel(
            KernelSourceCodeTable::Cuda,
            fmt::format(sourceCodeTemplate,
                        fmt::arg("LevelEncodingBody", levelEncodingKernelCode()),
                        fmt::arg("NREHashGridFeatureVolumeClassAlias", cudaCallPrefix()),
                        fmt::arg("NetworkEvalBody", networkEvalBodyStr),
                        fmt::arg("FirstBufferIdx", modelTensorIndex)));

        return Status();
    }

    virtual Status processKernelMemory_(
        const KernelMemoryBindings& memoryBindings,
        KernelMemoryBindings::BindingsFlag bindingsFlag,
        const std::vector<std::unique_ptr<KernelMemory>>& memory,
        ProcessMemoryFlag processFlag,
        uint64_t processQueueHandle,
        const Logger& logger) const override {

        Status status;

        if ((processFlag != ProcessMemoryFlag::Initialization) || (bindingsFlag != KernelMemoryBindings::Parameters) || !m_networkPtr) {
            return status;
        }

        const int modelTensorIndex = memoryBindings.registeredMemoryIndex(bindingsFlag, m_moduleParamsTensor.key);
        if ((modelTensorIndex < 0) || (modelTensorIndex >= static_cast<int>(memory.size())) || !memory[modelTensorIndex]) {
            RETURN_ERROR(logger, ErrorCode::BadInput, "NREHashGridFeatureVolume : resource %s not correctly binded.", m_moduleParamsTensor.key.c_str());
        }

        if (!m_moduleParamsHasJitLayout) {
            CHECK_STATUS_RETURN(memory[modelTensorIndex]->setFromHost(m_moduleParamsTensor.buffer.data(),
                                                                      m_moduleParamsTensor.buffer.size(),
                                                                      processQueueHandle,
                                                                      logger));
            m_networkPtr->set_params(nullptr,
                                     reinterpret_cast<TNetworkParams*>(memory[modelTensorIndex]->data()),
                                     nullptr);
            m_networkPtr->convert_params_to_jit_layout(reinterpret_cast<cudaStream_t>(processQueueHandle), true);
            m_networkPtr->set_params(nullptr, nullptr, nullptr);
            // copy the converted parameters to the host
            CUDA_CHECK_RETURN(cudaMemcpyAsync(m_moduleParamsTensor.buffer.data(),
                                              memory[modelTensorIndex]->data(),
                                              m_moduleParamsTensor.buffer.size(),
                                              cudaMemcpyDeviceToHost,
                                              reinterpret_cast<cudaStream_t>(processQueueHandle)),
                              logger);
            cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(processQueueHandle));
            m_moduleParamsHasJitLayout = true;
        }

        CHECK_STATUS_RETURN(memory[modelTensorIndex]->setFromHost(m_moduleParamsTensor.buffer.data(),
                                                                  m_moduleParamsTensor.buffer.size(),
                                                                  processQueueHandle,
                                                                  logger));

        return status;
    }
};

class NREHashGridObjectFeatureVolume : public NREHashGridFeatureVolume {

    bool m_timeDependent;

public:
    static constexpr char name[] = "hash-grid-object";

    NREHashGridObjectFeatureVolume(const nlohmann::json& config,
                                   const Logger& logger,
                                   const nlohmann::json& stateDict,
                                   const std::string& prefix)
        : NREHashGridFeatureVolume(config, logger, stateDict, prefix) {

        // discard skip time embedding for performance reason
        const std::string skipName = "skip-input-embedding";
        m_timeDependent            = config.contains("time_input_embedding") && (config["time_input_embedding"].value("name", skipName) != skipName);

        // setup the time and instance embeddings submodels
        std::vector<std::string> embeddingSubModelStrs = {"instance_input_embedding"};
        if (m_timeDependent) {
            embeddingSubModelStrs.push_back("time_input_embedding");
        }

        initializeSubModels(
            config,
            stateDict,
            prefix,
            embeddingSubModelStrs,
            logger);
    }

    virtual ~NREHashGridObjectFeatureVolume() = default;

protected:
    virtual Status registerKernelResources_(
        const KernelMemoryBindings& memoryBindings,
        const KernelSourceCodeTable& sourceCodeTable,
        KernelResourcesProvider::KernelOpts,
        const Logger& logger) const override {

        Status status;

        if (!m_networkPtr) {
            sourceCodeTable.registerKernel(
                KernelSourceCodeTable::Cuda,
                skipKernelCode());
            return status;
        }

        status = memoryBindings.registerMemory(KernelMemoryBindings::BindingsFlag::Parameters,
                                               m_moduleParamsTensor.key,
                                               KernelMemoryType::Buffer,
                                               logger);
        if (!status) {
            return status;
        }

        const std::string sourceCodeTemplate = R"(
            #include <nrend/kernels/cuda/models/nreFeatureVolume.cuh>
            
            {LevelEncodingBody}
            
            struct {NREHashGridObjectFeatureVolumeClassAlias}Evaluator
            {{
            protected:
               {NetworkEvalBody}
            }};
            using {NREHashGridObjectFeatureVolumeClassAlias} = 
               {NREHashGridObjectFeatureVolumeClass}<{InstanceEmbeddingClassAlias}, {TimeEmbeddingClassAlias}, 
                                                     {NREHashGridObjectFeatureVolumeClassAlias}LevelEncoding,
                                                     {NREHashGridObjectFeatureVolumeClassAlias}Evaluator, {FirstBufferIdx}>;
        )";

        const std::string networkEvalBodyStr = m_networkPtr->generate_device_function("_eval");
        const std::string classNameStr       = m_timeDependent ? "NREHashGridTimedObjectFeatureVolume" : "NREHashGridObjectFeatureVolume";
        const int modelTensorIndex           = memoryBindings.registeredMemoryIndex(KernelMemoryBindings::Parameters, m_moduleParamsTensor.key);

        sourceCodeTable.registerKernel(
            KernelSourceCodeTable::Cuda,
            fmt::format(sourceCodeTemplate,
                        fmt::arg("LevelEncodingBody", levelEncodingKernelCode()),
                        fmt::arg("NREHashGridObjectFeatureVolumeClassAlias", cudaCallPrefix()),
                        fmt::arg("NetworkEvalBody", networkEvalBodyStr),
                        fmt::arg("NREHashGridObjectFeatureVolumeClass", classNameStr),
                        fmt::arg("InstanceEmbeddingClassAlias", cudaCallPrefix("instance_input_embedding")),
                        fmt::arg("TimeEmbeddingClassAlias", m_timeDependent ? cudaCallPrefix("time_input_embedding") : "void"),
                        fmt::arg("FirstBufferIdx", modelTensorIndex)));

        return status;
    }
};
} // namespace nrend