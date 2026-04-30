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

class NREFullyFusedTexture : public NREModel {
    int m_inputFeatureDim;
    bool m_includeNormals;
    bool m_includeXYZ;
    int m_outputDims;

    using TNetworkParams = __half;
    using TNetwork       = tcnn::NetworkWithInputEncoding<TNetworkParams>;
    std::unique_ptr<TNetwork> m_networkPtr;

    mutable bool m_moduleParamsHasJitLayout = false;
    mutable TStateDictTensor m_moduleParamsTensor;

public:
    static constexpr char name[] = "fully-fused-texture";

    NREFullyFusedTexture(const nlohmann::json& config,
                         const Logger& logger,
                         const nlohmann::json& stateDict,
                         const std::string& prefix)
        : NREModel(config, logger, stateDict, prefix, {}) {

        m_inputFeatureDim = config.value("input_feature_dim", 16);
        m_includeNormals  = config.value("include_normals", false);
        if (m_includeNormals) {
            LOG_ERROR(logger, "NREFullyFusedTexture : <include_normals> not implemented.");
        }
        m_includeXYZ = config.value("include_xyz", false);
        m_outputDims = config.value("n_output_dims", 16);

        const std::string extraStateKey    = prefix + "_extra_state";
        const bool stateDictWithExtraState = stateDict.contains(extraStateKey);
        if (!stateDictWithExtraState) {
            LOG_ERROR(logger, "NREFullyFusedTexture : missing extra state <%s> in the state_dict.",
                      extraStateKey.c_str());
        }

        const std::string encodingConfigKey = "encoding_config";
        const std::string networkConfigKey  = "mlp_network_config";
        if (stateDictWithExtraState && stateDict[extraStateKey].contains(encodingConfigKey) && config.contains(networkConfigKey)) {
            const nlohmann::json& encodingConfig = stateDict[extraStateKey][encodingConfigKey];
            const nlohmann::json& networkConfig  = config[networkConfigKey];

            const int inputDim = m_inputFeatureDim + (m_includeXYZ ? 3 : 0) + (m_includeNormals ? 3 : 0) + 3;
            m_networkPtr       = std::make_unique<TNetwork>(inputDim, m_outputDims, encodingConfig, networkConfig);
            m_networkPtr->set_params(nullptr, nullptr, nullptr);
        } else {
            if (stateDictWithExtraState && !stateDict[extraStateKey].contains(encodingConfigKey)) {
                LOG_WARN(logger, "NREFullyFusedTexture : missing encoding configurations <%s> in the state_dict.",
                         encodingConfigKey.c_str());
            }
            if (!config.contains(networkConfigKey)) {
                LOG_WARN(logger, "NREFullyFusedTexture : missing mlp configurations <%s> in the config.",
                         networkConfigKey.c_str());
            }
        }

        m_moduleParamsTensor.key = prefix + "rgb_net.params";
        if (!readStateDictTensor(stateDict, m_moduleParamsTensor, m_networkPtr ? m_networkPtr->n_params() * sizeof(TNetworkParams) : 0)) {
            LOG_WARN(logger,
                     "NREFullyFusedTexture : missing valid encoding parameters tensor <%s> [%d/%d] in the state_dict.",
                     m_moduleParamsTensor.key.c_str(),
                     static_cast<int>(m_moduleParamsTensor.buffer.size() / sizeof(TNetworkParams)),
                     static_cast<int>(m_networkPtr ? m_networkPtr->n_params() : 0));
            m_networkPtr.reset();
        }
    }

    virtual ~NREFullyFusedTexture() = default;

protected:
    virtual std::string skipKernelCode() const {

        const std::string sourceCodeTemplate = R"(
            #include <nrend/kernels/cuda/models/nreTexture.cuh>
            using {NREFullyFusedTextureClassAlias} = NRESkipTexture<{InputDim}, {OutputDim}>;
        )";

        return fmt::format(sourceCodeTemplate,
                           fmt::arg("NREFullyFusedTextureClassAlias", cudaCallPrefix()),
                           fmt::arg("InputDim", m_inputFeatureDim),
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
            #include <nrend/kernels/cuda/models/nreTexture.cuh>
            
            struct {NREFullyFusedTextureClassAlias}Evaluator
            {{
            protected:
               {NetworkEvalBody}
            }};
            using {NREFullyFusedTextureClassAlias} = 
               NREFullyFusedTexture<{FeatureDim}, {PositionDim}, {NREFullyFusedTextureClassAlias}Evaluator, {FirstBufferIdx}>;
        )";

        const std::string networkEvalBodyStr = m_networkPtr->generate_device_function("_eval");
        const int modelTensorIndex           = memoryBindings.registeredMemoryIndex(KernelMemoryBindings::Parameters, m_moduleParamsTensor.key);

        sourceCodeTable.registerKernel(
            KernelSourceCodeTable::Cuda,
            fmt::format(sourceCodeTemplate,
                        fmt::arg("NREFullyFusedTextureClassAlias", cudaCallPrefix()),
                        fmt::arg("NetworkEvalBody", networkEvalBodyStr),
                        fmt::arg("FeatureDim", m_inputFeatureDim),
                        fmt::arg("PositionDim", (m_includeXYZ ? 3 : 0)),
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

        if ((processFlag != ProcessMemoryFlag::Initialization) || (bindingsFlag != KernelMemoryBindings::Parameters) || !m_networkPtr) {
            return status;
        }

        const int modelTensorIndex = memoryBindings.registeredMemoryIndex(bindingsFlag, m_moduleParamsTensor.key);
        if ((modelTensorIndex < 0) || (modelTensorIndex >= static_cast<int>(memory.size())) || !memory[modelTensorIndex]) {
            RETURN_ERROR(logger, ErrorCode::BadInput, "NREFullyFusedTexture : resource %s not correctly binded.", m_moduleParamsTensor.key.c_str());
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

} // namespace nrend