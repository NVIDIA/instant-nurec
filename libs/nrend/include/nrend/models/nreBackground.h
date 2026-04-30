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

#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/vec.h>

#include <nrend/utils/cuda/cudaCommon.h>

namespace nrend {

class NRESkipBackground : public NREModel {

public:
    static constexpr char name[] = "skip-background";

    NRESkipBackground(const nlohmann::json& config,
                      const Logger& logger,
                      const nlohmann::json& stateDict,
                      const std::string& prefix)
        : NREModel(config, logger, stateDict, prefix, {}) {
    }

    virtual ~NRESkipBackground() = default;

protected:
    virtual Status registerKernelResources_(
        const KernelMemoryBindings&,
        const KernelSourceCodeTable& sourceCodeTable,
        KernelResourcesProvider::KernelOpts,
        const Logger&) const override {

        sourceCodeTable.registerKernel(
            KernelSourceCodeTable::Cuda,
            fmt::format(R"(
                #include <nrend/kernels/cuda/models/nreBackground.cuh>
                
                using {NREBackgroundClassAlias} = NRESkipBackground;
            )",
                        fmt::arg("NREBackgroundClassAlias", cudaCallPrefix())));
        return Status();
    }
};

class NREColorBackground : public NREModel {
    tcnn::vec3 m_color;
    bool m_compositeInLinearSpace;

public:
    static constexpr char name[] = "background-color";

    NREColorBackground(const nlohmann::json& config,
                       const Logger& logger,
                       const nlohmann::json& stateDict,
                       const std::string& prefix)
        : NREModel(config, logger, stateDict, prefix, {}) {
        m_color                  = colorNameToVec(config.value("color", "black"));
        m_compositeInLinearSpace = config.value("composite_in_linear_space", false);
        if (m_compositeInLinearSpace) {
            LOG_ERROR(logger, "NREColorBackground : composition in linear space is not supported.");
        }
    }

    virtual ~NREColorBackground() = default;

protected:
    virtual Status registerKernelResources_(
        const KernelMemoryBindings&,
        const KernelSourceCodeTable& sourceCodeTable,
        KernelResourcesProvider::KernelOpts,
        const Logger&) const override {

        const std::string sourceCodeTemplate = R"(
            #include <nrend/kernels/cuda/models/nreBackground.cuh>

            struct {NREBackgroundClassAlias}Params
            {{
                static constexpr bool supportAppearanceEmbedding = false;
                const tcnn::vec3 color = tcnn::vec3({colorX}, {colorY}, {colorZ});
            }};
            using {NREBackgroundClassAlias} = NREColorBackground<{NREBackgroundClassAlias}Params>;
        )";

        sourceCodeTable.registerKernel(
            KernelSourceCodeTable::Cuda,
            fmt::format(sourceCodeTemplate,
                        fmt::arg("NREBackgroundClassAlias", cudaCallPrefix()),
                        fmt::arg("colorX", m_color[0]),
                        fmt::arg("colorY", m_color[1]),
                        fmt::arg("colorZ", m_color[2])));

        return Status();
    }

private:
    static inline tcnn::vec3 colorNameToVec(const std::string& color) {
        if (color == "white") {
            return tcnn::vec3::ones();
        }
        // random and black colors
        return tcnn::vec3::zero();
    }
};

class NREEnvMapBackground : public NREModel {
    using TParams = __half;

    tcnn::vec3 m_color;
    bool m_compositeInLinearSpace = false;
    bool m_saturateRadiance       = true;

    uint32_t m_width  = 512;
    uint32_t m_height = 512;

    enum EnvMapType {
        EQUIRECTANGULAR,
        CUBEMAP,
        INVALID
    };
    EnvMapType m_envMapType;
    std::string m_envMapTypeStr = "cubemap";

    TStateDictTensor m_texturesRGBA32F;

public:
    static constexpr char name[] = "sky-env-map";

    NREEnvMapBackground(const nlohmann::json& config,
                        const Logger& logger,
                        const nlohmann::json& stateDict,
                        const std::string& prefix)
        : NREModel(config, logger, stateDict, prefix, {}) {

        m_compositeInLinearSpace = config.value("composite_in_linear_space", m_compositeInLinearSpace);
        m_saturateRadiance       = config.value("saturate_radiance", m_saturateRadiance);
        m_width                  = config.value("width", m_width);
        m_height                 = config.value("height", m_height);
        m_envMapTypeStr          = config.value("envmap_type", m_envMapTypeStr);

        m_envMapType = envmapNameToEnum(m_envMapTypeStr, logger);
        if (m_envMapType == EnvMapType::INVALID) {
            LOG_ERROR(logger, "NREEnvMapBackground : invalid envmap type <%s>.", m_envMapTypeStr.c_str());
            return;
        }

        size_t nTexels = 0;
        if (m_envMapType == EnvMapType::CUBEMAP) {
            nTexels = m_width * m_height * 6;
        } else if (m_envMapType == EnvMapType::EQUIRECTANGULAR) {
            nTexels = m_width * m_height;
        } else {
            LOG_ERROR(logger, "NREEnvMapBackground : invalid envmap type <%s>.", m_envMapTypeStr.c_str());
            return;
        }

        // Read texture data from the state dict
        TStateDictTensor texturesRGB16F;
        texturesRGB16F.key = prefix + "textures";

        const size_t nBytes = nTexels * sizeof(TParams) * 3;

        if (!readStateDictTensor(stateDict, texturesRGB16F, nBytes)) {
            LOG_ERROR(logger,
                      "NREEnvMapBackground : missing valid texture %s tensor <%s> [%d/%d] in the state_dict.",
                      m_envMapTypeStr.c_str(),
                      texturesRGB16F.key.c_str(),
                      static_cast<int>(texturesRGB16F.buffer.size()),
                      static_cast<int>(nBytes));
            return;
        }

        // Convert to float4 texture
        // FIXME: Need to check if this initialization is taking too much time.
        // NOTE(qi): Perform conversion from float16 to float32 and padding on CPU, because the initialization is expected to be executed only once.
        m_texturesRGBA32F.key   = texturesRGB16F.key;
        m_texturesRGBA32F.shape = texturesRGB16F.shape;
        m_texturesRGBA32F.buffer.resize(nTexels * sizeof(float4));

        const TParams* srcPtr = reinterpret_cast<const TParams*>(texturesRGB16F.buffer.data());
        tcnn::vec4* dstPtr    = reinterpret_cast<tcnn::vec4*>(m_texturesRGBA32F.buffer.data());

        // FIXME: This is probably make no difference until we enable OpenMP. We can investigate potential multi-threading here.
#pragma omp parallel for
        for (size_t i = 0; i < nTexels; ++i) {
            const tcnn::vec3 vec3 = tcnn::vec3(srcPtr[i * 3 + 0], srcPtr[i * 3 + 1], srcPtr[i * 3 + 2]);
            dstPtr[i]             = tcnn::vec4(vec3, 1.0f);
        }
    }

protected:
    virtual Status registerKernelResources_(
        const KernelMemoryBindings& memoryBindings,
        const KernelSourceCodeTable& sourceCodeTable,
        KernelResourcesProvider::KernelOpts,
        const Logger& logger) const override {

        if (m_envMapType == EnvMapType::CUBEMAP) {
            CHECK_STATUS_RETURN(
                memoryBindings.registerMemory(KernelMemoryBindings::BindingsFlag::Parameters,
                                              m_texturesRGBA32F.key,
                                              KernelMemoryType::TextureCubeMap_RGBA_32F,
                                              logger));
        } else if (m_envMapType == EnvMapType::EQUIRECTANGULAR) {
            CHECK_STATUS_RETURN(
                memoryBindings.registerMemory(KernelMemoryBindings::BindingsFlag::Parameters,
                                              m_texturesRGBA32F.key,
                                              KernelMemoryType::Texture2D_RGBA_32F,
                                              logger));
        } else {
            LOG_ERROR(logger, "NREEnvMapBackground : invalid envmap type <%s>.", m_envMapTypeStr.c_str());
            return Status();
        }

        const std::string sourceCodeTemplate = R"(
            #include <nrend/kernels/cuda/models/nreBackground.cuh>

            struct {NREBackgroundClassAlias}Params
            {{
                static constexpr bool IsCubeMap         = {IsCubeMap};
                static constexpr int TextureHandleIndex = {TextureHandleIndex};    
                static constexpr bool SaturateRadiance  = {SaturateRadiance};
            }};
            using {NREBackgroundClassAlias} = NREEnvMapBackground<{NREBackgroundClassAlias}Params>;
        )";

        const int bindedTensorIndex = memoryBindings.registeredMemoryIndex(KernelMemoryBindings::BindingsFlag::Parameters, m_texturesRGBA32F.key);

        sourceCodeTable.registerKernel(
            KernelSourceCodeTable::Cuda,
            fmt::format(sourceCodeTemplate,
                        fmt::arg("NREBackgroundClassAlias", cudaCallPrefix()),
                        fmt::arg("TextureHandleIndex", bindedTensorIndex),
                        fmt::arg("IsCubeMap", m_envMapType == EnvMapType::CUBEMAP),
                        fmt::arg("SaturateRadiance", m_saturateRadiance)));

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

        if ((processFlag != ProcessMemoryFlag::Initialization) || (bindingsFlag != KernelMemoryBindings::Parameters)) {
            return status;
        }

        nrend::KernelMemoryExtend extend;
        if (m_envMapType == EnvMapType::CUBEMAP) {
            extend.type          = nrend::KernelMemoryType::TextureCubeMap_RGBA_32F;
            extend.cubeMap.width = m_width;
        } else if (m_envMapType == EnvMapType::EQUIRECTANGULAR) {
            extend.type         = nrend::KernelMemoryType::Texture2D_RGBA_32F;
            extend.tex2D.width  = m_width;
            extend.tex2D.height = m_height;
        } else {
            LOG_ERROR(logger, "NREEnvMapBackground : invalid envmap type <%s>.", m_envMapTypeStr.c_str());
            return status;
        }

        const int bindedTensorIndex = memoryBindings.registeredMemoryIndex(bindingsFlag, m_texturesRGBA32F.key);
        CHECK_STATUS_RETURN(memory[bindedTensorIndex]->setFromHost(m_texturesRGBA32F.buffer.data(), extend, processQueueHandle, logger));

        return status;
    }

private:
    static inline EnvMapType envmapNameToEnum(const std::string& type, const Logger& logger) {
        if (type == "cubemap") {
            return EnvMapType::CUBEMAP;
        }
        if (type == "equirectangular") {
            return EnvMapType::EQUIRECTANGULAR;
        }
        return EnvMapType::INVALID;
    }
};

class NRESkyMLPBackground : public NREModel {
    bool m_compositeInLinearSpace = false;
    bool m_useAppearanceEmbedding = false;
    int m_appearanceEmbeddingDim  = 0;

    using TNetworkParams = __half;
    using TEncoding      = tcnn::Encoding<TNetworkParams>;
    std::unique_ptr<TEncoding> m_encodingPtr;
    using TNetwork = tcnn::NetworkWithInputEncoding<TNetworkParams>;
    std::unique_ptr<TNetwork> m_networkPtr;

    struct TNetworkTensor {
        TStateDictTensor tensor;
        bool hasJitLayout = false; //< tcnn requires a specific layout for jit
    };
    enum NetworkTensorIds {
        EncodingNetworkId,
        MlpNetworkId,
        NumNetworkTensors
    };
    mutable std::array<TNetworkTensor, NumNetworkTensors> m_networkParams;

public:
    static constexpr char name[] = "sky-mlp";

    NRESkyMLPBackground(const nlohmann::json& config,
                        const Logger& logger,
                        const nlohmann::json& stateDict,
                        const std::string& prefix)
        : NREModel(config, logger, stateDict, prefix, {}) {

        m_compositeInLinearSpace = config.value("composite_in_linear_space", m_compositeInLinearSpace);
        m_useAppearanceEmbedding = config.value("n_use_appearance_embeddings", m_useAppearanceEmbedding);
        m_appearanceEmbeddingDim = config.value("appearance_embedding_dim", m_appearanceEmbeddingDim);

        const std::string encodingConfigKey = "dir_encoding_config";
        if (config.contains(encodingConfigKey)) {
            const nlohmann::json& encodingConfig = config[encodingConfigKey];
            m_encodingPtr.reset(tcnn::create_encoding<TNetworkParams>(3, encodingConfig));
            if (m_encodingPtr) {
                m_encodingPtr->set_params(nullptr, nullptr, nullptr);
            }
        } else {
            LOG_ERROR(logger, "NRESkyMLPBackground : missing encoding configuration <%s> in the config.",
                      encodingConfigKey.c_str());
        }

        const std::string mlpNetworkConfigKey = "mlp_network_config";
        if (config.contains(mlpNetworkConfigKey)) {
            const int inputDims  = (m_encodingPtr ? m_encodingPtr->padded_output_width() : 0) + m_appearanceEmbeddingDim;
            const int outputDims = 3;
            m_networkPtr.reset(new tcnn::NetworkWithInputEncoding<TNetworkParams>(inputDims, outputDims, {{"otype", "Identity"}}, config[mlpNetworkConfigKey]));
            if (m_networkPtr) {
                m_networkPtr->set_params(nullptr, nullptr, nullptr);
            }
        } else {
            LOG_ERROR(logger, "NRESkyMLPBackground : missing network configuration <%s> in the config.",
                      mlpNetworkConfigKey.c_str());
        }

        m_networkParams[EncodingNetworkId].tensor.key = prefix + "dir_encoding.params";
        if (!readStateDictTensor(stateDict, m_networkParams[EncodingNetworkId].tensor, m_encodingPtr ? m_encodingPtr->n_params() * sizeof(TNetworkParams) : 0)) {
            LOG_ERROR(logger,
                      "NRESkyMLPBackground : missing valid encoding parameters tensor <%s> [%d/%d] in the state_dict.",
                      m_networkParams[EncodingNetworkId].tensor.key.c_str(),
                      static_cast<int>(m_networkParams[EncodingNetworkId].tensor.buffer.size() / sizeof(TNetworkParams)),
                      static_cast<int>(m_encodingPtr ? m_encodingPtr->n_params() : 0));
        }

        m_networkParams[MlpNetworkId].tensor.key = prefix + "sky_mlp.params";
        if (!readStateDictTensor(stateDict, m_networkParams[MlpNetworkId].tensor, m_networkPtr ? m_networkPtr->n_params() * sizeof(TNetworkParams) : 0)) {
            LOG_ERROR(logger,
                      "NRESkyMLPBackground : missing valid encoding parameters tensor <%s> [%d/%d] in the state_dict.",
                      m_networkParams[MlpNetworkId].tensor.key.c_str(),
                      static_cast<int>(m_networkParams[MlpNetworkId].tensor.buffer.size() / sizeof(TNetworkParams)),
                      static_cast<int>(m_networkPtr ? m_networkPtr->n_params() : 0));
        }
    }

    virtual ~NRESkyMLPBackground() = default;

protected:
    virtual Status registerKernelResources_(
        const KernelMemoryBindings& memoryBindings,
        const KernelSourceCodeTable& sourceCodeTable,
        KernelResourcesProvider::KernelOpts,
        const Logger& logger) const override {

        Status status;

        for (const TNetworkTensor& tensor : m_networkParams) {
            status = memoryBindings.registerMemory(KernelMemoryBindings::BindingsFlag::Parameters,
                                                   tensor.tensor.key,
                                                   KernelMemoryType::Buffer,
                                                   logger);
            if (!status) {
                return status;
            }
        }

        const std::string sourceCodeTemplate = R"(
            #include <nrend/kernels/cuda/models/nreBackground.cuh>

            struct {NRESkyMLPBackgroundClassAlias}Evaluator
            {{
            protected:
                {EncodingEvalBody}
                {NetworkEvalBody}
            }};
            using {NRESkyMLPBackgroundClassAlias} = NRESkyMLPBackground<{EncodingDim}, {AppearenceEmbeddingDim}, {NRESkyMLPBackgroundClassAlias}Evaluator, {EncodingBufferIdx}, {NetworkBufferIdx}>;
        )";

        const std::string encodingEvalBodyStr =
            m_encodingPtr ? m_encodingPtr->generate_device_function("_evalEncoding") : std::string();
        const std::string networkEvalBodyStr =
            m_networkPtr ? m_networkPtr->generate_device_function("_evalNetwork") : std::string();

        std::array<int, NumNetworkTensors> bindedTensorIndex;
        for (size_t i = 0; i < NumNetworkTensors; ++i) {
            bindedTensorIndex[i] = memoryBindings.registeredMemoryIndex(KernelMemoryBindings::Parameters, m_networkParams[i].tensor.key);
        }

        sourceCodeTable.registerKernel(
            KernelSourceCodeTable::Cuda,
            fmt::format(sourceCodeTemplate,
                        fmt::arg("NRESkyMLPBackgroundClassAlias", cudaCallPrefix()),
                        fmt::arg("EncodingEvalBody", encodingEvalBodyStr),
                        fmt::arg("NetworkEvalBody", networkEvalBodyStr),
                        fmt::arg("EncodingDim", m_encodingPtr ? m_encodingPtr->padded_output_width() : 0),
                        fmt::arg("AppearenceEmbeddingDim", m_appearanceEmbeddingDim),
                        fmt::arg("EncodingBufferIdx", bindedTensorIndex[EncodingNetworkId]),
                        fmt::arg("NetworkBufferIdx", bindedTensorIndex[MlpNetworkId])));

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

        // fetch the binded memory indices
        std::array<int, NumNetworkTensors> bindedTensorIndex;
        for (size_t i = 0; i < NumNetworkTensors; ++i) {
            bindedTensorIndex[i] = memoryBindings.registeredMemoryIndex(bindingsFlag, m_networkParams[i].tensor.key);
            if ((bindedTensorIndex[i] < 0) || (bindedTensorIndex[i] >= memory.size()) || !memory[bindedTensorIndex[i]]) {
                RETURN_ERROR(logger, ErrorCode::BadInput, "NRESkyMLPBackground : resource %s not correctly binded.", m_networkParams[i].tensor.key.c_str());
            }
        }

        // convert the network parameters to the jit layout
        if (!m_networkParams[EncodingNetworkId].hasJitLayout) {
            CHECK_STATUS_RETURN(memory[bindedTensorIndex[EncodingNetworkId]]->setFromHost(
                m_networkParams[EncodingNetworkId].tensor.buffer.data(),
                m_networkParams[EncodingNetworkId].tensor.buffer.size(),
                processQueueHandle,
                logger));
            m_encodingPtr->set_params(nullptr,
                                      reinterpret_cast<TNetworkParams*>(memory[bindedTensorIndex[EncodingNetworkId]]->data()),
                                      nullptr);
            m_encodingPtr->convert_params_to_jit_layout(reinterpret_cast<cudaStream_t>(processQueueHandle), true);
            m_encodingPtr->set_params(nullptr, nullptr, nullptr);
            // copy the converted parameters to the host
            CUDA_CHECK_RETURN(cudaMemcpyAsync(m_networkParams[EncodingNetworkId].tensor.buffer.data(),
                                              memory[bindedTensorIndex[EncodingNetworkId]]->data(),
                                              m_networkParams[EncodingNetworkId].tensor.buffer.size(),
                                              cudaMemcpyDeviceToHost,
                                              reinterpret_cast<cudaStream_t>(processQueueHandle)),
                              logger);
            cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(processQueueHandle));
            m_networkParams[EncodingNetworkId].hasJitLayout = true;
        }
        if (!m_networkParams[MlpNetworkId].hasJitLayout) {
            CHECK_STATUS_RETURN(memory[bindedTensorIndex[MlpNetworkId]]->setFromHost(
                m_networkParams[MlpNetworkId].tensor.buffer.data(),
                m_networkParams[MlpNetworkId].tensor.buffer.size(),
                processQueueHandle,
                logger));
            m_networkPtr->set_params(nullptr,
                                     reinterpret_cast<TNetworkParams*>(memory[bindedTensorIndex[MlpNetworkId]]->data()),
                                     nullptr);
            m_networkPtr->convert_params_to_jit_layout(reinterpret_cast<cudaStream_t>(processQueueHandle), true);
            m_networkPtr->set_params(nullptr, nullptr, nullptr);
            // copy the converted parameters to the host
            CUDA_CHECK_RETURN(cudaMemcpyAsync(m_networkParams[MlpNetworkId].tensor.buffer.data(),
                                              memory[bindedTensorIndex[MlpNetworkId]]->data(),
                                              m_networkParams[MlpNetworkId].tensor.buffer.size(),
                                              cudaMemcpyDeviceToHost,
                                              reinterpret_cast<cudaStream_t>(processQueueHandle)),
                              logger);
            cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(processQueueHandle));
            m_networkParams[MlpNetworkId].hasJitLayout = true;
        }

        for (size_t i = 0; i < NumNetworkTensors; ++i) {
            CHECK_STATUS_RETURN(memory[bindedTensorIndex[i]]->setFromHost(m_networkParams[i].tensor.buffer.data(),
                                                                          m_networkParams[i].tensor.buffer.size(),
                                                                          processQueueHandle,
                                                                          logger));
        }

        return status;
    }
};

} // namespace nrend
