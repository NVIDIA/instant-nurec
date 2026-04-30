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

class NRESkipAppearanceEmbedding : public NREModel {
public:
    static constexpr char name[] = "skip-appearance";

    NRESkipAppearanceEmbedding(const nlohmann::json& config,
                               const Logger& logger,
                               const nlohmann::json& stateDict,
                               const std::string& prefix)
        : NREModel(config, logger, stateDict, prefix, {}) {
    }

    virtual ~NRESkipAppearanceEmbedding() = default;

protected:
    virtual Status registerKernelResources_(
        const KernelMemoryBindings&,
        const KernelSourceCodeTable& sourceCodeTable,
        KernelResourcesProvider::KernelOpts,
        const Logger&) const override {

        sourceCodeTable.registerKernel(
            KernelSourceCodeTable::Cuda,
            fmt::format(R"(
                #include <nrend/kernels/cuda/models/nreAppearanceEmbedding.cuh>
                
                using {NREAppearanceEmbeddingClassAlias} = NRESkipAppearanceEmbedding;
            )",
                        fmt::arg("NREAppearanceEmbeddingClassAlias", cudaCallPrefix())));

        return Status();
    }
};

class NREDefaultAppearanceEmbedding : public NRESkipAppearanceEmbedding {
public:
    static constexpr char name[] = "appearance_embedding";

    NREDefaultAppearanceEmbedding(const nlohmann::json& config,
                                  const Logger& logger,
                                  const nlohmann::json& stateDict,
                                  const std::string& prefix)
        : NRESkipAppearanceEmbedding(config, logger, stateDict, prefix) {
    }
    virtual ~NREDefaultAppearanceEmbedding() = default;
};

class NREGloAppearanceEmbedding : public NREModel {
    int m_cameraEmbeddingDim;
    int m_cameraNumEmbeddings;
    int m_frameEmbeddingDim;
    int m_frameNumEmbeddings;
    enum class LatentMode {
        ZeroLatent,
        MeanLatent,
        FirstLatent,
        PerElementLatent,
        UndefinedLatent
    };
    LatentMode m_cameraLatentMode;
    LatentMode m_frameLatentMode;

    static inline LatentMode strToLatentMode(const std::string& mode) {
        if (mode == "zeros-latent") {
            return LatentMode::ZeroLatent;
        } else if (mode == "mean-latent") {
            return LatentMode::MeanLatent;
        } else if ((mode == "first-frame-latent") || (mode == "first-camera-latent")) {
            return LatentMode::FirstLatent;
        } else if ((mode == "per-camera-latent") || (mode == "per-frame-latent")) {
            return LatentMode::PerElementLatent;
        }
        return LatentMode::UndefinedLatent;
    }

    std::vector<char> m_cameraFrameEmbeddingBuffer;
    std::vector<char> m_cameraEmbeddingBuffer;
    std::vector<char> m_frameEmbeddingBuffer;

public:
    static constexpr char name[] = "glo-embedding";

    NREGloAppearanceEmbedding(const nlohmann::json& config,
                              const Logger& logger,
                              const nlohmann::json& stateDict,
                              const std::string& prefix)
        : NREModel(config, logger, stateDict, prefix, {}) {
        m_cameraEmbeddingDim  = config.value("camera_embedding_dim", 0);
        m_cameraNumEmbeddings = 1;
        m_frameEmbeddingDim   = config.value("frame_embedding_dim", 16);
        m_frameNumEmbeddings  = 1;
        if (config.value("embedding_dim", 16) != m_frameEmbeddingDim + m_cameraEmbeddingDim) {
            LOG_WARN(logger,
                     "NREGloAppearanceEmbedding : invalid embedding dimension [%d / (%d + %d)].",
                     config.value("embedding_dim", 16),
                     m_frameEmbeddingDim,
                     m_cameraEmbeddingDim);
        }
        m_cameraLatentMode = strToLatentMode(config.value("eval_mode_cam", "zeros-latent"));
        m_frameLatentMode  = strToLatentMode(config.value("eval_mode_frame", "zeros-latent"));

        TStateDictTensor cameraEmbeddingParamsTensor;
        cameraEmbeddingParamsTensor.key = prefix + "camera_embedding.weight";
        if (embeddingNeedBuffer(m_cameraEmbeddingDim, m_cameraLatentMode)) {
            if (!readStateDictTensor(stateDict, cameraEmbeddingParamsTensor)) {
                LOG_ERROR(logger,
                          "NREGloAppearanceEmbedding : missing valid camera embedding parameters tensor <%s> [%d] in the state_dict.",
                          cameraEmbeddingParamsTensor.key.c_str(),
                          static_cast<int>(cameraEmbeddingParamsTensor.buffer.size() / sizeof(__half)));
            } else {
                const size_t numParams = fetchEmbeddingBuffer(mergedLatent() ? m_cameraFrameEmbeddingBuffer : m_cameraEmbeddingBuffer,
                                                              m_cameraEmbeddingDim, m_cameraLatentMode, cameraEmbeddingParamsTensor);
                m_cameraNumEmbeddings = (m_cameraLatentMode == LatentMode::PerElementLatent) ? numParams / m_cameraEmbeddingDim : 1;
            }
        }

        TStateDictTensor frameEmbeddingParamsTensor;
        frameEmbeddingParamsTensor.key = prefix + "frame_embedding.weight";
        if (embeddingNeedBuffer(m_frameEmbeddingDim, m_frameLatentMode)) {
            if (!readStateDictTensor(stateDict, frameEmbeddingParamsTensor)) {
                LOG_ERROR(logger,
                          "NREGloAppearanceEmbedding : missing valid frame embedding parameters tensor <%s> [%d] in the state_dict.",
                          frameEmbeddingParamsTensor.key.c_str(),
                          static_cast<int>(frameEmbeddingParamsTensor.buffer.size() / sizeof(__half)));
            } else {
                const size_t numParams = fetchEmbeddingBuffer(mergedLatent() ? m_cameraFrameEmbeddingBuffer : m_frameEmbeddingBuffer,
                                                              m_frameEmbeddingDim, m_frameLatentMode, frameEmbeddingParamsTensor);
                m_frameNumEmbeddings = (m_frameLatentMode == LatentMode::PerElementLatent) ? numParams / m_frameEmbeddingDim : 1;
            }
        }
    }

    virtual ~NREGloAppearanceEmbedding() = default;

protected:
    static inline size_t fetchEmbeddingBuffer(std::vector<char>& buffer, int dim, LatentMode mode, const TStateDictTensor& tensor) {

        const __half* tensorHalfBuffer = reinterpret_cast<const __half*>(tensor.buffer.data());

        const size_t bufferSz = buffer.size();
        const size_t latentSz = dim * sizeof(__half);

        if (mode == LatentMode::PerElementLatent) {

            buffer.resize(bufferSz + tensor.buffer.size());
            std::memcpy(&buffer[bufferSz], tensor.buffer.data(), tensor.buffer.size());

        } else if ((mode == LatentMode::FirstLatent) || ((mode == LatentMode::MeanLatent) && ((tensor.shape.size() < 2) || tensor.shape[0] < 2))) {

            buffer.resize(bufferSz + latentSz);
            std::memcpy(&buffer[bufferSz], tensor.buffer.data(), latentSz);

        } else if (mode == LatentMode::MeanLatent) {

            std::vector<float> floatBuffer(dim, 0.0f);
            const size_t numLatentVectors       = tensor.shape[0];
            const float oneOverNumLatentVectors = 1.0f / numLatentVectors;
            for (size_t i = 0; i < numLatentVectors; ++i) {
                for (size_t j = 0; j < dim; ++j) {
                    floatBuffer[j] += __half2float(tensorHalfBuffer[i * dim + j]) * oneOverNumLatentVectors;
                }
            }

            buffer.resize(bufferSz + latentSz);
            __half* halfBuffer = reinterpret_cast<__half*>(&buffer[bufferSz]);
            for (size_t i = 0; i < dim; ++i) {
                halfBuffer[i] = __float2half(floatBuffer[i]);
            }
        }

        return buffer.size() / sizeof(__half);
    }

    static inline bool embeddingNeedBuffer(int dim, LatentMode mode) {
        return (dim > 0) && (mode != LatentMode::ZeroLatent);
    }

    inline bool skipEmbedding() const {
        return (m_cameraEmbeddingDim == 0) && (m_frameEmbeddingDim == 0);
    }

    inline bool zeroLatentMode() const {
        return (m_cameraLatentMode == LatentMode::ZeroLatent) && (m_frameLatentMode == LatentMode::ZeroLatent);
    }

    inline bool mergedLatent() const {
        return (m_cameraEmbeddingDim > 0) && (m_frameEmbeddingDim > 0) &&
               (m_cameraLatentMode != LatentMode::PerElementLatent) && (m_frameLatentMode != LatentMode::PerElementLatent) &&
               (m_cameraLatentMode != LatentMode::ZeroLatent) && (m_frameLatentMode != LatentMode::ZeroLatent);
    }

    inline std::string embeddingDefinition(int dim, LatentMode mode, int bufferIdx, int numEmbeddings = 1, int rayEmbeddingIdx = 0) const {
        if (dim <= 0) {
            return "NRESkipAppearanceEmbedding";
        } else if (mode == LatentMode::ZeroLatent) {
            return fmt::format("NREZeroAppearanceEmbedding<{EmbeddingDim}>", fmt::arg("EmbeddingDim", dim));
        } else if (mode == LatentMode::PerElementLatent) {
            return fmt::format(
                "NREIndexableCachedAppearanceEmbedding<{EmbeddingDim}, {NumEmbeddings}, {BufferIdx}, {RayEmbeddingIdx}>",
                fmt::arg("EmbeddingDim", dim),
                fmt::arg("NumEmbeddings", numEmbeddings),
                fmt::arg("BufferIdx", bufferIdx),
                fmt::arg("RayEmbeddingIdx", rayEmbeddingIdx));
        } else {
            return fmt::format("NRECachedAppearanceEmbedding<{EmbeddingDim}, {BufferIdx}>", fmt::arg("EmbeddingDim", dim), fmt::arg("BufferIdx", bufferIdx));
        }
    }

    virtual Status registerKernelResources_(
        const KernelMemoryBindings& memoryBindings,
        const KernelSourceCodeTable& sourceCodeTable,
        KernelResourcesProvider::KernelOpts,
        const Logger& logger) const override {

        Status status;

        if (!m_cameraFrameEmbeddingBuffer.empty()) {
            status = memoryBindings.registerMemory(
                KernelMemoryBindings::BindingsFlag::Parameters,
                m_callPrefix + "camera_frame_embedding.weight", ///< FIXME remove merged latent case
                KernelMemoryType::Buffer,
                logger);
            if (!status) {
                return status;
            }
        }

        if (!m_cameraEmbeddingBuffer.empty()) {
            status = memoryBindings.registerMemory(
                KernelMemoryBindings::BindingsFlag::Parameters,
                m_callPrefix + "camera_embedding.weight",
                KernelMemoryType::Buffer,
                logger);
            if (!status) {
                return status;
            }
        }

        if (!m_frameEmbeddingBuffer.empty()) {
            status = memoryBindings.registerMemory(
                KernelMemoryBindings::BindingsFlag::Parameters,
                m_callPrefix + "frame_embedding.weight",
                KernelMemoryType::Buffer,
                logger);
            if (!status) {
                return status;
            }
        }

        const int cameraFrameEmbeddingIndex = memoryBindings.registeredMemoryIndex(KernelMemoryBindings::Parameters, m_callPrefix + "camera_frame_embedding.weight");
        const int cameraEmbeddingIndex      = memoryBindings.registeredMemoryIndex(KernelMemoryBindings::Parameters, m_callPrefix + "camera_embedding.weight");
        const int frameEmbeddingIndex       = memoryBindings.registeredMemoryIndex(KernelMemoryBindings::Parameters, m_callPrefix + "frame_embedding.weight");

        const std::string NREAppearanceEmbeddingClassAliasDefinition =
            skipEmbedding() ? fmt::format("using {NREAppearanceEmbeddingClassAlias} = NRESkipAppearanceEmbedding;\n",
                                          fmt::arg("NREAppearanceEmbeddingClassAlias", cudaCallPrefix()))

            : zeroLatentMode() ? fmt::format("using {NREAppearanceEmbeddingClassAlias} = NREZeroAppearanceEmbedding<{EmbeddingDim}>;\n",
                                             fmt::arg("NREAppearanceEmbeddingClassAlias", cudaCallPrefix()),
                                             fmt::arg("EmbeddingDim", m_cameraEmbeddingDim + m_frameEmbeddingDim))

            : mergedLatent() ? fmt::format("using {NREAppearanceEmbeddingClassAlias} = NRECachedAppearanceEmbedding<{EmbeddingDim}, {BufferIdx}>;\n",
                                           fmt::arg("NREAppearanceEmbeddingClassAlias", cudaCallPrefix()),
                                           fmt::arg("EmbeddingDim", m_cameraEmbeddingDim + m_frameEmbeddingDim),
                                           fmt::arg("BufferIdx", cameraFrameEmbeddingIndex))

                             : fmt::format(
                                   "using {NREAppearanceEmbeddingClassAlias}_Camera = {TCameraEmbeddingDefinition};\n"
                                   "using {NREAppearanceEmbeddingClassAlias}_Frame = {TFrameEmbeddingDefinition};\n"
                                   "using {NREAppearanceEmbeddingClassAlias} = NREGloAppearanceEmbedding<{NREAppearanceEmbeddingClassAlias}_Camera, {NREAppearanceEmbeddingClassAlias}_Frame>;\n",
                                   fmt::arg("NREAppearanceEmbeddingClassAlias", cudaCallPrefix()),
                                   fmt::arg("TCameraEmbeddingDefinition", embeddingDefinition(m_cameraEmbeddingDim, m_cameraLatentMode, cameraEmbeddingIndex, m_cameraNumEmbeddings, 0)),
                                   fmt::arg("TFrameEmbeddingDefinition", embeddingDefinition(m_frameEmbeddingDim, m_frameLatentMode, frameEmbeddingIndex, m_frameNumEmbeddings, 1)));

        sourceCodeTable.registerKernel(
            KernelSourceCodeTable::Cuda,
            fmt::format(R"(
                #include <nrend/kernels/cuda/models/nreAppearanceEmbedding.cuh>
                
                {NREAppearanceEmbeddingClassAliasDefinition}
            )",
                        fmt::arg("NREAppearanceEmbeddingClassAliasDefinition", NREAppearanceEmbeddingClassAliasDefinition)));

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

        const int cameraFrameEmbeddingIndex = memoryBindings.registeredMemoryIndex(bindingsFlag, m_callPrefix + "camera_frame_embedding.weight");
        if ((cameraFrameEmbeddingIndex >= 0) && (cameraFrameEmbeddingIndex < memory.size()) && memory[cameraFrameEmbeddingIndex]) {
            status = memory[cameraFrameEmbeddingIndex]->setFromHost(m_cameraFrameEmbeddingBuffer.data(), m_cameraFrameEmbeddingBuffer.size(), processQueueHandle, logger);
        }

        const int cameraEmbeddingIndex = memoryBindings.registeredMemoryIndex(bindingsFlag, m_callPrefix + "camera_embedding.weight");
        if (status && (cameraEmbeddingIndex >= 0) && (cameraEmbeddingIndex < memory.size()) && memory[cameraEmbeddingIndex]) {
            status = memory[cameraEmbeddingIndex]->setFromHost(m_cameraEmbeddingBuffer.data(), m_cameraEmbeddingBuffer.size(), processQueueHandle, logger);
        }

        const int frameEmbeddingIndex = memoryBindings.registeredMemoryIndex(bindingsFlag, m_callPrefix + "frame_embedding.weight");
        if (status && (frameEmbeddingIndex >= 0) && (frameEmbeddingIndex < memory.size()) && memory[frameEmbeddingIndex]) {
            status = memory[frameEmbeddingIndex]->setFromHost(m_frameEmbeddingBuffer.data(), m_frameEmbeddingBuffer.size(), processQueueHandle, logger);
        }

        return status;
    }
};

} // namespace nrend