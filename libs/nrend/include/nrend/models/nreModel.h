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

#include <nrend/kernelResources/kernelResourcesProvider.h>
#include <nrend/modelParameters.h>
#include <nrend/tracksParameters.h>
#include <nrend/utils/registrar.h>

#include <fmt/core.h>
#include <json/json.hpp>
#include <nrend/utils/logger.h>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/vec.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <regex>
#include <string>
#include <vector>

namespace nrend {

class NREModel : public IRegistrar<NREModel, NREModel, const nlohmann::json&, const std::string&>, public KernelResourcesProvider {

public:
    struct TStateDictTensor {
        std::string key;
        std::vector<int> shape;
        std::vector<char> buffer;
    };

    static inline bool readStateDictTensor(const nlohmann::json& stateDict, TStateDictTensor& tensor) {
        if (stateDict.contains(tensor.key)) {
            const nlohmann::json::binary_t& bytes = stateDict[tensor.key];
            tensor.buffer.resize(bytes.size());
            std::memcpy(tensor.buffer.data(), bytes.data(), bytes.size());

            const std::string shapeKeyStr = tensor.key + ".shape";
            if (stateDict.contains(shapeKeyStr)) {
                const auto& size = stateDict[shapeKeyStr];
                tensor.shape.clear();
                tensor.shape.insert(tensor.shape.begin(), size.begin(), size.end());
            } else {
                tensor.shape = std::vector<int>{static_cast<int>(tensor.buffer.size())};
            }
            return true;
        }
        return false;
    }

    static inline bool readStateDictTensor(const nlohmann::json& stateDict, TStateDictTensor& tensor, size_t expectedBufferSize) {
        return readStateDictTensor(stateDict, tensor) && (tensor.buffer.size() == expectedBufferSize);
    }

    template <typename T>
    static inline bool checkStateDictTensorSize(const TStateDictTensor& tensor) {
        return tensor.buffer.size() == (std::accumulate(tensor.shape.begin(), tensor.shape.end(), 1, std::multiplies<int>()) * sizeof(T));
    }

protected:
    std::vector<std::unique_ptr<NREModel>> m_subModelPtr;
    const std::string m_callPrefix;

    static inline std::string parseParentPrefix(const std::string& prefix) {
        std::string parsedPrefix = prefix;
        // remove last dot
        parsedPrefix.pop_back();
        // find last dot
        const size_t lastDotPos = parsedPrefix.find_last_of('.');
        return parsedPrefix.substr(0, lastDotPos + 1);
    }

    inline void initializeSubModels(
        const nlohmann::json& config,
        const nlohmann::json& stateDict,
        const std::string& prefix,
        const std::vector<std::string>& submodelStr,
        const Logger& logger) {
        initializeSubModels(config, stateDict, prefix, submodelStr, submodelStr, logger);
    }

    inline void initializeSubModels(
        const nlohmann::json& config,
        const nlohmann::json& stateDict,
        const std::string& prefix,
        const std::vector<std::string>& submodelStr,
        const std::vector<std::string>& submodelPrefixes,
        const Logger& logger) {

        m_subModelPtr.reserve(m_subModelPtr.size() + submodelStr.size());
        for (size_t i = 0; i < submodelStr.size(); ++i) {
            NREModel* ptr = nullptr;
            if (config.contains(submodelStr[i])) {
                if (config[submodelStr[i]].contains("name")) {
                    ptr = createFromJSON(config[submodelStr[i]], logger, stateDict, prefix + submodelPrefixes[i] + ".");
                } else {
                    ptr = createFromJSON(submodelStr[i], config[submodelStr[i]], logger, stateDict, prefix + submodelPrefixes[i] + ".");
                }
            }
            // backward compatibility : create a default model from an empty config
            else {
                ptr = createFromJSON(submodelStr[i], nlohmann::json{}, logger, stateDict, prefix + submodelPrefixes[i] + ".");
            }
            m_subModelPtr.emplace_back(ptr);
        }
    }

public:
    Status registerKernelResources(
        const KernelMemoryBindings& memoryBindings,
        const KernelSourceCodeTable& sourceCodeTable,
        KernelOpts kernelOpts,
        const Logger& logger) const override {

        Status status;
        for (size_t i = 0; status && i < m_subModelPtr.size(); ++i) {
            if (m_subModelPtr[i]) {
                status = m_subModelPtr[i]->registerKernelResources(memoryBindings, sourceCodeTable, kernelOpts, logger);
            }
        }

        if (status) {
            status = registerKernelResources_(memoryBindings, sourceCodeTable, kernelOpts, logger);
        }

        return status;
    }

    Status processKernelMemory(
        const KernelMemoryBindings& memoryBindings,
        KernelMemoryBindings::BindingsFlag bindingsFlag,
        const std::vector<std::unique_ptr<KernelMemory>>& memory,
        ProcessMemoryFlag processFlag,
        uint64_t processQueueHandle,
        const Logger& logger) const override {

        Status status;
        for (size_t i = 0; status && i < m_subModelPtr.size(); ++i) {
            if (m_subModelPtr[i]) {
                status = m_subModelPtr[i]->processKernelMemory(memoryBindings, bindingsFlag, memory, processFlag, processQueueHandle, logger);
            }
        }

        if (status) {
            status = processKernelMemory_(memoryBindings, bindingsFlag, memory, processFlag, processQueueHandle, logger);
        }

        return status;
    }

public:
    NREModel(const nlohmann::json& config,
             const Logger& logger,
             const nlohmann::json& stateDict,
             const std::string& prefix,
             const std::vector<const char*>& submodelCStr)
        : m_callPrefix(prefix) {
        initializeSubModels(config,
                            stateDict,
                            prefix,
                            std::vector<std::string>(submodelCStr.begin(), submodelCStr.end()),
                            logger);
    }
    virtual ~NREModel() = default;

    inline void initializeTrackInstances(const TrackInstancesUIdsSpan& trackInstancesStrUIds, const Logger& logger) {
        for (size_t i = 0; i < m_subModelPtr.size(); ++i) {
            if (m_subModelPtr[i]) {
                m_subModelPtr[i]->initializeTrackInstances(trackInstancesStrUIds, logger);
            }
        }
        initializeTrackInstances_(trackInstancesStrUIds, logger);
    }

    /// @return call prefix, compatible with cuda code
    inline std::string cudaCallPrefix(const char* subModelPrefix = "") const {
        // add a trailing dot to the given subModelPrefix
        std::string subModelPrefixStr = subModelPrefix;
        if (!subModelPrefixStr.empty()) {
            subModelPrefixStr += ".";
        }
        // add root "model" to the prefix
        auto cudaCallPrefix = "model" + m_callPrefix + subModelPrefixStr;
        // duplicate every underscore
        cudaCallPrefix = std::regex_replace(cudaCallPrefix, std::regex("\\_"), "__");
        // replace . by underscore
        cudaCallPrefix = std::regex_replace(cudaCallPrefix, std::regex("\\."), "_");
        return cudaCallPrefix;
    }

    static RegisterInstantiatorMap s_registeredInstantiators;

    struct FeaturesLayout {
        // base features : mandatory features stored with the density (radiance)
        int baseFeaturesDim; ///< radiance : 3 (RGB) if model output radiance, 0 otherwise
        // extended features : optional features (eg : normals, raydrop, intensity, uncertainty, semantic_logits, dinov2_feats, etc)
        int extendedFeaturesDim;
        // sensor specific extended features
        int cameraExtendedFeaturesDim;
        int lidarExtendedFeaturesDim;
        // TODO : add extended features layout
    };
    virtual FeaturesLayout featuresLayout() const { return FeaturesLayout{3, 0, 0, 0}; }

private:
    virtual void initializeTrackInstances_(const TrackInstancesUIdsSpan& trackInstancesStrUIds, const Logger& logger) {
    }

    virtual Status registerKernelResources_(
        const KernelMemoryBindings&,
        const KernelSourceCodeTable&,
        KernelResourcesProvider::KernelOpts,
        const Logger&) const { return Status(); }

    virtual Status processKernelMemory_(
        const KernelMemoryBindings&,
        KernelMemoryBindings::BindingsFlag,
        const std::vector<std::unique_ptr<KernelMemory>>&,
        ProcessMemoryFlag,
        uint64_t /*processQueueHandle*/,
        const Logger&) const { return Status(); }
};

#define REGISTER_NREMODEL_IMPLEMENTATION(ModelClass) \
    const static bool _NREModel##ModelClass##Registered_ = NREModel::registerInstantiator<ModelClass>();

} // namespace nrend