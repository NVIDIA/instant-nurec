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

#include <tiny-cuda-nn/bounding_box.h>

namespace nrend {

class NREInstancesExtent final {

    uint16_t m_numInstances = 0;
    NREModel::TStateDictTensor m_instanceExpandedExtentTensor;

public:
    NREInstancesExtent() = default;
    NREInstancesExtent(const Logger& logger,
                       const nlohmann::json& stateDict,
                       const std::string& prefix) {

        // read the instance scene contractors
        NREModel::TStateDictTensor instanceExtentHalfTensor;
        instanceExtentHalfTensor.key = prefix + "aabb_extent";
        if (NREModel::readStateDictTensor(stateDict, instanceExtentHalfTensor)) {
            m_numInstances                       = instanceExtentHalfTensor.shape[0];
            m_instanceExpandedExtentTensor.key   = instanceExtentHalfTensor.key;
            m_instanceExpandedExtentTensor.shape = instanceExtentHalfTensor.shape;
            m_instanceExpandedExtentTensor.buffer.resize(m_numInstances * sizeof(tcnn::vec3));
            const tcnn::tvec<__half, 3>* instanceExtentHalfPtr = reinterpret_cast<tcnn::tvec<__half, 3>*>(instanceExtentHalfTensor.buffer.data());
            tcnn::vec3* instanceExtentPtr                      = reinterpret_cast<tcnn::vec3*>(m_instanceExpandedExtentTensor.buffer.data());
            for (uint16_t i = 0; i < m_numInstances; ++i) {
                instanceExtentPtr[i] = instanceExtentHalfPtr[i];
            }
        } else {
            LOG_INFO(logger, "NREInstancesExtent : missing instance extents buffer (%s) in the state_dict.", instanceExtentHalfTensor.key.c_str());
        }
    }
    virtual ~NREInstancesExtent() = default;

    inline uint16_t numInstances() const {
        return m_numInstances;
    }

    inline float maxExtent() const {
        float maxExtent                   = 1e-06f;
        const tcnn::vec3* expandedExtents = reinterpret_cast<const tcnn::vec3*>(m_instanceExpandedExtentTensor.buffer.data());
        for (uint16_t i = 0; i < m_numInstances; ++i) {
            maxExtent = std::max(maxExtent, std::max(expandedExtents[i].x, std::max(expandedExtents[i].y, expandedExtents[i].z)));
        }
        return maxExtent;
    }

    inline std::string sourceDefinition(const std::string& prefix) const {
        const std::string sourceCodeTemplate = R"(
            #include <nrend/kernels/cuda/models/nreInstancesExtent.cuh>

            struct {NREInstancesExtentClassAlias}Params
            {{
                static constexpr int NumInstances = {numInstances};
                static const float instanceExpandedExtents[3*{extentsArraySize}];
            }};
            const float {NREInstancesExtentClassAlias}Params::instanceExpandedExtents[3*{extentsArraySize}] = {instanceExtents};
            using {NREInstancesExtentClassAlias} = NREInstancesExtent<{NREInstancesExtentClassAlias}Params>;
        )";

        return fmt::format(sourceCodeTemplate,
                           fmt::arg("NREInstancesExtentClassAlias", prefix),
                           fmt::arg("numInstances", m_numInstances),
                           fmt::arg("extentsArraySize", std::max<uint16_t>(1, m_numInstances)), //< empty array not allowed
                           fmt::arg("instanceExtents", instanceExpandedExtentDefsStr()));
    }

private:
    // return inline code definition of an array containing the instances extent
    inline std::string instanceExpandedExtentDefsStr() const {

        if (m_numInstances <= 0) {
            return "{0.f,0.f,0.f}";
        }

        std::string defStr                = "{";
        const tcnn::vec3* expandedExtents = reinterpret_cast<const tcnn::vec3*>(m_instanceExpandedExtentTensor.buffer.data());
        for (uint16_t i = 0; i < m_numInstances; ++i) {
            defStr += "\n   " + std::to_string(expandedExtents[i].x) + "," + std::to_string(expandedExtents[i].y) + "," + std::to_string(expandedExtents[i].z) + ",";
        }
        defStr.pop_back(); //< remove last coma
        defStr += "}";

        return defStr;
    }
};

} // namespace nrend