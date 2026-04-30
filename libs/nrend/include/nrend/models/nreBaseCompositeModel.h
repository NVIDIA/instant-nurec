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

#include <cstring>

namespace nrend {

class NREBaseCompositeModel : public NREModel {

protected:
    float m_transmittanceThreshold       = 0.0001f;
    int m_maxPrimitiveInstancesBatchSize = 16; //< size of the batch of instances to step at a time

    std::vector<std::string> m_backgroundPrimitives;
    struct PrimitiveInstance {
        std::string trackInstanceStrUId;

        // Kernel packs those into 16b (6b : primitiveId, 10b instanceId)
        uint8_t primitiveId;
        uint16_t instanceId;
        std::vector<uint16_t> trackMappingIndex; ///< a primitive instance may be mapped several times

        static constexpr uint8_t InvalidPrimitiveId           = 0x3F;
        static constexpr uint16_t InvalidInstanceId           = 0x03FF;
        static constexpr uint16_t InvalidPackedIdx            = 0xFFFF;
        static constexpr uint8_t MaxNumPrimitives             = InvalidPrimitiveId - 1;
        static constexpr uint16_t MaxNumInstancesPerPrimitive = InvalidInstanceId - 1;
        static constexpr uint32_t MaxNumPrimitiveInstances    = MaxNumPrimitives * MaxNumInstancesPerPrimitive;

        inline uint16_t packedIdx(uint8_t primitiveIdOffset) const {
            return (static_cast<uint16_t>(primitiveId + primitiveIdOffset) << 10) | (instanceId & InvalidInstanceId);
        }
    };
    std::vector<PrimitiveInstance> m_primitivesInstancesMap;

    struct Primitive {
        const std::string key;
        uint16_t numInstances;
        uint16_t numActiveInstances;
    };
    std::vector<Primitive> m_primitives;

public:
    NREBaseCompositeModel(const nlohmann::json& config,
                          const Logger& logger,
                          const nlohmann::json& stateDict,
                          const std::string& prefix,
                          const std::vector<const char*>& submodelStr,
                          const std::vector<const char*>& submodelPrefixes,
                          const char* primitivesKey,
                          const char* primitivesPrefix)
        // TODO : difference between submodel creation and model creation (may use different prefix / dict)
        : NREModel(config,
                   logger,
                   stateDict,
                   prefix,
                   {}) {

        initializeSubModels(config,
                            stateDict,
                            prefix,
                            std::vector<std::string>(submodelStr.begin(), submodelStr.end()),
                            std::vector<std::string>(submodelPrefixes.begin(), submodelPrefixes.end()),
                            logger);

        m_transmittanceThreshold = config.value("transmittance_threshold", 0.0001f);

        if (config.contains(primitivesKey)) {

            const std::string extraStateKey     = prefix + "_extra_state";
            const bool stateDictWithExtraState  = stateDict.contains(extraStateKey);
            constexpr char objTrackIdsKey[]     = "obj_track_ids";
            const bool stateDictWithObjTrackIds = stateDictWithExtraState && stateDict[extraStateKey].contains(objTrackIdsKey);
            if (!stateDictWithExtraState) {
                LOG_WARN(logger, "NREBaseComposite : missing extra state <%s> in the state_dict.",
                         extraStateKey.c_str());
            } else if (!stateDictWithObjTrackIds) {
                LOG_WARN(logger, "NREBaseComposite : missing object track ids <%s> in the state_dict.",
                         objTrackIdsKey);
            }

            m_backgroundPrimitives.reserve(2);
            m_primitives.reserve(8);
            m_primitivesInstancesMap.reserve(64);
            for (auto& [key, val] : config[primitivesKey].items()) {

                if (m_primitives.size() + m_backgroundPrimitives.size() == PrimitiveInstance::MaxNumPrimitives) {
                    LOG_ERROR(logger, "NREBaseComposite : maximum number of primitives %d has been overflown with for primitive <%s>.",
                              PrimitiveInstance::MaxNumPrimitives, key.c_str());
                    break;
                }

                LOG_DEBUG(logger, "NREBaseComposite : processing primitive <%s>.", key.c_str());

                // update the primitiveInstancesMap
                bool backgroundPrimitive          = true;
                uint16_t primitiveLocalInstanceId = 0;
                if (stateDictWithObjTrackIds && stateDict[extraStateKey][objTrackIdsKey].contains(key) && (m_primitivesInstancesMap.size() < PrimitiveInstance::MaxNumPrimitiveInstances)) {

                    for (auto& [key, val] : stateDict[extraStateKey][objTrackIdsKey][key].items()) {

                        backgroundPrimitive = false;

                        if (primitiveLocalInstanceId == PrimitiveInstance::MaxNumInstancesPerPrimitive) {
                            LOG_ERROR(logger, "NREBaseComposite : maximum number of instances %d has been reached for primitive <%s>.",
                                      primitiveLocalInstanceId, key.c_str());
                            break;
                        }

                        if (m_primitivesInstancesMap.size() == PrimitiveInstance::MaxNumPrimitiveInstances) {
                            LOG_ERROR(logger, "NREBaseComposite : maximum number of primitive instances %d has been reached while processing primitive <%s>.",
                                      static_cast<int>(m_primitivesInstancesMap.size()), key.c_str());
                            break;
                        }

                        m_primitivesInstancesMap.push_back(
                            PrimitiveInstance{static_cast<std::string>(val),
                                              static_cast<uint8_t>(m_primitives.size()),
                                              primitiveLocalInstanceId,
                                              std::vector<uint16_t>{}});

                        LOG_DEBUG(logger, "NREBaseComposite : primitive track <%s> instance %d.", key.c_str(), primitiveLocalInstanceId);

                        primitiveLocalInstanceId += 1;
                    }
                }

                if (backgroundPrimitive) {
                    m_backgroundPrimitives.push_back(key);
                } else {
                    // add the primitive : has to be done after the primitiveInstancesMap update
                    // primitive instances are activated through initializeTrackInstances_
                    m_primitives.push_back(Primitive{key, primitiveLocalInstanceId, 0});
                }
            }

            std::vector<std::string> primitivesKeyStr;
            primitivesKeyStr.reserve(m_backgroundPrimitives.size() + m_primitives.size());
            for (const auto& key : m_backgroundPrimitives) {
                primitivesKeyStr.push_back(key);
            }
            for (const auto& primitive : m_primitives) {
                primitivesKeyStr.push_back(primitive.key);
            }

            initializeSubModels(
                config[primitivesKey],
                stateDict,
                prefix + primitivesPrefix + ".",
                primitivesKeyStr,
                logger);

        } else {
            LOG_WARN(logger, "NREBaseComposite: no primitive defined in the configuration.");
        }
    }

    virtual ~NREBaseCompositeModel() = default;

protected:
    inline std::string getVariadicActivePrimitiveTypesStr(const std::string& prefix) const {
        // generate the list of active primitive types
        std::string variadicPrimitiveTypesStr;
        for (const auto& primitive : m_backgroundPrimitives) {
            variadicPrimitiveTypesStr += cudaCallPrefix((prefix + "." + primitive).c_str()) + ",";
        }
        for (const auto& primitive : m_primitives) {
            if (primitive.numActiveInstances) {
                variadicPrimitiveTypesStr += cudaCallPrefix((prefix + "." + primitive.key).c_str()) + ",";
            }
        }
        if (!variadicPrimitiveTypesStr.empty()) {
            variadicPrimitiveTypesStr.pop_back();
        }
        return variadicPrimitiveTypesStr;
    }

private:
    virtual void initializeTrackInstances_(const TrackInstancesUIdsSpan& trackInstancesStrUIds, const Logger& logger) override {

        // reset the activation state
        for (auto& primitive : m_primitives) {
            primitive.numActiveInstances = 0;
        }
        for (auto& primitiveInstance : m_primitivesInstancesMap) {
            primitiveInstance.trackMappingIndex.clear();
        }

        // sort the primitiveInstancesMap wrt the object uid
        std::sort(m_primitivesInstancesMap.begin(), m_primitivesInstancesMap.end(), [](const PrimitiveInstance& lhs, const PrimitiveInstance& rhs) { return lhs.trackInstanceStrUId < rhs.trackInstanceStrUId; });

        // create the trackInstancesStrUId with trackMappingIndex map
        if (trackInstancesStrUIds.size > PrimitiveInstance::MaxNumPrimitiveInstances) {
            LOG_ERROR(logger, "NREBaseComposite: input object uids list contains more than the maximum supported number (%d/%d).",
                      static_cast<int>(trackInstancesStrUIds.size), static_cast<int>(PrimitiveInstance::MaxNumPrimitiveInstances));
        }
        const uint16_t numValidTrackInstances = std::min<uint16_t>(trackInstancesStrUIds.size, PrimitiveInstance::MaxNumPrimitiveInstances);
        struct TrackInstanceUIdWithMappingIndex {
            const char* cStrUId;
            uint16_t mappingIndex;
        };
        std::vector<TrackInstanceUIdWithMappingIndex> trackInstanceUIdMapping(numValidTrackInstances);
        for (uint16_t i = 0; i < numValidTrackInstances; ++i) {
            trackInstanceUIdMapping[i] = TrackInstanceUIdWithMappingIndex{trackInstancesStrUIds.data[i], i};
        }

        // sort the mapping according the the string UIds / mapping index
        std::sort(trackInstanceUIdMapping.begin(), trackInstanceUIdMapping.end(), [](const TrackInstanceUIdWithMappingIndex& lhs, const TrackInstanceUIdWithMappingIndex& rhs) {
            const int strCmpRes = std::strcmp(lhs.cStrUId, rhs.cStrUId);
            return (strCmpRes == 0) ? (lhs.mappingIndex < rhs.mappingIndex) : (strCmpRes < 0);
        });

        // mapping the primitive instances map to the object uid (using the fact that both array are sorted according to the uid)
        for (uint16_t i = 0, j = 0; (i < m_primitivesInstancesMap.size()) && (j < trackInstanceUIdMapping.size());) {
            const int strCmpRes = std::strcmp(m_primitivesInstancesMap[i].trackInstanceStrUId.c_str(), trackInstanceUIdMapping[j].cStrUId);
            if (strCmpRes == 0) {
                m_primitivesInstancesMap[i].trackMappingIndex.push_back(trackInstanceUIdMapping[j].mappingIndex);
                m_primitives[m_primitivesInstancesMap[i].primitiveId].numActiveInstances++;
                // input mapping assigned : increment it (NB : we may have a primitive instance mapped to multiple inputs)
                j++; //< increment
            } else if (strCmpRes < 0) {
                // no more input are available for the current primitive instance
                i++;
            } else {
                // no input for the current primitive instance
                j++;
            }
        }
    }
};
} // namespace nrend