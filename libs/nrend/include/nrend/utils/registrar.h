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

#include <nrend/utils/status.h>

#include <json/json.hpp>

namespace nrend {
template <typename IRegistrarImp, typename IRegistered, typename... InstantiatorFunctionArgs>
struct IRegistrar {

    static inline IRegistered* createFromJSON(const std::string& namePrefix, const nlohmann::json& config, const Logger& logger, InstantiatorFunctionArgs&&... args) {

        std::string name   = namePrefix;
        const bool hasName = config.contains("name");
        if (!hasName && name.empty()) {
            SET_ERROR(logger, ErrorCode::BadInput, "Configuration does not contains a registered instance name.");
            return nullptr;
        } else if (hasName) {
            name += static_cast<std::string>(config["name"]);
        }
        LOG_INFO(logger, "Creating registered instance <%s>", name.c_str());

        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::tolower(c); });

        auto instanciatorIt = IRegistrarImp::s_registeredInstantiators.find(name);
        if (instanciatorIt == IRegistrarImp::s_registeredInstantiators.end()) {
            LOG_WARN(
                logger, "No registered instance named %s found. (%d instances registered)", name.c_str(), static_cast<int>(IRegistrarImp::s_registeredInstantiators.size()));
            return nullptr;
        }

        IRegistered* registeredPtr = nullptr;
        try {
            registeredPtr = instanciatorIt->second(config, logger, std::forward<InstantiatorFunctionArgs>(args)...);
        } catch (const std::exception& e) {
            SET_ERROR(logger, ErrorCode::BadInput, "Registered %s : exception while parsing the JSON : %s.", name.c_str(), e.what());
            return nullptr;
        }

        return registeredPtr;
    }

    static inline IRegistered* createFromJSON(const nlohmann::json& config, const Logger& logger, InstantiatorFunctionArgs&&... args) {
        return createFromJSON("", config, logger, std::forward<InstantiatorFunctionArgs>(args)...);
    }

    template <typename T>
    static inline IRegistered* createDefaultByName(const std::string& name, const Logger& logger, T testValidity, InstantiatorFunctionArgs&&... args) {
        const nlohmann::json emptyConfig;
        auto instantiorIt = IRegistrarImp::s_registeredInstantiators.find(name);
        if (instantiorIt != IRegistrarImp::s_registeredInstantiators.end()) {
            try {
                auto registeredPtr = instantiorIt->second(emptyConfig, logger, std::forward<InstantiatorFunctionArgs>(args)...);
                if (testValidity(registeredPtr)) {
                    return registeredPtr;
                }
            } catch (...) {
                // No Error reporting, the renderer is invalid
            }
        }
        return nullptr;
    }

    template <typename T>
    static inline IRegistered* createFirstValid(const Logger& logger, T testValidity, InstantiatorFunctionArgs&&... args) {
        const nlohmann::json emptyConfig;
        for (const auto& instantior : IRegistrarImp::s_registeredInstantiators) {
            auto registeredPtr = instantior.second(emptyConfig, logger, std::forward<InstantiatorFunctionArgs>(args)...);
            if (testValidity(registeredPtr)) {
                LOG_INFO(logger, "Creating registered instance <%s>", instantior.first.c_str());
                return registeredPtr;
            }
        }
        return nullptr;
    }

    virtual ~IRegistrar() = default;

public:
    using InstantiatorFunc        = IRegistered* (*)(const nlohmann::json&, const Logger&,
                                              InstantiatorFunctionArgs&&... args);
    using RegisterInstantiatorMap = std::unordered_map<std::string, InstantiatorFunc>;

    template <typename TRegistered>
    static inline bool registerInstantiator(const std::string& prefix = "") {
        std::string key = prefix + std::string(TRegistered::name);
        std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return std::tolower(c); });

        return IRegistrarImp::s_registeredInstantiators
            .insert(std::make_pair<std::string, InstantiatorFunc>(std::string(key), [](const nlohmann::json& config, const Logger& logger, InstantiatorFunctionArgs&&... args) { return dynamic_cast<IRegistered*>(new TRegistered(config, logger, std::forward<InstantiatorFunctionArgs>(args)...)); }))
            .second;
    }
};

#define REGISTER_PREFIXED_IMPLEMENTATION_EXT(Prefix, RegistrarClass, RegisteredClass) \
    const static bool _nrendRegistrar##RegisteredClass##Registered_ = RegistrarClass::registerInstantiator<RegisteredClass>(Prefix);

#define REGISTER_IMPLEMENTATION_EXT(RegistrarClass, RegisteredClass) REGISTER_PREFIXED_IMPLEMENTATION_EXT("", RegistrarClass, RegisteredClass)

#define REGISTER_IMPLEMENTATION(RegisteredClass) REGISTER_IMPLEMENTATION_EXT(RegisteredClass, RegisteredClass)

#define REGISTER_PREFIXED_IMPLEMENTATION(Prefix, RegisteredClass) REGISTER_PREFIXED_IMPLEMENTATION_EXT(Prefix, RegisteredClass, RegisteredClass)

} // namespace nrend