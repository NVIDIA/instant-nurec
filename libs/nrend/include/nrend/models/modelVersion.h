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

#include <json/json.hpp>

namespace nrend {

class ModelVersion final {
public:
    struct Number {
        int major = 0;
        int minor = 0;
        int patch = 0;

        inline bool operator>=(const Number& lhs) const {
            return (major > lhs.major) ||
                   ((major == lhs.major) && ((minor > lhs.minor) || ((minor == lhs.minor) && (patch >= lhs.patch))));
        }

        inline bool operator>(const Number& lhs) const {
            return (major > lhs.major) ||
                   ((major == lhs.major) && ((minor > lhs.minor) || ((minor == lhs.minor) && (patch > lhs.patch))));
        }

        inline bool operator<=(const Number& lhs) const {
            return !(*this > lhs);
        }

        inline bool operator<(const Number& lhs) const {
            return !(*this >= lhs);
        }
    };

    ModelVersion() = default;
    ModelVersion(const nlohmann::json& config) {
        if (config.contains("nre_data")) {
            if (config["nre_data"].contains("version") &&
                // if version field is 'null', no actual version number is available
                // (e.g., when executed in sandboxed unit tests), and we default to default-initialized 0.0.0
                config["nre_data"]["version"].is_string()) {
                std::stringstream iss{static_cast<std::string>(config["nre_data"]["version"])};
                std::string token;
                std::getline(iss, token, '.');
                m_number.major = std::stoi(token);
                std::getline(iss, token, '.');
                m_number.minor = std::stoi(token);
                std::getline(iss, token, '.');
                m_number.patch = std::stoi(token);
            }
            if (config["nre_data"].contains("model")) {
                m_model = config["nre_data"]["model"];
            } else {
                m_model = "ingp_nre";
            }
            if (config["nre_data"].contains("config") && config["nre_data"]["config"].contains("name")) {
                m_modelInstance = config["nre_data"]["config"]["name"];
            }
        }
    }
    virtual ~ModelVersion() = default;

    inline bool is(const std::string& model) const {
        return (m_model == model);
    }

    inline bool isInstance(const std::string& modelInstance) const {
        return (m_modelInstance == modelInstance);
    }

    inline std::string str() const {
        return std::to_string(m_number.major) + "." + std::to_string(m_number.minor) + "." + std::to_string(m_number.patch);
    }

    inline std::string extStr() const {
        return m_model + "_" + m_modelInstance + "@" + std::to_string(m_number.major) + "." + std::to_string(m_number.minor) + "." + std::to_string(m_number.patch);
    }

    inline const std::string& model() const {
        return m_model;
    }

    inline const std::string& modelInstance() const {
        return m_modelInstance;
    }

    inline const Number& number() const {
        return m_number;
    }

private:
    std::string m_model = "ingp"; ///< type of the model
    std::string m_modelInstance;  ///< instance of the model
    Number m_number;
};

} // namespace nrend
