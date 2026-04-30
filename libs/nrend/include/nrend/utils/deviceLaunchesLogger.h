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

#include <nrend/utils/logger.h>

#include <vector>

namespace nrend {

class DeviceLaunchesLogger {

    const Logger& m_logger;
    const int m_deviceIndex;
    const uint64_t m_deviceQueue;

    std::vector<const char*> m_profilingTags;

    inline bool enabled() const { return m_logger.deviceLaunchEnabled(); }

public:
    DeviceLaunchesLogger(const Logger& logger, int deviceIndex, uint64_t deviceQueue)
        : m_logger(logger), m_deviceIndex(deviceIndex), m_deviceQueue(deviceQueue) {
        if (enabled()) {
            m_profilingTags.reserve(8);
        }
    }
    ~DeviceLaunchesLogger() {
        for (size_t i = m_profilingTags.size(); i > 0; --i) {
            PROFILE_DEVICE_END(m_logger, m_profilingTags[i - 1], m_deviceIndex, m_deviceQueue);
        }
    }

    inline void push(const char* tag) {
        if (enabled()) {
            m_profilingTags.push_back(tag);
            PROFILE_DEVICE_START(m_logger, tag, m_deviceIndex, m_deviceQueue);
        }
    };

    inline void pop() {
        if (enabled()) {
            if (m_profilingTags.empty()) {
                LOG_ERROR(m_logger, "DeviceLaunchesLogger : cannot pop : no pushed launch");
            } else {
                PROFILE_DEVICE_END(m_logger, m_profilingTags.back(), m_deviceIndex, m_deviceQueue);
                m_profilingTags.pop_back();
            }
        }
    };

    inline void pop(const char* tag) {
        if (enabled()) {
            PROFILE_DEVICE_END(m_logger, tag, m_deviceIndex, m_deviceQueue);
            if (strcmp(m_profilingTags.back(), tag)) {
                LOG_ERROR(m_logger, "LogProfiler wrong tag : %s / %s.", m_profilingTags.back(), tag);
            } else {
                m_profilingTags.pop_back();
            }
        }
    };

    class ScopePush final {
        DeviceLaunchesLogger& m_logger;

    public:
        ScopePush(DeviceLaunchesLogger& logger, const char* tag)
            : m_logger(logger) {
            m_logger.push(tag);
        }
        ~ScopePush() {
            m_logger.pop();
        }
    };
};
} // namespace nrend