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

#ifndef NREND_LOGGER_CB
#ifdef _WIN32
#define NREND_LOGGER_CB __cdecl
#else
#define NREND_LOGGER_CB
#endif
#endif

#include <cstdint>

namespace nrend {

struct LoggerParameters {

    enum : uint8_t {
        Fatal,
        Error,
        Warning,
        Info,
        Debug,
        DebugSyncDevice,
        NumLevels
    };

    typedef void(NREND_LOGGER_CB* Callback)(uint8_t level, const char* msg, void* data);
    typedef void(NREND_LOGGER_CB* DeviceLaunchCallback)(bool start,
                                                        const char* tag,
                                                        int deviceIndex,
                                                        uint64_t deviceQueue,
                                                        void* data);

    uint8_t maximumLevel                     = Error;
    Callback callback                        = nullptr;
    void* callbackData                       = nullptr;
    DeviceLaunchCallback deviceLauncCallback = nullptr;
    void* deviceLaunchCallbackData           = nullptr;

    static const char* levelToString(uint8_t level) {
        switch (level) {
        case Fatal:
            return "FATAL";
        case Error:
            return "ERROR";
        case Warning:
            return "WARNING";
        case Info:
            return "INFO";
        case Debug:
            return "DEBUG";
        case DebugSyncDevice:
            return "DEBUG-SYNC-DEVICE";
        }
        return "UNDEFINED";
    }
};

} // namespace nrend