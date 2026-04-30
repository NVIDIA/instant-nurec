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

#include <nrend/loggerParameters.h>

#include <cstdint>
#include <iostream>

#ifndef NREND_ABI
#ifdef _WIN32
#define NREND_ABI __cdecl
#else
#define NREND_ABI
#endif
#endif

namespace nrend {

class Logger {
public:
    static void NREND_LOGGER_CB defaultLogCallback(uint8_t level, const char* msg, void* data) {
        std::ostream& stream = (level > LoggerParameters::Error) ? std::cout : std::cerr;
        stream << "[NRend][" << LoggerParameters::levelToString(level) << "] ::: " << msg << std::flush << std::endl;
    }

    Logger(const LoggerParameters& params)
        : m_maxLogLevel(params.maximumLevel), m_logFn(params.callback), m_logFnData(params.callbackData), m_logDeviceLaunchFn(params.deviceLauncCallback), m_logDeviceLaunchFnData(params.deviceLaunchCallbackData) {
    }
    Logger(uint8_t maxLogLevel                                            = LoggerParameters::Error,
           LoggerParameters::Callback logFn                               = defaultLogCallback,
           void* logFnData                                                = nullptr,
           LoggerParameters::DeviceLaunchCallback logDeviceLaunchCallback = nullptr,
           void* logDeviceLaunchCallbackData                              = nullptr)
        : m_maxLogLevel(maxLogLevel), m_logFn(logFn), m_logFnData(logFnData), m_logDeviceLaunchFn(logDeviceLaunchCallback), m_logDeviceLaunchFnData(logDeviceLaunchCallbackData) {
    }
    virtual ~Logger() = default;

    virtual inline void log(uint8_t level, const char* message) const {
        if (level <= m_maxLogLevel) {
            m_logFn(level, message, m_logFnData);
        }
    }

    virtual inline void logDeviceLaunch(bool start, const char* tag, int deviceIndex, uint64_t deviceQueue) const {
        if (m_logDeviceLaunchFn) {
            m_logDeviceLaunchFn(start, tag, deviceIndex, deviceQueue, m_logDeviceLaunchFnData);
        }
    }

    inline uint8_t level() const {
        return m_maxLogLevel;
    }

    inline bool deviceLaunchEnabled() const {
        return m_logDeviceLaunchFn;
    }

private:
    uint8_t m_maxLogLevel;

    LoggerParameters::Callback m_logFn;
    void* m_logFnData;
    LoggerParameters::DeviceLaunchCallback m_logDeviceLaunchFn;
    void* m_logDeviceLaunchFnData;
};

#define LOG_FMT(logger, level, fmt, ...)                 \
    do {                                                 \
        constexpr uint16_t maxMsgLength = 1024;          \
        char msg[maxMsgLength];                          \
        snprintf(msg, maxMsgLength, fmt, ##__VA_ARGS__); \
        logger.log(level, msg);                          \
    } while (0)

#define LOG_DEBUG(logger, fmt, ...) LOG_FMT(logger, nrend::LoggerParameters::Debug, fmt, ##__VA_ARGS__)
#define LOG_INFO(logger, fmt, ...) LOG_FMT(logger, nrend::LoggerParameters::Info, fmt, ##__VA_ARGS__)
#define LOG_WARN(logger, fmt, ...) LOG_FMT(logger, nrend::LoggerParameters::Warning, fmt, ##__VA_ARGS__)
#define LOG_ERROR(logger, fmt, ...) LOG_FMT(logger, nrend::LoggerParameters::Error, fmt, ##__VA_ARGS__)
#define LOG_FATAL(logger, fmt, ...) LOG_FMT(logger, nrend::LoggerParameters::Fatal, fmt, ##__VA_ARGS__)

#define PROFILE_DEVICE_START(logger, tag, deviceIndex, deviceQueue) \
    logger.logDeviceLaunch(true, tag, deviceIndex, deviceQueue)
#define PROFILE_DEVICE_END(logger, tag, deviceIndex, deviceQueue) \
    logger.logDeviceLaunch(false, tag, deviceIndex, deviceQueue)

} // namespace nrend
