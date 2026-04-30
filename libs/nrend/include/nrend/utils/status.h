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

#include <nrend/errorCodes.h>
#include <nrend/utils/logger.h>

namespace nrend {
class Status final {
public:
    inline static ErrorCode getLastError() {
        ErrorCode lastError = _lastError;
        _lastError          = ErrorCode::None;
        return lastError;
    };

    inline static const char* message(ErrorCode err) {
        return _errorMessage[static_cast<uint16_t>(err)];
    }

    Status(ErrorCode error = ErrorCode::None, bool setLast = true)
        : _error(error) {
        if ((error != ErrorCode::None) && setLast) {
            _lastError = error;
        }
    }

    inline bool success() const { return _error == ErrorCode::None; };

    inline operator bool() const { return success(); }

    inline operator ErrorCode() const { return _error; }

private:
    ErrorCode _error;

private:
    static constexpr char const* _errorMessage[static_cast<uint16_t>(ErrorCode::Num)] = {
        "Success",
        "Invalid resources",
        "Invalid inputs",
        "Out of memory",
        "Missing implementation"};
    static ErrorCode _lastError;
};

#define CHECK_STATUS_RETURN(status)   \
    do {                              \
        const auto __status = status; \
        if (!__status) {              \
            return __status;          \
        }                             \
    } while (0)

#define _SET_ERROR(logger, error, fmt, ...)         \
    static_assert(error != nrend::ErrorCode::None); \
    LOG_ERROR(logger, fmt, ##__VA_ARGS__);          \
    [[maybe_unused]] auto ___status = nrend::Status(error);

#define SET_ERROR(logger, error, fmt, ...)             \
    do {                                               \
        _SET_ERROR(logger, error, fmt, ##__VA_ARGS__); \
    } while (0)

#define RETURN_ERROR(logger, error, fmt, ...)          \
    do {                                               \
        _SET_ERROR(logger, error, fmt, ##__VA_ARGS__); \
        return ___status;                              \
    } while (0)

#define RETURN_ERROR_IF(condition, logger, error, fmt, ...)  \
    do {                                                     \
        if (condition) {                                     \
            RETURN_ERROR(logger, error, fmt, ##__VA_ARGS__); \
        }                                                    \
    } while (0)

} // namespace nrend
