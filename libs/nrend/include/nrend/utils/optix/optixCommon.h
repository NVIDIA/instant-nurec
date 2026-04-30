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

#include <optix.h>

#define OPTIX_SUCCEEDED(result) ((result) == OPTIX_SUCCESS)
#define OPTIX_FAILED(result) ((result) != OPTIX_SUCCESS)

#define OPTIX_CHECK(result, logger)                                                                                       \
    {                                                                                                                     \
        OptixResult _result = (result);                                                                                   \
        if (OPTIX_FAILED(_result)) {                                                                                      \
            LOG_ERROR(                                                                                                    \
                logger, "OPTIX error %d: %s - %s)", (_result), optixGetErrorName(_result), optixGetErrorString(_result)); \
        }                                                                                                                 \
    }

#define OPTIX_CHECK_RETURN(result, logger)                                                  \
    do {                                                                                    \
        OptixResult _result = (result);                                                     \
        if (OPTIX_FAILED(_result)) {                                                        \
            _SET_ERROR(logger, ErrorCode::Runtime, #result " failed: %d: %s - %s at %s:%d", \
                       (_result), optixGetErrorName(_result), optixGetErrorString(_result), \
                       __FILE__, __LINE__);                                                 \
            return ___status;                                                               \
        }                                                                                   \
    } while (0)

#define OPTIX_CHECK_LOG_RETURN(result, log, sizeofLog, logger)                  \
    do {                                                                        \
        OptixResult res = (result);                                             \
        if (OPTIX_FAILED(res) && (logger.level() >= LoggerParameters::Debug)) { \
            const size_t sizeofLogReturned = sizeofLog;                         \
            std::cout << "Optix call '" << #result << "' failed: " __FILE__ ":" \
                      << __LINE__ << ")\nLog:\n"                                \
                      << log                                                    \
                      << (sizeofLogReturned > sizeof(log) ? "<TRUNCATED>" : "") \
                      << "\n";                                                  \
        }                                                                       \
        sizeofLog = sizeof(log);                                                \
        OPTIX_CHECK_RETURN(res, logger);                                        \
    } while (0)
