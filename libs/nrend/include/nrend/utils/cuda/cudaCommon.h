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

#include <cuda_runtime.h>

#define CUDA_SUCCEEDED(result) ((result) == cudaSuccess)
#define CUDA_FAILED(result) ((result) != cudaSuccess)

#define CUDA_CHECK(logger, result)                                                                                     \
    {                                                                                                                  \
        cudaError_t _result = (result);                                                                                \
        if (CUDA_FAILED(_result)) {                                                                                    \
            LOG_ERROR(                                                                                                 \
                logger, "CUDA error %d: %s - %s)", (_result), cudaGetErrorName(_result), cudaGetErrorString(_result)); \
        }                                                                                                              \
    }

#define CUDA_CHECK_RETURN(result, logger)                                                   \
    do {                                                                                    \
        cudaError_t _result = result;                                                       \
        if (CUDA_FAILED(_result)) {                                                         \
            _SET_ERROR(logger, ErrorCode::Runtime, #result " failed: %d: %s - %s at %s:%d", \
                       (_result), cudaGetErrorName(_result), cudaGetErrorString(_result),   \
                       __FILE__, __LINE__);                                                 \
            return ___status;                                                               \
        }                                                                                   \
    } while (0)

#define CUDA_CHECK_STREAM_RETURN(stream, logger)                             \
    do {                                                                     \
        if (logger.level() >= LoggerParameters::DebugSyncDevice) {           \
            cudaStreamSynchronize(stream);                                   \
            std::cout << "<<< " << __FILE__ << "@" << __LINE__ << std::endl; \
        }                                                                    \
        CUDA_CHECK_RETURN(cudaGetLastError(), logger);                       \
    } while (0)

/// @brief : a scoped cuda device guard
class CudaCheckDeviceGuard {
    int _prevDeviceIndex = -1;
    bool _check          = true;

public:
    CudaCheckDeviceGuard(int deviceIndex) {
        int currDeviceIndex;
        _check = CUDA_SUCCEEDED(cudaGetDevice(&currDeviceIndex));
        if (_check && (deviceIndex != currDeviceIndex)) {
            _prevDeviceIndex = currDeviceIndex;
            _check           = CUDA_SUCCEEDED(cudaSetDevice(deviceIndex));
        }
    }
    ~CudaCheckDeviceGuard() {
        if (_check && _prevDeviceIndex != -1) {
            cudaSetDevice(_prevDeviceIndex);
        }
    }
    inline bool check() const {
        return _check;
    }
};
