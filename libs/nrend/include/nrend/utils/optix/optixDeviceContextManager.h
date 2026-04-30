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

#include <nrend/utils/cuda/cudaCommon.h>
#include <nrend/utils/optix/optixCommon.h>

#include <optix_stubs.h>

#include <memory>
#include <vector>

namespace nrend {

class OptixDeviceContextManager {

    static void contextLogCB(unsigned int level, const char* tag, const char* message, void* logger) {
        Logger* loggerPtr = reinterpret_cast<Logger*>(logger);
        std::string msg   = "[Optix][" + std::string(tag) + "] " + std::string(message);
        loggerPtr->log(level, msg.c_str());
    }

public:
    using Ptr = std::shared_ptr<OptixDeviceContext>;

    static Ptr get(int cudaDeviceId, const Logger& logger) {

        static bool m_isOptixInitialized = false;
        static Logger m_logger;
        static std::vector<std::weak_ptr<OptixDeviceContext>> m_contextPtrs;

        if (cudaDeviceId <= -1) {
            LOG_ERROR(logger, "OptixDeviceContextManager : Invalid CUDA device id: %d", cudaDeviceId);
            return nullptr;
        }

        if (cudaDeviceId < m_contextPtrs.size() && !m_contextPtrs[cudaDeviceId].expired()) {
            return m_contextPtrs[cudaDeviceId].lock();
        }

        if (!m_isOptixInitialized) {
            const OptixResult result = optixInit();
            OPTIX_CHECK(result, logger);
            if (OPTIX_FAILED(result)) {
                return nullptr;
            }
            m_isOptixInitialized = true;
            m_logger             = logger;
        }

        OptixDeviceContextOptions options = {};
        options.logCallbackFunction       = &contextLogCB;
        options.logCallbackLevel          = m_logger.level();
        options.logCallbackData           = &m_logger;
        options.validationMode =
            m_logger.level() >= LoggerParameters::DebugSyncDevice ? OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL : OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
        CudaCheckDeviceGuard guard(cudaDeviceId);
        CUcontext cuCtx = 0; // zero means take the current context

        Ptr contextPtr           = std::shared_ptr<OptixDeviceContext>(new OptixDeviceContext(0), [](OptixDeviceContext* ptr) {
            if (*ptr) {
                optixDeviceContextDestroy(*ptr);
            }
        });
        const OptixResult result = optixDeviceContextCreate(cuCtx, &options, contextPtr.get());
        OPTIX_CHECK(result, logger);
        if (OPTIX_FAILED(result)) {
            return nullptr;
        }

        m_contextPtrs.resize(cudaDeviceId + 1);
        m_contextPtrs[cudaDeviceId] = contextPtr;
        return contextPtr;
    }
};

} // namespace nrend