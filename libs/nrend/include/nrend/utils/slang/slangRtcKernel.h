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

#ifndef NO_SLANG_RTC
#include <slang.h>
#endif

#include <cuda_runtime_api.h>

#include <string>
#include <vector>

namespace nrend {

class SlangRtcKernel {
public:
    SlangRtcKernel()  = default;
    ~SlangRtcKernel() = default;

    Status compile(const std::string& name,
                   const std::string& kernelCode,
                   const std::vector<std::string>& includeDirs,
                   const std::string& cacheDir,
                   const std::vector<std::pair<std::string, const char*>>& extraIncludes,
                   const Logger& logger) {
        return ErrorCode::NotImplemented;
    }

    enum class IntermediateTarget {
        Cuda,
        PTX
    };

    static Status
    generateIntermediateTarget(IntermediateTarget target,
                               const std::string& kernelCode,
                               const std::vector<std::string>& includeDirs,
                               const std::string& cacheDir,
                               const std::vector<std::pair<std::string, const char*>>& extraIncludes,
                               std::string& targetBuffer,
                               const Logger& logger)
#ifdef NO_SLANG_RTC
    {
        return ErrorCode::NotImplemented;
    }
#else
        ;
#endif

    template <typename... Types>
    Status launch(dim3 blocks, dim3 threads, uint32_t shmem_size, cudaStream_t stream, const Logger& logger, Types&&... args) {
        return ErrorCode::NotImplemented;
    }

#ifndef NO_SLANG_RTC
private:
    slang::IModule* m_module = nullptr;
#endif
};

} // namespace nrend
