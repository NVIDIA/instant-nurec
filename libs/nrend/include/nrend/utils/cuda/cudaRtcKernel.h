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

#include <nrend/kernelResources/rtcKernel.h>

#include <cuda.h>

namespace nrend {

class CudaRtcKernel : public RtcKernel {

public:
    CudaRtcKernel(const CudaKernelOptions& kernelOptions,
                  const std::string& kernelCode,
                  const std::vector<std::string>& includeDirs,
                  const std::string& cacheDir,
                  const std::vector<std::pair<std::string, const char*>>& extraIncludes,
                  const Logger& logger,
                  Status& status);
    ~CudaRtcKernel();

    static Status generatePTX(std::vector<const char*> kernelNames,
                              const std::string& kernelCode,
                              const std::vector<std::string>& includeDirs,
                              const std::string& cacheDir,
                              const std::vector<std::pair<std::string, const char*>>& extraIncludes,
                              const std::vector<std::string>& options,
                              std::vector<char>& ptxBuffer,
                              std::vector<std::string>& kernelFunctionNames,
                              const Logger& logger);

    CUfunction getKernelFunction(uint32_t entryPointIndex, const Logger& logger) const;

    Status setKernelCacheConfig(uint32_t entryPointIndex, CUfunc_cache cacheConfig, const Logger& logger);

    template <typename... Types>
    Status launch(uint32_t kernelIndex,
                  dim3 blocks,
                  dim3 threads,
                  uint32_t shmemSize,
                  cudaStream_t stream,
                  const Logger& logger,
                  Types&&... args) {

        RETURN_ERROR_IF(kernelIndex >= m_kernels.size(), logger, ErrorCode::BadInput,
                        "CudaRtcKernel : invalid kernel index [%u / %zu].", kernelIndex, m_kernels.size());

        Status status;

        if (blocks.x * blocks.y * blocks.z == 0 || threads.x * threads.y * threads.z == 0) {
            return status;
        }

        // CUDA docs state that one has to opt-in for larger amounts of shmem than 48KiB == 49'152B
        if (shmemSize > 49152) {
            status = set(m_kernels[kernelIndex], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmemSize, logger);
            if (!status) {
                return status;
            }
        }

        const void* args_array[sizeof...(Types)] = {&args...};
        CUresult result                          = cuLaunchKernel(m_kernels[kernelIndex],
                                                                  blocks.x, blocks.y, blocks.z,
                                                                  threads.x, threads.y, threads.z,
                                                                  shmemSize,
                                                                  stream,
                                                                  (void**)args_array, nullptr);
        if (result != CUDA_SUCCESS) {
            const char* msg;
            cuGetErrorName(result, &msg);
            if (logger.level() >= LoggerParameters::Debug) {
                const std::string name = m_loweredKernelNames[kernelIndex];
                RETURN_ERROR(logger, ErrorCode::Runtime, "CudaRtcKernel : (%s) launch failed: (%d) %s", name.c_str(), static_cast<int>(result), msg);
            } else {
                RETURN_ERROR(logger, ErrorCode::Runtime, "CudaRtcKernel : launch failed: (%d) %s", static_cast<int>(result), msg);
            }
        }

        return Status();
    }

private:
    void clear();

    static Status set(CUfunction kernel, CUfunction_attribute attr, int value, const Logger& logger);

    CUmodule m_module = {};
    std::vector<CUfunction> m_kernels;
    std::vector<std::string> m_loweredKernelNames;
};

} // namespace nrend
