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

#include <nrend/utils/cuda/cudaBuffer.h>

#include <nrend/utils/optix/optixCommon.h>
#include <nrend/utils/optix/optixDeviceContextManager.h>

namespace nrend {

class OptixRtcPipeline : public RtcKernel {

    Status createPipeline(const OptixDeviceContext& context,
                          const OptixPipelineOptions& pipelineOptions,
                          const std::string& pipelineCode,
                          const std::vector<std::string>& includeDirs,
                          const std::string& cacheDir,
                          const std::vector<std::pair<std::string, const char*>>& extraIncludes,
                          OptixModule& module,
                          OptixPipeline& pipeline,
                          OptixShaderBindingTable& sbt,
                          cudaStream_t stream,
                          const Logger& logger);

public:
    OptixRtcPipeline(int cudaDeviceId,
                     const OptixPipelineOptions& pipelineOptions,
                     const std::string& pipelineCode,
                     const std::vector<std::string>& includeDirs,
                     const std::string& cacheDir,
                     const std::vector<std::pair<std::string, const char*>>& extraIncludes,
                     cudaStream_t stream,
                     const Logger& logger,
                     Status& status);
    ~OptixRtcPipeline();

    template <typename ParametersType>
    Status launch(dim3 blocks,
                  cudaStream_t stream,
                  const Logger& logger,
                  ParametersType* parametersDevicePtr) {
        OPTIX_CHECK_RETURN(optixLaunch(m_pipeline, stream, reinterpret_cast<CUdeviceptr>(parametersDevicePtr),
                                       sizeof(ParametersType), &m_sbt, blocks.x, blocks.y, blocks.z),
                           logger);
        return Status();
    }

private:
    OptixDeviceContextManager::Ptr m_contextPtr;
    OptixPipeline m_pipeline      = {};
    OptixModule m_module          = {};
    OptixShaderBindingTable m_sbt = {};
    CudaBuffer m_sbtRaygenBuffer;
    CudaBuffer m_sbtMissBuffer;
    CudaBuffer m_sbtHitgroupBuffer;
};

} // namespace nrend
