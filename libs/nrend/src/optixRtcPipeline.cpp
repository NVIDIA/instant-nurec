// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

/** This code has been adapted from tcnn/rtc_kernel.h
 *  @author Thomas Müller, NVIDIA
 */

#include <nrend/utils/optix/optixRtcPipeline.h>

#include <nrend/utils/cuda/cudaRtcKernel.h>

#include <tiny-cuda-nn/common_host.h>

#include <optix_function_table_definition.h>
#include <optix_stack_size.h>

#include <cstdlib>
#include <cstring>

namespace {

// Check if coverage mode is enabled via CUDA_COVERAGE_MODE=1
// When enabled, uses reduced optimization for better source correlation
bool isCoverageModeEnabled() {
    const char* env = std::getenv("CUDA_COVERAGE_MODE");
    return env && std::strcmp(env, "1") == 0;
}

// Get RTC optimization level from environment variable CUDA_RTC_OPT_LEVEL
// CUDA_COVERAGE_MODE=1: set opt level to 0 for source correlation
// Else by default returns -1, for Prod compilation
int getRtcOptLevelInt() {
    const char* env = std::getenv("CUDA_RTC_OPT_LEVEL");
    if (env) {
        int level = std::atoi(env);
        if (level >= 0 && level <= 3) {
            return level;
        }
    }
    return isCoverageModeEnabled() ? 0 : -1;
}

// Get OptiX optimization level for coverage mode
OptixCompileOptimizationLevel getOptixOptLevelForCoverage(int level) {
    switch (level) {
    case 0: return OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    case 1: return OPTIX_COMPILE_OPTIMIZATION_LEVEL_1;
    case 2: return OPTIX_COMPILE_OPTIMIZATION_LEVEL_2;
    case 3: return OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    default: return OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    }
}

template <typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct RayGenData {
    // No data needed
};
typedef SbtRecord<RayGenData> RayGenSbtRecord;

struct MissData {
    // No data needed
};
typedef SbtRecord<MissData> MissSbtRecord;

struct HitGroupData {
    // No data needed
};
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

}; // namespace

nrend::OptixRtcPipeline::OptixRtcPipeline(int cudaDeviceId,
                                          const OptixPipelineOptions& pipelineOptions,
                                          const std::string& pipelineCode,
                                          const std::vector<std::string>& includeDirs,
                                          const std::string& cacheDir,
                                          const std::vector<std::pair<std::string, const char*>>& extraIncludes,
                                          cudaStream_t stream,
                                          const Logger& logger,
                                          Status& status)
    : m_contextPtr(OptixDeviceContextManager::get(cudaDeviceId, logger)) {
    if (m_contextPtr) {
        status = createPipeline(*m_contextPtr, pipelineOptions, pipelineCode, includeDirs, cacheDir, extraIncludes, m_module, m_pipeline, m_sbt, stream, logger);
    } else {
        status = ErrorCode::InvalidResource;
    }
}

nrend::OptixRtcPipeline::~OptixRtcPipeline() {
    optixPipelineDestroy(m_pipeline);
    optixModuleDestroy(m_module);
}

nrend::Status nrend::OptixRtcPipeline::createPipeline(const OptixDeviceContext& context,
                                                      const OptixPipelineOptions& pipelineOptions,
                                                      const std::string& pipelineCode,
                                                      const std::vector<std::string>& includeDirs,
                                                      const std::string& cacheDir,
                                                      const std::vector<std::pair<std::string, const char*>>& extraIncludes,
                                                      OptixModule& module,
                                                      OptixPipeline& pipeline,
                                                      OptixShaderBindingTable& sbt,
                                                      cudaStream_t stream,
                                                      const Logger& logger) {

    char log[4096];
    size_t sizeofLog = sizeof(log);

    OptixPipelineCompileOptions pipelineCompileOptions = {};
    uint32_t maxTraversableGraphDepth                  = OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH;
    OptixModule builtinIsModule                        = nullptr;
    {
        OptixModuleCompileOptions moduleCompileOptions = {};
        moduleCompileOptions.maxRegisterCount          = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;

        // Coverage mode: use env-var-controlled optimization with debug info
        // Production mode: use original logger-based conditional behavior
        const int optLevel      = getRtcOptLevelInt();
        const bool coverageMode = (optLevel >= 0);
        if (coverageMode) {
            moduleCompileOptions.optLevel   = getOptixOptLevelForCoverage(optLevel);
            moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
        } else {
            // Prod behavior: logger-based conditional
            moduleCompileOptions.optLevel =
                logger.level() >= LoggerParameters::DebugSyncDevice ? OPTIX_COMPILE_OPTIMIZATION_LEVEL_0 : OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
            moduleCompileOptions.debugLevel =
                logger.level() >= LoggerParameters::DebugSyncDevice ? OPTIX_COMPILE_DEBUG_LEVEL_FULL : OPTIX_COMPILE_DEBUG_LEVEL_NONE;
        }

        pipelineCompileOptions.usesMotionBlur        = pipelineOptions.flags & OptixPipelineOptions::EnableMotionBlur;
        pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
        if (pipelineOptions.flags & OptixPipelineOptions::AllowSingleGas) {
            pipelineCompileOptions.traversableGraphFlags |= OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
            maxTraversableGraphDepth = 1;
        }
        if (pipelineOptions.flags & OptixPipelineOptions::AllowSingleLevelInstancing) {
            pipelineCompileOptions.traversableGraphFlags |= OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
            maxTraversableGraphDepth = 2;
        }
        pipelineCompileOptions.numPayloadValues                 = pipelineOptions.numPayloadValues;
        pipelineCompileOptions.numAttributeValues               = pipelineOptions.numAttributeValues;
        pipelineCompileOptions.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
        pipelineCompileOptions.pipelineLaunchParamsVariableName = pipelineOptions.parametersVariableName;
        pipelineCompileOptions.usesPrimitiveTypeFlags           = 0;
        if (pipelineOptions.flags & OptixPipelineOptions::EnableTrianglePrimitives) {
            pipelineCompileOptions.usesPrimitiveTypeFlags |= OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
        }
        if (pipelineOptions.flags & OptixPipelineOptions::EnableCustomPrimitives) {
            pipelineCompileOptions.usesPrimitiveTypeFlags |= OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
        }
        if (pipelineOptions.flags & OptixPipelineOptions::EnableSpherePrimitives) {
            pipelineCompileOptions.usesPrimitiveTypeFlags |= OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;
        }

        std::vector<const char*> entryPointNames;
        if (pipelineOptions.raygenEntryPointName) {
            entryPointNames.push_back(pipelineOptions.raygenEntryPointName);
        }
        if (pipelineOptions.missEntryPointName) {
            entryPointNames.push_back(pipelineOptions.missEntryPointName);
        }
        if (pipelineOptions.anyHitEntryPointName) {
            entryPointNames.push_back(pipelineOptions.anyHitEntryPointName);
        }
        if (pipelineOptions.closestHitEntryPointName) {
            entryPointNames.push_back(pipelineOptions.closestHitEntryPointName);
        }
        if (pipelineOptions.intersectionEntryPointName) {
            entryPointNames.push_back(pipelineOptions.intersectionEntryPointName);
        }

        const uint32_t computeCapability        = tcnn::cuda_supported_compute_capability();
        std::vector<std::string> compileOptions = {
            fmt::format("--gpu-architecture=compute_{}", computeCapability),
            fmt::format("-DTCNN_MIN_GPU_ARCH={}", computeCapability),
            "--std=c++17",
            "--use_fast_math",
            "-rdc", "true",
            "-D__OPTIX__"};
        if (coverageMode) {
            // Coverage mode: control PTX assembler optimization
            compileOptions.push_back("-lineinfo");
            compileOptions.push_back(fmt::format("--ptxas-options=-O{}", optLevel));
        } else {
            compileOptions.push_back("-lineinfo"); // TODO:Is this really necessary in prod? else remove this
            compileOptions.push_back("--extra-device-vectorization");
        }

        std::vector<char> ptxBuffer;
        std::vector<std::string> loweredEntryPointNames;
        CHECK_STATUS_RETURN(CudaRtcKernel::generatePTX(entryPointNames,
                                                       pipelineCode,
                                                       includeDirs,
                                                       cacheDir,
                                                       extraIncludes,
                                                       compileOptions,
                                                       ptxBuffer,
                                                       loweredEntryPointNames,
                                                       logger));

        OPTIX_CHECK_LOG_RETURN(
            optixModuleCreate(
                context, &moduleCompileOptions, &pipelineCompileOptions, ptxBuffer.data(), ptxBuffer.size(), log, &sizeofLog, &module),
            log, sizeofLog, logger);

        if (pipelineOptions.flags & OptixPipelineOptions::EnableSpherePrimitives) {
            OptixBuiltinISOptions isOptions = {OPTIX_PRIMITIVE_TYPE_SPHERE, 0, OPTIX_BUILD_FLAG_PREFER_FAST_TRACE, 0};
            OPTIX_CHECK_LOG_RETURN(
                optixBuiltinISModuleGet(context, &moduleCompileOptions, &pipelineCompileOptions, &isOptions, &builtinIsModule),
                log, sizeofLog, logger);
        }
    }

    OptixProgramGroup raygenProgramGroup   = nullptr;
    OptixProgramGroup missProgramGroup     = nullptr;
    OptixProgramGroup hitgroupProgramGroup = nullptr;
    {
        OptixProgramGroupOptions programGroupOptions = {}; // Initialize to zeros
        OptixProgramGroupDesc raygenProgramGroupDesc = {}; //
        raygenProgramGroupDesc.kind                  = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        if (pipelineOptions.raygenEntryPointName) {
            raygenProgramGroupDesc.raygen.module            = module;
            raygenProgramGroupDesc.raygen.entryFunctionName = pipelineOptions.raygenEntryPointName;
        }
        OPTIX_CHECK_LOG_RETURN(
            optixProgramGroupCreate(context, &raygenProgramGroupDesc,
                                    1, // num program groups
                                    &programGroupOptions, log, &sizeofLog, &raygenProgramGroup),
            log, sizeofLog, logger);

        OptixProgramGroupDesc missProgramGroupDesc = {};
        missProgramGroupDesc.kind                  = OPTIX_PROGRAM_GROUP_KIND_MISS;
        if (pipelineOptions.missEntryPointName) {
            missProgramGroupDesc.miss.module            = module;
            missProgramGroupDesc.miss.entryFunctionName = pipelineOptions.missEntryPointName;
        }
        OPTIX_CHECK_LOG_RETURN(
            optixProgramGroupCreate(context, &missProgramGroupDesc,
                                    1, // num program groups
                                    &programGroupOptions, log, &sizeofLog, &missProgramGroup),
            log, sizeofLog, logger);

        OptixProgramGroupDesc hitgroupProgramGroupDesc = {};
        hitgroupProgramGroupDesc.kind                  = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        if (pipelineOptions.closestHitEntryPointName) {
            hitgroupProgramGroupDesc.hitgroup.moduleCH            = module;
            hitgroupProgramGroupDesc.hitgroup.entryFunctionNameCH = pipelineOptions.closestHitEntryPointName;
        }
        if (pipelineOptions.intersectionEntryPointName) {
            hitgroupProgramGroupDesc.hitgroup.moduleIS            = module;
            hitgroupProgramGroupDesc.hitgroup.entryFunctionNameIS = pipelineOptions.intersectionEntryPointName;
        } else if (pipelineOptions.flags & OptixPipelineOptions::EnableSpherePrimitives) {
            hitgroupProgramGroupDesc.hitgroup.moduleIS            = builtinIsModule;
            hitgroupProgramGroupDesc.hitgroup.entryFunctionNameIS = nullptr;
        }
        if (pipelineOptions.anyHitEntryPointName) {
            hitgroupProgramGroupDesc.hitgroup.moduleAH            = module;
            hitgroupProgramGroupDesc.hitgroup.entryFunctionNameAH = pipelineOptions.anyHitEntryPointName;
        }
        OPTIX_CHECK_LOG_RETURN(
            optixProgramGroupCreate(context, &hitgroupProgramGroupDesc,
                                    1, // num program groups
                                    &programGroupOptions, log, &sizeofLog, &hitgroupProgramGroup),
            log, sizeofLog, logger);
    }

    {
        std::vector<OptixProgramGroup> programGroups;
        programGroups.push_back(raygenProgramGroup);
        programGroups.push_back(missProgramGroup);
        programGroups.push_back(hitgroupProgramGroup);

        OptixPipelineLinkOptions pipelineLinkOptions = {};
        pipelineLinkOptions.maxTraceDepth            = pipelineOptions.maxTraceDepth;

        OPTIX_CHECK_LOG_RETURN(optixPipelineCreate(context, &pipelineCompileOptions, &pipelineLinkOptions,
                                                   programGroups.data(), static_cast<unsigned int>(programGroups.size()),
                                                   log, &sizeofLog, &pipeline),
                               log, sizeofLog, logger);

        OptixStackSizes stackSizes = {};
        for (auto& progGroup : programGroups) {
            OPTIX_CHECK_RETURN(optixUtilAccumulateStackSizes(progGroup, &stackSizes, pipeline), logger);
        }

        uint32_t directCallableStackSizeFromTraversal;
        uint32_t directCallableStackSizeFromState;
        uint32_t continuationStackSize;
        OPTIX_CHECK_RETURN(optixUtilComputeStackSizes(&stackSizes, pipelineOptions.maxTraceDepth,
                                                      0, // maxCCDepth
                                                      0, // maxDCDEpth
                                                      &directCallableStackSizeFromTraversal,
                                                      &directCallableStackSizeFromState, &continuationStackSize),
                           logger);
        OPTIX_CHECK_RETURN(optixPipelineSetStackSize(pipeline, directCallableStackSizeFromTraversal,
                                                     directCallableStackSizeFromState, continuationStackSize,
                                                     maxTraversableGraphDepth),
                           logger);
    }

    // Set up shader binding table
    {
        RayGenSbtRecord rgSbt;
        OPTIX_CHECK_RETURN(optixSbtRecordPackHeader(raygenProgramGroup, &rgSbt), logger);
        m_sbtRaygenBuffer.setFromHostVector(std::vector<RayGenSbtRecord>{rgSbt}, reinterpret_cast<uint64_t>(stream), logger);

        MissSbtRecord msSbt;
        OPTIX_CHECK_RETURN(optixSbtRecordPackHeader(missProgramGroup, &msSbt), logger);
        m_sbtMissBuffer.setFromHostVector(std::vector<MissSbtRecord>{msSbt}, reinterpret_cast<uint64_t>(stream), logger);

        HitGroupSbtRecord hgSbt;
        OPTIX_CHECK_RETURN(optixSbtRecordPackHeader(hitgroupProgramGroup, &hgSbt), logger);
        m_sbtHitgroupBuffer.setFromHostVector(std::vector<HitGroupSbtRecord>{hgSbt}, reinterpret_cast<uint64_t>(stream), logger);

        sbt.raygenRecord                = reinterpret_cast<CUdeviceptr>(m_sbtRaygenBuffer.data());
        sbt.missRecordBase              = reinterpret_cast<CUdeviceptr>(m_sbtMissBuffer.data());
        sbt.missRecordStrideInBytes     = sizeof(MissSbtRecord);
        sbt.missRecordCount             = 1;
        sbt.hitgroupRecordBase          = reinterpret_cast<CUdeviceptr>(m_sbtHitgroupBuffer.data());
        sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
        sbt.hitgroupRecordCount         = 1;
    }

    return Status();
}
