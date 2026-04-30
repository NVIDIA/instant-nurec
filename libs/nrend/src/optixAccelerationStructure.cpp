// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include <nrend/utils/optix/optixAccelerationStructure.h>

nrend::OptixAccelerationStructure::OptixAccelerationStructure(int cudaDeviceId, cudaStream_t cudaStream, const Logger& logger)
    : context(OptixDeviceContextManager::get(cudaDeviceId, logger)), scopedBuffer(reinterpret_cast<uint64_t>(cudaStream)), aabbScopedBuffer(reinterpret_cast<uint64_t>(cudaStream)) {}

nrend::Status nrend::OptixAccelerationStructure::buildAS(OptixAccelerationStructure& as,
                                                         InputType inputType,
                                                         uint32_t inputDataSize,
                                                         CUdeviceptr inputData,
                                                         uint32_t extInputDataSize,
                                                         CUdeviceptr extInputData,
                                                         uint32_t buildFlags,
                                                         const Logger& logger) {
    OptixAccelBuildOptions asOptions = {};
    asOptions.buildFlags             = OPTIX_BUILD_FLAG_NONE;
    // Fast trace as precedence
    if (buildFlags & FastTrace) {
        asOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    } else if (buildFlags & FastBuild) {
        asOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
    }
    if (buildFlags & AllowUpdate) {
        asOptions.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    }
    asOptions.operation             = buildFlags & Update ? OPTIX_BUILD_OPERATION_UPDATE : OPTIX_BUILD_OPERATION_BUILD;
    asOptions.motionOptions.numKeys = 0;

    uint32_t asInputFlags   = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
    OptixBuildInput asInput = {};
    if (inputType == Custom) {
        asInput.type                               = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        asInput.customPrimitiveArray.numPrimitives = inputDataSize / sizeof(OptixAabb);
        asInput.customPrimitiveArray.aabbBuffers   = &inputData;
        asInput.customPrimitiveArray.strideInBytes = 0;
        asInput.customPrimitiveArray.flags         = &asInputFlags;
        asInput.customPrimitiveArray.numSbtRecords = 1;
    } else if (inputType == Instance) {
        asInput.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        asInput.instanceArray.numInstances = inputDataSize / sizeof(OptixInstance);
        asInput.instanceArray.instances    = inputData;
    } else if (inputType == Sphere) {
        asInput.type                            = OPTIX_BUILD_INPUT_TYPE_SPHERES;
        asInput.sphereArray.vertexBuffers       = &inputData;
        asInput.sphereArray.vertexStrideInBytes = 0;
        asInput.sphereArray.numVertices         = inputDataSize / (sizeof(float) * 3);
        asInput.sphereArray.radiusBuffers       = &extInputData;
        asInput.sphereArray.radiusStrideInBytes = 0;
        asInput.sphereArray.singleRadius        = 0;
        asInput.sphereArray.flags               = &asInputFlags;
        asInput.sphereArray.numSbtRecords       = 1;
    } else if (inputType == TriangleMesh) {
        asInput.type                           = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        asInput.triangleArray.vertexFormat     = OPTIX_VERTEX_FORMAT_FLOAT3;
        asInput.triangleArray.numVertices      = inputDataSize / (sizeof(float) * 3);
        asInput.triangleArray.vertexBuffers    = &inputData;
        asInput.triangleArray.indexFormat      = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        asInput.triangleArray.numIndexTriplets = extInputDataSize / (sizeof(uint32_t) * 3);
        asInput.triangleArray.indexBuffer      = extInputData;
        asInput.triangleArray.flags            = &asInputFlags;
        asInput.triangleArray.numSbtRecords    = 1;
    } else {
        RETURN_ERROR(logger, ErrorCode::BadInput, "OptixAccelerationStructure: Invalid primitive type");
    }

    OptixAccelBufferSizes asBufferSizes;
    OPTIX_CHECK_RETURN(optixAccelComputeMemoryUsage(*as.context, &asOptions, &asInput,
                                                    1, // Number of build inputs
                                                    &asBufferSizes),
                       logger);

    ScopedCudaBuffer asBufferTmp(as.scopedBuffer.processQueueHandle());
    CHECK_STATUS_RETURN(asBufferTmp.resize(asBufferSizes.tempSizeInBytes, logger));
    CHECK_STATUS_RETURN(as.scopedBuffer.enlarge(asBufferSizes.outputSizeInBytes, logger));
    CHECK_STATUS_RETURN(as.aabbScopedBuffer.enlarge(sizeof(OptixAabb), logger));

    OptixAccelEmitDesc emitDesc = {};
    emitDesc.type               = OPTIX_PROPERTY_TYPE_AABBS;
    emitDesc.result             = reinterpret_cast<CUdeviceptr>(as.aabbScopedBuffer.data());

    OPTIX_CHECK_RETURN(optixAccelBuild(*as.context,
                                       reinterpret_cast<cudaStream_t>(as.scopedBuffer.processQueueHandle()), // CUDA stream
                                       &asOptions,
                                       &asInput,
                                       1, // num build inputs
                                       reinterpret_cast<CUdeviceptr>(asBufferTmp.data()),
                                       asBufferSizes.tempSizeInBytes,
                                       reinterpret_cast<CUdeviceptr>(as.scopedBuffer.data()),
                                       asBufferSizes.outputSizeInBytes,
                                       &as.handle,
                                       &emitDesc, // emitted property list
                                       1          // num emitted properties
                                       ),
                       logger);
    return Status();
}

nrend::Status nrend::OptixAccelerationStructure::buildUnitAABBInstanceAS(OptixAccelerationStructure& as,
                                                                         const Logger& logger) {
    OptixAccelBuildOptions asOptions = {};
    asOptions.buildFlags             = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    asOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

    OptixAabb hostOptixAabb{-1.f, -1.f, -1.f, 1.f, 1.f, 1.f};
    CHECK_STATUS_RETURN(as.aabbScopedBuffer.setFromHostVector(std::vector<OptixAabb>{hostOptixAabb}, logger));
    CUdeviceptr aabbDevicePtr = reinterpret_cast<CUdeviceptr>(as.aabbScopedBuffer.data());

    const uint32_t aabbInputFlags                = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
    OptixBuildInput aabbInput                    = {};
    aabbInput.type                               = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabbInput.customPrimitiveArray.numPrimitives = 1;
    aabbInput.customPrimitiveArray.aabbBuffers   = &aabbDevicePtr;
    aabbInput.customPrimitiveArray.strideInBytes = 0;
    aabbInput.customPrimitiveArray.flags         = &aabbInputFlags;
    aabbInput.customPrimitiveArray.numSbtRecords = 1;

    OptixAccelBufferSizes asBufferSizes;
    OPTIX_CHECK_RETURN(optixAccelComputeMemoryUsage(*as.context, &asOptions, &aabbInput,
                                                    1, // Number of build inputs
                                                    &asBufferSizes),
                       logger);
    ScopedCudaBuffer asBufferTmp(as.scopedBuffer.processQueueHandle());
    CHECK_STATUS_RETURN(asBufferTmp.resize(asBufferSizes.tempSizeInBytes, logger));

    CHECK_STATUS_RETURN(as.scopedBuffer.enlarge(asBufferSizes.outputSizeInBytes, logger));

    OPTIX_CHECK_RETURN(optixAccelBuild(*as.context,
                                       reinterpret_cast<cudaStream_t>(as.scopedBuffer.processQueueHandle()), // CUDA stream
                                       &asOptions,
                                       &aabbInput,
                                       1, // num build inputs
                                       reinterpret_cast<CUdeviceptr>(asBufferTmp.data()),
                                       asBufferSizes.tempSizeInBytes,
                                       reinterpret_cast<CUdeviceptr>(as.scopedBuffer.data()),
                                       asBufferSizes.outputSizeInBytes,
                                       &as.handle,
                                       nullptr, // emitted property list
                                       0        // num emitted properties
                                       ),
                       logger);
    return Status();
};
