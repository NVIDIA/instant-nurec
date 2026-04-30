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

#include <nrend/utils/cuda/cudaBuffer.h>
#include <nrend/utils/logger.h>
#include <nrend/utils/optix/optixCommon.h>
#include <nrend/utils/optix/optixDeviceContextManager.h>

namespace nrend {
struct OptixAccelerationStructure {

    OptixDeviceContextManager::Ptr context;
    OptixTraversableHandle handle = 0;
    ScopedCudaBuffer scopedBuffer;
    ScopedCudaBuffer aabbScopedBuffer;

    enum InputType {
        Custom,
        Instance,
        Sphere,
        TriangleMesh,
        NumInputTypes
    };

    enum BuildFlags {
        DefaultBuild = 0,
        AllowUpdate  = 1 << 0,
        Update       = 1 << 1,
        FastBuild    = 1 << 2,
        FastTrace    = 1 << 3,
    };

public:
    OptixAccelerationStructure(int cudaDeviceId, cudaStream_t cudaStream, const Logger& logger);
    virtual ~OptixAccelerationStructure() = default;

    static Status buildAS(OptixAccelerationStructure& as,
                          InputType inputType,
                          uint32_t inputDataSize,
                          CUdeviceptr inputData,
                          uint32_t extInputDataSize,
                          CUdeviceptr extInputData,
                          uint32_t buildFlags,
                          const Logger& logger);

    static Status buildUnitAABBInstanceAS(OptixAccelerationStructure& as, const Logger& logger);
};

} // namespace nrend
