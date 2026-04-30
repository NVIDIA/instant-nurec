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

#include <nrend/kernelResources/kernelMemory.h>
#include <nrend/utils/cuda/cudaCommon.h>
#include <nrend/utils/status.h>

namespace nrend {

// --------------------------------------------------------------
// CUDA 2D Texture
// --------------------------------------------------------------

class CudaTexture2D final : public KernelMemory {

public:
    CudaTexture2D() = default;
    ~CudaTexture2D();

    // FIXME: This could cause illegal memory access issue, we should really raise an
    //        error here once we have a better exception handling mechanism.
    virtual const void* data() const override { return nullptr; }
    virtual void* data() override { return nullptr; }

    virtual uint64_t handle() const override;

    virtual Status clear(uint64_t processQueueHandle, const Logger& logger) override;

    virtual Status setFromHost(const void* hostMemory, size_t size, uint64_t processQueueHandle, const Logger& logger) override;
    virtual Status setFromHost(const void* hostMemory, KernelMemoryExtend extend, uint64_t processQueueHandle, const Logger& logger) override;

    // FIXME: this is not implemented yet
    // Status setFromDevice(const void* deviceMemory, KernelMemoryExtend extend, uint64_t processQueueHandle, const Logger& logger);

    Status resize(KernelMemoryExtend extend, uint64_t processQueueHandle, const Logger& logger);

private:
    KernelMemoryType m_type = KernelMemoryType::Invalid;

    cudaArray_t m_array       = nullptr;
    cudaTextureObject_t m_tex = 0;

    size_t m_width       = 0;
    size_t m_height      = 0;
    size_t m_elementSize = 0;

    cudaChannelFormatDesc m_channelDesc;
    cudaResourceDesc m_res;
    cudaTextureDesc m_desc;

    Status setCudaChannelFormat(KernelMemoryType type) const;
};

// --------------------------------------------------------------
// CUDA Cube Map Texture
// --------------------------------------------------------------

class CudaTextureCubeMap final : public KernelMemory {

public:
    CudaTextureCubeMap() = default;
    ~CudaTextureCubeMap();

    // FIXME: This could cause illegal memory access issue, we should really raise an
    //        error here once we have a better exception handling mechanism.
    virtual const void* data() const override { return nullptr; }
    virtual void* data() override { return nullptr; }

    virtual uint64_t handle() const override;

    virtual Status clear(uint64_t processQueueHandle, const Logger& logger) override;

    virtual Status setFromHost(const void* hostMemory, size_t size, uint64_t processQueueHandle, const Logger& logger) override;
    virtual Status setFromHost(const void* hostMemory, KernelMemoryExtend extend, uint64_t processQueueHandle, const Logger& logger) override;

    // FIXME: this is doesnt work for some reason
    // Status setFromDevice(const void* deviceMemory, KernelMemoryExtend extend, uint64_t processQueueHandle, const Logger& logger);

    Status resize(KernelMemoryExtend extend, uint64_t processQueueHandle, const Logger& logger);

private:
    constexpr static int NUM_FACES = 6;

    KernelMemoryType m_type = KernelMemoryType::Invalid;

    cudaArray_t m_array       = nullptr;
    cudaTextureObject_t m_tex = 0;

    size_t m_width       = 0;
    size_t m_elementSize = 0;
    cudaExtent m_extent;

    cudaChannelFormatDesc m_channelDesc;
    cudaResourceDesc m_res;
    cudaTextureDesc m_desc;

    Status setCudaChannelFormat(KernelMemoryType type) const;
};

} // namespace nrend
