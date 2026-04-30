// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include <nrend/errorCodes.h>
#include <nrend/utils/cuda/cudaMemoryAllocator.h>
#include <nrend/utils/cuda/cudaTexture.h>

#include <cstring>

// --------------------------------------------------------------
// CUDA 2D Texture
// --------------------------------------------------------------

nrend::CudaTexture2D::~CudaTexture2D() {
    if (m_array) {
        CudaMemoryAllocator::get().freeArray(m_array, m_channelDesc, m_width, m_height);
    }
    if (m_tex) {
        cudaDestroyTextureObject(m_tex);
    }
    m_width  = 0;
    m_height = 0;
}

uint64_t nrend::CudaTexture2D::handle() const {
    static_assert(sizeof(cudaTextureObject_t) == sizeof(uint64_t), "cudaTextureObject_t is not the same size as uint64_t");
    return (uint64_t)m_tex;
}

nrend::Status nrend::CudaTexture2D::setFromHost(const void* hostMemory, size_t size, uint64_t processQueueHandle, const Logger& logger) {
    RETURN_ERROR(logger, ErrorCode::InvalidResource, "CudaTexture2D : setFromHost is not supported.");
}

nrend::Status nrend::CudaTexture2D::setFromHost(const void* hostMemory, KernelMemoryExtend extend, uint64_t processQueueHandle, const Logger& logger) {

    Status status = resize(extend, processQueueHandle, logger);
    if (status) {
        CUDA_CHECK_RETURN(
            cudaMemcpy2DToArray(
                m_array, 0, 0, hostMemory,
                m_width * m_elementSize,
                m_width * m_elementSize,
                m_height, cudaMemcpyHostToDevice),
            logger);
    }
    return status;
}

nrend::Status nrend::CudaTexture2D::resize(KernelMemoryExtend extend, uint64_t processQueueHandle, const Logger& logger) {

    const auto type   = extend.type;
    const auto width  = (size_t)extend.tex2D.width;
    const auto height = (size_t)extend.tex2D.height;

    if (type != m_type || width != m_width || height != m_height) {
        CHECK_STATUS_RETURN(clear(processQueueHandle, logger));

        m_type   = type;
        m_width  = width;
        m_height = height;

        if (width > 0 && height > 0) {
            switch (type) {
            case KernelMemoryType::Texture2D_RED_32F:
                m_channelDesc = cudaCreateChannelDesc<float>();
                break;
            case KernelMemoryType::Texture2D_RG_32F:
                m_channelDesc = cudaCreateChannelDesc<float2>();
                break;
            case KernelMemoryType::Texture2D_RGBA_32F:
                m_channelDesc = cudaCreateChannelDesc<float4>();
                break;
            default:
                RETURN_ERROR(logger, ErrorCode::InvalidResource, "CudaTexture2D : incorrect memory type <%d>", (int)m_type);
            }

            m_elementSize = (m_channelDesc.x + m_channelDesc.y + m_channelDesc.z + m_channelDesc.w) / 8U;

            CHECK_STATUS_RETURN(
                CudaMemoryAllocator::get().allocateArray(logger, &m_array, &m_channelDesc, m_width, m_height));

            memset(&m_res, 0, sizeof(cudaResourceDesc));
            m_res.resType         = cudaResourceTypeArray;
            m_res.res.array.array = m_array;

            memset(&m_desc, 0, sizeof(cudaTextureDesc));
            m_desc.normalizedCoords = true;
            m_desc.filterMode       = cudaFilterModeLinear;
            m_desc.addressMode[0]   = cudaAddressModeWrap;
            m_desc.addressMode[1]   = cudaAddressModeWrap;
            m_desc.readMode         = cudaReadModeElementType;

            CUDA_CHECK_RETURN(cudaCreateTextureObject(&m_tex, &m_res, &m_desc, NULL), logger);
        }
    }

    return Status();
}

nrend::Status nrend::CudaTexture2D::clear(uint64_t processQueueHandle, const Logger& logger) {
    if (m_array) {
        CHECK_STATUS_RETURN(CudaMemoryAllocator::get().freeArray(logger, m_array, m_channelDesc, m_width, m_height));
    }
    if (m_tex) {
        CUDA_CHECK_RETURN(cudaDestroyTextureObject(m_tex), logger);
    }
    m_width  = 0;
    m_height = 0;
    return Status();
}

// --------------------------------------------------------------
// CUDA Cube Map Texture
// --------------------------------------------------------------

nrend::CudaTextureCubeMap::~CudaTextureCubeMap() {
    if (m_array) {
        CudaMemoryAllocator::get().freeArray(m_array, m_channelDesc, m_width, m_width, NUM_FACES);
    }
    if (m_tex) {
        cudaDestroyTextureObject(m_tex);
    }
    m_width = 0;
}

uint64_t nrend::CudaTextureCubeMap::handle() const {
    static_assert(sizeof(cudaTextureObject_t) == sizeof(uint64_t), "cudaTextureObject_t is not the same size as uint64_t");
    return (uint64_t)m_tex;
}

nrend::Status nrend::CudaTextureCubeMap::setFromHost(const void* hostMemory, size_t size, uint64_t processQueueHandle, const Logger& logger) {
    RETURN_ERROR(logger, ErrorCode::InvalidResource, "CudaTextureCubeMap : setFromHost is not supported.");
}

nrend::Status nrend::CudaTextureCubeMap::setFromHost(const void* hostMemory, KernelMemoryExtend extend, uint64_t processQueueHandle, const Logger& logger) {
    Status status = resize(extend, processQueueHandle, logger);
    if (status) {
        cudaMemcpy3DParms parms = {0};
        parms.srcPtr            = make_cudaPitchedPtr((void*)hostMemory, m_width * m_elementSize, m_width, m_width);
        parms.dstArray          = m_array;
        parms.extent            = m_extent;
        parms.kind              = cudaMemcpyHostToDevice;
        CUDA_CHECK_RETURN(cudaMemcpy3D(&parms), logger);
    }
    return status;
}

// // FIXME: this cause illegal memory access issue?!
// nrend::Status nrend::CudaTextureCubeMap::setFromDevice(const void* deviceMemory, KernelMemoryExtend extend, uint64_t processQueueHandle, const Logger& logger) {
//     Status status = resize(extend, processQueueHandle, logger);
//     if (status) {
//         cudaMemcpy3DParms parms = {0};
//         parms.srcPtr            = make_cudaPitchedPtr((void*)deviceMemory, m_width * m_elementSize, m_width, m_width);
//         parms.dstArray          = m_array;
//         parms.extent            = m_extent;
//         parms.kind              = cudaMemcpyDeviceToDevice;
//         CUDA_CHECK_RETURN(cudaMemcpy3D(&parms), logger);
//     }
//     return status;
// }

nrend::Status nrend::CudaTextureCubeMap::resize(KernelMemoryExtend extend, uint64_t processQueueHandle, const Logger& logger) {
    const auto type  = extend.type;
    const auto width = (size_t)extend.cubeMap.width;

    if (type != m_type || width != m_width) {
        CHECK_STATUS_RETURN(clear(processQueueHandle, logger));

        if (width > 0) {
            m_extent = make_cudaExtent(width, width, NUM_FACES);
            m_type   = type;

            switch (m_type) {
            case KernelMemoryType::TextureCubeMap_RED_32F:
                m_channelDesc = cudaCreateChannelDesc<float>();
                break;
            case KernelMemoryType::TextureCubeMap_RG_32F:
                m_channelDesc = cudaCreateChannelDesc<float2>();
                break;
            case KernelMemoryType::TextureCubeMap_RGBA_32F:
                m_channelDesc = cudaCreateChannelDesc<float4>();
                break;
            default:
                RETURN_ERROR(logger, ErrorCode::InvalidResource, "CudaTextureCubeMap : incorrect memory type <%d>", (int)m_type);
            }

            m_elementSize = (m_channelDesc.x + m_channelDesc.y + m_channelDesc.z + m_channelDesc.w) / 8U;

            CHECK_STATUS_RETURN(
                CudaMemoryAllocator::get().allocateArray3D(
                    logger,
                    &m_array,
                    &m_channelDesc,
                    m_extent,
                    cudaArrayCubemap));

            memset(&m_res, 0, sizeof(cudaResourceDesc));
            m_res.resType         = cudaResourceTypeArray;
            m_res.res.array.array = m_array;

            memset(&m_desc, 0, sizeof(cudaTextureDesc));
            m_desc.normalizedCoords = true;
            m_desc.filterMode       = cudaFilterModeLinear;
            m_desc.addressMode[0]   = cudaAddressModeClamp;
            m_desc.addressMode[1]   = cudaAddressModeClamp;
            m_desc.addressMode[2]   = cudaAddressModeClamp;
            m_desc.readMode         = cudaReadModeElementType;

            CUDA_CHECK_RETURN(cudaCreateTextureObject(&m_tex, &m_res, &m_desc, NULL), logger);
        }
        m_width = width;
    }

    return Status();
}

nrend::Status nrend::CudaTextureCubeMap::clear(uint64_t processQueueHandle, const Logger& logger) {
    if (m_array) {
        CHECK_STATUS_RETURN(CudaMemoryAllocator::get().freeArray(logger, m_array, m_channelDesc, m_width, m_width, NUM_FACES));
    }
    if (m_tex) {
        CUDA_CHECK_RETURN(cudaDestroyTextureObject(m_tex), logger);
    }
    m_width = 0;
    return Status();
}
