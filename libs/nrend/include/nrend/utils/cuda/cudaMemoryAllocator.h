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

#include <nrend/allocatorParameters.h>

#include <nrend/utils/cuda/cudaCommon.h>
#include <nrend/utils/status.h>

namespace nrend {

struct CudaMemoryAllocator final {

    CudaMemoryAllocator()                           = default;
    CudaMemoryAllocator(CudaMemoryAllocator const&) = delete;
    void operator=(CudaMemoryAllocator const&)      = delete;

public:
    ~CudaMemoryAllocator() = default;

    static inline CudaMemoryAllocator& get() {
        static CudaMemoryAllocator instance;
        return instance;
    }

    static inline size_t deviceUsedMemory(int deviceIndex) {
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        return total - free;
    }

    inline void setAllocator(const DeviceMemoryAllocator& allocator) {
        m_allocatorImpl = allocator;
    }

    inline Status allocateAsync(void*& ptr, size_t size, cudaStream_t stream, const Logger& logger) {
        if (m_allocatorImpl.allocAsync != nullptr) {
            ErrorCode errorCode = m_allocatorImpl.allocAsync(ptr, size, reinterpret_cast<uint64_t>(stream), logger);
            if (errorCode != ErrorCode::None) {
                return Status(errorCode);
            }
        } else {
            CUDA_CHECK_RETURN(cudaMallocAsync(&ptr, size, stream), logger);
        }
        m_totalAllocatedSize += size;
        return Status();
    }

    inline Status freeAsync(void* ptr, size_t size, cudaStream_t stream, const Logger& logger) {
        if (m_allocatorImpl.freeAsync != nullptr) {
            ErrorCode errorCode = m_allocatorImpl.freeAsync(ptr, reinterpret_cast<uint64_t>(stream), logger);
            if (errorCode != ErrorCode::None) {
                return Status(errorCode);
            }
        } else {
            CUDA_CHECK_RETURN(cudaFreeAsync(ptr, stream), logger);
        }
        if (m_totalAllocatedSize < size) {
            RETURN_ERROR(logger,
                         ErrorCode::BadInput,
                         "CudaMemoryAllocator : freeing more memory than allocated [%zu/%zu]",
                         size, m_totalAllocatedSize);
        }
        m_totalAllocatedSize -= size;
        return Status();
    }

    inline Status allocateArray(const Logger& logger, cudaArray_t* array, const cudaChannelFormatDesc* desc, size_t width, size_t height = 0, unsigned int flags = 0) {
        CUDA_CHECK_RETURN(cudaMallocArray(array, desc, width, height, flags), logger);
        const size_t elementSize = (desc->x + desc->y + desc->z + desc->w) / 8U;
        // TODO(qi): This should be the actual size of the array, but we don't have a way to get it from cudaArray_t
        if (height > 0) {
            m_totalAllocatedSize += elementSize * width * height;
        } else {
            m_totalAllocatedSize += elementSize * width;
        }
        return Status();
    }

    inline Status allocateArray3D(const Logger& logger, cudaArray_t* array, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int flags = 0) {
        CUDA_CHECK_RETURN(cudaMalloc3DArray(array, desc, extent, flags), logger);
        const size_t elementSize = (desc->x + desc->y + desc->z + desc->w) / 8U;
        size_t nElements         = extent.width;
        if (extent.height > 0) {
            nElements *= extent.height;
        }
        if (extent.depth > 0) {
            nElements *= extent.depth;
        }
        // TODO(qi): This should be the actual size of the array, but we don't have a way to get it from cudaArray_t
        m_totalAllocatedSize += elementSize * nElements;
        return Status();
    }

    inline Status freeArray(const Logger& logger, cudaArray_t array, cudaChannelFormatDesc desc, size_t width, size_t height = 0, size_t depth = 0) {
        const size_t elementSize = (desc.x + desc.y + desc.z + desc.w) / 8U;
        size_t nElements         = width;
        if (height > 0) {
            nElements *= height;
        }
        if (depth > 0) {
            nElements *= depth;
        }
        CUDA_CHECK_RETURN(cudaFreeArray(array), logger);
        m_totalAllocatedSize -= elementSize * nElements;
        return Status();
    }

    inline void free(void* ptr, size_t size) {
        if (m_allocatorImpl.free != nullptr) {
            [[maybe_unused]] ErrorCode errorCode = m_allocatorImpl.free(ptr, Logger());
        } else {
            cudaFree(ptr);
        }
        m_totalAllocatedSize -= size;
    }

    inline void freeArray(cudaArray_t array, cudaChannelFormatDesc desc, size_t width, size_t height = 0, size_t depth = 0) {
        const size_t elementSize = (desc.x + desc.y + desc.z + desc.w) / 8U;
        size_t nElements         = width;
        if (height > 0) {
            nElements *= height;
        }
        if (depth > 0) {
            nElements *= depth;
        }
        cudaFreeArray(array);
        m_totalAllocatedSize -= elementSize * nElements;
    }

    inline size_t currentlyAllocated() const {
        return m_totalAllocatedSize;
    }

private:
    DeviceMemoryAllocator m_allocatorImpl;
    size_t m_totalAllocatedSize = 0;
};

} // namespace nrend
