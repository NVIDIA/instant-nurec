// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include <nrend/utils/cuda/cudaBuffer.h>
#include <nrend/utils/cuda/cudaMemoryAllocator.h>

nrend::CudaBuffer::~CudaBuffer() {
    if (m_owner && (m_size > 0)) {
        CudaMemoryAllocator::get().free(m_data, m_size);
    }
}

size_t nrend::CudaBuffer::size() const {
    return m_size;
}

const void* nrend::CudaBuffer::data() const {
    return m_data;
}

void* nrend::CudaBuffer::data() {
    return m_data;
}

uint64_t nrend::CudaBuffer::handle() const {
    return reinterpret_cast<uint64_t>(m_data);
}

nrend::Status nrend::CudaBuffer::resize(size_t size,
                                        uint64_t processQueueHandle,
                                        const Logger& logger) {
    if (size != m_size) {
        CHECK_STATUS_RETURN(clear(processQueueHandle, logger));
        if (size > 0) {
            CHECK_STATUS_RETURN(CudaMemoryAllocator::get().allocateAsync(
                m_data,
                size,
                reinterpret_cast<cudaStream_t>(processQueueHandle),
                logger));
        }
        m_size = size;
    }
    return Status();
}

nrend::Status nrend::CudaBuffer::enlarge(size_t size,
                                         uint64_t processQueueHandle,
                                         const Logger& logger) {
    if (size <= m_size) {
        return Status();
    }
    return resize(size, processQueueHandle, logger);
}

nrend::Status nrend::CudaBuffer::setFromHost(const void* hostMemory,
                                             size_t size,
                                             uint64_t processQueueHandle,
                                             const Logger& logger) {

    Status status = detach(false, processQueueHandle, logger);
    if (!status) {
        return status;
    }

    status = resize(size, processQueueHandle, logger);
    if (status) {
        CUDA_CHECK_RETURN(cudaMemcpyAsync(m_data, hostMemory, size, cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(processQueueHandle)), logger);
    }
    return status;
}

nrend::Status nrend::CudaBuffer::setFromHost(const void* hostMemory,
                                             KernelMemoryExtend extend,
                                             uint64_t processQueueHandle,
                                             const Logger& logger) {
    const auto type = extend.type;
    if (type != KernelMemoryType::Buffer) {
        RETURN_ERROR(logger, ErrorCode::BadInput, "CudaBuffer : kernel memory type is not a CudaBuffer");
    }
    const auto nbytes = (size_t)extend.buffer.nbytes;
    return setFromHost(hostMemory, nbytes, processQueueHandle, logger);
}

nrend::Status nrend::CudaBuffer::setFromDevice(const void* deviceMemory,
                                               size_t size,
                                               bool attach,
                                               uint64_t processQueueHandle,
                                               const Logger& logger) {

    Status status = detach(false, processQueueHandle, logger);
    if (!status) {
        return status;
    }

    if (attach) {
        status = clear(processQueueHandle, logger);
        if (status) {
            m_size  = size;
            m_data  = const_cast<void*>(deviceMemory);
            m_owner = false;
        }
    } else {
        status = resize(size, processQueueHandle, logger);
        if (status) {
            CUDA_CHECK_RETURN(cudaMemcpyAsync(m_data, deviceMemory, size, cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(processQueueHandle)), logger);
        }
    }

    return status;
}

nrend::Status nrend::CudaBuffer::copyFromDevice(const void* deviceMemory,
                                                size_t size,
                                                size_t offset,
                                                uint64_t processQueueHandle,
                                                const Logger& logger) {
    if ((deviceMemory == nullptr) && (size > 0)) {
        RETURN_ERROR(logger, ErrorCode::BadInput, "CudaBuffer::copyFromDevice : deviceMemory is invalid");
    }
    if (offset + size > m_size) {
        LOG_DEBUG(logger, "CudaBuffer::copyFromDevice : input size is greater than the buffer size [%zu + %zu > %zu]", offset, size, m_size);
    }
    CUDA_CHECK_RETURN(cudaMemcpyAsync(reinterpret_cast<uint8_t*>(m_data) + offset,
                                      deviceMemory,
                                      std::min(size, m_size - offset),
                                      cudaMemcpyDeviceToDevice,
                                      reinterpret_cast<cudaStream_t>(processQueueHandle)),
                      logger);
    return Status();
}

bool nrend::CudaBuffer::attached() const {
    return !m_owner;
}

nrend::Status nrend::CudaBuffer::detach(bool copy, uint64_t processQueueHandle, const Logger& logger) {
    if (!attached()) {
        return Status();
    }
    return copy ? setFromDevice(m_data, m_size, false, processQueueHandle, logger) : clear(processQueueHandle, logger);
}

nrend::Status nrend::CudaBuffer::clear(uint64_t processQueueHandle, const Logger& logger) {
    if (m_owner && (m_size > 0)) {
        CHECK_STATUS_RETURN(
            CudaMemoryAllocator::get().freeAsync(m_data,
                                                 m_size,
                                                 reinterpret_cast<cudaStream_t>(processQueueHandle),
                                                 logger));
    }
    m_size  = 0;
    m_data  = nullptr;
    m_owner = true;

    return Status();
}