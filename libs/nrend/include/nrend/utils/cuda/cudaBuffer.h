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

class CudaBuffer final : public KernelMemory {

public:
    CudaBuffer() = default;
    ~CudaBuffer();

    size_t size() const;

    virtual const void* data() const override;
    virtual void* data() override;

    virtual uint64_t handle() const override;

    template <typename T>
    inline T* ptr() {
        return reinterpret_cast<T*>(data());
    }
    template <typename T>
    inline const T* ptr() const {
        return reinterpret_cast<const T*>(data());
    }

    Status resize(size_t size, uint64_t processQueueHandle, const Logger& logger);

    Status enlarge(size_t size, uint64_t processQueueHandle, const Logger& logger);

    virtual Status setFromHost(const void* hostMemory,
                               size_t size,
                               uint64_t processQueueHandle,
                               const Logger& logger) override;

    virtual Status setFromHost(const void* hostMemory,
                               KernelMemoryExtend extend,
                               uint64_t processQueueHandle,
                               const Logger& logger) override;

    template <typename T>
    inline Status setFromHostVector(const std::vector<T>& hostVector, uint64_t processQueueHandle, const Logger& logger) {
        return setFromHost(hostVector.data(), hostVector.size() * sizeof(T), processQueueHandle, logger);
    }

    Status setFromDevice(const void* deviceMemory,
                         size_t size,
                         bool attach,
                         uint64_t processQueueHandle,
                         const Logger& logger);

    virtual Status clear(uint64_t processQueueHandle,
                         const Logger& logger) override;

    bool attached() const;

    Status detach(bool copy, uint64_t processQueueHandle, const Logger& logger);

    Status copyFromDevice(const void* deviceMemory, size_t size, size_t offset, uint64_t processQueueHandle, const Logger& logger);

    inline Status copyFromDevice(const void* deviceMemory, size_t size, uint64_t processQueueHandle, const Logger& logger) {
        return copyFromDevice(deviceMemory, size, 0, processQueueHandle, logger);
    }

    inline Status copyFromDevice(const CudaBuffer& deviceBuffer, uint64_t processQueueHandle, const Logger& logger) {
        return copyFromDevice(deviceBuffer.data(), deviceBuffer.size(), 0, processQueueHandle, logger);
    }

private:
    size_t m_size = 0;
    void* m_data  = nullptr;
    bool m_owner  = true;
};

struct ScopedCudaBuffer final : public ScopedKernelMemory<CudaBuffer> {
public:
    ScopedCudaBuffer(uint64_t processQueueHandle)
        : ScopedKernelMemory<CudaBuffer>(processQueueHandle) {}
    ~ScopedCudaBuffer() = default;

    template <typename T>
    inline T* ptr() {
        return reinterpret_cast<CudaBuffer*>(m_kernelMemoryPtr)->ptr<T>();
    }
    template <typename T>
    inline const T* ptr() const {
        return reinterpret_cast<const CudaBuffer*>(m_kernelMemoryPtr)->ptr<T>();
    }

    // size
    inline size_t size() const {
        return reinterpret_cast<CudaBuffer*>(m_kernelMemoryPtr)->size();
    }

    // resize
    inline Status resize(size_t size, const Logger& logger) {
        return reinterpret_cast<CudaBuffer*>(m_kernelMemoryPtr)->resize(size, m_processQueueHandle, logger);
    }

    // enlarge
    inline Status enlarge(size_t size, const Logger& logger) {
        return reinterpret_cast<CudaBuffer*>(m_kernelMemoryPtr)->enlarge(size, m_processQueueHandle, logger);
    }

    template <typename T>
    inline Status setFromHostVector(const std::vector<T>& hostVector, const Logger& logger) {
        return reinterpret_cast<CudaBuffer*>(m_kernelMemoryPtr)->template setFromHostVector<T>(hostVector, m_processQueueHandle, logger);
    }

    // setFromDevice
    inline Status setFromDevice(const void* deviceMemory, size_t size, bool attach, const Logger& logger) {
        return reinterpret_cast<CudaBuffer*>(m_kernelMemoryPtr)->setFromDevice(deviceMemory, size, attach, m_processQueueHandle, logger);
    }

    // attached
    inline bool attached() const {
        return reinterpret_cast<CudaBuffer*>(m_kernelMemoryPtr)->attached();
    }

    // detach
    inline Status detach(bool copy, const Logger& logger) {
        return reinterpret_cast<CudaBuffer*>(m_kernelMemoryPtr)->detach(copy, m_processQueueHandle, logger);
    }

    // copyFromDevice
    inline Status copyFromDevice(const void* deviceMemory, size_t size, size_t offset, const Logger& logger) {
        return reinterpret_cast<CudaBuffer*>(m_kernelMemoryPtr)->copyFromDevice(deviceMemory, size, offset, m_processQueueHandle, logger);
    }

    // copyFromDevice
    inline Status copyFromDevice(const void* deviceMemory, size_t size, const Logger& logger) {
        return reinterpret_cast<CudaBuffer*>(m_kernelMemoryPtr)->copyFromDevice(deviceMemory, size, m_processQueueHandle, logger);
    }

    // copyFromDevice
    inline Status copyFromDevice(const CudaBuffer& deviceBuffer, const Logger& logger) {
        return reinterpret_cast<CudaBuffer*>(m_kernelMemoryPtr)->copyFromDevice(deviceBuffer, m_processQueueHandle, logger);
    }
};

} // namespace nrend