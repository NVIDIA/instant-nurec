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

#include <nrend/utils/status.h>

#include <memory>
#include <vector>

namespace nrend {

enum struct KernelMemoryType {
    Buffer,

    // Following OpenGL's naming convention for texture formats
    // NOTE(qi): CUDA Texture only support 1-, 2-, 4-channel format

    Texture2D_RED_32F, // TODO(qi): add RED format
    Texture2D_RG_32F,  // TODO(qi): add RG format
    Texture2D_RGBA_32F,

    TextureCubeMap_RED_32F,
    TextureCubeMap_RG_32F,
    TextureCubeMap_RGBA_32F,

    Invalid,
};

struct KernelMemoryExtend {
    KernelMemoryType type = KernelMemoryType::Invalid;

    struct Buffer {
        size_t nbytes;
    };

    struct Tex2D {
        int width;
        int height;
    };

    struct CubeMap {
        int width;
    };

    union {
        Buffer buffer;
        Tex2D tex2D;
        CubeMap cubeMap;
    };
};

class KernelMemory {

public:
    virtual ~KernelMemory() = default;

    virtual const void* data() const = 0;
    virtual void* data()             = 0;

    virtual uint64_t handle() const = 0;

    // NOTE(qi): in my opinion, only setFromHost make sense to be exposed in the abstract class, because
    //           the abstraction only makes sense if device memory is polymorphic. For code that calls
    //           setFromDevice, device memory type should already be known, so the caller should be able
    //           to dynamic cast to the correct type.
    virtual Status setFromHost(const void* hostMemory, size_t size, uint64_t processQueueHandle, const Logger& logger)               = 0;
    virtual Status setFromHost(const void* hostMemory, KernelMemoryExtend extend, uint64_t processQueueHandle, const Logger& logger) = 0;

    virtual Status clear(uint64_t processQueueHandle, const Logger& logger) = 0;

    template <typename T>
    T* as() { return dynamic_cast<T*>(this); }

    template <typename T>
    const T* as() const { return dynamic_cast<const T*>(this); }
};

using KernelMemoryPtr    = std::unique_ptr<KernelMemory>;
using KernelMemoryPtrVec = std::vector<KernelMemoryPtr>;

class IScopedKernelMemory {

protected:
    KernelMemory* m_kernelMemoryPtr = nullptr;
    uint64_t m_processQueueHandle;

    IScopedKernelMemory() = delete;
    IScopedKernelMemory(uint64_t processQueueHandle)
        : m_processQueueHandle(processQueueHandle) {}

public:
    // no virtual call on m_kernelMemoryPtr allowed in the destructor
    virtual ~IScopedKernelMemory() = default;

    inline const void* data() const { return m_kernelMemoryPtr->data(); }
    inline void* data() { return m_kernelMemoryPtr->data(); }

    inline uint64_t handle() const { return m_kernelMemoryPtr->handle(); }

    inline Status setFromHost(const void* hostMemory, size_t size, const Logger& logger) {
        return m_kernelMemoryPtr->setFromHost(hostMemory, size, m_processQueueHandle, logger);
    }

    inline Status setFromHost(const void* hostMemory, KernelMemoryExtend extend, uint64_t processQueueHandle, const Logger& logger) {
        return m_kernelMemoryPtr->setFromHost(hostMemory, extend, m_processQueueHandle, logger);
    }

    inline Status clear(const Logger& logger) {
        return m_kernelMemoryPtr->clear(m_processQueueHandle, logger);
    }

    inline uint64_t processQueueHandle() const { return m_processQueueHandle; }
};

template <typename TKernelMemoryImp>
class ScopedKernelMemory : public IScopedKernelMemory {
    TKernelMemoryImp m_kernelMemoryImp;

public:
    ScopedKernelMemory(uint64_t processQueueHandle)
        : IScopedKernelMemory(processQueueHandle) {
        // cannot be set in the parent constructor
        m_kernelMemoryPtr = &m_kernelMemoryImp;
    }
    virtual ~ScopedKernelMemory() { clear({}); }
};
} // namespace nrend
