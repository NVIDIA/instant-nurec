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

#include "tracy_utils.h"
#include <cuda_runtime.h>

#ifdef TRACY_ENABLE
#include <tracy/TracyC.h>
// Forward declaration
namespace tracy {
class CUDACtx;
}
#endif

namespace nre {
namespace tracy_utils {

/**
 * GPU profiling context for Tracy 0.12.2
 *
 * Tracy automatically tracks GPU operations through CUPTI:
 * - CUDA kernel launches
 * - Memory operations (cudaMemcpy, cudaMemset)
 * - Synchronization operations
 * - Memory allocations/deallocations
 *
 * Thread Safety:
 * - Individual TracyGpuContext instances are NOT thread-safe
 * - Each instance should only be used by one thread at a time
 * - Multiple instances can be used concurrently from different threads
 * - The global GPU context management functions are thread-safe
 */
class TracyGpuContext {
public:
    TracyGpuContext(const char* name, cudaStream_t stream = 0);
    ~TracyGpuContext();

    // Delete copy constructor and copy assignment operator
    // TracyGpuContext manages GPU resources that should not be copied
    TracyGpuContext(const TracyGpuContext&)            = delete;
    TracyGpuContext& operator=(const TracyGpuContext&) = delete;

    // Delete move constructor and move assignment operator
    // GPU context is tied to specific CUDA streams and should not be moved
    TracyGpuContext(TracyGpuContext&&)            = delete;
    TracyGpuContext& operator=(TracyGpuContext&&) = delete;

    // Collect GPU timestamps periodically (flushes CUPTI events)
    void collect();

    cudaStream_t getStream() const { return stream; }
    bool isEnabled() const { return enabled; }

private:
#ifdef TRACY_ENABLE
    tracy::CUDACtx* context;
#else
    void* context;
#endif
    cudaStream_t stream;
    bool enabled;
};

// Global GPU context management
TracyGpuContext* getGlobalGpuContext(cudaStream_t stream = 0);
void initializeGlobalGpuContext(const char* name = "CUDA", cudaStream_t stream = 0);
void destroyGlobalGpuContext();
void collectAllGpuContexts();

/**
 * Utility functions for GPU memory tracking
 * Note: CUPTI also automatically tracks cudaMalloc/cudaFree operations
 */
void tracyGpuAlloc(void* ptr, size_t size, const char* name = nullptr);
void tracyGpuFree(void* ptr, const char* name = nullptr);

} // namespace tracy_utils
} // namespace nre
