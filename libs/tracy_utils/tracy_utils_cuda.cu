// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include "tracy_utils_cuda.cuh"
#include <cstring>
#include <iostream>
#include <mutex>
#include <unordered_map>

#ifdef TRACY_ENABLE
#include <tracy/TracyC.h>
#define TRACY_ENABLE_CUDA
#include <tracy/TracyCUDA.hpp>
#endif

namespace nre {
namespace tracy_utils {

namespace {
std::mutex g_gpuContextMutex;
std::unordered_map<cudaStream_t, std::unique_ptr<TracyGpuContext>> g_gpuContexts;
} // namespace

// TracyGpuContext implementation
TracyGpuContext::TracyGpuContext(const char* name, cudaStream_t stream)
    : context(nullptr), stream(stream), enabled(false){
#ifdef TRACY_ENABLE
                                            {try {// Check if CUDA is initialized
                                                  cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    std::cerr << "[TracyGpuContext] Warning: CUDA error before Tracy GPU init: "
              << cudaGetErrorString(err) << std::endl;
    // Clear the error
    cudaGetLastError();
}

// In Tracy 0.12.2, TracyCUDAContext() returns a tracy::CUDACtx*
context = TracyCUDAContext();
if (context && name) {
    TracyCUDAContextName(context, name, strlen(name));
}
// Start GPU profiling immediately
TracyCUDAStartProfiling(context);
enabled = true;

// Verify no CUDA errors after initialization
err = cudaGetLastError();
if (err != cudaSuccess) {
    std::cerr << "[TracyGpuContext] Warning: CUDA error after Tracy GPU init: "
              << cudaGetErrorString(err) << std::endl;
    cudaGetLastError(); // Clear the error
}
}
catch (const std::exception& e) {
    std::cerr << "[TracyGpuContext] Exception during GPU context creation: " << e.what() << std::endl;
    context = nullptr;
    enabled = false;
}
}
#endif
}

TracyGpuContext::~TracyGpuContext() {
#ifdef TRACY_ENABLE
    if (context) {
        // Stop GPU profiling before destroying context
        TracyCUDAStopProfiling(context);
        TracyCUDAContextDestroy(context);
    }
#endif
}

// Note: Manual GPU zone tracking methods have been removed
// Tracy 0.12.2 automatically tracks all GPU operations through CUPTI

void TracyGpuContext::collect() {
#ifdef TRACY_ENABLE
    if (enabled && context) {
        TracyCUDACollect(context);
    }
#endif
}

// Note: ScopedTracyGpuZone has been removed
// GPU zones are automatically tracked by Tracy's CUPTI integration

// Global GPU context management
TracyGpuContext* getGlobalGpuContext(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(g_gpuContextMutex);
    auto it = g_gpuContexts.find(stream);
    if (it != g_gpuContexts.end()) {
        return it->second.get();
    }
    return nullptr;
}

void initializeGlobalGpuContext(const char* name, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(g_gpuContextMutex);
    if (g_gpuContexts.find(stream) == g_gpuContexts.end()) {
        g_gpuContexts[stream] = std::make_unique<TracyGpuContext>(name, stream);
    }
}

void destroyGlobalGpuContext() {
    std::lock_guard<std::mutex> lock(g_gpuContextMutex);
    g_gpuContexts.clear();
}

void collectAllGpuContexts() {
    std::lock_guard<std::mutex> lock(g_gpuContextMutex);
    for (auto& pair : g_gpuContexts) {
        if (pair.second) {
            pair.second->collect();
        }
    }
}

// GPU memory tracking
void tracyGpuAlloc(void* ptr, size_t size, const char* name) {
#ifdef TRACY_ENABLE
    {
        if (name) {
            TracyAllocN(ptr, size, name);
        } else {
            TracyAlloc(ptr, size);
        }
    }
#endif
}

void tracyGpuFree(void* ptr, const char* name) {
#ifdef TRACY_ENABLE
    {
        if (name) {
            TracyFreeN(ptr, name);
        } else {
            TracyFree(ptr);
        }
    }
#endif
}

} // namespace tracy_utils
} // namespace nre
