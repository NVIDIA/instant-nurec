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

#include <nrend/errorCodes.h>

#include <cstddef>
#include <cstdint>

#ifndef NREND_ALLOCATOR_CB
#ifdef _WIN32
#define NREND_ALLOCATOR_CB __cdecl
#else
#define NREND_ALLOCATOR_CB
#endif
#endif

namespace nrend {

class Logger;

struct DeviceMemoryAllocator {
    typedef ErrorCode(NREND_ALLOCATOR_CB* AllocAsyncCallback)(void*& ptr,
                                                              size_t size,
                                                              uint64_t stream,
                                                              const Logger& logger);
    typedef ErrorCode(NREND_ALLOCATOR_CB* FreeAsyncCallback)(void* ptr,
                                                             uint64_t stream,
                                                             const Logger& logger);
    typedef ErrorCode(NREND_ALLOCATOR_CB* FreeCallback)(void* ptr,
                                                        const Logger& logger);

    AllocAsyncCallback allocAsync = nullptr;
    FreeAsyncCallback freeAsync   = nullptr;
    FreeCallback free             = nullptr;
}; // struct DeviceMemoryAllocator
} // namespace nrend