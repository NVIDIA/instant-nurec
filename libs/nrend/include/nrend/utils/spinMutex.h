// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#pragma once

#include <atomic>
#include <mutex>

#if defined(__x86_64__) /*GCC*/ || defined(_M_X64) /*MSVC*/
#if defined(_MSC_VER)
#pragma intrinsic(_mm_pause)
#define NREND_HARDWARE_PAUSE() _mm_pause()
#else
#define NREND_HARDWARE_PAUSE() __builtin_ia32_pause()
#endif
#elif defined(__aarch64__)
#define NREND_HARDWARE_PAUSE() __asm__ __volatile__("yield" :: \
                                                        : "memory")
#endif

namespace nrend {

struct SpinMutex final {
#if defined(__x86_64__) /*GCC*/ || defined(_M_X64) /*MSVC*/
#if defined(_MSC_VER)
#pragma intrinsic(_mm_pause)
#endif
#endif
    void lock() noexcept {
        for (;;) {
            // Optimistically assume the lock is free on the first try
            if (!lock_.exchange(true, std::memory_order_acquire)) {
                return;
            }
            // Wait for lock to be released without generating cache misses
            while (lock_.load(std::memory_order_relaxed)) {
                NREND_HARDWARE_PAUSE();
            }
        }
    }

    bool try_lock() noexcept {
        // First do a relaxed load to check if lock is free in order to prevent
        // unnecessary cache misses if someone does while(!try_lock())
        return !lock_.load(std::memory_order_relaxed) && !lock_.exchange(true, std::memory_order_acquire);
    }

    void unlock() noexcept {
        lock_.store(false, std::memory_order_release);
    }

private:
    std::atomic<bool> lock_ = {0};
};

} // namespace nrend
