// SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#pragma once

#include "helper_math.cuh"

template <typename idx_t>
inline __host__ __device__ idx_t div_round_up(idx_t val, idx_t divisor) {
    return (val + divisor - 1) / divisor;
}

inline __host__ __device__ unsigned int nextPowerOfTwo_unsafe(unsigned int v) {
    // Compute the next higher power-of-two for a given integer up to 2^31.
    // Unsafe as 2^32 can't be represented with unsigned int, in which case an overflow of 0 will be returned

    v += (v == 0);

    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;

    return v;
}

inline __host__ __device__ unsigned int nextPowerOfTwoCapped(unsigned int v, unsigned int max) {
    // Compute the next higher power-of-two for a given integer capped by a maximum value.
    // The maximum value will also be returned if the next power of two doesn't fit into
    // the range of values of unsigned integers

    v = nextPowerOfTwo_unsafe(v);

    if (v >= max || v == 0)
        return max;

    return v;
}
