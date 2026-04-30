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

#ifdef __CUDACC__

template <unsigned int N = 16>
static __host__ __device__ __inline__ unsigned int tea(unsigned int val0, unsigned int val1) {
    unsigned int v0 = val0;
    unsigned int v1 = val1;
    unsigned int s0 = 0;

    for (unsigned int n = 0; n < N; n++) {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }

    return v0;
}

// Generate random unsigned int in [0, 2^24)
static __host__ __device__ __inline__ unsigned int lcg(unsigned int& prev) {
    const unsigned int LCG_A = 1664525u;
    const unsigned int LCG_C = 1013904223u;
    prev                     = (LCG_A * prev + LCG_C);
    return prev & 0x00FFFFFF;
}

static __host__ __device__ __inline__ unsigned int lcg2(unsigned int& prev) {
    prev = (prev * 8121 + 28411) % 134456;
    return prev;
}

// Generate random float in [0, 1)
static __host__ __device__ __inline__ float rnd(unsigned int& prev) {
    return ((float)lcg(prev) / (float)0x01000000);
}

#endif