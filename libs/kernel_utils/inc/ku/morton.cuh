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

// morton utils
inline __host__ __device__ uint32_t __expand_bits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

inline __host__ __device__ uint32_t __morton3D(uint32_t x, uint32_t y, uint32_t z) {
    uint32_t xx = __expand_bits(x);
    uint32_t yy = __expand_bits(y);
    uint32_t zz = __expand_bits(z);
    return xx | (yy << 1) | (zz << 2);
}

inline __host__ __device__ uint32_t __morton3D_invert(uint32_t x) {
    x = x & 0x49249249;
    x = (x | (x >> 2)) & 0xc30c30c3;
    x = (x | (x >> 4)) & 0x0f00f00f;
    x = (x | (x >> 8)) & 0xff0000ff;
    x = (x | (x >> 16)) & 0x0000ffff;
    return x;
}
