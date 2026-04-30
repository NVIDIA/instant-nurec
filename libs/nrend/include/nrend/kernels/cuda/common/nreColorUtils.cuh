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

inline __device__ float srgbToLinear(float srgb) {
    return (srgb <= 0.04045f) ? srgb / 12.92f : powf((srgb + 0.055f) / 1.055f, 2.4f);
}

inline __device__ float linearToSrgb(float linear) {
    return (linear < 0.0031308f) ? 12.92f * linear : 1.055f * powf(linear, 0.41666f) - 0.055f;
}
