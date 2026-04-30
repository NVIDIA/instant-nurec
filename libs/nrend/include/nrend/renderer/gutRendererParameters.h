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

namespace nrend {
namespace GUTParameters {

static constexpr uint32_t LinearLaunchSize   = 256U;
static constexpr uint32_t InvalidTileIdx     = -1U;
static constexpr uint32_t InvalidParticleIdx = -1U;
static constexpr uint32_t WarpSize           = 32;
static constexpr uint32_t WarpMask           = 0xFFFFFFFFU;

struct DefaultTiling {
    static constexpr uint32_t BlockX            = 16;
    static constexpr uint32_t BlockY            = 16;
    static constexpr uint32_t BlockSize         = BlockX * BlockY;
    static constexpr uint32_t NumWarps          = BlockSize / nrend::GUTParameters::WarpSize;
    static constexpr bool EnableRayBasedCulling = false;
};

} // namespace GUTParameters
} // namespace nrend
