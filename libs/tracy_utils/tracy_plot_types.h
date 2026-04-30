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

// Centralized plot type definitions for Tracy profiling
// This header is shared between C++ and Python bindings

enum PlotType {
    // GPU metrics
    PLOT_GPU_MEM_ALLOCATED_MB = 0,
    PLOT_GPU_MEM_RESERVED_MB,
    PLOT_GPU_MEM_DURING_RENDER_MB,

    // System metrics
    PLOT_CPU_MEMORY_MB,

    // Always keep this last
    PLOT_TYPE_COUNT
};

// Compile-time string names for each plot type
// Must match the order of PlotType enum exactly
constexpr const char* PLOT_NAMES[PLOT_TYPE_COUNT] = {
    "gpu_memory_allocated_mb", // PLOT_GPU_MEM_ALLOCATED_MB
    "gpu_memory_reserved_mb",  // PLOT_GPU_MEM_RESERVED_MB
    "gpu_memory_render_mb",    // PLOT_GPU_MEM_DURING_RENDER_MB
    "cpu_memory_mb"            // PLOT_CPU_MEMORY_MB
};

// Helper to validate enum values
static_assert(PLOT_TYPE_COUNT == 4, "Update PLOT_NAMES array when adding/removing plot types");
