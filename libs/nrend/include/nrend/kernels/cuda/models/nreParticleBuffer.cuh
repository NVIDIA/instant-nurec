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

/**
 * @file nreParticleBuffer.cuh
 * @brief Shared buffer classes for particle model components
 *
 * Provides common buffer management for particle data with support for:
 * - Forward-only buffers (ptr only)
 * - Differentiable buffers (ptr + gradPtr)
 * - Optional buffers (enabled/disabled via template parameter)
 * - Memory-efficient conditional compilation
 *
 * ASCII Schematic - Buffer Layout:
 *   Non-Differentiable          Differentiable
 *   ┌─────────────────┐         ┌─────────────────┐
 *   │ TBuffer* ptr    │         │ TBuffer* ptr    │
 *   └─────────────────┘         │ TBuffer* gradPtr│
 *                               └─────────────────┘
 */

template <typename TBuffer, bool TDifferentiable>
struct NREParticleBuffer {
    TBuffer* ptr = nullptr;
};

template <typename TBuffer>
struct NREParticleBuffer<TBuffer, true> {
    TBuffer* ptr     = nullptr;
    TBuffer* gradPtr = nullptr;
};

template <typename TBuffer, bool TDifferentiable, bool Enabled>
struct NREParticleOptionalBuffer {
    // Empty struct when disabled - optimizes memory usage
};

template <typename TBuffer, bool TDifferentiable>
struct NREParticleOptionalBuffer<TBuffer, TDifferentiable, true> : NREParticleBuffer<TBuffer, TDifferentiable> {
    // Inherits buffer functionality when enabled
};
