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

#include <nrend/renderer/renderParameters.h> //< TTimestamp
#include <nrend/utils/nreVec.h>

struct NRESkipAppearanceEmbedding {
    static constexpr int Dim = 0;

    inline __device__ void eval(nrend::TTimestamp, nrend::MemoryHandles, uint32_t, const tcnn::ivec2*) {
    }

    template <int N, int O>
    inline __device__ void fetch(tcnn::vec<N>& vec) const {
    }
};

template <int EmbeddingDim>
struct NREZeroAppearanceEmbedding {
    static constexpr int Dim = EmbeddingDim;

    inline __device__ void eval(nrend::TTimestamp, nrend::MemoryHandles, uint32_t, const tcnn::ivec2*) {
    }

    template <int N, int O>
    inline __device__ void fetch(tcnn::vec<N>& inputVec) const {
        nrend::copyVec<N, Dim, O>(inputVec, tcnn::vec<Dim>::zero());
    }
};

template <int EmbeddingDim, int BufferIdx>
struct NRECachedAppearanceEmbedding {
    static constexpr int Dim = EmbeddingDim;

    tcnn::vec<Dim> m_embedding;

    inline __device__ void eval(nrend::TTimestamp, nrend::MemoryHandles parameters, uint32_t, const tcnn::ivec2*) {
        m_embedding = *parameters.bufferPtr<const tcnn::tvec<__half, Dim>>(BufferIdx);
    }

    template <int N, int O>
    inline __device__ void fetch(tcnn::vec<N>& inputVec) const {
        nrend::copyVec<N, Dim, O>(inputVec, m_embedding);
    }
};

// draft implementation of indexable cache for per-element appareance embedding mode
template <int EmbeddingDim, int NumEmbeddings, int BufferIdx, int RayEmbeddingIdx>
struct NREIndexableCachedAppearanceEmbedding {
    static constexpr int Dim = EmbeddingDim;

    tcnn::vec<Dim> m_embedding;

    inline __device__ void eval(nrend::TTimestamp, nrend::MemoryHandles parameters, uint32_t /*rayIdx*/, const tcnn::ivec2* __restrict__ rayEmbeddingIdxPtr) {
        const int index = tcnn::clamp(rayEmbeddingIdxPtr ? rayEmbeddingIdxPtr[0][RayEmbeddingIdx] : 0, 0, NumEmbeddings - 1);
        m_embedding     = parameters.bufferPtr<const tcnn::tvec<__half, Dim>>(BufferIdx)[index];
    }

    template <int N, int O>
    inline __device__ void fetch(tcnn::vec<N>& inputVec) const {
        nrend::copyVec<N, Dim, O>(inputVec, m_embedding);
    }
};

template <typename TCameraCachedEmbedding, typename TFrameCachedEmbedding>
struct NREGloAppearanceEmbedding : TCameraCachedEmbedding, TFrameCachedEmbedding {
    static constexpr int Dim = TCameraCachedEmbedding::Dim + TFrameCachedEmbedding::Dim;

    inline __device__ void eval(nrend::TTimestamp ts, nrend::MemoryHandles parameters, uint32_t rayIdx, const tcnn::ivec2* __restrict__ rayEmbeddingIdxPtr) {
        TCameraCachedEmbedding::eval(ts, parameters, rayIdx, rayEmbeddingIdxPtr);
        TFrameCachedEmbedding::eval(ts, parameters, rayIdx, rayEmbeddingIdxPtr);
    }

    template <int N, int O>
    inline __device__ void fetch(tcnn::vec<N>& inputVec) const {
        TCameraCachedEmbedding::fetch<N, O>(inputVec);
        TFrameCachedEmbedding::fetch<N, O + TCameraCachedEmbedding::Dim>(inputVec);
    }
};
