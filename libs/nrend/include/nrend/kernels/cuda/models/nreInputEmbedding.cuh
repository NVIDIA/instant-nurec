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

#include <nrend/renderer/renderParameters.h>
#include <nrend/utils/nreVec.h>

template <int NumEmbeddings, int EmbeddingDims, int FirstBufferIdx>
struct NREWeightedInstanceInputEmbedding {

    static constexpr int Dim = EmbeddingDims;

    static inline __device__ tcnn::vec<EmbeddingDims> eval(uint32_t idx, nrend::MemoryHandles parameters) {
        return parameters.bufferPtr<const tcnn::tvec<__half, EmbeddingDims>>(FirstBufferIdx)[tcnn::clamp<uint32_t>(idx, 0, NumEmbeddings - 1)];
    }
};

struct NREIndividualRemapTimeInputEmbeddingParams {
    static constexpr float remapMin   = 0.0f;
    static constexpr float remapRange = 1.0f;
};

template <int NumEmbeddings, int FirstBufferIdx, typename TParams = NREIndividualRemapTimeInputEmbeddingParams>
struct NREIndividualRemapTimeInputEmbedding {

    static constexpr int Dim = 1;

    template <typename TVal>
    static inline __device__ tcnn::vec<Dim> eval(uint32_t idx, TVal val, nrend::MemoryHandles parameters) {
        using TValRange = tcnn::tvec<TVal, 2>;

        auto valRange = parameters.bufferPtr<const TValRange>(FirstBufferIdx)[tcnn::clamp<uint32_t>(idx, 0, NumEmbeddings - 1)];

        const float ratio = static_cast<double>(val - valRange[0]) / static_cast<double>(valRange[1] - valRange[0]);
        return tcnn::vec<Dim>{tcnn::clamp<float>(ratio, 0, 1) * TParams::remapRange + TParams::remapMin};
    }
};
