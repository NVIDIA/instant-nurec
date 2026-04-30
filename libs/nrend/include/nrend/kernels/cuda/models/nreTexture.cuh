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

template <int InputDim, int OutputDim>
struct NRESkipTexture {
    template <typename TAppearanceEmbedding>
    __device__ auto eval(const tcnn::vec<InputDim>&,
                         const TAppearanceEmbedding&,
                         const tcnn::vec3&,
                         const tcnn::vec3&,
                         nrend::MemoryHandles) {
        return tcnn::vec<OutputDim>::zero();
    }
};

template <int InputDim, int OutputDim>
struct NREFullyFusedTextureEvaluator {
protected:
    __device__ tcnn::vec<OutputDim> _eval(const tcnn::vec<InputDim>& input,
                                          const __half* __restrict__,
                                          uint8_t* __restrict__) {
        return tcnn::vec<OutputDim>::zero();
    }
};

template <int FeatureAndEmbeddingDim,
          int PositionDim,
          typename TEvaluator = NREFullyFusedTextureEvaluator<FeatureAndEmbeddingDim + PositionDim + 3, 3>,
          int FirstBufferIdx  = 0>
struct NREFullyFusedTexture : TEvaluator {

    template <typename TAppearanceEmbedding>
    __device__ auto eval(const tcnn::vec<FeatureAndEmbeddingDim - TAppearanceEmbedding::Dim>& features,
                         const TAppearanceEmbedding& appearanceEmbedding,
                         const tcnn::vec3& normalizedPosition,
                         const tcnn::vec3& direction,
                         nrend::MemoryHandles parameters) {

        constexpr int InputDim   = FeatureAndEmbeddingDim + PositionDim + 3;
        constexpr int FeatureDim = FeatureAndEmbeddingDim - TAppearanceEmbedding::Dim;

        tcnn::vec<InputDim> evalInputVec;
        nrend::copyVec<InputDim, FeatureDim>(evalInputVec, features);
        appearanceEmbedding.fetch<InputDim, FeatureDim>(evalInputVec);
        if (PositionDim > 0) {
            nrend::copyVec<InputDim, 3, FeatureAndEmbeddingDim, PositionDim>(evalInputVec, 2.0f * normalizedPosition - tcnn::vec3{1.0f});
        }
        nrend::copyVec<InputDim, 3, FeatureAndEmbeddingDim + PositionDim>(evalInputVec, (direction + 1.0f) * 0.5f);

        return TEvaluator::_eval(evalInputVec, parameters.bufferPtr<const __half>(FirstBufferIdx), nullptr);
    }
};
