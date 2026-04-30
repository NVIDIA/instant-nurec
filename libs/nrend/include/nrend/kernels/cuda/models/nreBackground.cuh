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

struct NRESkipBackground {
    static constexpr bool Enabled           = false;
    static constexpr bool RequireThreadSync = false;

    template <typename TAppearanceEmbedding>
    inline __device__ tcnn::vec3 eval(const tcnn::vec3&, nrend::TTimestamp, const TAppearanceEmbedding&, nrend::MemoryHandles) const {
        return tcnn::vec3::zero();
    }
};

struct NREColorBackgroundDefaultParams {
    const tcnn::vec3 color = {0.f, 0.f, 0.f};
};

template <typename TParams = NREColorBackgroundDefaultParams>
struct NREColorBackground : public TParams {
    static constexpr bool Enabled           = true;
    static constexpr bool RequireThreadSync = false;

    template <typename TAppearanceEmbedding>
    inline __device__ tcnn::vec3 eval(const tcnn::vec3&, nrend::TTimestamp, const TAppearanceEmbedding&, nrend::MemoryHandles) {
        return TParams::color;
    }
};

struct NREEnvMapBackgroundDefaultParams {
    static constexpr bool IsCubeMap         = true;
    static constexpr int TextureHandleIndex = 0;
};

template <typename TParams = NREEnvMapBackgroundDefaultParams>
struct NREEnvMapBackground : public TParams {
    static constexpr bool Enabled           = true;
    static constexpr bool RequireThreadSync = false;

    template <typename TAppearanceEmbedding>
    inline __device__ tcnn::vec3 eval(const tcnn::vec3& direction, nrend::TTimestamp, const TAppearanceEmbedding&, nrend::MemoryHandles parameters) {
        // NRE: x forward, y left, z up
        // OpenGL: x right, y up, z back
        const auto dir          = tcnn::vec3{-direction.y, direction.z, -direction.x};
        cudaTextureObject_t tex = (cudaTextureObject_t)parameters.handles[TParams::TextureHandleIndex];
        if constexpr (TParams::IsCubeMap) {
            const auto C = texCubemap<float4>(tex, dir.x, dir.y, dir.z);
            return TParams::SaturateRadiance ? tcnn::clamp(tcnn::vec3{C.x, C.y, C.z}, 0.f, 1.f) : tcnn::max(tcnn::vec3{C.x, C.y, C.z}, 0.f);
        } else {
            const auto uv = _cartesianToUV(dir);
            const auto C  = tex2D<float4>(tex, uv.x, uv.y);
            return TParams::SaturateRadiance ? tcnn::clamp(tcnn::vec3{C.x, C.y, C.z}, 0.f, 1.f) : tcnn::max(tcnn::vec3{C.x, C.y, C.z}, 0.f);
        }
    }
    inline __device__ tcnn::vec3 _cartesianToUV(const tcnn::vec3& v) {
        const float r     = tcnn::length(v);
        const float phi   = acosf(v.z / r);
        const float theta = atan2f(v.x, v.y);
        return tcnn::vec2{phi / tcnn::PI() * 2.f - 1.f, theta / tcnn::PI()};
    }
};

template <int InputDim, int EncodingDim, int OutputDim>
struct NRESkyMLPBackgroundDefaultEvaluator {
protected:
    __device__ tcnn::vec<EncodingDim> _evalEncoding(const tcnn::vec<InputDim>& input, const __half*) {
        return tcnn::vec<EncodingDim>::zero();
    }
    __device__ tcnn::vec<OutputDim> _evalNetwork(const tcnn::vec<EncodingDim>& input,
                                                 const __half* __restrict__,
                                                 uint8_t* __restrict__) {
        return tcnn::vec<OutputDim>::zero();
    }
};
template <int EncodingDim,
          int AppearanceEmbeddingDim,
          typename TEvaluator   = NRESkyMLPBackgroundDefaultEvaluator<3 + AppearanceEmbeddingDim, 16, 3>,
          int EncodingBufferIdx = 0,
          int NetworkBufferIdx  = EncodingBufferIdx + 1>
struct NRESkyMLPBackground : TEvaluator {
    static constexpr bool Enabled           = true;
    static constexpr bool RequireThreadSync = true;

    template <typename TAppearanceEmbedding>
    __device__ auto eval(const tcnn::vec3& direction,
                         nrend::TTimestamp,
                         const TAppearanceEmbedding& appearanceEmbedding,
                         nrend::MemoryHandles parameters) {

        static_assert(AppearanceEmbeddingDim == TAppearanceEmbedding::Dim, "AppearanceEmbedding dimensions mismatch.");

        const tcnn::vec<EncodingDim> encodingVec = TEvaluator::_evalEncoding(
            (direction + 1.0f) * 0.5f, parameters.bufferPtr<const __half>(EncodingBufferIdx), nullptr);

        tcnn::vec<EncodingDim + AppearanceEmbeddingDim> networkInputVec;
        nrend::copyVec<EncodingDim + AppearanceEmbeddingDim, EncodingDim>(networkInputVec, encodingVec);
        if (AppearanceEmbeddingDim) {
            appearanceEmbedding.fetch<EncodingDim + AppearanceEmbeddingDim, EncodingDim>(networkInputVec);
        }

        return TEvaluator::_evalNetwork(
            networkInputVec, parameters.bufferPtr<const __half>(NetworkBufferIdx), nullptr);
    }
};
