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

template <int OutputDim>
struct NRESkipFeatureVolume {
    static inline __device__ auto eval(const tcnn::vec<3>&, nrend::TTimestamp, uint16_t, nrend::MemoryHandles) {
        return tcnn::vec<OutputDim>::zero();
    }
};

struct NREHashGridFeatureVolumeLevelEncoding {
    static constexpr int Dim = 1;

    static inline __device__ tcnn::vec<1> eval() {
        return tcnn::vec<1>{encoding};
    }

protected:
    static constexpr float encoding = 0.0f;
};

template <int InputDim, int OutputDim>
struct NREHashGridFeatureVolumeEvaluator {
protected:
    __device__ tcnn::vec<OutputDim> _eval(const tcnn::vec<InputDim>& input,
                                          const __half* __restrict__,
                                          uint8_t* __restrict__) {
        return tcnn::vec<OutputDim>::zero();
    }
};
template <typename TLevelEncoding = NREHashGridFeatureVolumeLevelEncoding, typename TEvaluator = NREHashGridFeatureVolumeEvaluator<3, 3>, int FirstBufferIdx = 0>
struct NREHashGridFeatureVolume : TEvaluator {
    static constexpr int InputDim = 3 + NREHashGridFeatureVolumeLevelEncoding::Dim;

    __device__ auto eval(const tcnn::vec<3>& normalizedPosition, nrend::TTimestamp, uint16_t /*instanceId*/, nrend::MemoryHandles parameters) {
        tcnn::vec<InputDim> input;
        nrend::copyVec<InputDim, 3>(input, normalizedPosition);
        if (NREHashGridFeatureVolumeLevelEncoding::Dim > 1) {
            nrend::copyVec<InputDim, NREHashGridFeatureVolumeLevelEncoding::Dim, 3>(input, TLevelEncoding::eval());
        }
        return TEvaluator::_eval(
            input, parameters.bufferPtr<const __half>(FirstBufferIdx), nullptr);
    }
};

template <typename TInstanceEmbedding, typename TDummyTimeEmbedding, typename TLevelEncoding, typename TEvaluator, int FirstBufferIdx = 0>
struct NREHashGridObjectFeatureVolume : TEvaluator {
    static constexpr int InputDim = 3 + TInstanceEmbedding::Dim + TLevelEncoding::Dim;

    __device__ auto eval(const tcnn::vec<3>& normalizedPosition, nrend::TTimestamp, uint16_t instanceId, nrend::MemoryHandles parameters) {
        tcnn::vec<InputDim> input;
        nrend::copyVec<InputDim, 3>(input, tcnn::clamp(normalizedPosition, 0.f, 1.f));
        nrend::copyVec<InputDim, TInstanceEmbedding::Dim, 3>(input, TInstanceEmbedding::eval(instanceId, parameters));
        if (NREHashGridFeatureVolumeLevelEncoding::Dim > 0) {
            nrend::copyVec<InputDim, NREHashGridFeatureVolumeLevelEncoding::Dim, 3 + TInstanceEmbedding::Dim>(input, TLevelEncoding::eval());
        }
        return TEvaluator::_eval(input, parameters.bufferPtr<const __half>(FirstBufferIdx), nullptr);
    }
};

template <typename TInstanceEmbedding, typename TTimeEmbedding, typename TLevelEncoding, typename TEvaluator, int FirstBufferIdx = 0>
struct NREHashGridTimedObjectFeatureVolume : TEvaluator {
    static constexpr int InputDim = 3 + TInstanceEmbedding::Dim + TTimeEmbedding::Dim + TLevelEncoding::Dim;

    __device__ auto eval(const tcnn::vec<3>& normalizedPosition, nrend::TTimestamp timestamp, uint16_t instanceId, nrend::MemoryHandles parameters) {
        tcnn::vec<InputDim> input;
        nrend::copyVec<InputDim, 3>(input, tcnn::clamp(normalizedPosition, 0.f, 1.f));
        nrend::copyVec<InputDim, TInstanceEmbedding::Dim, 3>(input, TInstanceEmbedding::eval(instanceId, parameters));
        nrend::copyVec<InputDim, TTimeEmbedding::Dim, TInstanceEmbedding::Dim + 3>(input, TTimeEmbedding::eval(instanceId, timestamp, parameters));
        if (NREHashGridFeatureVolumeLevelEncoding::Dim > 0) {
            nrend::copyVec<InputDim, NREHashGridFeatureVolumeLevelEncoding::Dim, 3 + TInstanceEmbedding::Dim + TTimeEmbedding::Dim>(input, TLevelEncoding::eval());
        }
        return TEvaluator::_eval(input, parameters.bufferPtr<const __half>(FirstBufferIdx), nullptr);
    }
};
