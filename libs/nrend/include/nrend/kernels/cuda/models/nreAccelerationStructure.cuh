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

#include <nrend/kernels/cuda/models/nreInstancesExtent.cuh>

#include <nrend/kernels/cuda/common/nreStdUtils.cuh>
#include <nrend/kernels/cuda/common/random.cuh>
#include <nrend/renderer/renderParameters.h>

#include <tiny-cuda-nn/bounding_box.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/vec.h>
struct NREDenseObjectAccStructureDefaultParams {
    static constexpr bool stratifiedSampling     = true;
    static constexpr bool singleJitter           = false;
    static constexpr float uniformStepSize       = 0.05f;
    static constexpr float alphaInvarianceOffset = 1.0f;
};

template <typename TInstanceExtent,
          typename Params = NREDenseObjectAccStructureDefaultParams>
struct NREDenseObjectAccStructure : Params {

    inline __device__ tcnn::vec2 initialize(const tcnn::vec3& rayOrigin,
                                            const tcnn::vec3& rayDirection,
                                            nrend::TTimestamp /*rayTimestamp*/,
                                            const tcnn::vec2& rayMinMax,
                                            uint32_t& rndSeed,
                                            nrend::MemoryHandles /*parameters*/,
                                            uint16_t instanceId) const {

        // intersection with object AABB
        const tcnn::vec3 instanceHalfExtent = 0.5f * TInstanceExtent::fetchExpandedExtent(instanceId);
        tcnn::vec2 intersectMinMax          = tcnn::BoundingBox(-instanceHalfExtent, instanceHalfExtent).ray_intersect(rayOrigin, rayDirection);
        const float initialStep             = fminf(Params::uniformStepSize + jitter(rndSeed, true), intersectMinMax.y - intersectMinMax.x);
        intersectMinMax.x                   = fmaxf(rayMinMax.x, intersectMinMax.x + 0.5f * initialStep);
        intersectMinMax.y                   = fminf(rayMinMax.y, intersectMinMax.y + 0.5f * Params::uniformStepSize);

        return intersectMinMax;
    }

    inline __device__ tcnn::vec2 step(const tcnn::vec3& rayOrigin,
                                      const tcnn::vec3& rayDirection,
                                      const tcnn::vec2& rayMinMax,
                                      uint32_t& rndSeed,
                                      nrend::MemoryHandles /*parameters*/) const {

        return tcnn::vec2{rayMinMax.x, fminf(Params::uniformStepSize + jitter(rndSeed, false), rayMinMax.y - rayMinMax.x)};
    }

    inline __device__ const tcnn::vec4 contract_position(const tcnn::vec3& position, uint16_t instanceId) const {
        return tcnn::vec4(TInstanceExtent::contract_position(position, instanceId), Params::alphaInvarianceOffset /*-logf(instanceMaxExtent)*/);
    }

private:
    static inline __device__ float jitter(uint32_t& rndSeed, bool jitterInit) {
        if (Params::stratifiedSampling && (!Params::singleJitter || jitterInit)) {
            return Params::uniformStepSize * rnd(rndSeed);
        } else {
            return 0.0f;
        }
    }
};

enum class NREContractionType : int {
    NoContraction,
    DegreeInf,
    Degree2,
    Merf
};

struct NREAccNeRFAccAccStructureDefaultParams {
    static constexpr NREContractionType contractionType = NREContractionType::NoContraction;

    static constexpr float alphaInvarianceOffset = 1.0f;
    static constexpr float aabbScale             = 1.0f;
    static constexpr bool stratifiedSampling     = true;

    static constexpr int occupancyBitfieldBufferIdx = 0;

    static constexpr uint32_t gridResolution = 128;
    static constexpr uint32_t gridNumCells   = gridResolution * gridResolution * gridResolution;
    static constexpr uint32_t minCascade     = 0;
    static constexpr uint32_t maxCascade     = 4;

    static constexpr float coneAngle = 1.0f;
    static constexpr float dtMin     = 1.0f;
    static constexpr float dtMax     = 1.0f;

    static const float aabb[6];
};
const float NREAccNeRFAccAccStructureDefaultParams::aabb[6] = {-1.f, -1.f, -1.f, 1.f, 1.f, 1.f};

template <typename Params = NREAccNeRFAccAccStructureDefaultParams>
struct NREAccNeRFAccAccStructure : Params {

    inline __device__ tcnn::vec2 initialize(const tcnn::vec3& rayOrigin,
                                            const tcnn::vec3& rayDirection,
                                            nrend::TTimestamp /*rayTimestamp*/,
                                            const tcnn::vec2& rayMinMax,
                                            uint32_t& rndSeed,
                                            nrend::MemoryHandles /*parameters*/,
                                            uint16_t /*instanceId*/) const {

        // intersection with AABB
        tcnn::vec2 instersectMinMax = fetchAABB().ray_intersect(rayOrigin, rayDirection);
        instersectMinMax.x          = fmaxf(rayMinMax.x, instersectMinMax.x);
        instersectMinMax.y          = fminf(rayMinMax.y, instersectMinMax.y);

        if (Params::stratifiedSampling) {
            instersectMinMax.x += Params::dtMin * rnd(rndSeed);
        }

        return instersectMinMax;
    }

    inline __device__ tcnn::vec2 step(const tcnn::vec3& rayOrigin,
                                      const tcnn::vec3& rayDirection,
                                      const tcnn::vec2& rayMinMax,
                                      uint32_t& rndSeed,
                                      nrend::MemoryHandles parameters) const {

        const uint8_t* occupancyBitfield = parameters.bufferPtr<uint8_t>(Params::occupancyBitfieldBufferIdx);

        float minT = rayMinMax.x;

        while (true) {
            if (minT >= rayMinMax.y) {
                minT = rayMinMax.y;
                break;
            }

            const tcnn::vec3 pos    = rayOrigin + minT * rayDirection;
            const tcnn::vec3 relPos = pos / Params::aabbScale;

            const int mip = mipFromPos(relPos);

            if (fetchOccupancyBitfield(relPos, occupancyBitfield, mip)) {
                break;
            }

            const float tNext = minT + distanceToNextVoxel(relPos, rayDirection, mip) * Params::aabbScale;
            const float dt    = calcDt(minT);
            minT += dt * tcnn::max(static_cast<int>(((tNext - minT) / dt) + 0.5f), 1);
        }

        return tcnn::vec2{minT, calcDt(minT)};
    }

    template <NREContractionType contractionType, typename nreEnableIf<(contractionType == NREContractionType::NoContraction), bool>::type = true>
    inline __device__ const tcnn::vec3 contract_position_xyz(const tcnn::vec3& position, uint16_t /*instanceId*/) const {
        return fetchAABB().relative_pos(position);
    }

    template <NREContractionType contractionType, typename nreEnableIf<(contractionType == NREContractionType::DegreeInf), bool>::type = true>
    inline __device__ const tcnn::vec3 contract_position_xyz(const tcnn::vec3& position, uint16_t /*instanceId*/) const {
        tcnn::vec3 pos = fetchAABB().relative_pos(position) * 2.f - 1.f; // points in [-1,1]
        float norm     = tcnn::max(tcnn::abs(pos));
        if (norm >= 1.f) {
            pos = (2.f - 1.f / norm) * (pos / norm);
        }
        return pos * 0.25f + 0.5f; // [-inf, inf] is at [0, 1]
    }

    template <NREContractionType contractionType, typename nreEnableIf<(contractionType == NREContractionType::Degree2), bool>::type = true>
    inline __device__ const tcnn::vec3 contract_position_xyz<NREContractionType::Degree2>(const tcnn::vec3& position, uint16_t /*instanceId*/) const {
        tcnn::vec3 pos = fetchAABB().relative_pos(position) * 2.f - 1.f; // points in [-1,1]
        float norm     = tcnn::length(pos);
        if (norm >= 1) {
            pos = (2 - 1 / norm) * (pos / norm);
        }
        return pos * 0.25f + 0.5f; // [-inf, inf] is at [0, 1]
    }

    template <NREContractionType contractionType, typename nreEnableIf<(contractionType == NREContractionType::Merf), bool>::type = true>
    inline __device__ const tcnn::vec3 contract_position_xyz<NREContractionType::Merf>(const tcnn::vec3& position, uint16_t /*instanceId*/) const {
        tcnn::vec3 pos     = fetchAABB().relative_pos(position) * 2.f - 1.f; // points in [-1,1]
        tcnn::vec3 pos_abs = tcnn::abs(pos);
        float norm         = tcnn::max(pos_abs);
        float scale        = std::min(1.f, norm);

#pragma unroll
        for (uint32_t dim = 0; dim < 3; ++dim) {
            if (fabsf(pos_abs[dim] - norm) < 1.0e-7f) {
                pos = (2.f - 1.f / scale) * (pos / scale);
            } else {
                pos = pos / scale;
            }
        }
        return pos * 0.25f + 0.5f; // [-inf, inf] is at [0, 1]
    }

    inline __device__ const tcnn::vec4 contract_position(const tcnn::vec3& position, uint16_t /*instanceId*/ _) const {
        return tcnn::vec4(contract_position_xyz<Params::contractionType>(position, _), Params::alphaInvarianceOffset);
    }

private:
    inline __device__ tcnn::BoundingBox fetchAABB() const {
        return tcnn::BoundingBox{{Params::aabb[0], Params::aabb[1], Params::aabb[2]}, {Params::aabb[3], Params::aabb[4], Params::aabb[5]}};
    }

    static inline __device__ float calcDt(float t) {

        return tcnn::clamp(t * Params::coneAngle, Params::dtMin, Params::dtMax);
    }

    static inline __device__ int mipFromPos(const tcnn::vec3& pos) {

        int exponent;
        // frexpf(x) * 2^(exponent) < x <= frexpf(x) * 2^(exponent-1)
        frexpf(tcnn::max(tcnn::abs(pos)), &exponent);
        return tcnn::clamp<int>(exponent + 1, Params::minCascade, Params::maxCascade);
    }

    static inline __device__ float distanceToNextVoxel(const tcnn::vec3& pos, const tcnn::vec3& dir, int mip) {

        const float res    = scalbnf(Params::gridResolution, -mip);                 ///< res = resolution / 2^mip
        const tcnn::vec3 p = pos * res + tcnn::vec3(0.5f * Params::gridResolution); ///< 2^(mip-1) > pos > -2^(mip-1)  =>
                                                                                    ///< resolution > p > 0
        // dda like step
        float tx = (floorf(p.x + 0.5f + 0.5f * sign(dir.x)) - p.x) / dir.x;
        float ty = (floorf(p.y + 0.5f + 0.5f * sign(dir.y)) - p.y) / dir.y;
        float tz = (floorf(p.z + 0.5f + 0.5f * sign(dir.z)) - p.z) / dir.z;
        float t  = tcnn::min(tcnn::min(tx, ty), tz);

        return fmaxf(t / res, 0.0f) + 1e-06f;
    }

    static inline __device__ uint32_t cascadedGridIdxAt(const tcnn::vec3& pos, int mip) {

        // 2^(mip-1) > pos > -2^(mip-1)  => resolution / 2 > pos * scalbnf(resolution, -mip) > -resolution / 2
        const tcnn::vec3 i =
            pos * scalbnf(Params::gridResolution, -mip) + tcnn::vec3(0.5f * Params::gridResolution);
        if (i.x < 0 || i.x >= Params::gridResolution || i.y < 0 || i.y >= Params::gridResolution || i.z < 0 ||
            i.z >= Params::gridResolution) {
            return 0xFFFFFFFF;
        }
        return tcnn::morton3D(i.x, i.y, i.z);
    }

    static inline __device__ bool fetchOccupancyBitfield(const tcnn::vec3& pos,
                                                         const uint8_t* __restrict__ occupancyBitfield,
                                                         int mip) {

        uint32_t idx = cascadedGridIdxAt(pos, mip);
        if (idx == 0xFFFFFFFF) {
            return false;
        }
        return occupancyBitfield[idx / 8 + (Params::gridNumCells * mip) / 8] & (1 << (idx % 8));
    }
};
