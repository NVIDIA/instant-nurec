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

#include <tiny-cuda-nn/common.h>

namespace nrend {

// using SNIFAE to be noop when copying a NULL size vector
template <int toVecSize,
          int fromVecSize,
          int toVecCopyOffset    = 0,
          int toVecCopySize      = fromVecSize,
          std::enable_if_t<(toVecCopySize == 0) && (toVecCopyOffset + toVecCopySize <= toVecSize) && (fromVecSize > toVecCopySize),
                           bool> = true>
inline TCNN_HOST_DEVICE void copyVec(tcnn::vec<toVecSize>& toVec, const tcnn::vec<fromVecSize>& fromVec) {
}

template <int toVecSize,
          int fromVecSize,
          int toVecCopyOffset    = 0,
          int toVecCopySize      = fromVecSize,
          std::enable_if_t<(toVecCopySize > 0) && (toVecCopyOffset + toVecCopySize <= toVecSize) && (fromVecSize == toVecCopySize),
                           bool> = true>
inline TCNN_HOST_DEVICE void copyVec(tcnn::vec<toVecSize>& toVec, const tcnn::vec<fromVecSize>& fromVec) {
    // FIXME :: this does not compile
    // toVec.slice<toVecCopyOffset, toVecCopySize>() = fromVec;
    //
    *(tcnn::vec<toVecCopySize>*)(toVec.data() + toVecCopyOffset) = fromVec;
}

// tcnn::vec::slice method is not compiling
template <uint32_t Offset, uint32_t OutSize, typename T, uint32_t InSize, size_t A>
inline TCNN_HOST_DEVICE tcnn::tvec<T, OutSize, A>& sliceVec(const tcnn::tvec<T, InSize, A>& vec) {
    return *(tcnn::tvec<T, OutSize, A>*)(vec.data() + Offset);
}

template <typename T, int N, int M>
inline TCNN_HOST_DEVICE tcnn::tvec<T, N> mul(const tcnn::tvec<T, M>& vec, const tcnn::tmat<T, N, M>& mat) {
    tcnn::tvec<T, N> ret;
    TCNN_PRAGMA_UNROLL
    for (int i = 0; i < M; ++i) {
        ret[i] = tcnn::dot(vec, mat[i]);
    }
    return ret;
}

template <typename T>
inline TCNN_HOST_DEVICE tcnn::tquat<T> mul(const tcnn::tquat<T>& quat1, const tcnn::tquat<T>& quat2) {
    return tcnn::tquat<T>(
        quat1.w * quat2.w - quat1.x * quat2.x - quat1.y * quat2.y - quat1.z * quat2.z,
        quat1.w * quat2.x + quat1.x * quat2.w + quat1.y * quat2.z - quat1.z * quat2.y,
        quat1.w * quat2.y - quat1.x * quat2.z + quat1.y * quat2.w + quat1.z * quat2.x,
        quat1.w * quat2.z + quat1.x * quat2.y - quat1.y * quat2.x + quat1.z * quat2.w);
}

template <int N, typename T, bool Valid>
struct OptionalVec {
    tcnn::tvec<T, N> vec;

    inline TCNN_HOST_DEVICE tcnn::tvec<T, N>* ptr() {
        static_assert(Valid, "OptionalVec is not valid");
        return &vec;
    }

    inline TCNN_HOST_DEVICE const tcnn::tvec<T, N>* ptr() const {
        static_assert(Valid, "OptionalVec is not valid");
        return &vec;
    }
};

template <int N, typename T>
struct OptionalVec<N, T, false> {
    static inline TCNN_HOST_DEVICE tcnn::tvec<T, N>* ptr() {
        return nullptr;
    }
};

// TODO : WSOReduce optimization
template <int N, typename T, bool Atomic = true, uint32_t WarpSize = 32>
inline TCNN_HOST_DEVICE void reduceWarpSumToBuffer(tcnn::tvec<T, N>& vec,
                                                   tcnn::tvec<T, N>* bufferPtr,
                                                   uint32_t tileThreadIdx) {
    const uint32_t warpMask = 0xFFFFFFFF;
#pragma unroll
    for (uint32_t offset = WarpSize >> 1; offset > 0; offset >>= 1) {
#pragma unroll
        for (int i = 0; i < N; ++i) {
            vec[i] += __shfl_down_sync(warpMask, vec[i], offset);
        }
    }
    if ((tileThreadIdx & (WarpSize - 1)) == 0) {
        if constexpr (Atomic) {
#pragma unroll
            for (int i = 0; i < N; i++) {
                atomicAdd(&((*bufferPtr)[i]), vec[i]);
            }
        } else {
            for (int i = 0; i < N; i++) {
                (*bufferPtr)[i] += vec[i];
            }
        }
    }
}

#ifdef __CUDACC__
// Specialized version for single scalar value with active mask
template <typename T, bool Atomic = true, uint32_t WarpSize = 32>
inline TCNN_DEVICE void reduceActiveWarpSumToBufferScalar(T& value,
                                                          T* bufferPtr,
                                                          uint32_t tileThreadIdx) {
    const uint32_t warpMask = __activemask();
    // Find lowest active lane
    const int leaderLane = __ffs(warpMask) - 1; // first set bit (0-indexed)

    // Warp reduction
    for (uint32_t offset = WarpSize >> 1; offset > 0; offset >>= 1) {
        T other           = __shfl_down_sync(warpMask, value, offset);
        const int srcLane = (tileThreadIdx & (WarpSize - 1)) + offset;
        if ((warpMask >> srcLane) & 1) { // source lane is active
            value += other;
        }
    }

    // Leader lane (lowest active) writes result
    if ((tileThreadIdx & (WarpSize - 1)) == leaderLane) {
        if constexpr (Atomic) {
            atomicAdd(bufferPtr, value);
        } else {
            *bufferPtr += value;
        }
    }
}
#endif // __CUDACC__

} // namespace nrend