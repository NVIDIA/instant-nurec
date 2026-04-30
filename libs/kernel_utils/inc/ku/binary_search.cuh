// SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#pragma once

template <typename scalar_t, typename idx_t>
inline __device__ idx_t binary_search_unsafe(scalar_t val, scalar_t const* data, idx_t length) {
    // Borrowed from intant-ngp
    // Returns "right" bound index of the found interval.
    // (None or data[return-1]) <= val < data[return]
    // Allows val less than the minimum data.
    // Disallows val larger than the maximum data. (will return wrong index)
    // Returns zero if length <= 0
    idx_t it;
    idx_t count, step;
    count = length;

    idx_t first = 0;
    while (count > 0) {
        it   = first;
        step = count / 2;
        it += step;
        if (data[it] < val) {
            first = ++it;
            count -= step + 1;
        } else {
            count = step;
        }
    }
    return first;
}

template <typename scalar_t, typename idx_t>
inline __device__ idx_t binary_search(scalar_t val, scalar_t const* data, idx_t length) {
    // Returns the upper / "right" bound index such that
    //
    // (None or data[returned-index-1]) <= val < (None or data[returned-index])
    //
    // assuming non-repeating data values.
    //
    // Allows 'val' to be both less than the minimum data value and larger than the maximum data value.
    //
    // 'returned-index' is in the range [0, length], therefore potentially points *past-the-end*
    // of the valid indices [0, length - 1] of the data and should not be used for indexing
    // directly into the range of data values without validation checks
    //
    // Handled special cases:
    // - cases not covered by binary_search_unsafe
    {
        // early exit: data-range is empty -> return *zero* index by definition
        // this check is needed to safely check data points afterwards
        if constexpr (std::is_signed<idx_t>::value) {
            if (length <= 0)
                return 0;
        } else {
            if (length == 0)
                return 0;
        }

        // query-point exactly matches the first data element -> return subsequent index
        if (val == data[0])
            return 1;

        // query-point is out of a non-empty data-range -> return *past-the-end* index by definition
        if (val >= data[length - 1])
            return length;
    }

    // We are now guaranteed that the inputs meet the requirements of binary_search_unsafe
    return binary_search_unsafe<scalar_t>(val, data, length);
}

template <typename scalar_t, typename idx_t>
inline __device__ idx_t binary_search_interp(scalar_t val, scalar_t const* data, idx_t length) {
    // The same as binary_search, but returns
    //
    // (None or data[returned-index-1]) < val <= (None or data[returned-index])
    //
    // in the special case that val == data[length - 1] *and* length > 1. The 'returned-index' is 'length - 1'
    // in this special case, meaning the function considers a left-open interval instead of a right-open interval.
    //
    // This is important to allow interpolating values at 'returned-index' and 'returned-index - 1' if it's
    // known that the query value is inside the full range of values. For query values outside of the
    // range of values, the function behaves like binary_search in that the returned value is in [0, length].
    //
    // Handled special cases:
    // - cases not covered by binary_search_unsafe
    {
        // early exit: data-range is empty -> return *zero* index by definition
        // this check is needed to safely check data points afterwards
        if constexpr (std::is_signed<idx_t>::value) {
            if (length <= 0)
                return 0;
        } else {
            if (length == 0)
                return 0;
        }

        // query-point exactly matches the first data element -> return subsequent index
        if (val == data[0])
            return 1;

        // query-point is at the end of a non-empty data-range and length > 1 (otherwise the
        // previous check would have triggered) -> return *end* index by definition of the
        // special case condition for this function
        if (val == data[length - 1])
            return length - 1;

        // query-point is out of a non-empty data-range -> return *past-the-end* index by definition
        if (val > data[length - 1])
            return length;
    }

    // We are now guaranteed that the inputs meet the requirements of binary_search_unsafe
    return binary_search_unsafe(val, data, length);
}
