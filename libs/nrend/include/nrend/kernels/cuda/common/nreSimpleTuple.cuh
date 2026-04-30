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

// partial implementation of std::tuple

template <int, typename...>
struct NreSimpleTuple;

template <>
struct NreSimpleTuple<0> {
    using THead = void;
};

template <int N, typename Head, typename... Tail>
struct NreSimpleTuple<N, Head, Tail...> {
    using THead = Head;

    inline __device__ const THead& get() const { return head; }
    inline __device__ THead& get() { return head; }

    inline __device__ const auto& next() const { return tail; }
    inline __device__ auto& next() { return tail; }

private:
    THead head;
    NreSimpleTuple<N - 1, Tail...> tail;
};
