// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include <cuda.h>
#include <cuda_runtime.h>

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <array>

// CUDA kernel for ensemble operation, used for lidar segmentation
__global__ void lidar_seg_ensemble_kernel(
    torch::PackedTensorAccessor32<unsigned char, 2, torch::RestrictPtrTraits> points,
    torch::PackedTensorAccessor32<unsigned char, 1, torch::RestrictPtrTraits> result,
    unsigned char ignore_label) {

    auto const num_points  = points.size(0),
               num_cameras = points.size(1);

    auto const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points)
        return;

    // Create a counting array for the current point
    auto counts           = std::array<unsigned char, 256>{};
    auto max_count        = 0;
    auto first_appearance = std::array<int, 256>{};
#pragma unroll
    for (auto i = 0; i < 256; ++i) {
        first_appearance[i] = num_cameras;
    }

    auto const point_label = points[idx];

    // First pass: Count the occurrences of each label and the first appearance
    for (auto c = 0; c < num_cameras; ++c) {
        auto const label = point_label[c];

        if (label == ignore_label)
            continue;

        if (counts[label] == 0)
            first_appearance[label] = c;

        counts[label]++;

        if (counts[label] > max_count)
            max_count = counts[label];
    }

    // Find the label with max count and earliest appearance
    auto max_label           = ignore_label;
    auto earliest_appearance = num_cameras;
#pragma unroll
    for (auto i = 0; i < 256; i++) {
        if (counts[i] == max_count && first_appearance[i] < earliest_appearance) {
            max_label           = i;
            earliest_appearance = first_appearance[i];
        }
    }

    // Set the result
    result[idx] = max_label;
}

torch::Tensor lidar_seg_ensemble_cu(torch::Tensor points,
                                    torch::Tensor result,
                                    unsigned char ignore_label) {
    auto const points_arg = torch::TensorArg{points, "points", 1};
    auto const result_arg = torch::TensorArg{result, "result", 2};

    torch::checkScalarType(__func__, points_arg, torch::kUInt8);
    torch::checkScalarType(__func__, result_arg, torch::kUInt8);
    torch::checkAllSameGPU(__func__, {points_arg, result_arg});
    torch::checkAllContiguous(__func__, {points_arg, result_arg});

    auto const num_points  = points.size(0);
    auto const num_cameras = points.size(1);

    torch::checkSize(__func__, points_arg, {num_points, num_cameras});
    torch::checkSize(__func__, result_arg, {num_points});

    auto const threads = 256;
    auto const blocks  = (num_points + threads - 1) / threads;
    auto const stream  = c10::cuda::getCurrentCUDAStream().stream();

    // Launch the CUDA kernel
    lidar_seg_ensemble_kernel<<<blocks, threads, 0, stream>>>(
        points.packed_accessor32<unsigned char, 2, torch::RestrictPtrTraits>(),
        result.packed_accessor32<unsigned char, 1, torch::RestrictPtrTraits>(),
        ignore_label);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}
