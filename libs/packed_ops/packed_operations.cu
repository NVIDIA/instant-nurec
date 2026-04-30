// SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include <functional>

#include "utils.h"

#include <ku/binary_search.cuh>
#include <ku/common.cuh>

#include <c10/cuda/CUDAGuard.h>

template <typename scalar_t>
__global__ void kernel_arange_interleave(
    // Inputs
    const uint32_t num_packs,
    const int64_t* __restrict__ num_steps,
    const int64_t* __restrict__ num_steps_cumsum,
    const scalar_t* __restrict__ starts,
    const scalar_t* __restrict__ step_sizes,
    scalar_t start,
    scalar_t step_size,
    // Outputs
    scalar_t* out,
    int64_t* nidx = nullptr) {
    const uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx >= num_packs)
        return;

    uint32_t begin    = tidx > 0 ? num_steps_cumsum[tidx - 1] : 0;
    uint32_t num_step = num_steps[tidx];

    if (starts)
        start = starts[tidx];
    if (step_sizes)
        step_size = step_sizes[tidx];

    if (nidx) {
        out += begin;
        nidx += begin;
        for (uint32_t j = 0; j < num_step; ++j) {
            out[j]  = start + (scalar_t)j * step_size;
            nidx[j] = tidx;
        }
    } else {
        out += begin;
        for (uint32_t j = 0; j < num_step; ++j) {
            out[j] = start + (scalar_t)j * step_size;
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> arange_interleave_cu(
    const torch::Tensor stop,
    bool return_nidx) {
    torch::TensorArg stop_arg{stop, "stop", 1};
    torch::checkDim(__func__, stop_arg, 1);
    torch::checkContiguous(__func__, stop_arg);
    torch::checkScalarType(__func__, stop_arg, torch::kLong);
    torch::checkDeviceType(__func__, {stop}, torch::kCUDA);

    uint32_t num_packs             = stop.size(0);
    torch::Tensor num_steps_cumsum = stop.cumsum(0);
    uint32_t num                   = num_steps_cumsum[-1].item<int64_t>();

    torch::Tensor out = torch::empty({num}, stop.options());
    torch::Tensor nidx;
    if (return_nidx) {
        nidx = torch::empty({num}, stop.options());
    }

    static constexpr uint32_t num_threads = 128;

    const c10::cuda::OptionalCUDAGuard device_guard(torch::device_of(stop));
    auto stream = c10::cuda::getCurrentCUDAStream();
    kernel_arange_interleave<int64_t><<<div_round_up(num_packs, num_threads), num_threads, 0, stream>>>(
        num_packs, stop.data_ptr<int64_t>(), num_steps_cumsum.data_ptr<int64_t>(),
        nullptr, nullptr, 0, 1, out.data_ptr<int64_t>(),
        return_nidx ? nidx.data_ptr<int64_t>() : nullptr);
    return {out, nidx};
}

std::tuple<torch::Tensor, torch::Tensor> linstep_interleave_impl(
    const torch::Tensor start,
    const torch::Tensor num_steps,
    const torch::Tensor step_size,
    bool return_nidx) {
    torch::Tensor num_steps_cumsum = num_steps.cumsum(0);
    uint32_t num_packs             = start.size(0);
    uint32_t num                   = num_steps_cumsum[-1].item<int64_t>();

    torch::Tensor out = torch::empty({num}, start.options());
    torch::Tensor nidx;
    if (return_nidx) {
        nidx = torch::empty({num}, num_steps.options());
    }

    static constexpr uint32_t num_threads = 128;

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, start.scalar_type(), "arange_interleave_tensor", ([&] {
                                  const c10::cuda::OptionalCUDAGuard device_guard(torch::device_of(start));
                                  auto stream = c10::cuda::getCurrentCUDAStream();
                                  kernel_arange_interleave<scalar_t><<<div_round_up(num_packs, num_threads), num_threads, 0, stream>>>(
                                      num_packs, num_steps.data_ptr<int64_t>(), num_steps_cumsum.data_ptr<int64_t>(),
                                      start.data_ptr<scalar_t>(), step_size.data_ptr<scalar_t>(), 0, 0, out.data_ptr<scalar_t>(),
                                      return_nidx ? nidx.data_ptr<int64_t>() : nullptr);
                              }));

    return {out, nidx};
};

std::tuple<torch::Tensor, torch::Tensor> linstep_interleave_impl(
    const torch::Tensor start,
    const torch::Tensor num_steps,
    const torch::Scalar step_size,
    bool return_nidx) {
    torch::Tensor num_steps_cumsum = num_steps.cumsum(0);
    uint32_t num_packs             = start.size(0);
    uint32_t num                   = num_steps_cumsum[-1].item<int64_t>();

    torch::Tensor out = torch::empty({num}, start.options());
    torch::Tensor nidx;
    if (return_nidx) {
        nidx = torch::empty({num}, num_steps.options());
    }

    static constexpr uint32_t num_threads = 128;

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, start.scalar_type(), "arange_interleave_scalar", ([&] {
                                  const c10::cuda::OptionalCUDAGuard device_guard(torch::device_of(start));
                                  auto stream = c10::cuda::getCurrentCUDAStream();
                                  kernel_arange_interleave<scalar_t><<<div_round_up(num_packs, num_threads), num_threads, 0, stream>>>(
                                      num_packs, num_steps.data_ptr<int64_t>(), num_steps_cumsum.data_ptr<int64_t>(),
                                      start.data_ptr<scalar_t>(), nullptr, 0, step_size.to<scalar_t>(), out.data_ptr<scalar_t>(),
                                      return_nidx ? nidx.data_ptr<int64_t>() : nullptr);
                              }));

    return {out, nidx};
};

std::tuple<torch::Tensor, torch::Tensor> linstep_interleave_cu(
    const torch::Tensor start,
    const torch::Tensor num_steps,
    const torch::Tensor step_size,
    bool return_nidx) {
    torch::TensorArg start_arg{start, "start", 1};
    torch::TensorArg num_steps_arg{num_steps, "num_steps", 2};
    torch::TensorArg step_size_arg{step_size, "step_size", 3};

    torch::checkDim(__func__, start_arg, 1);
    torch::checkDim(__func__, num_steps_arg, 1);
    torch::checkDim(__func__, step_size_arg, 1);
    torch::checkSameSize(__func__, start_arg, num_steps_arg);
    torch::checkAllContiguous(__func__, {start_arg, num_steps_arg, step_size_arg});
    torch::checkAllSameGPU(__func__, {start_arg, num_steps_arg, step_size_arg});
    torch::checkSameType(__func__, start_arg, step_size_arg);
    torch::checkScalarType(__func__, num_steps_arg, torch::kLong);
    return linstep_interleave_impl(start, num_steps, step_size, return_nidx);
}

std::tuple<torch::Tensor, torch::Tensor> linstep_interleave_cu(
    const torch::Tensor start,
    const torch::Tensor num_steps,
    double step_size,
    bool return_nidx) {
    torch::TensorArg start_arg{start, "start", 1};
    torch::TensorArg num_steps_arg{num_steps, "num_steps", 2};
    torch::checkDim(__func__, start_arg, 1);
    torch::checkDim(__func__, num_steps_arg, 1);
    torch::checkSameSize(__func__, start_arg, num_steps_arg);
    torch::checkSameGPU(__func__, start_arg, num_steps_arg);
    torch::checkScalarType(__func__, num_steps_arg, torch::kLong);
    return linstep_interleave_impl(start, num_steps, step_size, return_nidx);
}

std::tuple<torch::Tensor, torch::Tensor> linstep_interleave_cu(
    const torch::Tensor start,
    const torch::Tensor num_steps,
    int32_t step_size,
    bool return_nidx) {
    torch::TensorArg start_arg{start, "start", 1};
    torch::TensorArg num_steps_arg{num_steps, "num_steps", 2};
    torch::checkDim(__func__, start_arg, 1);
    torch::checkDim(__func__, num_steps_arg, 1);
    torch::checkSameSize(__func__, start_arg, num_steps_arg);
    torch::checkSameGPU(__func__, start_arg, num_steps_arg);
    torch::checkScalarType(__func__, num_steps_arg, torch::kLong);
    return linstep_interleave_impl(start, num_steps, step_size, return_nidx);
}

template <typename scalar_t>
__global__ void packed_min_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> vals,
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> pack_info,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> vals_min,
    torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> indices) {
    const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= pack_info.size(0))
        return;

    const int start_idx = pack_info[tidx][0], N_samples = pack_info[tidx][1];

    if (N_samples == 0) {
        vals_min[tidx] = (scalar_t)0;
        indices[tidx]  = -1;
        return;
    }

    scalar_t vmin = vals[start_idx];
    int ind       = start_idx;
    int samples   = 1;
    while (samples < N_samples) {
        const int s = start_idx + samples;
        if (vals[s] < vmin) {
            vmin = vals[s];
            ind  = s;
        }
        samples++;
    }
    vals_min[tidx] = vmin;
    indices[tidx]  = ind;
}

std::tuple<torch::Tensor, torch::Tensor> packed_min_cu(
    const torch::Tensor vals,
    const torch::Tensor pack_info) {
    auto vals_arg      = torch::TensorArg{vals, "vals", 1};
    auto pack_info_arg = torch::TensorArg{pack_info, "pack_info", 2};

    torch::checkScalarType(__func__, pack_info_arg, torch::kInt32);

    torch::checkAllSameGPU(__func__, {vals_arg, pack_info_arg});
    torch::checkAllContiguous(__func__, {vals_arg, pack_info_arg});
    torch::checkDim(__func__, vals_arg, 1);
    torch::checkDim(__func__, pack_info_arg, 2);

    const int N_rays = pack_info.size(0);
    auto vals_min    = torch::zeros({N_rays}, vals.options());
    auto indices     = torch::zeros({N_rays}, pack_info.options());

    const int threads = 256, blocks = (N_rays + threads - 1) / threads;

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, vals.scalar_type(), "packed_min_cu",
                              ([&] { packed_min_kernel<scalar_t><<<blocks, threads>>>(
                                         vals.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                         pack_info.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                                         vals_min.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                         indices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>()); }));

    return {vals_min, indices};
}

template <typename scalar_t>
__global__ void packed_max_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> vals,
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> pack_info,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> vals_max,
    torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> indices) {
    const int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= pack_info.size(0))
        return;

    const int start_idx = pack_info[tidx][0], N_samples = pack_info[tidx][1];

    if (N_samples == 0) {
        vals_max[tidx] = (scalar_t)0;
        indices[tidx]  = -1;
        return;
    }

    scalar_t vmax = vals[start_idx];
    int ind       = start_idx;
    int samples   = 1;
    while (samples < N_samples) {
        const int s = start_idx + samples;
        if (vals[s] > vmax) {
            vmax = vals[s];
            ind  = s;
        }
        samples++;
    }
    vals_max[tidx] = vmax;
    indices[tidx]  = ind;
}

std::tuple<torch::Tensor, torch::Tensor> packed_max_cu(
    const torch::Tensor vals,
    const torch::Tensor pack_info) {
    auto vals_arg      = torch::TensorArg{vals, "vals", 1};
    auto pack_info_arg = torch::TensorArg{pack_info, "pack_info", 2};

    torch::checkScalarType(__func__, pack_info_arg, torch::kInt32);

    torch::checkAllSameGPU(__func__, {vals_arg, pack_info_arg});
    torch::checkAllContiguous(__func__, {vals_arg, pack_info_arg});
    torch::checkDim(__func__, vals_arg, 1);
    torch::checkDim(__func__, pack_info_arg, 2);

    const int N_rays = pack_info.size(0);
    auto vals_max    = torch::zeros({N_rays}, vals.options());
    auto indices     = torch::zeros({N_rays}, pack_info.options());

    const int threads = 256, blocks = (N_rays + threads - 1) / threads;

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, vals.scalar_type(), "packed_max_cu",
                              ([&] { packed_max_kernel<scalar_t><<<blocks, threads>>>(
                                         vals.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                         pack_info.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                                         vals_max.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                         indices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>()); }));

    return {vals_max, indices};
}

template <typename scalar_t>
__global__ void merge_two_packs_sorted_aligned_fw_kernel(
    const int32_t num_packs,
    const int32_t num_feats_a,
    const scalar_t* __restrict__ vals_a,
    const int32_t* __restrict__ pack_info_a,
    const int32_t num_feats_b,
    const scalar_t* __restrict__ vals_b,
    const int32_t* __restrict__ pack_info_b,
    const int32_t* __restrict__ pack_info_merged,
    int32_t* __restrict__ ranks_a,
    int32_t* __restrict__ ranks_b) {
    int32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs)
        return;

    const int32_t begin_a  = pack_info_a[tidx * 2];
    const int32_t length_a = pack_info_a[tidx * 2 + 1];

    const int32_t begin_b  = pack_info_b[tidx * 2];
    const int32_t length_b = pack_info_b[tidx * 2 + 1];

    const int32_t begin_out = pack_info_merged[tidx * 2];

    vals_a += begin_a;
    ranks_a += begin_a;

    vals_b += begin_b;
    ranks_b += begin_b;

    // If tere are no samples in a skip this step and directly fill in the ranks_b
    if (length_a > 0) {
        int32_t last_i = 0;
        for (int32_t j = 0; j < length_b; ++j) {
            int32_t i  = binary_search<scalar_t, int32_t>(vals_b[j], vals_a + last_i, length_a - last_i) + last_i;
            ranks_b[j] = i;
            if (i < length_a)
                ranks_a[i]++;
            last_i = i;
        }

        // From i count to `ranks_a` offset
        ranks_a[0] += begin_out;
        for (int32_t i = 1; i < length_a; ++i) {
            ranks_a[i] += ranks_a[i - 1] + 1;
        }
    }

    // From i to `ranks_b` offset
    int32_t acc    = 1;
    int32_t last_i = -1;
    for (int32_t j = 0; j < length_b; ++j) {
        int32_t i  = ranks_b[j];
        ranks_b[j] = ((i == last_i) ? (++acc) : (acc = 0)) + ((i == 0) ? begin_out : (ranks_a[i - 1] + 1));
        last_i     = i;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> merge_two_packs_sorted_aligned_fw_cu(
    const torch::Tensor vals_a,
    const torch::Tensor pack_info_a,
    const torch::Tensor vals_b,
    const torch::Tensor pack_info_b) {

    const int32_t num_packs   = pack_info_a.size(0);
    const int32_t num_feats_a = vals_a.size(0);
    const int32_t num_feats_b = vals_b.size(0);

    at::TensorArg vals_a_arg{vals_a, "vals_a", 1};
    at::TensorArg pack_info_a_arg{pack_info_a, "pack_info_a", 2};
    at::TensorArg vals_b_arg{vals_b, "vals_b", 3};
    at::TensorArg pack_info_b_arg{pack_info_b, "pack_info_b", 4};

    at::checkDim(__func__, vals_a_arg, 1);
    at::checkDim(__func__, vals_b_arg, 1);
    at::checkDim(__func__, pack_info_a_arg, 2);
    at::checkDim(__func__, pack_info_b_arg, 2);
    at::checkAllSameGPU(__func__, {vals_a_arg, vals_b_arg, pack_info_a_arg, pack_info_b_arg});
    at::checkAllContiguous(__func__, {vals_a_arg, vals_b_arg, pack_info_a_arg, pack_info_b_arg});
    at::checkSameType(__func__, vals_a_arg, vals_b_arg);
    at::checkScalarType(__func__, pack_info_a_arg, at::kInt);
    at::checkScalarType(__func__, pack_info_b_arg, at::kInt);
    at::checkSize(__func__, pack_info_b_arg, {pack_info_a.size(0), 2});

    // Check vals' size with pack_info
    if (num_feats_a > 0) {
        at::checkSize(__func__, vals_a_arg, 0, pack_info_a.index({-1, 0}).item<int64_t>() + pack_info_a.index({-1, 1}).item<int64_t>());
    }
    if (num_feats_b > 0) {
        at::checkSize(__func__, vals_b_arg, 0, pack_info_b.index({-1, 0}).item<int64_t>() + pack_info_b.index({-1, 1}).item<int64_t>());
    }

    torch::Tensor n_per_pack = pack_info_a.select(1, 1) + pack_info_b.select(1, 1);
    torch::Tensor cumsum     = n_per_pack.cumsum(0, torch::kInt);
    torch::Tensor pack_info  = torch::stack({cumsum - n_per_pack, n_per_pack}, 1);

    torch::Tensor ranks_a = torch::zeros({num_feats_a}, pack_info_a.options());
    torch::Tensor ranks_b = torch::zeros({num_feats_b}, pack_info_b.options());

    const int threads = 256, blocks = div_round_up(num_packs, threads);

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, vals_a.scalar_type(), "merge_two_packs_sorted_aligned_fw_cu", ([&] {
                                  auto const device_guard = c10::cuda::OptionalCUDAGuard(torch::device_of(vals_a));
                                  merge_two_packs_sorted_aligned_fw_kernel<scalar_t><<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
                                      num_packs,
                                      num_feats_a,
                                      vals_a.data_ptr<scalar_t>(),
                                      pack_info_a.data_ptr<int32_t>(),
                                      num_feats_b,
                                      vals_b.data_ptr<scalar_t>(),
                                      pack_info_b.data_ptr<int32_t>(),
                                      pack_info.data_ptr<int32_t>(),
                                      ranks_a.data_ptr<int32_t>(),
                                      ranks_b.data_ptr<int32_t>());
                              }));

    return {ranks_a, ranks_b, pack_info};
}

template <typename scalar_t>
__global__ void packed_cumsum_kernel(
    const int32_t num_packs,
    const int32_t num_feats,
    const int32_t feat_dim,
    const scalar_t* __restrict__ feats_in,
    const int32_t* __restrict__ pack_info,
    const int32_t offset,
    scalar_t* __restrict__ feats_out) {
    int32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) {
        return;
    }

    int32_t begin = pack_info[tidx * 2];
    int32_t end   = begin + pack_info[tidx * 2 + 1];

    // pack is empty / no scan to compute
    if (begin == end) {
        return;
    }

    if (offset == 0) {
        for (int32_t j = 0; j < feat_dim; ++j) {
            feats_out[begin * feat_dim + j] = feats_in[begin * feat_dim + j];
        }
    }
    // For loop on feat_dim first.
    for (int32_t j = 0; j < feat_dim; ++j) {
        for (int32_t i = begin + 1; i < end; ++i) {
            feats_out[i * feat_dim + j] = feats_in[(i - offset) * feat_dim + j] + feats_out[(i - 1) * feat_dim + j];
        }
    }
}

template <typename scalar_t>
__global__ void packed_cumsum_reverse_kernel(
    // Inputs
    const int32_t num_packs,
    const int32_t num_feats,
    const int32_t feat_dim,
    const scalar_t* __restrict__ feats_in,
    const int32_t* __restrict__ pack_info,
    const int32_t offset,
    // Outputs
    scalar_t* __restrict__ feats_out) {
    int32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) {
        return;
    }

    int32_t begin = pack_info[tidx * 2];
    int32_t end   = begin + pack_info[tidx * 2 + 1];

    // pack is empty / no scan to compute
    if (begin == end) {
        return;
    }

    if (offset == 0) {
        for (int32_t j = 0; j < feat_dim; ++j) {
            feats_out[(end - 1) * feat_dim + j] = feats_in[(end - 1) * feat_dim + j];
        }
    }
    // For loop on feat_dim first.
    for (int32_t j = 0; j < feat_dim; ++j) {
        for (int32_t i = end - 2; i >= begin; --i) {
            feats_out[i * feat_dim + j] = feats_in[(i + offset) * feat_dim + j] + feats_out[(i + 1) * feat_dim + j];
        }
    }
}

torch::Tensor packed_cumsum_cu(
    const torch::Tensor data,
    const torch::Tensor pack_info,
    bool exclusive,
    bool reverse) {

    auto data_arg      = torch::TensorArg{data, "data", 1};
    auto pack_info_arg = torch::TensorArg{pack_info, "pack_info", 2};

    torch::checkScalarType(__func__, pack_info_arg, torch::kInt32);

    torch::checkAllSameGPU(__func__, {data_arg, pack_info_arg});
    torch::checkAllContiguous(__func__, {data_arg, pack_info_arg});
    torch::checkDimRange(__func__, data_arg, 1, 3);
    torch::checkDim(__func__, pack_info_arg, 2);

    int32_t num_feats       = data.size(0);
    int32_t num_packs       = pack_info.size(0);
    int32_t feat_dim        = data.dim() == 1 ? 1 : data.size(1);
    torch::Tensor feats_out = data.dim() == 1 ? torch::zeros({num_feats}, data.options()) : torch::zeros({num_feats, feat_dim}, data.options());
    int32_t offset          = exclusive ? 1 : 0;

    const int threads = 256, blocks = div_round_up(num_packs, threads);

    if (reverse) {
        AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, data.scalar_type(), "packed_cumsum_cu", ([&] {
                                      auto const device_guard = c10::cuda::OptionalCUDAGuard(torch::device_of(data));
                                      packed_cumsum_reverse_kernel<scalar_t><<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
                                          num_packs,
                                          num_feats,
                                          feat_dim,
                                          data.data_ptr<scalar_t>(),
                                          pack_info.data_ptr<int32_t>(),
                                          offset,
                                          feats_out.data_ptr<scalar_t>());
                                  }));
    } else {
        AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, data.scalar_type(), "packed_cumsum_cu", ([&] {
                                      auto const device_guard = c10::cuda::OptionalCUDAGuard(torch::device_of(data));
                                      packed_cumsum_kernel<scalar_t><<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
                                          num_packs,
                                          num_feats,
                                          feat_dim,
                                          data.data_ptr<scalar_t>(),
                                          pack_info.data_ptr<int32_t>(),
                                          offset,
                                          feats_out.data_ptr<scalar_t>());
                                  }));
    }

    return feats_out;
}

// Modified from https://github.com/NVIDIAGameWorks/kaolin
template <typename scalar_t>
__global__ void packed_cumprod_kernel(
    // Inputs
    const int32_t num_packs,
    const int32_t num_feats,
    const int32_t feat_dim,
    const scalar_t* __restrict__ feats_in,
    const int32_t* __restrict__ pack_info,
    const int32_t offset,
    // Outputs
    scalar_t* __restrict__ feats_out) {
    int32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) {
        return;
    }

    int32_t begin = pack_info[tidx * 2];
    int32_t end   = begin + pack_info[tidx * 2 + 1];

    // pack is empty / no scan to compute
    if (begin == end) {
        return;
    }

    if (offset == 0) {
        for (int32_t j = 0; j < feat_dim; ++j) {
            feats_out[begin * feat_dim + j] = feats_in[begin * feat_dim + j];
        }
    }

    if (offset == 1) {
        for (int32_t j = 0; j < feat_dim; ++j) {
            feats_out[begin * feat_dim + j] = 1.0;
        }
    }

    // For loop on feat_dim first.
    for (int32_t j = 0; j < feat_dim; ++j) {
        for (int32_t i = begin + 1; i < end; ++i) {
            feats_out[i * feat_dim + j] = feats_in[(i - offset) * feat_dim + j] * feats_out[(i - 1) * feat_dim + j];
        }
    }
}

template <typename scalar_t>
__global__ void packed_cumprod_reverse_kernel(
    // Inputs
    const int32_t num_packs,
    const int32_t num_feats,
    const int32_t feat_dim,
    const scalar_t* __restrict__ feats_in,
    const int32_t* __restrict__ pack_info,
    const int32_t offset,
    // Outputs
    scalar_t* __restrict__ feats_out) {
    int32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs) {
        return;
    }

    int32_t begin = pack_info[tidx * 2];
    int32_t end   = begin + pack_info[tidx * 2 + 1];

    // pack is empty / no scan to compute
    if (begin == end) {
        return;
    }

    if (offset == 0) {
        for (int32_t j = 0; j < feat_dim; ++j) {
            feats_out[(end - 1) * feat_dim + j] = feats_in[(end - 1) * feat_dim + j];
        }
    }

    if (offset == 1) {
        for (int32_t j = 0; j < feat_dim; ++j) {
            feats_out[(end - 1) * feat_dim + j] = 1.0;
        }
    }

    // For loop on feat_dim first.
    for (int32_t j = 0; j < feat_dim; ++j) {
        for (int32_t i = end - 2; i >= begin; --i) {
            feats_out[i * feat_dim + j] = feats_in[(i + offset) * feat_dim + j] * feats_out[(i + 1) * feat_dim + j];
        }
    }
}

torch::Tensor packed_cumprod_cu(
    const torch::Tensor data,
    const torch::Tensor pack_info,
    bool exclusive,
    bool reverse) {

    auto data_arg      = torch::TensorArg{data, "data", 1};
    auto pack_info_arg = torch::TensorArg{pack_info, "pack_info", 2};

    torch::checkScalarType(__func__, pack_info_arg, torch::kInt32);

    torch::checkAllSameGPU(__func__, {data_arg, pack_info_arg});
    torch::checkAllContiguous(__func__, {data_arg, pack_info_arg});
    torch::checkDimRange(__func__, data_arg, 1, 3);
    torch::checkDim(__func__, pack_info_arg, 2);

    int32_t num_feats       = data.size(0);
    int32_t num_packs       = pack_info.size(0);
    int32_t feat_dim        = data.dim() == 1 ? 1 : data.size(1);
    torch::Tensor feats_out = data.dim() == 1 ? torch::zeros({num_feats}, data.options()) : torch::zeros({num_feats, feat_dim}, data.options());
    int32_t offset          = exclusive ? 1 : 0;

    const int threads = 256, blocks = div_round_up(num_packs, threads);

    if (reverse) {
        AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, data.scalar_type(), "packed_cumprod_cu", ([&] {
                                      auto const device_guard = c10::cuda::OptionalCUDAGuard(torch::device_of(data));
                                      packed_cumprod_reverse_kernel<scalar_t><<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
                                          num_packs,
                                          num_feats,
                                          feat_dim,
                                          data.data_ptr<scalar_t>(),
                                          pack_info.data_ptr<int32_t>(),
                                          offset,
                                          feats_out.data_ptr<scalar_t>());
                                  }));
    } else {
        AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, data.scalar_type(), "packed_cumprod_cu", ([&] {
                                      auto const device_guard = c10::cuda::OptionalCUDAGuard(torch::device_of(data));
                                      packed_cumprod_kernel<scalar_t><<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
                                          num_packs,
                                          num_feats,
                                          feat_dim,
                                          data.data_ptr<scalar_t>(),
                                          pack_info.data_ptr<int32_t>(),
                                          offset,
                                          feats_out.data_ptr<scalar_t>());
                                  }));
    }

    return feats_out;
}

template <typename scalar_t, typename ArithmeticOp>
__global__ void packed_arithmetic_fw_kernel(
    // Inputs
    const int32_t num_packs,
    const int32_t num_feats,
    const int32_t feat_dim,
    const scalar_t* __restrict__ feats_in,
    const scalar_t* __restrict__ other_in,
    const int32_t* __restrict__ pack_info,
    const ArithmeticOp op,
    // Outputs
    scalar_t* __restrict__ feats_out) {

    int32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx >= num_packs)
        return;

    other_in += tidx * feat_dim;
    int32_t begin = pack_info[tidx * 2];
    int32_t end   = begin + pack_info[tidx * 2 + 1];

    // loop over feat_dim first as data is stored in column major
    for (int32_t j = 0; j < feat_dim; ++j) {
        for (int32_t i = begin; i < end; ++i) {
            feats_out[i * feat_dim + j] = op(feats_out[i * feat_dim + j], other_in[j]);
        }
    }
}

template <template <typename T> class ArithmeticOpT>
torch::Tensor packed_arithmetic_cu(
    const torch::Tensor data,
    const torch::Tensor other,
    const torch::Tensor pack_info) {

    auto data_arg      = torch::TensorArg{data, "data", 1};
    auto other_arg     = torch::TensorArg{other, "other", 2};
    auto pack_info_arg = torch::TensorArg{pack_info, "pack_info", 3};

    torch::checkSameDim(__func__, data_arg, other_arg);
    torch::checkScalarType(__func__, pack_info_arg, torch::kInt32);

    torch::checkAllSameGPU(__func__, {data_arg, other_arg, pack_info_arg});
    torch::checkAllContiguous(__func__, {data_arg, other_arg, pack_info_arg});
    torch::checkDimRange(__func__, data_arg, 1, 3);
    torch::checkDim(__func__, pack_info_arg, 2);

    int32_t num_packs      = pack_info.size(0);
    int32_t num_feats      = data.size(0);
    int32_t feat_dim       = data.dim() == 1 ? 1 : data.size(1);
    torch::Tensor data_out = data.clone();

    torch::checkSize(__func__, other_arg, 0, num_packs);
    if (data.dim() == 2)
        torch::checkSize(__func__, other_arg, 1, feat_dim);

    const int threads = 256, blocks = div_round_up(num_packs, threads);

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, data.scalar_type(), "packed_arithmetic_cu", ([&] {
                                  auto const device_guard = c10::cuda::OptionalCUDAGuard(torch::device_of(data));
                                  packed_arithmetic_fw_kernel<scalar_t><<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
                                      num_packs, num_feats, feat_dim,
                                      data.data_ptr<scalar_t>(),
                                      other.data_ptr<scalar_t>(),
                                      pack_info.data_ptr<int32_t>(),
                                      ArithmeticOpT<scalar_t>(),
                                      data_out.data_ptr<scalar_t>());
                              }));

    return data_out;
}

torch::Tensor packed_add_cu(
    const torch::Tensor data,
    const torch::Tensor other,
    const torch::Tensor pack_info) {

    return packed_arithmetic_cu<std::plus>(data, other, pack_info);
}

torch::Tensor packed_sub_cu(
    const torch::Tensor data,
    const torch::Tensor other,
    const torch::Tensor pack_info) {

    return packed_arithmetic_cu<std::minus>(data, other, pack_info);
}

torch::Tensor packed_mul_cu(
    const torch::Tensor data,
    const torch::Tensor other,
    const torch::Tensor pack_info) {

    return packed_arithmetic_cu<std::multiplies>(data, other, pack_info);
}

torch::Tensor packed_div_cu(
    const torch::Tensor data,
    const torch::Tensor other,
    const torch::Tensor pack_info) {

    return packed_arithmetic_cu<std::divides>(data, other, pack_info);
}

template <typename scalar_t>
__global__ void packed_invert_cdf_fw_kernel(
    const int32_t num_packs,
    const int32_t num_bins,
    const scalar_t* __restrict__ bins,
    const scalar_t* __restrict__ cdfs,
    const int32_t* __restrict__ bins_pack_info, // [num_packs, 2]
    const scalar_t* __restrict__ u_vals,
    const int32_t* __restrict__ u_pack_info, // [num_packs, 2]
    scalar_t* __restrict__ samples,
    int32_t* __restrict__ bin_idx,
    const float eps) {
    int32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs)
        return;

    const int32_t begin  = bins_pack_info[tidx * 2];
    const int32_t length = bins_pack_info[tidx * 2 + 1];

    if (length == 0) {
        return;
    }

    const int32_t out_begin     = u_pack_info[tidx * 2];
    const int32_t num_to_sample = u_pack_info[tidx * 2 + 1];

    bins += begin;
    cdfs += begin;
    u_vals += out_begin;
    bin_idx += out_begin;
    samples += out_begin;
    for (int32_t i = 0; i < num_to_sample; ++i) {
        scalar_t u = u_vals[i];

        int32_t pos = binary_search(u, cdfs, length); // returns indices in [0, length] - make sure to not sample cdf values out-of-bounds
                                                      // (shouldn't happen if u's are consistent with cdf / are all strictly inside the cdf range)

        bin_idx[i] = std::min(pos, length - 1);
        if (pos == 0) {
            samples[i] = bins[0];
        } else if (pos == length) {
            samples[i] = bins[length - 1];
        } else {
            int32_t pos_prev = pos - 1;
            scalar_t pmf     = cdfs[pos] - cdfs[pos_prev];
            samples[i]       = pmf < eps ? bins[pos_prev] : (bins[pos_prev] + ((u - cdfs[pos_prev]) / pmf) * (bins[pos] - bins[pos_prev]));
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> packed_invert_cdf_cu(
    const torch::Tensor bins,
    const torch::Tensor cdfs,
    const torch::Tensor bins_pack_info,
    const torch::Tensor u_vals,
    const torch::Tensor u_pack_info,
    const float eps) {

    torch::TensorArg bins_arg{bins, "bins", 1};
    torch::TensorArg cdfs_arg{cdfs, "cdfs", 2};
    torch::TensorArg bins_pack_info_arg{bins_pack_info, "bins_pack_info", 3};
    torch::TensorArg u_vals_arg{u_vals, "u_vals", 4};
    torch::TensorArg u_pack_info_arg{u_pack_info, "u_pack_info", 5};

    torch::checkDim(__func__, bins_arg, 1);
    torch::checkDim(__func__, cdfs_arg, 1);
    torch::checkDim(__func__, bins_pack_info_arg, 2);
    torch::checkDim(__func__, u_vals_arg, 1);
    torch::checkDim(__func__, u_pack_info_arg, 2);
    torch::checkAllSameGPU(__func__, {bins_arg, cdfs_arg, u_vals_arg, bins_pack_info_arg, u_pack_info_arg});
    torch::checkAllContiguous(__func__, {bins_arg, cdfs_arg, u_vals_arg, bins_pack_info_arg, u_pack_info_arg});
    torch::checkScalarTypes(__func__, u_vals_arg, {at::kHalf, at::kFloat, at::kDouble});
    torch::checkAllSameType(__func__, {bins_arg, cdfs_arg, u_vals_arg});
    torch::checkScalarType(__func__, bins_pack_info_arg, at::kInt);
    torch::checkScalarType(__func__, u_pack_info_arg, at::kInt);
    torch::checkSameSize(__func__, bins_arg, cdfs_arg);
    torch::checkSameSize(__func__, bins_pack_info_arg, u_pack_info_arg);

    torch::checkSize(__func__, bins_arg, 0, bins_pack_info.index({-1, 0}).item<int32_t>() + bins_pack_info.index({-1, 1}).item<int32_t>());
    torch::checkSize(__func__, u_vals_arg, 0, u_pack_info.index({-1, 0}).item<int32_t>() + u_pack_info.index({-1, 1}).item<int32_t>());

    const int32_t num_bins  = bins.size(0);
    const int32_t num_packs = bins_pack_info.size(0);

    // `bin_idx` should always of the same size as u_vals;
    torch::Tensor bin_idx   = torch::full_like(u_vals, -1, u_vals.options().dtype(torch::kInt));
    torch::Tensor t_samples = torch::zeros_like(u_vals, u_vals.options());

    const int threads = 256, blocks = div_round_up(num_packs, threads);

    AT_DISPATCH_FLOATING_TYPES_AND(torch::ScalarType::Half, bins.scalar_type(), "packed_invert_cdf_cu", ([&] {
                                       auto const device_guard = c10::cuda::OptionalCUDAGuard(torch::device_of(bins));
                                       packed_invert_cdf_fw_kernel<scalar_t><<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
                                           num_packs, num_bins,
                                           bins.data_ptr<scalar_t>(),
                                           cdfs.data_ptr<scalar_t>(),
                                           bins_pack_info.data_ptr<int32_t>(),
                                           u_vals.data_ptr<scalar_t>(),
                                           u_pack_info.data_ptr<int32_t>(),
                                           t_samples.data_ptr<scalar_t>(),
                                           bin_idx.data_ptr<int32_t>(),
                                           eps);
                                   }));

    return {t_samples, bin_idx};
}

template <typename scalar_t>
__global__ void packed_interp_fw_kernel(
    const int32_t num_packs,
    const int32_t num_bins,
    const scalar_t* __restrict__ bins,
    const scalar_t* __restrict__ vals,
    const int32_t* __restrict__ bins_pack_info, // [num_packs, 2]
    const scalar_t* __restrict__ query_pts,
    const int32_t* __restrict__ query_pack_info, // [num_packs, 2]
    scalar_t* __restrict__ interpolated,
    int32_t* __restrict__ bin_idx,
    const float eps) {
    int32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs)
        return;

    const int32_t begin  = bins_pack_info[tidx * 2];
    const int32_t length = bins_pack_info[tidx * 2 + 1];

    if (length == 0) {
        return;
    }

    const int32_t out_begin     = query_pack_info[tidx * 2];
    const int32_t num_to_sample = query_pack_info[tidx * 2 + 1];

    bins += begin;
    vals += begin;
    query_pts += out_begin;
    bin_idx += out_begin;
    interpolated += out_begin;
    for (int32_t i = 0; i < num_to_sample; ++i) {
        scalar_t q = query_pts[i];

        int32_t pos = binary_search(q, bins, length); // returns indices in [0, length]
        bin_idx[i]  = std::min(pos, length - 1);
        if (pos == 0) {
            interpolated[i] = vals[0];
        } else if (pos == length) {
            interpolated[i] = vals[length - 1];
        } else {
            int32_t pos_prev = pos - 1;
            scalar_t denom   = bins[pos] - bins[pos_prev];
            interpolated[i]  = denom < eps ? vals[pos_prev] : (vals[pos_prev] + ((q - bins[pos_prev]) / denom) * (vals[pos] - vals[pos_prev]));
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> packed_interp_cu(
    const torch::Tensor bins,
    const torch::Tensor vals,
    const torch::Tensor bins_pack_info,
    const torch::Tensor query_pts,
    const torch::Tensor query_pack_info,
    const float eps) {

    torch::TensorArg bins_arg{bins, "bins", 1};
    torch::TensorArg vals_arg{vals, "vals", 2};
    torch::TensorArg bins_pack_info_arg{bins_pack_info, "bins_pack_info", 3};
    torch::TensorArg query_pts_arg{query_pts, "query_pts", 4};
    torch::TensorArg query_pack_info_arg{query_pack_info, "query_pack_info", 5};

    torch::checkDim(__func__, bins_arg, 1);
    torch::checkDim(__func__, vals_arg, 1);
    torch::checkDim(__func__, bins_pack_info_arg, 2);
    torch::checkDim(__func__, query_pts_arg, 1);
    torch::checkDim(__func__, query_pack_info_arg, 2);
    torch::checkAllSameGPU(__func__, {bins_arg, vals_arg, query_pts_arg, bins_pack_info_arg, query_pack_info_arg});
    torch::checkAllContiguous(__func__, {bins_arg, vals_arg, query_pts_arg, bins_pack_info_arg, query_pack_info_arg});
    torch::checkScalarTypes(__func__, query_pts_arg, {at::kHalf, at::kFloat, at::kDouble});
    torch::checkAllSameType(__func__, {bins_arg, vals_arg, query_pts_arg});
    torch::checkScalarType(__func__, bins_pack_info_arg, at::kInt);
    torch::checkScalarType(__func__, query_pack_info_arg, at::kInt);
    torch::checkSameSize(__func__, bins_arg, vals_arg);
    torch::checkSameSize(__func__, bins_pack_info_arg, query_pack_info_arg);

    torch::checkSize(__func__, bins_arg, 0, bins_pack_info.index({-1, 0}).item<int32_t>() + bins_pack_info.index({-1, 1}).item<int32_t>());
    torch::checkSize(__func__, query_pts_arg, 0, query_pack_info.index({-1, 0}).item<int32_t>() + query_pack_info.index({-1, 1}).item<int32_t>());

    const int32_t num_bins  = bins.size(0);
    const int32_t num_packs = bins_pack_info.size(0);

    // `bin_idx` should always of the same size as query_pts;
    torch::Tensor bin_idx      = torch::full_like(query_pts, -1, query_pts.options().dtype(torch::kInt));
    torch::Tensor interpolated = torch::zeros_like(query_pts, query_pts.options());

    const int threads = 256, blocks = div_round_up(num_packs, threads);

    AT_DISPATCH_FLOATING_TYPES_AND(torch::ScalarType::Half, bins.scalar_type(), "packed_interp_cu", ([&] {
                                       auto const device_guard = c10::cuda::OptionalCUDAGuard(torch::device_of(bins));
                                       packed_interp_fw_kernel<scalar_t><<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
                                           num_packs, num_bins,
                                           bins.data_ptr<scalar_t>(),
                                           vals.data_ptr<scalar_t>(),
                                           bins_pack_info.data_ptr<int32_t>(),
                                           query_pts.data_ptr<scalar_t>(),
                                           query_pack_info.data_ptr<int32_t>(),
                                           interpolated.data_ptr<scalar_t>(),
                                           bin_idx.data_ptr<int32_t>(),
                                           eps);
                                   }));

    return {interpolated, bin_idx};
}

template <typename scalar_t>
__global__ void packed_sum_fw_kernel(
    const int32_t num_packs,
    const int32_t num_feats,
    const int32_t feat_dim,
    const scalar_t* __restrict__ data,
    const int32_t* __restrict__ pack_info,
    scalar_t* __restrict__ feats_out) {

    int32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs)
        return;

    int32_t begin = pack_info[tidx * 2];
    int32_t end   = begin + pack_info[tidx * 2 + 1];

    for (int32_t i = begin; i < end; ++i) {
        for (int32_t j = 0; j < feat_dim; ++j) {
            feats_out[tidx * feat_dim + j] += data[i * feat_dim + j];
        }
    }
}

torch::Tensor packed_sum_cu(
    const torch::Tensor data,
    const torch::Tensor pack_info) {

    auto data_arg      = torch::TensorArg{data, "data", 1};
    auto pack_info_arg = torch::TensorArg{pack_info, "pack_info", 2};

    torch::checkScalarType(__func__, pack_info_arg, torch::kInt32);

    torch::checkAllSameGPU(__func__, {data_arg, pack_info_arg});
    torch::checkAllContiguous(__func__, {data_arg, pack_info_arg});
    torch::checkDimRange(__func__, data_arg, 1, 3);
    torch::checkDim(__func__, pack_info_arg, 2);

    int32_t num_packs = pack_info.size(0);
    int32_t num_feats = data.size(0);

    int32_t feat_dim        = data.dim() == 1 ? 1 : data.size(1);
    torch::Tensor feats_out = data.dim() == 1 ? torch::zeros({num_packs}, data.options()) : torch::zeros({num_packs, feat_dim}, data.options());

    const int threads = 256, blocks = div_round_up(num_packs, threads);

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, data.scalar_type(), "packed_sum_cu", ([&] {
                                  auto const device_guard = c10::cuda::OptionalCUDAGuard(torch::device_of(data));
                                  packed_sum_fw_kernel<scalar_t><<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
                                      num_packs,
                                      num_feats,
                                      feat_dim,
                                      data.data_ptr<scalar_t>(),
                                      pack_info.data_ptr<int32_t>(),
                                      feats_out.data_ptr<scalar_t>());
                              }));

    return feats_out;
}

template <typename scalar_t>
__global__ void packed_sum_bw_kernel(
    const int32_t num_packs,
    const int32_t num_feats,
    const int32_t feat_dim,
    const scalar_t* __restrict__ dL_dsum,
    const int32_t* __restrict__ pack_info,
    scalar_t* __restrict__ dL_ddata) {

    int32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs)
        return;

    int32_t begin = pack_info[tidx * 2];
    int32_t end   = begin + pack_info[tidx * 2 + 1];

    for (int32_t i = begin; i < end; ++i) {
        for (int32_t j = 0; j < feat_dim; ++j) {
            dL_ddata[i * feat_dim + j] = dL_dsum[tidx * feat_dim + j];
        }
    }
}

torch::Tensor packed_sum_bw_cu(
    const torch::Tensor data,
    const torch::Tensor pack_info,
    const torch::Tensor dL_dsum) {

    auto data_arg      = torch::TensorArg{data, "data", 1};
    auto pack_info_arg = torch::TensorArg{pack_info, "pack_info", 2};
    auto dL_dsum_arg   = torch::TensorArg{dL_dsum, "dL_dsum", 3};

    torch::checkScalarType(__func__, pack_info_arg, torch::kInt32);

    torch::checkAllSameGPU(__func__, {data_arg, pack_info_arg, dL_dsum_arg});
    torch::checkAllContiguous(__func__, {data_arg, pack_info_arg, dL_dsum_arg});
    torch::checkDimRange(__func__, data_arg, 1, 3);
    torch::checkDim(__func__, pack_info_arg, 2);
    torch::checkSameType(__func__, data_arg, dL_dsum_arg);

    int32_t num_packs = pack_info.size(0);
    int32_t num_feats = data.size(0);
    int32_t feat_dim  = data.dim() == 1 ? 1 : data.size(1);

    if (data.dim() == 1)
        torch::checkSize(__func__, dL_dsum_arg, {num_packs});
    else
        torch::checkSize(__func__, dL_dsum_arg, {num_packs, feat_dim});

    torch::Tensor dL_ddata = torch::zeros_like(data);

    const int threads = 256, blocks = div_round_up(num_packs, threads);

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, data.scalar_type(), "packed_sum_bw_cu", ([&] {
                                  auto const device_guard = c10::cuda::OptionalCUDAGuard(torch::device_of(data));
                                  packed_sum_bw_kernel<scalar_t><<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
                                      num_packs,
                                      num_feats,
                                      feat_dim,
                                      dL_dsum.data_ptr<scalar_t>(),
                                      pack_info.data_ptr<int32_t>(),
                                      dL_ddata.data_ptr<scalar_t>());
                              }));

    return dL_ddata;
}

template <typename scalar_t>
__global__ void packed_weigthed_sum_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> data,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> weights,
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> pack_info,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> accumulated_data) {
    const int32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= pack_info.size(0))
        return;

    const int32_t start_idx = pack_info[tidx][0];
    const int32_t N_samples = pack_info[tidx][1];
    int32_t sample          = 0;

    while (sample < N_samples) {
        const int32_t s = start_idx + sample;
        for (int32_t i = 0; i < data.size(1); i++) {
            accumulated_data[tidx][i] += weights[s] * data[s][i];
        }
        sample++;
    }
}

torch::Tensor packed_weighted_sum_cu(
    const torch::Tensor data,
    const torch::Tensor weights,
    const torch::Tensor pack_info) {

    auto data_arg      = torch::TensorArg{data, "data", 1};
    auto weights_arg   = torch::TensorArg{weights, "weights", 2};
    auto pack_info_arg = torch::TensorArg{pack_info, "pack_info", 3};

    torch::checkScalarType(__func__, pack_info_arg, torch::kInt32);

    torch::checkAllSameGPU(__func__, {data_arg, weights_arg, pack_info_arg});
    torch::checkAllContiguous(__func__, {data_arg, weights_arg, pack_info_arg});
    torch::checkDim(__func__, weights_arg, 1);
    torch::checkDim(__func__, data_arg, 2);
    torch::checkDim(__func__, pack_info_arg, 2);

    const int32_t num_packs = pack_info.size(0);

    auto accumulated_data = torch::zeros({num_packs, data.size(1)}, data.options());

    const int32_t threads = 512, blocks = div_round_up(num_packs, threads);

    AT_DISPATCH_FLOATING_TYPES_AND(torch::ScalarType::Half, data.scalar_type(), "packed_weighted_sum_cu",
                                   ([&] {
                                       auto const device_guard = c10::cuda::OptionalCUDAGuard(torch::device_of(data));
                                       packed_weigthed_sum_kernel<scalar_t><<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
                                           data.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                           weights.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                           pack_info.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                                           accumulated_data.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
                                   }));

    return accumulated_data;
}

template <typename scalar_t>
__global__ void packed_weigthed_sum_bw_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> data,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> weights,
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits> pack_info,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> dL_daccumulated_data,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> dL_ddata,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> dL_dweights) {
    const int32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= pack_info.size(0))
        return;

    const int32_t start_idx = pack_info[tidx][0];
    const int32_t N_samples = pack_info[tidx][1];
    int32_t sample          = 0;

    while (sample < N_samples) {
        const int32_t s = start_idx + sample;
        for (int32_t i = 0; i < data.size(1); i++) {
            auto const grad_feat = dL_daccumulated_data[tidx][i];
            dL_ddata[s][i]       = weights[s] * grad_feat;
            dL_dweights[s] += data[s][i] * grad_feat;
        }
        sample++;
    }
}

std::tuple<torch::Tensor, torch::Tensor> packed_weighted_sum_bw_cu(
    const torch::Tensor data,
    const torch::Tensor weights,
    const torch::Tensor pack_info,
    const torch::Tensor dL_daccumulated_data) {

    auto data_arg                 = torch::TensorArg{data, "data", 1};
    auto weights_arg              = torch::TensorArg{weights, "weights", 2};
    auto pack_info_arg            = torch::TensorArg{pack_info, "pack_info", 3};
    auto dL_daccumulated_data_arg = torch::TensorArg{dL_daccumulated_data, "dL_daccumulated_data", 4};

    torch::checkScalarType(__func__, pack_info_arg, torch::kInt32);

    torch::checkAllSameGPU(__func__, {data_arg, weights_arg, pack_info_arg, dL_daccumulated_data_arg});
    torch::checkAllContiguous(__func__, {data_arg, weights_arg, pack_info_arg, dL_daccumulated_data_arg});
    torch::checkDim(__func__, weights_arg, 1);
    torch::checkDim(__func__, data_arg, 2);
    torch::checkDim(__func__, pack_info_arg, 2);
    torch::checkAllSameType(__func__, {data_arg, weights_arg, dL_daccumulated_data_arg});

    const int32_t num_packs = pack_info.size(0);
    const int32_t feat_dim  = data.size(1);

    torch::checkSize(__func__, dL_daccumulated_data_arg, {num_packs, feat_dim});

    auto dL_ddata    = torch::zeros_like(data);
    auto dL_dweights = torch::zeros_like(weights);

    const int32_t threads = 512, blocks = div_round_up(num_packs, threads);

    AT_DISPATCH_FLOATING_TYPES_AND(torch::ScalarType::Half, data.scalar_type(), "packed_weighted_sum_bw_cu",
                                   ([&] {
                                       auto const device_guard = c10::cuda::OptionalCUDAGuard(torch::device_of(data));
                                       packed_weigthed_sum_bw_kernel<scalar_t><<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
                                           data.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                           weights.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                           pack_info.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
                                           dL_daccumulated_data.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                           dL_ddata.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                           dL_dweights.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>());
                                   }));

    return {dL_ddata, dL_dweights};
}

template <typename scalar_t, typename idx_t>
__global__ void kernel_packed_searchsorted(
    // Inputs
    uint32_t num_packs,
    uint32_t num_feats,
    scalar_t const* __restrict__ bins,
    scalar_t const* __restrict__ vals,
    idx_t const* __restrict__ pack_info,
    idx_t num_to_search,
    idx_t const* __restrict__ pack_info_to_search, // Optional, if different num_to_search for each each pack
    // Outputs
    idx_t* __restrict__ pidx) {
    auto const tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs)
        return;

    auto const begin  = pack_info[tidx * 2];
    auto const length = pack_info[tidx * 2 + 1];

    idx_t out_begin;
    if (pack_info_to_search) {
        out_begin     = pack_info_to_search[tidx * 2];
        num_to_search = pack_info_to_search[tidx * 2 + 1];
    } else {
        out_begin = tidx * num_to_search;
    }

    bins += begin;
    pidx += out_begin;
    vals += out_begin;
    for (idx_t i = 0; i < num_to_search; ++i) {
        pidx[i] = begin + binary_search(vals[i], bins, length);
    }
}
template <typename scalar_t, typename idx_t>
__global__ void kernel_packed_searchsorted_indexed(
    // Inputs
    uint32_t num_vals,
    scalar_t const* __restrict__ bins,
    scalar_t const* __restrict__ vals,
    idx_t const* __restrict__ pack_infos,
    idx_t const* __restrict__ vals_indices,
    // Outputs
    idx_t* __restrict__ pidx) {
    auto const tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_vals)
        return;

    auto const bin_idx = vals_indices[tidx];

    auto const begin  = pack_infos[bin_idx * 2];
    auto const length = pack_infos[bin_idx * 2 + 1];

    bins += begin;
    pidx[tidx] = begin + binary_search(vals[tidx], bins, length);
}

torch::Tensor packed_searchsorted_indexed_vals_cu(
    const torch::Tensor bins,        // [num_feats]
    const torch::Tensor pack_infos,  // [num_pack, 2]
    const torch::Tensor vals,        // [num_feats_to_search]
    const torch::Tensor vals_indices // [num_feats_to_search]
) {
    torch::TensorArg bins_arg{bins, "bins", 1};
    torch::TensorArg pack_infos_arg{pack_infos, "pack_infos", 2};
    torch::TensorArg vals_arg{vals, "vals", 3};
    torch::TensorArg vals_indices_arg{vals_indices, "vals_indices", 4};

    torch::checkDim(__func__, bins_arg, 1);
    torch::checkDim(__func__, vals_arg, 1);
    torch::checkDim(__func__, pack_infos_arg, 2);
    torch::checkDim(__func__, vals_indices_arg, 1);
    torch::checkAllSameGPU(__func__, {bins_arg, vals_arg, pack_infos_arg, vals_indices_arg});
    torch::checkAllContiguous(__func__, {bins_arg, vals_arg, pack_infos_arg, vals_indices_arg});
    torch::checkSameType(__func__, bins_arg, vals_arg);
    torch::checkSameType(__func__, pack_infos_arg, vals_indices_arg);

    torch::checkSize(__func__, bins_arg, 0, pack_infos.index({-1, 0}).item<int32_t>() + pack_infos.index({-1, 1}).item<int32_t>());
    torch::checkSize(__func__, vals_arg, 0, vals_indices.size(0));
    torch::checkSize(__func__, vals_indices_arg, 0, vals.size(0));

    TORCH_CHECK(vals_indices.max().item<int32_t>() < pack_infos.size(0), "vals_indices out of bounds");

    uint32_t const num_vals = vals.size(0);

    // pidx should always of the same size as vals;
    torch::Tensor pidx                    = at::full_like(vals, -1, vals.options().dtype(at::kInt));
    static constexpr uint32_t num_threads = 128;

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, bins.scalar_type(), "packed_searchsorted_indexed", ([&] {
                                  auto const device_guard = c10::cuda::OptionalCUDAGuard(torch::device_of(bins));
                                  kernel_packed_searchsorted_indexed<scalar_t, int32_t><<<div_round_up(num_vals, num_threads), num_threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
                                      num_vals,
                                      bins.data_ptr<scalar_t>(),
                                      vals.data_ptr<scalar_t>(),
                                      pack_infos.data_ptr<int32_t>(),
                                      vals_indices.data_ptr<int32_t>(),
                                      pidx.data_ptr<int32_t>());
                              }));

    return pidx;
}

torch::Tensor packed_searchsorted_cu(
    const torch::Tensor bins,     // [num_feats]
    const torch::Tensor vals,     // [num_pack, num_to_search]
    const torch::Tensor pack_info // [num_pack, 2]
) {
    torch::TensorArg bins_arg{bins, "bins", 1};
    torch::TensorArg vals_arg{vals, "vals", 2};
    torch::TensorArg pack_info_arg{pack_info, "pack_info", 3};

    torch::checkDim(__func__, bins_arg, 1);
    torch::checkDim(__func__, vals_arg, 2);
    torch::checkDim(__func__, pack_info_arg, 2);
    torch::checkAllSameGPU(__func__, {bins_arg, vals_arg, pack_info_arg});
    torch::checkAllContiguous(__func__, {bins_arg, vals_arg, pack_info_arg});
    torch::checkSameType(__func__, bins_arg, vals_arg);
    torch::checkScalarType(__func__, pack_info_arg, torch::kLong);
    torch::checkSize(__func__, pack_info_arg, {vals.size(0), 2});

    torch::checkSize(__func__, bins_arg, 0, pack_info.index({-1, 0}).item<int64_t>() + pack_info.index({-1, 1}).item<int64_t>());

    uint32_t const num_packs     = pack_info.size(0);
    uint32_t const num_feats     = bins.size(0);
    uint32_t const num_to_search = vals.size(1);

    // pidx should always be of the same size as vals
    torch::Tensor pidx                    = torch::full_like(vals, -1, vals.options().dtype(torch::kLong));
    static constexpr uint32_t num_threads = 128;

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, bins.scalar_type(), "packed_searchsorted", ([&] {
                                  auto const device_guard = c10::cuda::OptionalCUDAGuard(torch::device_of(bins));
                                  kernel_packed_searchsorted<scalar_t, int64_t><<<div_round_up(num_packs, num_threads), num_threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
                                      num_packs, num_feats,
                                      bins.data_ptr<scalar_t>(), vals.data_ptr<scalar_t>(),
                                      pack_info.data_ptr<int64_t>(), num_to_search, nullptr, pidx.data_ptr<int64_t>());
                              }));

    return pidx;
}

torch::Tensor packed_searchsorted_packed_vals_cu(
    const torch::Tensor bins,          // [num_feats]
    const torch::Tensor pack_info,     // [num_pack, 2]
    const torch::Tensor vals,          // [num_feats_to_search]
    const torch::Tensor vals_pack_info // [num_pack, 2]
) {
    torch::TensorArg bins_arg{bins, "bins", 1};
    torch::TensorArg pack_info_arg{pack_info, "pack_info", 2};
    torch::TensorArg vals_arg{vals, "vals", 3};
    torch::TensorArg vals_pack_info_arg{vals_pack_info, "vals_pack_info", 4};

    torch::checkDim(__func__, bins_arg, 1);
    torch::checkDim(__func__, vals_arg, 1);
    torch::checkDim(__func__, pack_info_arg, 2);
    torch::checkDim(__func__, vals_pack_info_arg, 2);
    torch::checkAllSameGPU(__func__, {bins_arg, vals_arg, pack_info_arg, vals_pack_info_arg});
    torch::checkAllContiguous(__func__, {bins_arg, vals_arg, pack_info_arg, vals_pack_info_arg});
    torch::checkSameType(__func__, bins_arg, vals_arg);
    torch::checkScalarType(__func__, pack_info_arg, torch::kLong);
    torch::checkScalarType(__func__, vals_pack_info_arg, torch::kLong);
    torch::checkSize(__func__, vals_pack_info_arg, {pack_info.size(0), 2});

    torch::checkSize(__func__, bins_arg, 0, pack_info.index({-1, 0}).item<int64_t>() + pack_info.index({-1, 1}).item<int64_t>());
    torch::checkSize(__func__, vals_arg, 0, vals_pack_info.index({-1, 0}).item<int64_t>() + vals_pack_info.index({-1, 1}).item<int64_t>());

    uint32_t const num_packs = pack_info.size(0);
    uint32_t const num_feats = bins.size(0);

    // pidx should always of the same size as vals;
    torch::Tensor pidx                    = torch::full_like(vals, -1, vals.options().dtype(torch::kLong));
    static constexpr uint32_t num_threads = 128;

    AT_DISPATCH_ALL_TYPES_AND(torch::ScalarType::Half, bins.scalar_type(), "packed_searchsorted_packed_vals", ([&] {
                                  auto const device_guard = c10::cuda::OptionalCUDAGuard(torch::device_of(bins));
                                  kernel_packed_searchsorted<scalar_t, int64_t><<<div_round_up(num_packs, num_threads), num_threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
                                      num_packs, num_feats,
                                      bins.data_ptr<scalar_t>(), vals.data_ptr<scalar_t>(),
                                      pack_info.data_ptr<int64_t>(), 0, vals_pack_info.data_ptr<int64_t>(), pidx.data_ptr<int64_t>());
                              }));

    return pidx;
}

template <typename scalar_t>
__global__ void kernel_packed_diff(
    // Inputs
    const uint32_t num_packs,
    const uint32_t num_feats,
    const uint32_t feat_dim,
    const scalar_t* __restrict__ feats_in,

    const scalar_t* __restrict__ appends,
    const scalar_t* __restrict__ last_fills,

    const int64_t* __restrict__ pack_info,
    // Outputs
    scalar_t* __restrict__ feats_out) {
    uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs)
        return;

    uint32_t begin  = pack_info[tidx * 2];
    uint32_t length = pack_info[tidx * 2 + 1];
    if (length == 0) {
        return;
    }
    uint32_t end = begin + length;

    // For loop on feat_dim first.
    for (uint32_t j = 0; j < feat_dim; ++j) {
        for (uint32_t i = begin; i < end - 1; ++i) {
            feats_out[i * feat_dim + j] = feats_in[(i + 1) * feat_dim + j] - feats_in[i * feat_dim + j];
        }
    }

    if (appends) {
        for (uint32_t j = 0; j < feat_dim; ++j) {
            feats_out[(end - 1) * feat_dim + j] = appends[tidx * feat_dim + j] - feats_in[(end - 1) * feat_dim + j];
        }
    } else if (last_fills) {
        for (uint32_t j = 0; j < feat_dim; ++j) {
            feats_out[(end - 1) * feat_dim + j] = last_fills[tidx * feat_dim + j];
        }
    }
}

template <typename scalar_t>
__global__ void kernel_packed_backward_diff(
    // Inputs
    const uint32_t num_packs,
    const uint32_t num_feats,
    const uint32_t feat_dim,
    const scalar_t* __restrict__ feats_in,
    const scalar_t* __restrict__ prepends,
    const scalar_t* __restrict__ first_fill,
    const int64_t* __restrict__ pack_info,
    // Outputs
    scalar_t* __restrict__ feats_out) {
    uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= num_packs)
        return;

    uint32_t begin  = pack_info[tidx * 2];
    uint32_t length = pack_info[tidx * 2 + 1];
    if (length == 0) {
        return;
    }
    uint32_t end = begin + length;

    // For loop on feat_dim first.
    for (uint32_t j = 0; j < feat_dim; ++j) {
        for (uint32_t i = begin + 1; i < end; ++i) {
            feats_out[i * feat_dim + j] = feats_in[i * feat_dim + j] - feats_in[(i - 1) * feat_dim + j];
        }
    }

    if (prepends) {
        for (uint32_t j = 0; j < feat_dim; ++j) {
            feats_out[begin * feat_dim + j] = feats_in[begin * feat_dim + j] - prepends[tidx * feat_dim + j];
        }
    } else if (first_fill) {
        for (uint32_t j = 0; j < feat_dim; ++j) {
            feats_out[begin * feat_dim + j] = first_fill[tidx * feat_dim + j];
        }
    }
}

torch::Tensor packed_diff_cu(
    const torch::Tensor data,                          // [num_feats, feat_dim] or [num_feats]
    const torch::Tensor pack_info,                     // [num_packs, 2]
    c10::optional<const torch::Tensor> pack_appends_,  // [num_packs, feat_dim] or [num_packs]
    c10::optional<const torch::Tensor> pack_last_fill_ // [num_packs, feat_dim] or [num_packs]
) {
    // https://en.wikipedia.org/wiki/Finite_difference
    // forward difference: [next value] - [this value]

    torch::TensorArg feats_arg{data, "data", 1};
    torch::TensorArg pack_info_arg{pack_info, "pack_info", 2};

    torch::checkDimRange(__func__, feats_arg, 1, 3); // [1, 2] is allowed.
    torch::checkDim(__func__, pack_info_arg, 2);
    torch::checkAllContiguous(__func__, {feats_arg, pack_info_arg});
    torch::checkSameGPU(__func__, feats_arg, pack_info_arg);
    torch::checkScalarType(__func__, pack_info_arg, torch::kLong);

    if ((int32_t)pack_appends_.has_value() + (int32_t)pack_last_fill_.has_value() > 1) {
        throw std::runtime_error("You should only specify AT MOST one of [appends, prepends, last_fill, first_fill]");
    }

    torch::checkSize(__func__, feats_arg, 0, pack_info.index({-1, 0}).item<int64_t>() + pack_info.index({-1, 1}).item<int64_t>());

    uint32_t num_packs = pack_info.size(0);
    uint32_t num_feats = data.size(0);
    uint32_t feat_dim  = data.dim() == 1 ? 1 : data.size(1);

    torch::Tensor pack_appends;
    if (pack_appends_.has_value()) {
        pack_appends = pack_appends_.value();
        torch::TensorArg pack_appends_arg{pack_appends, "pack_appends", 3};
        torch::checkContiguous(__func__, pack_appends_arg);
        torch::checkSameGPU(__func__, feats_arg, pack_appends_arg);
        torch::checkSameType(__func__, feats_arg, pack_appends_arg);
        if (data.dim() == 1) {
            torch::checkSize(__func__, pack_appends_arg, {num_packs});
        } else {
            torch::checkSize(__func__, pack_appends_arg, {num_packs, feat_dim});
        }
    }

    torch::Tensor pack_last_fill;
    if (pack_last_fill_.has_value()) {
        pack_last_fill = pack_last_fill_.value();
        torch::TensorArg pack_last_fill_arg{pack_last_fill, "pack_last_fill", 4};
        torch::checkContiguous(__func__, pack_last_fill_arg);
        torch::checkSameGPU(__func__, feats_arg, pack_last_fill_arg);
        torch::checkSameType(__func__, feats_arg, pack_last_fill_arg);
        if (data.dim() == 1) {
            torch::checkSize(__func__, pack_last_fill_arg, {num_packs});
        } else {
            torch::checkSize(__func__, pack_last_fill_arg, {num_packs, feat_dim});
        }
    }

    torch::Tensor feats_out = data.dim() == 1 ? torch::zeros({num_feats}, data.options()) : torch::zeros({num_feats, feat_dim}, data.options());

    int64_t* pack_info_ptr = pack_info.data_ptr<int64_t>();

    static constexpr uint32_t num_threads = 128;

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, data.scalar_type(), "packed_diff", ([&] {
                                  const c10::cuda::OptionalCUDAGuard device_guard(torch::device_of(data));
                                  auto stream = c10::cuda::getCurrentCUDAStream();
                                  kernel_packed_diff<scalar_t><<<div_round_up(num_packs, num_threads), num_threads, 0, stream>>>(
                                      num_packs, num_feats, feat_dim,
                                      data.data_ptr<scalar_t>(),
                                      pack_appends_.has_value() ? pack_appends.data_ptr<scalar_t>() : nullptr,
                                      pack_last_fill_.has_value() ? pack_last_fill.data_ptr<scalar_t>() : nullptr,
                                      pack_info_ptr, feats_out.data_ptr<scalar_t>());
                              }));
    return feats_out;
}

torch::Tensor packed_backward_diff_cu(
    const torch::Tensor data,                           // [num_feats, feat_dim]
    const torch::Tensor pack_info,                      // [num_packs, 2]
    c10::optional<const torch::Tensor> pack_prepends_,  // [num_packs, feat_dim]
    c10::optional<const torch::Tensor> pack_first_fill_ // [num_packs, feat_dim]
) {
    // https://en.wikipedia.org/wiki/Finite_difference
    // backward difference: [this value] - [prev value]

    torch::TensorArg feats_arg{data, "data", 1};
    torch::TensorArg pack_info_arg{pack_info, "pack_info", 2};

    torch::checkDimRange(__func__, feats_arg, 1, 3); // [1, 2] is allowed.
    torch::checkDim(__func__, pack_info_arg, 2);
    torch::checkAllContiguous(__func__, {feats_arg, pack_info_arg});
    torch::checkSameGPU(__func__, feats_arg, pack_info_arg);
    torch::checkScalarType(__func__, pack_info_arg, torch::kLong);

    torch::checkSize(__func__, feats_arg, 0, pack_info.index({-1, 0}).item<int64_t>() + pack_info.index({-1, 1}).item<int64_t>());

    uint32_t num_packs = pack_info.size(0);
    uint32_t num_feats = data.size(0);
    uint32_t feat_dim  = data.dim() == 1 ? 1 : data.size(1);

    torch::Tensor pack_prepends;
    if (pack_prepends_.has_value()) {
        pack_prepends = pack_prepends_.value();
        torch::TensorArg pack_prepends_arg{pack_prepends, "pack_prepends", 3};
        torch::checkContiguous(__func__, pack_prepends_arg);
        torch::checkSameGPU(__func__, feats_arg, pack_prepends_arg);
        torch::checkSameType(__func__, feats_arg, pack_prepends_arg);
        if (data.dim() == 1) {
            torch::checkSize(__func__, pack_prepends_arg, {num_packs});
        } else {
            torch::checkSize(__func__, pack_prepends_arg, {num_packs, feat_dim});
        }
    }

    torch::Tensor pack_first_fill;
    if (pack_first_fill_.has_value()) {
        pack_first_fill = pack_first_fill_.value();
        torch::TensorArg pack_first_fill_arg{pack_first_fill, "pack_first_fill", 4};
        torch::checkContiguous(__func__, pack_first_fill_arg);
        torch::checkSameGPU(__func__, feats_arg, pack_first_fill_arg);
        torch::checkSameType(__func__, feats_arg, pack_first_fill_arg);
        if (data.dim() == 1) {
            torch::checkSize(__func__, pack_first_fill_arg, {num_packs});
        } else {
            torch::checkSize(__func__, pack_first_fill_arg, {num_packs, feat_dim});
        }
    }

    torch::Tensor feats_out = data.dim() == 1 ? torch::zeros({num_feats}, data.options()) : torch::zeros({num_feats, feat_dim}, data.options());

    int64_t* pack_info_ptr = pack_info.data_ptr<int64_t>();

    static constexpr uint32_t num_threads = 128;

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, data.scalar_type(), "packed_backward_diff", ([&] {
                                  const c10::cuda::OptionalCUDAGuard device_guard(torch::device_of(data));
                                  auto stream = c10::cuda::getCurrentCUDAStream();
                                  kernel_packed_backward_diff<scalar_t><<<div_round_up(num_packs, num_threads), num_threads, 0, stream>>>(
                                      num_packs, num_feats, feat_dim,
                                      data.data_ptr<scalar_t>(),
                                      pack_prepends_.has_value() ? pack_prepends.data_ptr<scalar_t>() : nullptr,
                                      pack_first_fill_.has_value() ? pack_first_fill.data_ptr<scalar_t>() : nullptr,
                                      pack_info_ptr, feats_out.data_ptr<scalar_t>());
                              }));
    return feats_out;
}
