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

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> arange_interleave_cu(
    const torch::Tensor stop,
    bool return_nidx);

std::tuple<torch::Tensor, torch::Tensor> linstep_interleave_cu(
    const torch::Tensor start,
    const torch::Tensor num_steps,
    const torch::Tensor step_size,
    bool return_nidx);

std::tuple<torch::Tensor, torch::Tensor> linstep_interleave_cu(
    const torch::Tensor start,
    const torch::Tensor num_steps,
    double step_size,
    bool return_nidx);

std::tuple<torch::Tensor, torch::Tensor> linstep_interleave_cu(
    const torch::Tensor start,
    const torch::Tensor num_steps,
    int32_t step_size,
    bool return_nidx);

torch::Tensor packed_weighted_sum_cu(
    const torch::Tensor data,
    const torch::Tensor weights,
    const torch::Tensor pack_info);

std::tuple<torch::Tensor, torch::Tensor> packed_weighted_sum_bw_cu(
    const torch::Tensor data,
    const torch::Tensor weights,
    const torch::Tensor pack_info,
    const torch::Tensor dL_daccumulated_data);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> merge_two_packs_sorted_aligned_fw_cu(
    const torch::Tensor vals_a,
    const torch::Tensor pack_info_a,
    const torch::Tensor vals_b,
    const torch::Tensor pack_info_b);

torch::Tensor packed_cumsum_cu(
    const torch::Tensor data,
    const torch::Tensor pack_info,
    bool exclusive,
    bool reverse);

torch::Tensor packed_cumprod_cu(
    const torch::Tensor data,
    const torch::Tensor pack_info,
    bool exclusive,
    bool reverse);

torch::Tensor packed_add_cu(
    const torch::Tensor data,
    const torch::Tensor other,
    const torch::Tensor pack_info);

torch::Tensor packed_sub_cu(
    const torch::Tensor data,
    const torch::Tensor other,
    const torch::Tensor pack_info);

torch::Tensor packed_mul_cu(
    const torch::Tensor data,
    const torch::Tensor other,
    const torch::Tensor pack_info);

torch::Tensor packed_div_cu(
    const torch::Tensor data,
    const torch::Tensor other,
    const torch::Tensor pack_info);

std::tuple<torch::Tensor, torch::Tensor> packed_invert_cdf_cu(
    const torch::Tensor bins,
    const torch::Tensor cdfs,
    const torch::Tensor bins_pack_info,
    const torch::Tensor u_vals,
    const torch::Tensor u_pack_info,
    const float eps);

std::tuple<torch::Tensor, torch::Tensor> packed_interp_cu(
    const torch::Tensor bins,
    const torch::Tensor vals,
    const torch::Tensor bins_pack_info,
    const torch::Tensor query_pts,
    const torch::Tensor query_pack_info,
    const float eps);

torch::Tensor packed_sum_cu(
    const torch::Tensor data,
    const torch::Tensor pack_info);

torch::Tensor packed_sum_bw_cu(
    const torch::Tensor data,
    const torch::Tensor pack_info,
    const torch::Tensor dL_dsum);

torch::Tensor packed_searchsorted_cu(
    const torch::Tensor bins,
    const torch::Tensor vals,
    const torch::Tensor pack_info);

torch::Tensor packed_searchsorted_packed_vals_cu(
    const torch::Tensor bins,
    const torch::Tensor pack_info,
    const torch::Tensor vals,
    const torch::Tensor vals_pack_info);

torch::Tensor packed_searchsorted_indexed_vals_cu(
    const torch::Tensor bins,
    const torch::Tensor pack_infos,
    const torch::Tensor vals,
    const torch::Tensor vals_indices);

std::tuple<torch::Tensor, torch::Tensor> packed_min_cu(
    const torch::Tensor vals,
    const torch::Tensor pack_info);

std::tuple<torch::Tensor, torch::Tensor> packed_max_cu(
    const torch::Tensor vals,
    const torch::Tensor pack_info);

torch::Tensor packed_diff_cu(
    const torch::Tensor data,
    const torch::Tensor pack_info,
    c10::optional<const torch::Tensor> pack_appends_,
    c10::optional<const torch::Tensor> pack_last_fill_);

torch::Tensor packed_backward_diff_cu(
    const torch::Tensor data,
    const torch::Tensor pack_info,
    c10::optional<const torch::Tensor> pack_prepends_,
    c10::optional<const torch::Tensor> pack_first_fill_);
