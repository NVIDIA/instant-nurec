// SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include "utils.h"

#include <ku/common_host.h>
#include <vector>

std::tuple<torch::Tensor, torch::Tensor> arange_interleave(
    torch::Tensor stop,
    bool return_nidx) {
    CHECK_INPUT(stop);
    return arange_interleave_cu(stop, return_nidx);
}

std::tuple<torch::Tensor, torch::Tensor> linstep_interleave(
    torch::Tensor start, torch::Tensor num_steps, torch::Tensor step_size,
    bool return_nidx) {
    return linstep_interleave_cu(start, num_steps, step_size, return_nidx);
}

std::tuple<torch::Tensor, torch::Tensor> linstep_interleave(
    torch::Tensor start, torch::Tensor num_steps, double step_size,
    bool return_nidx) {
    return linstep_interleave_cu(start, num_steps, step_size, return_nidx);
}

std::tuple<torch::Tensor, torch::Tensor> linstep_interleave(
    torch::Tensor start, torch::Tensor num_steps, int32_t step_size,
    bool return_nidx) {
    return linstep_interleave_cu(start, num_steps, step_size, return_nidx);
}

torch::Tensor packed_searchsorted_indexed_vals(
    const torch::Tensor bins,        // [num_feats]
    const torch::Tensor pack_infos,  // [num_pack, 2]
    const torch::Tensor vals,        // [num_feats_to_search]
    const torch::Tensor vals_indices // [num_feats_to_search]
) {
    CHECK_INPUT(bins);
    CHECK_INPUT(pack_infos);
    CHECK_INPUT(vals);
    CHECK_INPUT(vals_indices);

    return packed_searchsorted_indexed_vals_cu(bins, pack_infos, vals, vals_indices);
}
torch::Tensor packed_weighted_sum(
    const torch::Tensor data,
    const torch::Tensor weights,
    const torch::Tensor pack_info) {
    CHECK_INPUT(data);
    CHECK_INPUT(weights);
    CHECK_INPUT(pack_info);

    return packed_weighted_sum_cu(data, weights, pack_info);
}

std::tuple<torch::Tensor, torch::Tensor> packed_weighted_sum_bw(
    const torch::Tensor data,
    const torch::Tensor weights,
    const torch::Tensor pack_info,
    const torch::Tensor dL_daccumulated_data) {
    CHECK_INPUT(data);
    CHECK_INPUT(weights);
    CHECK_INPUT(pack_info);
    CHECK_INPUT(dL_daccumulated_data);

    return packed_weighted_sum_bw_cu(data, weights, pack_info, dL_daccumulated_data);
}

std::tuple<torch::Tensor, torch::Tensor> packed_min(
    const torch::Tensor weights,
    const torch::Tensor pack_info) {
    CHECK_INPUT(weights);
    CHECK_INPUT(pack_info);

    return packed_min_cu(weights, pack_info);
}

std::tuple<torch::Tensor, torch::Tensor> packed_max(
    const torch::Tensor weights,
    const torch::Tensor pack_info) {
    CHECK_INPUT(weights);
    CHECK_INPUT(pack_info);

    return packed_max_cu(weights, pack_info);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> merge_two_packs_sorted_aligned_fw(
    const torch::Tensor vals_a,
    const torch::Tensor pack_info_a,
    const torch::Tensor vals_b,
    const torch::Tensor pack_info_b) {
    CHECK_INPUT(vals_a);
    CHECK_INPUT(pack_info_a);
    CHECK_INPUT(vals_b);
    CHECK_INPUT(pack_info_b);

    return merge_two_packs_sorted_aligned_fw_cu(vals_a, pack_info_a, vals_b, pack_info_b);
}

torch::Tensor packed_cumsum(
    const torch::Tensor data,
    const torch::Tensor pack_info,
    bool exclusive,
    bool reverse) {
    CHECK_INPUT(data);
    CHECK_INPUT(pack_info);

    return packed_cumsum_cu(data, pack_info, exclusive, reverse);
}

torch::Tensor packed_cumprod(
    const torch::Tensor data,
    const torch::Tensor pack_info,
    bool exclusive,
    bool reverse) {
    CHECK_INPUT(data);
    CHECK_INPUT(pack_info);

    return packed_cumprod_cu(data, pack_info, exclusive, reverse);
}

torch::Tensor packed_add(
    const torch::Tensor data,
    const torch::Tensor other,
    const torch::Tensor pack_info) {
    CHECK_INPUT(data);
    CHECK_INPUT(other);
    CHECK_INPUT(pack_info);

    return packed_add_cu(data, other, pack_info);
}

torch::Tensor packed_sub(
    const torch::Tensor data,
    const torch::Tensor other,
    const torch::Tensor pack_info) {
    CHECK_INPUT(data);
    CHECK_INPUT(other);
    CHECK_INPUT(pack_info);

    return packed_sub_cu(data, other, pack_info);
}

torch::Tensor packed_mul(
    const torch::Tensor data,
    const torch::Tensor other,
    const torch::Tensor pack_info) {
    CHECK_INPUT(data);
    CHECK_INPUT(other);
    CHECK_INPUT(pack_info);

    return packed_mul_cu(data, other, pack_info);
}

torch::Tensor packed_div(
    const torch::Tensor data,
    const torch::Tensor other,
    const torch::Tensor pack_info) {
    CHECK_INPUT(data);
    CHECK_INPUT(other);
    CHECK_INPUT(pack_info);

    return packed_div_cu(data, other, pack_info);
}

std::tuple<torch::Tensor, torch::Tensor> packed_invert_cdf(
    const torch::Tensor bins,
    const torch::Tensor cdfs,
    const torch::Tensor bins_pack_info,
    const torch::Tensor u_vals,
    const torch::Tensor u_pack_info,
    const float eps) {
    CHECK_INPUT(bins);
    CHECK_INPUT(cdfs);
    CHECK_INPUT(bins_pack_info);
    CHECK_INPUT(u_vals);
    CHECK_INPUT(u_pack_info);

    return packed_invert_cdf_cu(bins, cdfs, bins_pack_info, u_vals, u_pack_info, eps);
}

std::tuple<torch::Tensor, torch::Tensor> packed_interp(
    const torch::Tensor bins,
    const torch::Tensor vals,
    const torch::Tensor bins_pack_info,
    const torch::Tensor query_pts,
    const torch::Tensor query_pack_info,
    const float eps) {
    CHECK_INPUT(bins);
    CHECK_INPUT(vals);
    CHECK_INPUT(bins_pack_info);
    CHECK_INPUT(query_pts);
    CHECK_INPUT(query_pack_info);

    return packed_interp_cu(bins, vals, bins_pack_info, query_pts, query_pack_info, eps);
}

torch::Tensor packed_sum(
    const torch::Tensor data,
    const torch::Tensor pack_info) {
    CHECK_INPUT(data);
    CHECK_INPUT(pack_info);

    return packed_sum_cu(data, pack_info);
}

torch::Tensor packed_sum_bw(
    const torch::Tensor data,
    const torch::Tensor pack_info,
    const torch::Tensor dL_dsum) {
    CHECK_INPUT(data);
    CHECK_INPUT(pack_info);
    CHECK_INPUT(dL_dsum);

    return packed_sum_bw_cu(data, pack_info, dL_dsum);
}

torch::Tensor packed_searchsorted(
    const torch::Tensor bins,     // [num_feats]
    const torch::Tensor vals,     // [num_pack, num_to_search]
    const torch::Tensor pack_info // [num_pack, 2]
) {
    CHECK_INPUT(bins);
    CHECK_INPUT(vals);
    CHECK_INPUT(pack_info);

    return packed_searchsorted_cu(bins, vals, pack_info);
}

torch::Tensor packed_searchsorted_packed_vals(
    const torch::Tensor bins,          // [num_feats]
    const torch::Tensor pack_info,     // [num_pack, 2]
    const torch::Tensor vals,          // [num_feats_to_search]
    const torch::Tensor vals_pack_info // [num_pack, 2]
) {
    CHECK_INPUT(bins);
    CHECK_INPUT(pack_info);
    CHECK_INPUT(vals);
    CHECK_INPUT(vals_pack_info);

    return packed_searchsorted_packed_vals_cu(bins, pack_info, vals, vals_pack_info);
}

torch::Tensor packed_diff(
    const torch::Tensor feats,
    const torch::Tensor pack_info,
    c10::optional<const torch::Tensor> pack_appends_,
    c10::optional<const torch::Tensor> pack_last_fill_) {
    CHECK_INPUT(feats);
    CHECK_INPUT(pack_info);
    if (pack_appends_.has_value()) {
        CHECK_INPUT(pack_appends_.value());
    }
    if (pack_last_fill_.has_value()) {
        CHECK_INPUT(pack_last_fill_.value());
    }
    TORCH_CHECK(!(pack_appends_.has_value() and pack_last_fill_.has_value()));
    return packed_diff_cu(feats, pack_info, pack_appends_, pack_last_fill_);
}

torch::Tensor packed_backward_diff(
    const torch::Tensor feats,
    const torch::Tensor pack_info,
    c10::optional<const torch::Tensor> pack_prepends_,
    c10::optional<const torch::Tensor> pack_first_fill_) {
    CHECK_INPUT(feats);
    CHECK_INPUT(pack_info);
    if (pack_prepends_.has_value()) {
        CHECK_INPUT(pack_prepends_.value());
    }
    if (pack_first_fill_.has_value()) {
        CHECK_INPUT(pack_first_fill_.value());
    }
    TORCH_CHECK(!(pack_prepends_.has_value() and pack_first_fill_.has_value()));
    return packed_backward_diff_cu(feats, pack_info, pack_prepends_, pack_first_fill_);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("merge_two_packs_sorted_aligned_fw", &merge_two_packs_sorted_aligned_fw);
    m.def("arange_interleave", &arange_interleave);
    m.def("linstep_interleave", py::overload_cast<torch::Tensor, torch::Tensor, torch::Tensor, bool>(&linstep_interleave));
    m.def("linstep_interleave", py::overload_cast<torch::Tensor, torch::Tensor, double, bool>(&linstep_interleave));
    m.def("linstep_interleave", py::overload_cast<torch::Tensor, torch::Tensor, int32_t, bool>(&linstep_interleave));
    m.def("packed_cumsum", &packed_cumsum);
    m.def("packed_cumprod", &packed_cumprod);
    m.def("packed_add", &packed_add);
    m.def("packed_sub", &packed_sub);
    m.def("packed_mul", &packed_mul);
    m.def("packed_div", &packed_div);
    m.def("packed_sum", &packed_sum);
    m.def("packed_sum_bw", &packed_sum_bw);
    m.def("packed_diff", &packed_diff);
    m.def("packed_backward_diff", &packed_backward_diff);
    m.def("packed_weighted_sum", &packed_weighted_sum);
    m.def("packed_weighted_sum_bw", &packed_weighted_sum_bw);
    m.def("packed_invert_cdf", &packed_invert_cdf);
    m.def("packed_interp", &packed_interp);
    m.def("packed_searchsorted", &packed_searchsorted);
    m.def("packed_searchsorted_indexed_vals", &packed_searchsorted_indexed_vals);
    m.def("packed_searchsorted_packed_vals", &packed_searchsorted_packed_vals);
    m.def("packed_max", &packed_max);
    m.def("packed_min", &packed_min);
}
