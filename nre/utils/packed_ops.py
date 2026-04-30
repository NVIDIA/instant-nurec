# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dataclasses import dataclass
from typing import Iterator, Optional

import torch

from torch.autograd.function import once_differentiable

from libs.packed_ops.interface import packed_ops  # type: ignore


class PackedWeightedSum(torch.autograd.Function):
    """
    Computes the weighted sum of a packed tensor representation

    Inputs:
        data: (N, M) tensor of data to be summed
        weights: (N,) weights for each sample in the data (each row)
        pack_info: (n_packs, 2) start_idx, N_samples
            meaning each entry corresponds to the a single pack,
            whose samples are [start_idx:start_idx+N_samples]

    Outputs:
        accumulated_data: (M) weighted sum of the input data along each column
    """

    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(ctx, data, weights, pack_info):
        ctx.save_for_backward(data, weights, pack_info)
        if pack_info.size(0) == 0:
            return torch.empty((0, *data.shape[1:]), dtype=data.dtype, device=data.device)
        else:
            return packed_ops.packed_weighted_sum(data, weights, pack_info)

    @staticmethod
    @once_differentiable
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dL_daccumulated_data):
        data, weights, pack_info = ctx.saved_tensors

        dL_ddata = dL_dweights = None
        if pack_info.size(0) != 0:
            dL_ddata, dL_dweights = packed_ops.packed_weighted_sum_bw(data, weights, pack_info, dL_daccumulated_data)

        return dL_ddata, dL_dweights, None


def packed_weighted_sum(
    data: torch.Tensor,
    weights: torch.Tensor,
    pack_info: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the weighted sum of a packed tensor representation
    """

    return PackedWeightedSum.apply(data, weights, pack_info.int())


def packed_weighted_sum_list(
    data: list[torch.Tensor],
    weights: torch.Tensor,
    pack_info: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """
    Computes a weighted sum of all data points along the 1st dimension
    """

    assert len(set([d.shape[0] for d in data])) == 1, "All elements need to have the same amount of data points"
    assert data[0].shape[0] == weights.shape[0], (
        "The number of data points needs to be the same as the number of weights"
    )
    assert torch.all(torch.tensor([len(d.shape) == 2 for d in data])), "All tensors need to be 2d"

    # Get the lenghts of datapoints
    chunks = torch.cumsum(torch.tensor([d.shape[1] for d in data]), 0)[:-1]

    data_cat = torch.cat(data, dim=1)

    accumulated_data = packed_weighted_sum(data_cat, weights, pack_info)

    return torch.tensor_split(accumulated_data, chunks, dim=1)


class PackedCumsum(torch.autograd.Function):
    """
    Compute the cumulative sum of a packed tensor

    Inputs:
        data: (N, m) data for which we want to compute the cumulative sum
        pack_info: (n_packs, 2) start_idx, N_samples
            meaning each entry corresponds to the a single pack,
            whose samples are [start_idx:start_idx+N_samples]

        exclusive: bool if true an exclusive cumulative sum will be computed
        reverse: bool if true a reverse (back to front) cumsum will be computed

    Outputs:
        cumsum: (N, m) cumulative sum of the input packed tensor
    """

    @staticmethod
    def forward(ctx, data, pack_info, exclusive, reverse):
        ctx.save_for_backward(pack_info)
        ctx.flags = (exclusive, reverse)
        if pack_info.size(0) == 0:
            cumsum = torch.empty((0, *data.shape[1:]), dtype=data.dtype, device=data.device)
        else:
            cumsum = packed_ops.packed_cumsum(data, pack_info, exclusive, reverse)
        return cumsum

    @staticmethod
    @once_differentiable
    def backward(ctx, dL_dcumsum):
        (pack_info,) = ctx.saved_tensors
        exclusive, reverse = ctx.flags

        dL_ddata = None
        if pack_info.size(0) != 0:
            dL_ddata = packed_ops.packed_cumsum(dL_dcumsum, pack_info, exclusive, not reverse)

        return dL_ddata, None, None, None


def packed_cumsum(
    data: torch.Tensor, pack_info: torch.Tensor, exclusive: bool = False, reverse: bool = False
) -> torch.Tensor:
    """
    Compute the cumulative sum of a packed tensor

    Inputs:
        data: (N, m) data for which we want to compute the cumulative sum
        pack_info: (n_packs, 2) start_idx, N_samples
                meaning each entry corresponds to the a single pack,
                whose samples are [start_idx:start_idx+N_samples]

        exclusive: bool if true an exclusive cumulative sum will be computed
        reverse: bool if true a reverse (back to front) cumsum will be computed

    Outputs:
        _ : (N, m) cumulative sum of the input packed tensor
    """

    return PackedCumsum.apply(data.contiguous(), pack_info.int(), exclusive, reverse)  # type: ignore


class PackedCumprod(torch.autograd.Function):
    """
    Compute the cumulative product of a packed tensor

    Inputs:
        data: (N, m) data for which we want to compute the cumulative sum
        pack_info: (n_packs, 2) start_idx, N_samples
                meaning each entry corresponds to the a single pack,
                whose samples are [start_idx:start_idx+N_samples]

        exclusive: bool if true an exclusive cumulative product will be computed
        reverse: bool if true a reverse (back to front) cumulative product will be computed

    Outputs:
        cumprod: (N, m) cumulative product of the input packed tensor
    """

    @staticmethod
    def forward(ctx, data, pack_info, exclusive, reverse):
        if pack_info.size(0) == 0:
            cumprod = torch.empty((0, *data.shape[1:]), dtype=data.dtype, device=data.device)
        else:
            cumprod = packed_ops.packed_cumprod(data, pack_info, exclusive, reverse)
        ctx.flags = (exclusive, reverse)
        ctx.save_for_backward(pack_info, cumprod, data)
        return cumprod

    @staticmethod
    @once_differentiable
    def backward(ctx, dL_dcumprod):
        # Gradient computation taken from tensorflow:
        # https://github.com/tensorflow/tensorflow/blob/51fd9c024c4544ba1ef60862ec3f55b6e3ae79b1/tensorflow/python/ops/math_grad.py#L891

        pack_info, cumprod, data = ctx.saved_tensors
        exclusive, reverse = ctx.flags

        dL_ddata = None
        if pack_info.size(0) != 0:
            out = packed_ops.packed_cumsum(dL_dcumprod * cumprod, pack_info, exclusive, not reverse)
            dL_ddata = out / data
            dL_ddata[dL_ddata.isnan()] = 0

        return dL_ddata, None, None, None


def packed_cumprod(
    data: torch.Tensor, pack_info: torch.Tensor, exclusive: bool = False, reverse: bool = False
) -> torch.Tensor:
    """
    Compute the cumulative product of a packed tensor

    Inputs:
        data: (N, m) data for which we want to compute the cumulative sum
        pack_info: (n_packs, 2) start_idx, N_samples
                meaning each entry corresponds to the a single pack,
                whose samples are [start_idx:start_idx+N_samples]

        exclusive: bool if true an exclusive cumulative product will be computed
        reverse: bool if true a reverse (back to front) cumulative product will be computed

    Outputs:
        cumprod: (N, m) cumulative product of the input packed tensor
    """

    return PackedCumprod.apply(data.contiguous(), pack_info.int(), exclusive, reverse)  # type: ignore


class PackedAdd(torch.autograd.Function):
    """Calculate pack-wise addition: data + other

    Args:
        data (torch.Tensor): [num_feats(, feat_dim)]
        other (torch.Tensor): [num_packs(, feat_dim)]
        pack_info: (n_packs, 2) start_idx, N_samples
            meaning each entry corresponds to the a single pack,
            whose samples are [start_idx:start_idx+N_samples]

    Returns:
        torch.Tensor: Pack-wise addition results
    """

    @staticmethod
    def forward(ctx, data, other, pack_info):
        ctx.save_for_backward(pack_info)
        if pack_info.size(0) == 0:
            return torch.empty((0, *data.shape[1:]), dtype=data.dtype, device=data.device)
        else:
            return packed_ops.packed_add(data, other, pack_info)

    @staticmethod
    @once_differentiable
    def backward(ctx, dL_dpacked_add):
        pack_info = ctx.saved_tensors[0]

        dL_ddata = dL_dother = None
        if pack_info.size(0) != 0:
            if ctx.needs_input_grad[0]:
                dL_ddata = dL_dpacked_add

            if ctx.needs_input_grad[1]:
                dL_dother = packed_ops.packed_sum(dL_dpacked_add, pack_info)

        return dL_ddata, dL_dother, None


def packed_add(data: torch.Tensor, other: torch.Tensor, pack_info: torch.Tensor) -> torch.Tensor:
    """Calculate pack-wise addition: data + other

    Args:
        data (torch.Tensor): [num_feats(, feat_dim)]
        other (torch.Tensor): [num_packs(, feat_dim)]
        pack_info: (n_packs, 2) start_idx, N_samples
            meaning each entry corresponds to the a single pack,
            whose samples are [start_idx:start_idx+N_samples]

    Returns:
        torch.Tensor: Pack-wise addition results
    """

    return PackedAdd.apply(data.contiguous(), other.contiguous(), pack_info.int())  # type: ignore


class PackedSub(torch.autograd.Function):
    """Calculate pack-wise subtraction: data - other

    Args:
        data (torch.Tensor): [num_feats(, feat_dim)]
        other (torch.Tensor): [num_packs(, feat_dim)]
        pack_info: (n_packs, 2) start_idx, N_samples
            meaning each entry corresponds to the a single pack,
            whose samples are [start_idx:start_idx+N_samples]

    Returns:
        torch.Tensor: Pack-wise division results
    """

    @staticmethod
    def forward(ctx, data, other, pack_info):
        ctx.save_for_backward(pack_info)
        if pack_info.size(0) == 0:
            return torch.empty((0, *data.shape[1:]), dtype=data.dtype, device=data.device)
        else:
            return packed_ops.packed_sub(data, other, pack_info)

    @staticmethod
    @once_differentiable
    def backward(ctx, dL_dpacked_sub):
        pack_info = ctx.saved_tensors[0]

        dL_ddata = dL_dother = None
        if pack_info.size(0) != 0:
            if ctx.needs_input_grad[0]:
                dL_ddata = dL_dpacked_sub
            if ctx.needs_input_grad[1]:
                dL_dother = -packed_ops.packed_sum(dL_dpacked_sub, pack_info)

        return dL_ddata, dL_dother, None


def packed_sub(data: torch.Tensor, other: torch.Tensor, pack_info: torch.Tensor) -> torch.Tensor:
    """Calculate pack-wise subtraction: data - other

    Args:
        data (torch.Tensor): [num_feats(, feat_dim)]
        other (torch.Tensor): [num_packs(, feat_dim)]
        pack_info: (n_packs, 2) start_idx, N_samples
            meaning each entry corresponds to the a single pack,
            whose samples are [start_idx:start_idx+N_samples]

    Returns:
        torch.Tensor: Pack-wise division results
    """

    return PackedSub.apply(data.contiguous(), other.contiguous(), pack_info.int())  # type: ignore


class PackedMul(torch.autograd.Function):
    """Calculate pack-wise multiplication: data * other

    Args:
        data (torch.Tensor): [num_feats(, feat_dim)]
        other (torch.Tensor): [num_packs(, feat_dim)]
        pack_info: (n_packs, 2) start_idx, N_samples
            meaning each entry corresponds to the a single pack,
            whose samples are [start_idx:start_idx+N_samples]

    Returns:
        torch.Tensor: Pack-wise multiplication results
    """

    @staticmethod
    def forward(ctx, data, other, pack_info):
        ctx.save_for_backward(data, other, pack_info)
        if pack_info.size(0) == 0:
            return torch.empty((0, *data.shape[1:]), dtype=data.dtype, device=data.device)
        else:
            return packed_ops.packed_mul(data, other, pack_info)

    @staticmethod
    @once_differentiable
    def backward(ctx, dL_dpacked_mul):
        data, other, pack_info = ctx.saved_tensors

        dL_ddata = dL_dother = None
        if pack_info.size(0) != 0:
            if ctx.needs_input_grad[0]:
                dL_ddata = packed_ops.packed_mul(dL_dpacked_mul, other, pack_info)

            if ctx.needs_input_grad[1]:
                dL_dother = packed_ops.packed_sum(dL_dpacked_mul * data, pack_info)

        return dL_ddata, dL_dother, None


def packed_mul(data: torch.Tensor, other: torch.Tensor, pack_info: torch.Tensor) -> torch.Tensor:
    """Calculate pack-wise multiplication: data * other

    Args:
        data (torch.Tensor): [num_feats(, feat_dim)]
        other (torch.Tensor): [num_packs(, feat_dim)]
        pack_info: (n_packs, 2) start_idx, N_samples
            meaning each entry corresponds to the a single pack,
            whose samples are [start_idx:start_idx+N_samples]

    Returns:
        torch.Tensor: Pack-wise multiplication results
    """

    return PackedMul.apply(data.contiguous(), other.contiguous(), pack_info.int())  # type: ignore


class PackedDiv(torch.autograd.Function):
    """Calculate pack-wise division: data / other

    Args:
        data (torch.Tensor): [num_feats(, feat_dim)]
        other (torch.Tensor): [num_packs(, feat_dim)]
        pack_info: (n_packs, 2) start_idx, N_samples
            meaning each entry corresponds to the a single pack,
            whose samples are [start_idx:start_idx+N_samples]

    Returns:
        torch.Tensor: Pack-wise division results
    """

    @staticmethod
    def forward(ctx, data, other, pack_info):
        ctx.save_for_backward(data, other, pack_info)
        if pack_info.size(0) == 0:
            return torch.empty((0, *data.shape[1:]), dtype=data.dtype, device=data.device)
        else:
            return packed_ops.packed_div(data, other, pack_info)

    @staticmethod
    @once_differentiable
    def backward(ctx, dL_dpacked_div):
        data, other, pack_info = ctx.saved_tensors

        dL_ddata = dL_dother = None
        if pack_info.size(0) != 0:
            if ctx.needs_input_grad[0]:
                dL_ddata = packed_ops.packed_div(dL_dpacked_div, other, pack_info)
            if ctx.needs_input_grad[1]:
                grad_other = packed_ops.packed_div(-dL_dpacked_div * data, other * other, pack_info)
                dL_dother = packed_ops.packed_sum(grad_other, pack_info)

        return dL_ddata, dL_dother, None


def packed_div(data: torch.Tensor, other: torch.Tensor, pack_info: torch.Tensor) -> torch.Tensor:
    """Calculate pack-wise division: data / other

    Args:
        data (torch.Tensor): [num_feats(, feat_dim)]
        other (torch.Tensor): [num_packs(, feat_dim)]
        pack_info: (n_packs, 2) start_idx, N_samples
            meaning each entry corresponds to the a single pack,
            whose samples are [start_idx:start_idx+N_samples]

    Returns:
        torch.Tensor: Pack-wise division results
    """
    return PackedDiv.apply(data.contiguous(), other.contiguous(), pack_info.int())  # type: ignore


class PackedSum(torch.autograd.Function):
    """
    Computes the sum of a packed tensor along the zero dimension

    Inputs:
        data: (N, m) data for which we want to compute the sum
        pack_info: (n_packs, 2) start_idx, N_samples
            meaning each entry corresponds to the a single pack,
            whose samples are [start_idx:start_idx+N_samples]
    Outputs:
        _ : (N_rays, m) sum of the input packed tensor
    """

    @staticmethod
    def forward(ctx, data, pack_info):
        ctx.save_for_backward(data, pack_info)
        if pack_info.size(0) == 0:
            return torch.zeros((pack_info.size(0), data.size(-1)), dtype=data.dtype, device=data.device)
        else:
            return packed_ops.packed_sum(data, pack_info)

    @staticmethod
    @once_differentiable
    def backward(ctx, dL_dsum):
        data, pack_info = ctx.saved_tensors

        dL_ddata = None
        if pack_info.size(0) != 0:
            dL_ddata = packed_ops.packed_sum_bw(data, pack_info, dL_dsum.contiguous())

        return dL_ddata, None


def packed_sum(data: torch.Tensor, pack_info: torch.Tensor) -> torch.Tensor:
    """
    Computes the sum of a packed tensor along the zero dimension

    Inputs:
        data: (N, m) data for which we want to compute the sum
        pack_info: (n_packs, 2) start_idx, N_samples
            meaning each entry corresponds to the a single pack,
            whose samples are [start_idx:start_idx+N_samples]
    Outputs:
        _ : (N_rays, m) sum of the input packed tensor
    """

    return PackedSum.apply(data.contiguous(), pack_info.int())


@torch.no_grad()
def packed_searchsorted(bins: torch.Tensor, vals: torch.Tensor, bins_pack_info: torch.Tensor) -> torch.Tensor:
    """
    Search a batch (vals) in a sorted pack (bins).
    For each pack in bins, the behavior is similar to torch.searchsorted(right=True)
        i.e. (None or bins[i-1]) < vals < (None or bins[i])
    """
    if vals.numel() == 0:
        return torch.empty_like(vals)
    elif bins.numel() == 0 or bins_pack_info.size(0) == 0:
        raise RuntimeError(
            "Should not invoke packed_searchsorted() with empty `vals` and non-empty `bins`. The results are undetermined."
        )
    return packed_ops.packed_searchsorted(bins.contiguous(), vals.contiguous(), bins_pack_info)


@torch.no_grad()
def packed_searchsorted_packed_vals(
    bins: torch.Tensor, bins_pack_info: torch.Tensor, vals: torch.Tensor, vals_pack_info: torch.Tensor
) -> torch.Tensor:
    """
    Search a pack (vals,vals_pack_info) in a sorted pack (bins)
    For each pack in bins, the behavior is similar to torch.searchsorted(right=True)
        i.e. (None or bins[i-1]) < vals < (None or bins[i])
    """
    if vals.numel() == 0 or vals_pack_info.size(0) == 0:
        return torch.empty((0, *vals.shape[1:]), dtype=vals.dtype, device=vals.device)
    elif bins.numel() == 0 or bins_pack_info.size(0) == 0:
        raise RuntimeError(
            "Should not invoke packed_searchsorted_packed_vals() with empty `vals` and non-empty `bins`. The results are undetermined."
        )
    return packed_ops.packed_searchsorted_packed_vals(
        bins.contiguous(), bins_pack_info.long(), vals.contiguous(), vals_pack_info.long()
    )


@dataclass(slots=True, frozen=True)
class ValuesAndPidx:
    """
    Contains the common result types
        - values: the resulting packed tensor [any] (n_samples, )
        - pidx: the pack index of each sample points [int] (n_samples, )
    """

    values: torch.Tensor
    pidx: torch.Tensor

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterator to support unpacking into values / pidx"""
        return iter((self.values, self.pidx))


@torch.no_grad()
def arange_interleave_simple(stop: torch.Tensor, return_idx: bool = False) -> ValuesAndPidx:
    """
    Returns:
        - values: a packed tensor, with each pack_i being the result of torch.arange(stop[i]).
        - pidx: if return_idx is True, will also return the per-sample pack indices. None by default
    """
    if stop.numel() == 0:
        return ValuesAndPidx(torch.empty_like(stop), torch.empty_like(stop, dtype=torch.long))
    return ValuesAndPidx(*packed_ops.arange_interleave(stop.contiguous(), return_idx))


@torch.no_grad()
def linstep_interleave(
    start: torch.Tensor, num_steps: torch.Tensor, step_size: torch.Tensor | int | float, return_idx: bool = False
) -> ValuesAndPidx:
    """
    Returns:
        - values: a packed tensor, with each pack_i being the result of torch.arange(start[i], stop[i], num_steps[i]),
            where stop[i] = start[i] + num_steps[i] * step_size[i]
        - pidx: if return_idx is True, will also return the per-sample pack indices. None by default
    """
    if start.numel() == 0:
        return ValuesAndPidx(torch.empty_like(start), torch.empty_like(num_steps))
    return ValuesAndPidx(
        *packed_ops.linstep_interleave(
            start.contiguous(),
            num_steps.contiguous().long(),
            step_size.contiguous() if isinstance(step_size, torch.Tensor) else step_size,
            return_idx,
        )
    )


@torch.no_grad()
def arange_interleave(
    start: torch.Tensor, stop: torch.Tensor, step_size: torch.Tensor | int | float, return_idx: bool = False
) -> ValuesAndPidx:
    """
    Returns:
        - values: a packed tensor, with each pack_i being the result of torch.arange(start[i], stop[i], step_size[i])
        - pidx: if return_idx is True, will also return the per-sample pack indices. None by default
    """
    num_steps = stop.subtract(start).div(step_size).ceil().long()
    return linstep_interleave(start, num_steps, step_size, return_idx)


@torch.no_grad()
def linspace_interleave(
    start: torch.Tensor, stop: torch.Tensor, num_steps: torch.Tensor | int, return_idx: bool = False
) -> ValuesAndPidx:
    """
    Returns:
        - values: a packed tensor, with each pack_i being the result of torch.linspace(start[i], stop[i], num_steps[i])
        - pidx: if return_idx is True, will also return the per-sample pack indices. None by default
    """
    denom = (num_steps - 1).clamp_min(1) if isinstance(num_steps, torch.Tensor) else max(1, num_steps - 1)
    step_size = (stop - start) / denom
    num_steps = (
        num_steps
        if isinstance(num_steps, torch.Tensor)
        else torch.full(start.shape, num_steps, device=start.device, dtype=torch.long)
    )
    return linstep_interleave(start, num_steps, step_size, return_idx=return_idx)


def packed_max(vals: torch.Tensor, pack_info: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-pack maximum; `vals_max = vals[indices]`; `indices` will be -1 for empty packs"""
    if pack_info.size(0) == 0:
        empty = torch.empty((0, *vals.shape[1:]), device=vals.device)
        return empty, empty

    vals_max, indices = packed_ops.packed_max(vals.squeeze(-1).contiguous(), pack_info.int())
    return vals_max, indices


def packed_min(vals: torch.Tensor, pack_info: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-pack minimum; `vals_min = vals[indices]`; `indices` will be -1 for empty packs"""
    if pack_info.size(0) == 0:
        empty = torch.empty((0, *vals.shape[1:]), device=vals.device)
        return empty, empty

    vals_min, indices = packed_ops.packed_min(vals.squeeze(-1).contiguous(), pack_info.int())
    return vals_min, indices


def merge_two_packs_sorted_aligned(
    vals_a: torch.Tensor, pack_info_a: torch.Tensor, vals_b: torch.Tensor, pack_info_b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Merge and sort two sorted packs. (both vals_a and vals_b should be sorted)
    """
    if vals_a.numel() == 0 or pack_info_a.size(0) == 0:
        inds_a = torch.empty((0,), dtype=torch.long, device=vals_a.device)
        inds_b = torch.arange(vals_b.numel(), device=vals_b.device)
        return pack_info_b, inds_b, inds_a, inds_b
    elif vals_b.numel() == 0 or pack_info_b.size(0) == 0:
        inds_a = torch.arange(vals_a.numel(), device=vals_a.device)
        inds_b = torch.empty((0,), dtype=torch.long, device=vals_a.device)
        return pack_info_a, inds_a, inds_a, inds_b

    ranks_a, ranks_b, pack_info = packed_ops.merge_two_packs_sorted_aligned_fw(
        vals_a, pack_info_a.int(), vals_b, pack_info_b.int()
    )

    indices = torch.zeros([len(vals_a) + len(vals_b)], dtype=torch.long, device=vals_a.device)
    indices[ranks_a] = torch.arange(len(vals_a), dtype=torch.long, device=vals_a.device)
    indices[ranks_b] = len(vals_a) + torch.arange(len(vals_b), dtype=torch.long, device=vals_a.device)

    return pack_info, indices, ranks_a, ranks_b


class PackedDiff(torch.autograd.Function):
    """
    Computes per-pack forward difference. out[i] = data[i+1] - data[i]. \
    Refer to `packed_diff()` for detailed documentation.
    """

    @staticmethod
    def forward(ctx, data, pack_info, pack_appends, pack_last_fill):
        ctx.save_for_backward(pack_info)
        ctx.flags = (pack_appends is not None, pack_last_fill is not None)
        if pack_info.size(0) == 0:
            return torch.empty((0, *data.shape[1:]), dtype=data.dtype, device=data.device)
        else:
            return packed_ops.packed_diff(data, pack_info, pack_appends, pack_last_fill)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (pack_info,) = ctx.saved_tensors
        has_append, has_last_fill = ctx.flags

        grad_feat = grad_append = grad_last_fill = None
        if pack_info.size(0) != 0:
            first_inds, n_per_pack = pack_info[..., 0], pack_info[..., 1]
            non_empty = n_per_pack > 0
            last_inds = first_inds + n_per_pack - 1
            if ctx.needs_input_grad[0]:
                # [-dL0, dL0-dL1, dL1-dL2, ..., dL(n-2)-dL(n-1)]
                grad_feat = -1 * packed_ops.packed_backward_diff(grad_output, pack_info, None, grad_output[first_inds])
                if not has_append:
                    second_last = grad_output[last_inds - 1]
                    grad_feat[last_inds[non_empty]] = torch.where(
                        n_per_pack[non_empty] > 1, second_last[non_empty], grad_output.new_tensor([0.0])
                    )

            if has_append and ctx.needs_input_grad[2]:
                grad_append = grad_output[last_inds]
                grad_append[non_empty.logical_not()] = 0

            if has_last_fill and ctx.needs_input_grad[3]:
                grad_last_fill = grad_output[last_inds]
                grad_last_fill[non_empty.logical_not()] = 0

        return grad_feat, None, grad_append, grad_last_fill


def packed_diff(
    data: torch.Tensor,
    pack_info: torch.Tensor,
    pack_appends: Optional[torch.Tensor] = None,
    pack_last_fill: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes per-pack forward difference. out[i] = data[i+1] - data[i]. \
    See `pack_appends` and `pack_last_fill` for how to handle boundaries. \
    By default, the behavior is equivalent to pack_last_fill = torch.zeros() \
    This operation is differentiable.
    
    The behavior is similar to pytorch, only differs by adding a zero padding \
    to keep the pack_info un-changed when `pack_appends` is not specified.
    e.g.:
        packed_diff([3,5,10]]) = [2,5,0]
        packed_diff([3,5]]) = [2,0]
        packed_diff([3]]) = [0]
        packed_diff([]) = []
    If `pack_appends` is provided, then:
        packed_diff([3,5]]) = [2,appended_value-5]
        packed_diff([3]]) = [appended_value-3]
        packed_diff([]]) = []
    Or, if `pack_last_fill` is provided, then:
        packed_diff([3,5]]) = [2,filled_value]
        packed_diff([3]]) = [filled_value]
        packed_diff([]]) = []
    
    Args:
        data (torch.Tensor): (n_data,) or (n_data, n_feat_dim) packed tensor data
        pack_info (torch.Tensor): (n_packs, 2) start_idx, N_samples
            meaning each entry corresponds to the a single pack,
            whose samples are [start_idx:start_idx+N_samples]
        pack_appends (Optional[torch.Tensor]): (n_packs,) \
            Optional values appended to each pack's tail before diff. Defaults to None.
        pack_last_fill (Optional[torch.Tensor]): (n_packs,). \
            Optional values filled into the last item of each diff result pack. Defaults to None.
            Should not specify both of `pack_appends` and `pack_last_fill` simultaneously

    Returns:
        torch.Tensor: (n_data,) or (n_data, n_feat_dim) packed diff results
    """

    return PackedDiff.apply(data.contiguous(), pack_info.long(), pack_appends, pack_last_fill)


class PackedBackwardDiff(torch.autograd.Function):
    """
    Computes per-pack backward difference. out[i] = data[i] - data[i-1]. \
    Refer to `packed_backward_diff()` for detailed documentation.
    """

    @staticmethod
    def forward(ctx, data, pack_info, pack_prepends, pack_first_fill):
        ctx.save_for_backward(pack_info)
        ctx.flags = (pack_prepends is not None, pack_first_fill is not None)
        if pack_info.size(0) == 0:
            return torch.empty((0, *data.shape[1:]), dtype=data.dtype, device=data.device)
        else:
            return packed_ops.packed_backward_diff(data, pack_info, pack_prepends, pack_first_fill)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (pack_info,) = ctx.saved_tensors
        has_prepend, has_first_fill = ctx.flags

        grad_feat = grad_prepend = grad_first_fill = None
        if pack_info.size(0) != 0:
            first_inds, n_per_pack = pack_info[..., 0], pack_info[..., 1]
            non_empty = n_per_pack > 0
            last_inds = first_inds + n_per_pack - 1
            if ctx.needs_input_grad[0]:
                grad_feat = -1 * packed_ops.packed_diff(grad_output, pack_info, None, -grad_output[last_inds])
                if not has_prepend:
                    second = grad_output[first_inds + 1]
                    grad_feat[first_inds[non_empty]] = torch.where(
                        n_per_pack[non_empty] > 1, -second[non_empty], grad_output.new_tensor([0.0])
                    )

            if has_prepend or ctx.needs_input_grad[2]:
                grad_prepend = -1 * grad_output[first_inds]
                grad_prepend[non_empty.logical_not()] = 0

            if has_first_fill or ctx.needs_input_grad[3]:
                grad_first_fill = grad_output[first_inds]
                grad_first_fill[non_empty.logical_not()] == 0

        return grad_feat, None, grad_prepend, grad_first_fill


def packed_backward_diff(
    data: torch.Tensor,
    pack_info: torch.Tensor,
    pack_prepends: Optional[torch.Tensor] = None,
    pack_first_fill: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes per-pack backward difference. out[i] = data[i] - data[i-1]. \
    See `pack_prepends` and `pack_first_fill` for how to handle boundaries. \
    By default, the behavior is equivalent to pack_first_fill = torch.zeros() \
    This operation is differentiable.

    The behavior is similar to pytorch, only differs by adding a zero padding \
    to keep the pack_info un-changed when `pack_prepends` is not specified.
    e.g.:
        packed_backward_diff([3,5,10]]) = [0,2,5]
        packed_backward_diff([3,5]]) = [0,2]
        packed_backward_diff([3]]) = [0]
        packed_backward_diff([]) = []
    If `pack_appends` is provided, then:
        packed_backward_diff([3,5]]) = [3-prepended_value,2]
        packed_backward_diff([3]]) = [3-prepended_value]
        packed_backward_diff([]]) = []
    Or, if `pack_first_fill` is provided, then:
        packed_backward_diff([3,5]]) = [filled_value,2]
        packed_backward_diff([3]]) = [filled_value]
        packed_backward_diff([]]) = []

    Args:
        data (torch.Tensor): (n_data,) or (n_data, n_feat_dim) packed tensor data
        pack_info (torch.Tensor): (n_packs, 2) start_idx, N_samples
            meaning each entry corresponds to the a single pack,
            whose samples are [start_idx:start_idx+N_samples]
        pack_prepends (Optional[torch.Tensor]): (n_packs,) \
            Optional values prepended to each pack's head before diff. Defaults to None.
        pack_first_fill (Optional[torch.Tensor]): (n_packs,). \
            Optional values filled into the first item of each diff result pack. Defaults to None.
            Should not specify both of `pack_prepends` and `pack_first_fill` simultaneously

    Returns:
        torch.Tensor: (n_data,) or (n_data, n_feat_dim) packed diff results
    """

    return PackedBackwardDiff.apply(data.contiguous(), pack_info.long(), pack_prepends, pack_first_fill)


@torch.no_grad()
def packed_invert_cdf(
    bins: torch.Tensor,
    cdfs: torch.Tensor,
    bins_pack_info: torch.Tensor,
    u: torch.Tensor,
    u_pack_info: torch.Tensor,
    eps: float = 1.0e-7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Conduct packed version of inverse CDF sampling

    Args:
        bins (torch.Tensor): The endpoints of the contiguous intervals
        cdfs (torch.Tensor): CDF value of each bin point
        bins_pack_info (torch.Tensor): Pack info for bins
        u (torch.Tensor): The drawn uniform samples for inverse CDF sampling, within range [0,1]
        u_pack_info (torch.Tensor): Pack info for `u`

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
        - samples: Samples in bins, sampled via inverse CDF sampling
        - bin_idx: The bin indices that each sample belongs to
    """
    if u.numel() == 0:
        return torch.empty_like(u), torch.empty_like(u)

    if u_pack_info.size(0) == 0:
        return torch.empty_like(u[:0]), torch.empty_like(u[:0])

    if bins.numel() == 0 or bins_pack_info.size(0) == 0:
        raise RuntimeError(
            "Should not invoke packed_invert_cdf() with empty `bins` and non-empty `u`. The results are undetermined."
        )
    return packed_ops.packed_invert_cdf(bins, cdfs, bins_pack_info.int(), u, u_pack_info.int(), eps)


@torch.no_grad()
def packed_interp(
    bins: torch.Tensor,
    vals: torch.Tensor,
    bins_pack_info: torch.Tensor,
    query_pts: torch.Tensor,
    query_pack_info: torch.Tensor,
    eps: float = 1.0e-7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-pack one-dimensional linear interpolation for monotonically increasing sample points.
    Returns:
        tuple[torch.Tensor, torch.Tensor]:
        - samples: one-dimensional piecewise linear interpolant to a function with given discrete data points (`bins`, `vals`), evaluated at `query_pts`.
        - bin_idx: The bin indices that each interpolant belongs to
    """
    if query_pts.numel() == 0:
        return torch.empty_like(query_pts), torch.empty_like(query_pts)

    if query_pack_info.size(0) == 0:
        return torch.empty_like(query_pts[:0]), torch.empty_like(query_pts[:0])

    if bins.numel() == 0 or bins_pack_info.size(0) == 0:
        raise RuntimeError(
            "Should not invoke packed_interp() with empty `bins` and non-empty `query_pts`. The results are undetermined."
        )
    return packed_ops.packed_interp(bins, vals, bins_pack_info.int(), query_pts, query_pack_info.int(), eps)
