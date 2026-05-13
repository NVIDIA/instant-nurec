# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Two packed-array helpers consumed by the predict path:
``linstep_interleave`` and ``packed_searchsorted_indexed_vals``.

The "pack" surface is a flat tensor partitioned by a (P, 2) packinfo
tensor of ``(start_offset, count)`` rows. These helpers operate on the
per-pack semantics without materialising the row-major segments.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def linstep_interleave(
    start: torch.Tensor,
    num_steps: torch.Tensor,
    step_size: torch.Tensor | float | int,
) -> torch.Tensor:
    """Per-pack arange-style interleaved sequence.

    For each pack ``p``, emits ``num_steps[p]`` values
    ``start[p], start[p] + step_size, start[p] + 2*step_size, ..``
    Output is the flat concatenation of all packs' sequences.

    Args:
        start: (P,) per-pack starting value. Same dtype as the desired output.
        num_steps: (P,) int64 number of values per pack.
        step_size: scalar (int/float) or (P,) tensor of per-pack step sizes
            (same dtype as ``start``).

    Returns:
        (N,) tensor with N = num_steps.sum() and dtype matching ``start``.
    """
    if start.numel() == 0:
        return torch.empty_like(start)

    start = start.contiguous()
    num_steps_long = num_steps.contiguous().long()
    if isinstance(step_size, torch.Tensor):
        step_size = step_size.contiguous()

    num_packs = start.shape[0]
    if num_packs == 0:
        return torch.empty(0, dtype=start.dtype, device=start.device)

    total = int(num_steps_long.sum().item())
    if total == 0:
        return torch.empty(0, dtype=start.dtype, device=start.device)

    # Per-element offset within its pack: 0, 1, ..., num_steps[p]-1.
    cum = num_steps_long.cumsum(0)
    pack_starts = cum - num_steps_long
    within = torch.arange(total, dtype=torch.int64, device=start.device) - torch.repeat_interleave(
        pack_starts, num_steps_long
    )

    start_per_elem = torch.repeat_interleave(start, num_steps_long)
    if isinstance(step_size, torch.Tensor):
        step_per_elem = torch.repeat_interleave(step_size, num_steps_long)
        return start_per_elem + within.to(start.dtype) * step_per_elem
    return start_per_elem + within.to(start.dtype) * start.new_tensor(step_size)


def packed_searchsorted_indexed_vals(
    bins: torch.Tensor,
    pack_infos: torch.Tensor,
    vals: torch.Tensor,
    vals_indices: torch.Tensor,
) -> torch.Tensor:
    """Per-pack lower-bound binary search returning global indices.

    For each query ``i``, finds the smallest ``j`` in
    ``[pack_infos[vals_indices[i], 0], pack_infos[vals_indices[i], 0] + pack_infos[vals_indices[i], 1]]``
    such that ``bins[j] >= vals[i]``, and returns ``j`` (a global index into
    the flat ``bins`` tensor).

    Args:
        bins: (num_feats,) flat sorted-within-each-pack values (int or float).
        pack_infos: (num_pack, 2) int32, columns ``(start, count)``.
        vals: (num_vals,) queries; same dtype as ``bins``.
        vals_indices: (num_vals,) int32, which pack each query targets.

    Returns:
        (num_vals,) int32 tensor of global lower-bound indices.

    Strategy: shift each pack's bins by a per-pack offset large enough to
    make the global ``shifted_bins`` strictly sorted across pack boundaries,
    apply the same shift to ``vals``, then run a single ``torch.searchsorted``.
    Avoids a per-query python loop while preserving exact lower-bound semantics.
    """
    if vals.numel() == 0:
        return torch.empty(0, dtype=torch.int32, device=vals.device)

    num_packs = pack_infos.shape[0]
    bins_min = bins.min()
    bins_max = bins.max()
    # Per-pack offset: enough to push pack p+1's smallest bin above pack p's largest.
    bin_range = bins_max - bins_min + bins.new_tensor(1)
    pack_lengths = pack_infos[:, 1].to(torch.int64)
    pack_idx_per_bin = torch.repeat_interleave(
        torch.arange(num_packs, dtype=torch.int64, device=bins.device), pack_lengths
    )
    shifted_bins = bins + bin_range * pack_idx_per_bin.to(bins.dtype)
    shifted_vals = vals + bin_range * vals_indices.to(vals.dtype)

    raw = torch.searchsorted(shifted_bins, shifted_vals).to(torch.int64)
    # Queries past their pack's max can spill into neighbouring packs in the
    # globally-shifted space — clamp each result back into its query's pack range.
    starts_int = pack_infos[:, 0].to(torch.int64)
    q_idx = vals_indices.to(torch.int64)
    q_starts = starts_int[q_idx]
    q_ends = q_starts + pack_lengths[q_idx]
    return raw.clamp(min=q_starts, max=q_ends).to(torch.int32)
