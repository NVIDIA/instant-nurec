# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Phase 1 step 4.3: predict-only standalone keeps just the
# linstep_interleave helper used by tracks.py. The rest of the
# packed-ops surface (PackedWeightedSum / PackedCumsum / PackedAdd /
# PackedSub / PackedMul / PackedDiv / PackedSum / packed_searchsorted /
# arange_interleave / linspace_interleave / packed_max / packed_min /
# merge_two_packs_sorted_aligned / PackedDiff / PackedBackwardDiff /
# packed_invert_cdf / packed_interp) was training-side and is removed.

from __future__ import annotations

from dataclasses import dataclass

import torch

from instant_nurec._pkg.utils._packed_ops_torch import (
    linstep_interleave as _linstep_interleave_torch,
)


@dataclass(slots=True, frozen=True)
class ValuesAndPidx:
    values: torch.Tensor
    pidx: torch.Tensor


@torch.no_grad()
def linstep_interleave(
    start: torch.Tensor,
    num_steps: torch.Tensor,
    step_size: torch.Tensor | int | float,
    return_idx: bool = False,
) -> ValuesAndPidx:
    """Returns interleaved per-pack arange-style sequences, one step_size apart."""
    if start.numel() == 0:
        return ValuesAndPidx(torch.empty_like(start), torch.empty_like(num_steps))
    out, nidx = _linstep_interleave_torch(
        start.contiguous(),
        num_steps.contiguous().long(),
        step_size.contiguous() if isinstance(step_size, torch.Tensor) else step_size,
        return_idx,
    )
    return ValuesAndPidx(out, nidx if nidx is not None else torch.empty_like(num_steps))
