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

"""Branch-coverage tests for ``instant_nurec.utils.packed_ops``."""

from __future__ import annotations

import sys

from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


from instant_nurec.utils.packed_ops import (  # noqa: E402
    linstep_interleave,
    packed_searchsorted_indexed_vals,
)


# ---------------------------------------------------------------------------
# linstep_interleave
# ---------------------------------------------------------------------------


def test_linstep_interleave_empty_start_short_circuits():
    out = linstep_interleave(
        start=torch.empty(0, dtype=torch.float32),
        num_steps=torch.empty(0, dtype=torch.int64),
        step_size=1.0,
    )
    assert out.numel() == 0


def test_linstep_interleave_simple_three_packs_int_step():
    start = torch.tensor([0.0, 10.0, 100.0])
    num_steps = torch.tensor([3, 2, 4], dtype=torch.int64)
    out = linstep_interleave(start, num_steps, 1)
    expected = torch.tensor([0.0, 1.0, 2.0, 10.0, 11.0, 100.0, 101.0, 102.0, 103.0])
    assert torch.equal(out, expected)


def test_linstep_interleave_emits_per_pack_arange_sequence():
    start = torch.tensor([0.0, 5.0])
    num_steps = torch.tensor([2, 3])
    out = linstep_interleave(start=start, num_steps=num_steps, step_size=0.5)
    assert torch.allclose(out, torch.tensor([0.0, 0.5, 5.0, 5.5, 6.0]))


def test_linstep_interleave_with_per_pack_step_size_tensor():
    start = torch.tensor([0.0, 10.0])
    num_steps = torch.tensor([3, 2], dtype=torch.int64)
    step_size = torch.tensor([0.5, 2.0])
    out = linstep_interleave(start, num_steps, step_size)
    expected = torch.tensor([0.0, 0.5, 1.0, 10.0, 12.0])
    assert torch.allclose(out, expected)


def test_linstep_interleave_per_pack_tensor_step_size():
    out = linstep_interleave(
        start=torch.tensor([0.0, 10.0]),
        num_steps=torch.tensor([2, 2]),
        step_size=torch.tensor([0.5, 1.0]),
    )
    assert torch.allclose(out, torch.tensor([0.0, 0.5, 10.0, 11.0]))


def test_linstep_interleave_zero_steps_packs_yield_no_elements():
    start = torch.tensor([5.0, 7.0, 9.0])
    num_steps = torch.tensor([0, 3, 0], dtype=torch.int64)
    out = linstep_interleave(start, num_steps, 1)
    assert torch.equal(out, torch.tensor([7.0, 8.0, 9.0]))


def test_linstep_interleave_all_packs_zero_yield_empty_output():
    start = torch.tensor([5.0, 7.0])
    num_steps = torch.tensor([0, 0], dtype=torch.int64)
    out = linstep_interleave(start, num_steps, 1)
    assert out.numel() == 0


def test_linstep_interleave_int_input_int_step():
    """The tracks.py call site uses ``track_starts`` (int64) + step_size=1."""
    start = torch.tensor([100, 200, 300], dtype=torch.int64)
    num_steps = torch.tensor([2, 3, 1], dtype=torch.int64)
    out = linstep_interleave(start, num_steps, 1)
    assert out.dtype == torch.int64
    assert torch.equal(out, torch.tensor([100, 101, 200, 201, 202, 300]))


def test_linstep_interleave_int_dtype_preserved():
    out = linstep_interleave(
        start=torch.tensor([100, 200], dtype=torch.int64),
        num_steps=torch.tensor([2, 1]),
        step_size=1,
    )
    assert out.dtype == torch.int64
    assert torch.equal(out, torch.tensor([100, 101, 200]))


def test_linstep_interleave_preserves_dtype_float32():
    start = torch.tensor([0.0, 1.0], dtype=torch.float32)
    num_steps = torch.tensor([2, 2], dtype=torch.int64)
    out = linstep_interleave(start, num_steps, 0.5)
    assert out.dtype == torch.float32


def test_linstep_interleave_preserves_device():
    start = torch.tensor([0.0, 1.0])
    num_steps = torch.tensor([2, 2], dtype=torch.int64)
    out = linstep_interleave(start, num_steps, 1)
    assert out.device == start.device


# ---------------------------------------------------------------------------
# packed_searchsorted_indexed_vals
# ---------------------------------------------------------------------------


def test_packed_searchsorted_simple_two_packs_int():
    """bins concatenates two sorted packs:
       pack 0 = [10, 20, 30] (start=0, count=3)
       pack 1 = [100, 200, 300, 400] (start=3, count=4)
       queries: vals=[15, 25, 250, 350], indices=[0, 0, 1, 1]
       expected lower-bounds: [1, 2, 5, 6] (global indices into bins).
    """
    bins = torch.tensor([10, 20, 30, 100, 200, 300, 400], dtype=torch.int64)
    pack_infos = torch.tensor([[0, 3], [3, 4]], dtype=torch.int32)
    vals = torch.tensor([15, 25, 250, 350], dtype=torch.int64)
    vals_indices = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    out = packed_searchsorted_indexed_vals(bins, pack_infos, vals, vals_indices)
    assert out.dtype == torch.int32
    assert torch.equal(out, torch.tensor([1, 2, 5, 6], dtype=torch.int32))


def test_packed_searchsorted_query_below_pack_min_returns_pack_start():
    bins = torch.tensor([10, 20, 30, 100, 200], dtype=torch.int64)
    pack_infos = torch.tensor([[0, 3], [3, 2]], dtype=torch.int32)
    vals = torch.tensor([5, 50], dtype=torch.int64)
    vals_indices = torch.tensor([0, 1], dtype=torch.int32)
    out = packed_searchsorted_indexed_vals(bins, pack_infos, vals, vals_indices)
    assert torch.equal(out, torch.tensor([0, 3], dtype=torch.int32))


def test_packed_searchsorted_query_above_pack_max_returns_pack_end():
    """Lower-bound: when val > bins[start+length-1], result is start+length."""
    bins = torch.tensor([10, 20, 30, 100, 200], dtype=torch.int64)
    pack_infos = torch.tensor([[0, 3], [3, 2]], dtype=torch.int32)
    vals = torch.tensor([1000, 1000], dtype=torch.int64)
    vals_indices = torch.tensor([0, 1], dtype=torch.int32)
    out = packed_searchsorted_indexed_vals(bins, pack_infos, vals, vals_indices)
    assert torch.equal(out, torch.tensor([3, 5], dtype=torch.int32))


def test_packed_searchsorted_exact_match_returns_left_index():
    """Default lower-bound semantics: bins[i] == val gives index i."""
    bins = torch.tensor([10, 20, 30, 100, 200], dtype=torch.int64)
    pack_infos = torch.tensor([[0, 3], [3, 2]], dtype=torch.int32)
    vals = torch.tensor([20, 200], dtype=torch.int64)
    vals_indices = torch.tensor([0, 1], dtype=torch.int32)
    out = packed_searchsorted_indexed_vals(bins, pack_infos, vals, vals_indices)
    assert torch.equal(out, torch.tensor([1, 4], dtype=torch.int32))


def test_packed_searchsorted_empty_vals():
    bins = torch.tensor([10, 20], dtype=torch.int64)
    pack_infos = torch.tensor([[0, 2]], dtype=torch.int32)
    vals = torch.empty(0, dtype=torch.int64)
    vals_indices = torch.empty(0, dtype=torch.int32)
    out = packed_searchsorted_indexed_vals(bins, pack_infos, vals, vals_indices)
    assert out.numel() == 0
    assert out.dtype == torch.int32


def test_packed_searchsorted_float_bins():
    bins = torch.tensor([1.0, 2.5, 4.0, 10.0, 20.0], dtype=torch.float32)
    pack_infos = torch.tensor([[0, 3], [3, 2]], dtype=torch.int32)
    vals = torch.tensor([2.0, 15.0], dtype=torch.float32)
    vals_indices = torch.tensor([0, 1], dtype=torch.int32)
    out = packed_searchsorted_indexed_vals(bins, pack_infos, vals, vals_indices)
    assert torch.equal(out, torch.tensor([1, 4], dtype=torch.int32))


def test_packed_searchsorted_single_pack():
    bins = torch.tensor([1, 3, 5, 7, 9], dtype=torch.int64)
    pack_infos = torch.tensor([[0, 5]], dtype=torch.int32)
    vals = torch.tensor([2, 6, 10], dtype=torch.int64)
    vals_indices = torch.tensor([0, 0, 0], dtype=torch.int32)
    out = packed_searchsorted_indexed_vals(bins, pack_infos, vals, vals_indices)
    assert torch.equal(out, torch.tensor([1, 3, 5], dtype=torch.int32))


def test_packed_searchsorted_use_case_from_interpolate_tracks_poses():
    """End-to-end shape check matching the interpolate_tracks_poses use site.

    Three tracks, each with 4 keyframes; flatten gives 12 timestamps.
    Query 5 random points across the 3 tracks; verify the per-pack
    lower-bound semantics.
    """
    bins = torch.tensor(
        [100, 200, 300, 400, 1000, 1100, 1200, 1300, 5000, 5100, 5200, 5300],
        dtype=torch.int64,
    )
    pack_infos = torch.tensor([[0, 4], [4, 4], [8, 4]], dtype=torch.int32)
    queries = torch.tensor([150, 350, 1050, 1250, 5150], dtype=torch.int64)
    indices = torch.tensor([0, 0, 1, 1, 2], dtype=torch.int32)
    out = packed_searchsorted_indexed_vals(bins, pack_infos, queries, indices)
    # Expected (lower-bound):
    #   150 in [100,200,300,400] → 1 (global = 0+1).
    #   350 → 3 (global = 0+3).
    #   1050 in [1000,1100,1200,1300] → 1 (global = 4+1 = 5).
    #   1250 → 3 (global = 4+3 = 7).
    #   5150 in [5000,5100,5200,5300] → 2 (global = 8+2 = 10).
    assert torch.equal(out, torch.tensor([1, 3, 5, 7, 10], dtype=torch.int32))
