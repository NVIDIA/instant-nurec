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

"""Branch-coverage tests for ``instant_nurec.utils.packed_ops``.

After Phase A.4 the wrapper delegates to the pure-torch impl in
``_packed_ops_torch``; the kernel-stub fixture is no longer needed.
The wrapper used to return a ``ValuesAndPidx`` dataclass with a
secondary ``.pidx`` field — both that field and the underlying
``return_idx`` parameter were dropped in the final dead-code pass
(no production caller ever read ``.pidx``).
"""

from __future__ import annotations

import sys

from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


from instant_nurec.utils import packed_ops as mod  # noqa: E402


def test_linstep_interleave_empty_start_short_circuits():
    """Empty input → empty output, no kernel call (preserved across A.4)."""
    out = mod.linstep_interleave(
        start=torch.empty(0, dtype=torch.float32),
        num_steps=torch.empty(0, dtype=torch.int64),
        step_size=1.0,
    )
    assert out.numel() == 0


def test_linstep_interleave_emits_per_pack_arange_sequence():
    """Wrapper returns the per-pack arange-style sequence flattened."""
    start = torch.tensor([0.0, 5.0])
    num_steps = torch.tensor([2, 3])
    out = mod.linstep_interleave(start=start, num_steps=num_steps, step_size=0.5)
    assert torch.allclose(out, torch.tensor([0.0, 0.5, 5.0, 5.5, 6.0]))


def test_linstep_interleave_per_pack_tensor_step_size():
    out = mod.linstep_interleave(
        start=torch.tensor([0.0, 10.0]),
        num_steps=torch.tensor([2, 2]),
        step_size=torch.tensor([0.5, 1.0]),
    )
    assert torch.allclose(out, torch.tensor([0.0, 0.5, 10.0, 11.0]))


def test_linstep_interleave_int_dtype_preserved():
    """tracks.py call site uses int64 starts; the dtype must round-trip."""
    out = mod.linstep_interleave(
        start=torch.tensor([100, 200], dtype=torch.int64),
        num_steps=torch.tensor([2, 1]),
        step_size=1,
    )
    assert out.dtype == torch.int64
    assert torch.equal(out, torch.tensor([100, 101, 200]))
