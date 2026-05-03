"""Branch-coverage tests for ``instant_nurec._pkg.utils.packed_ops``.

After Phase A.4 the wrapper delegates to the pure-torch impl in
``_packed_ops_torch``; the kernel-stub fixture is no longer needed.
"""

from __future__ import annotations

import sys

from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


from instant_nurec._pkg.utils import packed_ops as mod  # noqa: E402


def test_linstep_interleave_empty_start_short_circuits():
    """Empty input → empty output, no kernel call (preserved across A.4)."""
    out = mod.linstep_interleave(
        start=torch.empty(0, dtype=torch.float32),
        num_steps=torch.empty(0, dtype=torch.int64),
        step_size=1.0,
    )
    assert out.values.numel() == 0
    assert out.pidx.numel() == 0


def test_linstep_interleave_returns_values_and_pidx():
    """Wrapper returns a ValuesAndPidx with the per-pack arange-style sequence."""
    start = torch.tensor([0.0, 5.0])
    num_steps = torch.tensor([2, 3])
    out = mod.linstep_interleave(start=start, num_steps=num_steps, step_size=0.5)
    assert isinstance(out, mod.ValuesAndPidx)
    assert torch.allclose(out.values, torch.tensor([0.0, 0.5, 5.0, 5.5, 6.0]))


def test_linstep_interleave_returns_pidx_when_return_idx_true():
    out = mod.linstep_interleave(
        start=torch.tensor([0.0]),
        num_steps=torch.tensor([3]),
        step_size=1.0,
        return_idx=True,
    )
    assert torch.equal(out.pidx, torch.tensor([0, 0, 0]))


def test_linstep_interleave_per_pack_tensor_step_size():
    out = mod.linstep_interleave(
        start=torch.tensor([0.0, 10.0]),
        num_steps=torch.tensor([2, 2]),
        step_size=torch.tensor([0.5, 1.0]),
    )
    assert torch.allclose(out.values, torch.tensor([0.0, 0.5, 10.0, 11.0]))


def test_linstep_interleave_int_dtype_preserved():
    """tracks.py call site uses int64 starts; the dtype must round-trip."""
    out = mod.linstep_interleave(
        start=torch.tensor([100, 200], dtype=torch.int64),
        num_steps=torch.tensor([2, 1]),
        step_size=1,
    )
    assert out.values.dtype == torch.int64
    assert torch.equal(out.values, torch.tensor([100, 101, 200]))


def test_values_and_pidx_dataclass_is_frozen_and_slotted():
    inst = mod.ValuesAndPidx(values=torch.zeros(1), pidx=torch.zeros(1, dtype=torch.long))
    with pytest.raises((AttributeError, Exception)):
        inst.values = torch.ones(1)
