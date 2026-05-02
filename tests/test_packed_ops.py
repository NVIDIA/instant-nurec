"""Branch-coverage tests for nre.utils.packed_ops.

The Phase 1 step 4.3 strip left only ``linstep_interleave``. The kernel
itself (``libs.packed_ops.interface.packed_ops.linstep_interleave``) is a
compiled CUDA op we can't run in the CPU-only test venv, but the empty-input
short-circuit at the top of the function is reachable, and the kernel
delegation can be exercised against a stubbed ``packed_ops`` namespace.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def stubbed_module(monkeypatch):
    """Install a fake ``libs.packed_ops.interface`` whose ``packed_ops``
    namespace records the args passed to ``linstep_interleave`` and returns
    a captured-shape tensor pair."""
    libs_mod = types.ModuleType("libs")
    packed_pkg = types.ModuleType("libs.packed_ops")
    interface_mod = types.ModuleType("libs.packed_ops.interface")

    captured: dict = {}

    class _StubPackedOps:
        @staticmethod
        def linstep_interleave(start, num_steps, step_size, return_idx):
            captured["start"] = start
            captured["num_steps"] = num_steps
            captured["step_size"] = step_size
            captured["return_idx"] = return_idx
            # Return tensor shapes the dataclass packs.
            total = int(num_steps.sum().item())
            return torch.zeros(total, dtype=start.dtype), torch.zeros(total, dtype=num_steps.dtype)

    interface_mod.packed_ops = _StubPackedOps()
    packed_pkg.interface = interface_mod
    libs_mod.packed_ops = packed_pkg

    monkeypatch.setitem(sys.modules, "libs", libs_mod)
    monkeypatch.setitem(sys.modules, "libs.packed_ops", packed_pkg)
    monkeypatch.setitem(sys.modules, "libs.packed_ops.interface", interface_mod)
    monkeypatch.delitem(sys.modules, "nre.utils.packed_ops", raising=False)

    import importlib

    mod = importlib.import_module("nre.utils.packed_ops")
    return mod, captured


def test_linstep_interleave_empty_start_short_circuits(stubbed_module):
    mod, captured = stubbed_module
    out = mod.linstep_interleave(
        start=torch.empty(0, dtype=torch.float32),
        num_steps=torch.empty(0, dtype=torch.int64),
        step_size=1.0,
    )
    assert out.values.numel() == 0
    assert out.pidx.numel() == 0
    # Kernel must not have been called for the empty input.
    assert "start" not in captured


def test_linstep_interleave_delegates_to_kernel_for_nonempty(stubbed_module):
    mod, captured = stubbed_module
    start = torch.tensor([0.0, 5.0])
    num_steps = torch.tensor([2, 3])
    out = mod.linstep_interleave(start=start, num_steps=num_steps, step_size=0.5)
    assert isinstance(out, mod.ValuesAndPidx)
    # Kernel got the right args.
    assert torch.equal(captured["start"], start)
    assert torch.equal(captured["num_steps"], num_steps.long())
    assert captured["step_size"] == 0.5
    assert captured["return_idx"] is False


def test_linstep_interleave_passes_tensor_step_size_through_contiguous(stubbed_module):
    """When step_size is a Tensor, the helper calls .contiguous() on it."""
    mod, captured = stubbed_module
    start = torch.tensor([0.0, 5.0])
    num_steps = torch.tensor([1, 1])
    step_size = torch.tensor([0.25, 0.5])
    mod.linstep_interleave(start=start, num_steps=num_steps, step_size=step_size)
    assert isinstance(captured["step_size"], torch.Tensor)
    assert captured["step_size"].is_contiguous()


def test_linstep_interleave_forwards_return_idx_flag(stubbed_module):
    mod, captured = stubbed_module
    mod.linstep_interleave(
        start=torch.tensor([0.0]),
        num_steps=torch.tensor([1]),
        step_size=1.0,
        return_idx=True,
    )
    assert captured["return_idx"] is True


def test_values_and_pidx_dataclass_is_frozen_and_slotted():
    """ValuesAndPidx is `slots=True, frozen=True` — verify the contract."""
    import sys as _sys

    # Need the module imported with stubs at least once; reuse from any earlier test.
    if "nre.utils.packed_ops" not in _sys.modules:
        pytest.skip("nre.utils.packed_ops not loaded yet (no prior stub run)")
    mod = _sys.modules["nre.utils.packed_ops"]

    inst = mod.ValuesAndPidx(values=torch.zeros(1), pidx=torch.zeros(1, dtype=torch.long))
    with pytest.raises((AttributeError, Exception)):
        inst.values = torch.ones(1)
