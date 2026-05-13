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

"""Branch-coverage tests for the parity-gate helpers inside
``internal/scripts/export_kelvin_jit.py``.

The script's ``main`` body exercises GPU + a real ncorev4 sequence; that
end-to-end run is the production gate. The parity helpers are pure-CPU
and worth their own focused branch tests.
"""

from __future__ import annotations

import importlib.util
import sys

from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def export_mod():
    """Direct import of the export script as a module so we can call its
    private helpers and verify them in coverage."""
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "internal"))
    spec = importlib.util.spec_from_file_location(
        "export_kelvin_jit_module_under_test",
        str(REPO_ROOT / "internal" / "scripts" / "export_kelvin_jit.py"),
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------- _assert_close ----------


def test_assert_close_bitwise_pass(export_mod):
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    export_mod._assert_close(a, a.clone(), "x")


def test_assert_close_bitwise_fail(export_mod):
    a = torch.zeros(2, 3)
    b = a.clone()
    b[0, 0] = 1.0
    with pytest.raises(AssertionError, match="bitwise mismatch"):
        export_mod._assert_close(a, b, "x")


def test_assert_close_shape_mismatch_raises(export_mod):
    with pytest.raises(AssertionError, match="shape mismatch"):
        export_mod._assert_close(torch.zeros(3), torch.zeros(4), "x")


def test_assert_close_dtype_mismatch_raises(export_mod):
    with pytest.raises(AssertionError, match="dtype mismatch"):
        export_mod._assert_close(torch.zeros(3, dtype=torch.float32), torch.zeros(3, dtype=torch.float64), "x")


def test_assert_close_with_tolerance_passes_within_atol(export_mod):
    a = torch.zeros(3)
    b = torch.full((3,), 1e-7)
    export_mod._assert_close(a, b, "x", atol=1e-5, rtol=0.0)


def test_assert_close_with_tolerance_fails_above_atol(export_mod):
    a = torch.zeros(3)
    b = torch.full((3,), 1e-3)
    with pytest.raises(AssertionError):
        export_mod._assert_close(a, b, "x", atol=1e-5, rtol=0.0)


# ---------- _assert_count_close ----------


def test_assert_count_close_exact_shape_logs_max_diff(export_mod):
    a = torch.zeros(10, 3)
    b = a.clone()
    export_mod._assert_count_close(a, b, "positions")


def test_assert_count_close_within_delta_tolerance(export_mod):
    a = torch.zeros(100, 3)
    b = torch.zeros(110, 3)  # delta=10, within default 50
    export_mod._assert_count_close(a, b, "positions")


def test_assert_count_close_exceeds_delta_raises(export_mod):
    a = torch.zeros(100, 3)
    b = torch.zeros(200, 3)  # delta=100, > 50
    with pytest.raises(AssertionError, match="exceeds tolerance"):
        export_mod._assert_count_close(a, b, "positions")


def test_assert_count_close_custom_delta(export_mod):
    a = torch.zeros(100, 3)
    b = torch.zeros(110, 3)
    with pytest.raises(AssertionError, match="exceeds tolerance"):
        export_mod._assert_count_close(a, b, "positions", vertex_count_delta=5)


def test_assert_count_close_dtype_mismatch_raises(export_mod):
    a = torch.zeros(10, 3, dtype=torch.float32)
    b = torch.zeros(10, 3, dtype=torch.float64)
    with pytest.raises(AssertionError, match="dtype mismatch"):
        export_mod._assert_count_close(a, b, "positions")


def test_assert_count_close_ndim_mismatch_raises(export_mod):
    a = torch.zeros(10, 3)
    b = torch.zeros(10, 3, 1)
    with pytest.raises(AssertionError, match="ndim mismatch"):
        export_mod._assert_count_close(a, b, "positions")


def test_assert_count_close_trailing_shape_mismatch_raises(export_mod):
    a = torch.zeros(10, 3)
    b = torch.zeros(10, 4)  # different feature dim
    with pytest.raises(AssertionError, match="trailing-shape mismatch"):
        export_mod._assert_count_close(a, b, "positions")
