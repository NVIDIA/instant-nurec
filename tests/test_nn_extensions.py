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

"""Branch-coverage tests for instant_nurec.utils.nn_extensions.

The module is pure torch.nn — no heavy / compiled deps.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from instant_nurec.utils.nn_extensions import TypedModuleList, module_call_type


# ---------------------------------------------------------------------------
# TypedModuleList
# ---------------------------------------------------------------------------


def test_typed_module_list_empty_init():
    """No args → empty container."""
    m = TypedModuleList[nn.Linear]()
    assert len(m) == 0


def test_typed_module_list_init_with_iterable():
    m = TypedModuleList[nn.Linear]([nn.Linear(3, 4), nn.Linear(4, 5)])
    assert len(m) == 2


def test_typed_module_list_getitem():
    a, b = nn.Linear(3, 4), nn.Linear(4, 5)
    m = TypedModuleList[nn.Linear]([a, b])
    assert m[0] is a
    assert m[1] is b


def test_typed_module_list_setitem():
    m = TypedModuleList[nn.Linear]([nn.Linear(3, 4)])
    new = nn.Linear(7, 8)
    m[0] = new
    assert m[0] is new


def test_typed_module_list_append():
    m = TypedModuleList[nn.Linear]()
    a = nn.Linear(3, 4)
    m.append(a)
    assert len(m) == 1
    assert m[0] is a


def test_typed_module_list_extend():
    m = TypedModuleList[nn.Linear]([nn.Linear(3, 4)])
    m.extend([nn.Linear(4, 5), nn.Linear(5, 6)])
    assert len(m) == 3


def test_typed_module_list_insert():
    a, c = nn.Linear(3, 4), nn.Linear(5, 6)
    m = TypedModuleList[nn.Linear]([a, c])
    b = nn.Linear(4, 5)
    m.insert(1, b)
    assert len(m) == 3
    assert m[0] is a and m[1] is b and m[2] is c


def test_typed_module_list_pop_default_index():
    a, b = nn.Linear(3, 4), nn.Linear(4, 5)
    m = TypedModuleList[nn.Linear]([a, b])
    popped = m.pop()  # default = -1 (last)
    assert popped is b
    assert len(m) == 1


def test_typed_module_list_pop_explicit_index():
    a, b = nn.Linear(3, 4), nn.Linear(4, 5)
    m = TypedModuleList[nn.Linear]([a, b])
    popped = m.pop(0)
    assert popped is a
    assert len(m) == 1
    assert m[0] is b


def test_typed_module_list_iter():
    items = [nn.Linear(3, 4), nn.Linear(4, 5), nn.Linear(5, 6)]
    m = TypedModuleList[nn.Linear](items)
    seen = list(iter(m))
    assert seen == items  # same identity, order preserved


# ---------------------------------------------------------------------------
# module_call_type
# ---------------------------------------------------------------------------


class _Doubler(nn.Module):
    """Minimal nn.Module that just doubles its input — used to verify
    module_call_type wires through to nn.Module.__call__ (and therefore
    forward + hooks)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2

    __call__ = module_call_type(forward)


def test_module_call_type_dispatches_to_forward():
    m = _Doubler()
    out = m(torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(out, torch.tensor([2.0, 4.0, 6.0]))


def test_module_call_type_fires_forward_hooks():
    """The wrapper goes through nn.Module.__call__, so registered hooks must fire."""
    m = _Doubler()
    seen = {}

    def hook(_module, _inputs, output):
        seen["out"] = output

    m.register_forward_hook(hook)
    m(torch.tensor([5.0]))
    assert "out" in seen
    assert torch.equal(seen["out"], torch.tensor([10.0]))


def test_module_call_type_rejects_non_module_self():
    """If applied to a class that isn't an nn.Module, calling raises TypeError."""

    class _NotAModule:
        def forward(self, x):
            return x

        __call__ = module_call_type(forward)

    with pytest.raises(TypeError, match="does not derive from nn.Module"):
        _NotAModule()(123)
