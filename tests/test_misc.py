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

"""Branch-coverage tests for instant_nurec.utils.misc.

The helpers in misc.py are tiny and pure (modulo torch tensor I/O), so each
test exercises one input branch end to end.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from instant_nurec.utils.misc import (
    assert_same_type,
    collate_fn,
    dataclass_keys,
    get_pack_info_from_n,
    list_of_dicts_to_singleton_dict,
    to_torch,
    unpack_optional,
)


# ---------------------------------------------------------------------------
# get_pack_info_from_n
# ---------------------------------------------------------------------------


def test_get_pack_info_from_n_typical_case():
    n = torch.tensor([2, 3, 1], dtype=torch.int64)
    out = get_pack_info_from_n(n)
    assert torch.equal(out[:, 0], torch.tensor([0, 2, 5], dtype=torch.int64))
    assert torch.equal(out[:, 1], n)


def test_get_pack_info_from_n_empty_input():
    n = torch.tensor([], dtype=torch.int64)
    out = get_pack_info_from_n(n)
    assert out.shape == (0, 2)


def test_get_pack_info_from_n_rejects_floating_point():
    with pytest.raises(AssertionError):
        get_pack_info_from_n(torch.tensor([1.0, 2.0]))


def test_get_pack_info_from_n_rejects_2d_input():
    with pytest.raises(AssertionError):
        get_pack_info_from_n(torch.tensor([[1, 2], [3, 4]]))


# ---------------------------------------------------------------------------
# unpack_optional
# ---------------------------------------------------------------------------


def test_unpack_optional_returns_value_when_present():
    assert unpack_optional(7) == 7


def test_unpack_optional_returns_default_when_none():
    assert unpack_optional(None, default=42) == 42


def test_unpack_optional_raises_when_none_and_no_default():
    with pytest.raises(ValueError, match="Can't unpack"):
        unpack_optional(None)


def test_unpack_optional_includes_custom_message_in_error():
    with pytest.raises(ValueError, match="boom"):
        unpack_optional(None, msg="boom")


def test_unpack_optional_default_takes_precedence_over_msg():
    """When default is provided AND value is None, default wins; msg is unused."""
    assert unpack_optional(None, default="def", msg="ignored") == "def"


# ---------------------------------------------------------------------------
# to_torch
# ---------------------------------------------------------------------------


def test_to_torch_default_dtype_preserves_input_dtype():
    arr = np.array([1.0, 2.0], dtype=np.float32)
    out = to_torch(arr, device="cpu")
    assert out.dtype == torch.float32


def test_to_torch_with_explicit_dtype_casts():
    arr = np.array([1.0, 2.0], dtype=np.float32)
    out = to_torch(arr, device="cpu", dtype=torch.float64)
    assert out.dtype == torch.float64


def test_to_torch_returns_on_target_cpu_device():
    arr = np.array([1, 2, 3], dtype=np.int64)
    out = to_torch(arr, device="cpu")
    assert out.device.type == "cpu"


# ---------------------------------------------------------------------------
# dataclass_keys
# ---------------------------------------------------------------------------


def test_dataclass_keys_yields_field_names_in_order():
    @dataclass
    class Foo:
        a: int
        b: str
        c: float

    foo = Foo(1, "x", 3.14)
    assert list(dataclass_keys(foo)) == ["a", "b", "c"]


def test_dataclass_keys_rejects_non_dataclass():
    with pytest.raises(AssertionError):
        list(dataclass_keys({"a": 1}))


# ---------------------------------------------------------------------------
# assert_same_type
# ---------------------------------------------------------------------------


def test_assert_same_type_empty_sequence_returns_true():
    assert assert_same_type([]) is True


def test_assert_same_type_homogeneous_sequence_passes():
    # No assertion error → None implicit return
    assert_same_type([1, 2, 3])


def test_assert_same_type_mixed_sequence_raises():
    with pytest.raises(AssertionError):
        assert_same_type([1, "two", 3])


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------


def test_collate_fn_collates_tensors_via_concat():
    out = collate_fn([torch.tensor([1.0]), torch.tensor([2.0])], torch.device("cpu"))
    assert torch.equal(out, torch.tensor([1.0, 2.0]))


def test_collate_fn_returns_none_when_first_elem_none():
    assert collate_fn([None, None], torch.device("cpu")) is None


def test_collate_fn_uses_class_collate_fn_when_present():
    class Custom:
        @classmethod
        def collate_fn(cls, batch, device):  # noqa: ARG003
            return f"collated({len(batch)})"

    out = collate_fn([Custom(), Custom(), Custom()], torch.device("cpu"))
    assert out == "collated(3)"


def test_collate_fn_recurses_into_list():
    batch = [[torch.tensor([1.0]), torch.tensor([2.0])], [torch.tensor([3.0]), torch.tensor([4.0])]]
    out = collate_fn(batch, torch.device("cpu"))
    assert torch.equal(out[0], torch.tensor([1.0, 3.0]))
    assert torch.equal(out[1], torch.tensor([2.0, 4.0]))


def test_collate_fn_recurses_into_dict():
    batch = [{"a": torch.tensor([1.0])}, {"a": torch.tensor([2.0])}]
    out = collate_fn(batch, torch.device("cpu"))
    assert torch.equal(out["a"], torch.tensor([1.0, 2.0]))


def test_collate_fn_recurses_into_dataclass():
    @dataclass
    class Pair:
        x: torch.Tensor
        y: torch.Tensor

    batch = [Pair(torch.tensor([1.0]), torch.tensor([10.0])), Pair(torch.tensor([2.0]), torch.tensor([20.0]))]
    out = collate_fn(batch, torch.device("cpu"))
    assert torch.equal(out.x, torch.tensor([1.0, 2.0]))
    assert torch.equal(out.y, torch.tensor([10.0, 20.0]))


def test_collate_fn_unknown_type_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        collate_fn([object(), object()], torch.device("cpu"))


# ---------------------------------------------------------------------------
# list_of_dicts_to_singleton_dict
# ---------------------------------------------------------------------------


def test_list_of_dicts_to_singleton_dict_empty_input_returns_empty():
    assert list_of_dicts_to_singleton_dict([]) == {}


def test_list_of_dicts_to_singleton_dict_collapses_equal_values():
    out = list_of_dicts_to_singleton_dict([{"a": 1, "b": "x"}, {"a": 1, "b": "x"}])
    assert out == {"a": 1, "b": "x"}


def test_list_of_dicts_to_singleton_dict_rejects_disagreement():
    with pytest.raises(AssertionError, match="more than one unique element"):
        list_of_dicts_to_singleton_dict([{"a": 1}, {"a": 2}])


def test_list_of_dicts_to_singleton_dict_rejects_unhashable_value():
    with pytest.raises(TypeError, match="unhashable"):
        list_of_dicts_to_singleton_dict([{"a": [1]}, {"a": [1]}])


def test_list_of_dicts_to_singleton_dict_rejects_key_mismatch():
    with pytest.raises(AssertionError, match="same keys"):
        list_of_dicts_to_singleton_dict([{"a": 1}, {"b": 1}])
