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

from dataclasses import fields, is_dataclass, replace
from typing import (
    Any,
    Dict,
    Generator,
    Hashable,
    List,
    Optional,
    Sequence,
    TypeVar,
)

import numpy.typing as npt
import torch


T = TypeVar("T")


def get_pack_info_from_n(n_per_pack: torch.Tensor) -> torch.Tensor:
    """Given an array of N pack element counts, returns the corresponding (N, 2) pack_info with per-pack start_idx / N_elements"""

    assert n_per_pack.dim() == 1 and not n_per_pack.is_floating_point(), (
        "get_pack_info_from_n(): required 1d integer as number-per-pack input"
    )

    return torch.stack(
        [n_per_pack.cumsum(0, dtype=n_per_pack.dtype) - n_per_pack, n_per_pack], 1
    )  # no exclusive cumsum in pytorch, emulate by substraction


def unpack_optional(maybe_value: Optional[T], default: Optional[T] = None, msg: Optional[str] = None) -> T:
    """Unpacks the value of an optional or returns a default if provided, otherwise raises a ValueError with custom message (if provided)."""
    if maybe_value is None:
        # Check if we can return a default value instead
        if default is not None:
            return default
        # Not possible to unpack an empty optional and no default is given -> raise ValueError
        raise ValueError(msg or "Can't unpack empty optional")

    # If the optional is not empty, return its value
    return maybe_value


def to_torch(
    data: npt.NDArray, device: str | torch.device, dtype: Optional[torch.dtype] = None, non_blocking: bool = False
) -> torch.Tensor:
    """Converts a numpy array to a torch tensor on target device with optional type-casting"""
    return torch.from_numpy(data).to(device=device, dtype=dtype, non_blocking=non_blocking)


def dataclass_keys(dataclass_: Any) -> Generator[str, Any, None]:
    assert is_dataclass(dataclass_), "Only applicable to dataclasses"
    for field in fields(dataclass_):
        yield field.name


def assert_same_type(seq: Sequence):
    """
    Asserts that all elements of a sequence are of the same type
    """

    if not seq:  # if the sequence is empty, all elements are trivially of the same type
        return True

    first_type = type(seq[0])
    assert all(isinstance(item, first_type) for item in seq), (
        f"Not all elements in the sequence are of the same type {first_type}"
    )


def collate_fn(batch: List[Any], target_device: torch.device | None) -> Any:
    """
    Returns a collated version of possibly nested tensors and dataclasses.
    """
    elem = batch[0]

    if elem is None:
        return None
    elif isinstance(elem, torch.Tensor):
        return torch.concat(batch, dim=0).to(target_device)
    elif hasattr(elem, "collate_fn"):
        return type(elem).collate_fn(batch, device=target_device)
    elif is_dataclass(type(elem)):
        return replace(
            elem, **{k: collate_fn([getattr(e, k) for e in batch], target_device) for k in dataclass_keys(elem)}
        )
    elif isinstance(elem, list):
        return [collate_fn([b[i] for b in batch], target_device) for i in range(len(elem))]
    elif isinstance(elem, dict):
        return {k: collate_fn([b[k] for b in batch], target_device) for k in elem.keys()}
    else:
        raise NotImplementedError(f"Collating of type {type(elem)} is not supported.")


def list_of_dicts_to_singleton_dict(list_of_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert a list of dictionaries (with shared keys and equal hashable values
    for each key) to a single dictionary mapping each key to its unique value.
    """
    if len(list_of_dicts) == 0:
        return {}

    all_keys = set.union(*map(set, list_of_dicts))
    common_keys = set.intersection(*map(set, list_of_dicts))
    assert all_keys == common_keys, (
        f"All dictionaries must have the same keys, but got different keys. all keys: {all_keys} and common keys: {common_keys}"
    )

    out: Dict[str, Any] = {}
    for ki in common_keys:
        values = set()
        for di in list_of_dicts:
            if not isinstance(di[ki], Hashable):
                raise TypeError(
                    f"unhashable type: '{type(di[ki])}'. Can not apply set.add(a), where a is of type {type(di[ki])})."
                )
            values.add(di[ki])
        assert len(values) == 1, f"List for key {ki} has more than one unique element: {values}"
        out[ki] = values.pop()
    return out
