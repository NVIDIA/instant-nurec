# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import functools
import random
import traceback

from abc import ABC
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generator,
    Hashable,
    Iterable,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from torch.utils import data as torchdata


T = TypeVar("T")
U = TypeVar("U")
KT = TypeVar("KT", bound=Hashable)
VT = TypeVar("VT")


def rank_zero_only(fn):
    """Predict-only standalone is single-process; rank_zero_only is a no-op
    decorator. Self-invented: NRE used pytorch_lightning's rank_zero_only."""

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapped


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


def map_optional(maybe_value: Optional[T], func: Callable[[T], U]) -> Optional[U]:
    """Applies a function `func` to an optional value if it's set, otherwise returns None"""
    if maybe_value is None:
        # Can't apply function if there is no input value, return None
        return None

    # If the optional is not empty, apply the function to its value
    return func(maybe_value)


def strip_none_from_config(obj: Any) -> Any:
    """Recursively remove dict entries whose value is None. For JSON serialization."""
    if isinstance(obj, dict):
        return {k: strip_none_from_config(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [strip_none_from_config(v) for v in obj]
    return obj


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


def collate_fn(
    batch: List[Any],
    target_device: torch.device | None,
    name_hint: str | None = None,
    return_list_if_unknown: bool = False,
) -> Any:
    """
    Returns a collated version of possibly nested tensors and dataclasses.
    """
    elem = batch[0]

    if name_hint is not None:
        if name_hint in ["n_rm_samples", "n_vr_samples"]:
            return sum(batch)
        if name_hint == "pack_info":
            # Note pack_info is a packed auxiliary tensor containing starting indices and num samples for each ray [int] (n_rays, 2)
            num_samples = torch.concat(batch, dim=0).to(target_device)[:, 1]
            return get_pack_info_from_n(num_samples)
        if name_hint == "radiance_embedding_type":
            assert [b == elem for b in batch], "Collating different `radiance_embedding_type` is not supported."
            return batch[0]

    if elem is None:
        return None
    elif isinstance(elem, torch.Tensor):
        return torch.concat(batch, dim=0).to(target_device)
    elif hasattr(elem, "collate_fn"):
        return type(elem).collate_fn(batch, device=target_device)
    elif is_dataclass(type(elem)):
        return replace(
            elem, **{k: collate_fn([getattr(e, k) for e in batch], target_device, k) for k in dataclass_keys(elem)}
        )
    elif isinstance(elem, list):
        return [collate_fn([b[i] for b in batch], target_device) for i in range(len(elem))]
    elif isinstance(elem, dict):
        return {k: collate_fn([b[k] for b in batch], target_device) for k in elem.keys()}
    else:
        if not return_list_if_unknown:
            raise NotImplementedError(f"Collating of type {type(elem)} is not supported.")
        else:
            return batch



# https://github.com/nerfstudio-project/gsplat/blob/2323de5905d5e90e035f792fe65bad0fedd413e7/gsplat/distributed.py#L10
def stop_gradient(input: torch.Tensor) -> torch.Tensor:
    """
    Stop the gradient from flowing through the given tensor, but still keep the computation graph.
    """
    return input * 0 + input.detach()


def list_of_dicts_to_dict_of_lists(
    list_of_dicts: List[Dict[str, Any]], no_duplicates: bool = False, singleton: bool = False
) -> Dict[str, List[Any]]:
    """
    Convert a list of dictionaries to a dictionary of lists.

    If no_duplicates is True, the lists will only contain unique elements.
    If singleton is True, it fails if at least one list in the dict_of_lists has more than one unique element. Also, the value of the dictionary is the unique element and not a list of length one.

    If list_of_dicts is empty, returns an empty dictionary.

    Assumes all dictionaries have the same keys, throws assertion error otherwise.
    Assumes all values are hashable, if no_duplicates or singleton is True.
    """
    if len(list_of_dicts) == 0:
        return {}

    all_keys = set.union(*map(set, list_of_dicts))
    common_keys = set.intersection(*map(set, list_of_dicts))
    assert all_keys == common_keys, (
        f"All dictionaries must have the same keys, but got different keys. all keys: {all_keys} and common keys: {common_keys}"
    )

    if no_duplicates or singleton:
        # Iterate over all keys and collect the unique values into a set.
        dict_of_lists = {}
        for ki in common_keys:
            # Collect the unique values for the current key into a set.
            values = set()
            for di in list_of_dicts:
                if not isinstance(di[ki], Hashable):
                    raise TypeError(
                        f"unhashable type: '{type(di[ki])}'. Can not apply set.add(a), where a is of type {type(di[ki])})."
                    )
                values.add(di[ki])

            if singleton:
                assert len(values) == 1, f"List for key {ki} has more than one unique element: {values}"
                dict_of_lists[ki] = values.pop()
            else:
                dict_of_lists[ki] = list(values)

        return dict_of_lists
    else:
        return {k: [d[k] for d in list_of_dicts] for k in common_keys}
