# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
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
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
)

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_only
from torch.utils import data as torchdata


T = TypeVar("T")
U = TypeVar("U")
KT = TypeVar("KT", bound=Hashable)
VT = TypeVar("VT")


def is_env_true(name: str, default: bool) -> bool:
    """Check if an environment variable is set to a truthy ("True", "true", "TrUE", "1", ...) value."""
    return os.environ.get(name, "1" if default else "0").lower() in ["1", "true"]


def _get_rank() -> int:
    """Helper function to get the rank of the current process

    Based https://github.com/Lightning-AI/pytorch-lightning/blob/2.4.0/src/lightning/fabric/utilities/rank_zero.py#L39-L48

    Behaviour:
        - If distributed training is initialized, returns the rank of the current process.
        - If distributed training is not initialized, it checks the following environment variables in order:
            - LOCAL_RANK
            - RANK
            - SLURM_PROCID
            - JSM_NAMESPACE_RANK
        - If none of the environment variables are set, it will return 0.
    Returns:
        The rank of the current process
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        # In the case of single-node multi-GPU training, LOCAL_RANK is set to the rank of the current GPU. However, if
        # the job is launched inside a SLURM task, `SLURM_PROCID` will also be set to 0. This creates a conflict. So we
        # compare the two values. Note that if SLURM_PROCID is set correctly, it should not be smaller than LOCAL_RANK,
        # so the larger value is the correct rank ID.
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        # Assume multi-node distributed training
        rank_keys = ("RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
        for key in rank_keys:
            rank = os.environ.get(key)
            if rank is not None:
                return max(int(rank), local_rank)

        # Assume single-node multi-GPU training
        node_rank = os.environ.get("NODE_RANK")
        assert node_rank is None or node_rank == "0"

        return local_rank


def _get_local_rank() -> int:
    return max(int(os.environ.get("LOCAL_RANK", "0")), int(os.environ.get("SLURM_LOCALID", "0")))


# Reset global rank correctly
rank_zero_only.rank = _get_rank()


def set_default_device(local_rank: int | None = None) -> None:
    """Set the default CUDA device for the current process in distributed training.

    In distributed training, each process typically manages one GPU. The local_rank
    identifies which GPU this process should use within the current node/machine.

    Args:
        local_rank: The local GPU index within the current node to use as default.
            If None, automatically determines the local rank from environment variables
            (LOCAL_RANK, RANK, SLURM_PROCID, JSM_NAMESPACE_RANK, etc.) or the current
            distributed context if available.

    Returns:
        None
    """
    torch.cuda.set_device(_get_local_rank() if local_rank is None else local_rank)


def assert_default_device_on_local_rank(local_rank: int | None = None) -> None:
    """Verify that PyTorch's default CUDA device matches the expected local rank.

    This is a safety check to ensure the process is using the correct GPU in distributed
    training. Helps catch configuration errors where the wrong GPU is being used.

    In distributed setups, it's critical that each process uses its assigned GPU to
    avoid memory conflicts and ensure proper data placement.

    Args:
        local_rank: The expected local GPU index within the current node.
            If None, automatically determines the expected local rank from environment
            variables or distributed context if available.

    Returns:
        None

    Raises:
        AssertionError: If the current default CUDA device doesn't match the expected local_rank.
    """
    local_rank = _get_local_rank() if local_rank is None else local_rank
    assert local_rank == torch.cuda.current_device(), (
        f"Default CUDA device {torch.cuda.current_device()} does not match local rank {local_rank}"
    )


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


def dataclass_items(dataclass_: Any) -> Generator[tuple[str, Any], Any, None]:
    """
    Iterate over dataclass keys and values
    https://stackoverflow.com/a/77486666/24150771
    """
    assert is_dataclass(dataclass_), "Only applicable to dataclasses"
    for field in fields(dataclass_):
        yield field.name, getattr(dataclass_, field.name)


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


def flatten_list(x: list) -> list:
    """
    Deep flattens a list of lists in place. For example:
    Input: [1, 2, 3,[1, 2,[[3, 4,[5]], 7, 0, 1, 10], 100,[101,[101,[[101]], 2]],0]]
    Output: [1, 2, 3, 1, 2, 3, 4, 5, 7, 0, 1, 10, 100, 101, 101, 101, 2, 0]
    """
    if isinstance(x, list):
        return [a for i in x for a in flatten_list(i)]
    else:
        return [x]


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



def get_union_types(union_type: Any) -> Tuple[Type, ...]:
    """Get the union types of a type or union.

    For example, if union_type is `Union[int, float]`, this function will return `(int, float)`.
    """
    args = get_args(union_type)
    if args == ():  # if union_type is not a union, return the type itself
        return (union_type,)
    return args


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
