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


def to_numpy(data: torch.Tensor) -> npt.NDArray:
    """Convert a PyTorch tensor to a NumPy array.

    Handles the common pattern of detaching from the computation graph,
    moving to CPU, and converting to NumPy. Safe to use with tensors
    that require gradients or are on GPU.

    Args:
        data: The PyTorch tensor to convert.

    Returns:
        NumPy array with the same data as the input tensor.
    """
    return data.detach().cpu().numpy()


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



def power_ladder_max_output(p: float) -> float:
    """The limit of power_ladder(x, p) as x goes to infinity."""
    if p == float("-inf"):
        return 1
    elif p >= 0:
        return float("inf")
    else:
        return (p - 1) / p


def power_ladder(
    x: torch.Tensor, p: float, pre_mult: float | None = None, post_mult: float | None = None, eps: float = 1.0e-7
) -> torch.Tensor:
    """Tukey's power ladder, with a +1 on x, some scaling, and special cases."""

    # Compute sign(x) * |p - 1|/p * ((|x|/|p-1| + 1)^p - 1)
    if pre_mult is not None:
        x = x * pre_mult
    xp = torch.abs(x)
    xs = xp / max(abs(p - 1), eps)
    p_safe = eps if abs(p) < eps else p
    sign_x = torch.where(x < 0, torch.full_like(x, -1), torch.full_like(x, 1))
    mult: torch.Tensor
    match p:
        case 1:
            mult = xp
        case 0:
            mult = torch.log1p(xp)
        case float("-inf"):
            mult = -torch.expm1(-xp)
        case float("inf"):
            mult = torch.expm1(xp)
        case _:
            mult = abs(p_safe - 1) / p_safe * ((xs + 1) ** p_safe - 1)
    y = sign_x * mult
    if post_mult is not None:
        y = y * post_mult
    return y


def power_ladder_inv(
    y: torch.Tensor, p: float, pre_mult: float | None = None, post_mult: float | None = None, eps: float = 1.0e-7
) -> torch.Tensor:
    """The inverse of `power_ladder()`."""
    if post_mult is not None:
        y /= post_mult
    p_safe = eps if abs(p) < eps else p
    y_max = max(-eps, power_ladder_max_output(p))
    yp = torch.abs(y).clip(-y_max, y_max)
    sign_y = torch.where(y < 0, torch.full_like(y, -1), torch.full_like(y, 1))
    mult: torch.Tensor
    match p:
        case 1:
            mult = yp
        case 0:
            mult = torch.expm1(yp)
        case float("-inf"):
            mult = -torch.log1p(-yp)
        case float("inf"):
            mult = torch.log1p(yp)
        case _:
            mult = abs(p_safe - 1) * ((p_safe / max(abs(p_safe - 1), eps) * yp + 1) ** (1 / p_safe) - 1)

    x = sign_y * mult
    if pre_mult is not None:
        x /= pre_mult
    return x


def logistic_density(x: torch.Tensor, inv_s: torch.Tensor | float, normalized: bool = False) -> torch.Tensor:
    """Logistic density function
    Source: https://en.wikipedia.org/wiki/Logistic_distribution

    Args:
        x (torch.Tensor): Input
        inv_s (Union[torch.Tensor, Number]): The reciprocal of the distribution scaling factor.
        normalized (bool): Set true to normalize the logistic function (with constant peak value == 1.0, un-related to inv_s)

    Returns:
        torch.Tensor: Output
    """
    cosh = torch.cosh((inv_s * x / 2.0).clamp_(-20, 20))
    return ((1.0 / cosh) ** 2) if normalized else (0.25 * inv_s / (cosh**2))


def precision_to_dtype(precision: str | int) -> torch.dtype:
    return {
        "16": torch.float16,
        "16-no-cast": torch.float16,
        "32": torch.float32,
    }[str(precision)]


AbstractClass = TypeVar("AbstractClass", bound=ABC)


def create_dummy_class(abstract_class: Type[AbstractClass]) -> Type[AbstractClass]:
    """Dynamically generates a dummy subclass implementing all abstract members."""
    abstract_members = abstract_class.__abstractmethods__
    dummy_class_name = f"Dummy{abstract_class.__name__}"

    def create_dummy_method(name):
        def method(self, *args, **kwargs):
            return None

        method.__name__ = name
        method.__qualname__ = f"{dummy_class_name}.{name}"
        return method

    methods = {
        name: (
            property(lambda self: None)
            if isinstance(getattr(abstract_class, name, None), property)
            else create_dummy_method(name)
        )
        for name in abstract_members
    }

    return type(dummy_class_name, (abstract_class,), methods)


class _SingletonsRegistry:
    """Protocol defining the singleton associated to an interface"""

    _instances: ClassVar[Dict[Type[ABC], ABC]] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_instance(interface_cls: Type[ABC]) -> ABC:
        if interface_cls not in _SingletonsRegistry._instances:
            _SingletonsRegistry._instances[interface_cls] = create_dummy_class(interface_cls)()
        return _SingletonsRegistry._instances[interface_cls]

    @staticmethod
    def register(interface_cls: Type[ABC], impl_cls: Type[ABC]) -> None:
        _SingletonsRegistry._instances[interface_cls] = impl_cls()


def distributed_all_gather_nested(data: Any, only_on_rank_zero: bool = True) -> Any:
    """
    Gather and merge data from all distributed processes with support for nested structures.

    This funciton can can handle:
    - Nested dictionaries
    - Lists, tuples, sets
    - Dataclasses
    - Pydantic models
    - Any Python type (collected into lists for non-collections)

    Collections (lists, tuples, sets) are extended/merged across processes.
    Non-collection types (scalars, custom objects) are collected into lists.

    Args:
        data: Any data structure to be gathered from all processes
        only_on_rank_zero: If True, return combined data only on rank 0,
                           otherwise return on all ranks

    Returns:
        Combined data from all processes.

    Examples:
        # Example 1: Nested dictionaries with lists
        # Process 0: {"metrics": {"camera": {"psnr": [30.1], "loss": [0.1]}}}
        # Process 1: {"metrics": {"camera": {"psnr": [29.8], "loss": [0.2]}}}
        # Result: {"metrics": {"camera": {"psnr": [30.1, 29.8], "loss": [0.1, 0.2]}}}

        # Example 2: Mixed types
        # Process 0: {"count": 5, "name": "gpu0", "data": [1, 2]}
        # Process 1: {"count": 3, "name": "gpu1", "data": [3, 4]}
        # Result: {"count": [5, 3], "name": ["gpu0", "gpu1"], "data": [1, 2, 3, 4]}
    """

    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        # Not in distributed mode or single GPU, return as-is
        return data

    # Gather data from all processes
    all_data = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(all_data, data)

    # Only combine on rank 0 if requested
    if only_on_rank_zero and torch.distributed.get_rank() != 0:
        return data

    def merge_structures(structures: List[Any]) -> Any:
        """Recursively merge nested structures."""
        if not structures:
            return None

        first = structures[0]

        # Handle None explicitly
        if first is None:
            return None

        # Check if it's a Pydantic model
        # Pydantic models have model_dump() method
        if hasattr(first, "model_dump") and callable(getattr(first, "model_dump")):
            # Convert all Pydantic models to dicts
            dicts = []
            for struct in structures:
                dicts.append(struct.model_dump())
            # Merge the dictionaries
            merged_dict = merge_structures(dicts)
            # Reconstruct the Pydantic model from the merged dict
            return type(first)(**merged_dict)
        # Check for container types first (these need recursive merging)
        elif isinstance(first, dict):
            # Merge dictionaries recursively
            merged = {}
            all_keys = set()
            for struct in structures:
                all_keys.update(struct.keys())
            for key in all_keys:
                values_for_key = []
                for struct in structures:
                    if key in struct:
                        values_for_key.append(struct[key])
                if values_for_key:
                    merged[key] = merge_structures(values_for_key)
            return merged
        elif is_dataclass(type(first)):
            # Handle dataclasses by converting to dict and back
            dicts = []
            for struct in structures:
                dicts.append({k: getattr(struct, k) for k in dataclass_keys(struct)})
            merged_dict = merge_structures(dicts)
            return replace(first, **merged_dict)
        elif isinstance(first, list):
            list_result: list = []
            for v in structures:
                list_result.extend(v)
            return list_result
        elif isinstance(first, tuple):
            # Extend tuples
            tuple_result: list = []
            for v in structures:
                tuple_result.extend(v)
            return tuple(tuple_result)
        elif isinstance(first, set):
            # Union sets
            set_result: set = set()
            for v in structures:
                set_result.update(v)
            return set_result
        elif hasattr(first, "__iter__") and not isinstance(first, (str, bytes)):
            # For other iterables (but not strings/bytes), try to extend
            try:
                iterable_result: list = []
                for v in structures:
                    iterable_result.extend(v)
                # Try to convert back to original type
                return type(first)(iterable_result)
            except:
                # If that fails, just return as list
                return structures
        else:
            # Everything else is treated as a leaf value - collect into list
            return structures

    return merge_structures(all_data)


def singleton_get_instance(
    interface_cls: Type[ABC],
) -> ABC:
    """Associates a singleton to an interface."""

    return _SingletonsRegistry.get_instance(interface_cls)


def singleton_register(
    interface_cls: Type[ABC],
) -> Callable[[Type[ABC]], Type[ABC]]:
    """Register an implementation class to a interface singleton."""

    def decorator(impl_cls: Type[ABC]) -> Type[ABC]:
        _SingletonsRegistry.register(interface_cls, impl_cls)
        return impl_cls

    return decorator


def crop_mask_border(im: torch.Tensor, camera_mask_border: list[int]) -> torch.Tensor:
    """
    Crop the border of an image.

    Args:
        im (torch.Tensor): The image to crop.
        camera_mask_border (list[int]): The border to crop in top, right, bottom, left order.

    Returns:
        torch.Tensor: The cropped image.
    """
    copied_im = im.clone()
    if camera_mask_border[0] != 0:
        copied_im = copied_im[:, :, camera_mask_border[0] :]
    if camera_mask_border[1] != 0:
        copied_im = copied_im[:, :, :, camera_mask_border[1] :]
    if camera_mask_border[2] != 0:
        copied_im = copied_im[:, :, : -camera_mask_border[2]]
    if camera_mask_border[3] != 0:
        copied_im = copied_im[:, :, :, : -camera_mask_border[3]]
    return copied_im


def get_union_types(union_type: Any) -> Tuple[Type, ...]:
    """Get the union types of a type or union.

    For example, if union_type is `Union[int, float]`, this function will return `(int, float)`.
    """
    args = get_args(union_type)
    if args == ():  # if union_type is not a union, return the type itself
        return (union_type,)
    return args


# https://github.com/nerfstudio-project/gsplat/blob/2323de5905d5e90e035f792fe65bad0fedd413e7/gsplat/distributed.py#L10
def all_gather_int32(world_size: int, value: int, device: Optional[torch.device] = None) -> List[int]:
    """Gather an 32-bit integer from all ranks.

    .. note::
        This implementation is faster than using `torch.distributed.all_gather_object`.

    .. note::
        This function is not differentiable to the input tensor.

    Args:
        world_size: The total number of ranks.
        value: The integer to gather. Could be a scalar or a tensor.
        device: Only required if `value` is a scalar. The device to put the tensor on.

    Returns:
        A list of integers, where the i-th element is the value from the i-th rank.
        Could be a list of scalars or tensors based on the input `value`.
    """
    if world_size == 1:
        return [value]

    # move to CUDA
    if isinstance(value, int):
        assert device is not None, "device is required for scalar input"
        value_tensor = torch.tensor(value, dtype=torch.int, device=device)
    else:
        value_tensor = value
    assert value_tensor.is_cuda, "value should be on CUDA"

    # gather
    collected = torch.empty(world_size, dtype=value_tensor.dtype, device=value_tensor.device)
    torch.distributed.all_gather_into_tensor(collected, value_tensor)

    if isinstance(value, int):
        # return as list of integers on CPU
        return collected.tolist()
    else:
        # return as list of single-element tensors
        return collected.unbind()


def evenly_divisible_all_gather(world_size: int, data: torch.Tensor) -> torch.Tensor:
    """
    Gather tensors from all processes, handling uneven batch sizes automatically.

    In distributed training, different processes may have different batch sizes
    (especially for the last batch). This function pads smaller batches with NaN
    values, performs the all_gather operation, then removes the padding to return
    the concatenated data from all processes.

    This is essential for operations that need to see data from all processes
    (like computing global statistics) while handling the common case where
    batch sizes don't divide evenly across processes.

    Reference: https://github.com/pytorch/ignite/issues/1569#issuecomment-767247092

    Args:
        world_size: The total number of processes in the distributed group.
        data: Tensor to gather from this process. First dimension is the batch size.

    Returns:
        Concatenated tensor containing data from all processes, with padding removed.
        Shape: [total_samples_across_all_processes, *original_shape[1:]]
    """
    if world_size == 1:
        return data
    # make sure the data is evenly-divisible on multi-GPUs
    length = data.shape[0]
    all_lens = all_gather_int32(world_size, length, device=data.device)
    max_len = max(all_lens)
    if length < max_len:
        size = [max_len - length] + list(data.shape[1:])
        data = torch.cat([data, data.new_full(size, float("NaN"))], dim=0)
    # all gather across all processes (differentiable)
    data = torch.distributed.nn.functional.all_gather(data)  # type: ignore[attr-defined]
    # delete the padding NaN items
    return torch.cat([data[i][:l, ...] for i, l in enumerate(all_lens)], dim=0)


def all_gather_tensor_list(world_size: int, tensor_list: List[torch.Tensor]) -> List[torch.Tensor]:
    """Gather a list of tensors from all ranks.

    Reference: https://github.com/nerfstudio-project/gsplat/blob/2323de5905d5e90e035f792fe65bad0fedd413e7/gsplat/distributed.py#L102C1-L168C1

    .. note::
        This function expects the tensors in the `tensor_list` to have the same shape
        and data type across all ranks.

    .. note::
        This function is differentiable to the tensors in `tensor_list`.

    .. note::
        For efficiency, this function internally concatenates the tensors in `tensor_list`
        and performs a single gather operation. Thus it requires all tensors in the list
        to have the same first-dimension size.

    Args:
        world_size: The total number of ranks.
        tensor_list: A list of tensors to gather. The size of the first dimension of all
            the tensors in this list should be the same. The rest dimensions can be
            arbitrary. Shape: [(N, *), (N, *), ...]

    Returns:
        A list of tensors gathered from all ranks, where the i-th element is corresponding
        to the i-th tensor in `tensor_list`. The returned tensors have the shape
        [(N * world_size, *), (N * world_size, *), ...]

    Examples:

    .. code-block:: python

        >>> # on rank 0
        >>> # tensor_list = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        >>> # on rank 1
        >>> # tensor_list = [torch.tensor([7, 8, 9]), torch.tensor([10, 11, 12])]
        >>> collected = all_gather_tensor_list(world_size, tensor_list)
        >>> # on both ranks
        >>> # [torch.tensor([1, 2, 3, 7, 8, 9]), torch.tensor([4, 5, 6, 10, 11, 12])]

    """
    if world_size == 1:
        return tensor_list

    N = len(tensor_list[0])
    for tensor in tensor_list:
        assert len(tensor) == N, "All tensors should have the same first dimension size"

    # concatenate tensors and record their sizes
    # Handle edge case where N == 0 (empty tensors)
    if N == 0:
        reshaped_tensors = []
        for t in tensor_list:
            # Empty tensor - infer feature_dim from shape
            feature_dim = 1
            for dim in t.shape[1:]:
                feature_dim *= dim
            reshaped_tensors.append(t.reshape(0, feature_dim))
        data = torch.cat(reshaped_tensors, dim=-1)
        sizes = [t.shape[-1] for t in reshaped_tensors]
    else:
        data = torch.cat([t.reshape(N, -1) for t in tensor_list], dim=-1)
        sizes = [t.numel() // N for t in tensor_list]

    collected = evenly_divisible_all_gather(world_size, data)

    # split the collected tensor and reshape to the original shape
    out_tensor_tuple = torch.split(collected, sizes, dim=-1)
    out_tensor_list = []
    for out_tensor, tensor in zip(out_tensor_tuple, tensor_list):
        out_tensor = out_tensor.reshape(collected.shape[0], *tensor.shape[1:])  # [~ N * world_size, *]
        out_tensor_list.append(out_tensor)
    return out_tensor_list


class SetZeroFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(input)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None
        return torch.zeros_like(grad_output)


class SetZeroScalarFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.input_shape = input.shape
        return torch.tensor(0.0, device=input.device, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None
        return torch.zeros(ctx.input_shape, device=grad_output.device, dtype=grad_output.dtype)


def set_zero(input: torch.Tensor) -> torch.Tensor:
    """
    Set the given tensor to zero, but still keep the computation graph.
    i.e. set_zero(x) -> x * 0
    This is useful e.g. in DDP setting where we still want to keep the gradient flow to the leaf parameters,
    so that we no longer need find_unused_parameters=True.
    """
    return SetZeroFunction.apply(input)


def set_zero_scalar(input: torch.Tensor) -> torch.Tensor:
    """
    Similar to set_zero, but will return a scalar zero tensor.
    """
    return SetZeroScalarFunction.apply(input)


def stop_gradient(input: torch.Tensor) -> torch.Tensor:
    """
    Stop the gradient from flowing through the given tensor, but still keep the computation graph.
    """
    return set_zero(input) + input.detach()


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
