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


def assert_device_on_local_rank(device: torch.device | str, local_rank: int | None = None) -> None:
    """Verify that a specific device matches the expected local rank.

    This validation ensures that the given device is on the correct GPU managed by the current process.
    Useful for debugging device placement issues.

    Args:
        device: The device to validate.
        local_rank: The expected local GPU index within the current node.
            If None, automatically determines the expected local rank from environment variables or
            distributed context if available.
    """
    local_rank = _get_local_rank() if local_rank is None else local_rank
    device_idx = torch.cuda.device(device).idx
    assert device_idx == local_rank, f"Device ID {device_idx} mismatch local rank {local_rank}"


def sync_objects(func: Callable[[], Any], master_rank: int = 0) -> Any:
    """Synchronize arbitrary Python objects across all processes in distributed training.

    This function ensures all processes get the same object by having only the master rank
    execute the function, then broadcasting the result to all other ranks. This is useful
    for operations that should only happen once (like random number generation, file I/O,
    or expensive computations) but where all processes need the same result.

    Common use cases:
    - Generating random seeds that should be consistent across processes
    - Loading configuration or data that should be identical everywhere
    - Computing expensive values once and sharing them
    - Ensuring deterministic behavior in distributed settings

    Note: This uses PyTorch's broadcast_object_list which handles arbitrary Python objects
    but is not differentiable. For tensor synchronization with gradients, use other methods.

    Args:
        func: A callable that returns the object to synchronize. Only executed on master_rank.
        master_rank: The process rank that executes the function and broadcasts the result.
            All other ranks receive this result. Defaults to 0.

    Returns:
        The result of func() from the master rank, identical across all processes.
        If not in distributed mode, simply returns func().

    Example:
        # Ensure all processes use the same random seed
        seed = sync_objects(lambda: random.randint(0, 1000000))

        # Load config once and share across processes
        config = sync_objects(lambda: load_expensive_config())
    """
    # Return directly if we are not in distributed mode
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return func()

    # We need to synchronize across all ranks
    assert torch.distributed.is_initialized(), "Distributed must be initialized to use sync_objects"

    object_list = [func() if torch.distributed.get_rank() == master_rank else None]
    torch.distributed.broadcast_object_list(object_list, src=master_rank)

    return object_list[0]


class SyncedCallError(RuntimeError):
    """Raised on non-master ranks when sync_objects_or_raise() receives a remote failure."""


@dataclass(frozen=True)
class SyncedSuccessPayload:
    """Serializable success payload broadcast by sync_objects_or_raise()."""

    value: Any
    ok: bool = True


@dataclass(frozen=True)
class SyncedExceptionPayload:
    """Serializable failure metadata broadcast by sync_objects_or_raise()."""

    master_rank: int
    message: str
    traceback: str
    ok: bool = False

    @classmethod
    def from_exception(cls, exc: BaseException, master_rank: int) -> "SyncedExceptionPayload":
        return cls(
            master_rank=master_rank,
            message=str(exc),
            traceback=traceback.format_exc(),
        )


SyncedPayload = SyncedSuccessPayload | SyncedExceptionPayload


def _coerce_synced_payload(payload: object) -> SyncedPayload:
    if isinstance(payload, (SyncedSuccessPayload, SyncedExceptionPayload)):
        return payload

    # Keep dict payloads for rolling upgrades from older ranks that still broadcast the legacy shape.
    if not isinstance(payload, dict):
        raise TypeError(f"sync_objects_or_raise() expected a synced payload dataclass, got {type(payload).__name__}")

    if payload.get("ok"):
        return SyncedSuccessPayload(value=payload["value"])

    return SyncedExceptionPayload(
        master_rank=payload["master_rank"],
        message=payload["message"],
        traceback=payload["traceback"],
    )


def _reconstruct_synced_exception(payload: SyncedExceptionPayload) -> BaseException:
    exc = SyncedCallError(f"Callable failed on rank {payload.master_rank}: {payload.message}")

    if payload.traceback:
        exc.add_note(f"Raised on rank {payload.master_rank} during sync_objects_or_raise().")
        exc.add_note(payload.traceback)

    return exc


# Prefer this over sync_objects(): old callers remain there, but rank-0 failure may leave peers waiting.
def sync_objects_or_raise(func: Callable[[], Any], master_rank: int = 0) -> Any:
    """Synchronize arbitrary Python objects across ranks and re-raise master-rank failures everywhere.

    This is similar to sync_objects(), but preserves failure visibility for rank-0-only callables:
    if func() raises on master_rank, the failure payload is broadcast first and then re-raised on
    all ranks instead of leaving non-master ranks waiting inside the collective.
    """
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return func()

    object_list: list[SyncedPayload | None]
    if torch.distributed.get_rank() == master_rank:
        try:
            success_payload = SyncedSuccessPayload(value=func())
            object_list = [success_payload]
        except BaseException as exc:
            object_list = [SyncedExceptionPayload.from_exception(exc, master_rank)]
            torch.distributed.broadcast_object_list(object_list, src=master_rank)
            raise
        torch.distributed.broadcast_object_list(object_list, src=master_rank)
        return success_payload.value

    object_list = [None]
    torch.distributed.broadcast_object_list(object_list, src=master_rank)
    payload = _coerce_synced_payload(unpack_optional(object_list[0]))
    if isinstance(payload, SyncedSuccessPayload):
        return payload.value
    raise _reconstruct_synced_exception(payload)


def compute_process_local_rng_seed(global_seed: Optional[int] = None):
    """Computes a deterministic local random seed for each distributed process.

    This function ensures that each process in a distributed setting gets a unique but
    deterministic random seed derived from a global seed. This is useful for maintaining
    reproducibility while allowing different processes to have different random sequences.

    Args:
        global_seed (Optional[int]): The global seed to use as a base. If None, will try to
            use PL_GLOBAL_SEED from environment or default to 0.

    Returns:
        int: A unique local seed for the current process. For rank 0, this will be the
            global_seed. For other ranks, it will be a deterministic derivative.
    """
    if global_seed is None:
        if "PL_GLOBAL_SEED" in os.environ:
            # This is set by pytorch_lightning.seed_everything call in run/main.py
            global_seed = int(os.environ["PL_GLOBAL_SEED"])
        else:
            global_seed = 0

    rank = _get_rank()

    # Rank 0 always get the global_seed.
    if rank == 0:
        local_seed = global_seed
    # Seed for all others is an unique derivative of rank 0's seed.
    else:
        rng = random.Random(global_seed)

        # Advance the global seed to get a unique seed for each rank below current
        for _ in range(rank):
            rng.random()

        local_seed = rng.randrange(2**31 - 1)

    return local_seed


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    """Converts a linear RGB value to sRGB"""
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x**0.41666 - 0.055)


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """Converts a sRGB value to linear RGB"""
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def get_worker_id() -> Optional[int]:
    match torchdata.get_worker_info():
        case None:
            return None
        case torch.utils.data._utils.worker.WorkerInfo(id=worker_id):  # type:ignore
            return worker_id
        case _:
            raise ValueError("get_worker_info() resulting in an invalid return")


@torch.no_grad()
def get_pack_info_from_n(n_per_pack: torch.Tensor) -> torch.Tensor:
    """Given an array of N pack element counts, returns the corresponding (N, 2) pack_info with per-pack start_idx / N_elements"""

    assert n_per_pack.dim() == 1 and not n_per_pack.is_floating_point(), (
        "get_pack_info_from_n(): required 1d integer as number-per-pack input"
    )

    return torch.stack(
        [n_per_pack.cumsum(0, dtype=n_per_pack.dtype) - n_per_pack, n_per_pack], 1
    )  # no exclusive cumsum in pytorch, emulate by substraction


def decode_datatype(type_name: str) -> torch.dtype:
    """Convert a string representation of a data type to a PyTorch dtype.

    Useful for configuration files or command-line arguments where data types
    are specified as strings and need to be converted to actual PyTorch dtypes.

    Args:
        type_name: String name of the data type ("float16" or "float32").

    Returns:
        The corresponding PyTorch dtype.

    Raises:
        KeyError: If type_name is not a supported data type.
    """
    data_types = {
        "float16": torch.float16,
        "float32": torch.float32,
    }

    return data_types[type_name]


def lexsort(keys: list[torch.Tensor], dim: int = -1, descending: bool = False) -> torch.Tensor:
    """
    Lexicographically sorts the keys provided in the list.
    The first key is the most significant, and the last key is the least significant.

    Returns:
        - indices: indices to sort the keys [long]. \
            If the keys are 1-D, `indices` will also be 1-D. \
            If the keys are multi-dimensional, `indices` will have the same number of dimensions \
                and should be used with `torch.gather(key, dim=dim, index=indices)`
    """
    assert len(keys) > 0, "Expect at least one key"

    kwargs: dict = dict(descending=descending, dim=dim, stable=True)
    keys = keys[::-1]
    out = keys[0].argsort(**kwargs)

    if keys[0].dim() == 1:
        for k in keys[1:]:
            out = out[k[out].argsort(**kwargs)]
    else:
        for k in keys[1:]:
            idx = torch.gather(k, dim=dim, index=out).argsort(**kwargs)
            out = torch.gather(out, dim=dim, index=idx)
    return out


def merge_pack_info(
    pack_info: list[torch.Tensor], secondary_sample_keys: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Merge pack_info information, where pack_info is a list of packinfo with sizes [n_rays, 2].
    It allows for an additional secondary key which is of the same size of samples.
    TODO[JH]: Replace this with a custom cuda function

    Returns:
        - pack_info: merged pack_info (n_rays, 2) [int]
        - indices: indices to perturb the sample keys (n_samples, ) [long]
    """
    assert len(pack_info) == len(secondary_sample_keys)
    assert len(pack_info) > 0, "pack_info must not be empty"

    num_rays = pack_info[0].size(0)

    primal_sample_keys = []
    for i, pack_info_item in enumerate(pack_info):
        rays_ridx = torch.arange(pack_info_item.size(0), device=pack_info_item.device).repeat_interleave(
            pack_info_item[:, 1]
        )
        assert pack_info_item.size(0) == num_rays
        assert rays_ridx.size(0) == secondary_sample_keys[i].size(0)
        primal_sample_keys.append(rays_ridx)

    primal_sample_key = torch.cat(primal_sample_keys)
    del primal_sample_keys

    argsort_inter = lexsort([primal_sample_key, torch.cat(secondary_sample_keys)])
    primal_sample_key = primal_sample_key[argsort_inter]

    count = torch.bincount(primal_sample_key, minlength=num_rays)
    new_pack_info = get_pack_info_from_n(count).int()
    return new_pack_info, argsort_inter


T = TypeVar("T")
U = TypeVar("U")
KT = TypeVar("KT")
VT = TypeVar("VT")


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


def to_torch_optional(
    data: Optional[npt.NDArray], device: str | torch.device, dtype: Optional[torch.dtype] = None
) -> Optional[torch.Tensor]:
    """Wrapper of 'to_torch', simply bypassing tensor creation if input is 'None'"""
    return to_torch(data, device, dtype) if data is not None else None


def to_numpy_optional(data: Optional[torch.Tensor]) -> Optional[npt.NDArray]:
    """Wrapper of 'to_numpy', simply bypassing array creation if input is 'None'"""
    return to_numpy(data) if data is not None else None


def decorate_all(decorators: list[Callable]) -> Callable:
    """A decorator to decorate all member functions of a class

    Args:
        decorators: list of decorators to add to all functions in the class
    """

    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)) and attr != "__init__":
                for decorator in decorators:
                    setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls

    return decorate


def tree_map(maybe_dataclass: Any, fn: Callable) -> Any:
    """
    Applies a function recursively to a tree of dataclasses
    """
    if is_dataclass(type(maybe_dataclass)):
        new_fields = {}
        for field in fields(maybe_dataclass):
            value = getattr(maybe_dataclass, field.name)
            new_fields[field.name] = tree_map(value, fn)

        return replace(maybe_dataclass, **new_fields)
    else:
        return fn(maybe_dataclass)


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


def to_float_device(
    device: torch.device,
) -> Callable[
    [
        Any,
    ],
    Any,
]:
    """
    Creates a function which maps tensors to a given device and float32 (if they are floating point)
    while doing nothing to non-tensors.
    Used alongside `tree_map` to prepare structs like `RayBundle` for inference by

    ```
    rays_device = tree_map(rays_cpu, to_float_device(device))
    ```
    """

    def apply(maybe_tensor: Any) -> Any:
        if torch.is_tensor(maybe_tensor):
            if torch.is_floating_point(maybe_tensor):
                maybe_tensor = maybe_tensor.to(torch.float32)
            return maybe_tensor.to(device)
        else:
            return maybe_tensor

    return apply


def check_same_size_and_device(tensors: list[torch.Tensor], compare_first_dim_only: bool = False) -> bool:
    # Check if the list is empty
    if len(tensors) == 0:
        return True

    # Get size and device of the first tensor
    size: int | torch.Size
    if compare_first_dim_only:
        size = tensors[0].size(0)
    else:
        size = tensors[0].size()

    device = tensors[0].device

    # Iterate through the rest of the tensors
    for tensor in tensors[1:]:
        # Check if the size and device match
        if compare_first_dim_only:
            if tensor.size(0) != size or tensor.device != device:
                return False
        else:
            if tensor.size() != size or tensor.device != device:
                return False

    return True


def index_tensors_with_mask(mask: torch.Tensor, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Given a boolean mask and a list of torch.Tensors, returns a list of indexed tensors at the non-masked
    elements along their first dimension
    """

    assert check_same_size_and_device(tensors, compare_first_dim_only=True), "Tensors need to be on the same device"
    assert mask.size(0) == tensors[0].size(0), "Mask and tensors need to have the same size of the first dimension"
    assert mask.dtype == torch.bool, "Expected data type for the mask is bool"

    # TODO: check if this is actually a faster way of indexing
    indices = mask.nonzero().long()[..., 0]

    return [tensor[indices] for tensor in tensors]


def geometric_mean(input: torch.Tensor, dim: Optional[int] = None, keepdim: bool = False) -> torch.Tensor:
    """
    Returns the geometric mean of `input` tensor at the optionally given `dim`.
    Both `dim` and `keepdim` are directly passed to `torch.mean()` function.
    """
    return input.log().mean(dim=dim, keepdim=keepdim).exp()


def scale_distances_on_dirs(dists: torch.Tensor, dirs: torch.Tensor, scales_nd: torch.Tensor) -> torch.Tensor:
    """
    Given oriented distances `dists` with their directions given by `dirs` unit vectors,
    returns the scaled distance with potentially rectangular (anisotropic) scaling.

    Args:
        dists (torch.Tensor): [...] The given distances
        dirs (torch.Tensor): [..., nd] The given N-D unit direction vectors corresponding to each given input distance
        scales_nd (torch.Tensor): [nd] The potentially rectangular (anisotropic) N-D scaling

    Returns:
        torch.Tensor: The scaled distances
    """
    nd = scales_nd.size(0)
    dirs = F.normalize(dirs, dim=-1).expand(*dists.shape, nd)
    scales_on_dirs = (dirs * scales_nd.view(*[1] * (dirs.dim() - 1), nd)).square().sum(dim=-1).sqrt()
    return dists * scales_on_dirs


def zip_dict(d: dict) -> Generator[dict, Any, Any]:
    # https://stackoverflow.com/a/69578838/11121534
    """
    A generator that zip dict's values

    Example:
        >>> d = dict(
        >>>     x=[1,3,5],
        >>>     y=[2,4,6])

        >>> for t in zip_dict(d):
        >>>     print(t)

        The result:
            {'x': 1, 'y': 2}
            {'x': 3, 'y': 4}
            {'x': 5, 'y': 6}
    """
    for vals in zip(*(d.values())):
        yield dict(zip(d.keys(), vals))


def nested_dict_keys(d: dict, pre=None) -> Generator[KT, Any, Any]:
    """
    A generator for DFS traversal of a nested dict's key tree
    """
    pre = pre[:] if pre else []
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from nested_dict_keys(v, pre + [k])
            else:
                yield pre + [k]
    else:
        yield pre


def nested_dict_values(d: dict) -> Generator[VT, Any, Any]:
    """
    A generator for DFS traversal of a nested dict's values
    """
    for v in d.values():
        if isinstance(v, dict):
            yield from nested_dict_values(v)
        else:
            yield v


def nested_dict_items(d: dict, pre=None) -> Generator[list, Any, Any]:
    # https://stackoverflow.com/questions/12507206/how-to-completely-traverse-a-complex-dictionary-of-unknown-depth
    """
    A generator for DFS traversal of a nested dict's keytree-value pairs

    Example:
        >>> for *k, v in nested_dict_items(d):
        >>>    print(k, v)
    """
    pre = pre[:] if pre else []
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                # for d in nested_dict_items(value, pre + [k]):
                #     yield d
                # NOTE: equivalent
                yield from nested_dict_items(v, pre + [k])
            else:
                yield pre + [k, v]
    else:
        yield pre + [d]


def nested_dict_del(d: dict):
    """
    DFS deletion of a nested dict
    """
    for k in list(d.keys()):
        if isinstance(d[k], dict):
            nested_dict_del(d[k])
        del d[k]


def nested_dict(iterable: Iterable[Tuple[list[KT], VT]]) -> dict:
    """
    Construct a nested dict given lists of keytree-value pairs
    """
    d: dict = {}
    for ks, v in iterable:
        _d = d
        for _k in ks[:-1]:
            if _k not in _d:
                _d[_k] = dict()
            _d = _d[_k]
        _d[ks[-1]] = v
    return d


def zip_nested_dict(d: dict) -> Generator[dict, Any, Any]:
    """
    `zip` a (nested) dict (also support normal dict)

    Example:
        For a nested dict
        >>> nested_d = dict(
        >>>     x=[1,3,5,7,9],
        >>>     y=[2,4,6,8,10],
        >>>     z=dict(
        >>>         z1=[0.1, 0.2, 0.3, 0.4, 0.5],
        >>>         z2=[-0.1, -0.2, -0.3, -0.4, -0.5]
        >>>     )
        >>> )
        >>> for d in zip_nested_dict(nested_d):
        >>>     print(d)

        => Output
            {'x': 1, 'y': 2, 'z': {'z1': 0.1, 'z2': -0.1}}
            {'x': 3, 'y': 4, 'z': {'z1': 0.2, 'z2': -0.2}}
            {'x': 5, 'y': 6, 'z': {'z1': 0.3, 'z2': -0.3}}
            {'x': 7, 'y': 8, 'z': {'z1': 0.4, 'z2': -0.4}}
            {'x': 9, 'y': 10, 'z': {'z1': 0.5, 'z2': -0.5}}


        Compatible with non-nested dict
        >>> normal_d = dict(
        >>>     x=[1,3,5,7,9],
        >>>     y=[2,4,6,8,10]
        >>> )
        >>> for d in zip_nested_dict(normal_d):
        >>>     print(d)

        => Output
            {'x': 1, 'y': 2}
            {'x': 3, 'y': 4}
            {'x': 5, 'y': 6}
            {'x': 7, 'y': 8}
            {'x': 9, 'y': 10}

    """
    for vals in zip(*(nested_dict_values(d))):
        yield nested_dict(zip(nested_dict_keys(d), vals))


def torch_interp1d(t_keyframes: torch.Tensor, y_keyframes: torch.Tensor, t: torch.Tensor):
    """pytorch interp1d interpolation"""
    return torch_interp1d_general(t_keyframes, y_keyframes, t, interp_fn=torch.lerp)


def torch_interp1d_general(
    t_keyframes: torch.Tensor,
    y_keyframes: torch.Tensor,
    t: torch.Tensor,
    interp_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] = torch.lerp,
) -> torch.Tensor:
    """
    - no extrapolation (snap to boundary seq vals for out-of-bound `t`)
    - support multi-dimensional `y_keyframes` (support arbitary data dimensions)

    e.g. No extrapolation (snap to boundary seq vals for out-of-bound `t`)
    >>> seq = torch.tensor([2, 3, 4], dtype=torch.float)
    >>> y = torch.tensor([20., 10., 30.], dtype=torch.float)
    >>> t = torch.tensor([-45, 1.5, 2.5, 3.5, 4.5, 100.5], dtype=torch.float)
    `inds` would be: [0, 0, 1, 2, 3, 3]
    `vals` would be: [20.0, 20.0, 15.0, 20.0, 30.0, 30.0]

    Args:
        t_keyframes (torch.Tensor): _description_
        y_keyframes (torch.Tensor): _description_
        t (torch.Tensor): _description_
        interp_fn (Callable[[torch.Tensor,torch.Tensor,torch.Tensor], torch.Tensor]): \
            A function that takes v0,v1,w as input and outputs the interpolated tensor,\
            where v0, v1 is the two boundary tensors, \
            and w is the weighting tensor in range [0,1]. w=0 indicates v0.

    Returns:
        torch.Tensor: The interpolated value
    """
    assert t_keyframes.dim() == 1, "`t_keyframes` must be 1D sequences"
    assert y_keyframes.size(0) == t_keyframes.size(0), "`y_keyframes` should correspond to `t_keyframes`"

    inds = torch.searchsorted(t_keyframes, t)  # in range [0, len]
    below, above = torch.clamp_min(inds - 1, 0), torch.clamp_max(inds, len(t_keyframes) - 1)
    inds_g = torch.stack([below, above], 0)
    bins_g = t_keyframes[inds_g]
    vals_g = y_keyframes[inds_g]
    denom = bins_g[1] - bins_g[0]
    w = (t - bins_g[0]) / denom.clamp_min(1e-5)
    w = w.view([*w.shape, *[1] * (y_keyframes.dim() - t_keyframes.dim())])
    # vals = vals_g[0] + w * (vals_g[1] - vals_g[0]) # i.e. If `interp_fn` is `torch.lerp`
    vals = interp_fn(vals_g[0], vals_g[1], w)
    return vals


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


@torch.no_grad()
def _multinomial_sample(probabilities: torch.Tensor, n: int, replacement: bool = True) -> torch.Tensor:
    """Sample from a distribution using torch.multinomial or numpy.random.choice.


    Sample points from a distribution either with  `torch.multinomial` or `numpy.random.choice`
    based on the number of elements in `probabilities`. If the number of elements exceeds
    the torch.multinomial limit (2^24), it falls back to using `numpy.random.choice`.
    This is a workaround until https://github.com/pytorch/pytorch/issues/2576 is solved.

    Args:
        probabilities (Tensor): probabilitiy of sampling each element.
        n (int): The number of samples to draw.
        replacement (bool): Whether to sample with replacement

    Returns:
        Tensor: A 1D tensor of sampled indices.
    """

    assert len(probabilities.size()) == 1, "_multinomial_sample expects a flat tensor as input"
    num_elements = probabilities.size(0)

    if num_elements <= 2**24:
        # Use torch.multinomial for elements within the limit
        return torch.multinomial(probabilities, n, replacement=replacement)
    else:
        # Fallback to numpy.random.choice for larger element spaces
        weights = probabilities / probabilities.sum()
        weights_np = weights.detach().cpu().numpy()
        sampled_idxs_np = np.random.choice(num_elements, size=n, p=weights_np, replace=replacement)
        sampled_idxs = torch.from_numpy(sampled_idxs_np)

        # Return the sampled indices on the original device
        return sampled_idxs.to(weights.device)


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
