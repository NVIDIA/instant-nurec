# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
This module contains extensions to the torch.nn module that are used in the NRE codebase.
Its dependencies should be kept minimal to avoid circular dependencies (i.e. depend only on third party and potentially minor internal utils).
"""

from __future__ import annotations

from typing import Any, Callable, Generic, Iterable, Iterator, Mapping, Optional, Type, TypeVar, cast

import lietorch as lt
import torch
import torch.nn as nn

from nre.utils.geometry import matrix_to_rotation_6d, rotation_6d_to_matrix, se3_matrix_to_se3


G = TypeVar("G", bound=lt.groups.LieGroup)


class LieGroupBuffer(nn.Module, Generic[G]):
    """
    A wrapper for a lie group which behaves as a regular torch module.
    It parameterizes the group directly -> .get() is zero-cost but optimization is impossible.
    """

    data: nn.Buffer

    def __init__(self, group: G):
        super().__init__()
        self.group_type: Type[G] = type(group)
        self.data = nn.Buffer(group.data)

    def get(self) -> G:
        return self.group_type(self.data)

    @property
    def device(self) -> torch.device:
        return self.data.device

    def requires_grad_(self, requires_grad: bool = True) -> LieGroupBuffer:
        # can never be optimized
        return self


class LieGroupParameter(nn.Module, Generic[G]):
    """
    A wrapper for a lie group which behaves as a regular torch module.
    It parameterizes the group in log-space -> .get() has some cost to enable optimization.
    """

    def __init__(self, group_type: Type[G], log: torch.Tensor, requires_grad: bool = True):
        super().__init__()
        self.group_type: Type[G] = group_type
        self.log = nn.Parameter(log, requires_grad=requires_grad)

        assert (log_dim := self.log.shape[-1]) == self.group_type.manifold_dim, (
            f"log has incompatible dimension {log_dim} for group "
            f"{self.group_type.group_name} with manifold dimension {self.group_type.manifold_dim}"
        )

    def get(self) -> G:
        return self.group_type.exp(self.log)

    @property
    def device(self) -> torch.device:
        return self.log.device

    def freeze(self) -> LieGroupBuffer[G]:
        """Freeze and convert to a directly parameterized LieGroupBuffer."""
        return LieGroupBuffer(self.get())


class LieGroupWithDeltaParameter(nn.Module, Generic[G]):
    """
    A Lie group expressed as a fixed `base` plus an (optionally) learnable offset `delta`.

    Post-multiplies the delta by the base group if `postmult` is True. If the base transformation
    is a pose from a local frame (e.g., sensor in a T_sensor_rig extrinsic transformation,
    or rig in a T_rig_world pose) it is advisable to postmultiply the delta by the base group,
    as the delta are small local perturbations around the local frame
    otherwise numerical lever effects can occur if delta transformations are around the global frame
    reducing the stability of the estimation).
    """

    def __init__(self, base: G, delta: G | None, requires_grad: bool = True, postmult: bool = True) -> None:
        super().__init__()
        self.group = LieGroupBuffer(base)
        self.postmult = postmult

        if delta is None:
            log_delta = torch.zeros(base.shape + (base.manifold_dim,), device=base.device, dtype=base.dtype)
        else:
            log_delta = delta.log()

        assert log_delta.shape == base.shape + (base.manifold_dim,), (
            f"delta log has incompatible shape {log_delta.shape} for base group "
            f"{base.group_name} with shape {base.shape} and manifold dimension {base.manifold_dim}"
        )

        self.delta_group: LieGroupParameter[G] = LieGroupParameter(
            group_type=type(base),
            log=log_delta,
            requires_grad=requires_grad,
        )

    def get(
        self, idxs: torch.Tensor | None | slice, with_delta: bool = True, skip_delta_idxs: torch.Tensor | None = None
    ) -> G:
        if idxs is None:
            idxs = slice(None)  # grab the full tensor

        if not with_delta:
            return self.group.get()[idxs]

        if skip_delta_idxs is None:
            delta = self.delta_group.get()
        else:
            log_delta = self.delta_group.log.clone()
            log_delta[skip_delta_idxs] = 0.0
            delta = self.delta_group.group_type.exp(log_delta)

        # If idxs oversamples (is a long tensor with repeating entries) it's cheaper to multiply -> overindex.
        group = self.group.get()
        if (group_indexed := group[idxs]).shape[0] > group.shape[0]:
            # multiply first and overindex
            return delta.mul(group)[idxs] if not self.postmult else group.mul(delta)[idxs]
        else:
            # index first and multiply subset only
            return delta[idxs].mul(group_indexed) if not self.postmult else group_indexed.mul(delta[idxs])


class SixDPoseWithDeltaParameter(nn.Module):
    """
    Poses encoded as 3d translation and 6d rotation expressed as a fixed `base` 4x4 poses plus an (optionally) learnable offset `delta`.

    Deltas are always applied in a post-multiplication way to the base pose.
    """

    def __init__(self, base: torch.Tensor, delta: torch.Tensor | None, requires_grad: bool = True) -> None:
        super().__init__()

        self.base = nn.Buffer(base)

        assert base.shape[-2:] == (4, 4), f"Base pose must be a batch 4x4 matrices, got {base.shape[-2:]}"

        batch_shape = base.shape[:-2]

        # representation of identity (3d translation + 6d rotation, 9d total)
        delta_repr = torch.zeros(size=batch_shape + (9,), device=self.base.device)
        if delta is not None:
            assert delta.shape[-2:] == (4, 4), f"Delta pose must be a batch 4x4 matrices, got {delta.shape[-2:]}"
            assert delta.shape[:-2] == batch_shape, (
                f"Delta pose must be a batch of 4x4 matrices of same batch size as the base poses"
            )

            delta_repr[..., :3] = delta[..., :3, 3]  # translation around zero element
            delta_repr[..., 3:] = matrix_to_rotation_6d(delta[..., :3, :3])  # rotation around zero element

        self.delta_repr = nn.Parameter(delta_repr, requires_grad=requires_grad)

    def get(
        self,
        idxs: torch.Tensor | None | slice,
        with_delta: bool = True,
        skip_delta_idxs: torch.Tensor | None = None,
    ) -> lt.SE3:
        if idxs is None:
            idxs = slice(None)  # grab the full tensor

        if not with_delta:
            return se3_matrix_to_se3(self.base[idxs], unbatch=False)

        delta_repr: torch.Tensor | torch.nn.Parameter
        if skip_delta_idxs is None:
            delta_repr = self.delta_repr
        else:
            delta_repr = self.delta_repr.clone()
            delta_repr[skip_delta_idxs] = 0.0

        dx, drot = delta_repr[..., :3], delta_repr[..., 3:]
        delta_transform = torch.eye(4, device=delta_repr.device).expand(delta_repr.shape[:-1] + (4, 4)).clone()
        delta_transform[..., :3, :3] = rotation_6d_to_matrix(drot)
        delta_transform[..., :3, 3] = dx

        # If idxs oversamples (is a long tensor with repeating entries) it's cheaper to multiply -> overindex.
        if len(base_indexed := self.base[idxs]) > len(self.base):
            # multiply first and overindex
            poses = (self.base @ delta_transform)[idxs]
        else:
            # index first and multiply subset only
            poses = base_indexed @ delta_transform[idxs]

        # convert to SE3
        return se3_matrix_to_se3(poses, unbatch=False)


class BufferList(nn.ParameterList):
    def __setitem__(self, idx: int, buffer: Any) -> None:
        idx = self._get_abs_string_index(idx)
        if isinstance(buffer, torch.Tensor) and not isinstance(buffer, nn.Buffer):
            buffer = nn.Buffer(buffer)
        assert isinstance(buffer, nn.Buffer)
        return setattr(self, str(idx), buffer)


V = TypeVar("V", bound=nn.Module)


class TypedModuleDict(nn.ModuleDict, Generic[V]):
    """
    An nn.ModuleDict that enforces type checking of the contained modules.
    Generic only over values V, as keys are always strings.

    We violate the Liskov substitution principle, hence the type: ignore[override] annotations.

    This has no runtime consequences and shouldn't matter unless storing the TypedModuleDict in
    a container like list[nn.ModuleDict], which would allow for type erasure of the TypedModuleDict[V].
    Can be addressed later if it becomes a problem.
    """

    def __init__(self, modules: Optional[Mapping[str, V]] = None):
        super().__init__(modules)

    def __getitem__(self, key: str) -> V:
        return super().__getitem__(key)

    def __setitem__(self, key: str, module: V) -> None:  # type: ignore[override]
        super().__setitem__(key, module)

    def pop(self, key: str) -> V:
        return cast(V, super().pop(key))

    def values(self) -> Iterable[V]:
        return cast(Iterable[V], super().values())

    def items(self) -> Iterable[tuple[str, V]]:
        return cast(Iterable[tuple[str, V]], super().items())

    def update(self, modules: Mapping[str, V]) -> None:  # type: ignore[override]
        super().update(modules)


class TypedModuleList(nn.ModuleList, Generic[V]):
    """
    An nn.ModuleList that enforces type checking of the contained modules.

    We violate the Liskov substitution principle, hence the type: ignore[override] annotations.

    This has no runtime consequences and shouldn't matter unless storing the TypedModuleList in
    a container like list[nn.ModuleList], which would allow for type erasure of the TypedModuleList[V].
    Can be addressed later if it becomes a problem.
    """

    def __init__(self, modules: Optional[Iterable[V]] = None):
        super().__init__(modules)

    def __getitem__(self, key: int) -> V:  # type: ignore[override]
        return cast(V, super().__getitem__(key))

    def __setitem__(self, key: int, module: V) -> None:  # type: ignore[override]
        super().__setitem__(key, module)

    def append(self, module: V) -> None:  # type: ignore[override]
        super().append(module)

    def extend(self, modules: Iterable[V]) -> None:  # type: ignore[override]
        super().extend(modules)

    def insert(self, index: int, module: V) -> None:  # type: ignore[override]
        super().insert(index, module)

    def pop(self, index: int = -1) -> V:  # type: ignore[override]
        return cast(V, super().pop(index))

    def __iter__(self) -> Iterator[V]:  # type: ignore[override]
        return cast(Iterator[V], super().__iter__())


C = TypeVar("C", bound=Callable)


def module_call_type(forward_fn: C) -> C:  # `forward_fn` is unused but its type `C` is
    """
    When defining an nn.Module (or subclass) use this to create the `__call__` method like so:

    ```
    class MyModule(nn.Module):
        def forward(self, arg1: Type1, arg2: Type2) -> ReturnType: ...
        __call__ = module_call_type(forward)
    ```

    This will make calls like

    ```
    module = MyModule()
    result = module(arg1, arg2)
    ```

    correctly typed and benefit from mypy. Since we usually define `forward` but call via `__call__`,
    without this helper mypy treats `__call__` as untyped.

    This function works by "stealing" the annotations from `forward` and applying them to `__call__`.
    It needs to be re-applied whenever the signature of `forward` changes w.r.t. superclass (this is
    why we can't simply put it in `BaseModel` and be done).
    """

    def call(self, *args, **kwargs):
        """A closure which calls Module.__call__ (which internally calls forward + various hooks)"""
        if not isinstance(self, nn.Module):
            raise TypeError(f"`module_call_type` used on a class which does not derive from nn.Module.")
        return nn.Module.__call__(self, *args, **kwargs)

    # cast the type of `call` to align with the type of `forward_fn`.
    # it would be nicer to write `cast(C, call)` but for some reason mypy rejects that
    # so we just type ignore.
    return call  # type: ignore
