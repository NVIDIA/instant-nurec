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

"""Extensions to ``torch.nn``.

Dependencies kept minimal to avoid circular imports (third-party + minor
internal utils only).
"""

from __future__ import annotations

from typing import Callable, Generic, Iterable, Iterator, Optional, TypeVar, cast

import torch.nn as nn


V = TypeVar("V", bound=nn.Module)


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
            raise TypeError("`module_call_type` used on a class which does not derive from nn.Module.")
        return nn.Module.__call__(self, *args, **kwargs)

    # cast the type of `call` to align with the type of `forward_fn`.
    # it would be nicer to write `cast(C, call)` but for some reason mypy rejects that
    # so we just type ignore.
    return call  # type: ignore
