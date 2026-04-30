# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import functools
import os
import re

from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch._dynamo


_init_torch_compile_done = False


def _init_torch_compile():
    global _init_torch_compile_done
    if _init_torch_compile_done:
        return

    # Allows to compile functions that can fit for any shapes (avoids recompilation when shapes change)
    torch._dynamo.config.force_parameter_static_shapes = False
    torch._dynamo.config.force_nn_module_property_static_shapes = False
    torch._dynamo.config.compiled_autograd = True
    torch._dynamo.config.compiled_autograd_kwargs_override = {"fullgraph": True}

    _init_torch_compile_done = True


class TorchCompile:
    # Putting this decorator inside a class allows us to work with pycena more easily.
    @staticmethod
    def conditional(*decorator_args: Any, **decorator_kwargs: Any) -> Callable[[Callable], Callable]:
        """
        Decorator to conditionally compile the function with torch.compile.
        The compiled function will be executed if the extra argument `enable_torch_compile` is True, otherwise it will be executed normally.

        Usage:
        @TorchCompile.conditional(fullgraph=True, dynamic=True)
        def my_function(*args, **kwargs) -> Any:
            ...

        @TorchCompile.conditional() # Don't forget () even if there are no arguments
        def my_function(*args, **kwargs) -> Any:
            ...

        my_function(..., enable_torch_compile=True)
        my_function(..., enable_torch_compile=False)

        Args:
            decorator_args: Positional arguments to pass to torch.compile.
            decorator_kwargs: Keyword arguments to pass to torch.compile.
        """

        def decorator(func: Callable) -> Callable:
            class ConditionalTorchCompile:
                def __init__(self, func: Callable) -> None:
                    self.func = func
                    self.func_compiled: Optional[Callable] = None

                @functools.wraps(func)
                def __call__(self, *args: Any, enable_torch_compile: bool = False, **kwargs: Any) -> Any:
                    if enable_torch_compile:
                        if self.func_compiled is None:
                            _init_torch_compile()
                            self.func_compiled = torch.compile(*decorator_args, **decorator_kwargs)(self.func)
                        assert self.func_compiled is not None, "Function should be compiled"
                        return self.func_compiled(*args, **kwargs)
                    else:
                        return self.func(*args, **kwargs)

            return ConditionalTorchCompile(func)

        return decorator
