# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import os

from contextlib import nullcontext
from typing import Any, Callable, ContextManager

import torch


logger = logging.getLogger(__name__)


# Define a profiling context manager not activated in production code
# with minimal overhead when disabled.
#
# profile() is used to instrument the code with nvtx ranges.
# enable_torch_nvtx() is mostly useful for tests.
USE_NVTX_PROFILE = False

profile: Callable[..., ContextManager[Any]]
if USE_NVTX_PROFILE:

    def profile(*args, **kwargs):
        return torch.cuda.nvtx.range(*args, **kwargs)

    def enable_torch_nvtx():
        return torch.autograd.profiler.emit_nvtx()

else:

    def profile(*_, **__):
        return nullcontext()

    def enable_torch_nvtx():
        return nullcontext()


def div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


_cached_slang_is_ninja_in_path: bool = False


def add_ninja_to_path() -> None:
    global _cached_slang_is_ninja_in_path
    if not _cached_slang_is_ninja_in_path:
        from python.runfiles import runfiles

        R = runfiles.Create()
        if R:
            ninja_path = R.Rlocation("nre_repo/libs/slang_utils/ninja")
            if ninja_path:
                os.environ["PATH"] = (
                    os.path.realpath(os.path.dirname(ninja_path)) + os.path.pathsep + os.environ["PATH"]
                )

        _cached_slang_is_ninja_in_path = True
