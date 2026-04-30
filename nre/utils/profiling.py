# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Phase 1 step 4.3: predict-only standalone keeps a no-op profiling shim.
# The full implementation lived under torch.profiler / Tracy / cProfile and
# was used during training and benchmarking — predict only needs the symbols
# (decorator + context manager) so the imports resolve.

from __future__ import annotations

from enum import IntEnum
from typing import Any, Callable


class TimingTag(IntEnum):
    DEFAULT = 0
    DATALOADER = 1
    TRAIN_STEP = 2
    BACKWARD = 3
    OPTIMIZER = 4


class ScopedTimer:
    """No-op context manager + decorator. Real implementation removed."""

    def __init__(self, name: str = "", tag: TimingTag = TimingTag.DEFAULT, *_args: Any, **_kwargs: Any):
        self.name = name
        self.tag = tag

    def __enter__(self) -> "ScopedTimer":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def __call__(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn


def mark_frame_boundary(_name: str | None = None) -> None:
    return None
