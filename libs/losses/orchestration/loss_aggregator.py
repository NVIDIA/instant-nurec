# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Phase 1 step 4.3 stub: predict-only standalone never instantiates a loss.
# `LossAggregator` is imported by nre/nrm/systems/base.py only for the type
# annotation on `self.loss`; the constructor is called with
# `force_disable_cuda=True` and never invoked further.

from __future__ import annotations

from typing import Any


class LossAggregator:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.losses = []
        self.in_cuda = None

    def __getattr__(self, _name: str) -> Any:
        return self._noop

    @staticmethod
    def _noop(*_args: Any, **_kwargs: Any) -> None:
        return None


__all__ = ["LossAggregator"]
