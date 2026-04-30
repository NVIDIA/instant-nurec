# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Phase 1 step 4.3: predict-only standalone keeps a no-op prober shim.
# The real TensorProber recorded reference tensors for unit-test bring-up
# of training kernels; predict only hits the ``get_global_prober()(...)``
# call sites under ``if global_step_for_prober is not None``, which never
# fires during inference. Returning ``None`` keeps the walrus-assignment
# branches inert.

from __future__ import annotations

from typing import Any


class TensorProber:
    def __call__(self, *_args: Any, **_kwargs: Any) -> None:
        return None


_global_prober: TensorProber | None = None


def get_global_prober() -> TensorProber:
    global _global_prober
    if _global_prober is None:
        _global_prober = TensorProber()
    return _global_prober
