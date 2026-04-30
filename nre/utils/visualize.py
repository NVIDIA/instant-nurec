# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Phase 1 step 4.3: predict-only standalone keeps a no-op visualization
# shim. flow2img/make_image_grid/scalar2img only fire under media-logger
# branches (training/validation), which predict never enters.

from __future__ import annotations

from typing import Any

import numpy as np


def scalar2img(*_args: Any, **_kwargs: Any) -> np.ndarray:
    return np.zeros((1, 1, 3), dtype=np.uint8)


def flow2img(*_args: Any, **_kwargs: Any) -> np.ndarray:
    return np.zeros((1, 1, 3), dtype=np.uint8)


def make_image_grid(*_args: Any, **_kwargs: Any) -> np.ndarray:
    return np.zeros((1, 1, 3), dtype=np.uint8)
