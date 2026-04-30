# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tracy profiler Python integration for NRE - interface module for compiled extensions."""

import importlib

from typing import Any


# Import Tracy utils - the correct variant is provided by the Bazel target
# - pylib → tracy_utils_py.so (default CPU-only Tracy)
# - pylib_gpu → tracy_utils_gpu_py.so (full GPU Tracy)
# - pylib_disabled → tracy_utils_disabled_py.so (no Tracy symbols)
try:
    tracy_utils_py: Any = importlib.import_module("libs.tracy_utils.tracy_utils_py")

    TRACY_AVAILABLE = tracy_utils_py.is_available()
    PlotType = tracy_utils_py.PlotType if TRACY_AVAILABLE else None
except ImportError:
    try:
        tracy_utils_py: Any = importlib.import_module("libs.tracy_utils.tracy_utils_gpu_py")

        TRACY_AVAILABLE = tracy_utils_py.is_available()
        PlotType = tracy_utils_py.PlotType if TRACY_AVAILABLE else None
    except ImportError:
        try:
            tracy_utils_py: Any = importlib.import_module("libs.tracy_utils.tracy_utils_disabled_py")

            TRACY_AVAILABLE = tracy_utils_py.is_available()
            PlotType = tracy_utils_py.PlotType if TRACY_AVAILABLE else None
        except ImportError:
            # No Tracy variant available
            tracy_utils_py = None
            TRACY_AVAILABLE = False
            PlotType = None
