# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Colored exception handling utilities."""

import sys
import traceback

from types import TracebackType
from typing import Optional, Type


_EXCEPTION_COUNT: int = 0


def enable_colored_exceptions() -> None:
    """Enable colored exception handling globally."""

    # Having a local function to avoid obfuscation missing the dependency
    def handle_exception(
        exc_type: Type[BaseException], exc_value: BaseException, exc_traceback: Optional[TracebackType]
    ) -> None:
        """Handle an uncaught exception with colored output."""
        global _EXCEPTION_COUNT

        red = "\033[91m"
        orange = "\033[38;2;255;165;0m"
        no_color = "\033[0m"

        # Format the exception
        lines = []
        for trace in traceback.format_exception(exc_type, exc_value, exc_traceback):
            lines.extend(trace.split("\n"))

        # Print the traceback lines normally, but color only the exception message line for main thread
        for line in lines:
            if len(line) == 0:
                continue

            first_word = line.split(":")[0]
            if first_word.endswith("Exception") or (first_word.endswith("Error") and first_word != "Error"):
                # Use red color for the first exception, yellow for the next ones
                color = red if _EXCEPTION_COUNT == 0 else orange  # orange for subsequent exceptions
                _EXCEPTION_COUNT += 1

                print(f"{color}{line.rstrip()}{no_color}", file=sys.stderr)

            else:
                print(line.rstrip(), file=sys.stderr)

    sys.excepthook = handle_exception


def disable_colored_exceptions() -> None:
    """Disable colored exception handling and reset to default."""
    sys.excepthook = sys.__excepthook__
