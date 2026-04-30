# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Base class for test resources (datasets, artifacts, etc.)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Resource(ABC):
    """Base class for test resources like datasets and artifacts."""

    name: str
    local_path: Path
    remote_path: str | None
    bazel_target: dict[str, str] | None = None

    @abstractmethod
    def get_runfiles_path(self) -> str | None:
        """Get the runfiles path from bazel_target for this resource type.

        Returns:
            The runfiles path string, or None if not available.
        """
        ...

    @abstractmethod
    def get_actual_path_from_runfiles(self, resolved_file: Path) -> Path:
        """Get the actual directory path from a resolved runfiles file path.

        Args:
            resolved_file: The resolved path to the marker file in runfiles.

        Returns:
            The actual resource directory path.
        """
        ...

    @abstractmethod
    def check_exists(self) -> bool:
        """Check if the resource exists locally with all required files.

        Returns:
            True if the resource is available and valid, False otherwise.
        """
        ...

    @property
    def resource_type(self) -> str:
        """Return the resource type name for logging."""
        return type(self).__name__.upper()

    def get_bazel_target_display(self) -> list[tuple[str, str]]:
        """Get resource-specific bazel_target fields for display.

        Returns:
            List of (label, value) tuples for printing.
        """
        return []
