# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import os
import re
import subprocess

from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any, Optional

import yaml

from pydantic import BaseModel, Field, field_serializer, field_validator


SENTINEL = "<sentinel>"


class Version(BaseModel):
    """Maintains full version information of the current NRE execution environment"""

    # Semantic version component
    version_major: int
    version_minor: int
    version_patch: int

    # Code-related component
    git_commit_sha_short: str = Field(default="0000000")
    git_tree_dirty: bool = Field(default=False)
    git_commit_date: datetime = Field(default=datetime.fromtimestamp(0))

    version_string: str = Field(
        default=SENTINEL,
        description="Not to be set by the user. Set in `model_post_init`.",
    )

    @staticmethod
    def empty() -> Version:
        """Returns an empty Version object with all fields set to default values"""
        return Version(
            version_major=0,
            version_minor=0,
            version_patch=0,
            git_commit_sha_short="0000000",
            git_tree_dirty=False,
            git_commit_date=datetime.fromtimestamp(0),
        )

    def semantic_string(self) -> str:
        """Returns string-representation of semantic version part 'VERSION_MAJOR.VERSION_MINOR.VERSION_PATCH'"""
        return f"{self.version_major}.{self.version_minor}.{self.version_patch}"

    def __repr__(self) -> str:
        """Returns string-representation as 'VERSION_MAJOR.VERSION_MINOR.VERSION_PATCH-GIT_COMMIT_SHA_SHORT[+GIT_TREE_DIRTY]'"""
        return f"{self.version_major}.{self.version_minor}.{self.version_patch}-{self.git_commit_sha_short}" + (
            "+dirty" if self.git_tree_dirty else ""
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, self.__class__)
            and repr(self) == repr(other)
            and self.git_commit_date == other.git_commit_date
        )

    def model_post_init(self, __context) -> None:
        self.version_string = repr(self)

    # OmegaConfig doesn't like datetime objects so we treat it as string outside of Pydantic

    @field_validator("git_commit_date", mode="before")
    @classmethod
    def parse_datetime(cls, value: Any) -> datetime:
        match value:
            case datetime():
                return value
            case str():
                return datetime.fromisoformat(value)
            case _:
                raise TypeError(f"Unexpected type {type(value)=}.")

    @field_serializer("git_commit_date")
    def serialize_datetime(self, value: datetime) -> str:
        return value.isoformat()


# This is required for pip install -e . to work properly in the python path.
Version.model_rebuild()


@cache  # Cache the result of get_version as current version doesn't change at runtime (to avoid repeated file access)
def get_version(allow_empty: bool = False) -> Optional[Version]:
    """Parses the current version from available runtime information in priority order

    1. build-time version file (created by bazel for deployed executions only that
       don't have access to the source git repo anymore, i.e., built images) - note that
       the version file is not an unconditional dependency of this module to prevent
       unnecessary build/test cache invalidations
    2. directly from a git-repository (both bazel / python-env)

    Doesn't return a version if executed in certain environments (e.g., test sandboxes),
    where no version information is available intentionally to prevent
    test result cache invalidation (unless `allow_empty` is set to True, in
    which case a default initialized `Version` object is returned).

    Please see bazel/version/BUILD.bazel for more information on NRE's version determination system.
    """

    version_file_yaml = Path("/version_file.yaml")
    workspace_status_script = Path("bazel/version/workspace_status.sh")

    # Runtime version is intentionally *not* available in the sandbox building the sphinx-based
    # docs to not invalidate build caches unnecessarily (this information is currently not exposed
    # in the rendered html docs) - return a surrogate empty version object in that case
    # TODO(jme): Extend this if we ever want to tag the current version in the html docs,
    #            which will require exposing more source-repository data to the sandbox
    if "DOCUTILSCONFIG" in os.environ:
        return Version.empty()

    if version_file_yaml.is_file():
        # Parse version info from version_file.yaml, which is available in Docker containers

        with open(version_file_yaml, "r", encoding="utf-8") as f:
            version_yaml_content = yaml.safe_load(f)

        # Check presence of required keys and their types in the parsed data
        required_keys = {"VERSION_STRING": str, "GIT_COMMIT_DATE": datetime}
        for key, typ in required_keys.items():
            try:
                value = version_yaml_content[key]
            except KeyError as key_error:
                raise RuntimeError(f"Required key '{key}' is missing from {version_file_yaml}") from key_error
            if not isinstance(value, typ):
                raise RuntimeError(
                    f"{key} is of type {type(value).__name__} instead of {typ.__name__} in {version_file_yaml}"
                )

        version_string = version_yaml_content["VERSION_STRING"]
        git_commit_date = version_yaml_content["GIT_COMMIT_DATE"]

    elif workspace_status_script.is_file():
        # Get version info from the workspace status script available in the source repository (available from git-repositories)

        # Execute the script in workspace root directory (or sphinx runfiles sandbox)
        workspace_root = (
            Path(__file__).parent.parent.parent.resolve()
            if not "BUILD_WORKSPACE_DIRECTORY" in os.environ
            else Path(os.environ["BUILD_WORKSPACE_DIRECTORY"])
        )

        p = subprocess.Popen(workspace_status_script, stdout=subprocess.PIPE, cwd=workspace_root)
        cout, cerr = p.communicate()

        if p.returncode != 0:
            if allow_empty:
                return Version.empty()
            return None  # No version information available in this case, which is expected in some environments (e.g., tests, docs)

        # Parse output of workspace status
        # https://bazel.build/docs/user-manual#workspace-status
        workspace_status_lines = cout.decode("utf-8").splitlines()
        workspace_status_dict = {}
        for line in workspace_status_lines:
            key, value = line.split(maxsplit=1)
            workspace_status_dict[key] = value

        version_string = workspace_status_dict["STABLE_VERSION_STRING"]
        git_commit_date = datetime.fromisoformat(workspace_status_dict["STABLE_GIT_COMMIT_DATE"])

    else:
        raise RuntimeError(
            f"Version information not available. Missing {version_file_yaml} or {workspace_status_script} failed to execute in source tree"
        )

    if not (
        m := re.match(
            r"(?P<version_major>\d+)\.(?P<version_minor>\d+)\.(?P<version_patch>\d+)-(?P<git_commit_sha_short>[\dabcdef]+)(?P<git_tree_dirty>\+dirty)?",
            version_string,
        )
    ):
        raise RuntimeError(f"Version string {version_string} cannot be parsed (does not match regex)")

    return Version(
        version_major=int(m.group("version_major")),
        version_minor=int(m.group("version_minor")),
        version_patch=int(m.group("version_patch")),
        git_commit_sha_short=m.group("git_commit_sha_short"),
        git_tree_dirty=m.group("git_tree_dirty") is not None,
        git_commit_date=git_commit_date,
    )
