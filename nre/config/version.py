# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

from __future__ import annotations

from datetime import datetime
from functools import cache
from typing import Any, Optional

from pydantic import BaseModel, Field, field_serializer, field_validator


class Version(BaseModel):
    """Stub version object for the predict-only standalone."""

    version_major: int = 0
    version_minor: int = 0
    version_patch: int = 0
    git_commit_sha_short: str = Field(default="0000000")
    git_tree_dirty: bool = Field(default=False)
    git_commit_date: datetime = Field(default=datetime.fromtimestamp(0))

    @staticmethod
    def empty() -> Version:
        return Version()

    def semantic_string(self) -> str:
        return f"{self.version_major}.{self.version_minor}.{self.version_patch}"

    def __repr__(self) -> str:
        return f"{self.semantic_string()}-{self.git_commit_sha_short}" + ("+dirty" if self.git_tree_dirty else "")

    @field_validator("git_commit_date", mode="before")
    @classmethod
    def _parse_datetime(cls, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(value)

    @field_serializer("git_commit_date")
    def _serialize_datetime(self, value: datetime) -> str:
        return value.isoformat()


@cache
def get_version(allow_empty: bool = False) -> Optional[Version]:
    return Version.empty()
