# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Custom Starlark lint checks for the NRE repository.

This module provides custom lint checks for Bazel Starlark files (.bazel, .bzl).
Add new checks by creating a LintCheck subclass and registering it in LINT_CHECKS.
"""

import os
import re
import subprocess
import sys

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LintError:
    """A lint error found in a file."""

    filepath: Path
    message: str
    suggestion: str | None = None


class LintCheck(ABC):
    """Base class for custom lint checks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name for the check."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this check does."""

    @abstractmethod
    def check(self, filepath: Path, content: str) -> LintError | None:
        """Check a file and return a LintError if there's a violation, None otherwise."""


class PyBinaryMissingMainCheck(LintCheck):
    """Check that py_binary and py_venv_binary targets have an explicit 'main' attribute.

    Matches py_binary, py_venv_binary, and aliased variants.

    Without explicit main, aspect_rules_py falls back to legacy name-based resolution
    which triggers some debug output in case there are multiple source files.
    py_venv_binary mandates this attribute.
    Reference: https://github.com/aspect-build/rules_py/pull/702
    """

    @property
    def name(self) -> str:
        return "py_binary_missing_main"

    @property
    def description(self) -> str:
        return "Ensure py_binary and py_venv_binary targets have explicit 'main' attribute"

    def check(self, filepath: Path, content: str) -> LintError | None:
        # Only check BUILD.bazel files
        if not str(filepath).endswith("BUILD.bazel"):
            return None

        # Find py_binary/py_venv_binary blocks without main attribute
        # Pattern: py_binary* or py_venv_binary* followed by content until closing )
        pattern = r"(py_(?:venv_)?binary\w*)\(\s*\n((?:.*\n)*?)\s*\)"
        matches = re.findall(pattern, content)

        missing_main_targets = []
        for rule_name, body in matches:
            has_main = "main =" in body or "main=" in body
            if not has_main:
                name_match = re.search(r'name\s*=\s*"([^"]+)"', body)
                if name_match:
                    missing_main_targets.append(f"{name_match.group(1)} ({rule_name})")

        if missing_main_targets:
            targets = ", ".join(missing_main_targets)
            return LintError(
                filepath=filepath,
                message=f"py_binary target(s) missing 'main' attribute: {targets}",
                suggestion="Add 'main = \"<name>.py\"' to each py_binary/py_venv_binary target",
            )

        return None


class ForbiddenLoadCheck(LintCheck):
    """Check that forbids loading specific symbols from a Bazel label.

    Files can opt-out by adding a comment with the exception marker.
    """

    def __init__(
        self,
        name: str,
        description: str,
        load_label: str,
        forbidden_symbols: list[str],
        exception_comment: str,
        suggested_label: str | None = None,
    ):
        self._name = name
        self._description = description
        self._load_label = load_label
        self._forbidden_symbols = forbidden_symbols
        self._exception_comment = exception_comment
        self._suggested_label = suggested_label

        # Build regex pattern: load("<label>".*"<symbol>"
        symbols_pattern = "|".join(re.escape(s) for s in forbidden_symbols)
        escaped_label = re.escape(load_label)
        self._pattern = re.compile(rf'load\("{escaped_label}".*"({symbols_pattern})"')

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def check(self, filepath: Path, content: str) -> LintError | None:
        if not self._pattern.search(content):
            return None

        if self._exception_comment in content:
            return None

        symbols = ", ".join(self._forbidden_symbols)
        suggestion = None
        if self._suggested_label:
            suggestion = f"Use {self._suggested_label} instead, or add comment: # {self._exception_comment}"

        return LintError(
            filepath=filepath,
            message=f"Forbidden load of [{symbols}] from {self._load_label}",
            suggestion=suggestion,
        )


# =============================================================================
# Register lint checks here
# =============================================================================

LINT_CHECKS: list[LintCheck] = [
    PyBinaryMissingMainCheck(),
    ForbiddenLoadCheck(
        name="py_binary_from_aspect",
        description="Ensure py_binary is loaded from @aspect_rules_py, not @rules_python",
        load_label="@rules_python//python:defs.bzl",
        forbidden_symbols=["py_binary"],
        exception_comment="nre:allow-rules-python-py-binary",
        suggested_label="@aspect_rules_py//py:defs.bzl",
    ),
    ForbiddenLoadCheck(
        name="py_test_from_aspect",
        description="Ensure py_test is loaded from @aspect_rules_py, not @rules_python",
        load_label="@rules_python//python:defs.bzl",
        forbidden_symbols=["py_test"],
        exception_comment="nre:allow-rules-python-py-test",
        suggested_label="@aspect_rules_py//py:defs.bzl",
    ),
    ForbiddenLoadCheck(
        name="pytest_test_rather_than_py_test",
        description="Prefer pytest_test over py_test for centralized mypy integration",
        load_label="@aspect_rules_py//py:defs.bzl",
        forbidden_symbols=["py_test"],
        exception_comment="nre:allow-direct-py-test",
        suggested_label="//bazel/pytest:defs.bzl (pytest_test)",
    ),
    ForbiddenLoadCheck(
        name="py_venv_binary_and_test_forbidden",
        description="Forbid py_venv_binary and py_venv_test except where explicitly allowed (build cost)",
        load_label="@aspect_rules_py//py/unstable:defs.bzl",
        forbidden_symbols=["py_venv_binary", "py_venv_test"],
        exception_comment="nre:allow-py-venv-binary",
        suggested_label="@aspect_rules_py//py:defs.bzl (py_binary / py_test)",
    ),
]


# =============================================================================
# Main runner
# =============================================================================


def get_starlark_files(workspace_dir: Path) -> list[Path]:
    """Get list of Starlark files using git ls-files (respects .gitignore).

    Includes both tracked files and untracked files that aren't gitignored.
    """
    try:
        # Get tracked files
        tracked = subprocess.run(
            ["git", "ls-files", "*.bazel", "*.bzl"],
            cwd=workspace_dir,
            capture_output=True,
            text=True,
            check=True,
        )

        # Get untracked files that aren't ignored (new files not yet committed)
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "*.bazel", "*.bzl"],
            cwd=workspace_dir,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("Failed to get list of Bazel files through Git")

    files = set()
    for output in [tracked.stdout, untracked.stdout]:
        for line in output.strip().split("\n"):
            if line:
                files.add(Path(line))

    return sorted(files)


def run_checks(workspace_dir: Path) -> list[LintError]:
    """Run all lint checks on Bazel files in the workspace."""
    errors: list[LintError] = []

    for relative_path in get_starlark_files(workspace_dir):
        filepath = workspace_dir / relative_path
        content = filepath.read_text()

        for check in LINT_CHECKS:
            error = check.check(relative_path, content)
            if error:
                errors.append(error)

    return errors


def print_errors(errors: list[LintError]) -> None:
    """Print lint errors to stderr."""
    for error in errors:
        print(f"ERROR: {error.filepath}", file=sys.stderr)
        print(f"       {error.message}", file=sys.stderr)
        if error.suggestion:
            print(f"       {error.suggestion}", file=sys.stderr)
        print(file=sys.stderr)


def main() -> int:
    workspace_dir = Path(os.environ.get("BUILD_WORKSPACE_DIRECTORY", ".")).resolve()
    os.chdir(workspace_dir)

    errors = run_checks(workspace_dir)

    if errors:
        print_errors(errors)
        print(f"Found {len(errors)} lint error(s)", file=sys.stderr)
        return 1

    print("All custom Bazel lint checks passed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
