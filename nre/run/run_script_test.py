# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Test that example scripts can be run via run_script."""

from pathlib import Path

from click.testing import CliRunner
from python.runfiles import runfiles

from nre.run.run_script import run_script


def test_examples_script() -> None:
    """Test that all example scripts run successfully."""
    # Discover all Python example scripts in the examples directory
    examples_path = Path(runfiles.Create().Rlocation("_main/docs/architecture/examples"))
    if not examples_path.exists():
        raise AssertionError(
            f"Examples directory not found. This is an issue with your filesystem/test suite. Missing {examples_path=}"
        )

    example_scripts = sorted(examples_path.glob("*.py"))
    assert len(example_scripts) > 0, "No example scripts found in docs/architecture/examples"

    for script_path in example_scripts:
        print(f"\n{'=' * 60}")
        print(f"Testing example script: {script_path.name}")
        print("=" * 60)

        result = CliRunner().invoke(
            run_script,
            [str(script_path)],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, (
            f"Script {script_path.name} failed with exit code {result.exit_code}\nOutput: {result.output}"
        )
