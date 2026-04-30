# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Pytest wrapper for sync_test_plan verify mode."""

from click.testing import CliRunner

from internal.sqa.test_cases.sync_test_plan import sync_test_plan


def test_sync_test_plan_verify() -> None:
    """Test that test plan files are in sync."""
    runner = CliRunner()
    result = runner.invoke(sync_test_plan, ["--verify"])
    assert result.exit_code == 0, f"sync_test_plan --verify failed:\n{result.output}"
