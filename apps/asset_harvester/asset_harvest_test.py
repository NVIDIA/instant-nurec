# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os

from pathlib import Path

import pytest

from click.testing import CliRunner
from python.runfiles import runfiles

from apps.asset_harvester.asset_harvest import asset_harvest


RUNFILES = runfiles.Create()


@pytest.fixture
def test_dataset_path() -> Path:
    path = Path(
        RUNFILES.Rlocation("test_data_asset_harvester/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8.zarr.itar"),
    )
    if not path.exists():
        raise AssertionError(
            f"Test dataset not found. This is an issue with your filesystem/test suite, not the code under test. Missing {path=}"
        )

    return path


def test_asset_harvest(test_dataset_path: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Integration test to validate execution of asset_harvest cli on small test data set"""
    # Use environment variable for cache dir if available, otherwise create temp directory
    if cache_dir_env := os.environ.get("ASSET_HARVESTER_CACHE_DIR"):
        cache_dir = Path(cache_dir_env)
    else:
        cache_dir = tmp_path_factory.mktemp("asset_harvester_cache")

    # Use environment variable for output dir if available, otherwise create temp directory
    output_dir_env = os.environ.get("ASSET_HARVESTER_OUTPUT_DIR")

    # 3dgs test
    if output_dir_env:
        output_root = Path(output_dir_env) / "3dgs"
        output_root.mkdir(parents=True, exist_ok=True)
    else:
        output_root = tmp_path_factory.mktemp("asset_harvester_3dgs_test")
    result = CliRunner().invoke(
        asset_harvest,
        [
            f"--component-store={test_dataset_path}",
            f"--output-dir={output_root}",
            f"--cache-dir={cache_dir}",
            f"--track-ids=39,44",
            f'ncore_parser.camera_ids=["camera_front_wide_120fov","camera_cross_right_120fov","camera_cross_left_120fov"]',
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
