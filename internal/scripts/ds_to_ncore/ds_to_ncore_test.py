# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os.path

from pathlib import Path

import pytest

from click.testing import CliRunner
from python.runfiles import runfiles

from internal.scripts.ds_to_ncore.ds_to_ncore import ds_to_ncore


RUNFILES = runfiles.Create()


@pytest.fixture
def small_dataset_path() -> Path:
    path = Path(RUNFILES.Rlocation("test_data_ds2ncore/ego_dynamics.json")).parent
    if not path.exists():
        raise AssertionError(
            f"Test dataset not found. This is an issue with your filesystem/test suite, not the code under test. Missing {path=}"
        )
    return path


def test_training_config(small_dataset_path: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Test to validate ds2ncore runs without crashing"""

    print("Output root:", output_root := tmp_path_factory.mktemp(str(small_dataset_path).replace("/", "_")))

    result = CliRunner().invoke(
        ds_to_ncore,
        [
            f"--input-dir={small_dataset_path}",
            "--run-id=ds2ncore_output",
            f"--output-dir={output_root}",
            "--camera-ids=camera_front_wide_120fov",
            "--lidar-ids=lidar_gt_top_p128_v4p5",
            "--n-shards=1",
        ],
        catch_exceptions=False,
    )
    assert os.path.exists(os.path.join(output_root, "ds2ncore_output_0-1.zarr.itar"))
    assert os.path.exists(os.path.join(output_root, "ds2ncore_output_0-1.aux.sseg.zarr.itar"))
    assert os.path.exists(os.path.join(output_root, "ds2ncore_output_0-1.aux.nrml.zarr.itar"))
    assert os.path.exists(os.path.join(output_root, "ds2ncore_output_0-1.aux.depth.zarr.itar"))
    assert result.exit_code == 0
