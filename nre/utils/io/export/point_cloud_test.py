# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from pathlib import Path

import pytest

from click.testing import CliRunner
from python.runfiles import runfiles

from nre.utils.io.export.point_cloud import export_point_cloud


RUNFILES = runfiles.Create()


@pytest.fixture
def dataset_path_str() -> str:
    path = Path(
        RUNFILES.Rlocation(
            "test_data_ncore/cf5ff7f6-5c82-11ed-806f-00044bf655de_1667597307250262_1667597318349978_1667597307250262_1667597308250262.json"
        ),
    )
    if not path.is_file():
        raise AssertionError(f"Test dataset not found at {path}.")

    # Ensure the path is a *quoted* string for Hydra compatibility with bazel's `~`-separated paths
    return '"{0}"'.format(str(path))


def test_point_cloud_rgb_colors_fused(dataset_path_str: str, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        export_point_cloud,
        [
            "--config-name=configs/tests/ncore_ds",
            f"dataset.path={dataset_path_str}",
            "--output-dir",
            str(tmp_path),
            "--colorizer",
            "rgb",
        ],
        catch_exceptions=False,
    )
    print(result.output)
    assert result.exit_code == 0
    file_path = tmp_path / "colored_point_cloud.ply"
    assert file_path.is_file(), str(file_path) + " not found"


def test_point_cloud_semantic_colors_split_per_frame_per_class(dataset_path_str: str, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        export_point_cloud,
        [
            "--config-name=configs/tests/ncore_ds",
            f"dataset.path={dataset_path_str}",
            "--output-dir",
            str(tmp_path),
            "--colorizer",
            "semantic",
            "--per-frame",
            "--per-class",
            "--frame-step",
            "8",
        ],
        catch_exceptions=False,
    )
    print(result.output)

    assert result.exit_code == 0

    for frame_idx in [0, 8]:
        for class_name in [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "car",
        ]:
            file_path = tmp_path / f"semantic_point_cloud_{frame_idx:04d}_{class_name}.ply"
            assert file_path.is_file(), str(file_path) + " not found"


def test_point_cloud_road_colors_split_per_class(dataset_path_str: str, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        export_point_cloud,
        [
            "--config-name=configs/tests/ncore_ds",
            f"dataset.path={dataset_path_str}",
            "--output-dir",
            str(tmp_path),
            "--colorizer",
            "road",
            "--per-class",
        ],
        catch_exceptions=False,
    )
    print(result.output)
    assert result.exit_code == 0
    for class_name in ["road", "nonroad"]:
        file_path = tmp_path / f"segmented_point_cloud_{class_name}.ply"
        assert file_path.is_file(), str(file_path) + " not found"
