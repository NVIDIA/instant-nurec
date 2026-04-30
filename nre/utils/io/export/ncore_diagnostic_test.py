# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nre.utils.io.export.ncore_diagnostic import export_ncore_diagnostic


RUNFILES = runfiles.Create()


@pytest.fixture
def dataset_path() -> Path:
    path = Path(
        RUNFILES.Rlocation(
            "test_data_ncore/cf5ff7f6-5c82-11ed-806f-00044bf655de_1667597307250262_1667597318349978_1667597307250262_1667597308250262.zarr.itar"
        ),
    )
    if not path.is_file():
        raise AssertionError(f"Test dataset not found at {path}.")
    return path


def test_ncore_diagnostic_export(dataset_path: Path, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        export_ncore_diagnostic,
        [
            "--shard-file-pattern",
            str(dataset_path),
            "--output-dir",
            str(tmp_path),
            "--camera-images",
            "--semantic-labelmaps",
            "--semantic-overlays",
            "--lidar-points",
            "--lidar-points-fused",
            "--meta",
            "--frame-naming",
            "index",
            "--format",
            "image+video",
            "--frame-step-camera",
            "20",
            "--frame-step-lidar",
            "8",
        ],
        catch_exceptions=False,
    )
    print(result.output)
    assert result.exit_code == 0

    expected_files = [
        tmp_path / "meta.yaml",
        tmp_path / "meta_overview.yaml",
        tmp_path / "camera_images" / "camera_front_wide_120fov" / "000000.jpg",
        tmp_path / "camera_images" / "camera_front_wide_120fov" / "000020.jpg",
        tmp_path / "camera_images" / "camera_front_wide_120fov_input.mp4",
        tmp_path / "camera_semantic_labelmaps" / "camera_front_wide_120fov" / "000000.png",
        tmp_path / "camera_semantic_labelmaps" / "camera_front_wide_120fov" / "000020.png",
        tmp_path / "camera_semantic_labelmaps" / "camera_front_wide_120fov_semantic_classes.png",
        tmp_path / "camera_semantic_labelmaps" / "camera_front_wide_120fov_semantic_labelmaps.mp4",
        tmp_path / "camera_semantic_overlays" / "camera_front_wide_120fov" / "000000.jpg",
        tmp_path / "camera_semantic_overlays" / "camera_front_wide_120fov" / "000020.jpg",
        tmp_path / "camera_semantic_overlays" / "camera_front_wide_120fov_semantic_classes.png",
        tmp_path / "camera_semantic_overlays" / "camera_front_wide_120fov_semantic_overlays.mp4",
        tmp_path / "lidar_point_clouds" / "lidar_gt_top_p128_v4p5" / "world_point_cloud_000000.ply",
        tmp_path / "lidar_point_clouds" / "lidar_gt_top_p128_v4p5" / "world_point_cloud_000008.ply",
        tmp_path / "lidar_point_clouds" / "lidar_gt_top_p128_v4p5" / "semantic_classes.png",
    ]

    for expected_file in expected_files:
        assert expected_file.is_file(), str(expected_file) + " not exported"
        print(f"Found {expected_file}")
