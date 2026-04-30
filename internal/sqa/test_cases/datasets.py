# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dataclasses import dataclass, field
from pathlib import Path

from internal.sqa.test_cases.resource import Resource


@dataclass
class DatasetConfig:
    local_path: Path


@dataclass
class Sensors:
    camera_ids: list[str] = field(default_factory=list)
    lidar_id: list[str] = field(default_factory=list)


@dataclass
class Dataset(Resource):
    sensors: Sensors = field(default_factory=Sensors)

    def get_runfiles_path(self) -> str | None:
        """Get the JSON runfiles path from bazel_target."""
        if self.bazel_target is None:
            return None
        return self.bazel_target.get("json_path")

    def get_json_file(self) -> Path:
        """Get the .json file path for tests like nre_image_trainval."""
        dataset_name = self.local_path.name
        return self.local_path / f"{dataset_name}.json"

    def get_zarr_itar_file(self) -> Path:
        """Get the .zarr.itar file path for tests like grpc_api_test and nre_tools."""
        dataset_name = self.local_path.name
        return self.local_path / f"{dataset_name}.zarr.itar"

    def get_actual_path_from_runfiles(self, resolved_file: Path) -> Path:
        """Get the actual directory path from a resolved runfiles file path.

        For datasets, the dataset is the parent directory of the .json file.
        """
        return resolved_file.parent

    def check_exists(self) -> bool:
        """Check if the dataset exists locally and has the required files."""
        if not self.local_path.exists():
            print(f"[{self.resource_type}] WARNING: Path {self.local_path} not found")
            return False

        # Check for JSON file (required for nre_image_trainval)
        json_file = self.get_json_file()
        if not json_file.exists():
            print(f"[{self.resource_type}] WARNING: File {json_file} not found")
            return False

        return True

    def get_bazel_target_display(self) -> list[tuple[str, str]]:
        """Get JSON path for display."""
        if self.bazel_target and "json_path" in self.bazel_target:
            return [("JSON path", self.bazel_target["json_path"])]
        return []


sensor_sets = {
    "h81_1cam_1lidar": Sensors(
        camera_ids=["camera_front_wide_120fov"],
        lidar_id=["lidar_gt_top_p128"],
    ),
    "h81_4cam_1lidar": Sensors(
        camera_ids=[
            "camera_front_wide_120fov",
            "camera_cross_right_120fov",
            "camera_cross_left_120fov",
            "camera_front_tele_30fov",
        ],
        lidar_id=["lidar_gt_top_p128"],
    ),
    "h81_4cam_lidarfree": Sensors(
        camera_ids=[
            "camera_front_wide_120fov",
            "camera_cross_right_120fov",
            "camera_cross_left_120fov",
            "camera_front_tele_30fov",
        ],
        lidar_id=["lidar_virtual"],
    ),
    "waymo_3cam_1lidar": Sensors(
        camera_ids=[
            "camera_front_right_50fov",
            "camera_front_left_50fov",
            "camera_front_50fov",
        ],
        lidar_id=["lidar_top"],
    ),
    "drive_sim_1cam_1lidar": Sensors(
        camera_ids=["camera_front_wide_120fov"],
        lidar_id=["lidar_gt_top_p128_v4p5"],
    ),
}


def sqa_test_datasets(config: DatasetConfig) -> dict[str, Dataset]:
    local_path = config.local_path

    h81_panda128_6b0e750d_local_path = "H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8"
    h81_panda128_0d59b8c8_local_path = "H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840"
    h81_at128_6b0e750d_local_path = "H81/AT128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8"
    h81_at128_0d59b8c8_local_path = "H81/AT128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840"
    h81_6b0e750d_lidarfree_local_path = "H81/lidar-free/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8"
    h81_0d59b8c8_lidarfree_local_path = "H81/lidar-free/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840"
    waymo_11017034898130016754_local_path = "Waymo/11017034898130016754_697_830_717_830"
    test_data_ncore_local_path = "test_data_ncore/clipgt-9048443e-c482-4228-8326-5b3dff3be711"

    h81_panda128_6b0e750d_remote_path = (
        "pdx-team-ncore:/sqa/dataset/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8"
    )
    h81_panda128_0d59b8c8_remote_path = (
        "pdx-team-ncore:/sqa/dataset/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840"
    )
    h81_at128_6b0e750d_remote_path = "pdx-team-ncore:/sqa/dataset/H81/AT128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8"
    h81_at128_0d59b8c8_remote_path = "pdx-team-ncore:/sqa/dataset/H81/AT128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840"
    h81_6b0e750d_lidarfree_remote_path = (
        "pdx-team-ncore:/sqa/dataset/H81/lidar-free/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8"
    )
    h81_0d59b8c8_lidarfree_remote_path = (
        "pdx-team-ncore:/sqa/dataset/H81/lidar-free/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840"
    )
    h81_panda128_6b0e750d_v4_local_path = "v4/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8"
    h81_panda128_0d59b8c8_v4_local_path = "v4/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840"
    h81_panda128_6b0e750d_v4_remote_path = (
        "pdx-team-ncore:/sqa/dataset/v4/H81/Panda128/clipgt-6b0e750d-5c40-47e9-879d-6f840dc6d8b8"
    )
    h81_panda128_0d59b8c8_v4_remote_path = (
        "pdx-team-ncore:/sqa/dataset/v4/H81/Panda128/clipgt-0d59b8c8-3706-4e32-82ed-3f6e0cfb6840"
    )

    waymo_11017034898130016754_remote_path = "pdx-team-ncore:/sqa/dataset/Waymo/11017034898130016754_697_830_717_830"
    test_data_ncore_bazel_target = {
        "target": "@test_data_ncore//:ncore_clipgt_1sec",
        "json_path": "test_data_ncore/clipgt-9048443e-c482-4228-8326-5b3dff3be711.json",
    }

    return {
        "H81_Panda128_6b0e750d_1cam_1lidar": Dataset(
            name="H81_Panda128_6b0e750d_1cam_1lidar",
            local_path=local_path / h81_panda128_6b0e750d_local_path,
            remote_path=h81_panda128_6b0e750d_remote_path,
            bazel_target=None,
            sensors=sensor_sets["h81_1cam_1lidar"],
        ),
        "H81_Panda128_6b0e750d_4cam_1lidar": Dataset(
            name="H81_Panda128_6b0e750d_4cam_1lidar",
            local_path=local_path / h81_panda128_6b0e750d_local_path,
            remote_path=h81_panda128_6b0e750d_remote_path,
            bazel_target=None,
            sensors=sensor_sets["h81_4cam_1lidar"],
        ),
        "H81_6b0e750d_4cam_lidarfree": Dataset(
            name="H81_6b0e750d_4cam_lidarfree",
            local_path=local_path / h81_6b0e750d_lidarfree_local_path,
            remote_path=h81_6b0e750d_lidarfree_remote_path,
            bazel_target=None,
            sensors=sensor_sets["h81_4cam_lidarfree"],
        ),
        "H81_0d59b8c8_4cam_lidarfree": Dataset(
            name="H81_0d59b8c8_4cam_lidarfree",
            local_path=local_path / h81_0d59b8c8_lidarfree_local_path,
            remote_path=h81_0d59b8c8_lidarfree_remote_path,
            bazel_target=None,
            sensors=sensor_sets["h81_4cam_lidarfree"],
        ),
        "H81_Panda128_0d59b8c8_1cam_1lidar": Dataset(
            name="H81_Panda128_0d59b8c8_1cam_1lidar",
            local_path=local_path / h81_panda128_0d59b8c8_local_path,
            remote_path=h81_panda128_0d59b8c8_remote_path,
            bazel_target=None,
            sensors=sensor_sets["h81_1cam_1lidar"],
        ),
        "H81_Panda128_0d59b8c8_4cam_1lidar": Dataset(
            name="H81_Panda128_0d59b8c8_4cam_1lidar",
            local_path=local_path / h81_panda128_0d59b8c8_local_path,
            remote_path=h81_panda128_0d59b8c8_remote_path,
            bazel_target=None,
            sensors=sensor_sets["h81_4cam_1lidar"],
        ),
        "Waymo_11017034898130016754_697_830_717_830_3cam_1lidar": Dataset(
            name="Waymo_11017034898130016754_697_830_717_830_3cam_1lidar",
            local_path=local_path / waymo_11017034898130016754_local_path,
            remote_path=waymo_11017034898130016754_remote_path,
            bazel_target=None,
            sensors=sensor_sets["waymo_3cam_1lidar"],
        ),
        "test_data_ncore": Dataset(
            name="test_data_ncore",
            local_path=local_path / test_data_ncore_local_path,
            remote_path=None,
            bazel_target=test_data_ncore_bazel_target,
            sensors=sensor_sets["h81_1cam_1lidar"],
        ),
        "H81_AT128_6b0e750d_1cam_1lidar": Dataset(
            name="H81_AT128_6b0e750d_1cam_1lidar",
            local_path=local_path / h81_at128_6b0e750d_local_path,
            remote_path=h81_at128_6b0e750d_remote_path,
            bazel_target=None,
            sensors=sensor_sets["h81_1cam_1lidar"],
        ),
        "H81_AT128_0d59b8c8_1cam_1lidar": Dataset(
            name="H81_AT128_0d59b8c8_1cam_1lidar",
            local_path=local_path / h81_at128_0d59b8c8_local_path,
            remote_path=h81_at128_0d59b8c8_remote_path,
            bazel_target=None,
            sensors=sensor_sets["h81_1cam_1lidar"],
        ),
        "H81_Panda128_6b0e750d_4cam_1lidar_v4": Dataset(
            name="H81_Panda128_6b0e750d_4cam_1lidar_v4",
            local_path=local_path / h81_panda128_6b0e750d_v4_local_path,
            remote_path=h81_panda128_6b0e750d_v4_remote_path,
            bazel_target=None,
            sensors=sensor_sets["h81_4cam_1lidar"],
        ),
        "H81_Panda128_0d59b8c8_4cam_1lidar_v4": Dataset(
            name="H81_Panda128_0d59b8c8_4cam_1lidar_v4",
            local_path=local_path / h81_panda128_0d59b8c8_v4_local_path,
            remote_path=h81_panda128_0d59b8c8_v4_remote_path,
            bazel_target=None,
            sensors=sensor_sets["h81_4cam_1lidar"],
        ),
    }
