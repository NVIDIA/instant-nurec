# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dataclasses import dataclass
from pathlib import Path

from internal.sqa.test_cases.resource import Resource


@dataclass
class ArtifactsConfig:
    local_path: Path  # Base path receiving artifacts download/copy


@dataclass
class Artifacts(Resource):
    """Pre-computed training and validation outputs."""

    remote_include_patterns: list[str] | None = (
        None  # Optional patterns to filter which files or subdirectories to include when copying artifacts from remote storage
    )

    def get_runfiles_path(self) -> str | None:
        """Get the USDZ runfiles path from bazel_target."""
        if self.bazel_target is None:
            return None
        return self.bazel_target.get("usdz_path")

    def get_actual_path_from_runfiles(self, resolved_file: Path) -> Path:
        """Get the actual directory path from a resolved runfiles file path.

        For artifacts, the archive root is two levels up from the USDZ file.
        (e.g., artifacts/last.usdz -> artifacts/ -> root/)
        """
        return resolved_file.parent.parent

    def check_exists(self) -> bool:
        """Check if the artifacts archive exists locally with all required files."""
        if not self.local_path.exists():
            print(f"[{self.resource_type}] WARNING: Path {self.local_path} not found")
            return False

        # Check for the USDZ artifact file
        usdz_file = self.local_path / "artifacts" / "last.usdz"
        if not usdz_file.exists():
            print(f"[{self.resource_type}] WARNING: USDZ file {usdz_file} not found")
            return False

        return True

    def get_bazel_target_display(self) -> list[tuple[str, str]]:
        """Get USDZ path for display."""
        if self.bazel_target and "usdz_path" in self.bazel_target:
            return [("USDZ path", self.bazel_target["usdz_path"])]
        return []


def sqa_test_artifacts(config: ArtifactsConfig) -> dict[str, Artifacts]:
    """
    Define all available artifact archives for SQA tests, which can be used
    instead of running train_val on-the-fly.
    """
    local_path = config.local_path
    # common pattern to include only validation RGB predictions and the USDZ file for previous artifacts for full forward rendering tests
    include_patterns_pred_and_last_usdz = ["val/pred_rgb/**", "artifacts/last.usdz"]

    # Define artifact archives
    test_data_ncore_sqa_default_25_06_artifacts_local_path = "test_data_ncore_sqa_default_25.06_artifacts"
    h81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25_07_artifacts_local_path = (
        "H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.07_artifacts"
    )
    h81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25_08_artifacts_local_path = (
        "H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.08_artifacts"
    )
    h81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25_09_artifacts_local_path = (
        "H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.09_artifacts"
    )
    h81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25_11_artifacts_local_path = (
        "H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.11_artifacts"
    )
    h81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25_12_artifacts_local_path = (
        "H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.12_artifacts"
    )
    h81_Panda128_6b0e750d_1cam_1lidar_sqa_default_26_01_artifacts_local_path = (
        "H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_26.01_artifacts"
    )

    test_data_ncore_sqa_default_25_06_artifacts_bazel_target = {
        "target": "@test_data_ncore_sqa_default_25.06_artifacts//:all",
        "usdz_path": "test_data_ncore_sqa_default_25.06_artifacts/artifacts/last.usdz",
    }

    # TODO: Unify archived artifact names and locations
    h81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25_07_artifacts_remote_path = (
        "pdx-team-ncore:scratch-adrajeev/25_07/L20/RC5/sqa_default_6b/fP3DV4HAYAcQ9yP65Rcnsb"
    )
    h81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25_08_artifacts_remote_path = (
        "pdx-team-ncore:scratch-adrajeev/25_08/L40S/RC5/train_val_sqa_default_6b/9b68oNgdZeGdeAHRMWXqwP"
    )
    h81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25_09_artifacts_remote_path = (
        "pdx-team-ncore:scratch-adrajeev/25_09/A100/RC10/TrainVal_GRPC/train_val_sqa_default_6b/86nfd6RbvJMFNiQr26tEX4"
    )
    h81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25_11_artifacts_remote_path = (
        "pdx-team-ncore:scratch-adrajeev/25_11/L40S/RC5/TrainVal_GRPC/train_val_sqa_default_6b/7Cfeq2NzXnDokBHAVYwjFi"
    )
    h81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25_12_artifacts_remote_path = (
        "pdx-team-ncore:scratch-adrajeev/25_12/A40/RC2/TrainVal_GRPC/train_val_sqa_default_6b/75gJzJ6YHNHP9PA3mBqEnr"
    )
    h81_Panda128_6b0e750d_1cam_1lidar_sqa_default_26_01_artifacts_remote_path = (
        "pdx-team-ncore:scratch-adrajeev/26_01/L40S/RC7/TrainVal_GRPC/train_val_sqa_default_6b/RZY4L2b4SK9xnjSYENX4Z9"
    )

    return {
        "test_data_ncore_sqa_default_25.06_artifacts": Artifacts(
            name="test_data_ncore_sqa_default_25.06_artifacts",
            local_path=local_path / test_data_ncore_sqa_default_25_06_artifacts_local_path,
            remote_path=None,
            remote_include_patterns=None,
            bazel_target=test_data_ncore_sqa_default_25_06_artifacts_bazel_target,
        ),
        "H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.07_artifacts": Artifacts(
            name="H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.07_artifacts",
            local_path=local_path / h81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25_07_artifacts_local_path,
            remote_path=h81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25_07_artifacts_remote_path,
            remote_include_patterns=include_patterns_pred_and_last_usdz,
            bazel_target=None,
        ),
        "H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.08_artifacts": Artifacts(
            name="H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.08_artifacts",
            local_path=local_path / h81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25_08_artifacts_local_path,
            remote_path=h81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25_08_artifacts_remote_path,
            remote_include_patterns=include_patterns_pred_and_last_usdz,
            bazel_target=None,
        ),
        "H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.09_artifacts": Artifacts(
            name="H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.09_artifacts",
            local_path=local_path / h81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25_09_artifacts_local_path,
            remote_path=h81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25_09_artifacts_remote_path,
            remote_include_patterns=include_patterns_pred_and_last_usdz,
            bazel_target=None,
        ),
        "H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.11_artifacts": Artifacts(
            name="H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.11_artifacts",
            local_path=local_path / h81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25_11_artifacts_local_path,
            remote_path=h81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25_11_artifacts_remote_path,
            remote_include_patterns=include_patterns_pred_and_last_usdz,
            bazel_target=None,
        ),
        "H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.12_artifacts": Artifacts(
            name="H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25.12_artifacts",
            local_path=local_path / h81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25_12_artifacts_local_path,
            remote_path=h81_Panda128_6b0e750d_1cam_1lidar_sqa_default_25_12_artifacts_remote_path,
            remote_include_patterns=include_patterns_pred_and_last_usdz,
            bazel_target=None,
        ),
        "H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_26.01_artifacts": Artifacts(
            name="H81_Panda128_6b0e750d_1cam_1lidar_sqa_default_26.01_artifacts",
            local_path=local_path / h81_Panda128_6b0e750d_1cam_1lidar_sqa_default_26_01_artifacts_local_path,
            remote_path=h81_Panda128_6b0e750d_1cam_1lidar_sqa_default_26_01_artifacts_remote_path,
            remote_include_patterns=include_patterns_pred_and_last_usdz,
            bazel_target=None,
        ),
    }
