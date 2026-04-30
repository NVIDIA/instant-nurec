# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os.path

from io import BytesIO
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pytest

from click.testing import CliRunner
from PIL import Image
from python.runfiles import runfiles

from nre import __version__
from nre.grpc.protos.common_pb2 import AvailableScenesReturn, Empty, VersionId
from nre.grpc.protos.sensorsim_pb2 import (
    AvailableCamerasRequest,
    AvailableCamerasReturn,
    AvailableDynamicObjectsRequest,
    AvailableDynamicObjectsReturn,
    AvailableEgoMasksReturn,
    AvailableTrajectoriesRequest,
    AvailableTrajectoriesReturn,
    DynamicObject,
    ImageFormat,
    PosePair,
    RGBRenderRequest,
    RGBRenderReturn,
)
from nre.grpc.protos.sensorsim_pb2_grpc import SensorsimServiceStub
from nre.grpc.serve import (
    SensorSimService,
    add_SensorsimServiceServicer_to_server,
    se3_to_grpc_pose,
)
from nre.run.main import main


RUNFILES = runfiles.Create()


@pytest.fixture
def lidarfree_dataset_path() -> Path:
    path = Path(
        RUNFILES.Rlocation("test_data_ncore/clipgt-34d6855b-4913-4e13-8a37-e4f03d34911c.json"),
    )
    if not path.exists():
        raise AssertionError(
            f"Test dataset not found. This is an issue with your filesystem/test suite, not the code under test. Missing {path=}"
        )
    return path


@pytest.fixture(scope="session")
def output_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Common temporary output folder for tests that execute *both* 'train' (test_training) and 'val' (test_validation_camera) stages and need a common shared folder"""

    return tmp_path_factory.mktemp("nre_training_pytest_output")


@pytest.mark.dependency()
def test_training(lidarfree_dataset_path: Path, output_root: Path) -> None:
    """Test validating 'train' mode on a default config - output will be stored in 'output_root', to be consumed by subsequent test_validation tests"""

    print("Output root:", output_root)

    # Ensure the path is a *quoted* string for Hydra compatibility with bazel's `~`-separated paths
    lidarfree_dataset_path_str = '"{0}"'.format(str(lidarfree_dataset_path))

    result = CliRunner().invoke(
        main,
        [
            "--config-name=configs/apps/prod/Hyperion-8.1/car2sim_lidarfree.yaml",
            f"dataset.path={lidarfree_dataset_path_str}",
            "dataset.camera_ids=[camera_front_wide_120fov]",
            # Override for faster test execution (config defaults are much larger)
            "dataset.n_samples_per_epoch=2",
            "dataset.samplers.batch_sampler.camera_pixel_sampler.subsample=2",
            "model.layers.background.initialization.num_point_cloud_points=10000",
            "model.layers.background.initialization.num_near_points=1000",
            "model.layers.background.initialization.num_far_points=1000",
            "model.layers.dynamic_rigids.initialization.num_point_cloud_points_in_layer=5000",
            "model.layers.dynamic_deformables.initialization.num_point_cloud_points_in_layer=5000",
            "model.layers.road.initialization.voxel_size=2.0",
            # Disable traffic light layer (test data has no traffic lights)
            "dataset.generate_static_rigid_cuboid_tracks.enabled=false",
            "~model.layers.traffic_light",
            "model.layers.background.ignore_classes_from_layers=[]",
            # Enable artifacts for serving smoke tests
            "checkpoint.artifact.enabled=true",
            "checkpoint.artifact.sequence_tracks.enabled=true",
            "checkpoint.artifact.rig_trajectories.enabled=true",
            "mode=train",
            f"out_dir={output_root}",
            "logger=wandb",
            "logger.offline=true",
            "logger.run_id=out",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    # Assert core outputs exist
    out_dir = os.path.join(output_root, "out")
    assert os.path.isdir(out_dir)
    assert os.path.isfile(os.path.join(out_dir, "checkpoints", "last.ckpt"))
    assert os.path.isfile(os.path.join(out_dir, "config", "parsed.yaml"))


@pytest.mark.dependency(depends=["test_training"])
def test_validation_camera(output_root: Path) -> None:
    """Verifies 'val' mode executed on 'train' results of earlier 'test_training' test that outputted results to 'output_root'"""

    print("Output root:", output_root)

    config_file = output_root / "out" / "config" / "parsed.yaml"

    result = CliRunner().invoke(
        main,
        [
            f"--config-name={config_file}",
            "mode=val",
            "resume=last.ckpt",
            "dataset.val_camera=true",
            "dataset.val_lidar=false",
            "dataset.n_val_image_subsample=2",
            "dataset.val_camera_frame_step=10",
            "system.test.save_results=true",
            "system.test.save_inputs=true",
            "system.test.save_videos=true",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    # Verify a couple of expected outputs
    out_dir = os.path.join(output_root, "out", "val")
    assert os.path.isdir(out_dir)
    assert os.path.isfile(os.path.join(out_dir, "pred_rgb", "cam_00", "000000.png"))


@pytest.fixture(scope="module")
def grpc_add_to_server():
    return add_SensorsimServiceServicer_to_server


@pytest.fixture(scope="module")
def grpc_servicer(output_root: Path) -> SensorSimService:
    # there will possibly be more than just the artifact at output_root but
    # SensorSimService should figure out the relevant subdirectories
    return SensorSimService(
        server=None,
        artifacts_glob=f"{output_root}/**/last.usdz",
        ray_chunk_size=1024,
    )


@pytest.fixture(scope="module")
def grpc_stub_cls(grpc_channel):
    return SensorsimServiceStub


def _create_rgb_request(
    scene_id: str,
    grpc_stub: SensorsimServiceStub,
    start_timestamp_us: int,
    end_timestamp_us: int,
    image_format: Literal["PNG", "JPEG"],
    dynamic_objects: Optional[list[DynamicObject]] = None,
) -> RGBRenderRequest:
    # test available cameras request
    available_cameras_request = AvailableCamerasRequest(scene_id=scene_id)
    available_cameras_return: AvailableCamerasReturn = grpc_stub.get_available_cameras(available_cameras_request)

    id_pose_pair = PosePair(start_pose=se3_to_grpc_pose(np.eye(4)), end_pose=se3_to_grpc_pose(np.eye(4)))

    if dynamic_objects is None:
        dynamic_objects = []

    return RGBRenderRequest(
        scene_id=scene_id,
        resolution_h=80,
        resolution_w=128,
        camera_intrinsics=available_cameras_return.available_cameras[0].intrinsics,
        frame_start_us=start_timestamp_us,
        frame_end_us=end_timestamp_us,
        sensor_pose=id_pose_pair,
        image_format={
            "PNG": ImageFormat.PNG,
            "JPEG": ImageFormat.JPEG,
        }[image_format],
        dynamic_objects=dynamic_objects,
        image_quality=95,  # ignored in case of PNG
    )


@pytest.mark.dependency(depends=["test_training"])
def test_server_get_version(grpc_stub: SensorsimServiceStub) -> None:
    version: VersionId = grpc_stub.get_version(Empty())
    # NRE runtime version is only available in non sandboxed environments
    # (e.g., when `run`-ing this unit test, not in `test` sandboxes) - hence,
    # only validate version consistency if NRE version is available
    if __version__ is not None:
        assert version.version_id == __version__.semantic_string()


@pytest.mark.dependency(depends=["test_training"])
def test_server_get_available_cameras(grpc_stub: SensorsimServiceStub) -> None:
    available_scenes: AvailableScenesReturn = grpc_stub.get_available_scenes(Empty())
    (scene_id,) = available_scenes.scene_ids

    available_cameras_request = AvailableCamerasRequest(scene_id=scene_id)
    available_cameras_return: AvailableCamerasReturn = grpc_stub.get_available_cameras(available_cameras_request)
    assert len(available_cameras_return.available_cameras) > 0


@pytest.mark.dependency(depends=["test_training"])
def test_server_get_available_trajectories(grpc_stub: SensorsimServiceStub) -> None:
    available_scenes: AvailableScenesReturn = grpc_stub.get_available_scenes(Empty())
    (scene_id,) = available_scenes.scene_ids

    available_trajectories_request = AvailableTrajectoriesRequest(scene_id=scene_id)
    available_trajectories_return: AvailableTrajectoriesReturn = grpc_stub.get_available_trajectories(
        available_trajectories_request
    )
    assert len(available_trajectories_return.available_trajectories) > 0


@pytest.mark.dependency(depends=["test_training"])
def test_server_get_available_ego_masks(grpc_stub: SensorsimServiceStub) -> None:
    """
    Smoke test to ensure the RPC call is available.
    We don't current create a mock ego hood directory at test time.
    """
    _available_ego_masks_return: AvailableEgoMasksReturn = grpc_stub.get_available_ego_masks(Empty())


@pytest.mark.dependency(depends=["test_training"])
def test_server_get_dynamic_objects(grpc_stub: SensorsimServiceStub) -> None:
    """Test getting dynamic objects data with full track information."""
    available_scenes: AvailableScenesReturn = grpc_stub.get_available_scenes(Empty())
    (scene_id,) = available_scenes.scene_ids

    # Test the new API
    dynamic_objects_request = AvailableDynamicObjectsRequest(scene_id=scene_id)
    dynamic_objects_response: AvailableDynamicObjectsReturn = grpc_stub.get_dynamic_objects(dynamic_objects_request)

    # Validate dynamic objects structure (may or may not have dynamic objects)
    for dynamic_object in dynamic_objects_response.dynamic_objects:
        # Validate field types/values
        assert isinstance(dynamic_object.id, str), f"Expected id to be str, got {type(dynamic_object.id)}"
        assert len(dynamic_object.id) > 0, "Track ID should not be empty"

        # Validate track dimensions (AABB) if present
        if dynamic_object.object_size.size_x > 0:
            assert dynamic_object.object_size.size_y > 0
            assert dynamic_object.object_size.size_z > 0


@pytest.mark.dependency(depends=["test_training"])
def test_server_render_rgb(grpc_stub: SensorsimServiceStub) -> None:
    available_scenes: AvailableScenesReturn = grpc_stub.get_available_scenes(Empty())
    (scene_id,) = available_scenes.scene_ids

    available_trajectories_request = AvailableTrajectoriesRequest(scene_id=scene_id)
    available_trajectories_return: AvailableTrajectoriesReturn = grpc_stub.get_available_trajectories(
        available_trajectories_request
    )
    (trajectory,) = available_trajectories_return.available_trajectories
    timestamp_us = trajectory.trajectory.poses[0].timestamp_us

    request = _create_rgb_request(scene_id, grpc_stub, timestamp_us, timestamp_us + 1, image_format="PNG")
    response: RGBRenderReturn = grpc_stub.render_rgb(request)
    img = Image.open(BytesIO(response.image_bytes))
    assert img.height > 0
