# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os.path
import zipfile

from io import BytesIO
from pathlib import Path
from typing import Any, Literal, Optional

import grpc
import numpy as np
import pytest
import torch

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
    BatchRGBRenderRequest,
    BatchRGBRenderRequestItem,
    BatchRGBRenderReturn,
    DynamicObject,
    ImageFormat,
    LidarDeviceType,
    LidarRenderRequest,
    LidarRenderReturn,
    LidarSpec,
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
from nre.run.main import main, parse_typed_config
from nre.systems import make as make_system
from nre.systems.gaussians import GaussiansSystem
from nre.utils.batch import DataBatch
from nre.utils.misc import unpack_optional


RUNFILES = runfiles.Create()


@pytest.fixture
def small_clipgt_dataset_path() -> Path:
    path = Path(
        RUNFILES.Rlocation("test_data_ncore/clipgt-9048443e-c482-4228-8326-5b3dff3be711.json"),
    )
    if not path.exists():
        raise AssertionError(
            f"Test dataset not found. This is an issue with your filesystem/test suite, not the code under test. Missing {path=}"
        )
    return path


@pytest.mark.parametrize(
    "config_name",
    ["apps/Alpasim/alpasim_3dgut_speed"],
)
def test_training_config(
    config_name: str, small_clipgt_dataset_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Test to validate 'train' mode on specified configurations"""

    print("Output root:", output_root := tmp_path_factory.mktemp(config_name.replace("/", "_")))

    # Ensure the path is a *quoted* string for Hydra compatibility with bazel's `~`-separated paths
    small_clipgt_dataset_path_str = '"{0}"'.format(str(small_clipgt_dataset_path))

    result = CliRunner().invoke(
        main,
        [
            f"--config-name={config_name}",
            f"dataset.path={small_clipgt_dataset_path_str}",
            "dataset.n_samples_per_epoch=2",  # Using a low number of iterations speeds the test up
            "dataset.camera_ids=[camera_front_wide_120fov]",
            "dataset.lidar_ids=[lidar_gt_top_p128]",
            "dataset.samplers.batch_sampler.camera_pixel_sampler.subsample=2",  # test data is 480x270
            "dataset.n_train_sample_lidar_rays=10",
            "checkpoint.artifact.mesh.generic.step_frame=5",
            "checkpoint.artifact.mesh.ground.step_frame=5",
            "mode=train",
            f"out_dir={output_root}",
            "logger=wandb",
            "logger.offline=true",
            "logger.run_id=out",  # fixes the output subdir, avoids generating a random hash
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0


@pytest.fixture(scope="session")
def output_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Common temporary output folder for tests that execute *both* 'train' (test_training) and 'val' (test_validation_camera / test_validation_lidar) stages and need a common shared folder"""

    return tmp_path_factory.mktemp("nre_training_pytest_output")


@pytest.mark.dependency()
def test_training(small_clipgt_dataset_path: Path, output_root: Path) -> None:
    """Test validating 'train' mode on a default config - output will be stored in 'output_root', to be consumed by subsequent test_validation tests"""

    print("Output root:", output_root)

    # Ensure the path is a *quoted* string for Hydra compatibility with bazel's `~`-separated paths
    small_clipgt_dataset_path_str = '"{0}"'.format(str(small_clipgt_dataset_path))

    result = CliRunner().invoke(
        main,
        [
            "--config-name=apps/AV/NV/3dgut_dynamic",
            f"dataset.path={small_clipgt_dataset_path_str}",
            "dataset.n_samples_per_epoch=2",  # Using a low number of iterations speeds the test up
            "dataset.n_train_sample_lidar_rays=10",
            "dataset.lidar_ids=[lidar_gt_top_p128]",
            "dataset.camera_ids=[camera_front_wide_120fov]",
            "dataset.samplers.batch_sampler.camera_pixel_sampler.subsample=2",
            "mode=train",
            f"out_dir={output_root}",
            "logger=wandb",
            "logger.offline=true",
            "logger.run_id=out",  # fixes the output subdir, avoids generating a random hash
            # save NRE artifact including sequence tracks and rig trajectories so we can use them in `test_serving` and `test_simple_serving`
            "checkpoint.artifact.enabled=true",
            "checkpoint.artifact.sequence_tracks.enabled=true",
            "checkpoint.artifact.rig_trajectories.enabled=true",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    # Test whether output directory and files have been created by the training process.
    out_dir = os.path.join(output_root, "out")
    assert os.path.isdir(out_dir)
    assert os.path.isfile(os.path.join(out_dir, "checkpoints", "last.ckpt"))
    assert os.path.isfile(os.path.join(out_dir, "config", "parsed.yaml"))

    assert os.path.isfile(artifact_file_path := os.path.join(out_dir, "artifacts", "last.usdz"))
    with zipfile.ZipFile(artifact_file_path, "r") as zip_file:
        zip_name_list = zip_file.namelist()
        assert "parsed_config.yaml" in zip_name_list
        assert "checkpoint.ckpt" in zip_name_list
        assert "sequence_tracks.json" in zip_name_list
        assert "rig_trajectories.json" in zip_name_list


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
            "dataset.n_val_image_subsample=2",  # >1 tests the subsampling feature and is also faster (less rays)
            # reduce runtime / number of rendered frames
            "dataset.val_camera_frame_step=10",
            "system.test.save_results=true",
            "system.test.save_inputs=true",
            "system.test.save_videos=true",
            "dataset.val_lidar=false",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    # Test whether the expected output directory and files have been created by the validation process.
    # Assuming the images are exported in order, any failure would prevent the process to produce the last image.
    # So only test the first and last image of each sequence of images.
    # Testing the first image gives info about whether any image have been exported.
    out_dir = os.path.join(output_root, "out", "val")
    assert os.path.isdir(out_dir)
    assert os.path.isfile(os.path.join(out_dir, "input_rgb", "cam_00", "000000.png"))
    assert os.path.isfile(os.path.join(out_dir, "input_rgb", "cam_00", "000001.png"))
    assert os.path.isfile(os.path.join(out_dir, "input_valid_mask", "cam_00", "000000.png"))
    assert os.path.isfile(os.path.join(out_dir, "input_valid_mask", "cam_00", "000001.png"))
    assert os.path.isfile(os.path.join(out_dir, "pred_distance", "cam_00", "000000.png"))
    assert os.path.isfile(os.path.join(out_dir, "pred_distance", "cam_00", "000001.png"))
    assert os.path.isfile(os.path.join(out_dir, "pred_rgb", "cam_00", "000000.png"))
    assert os.path.isfile(os.path.join(out_dir, "pred_rgb", "cam_00", "000001.png"))
    assert os.path.isfile(os.path.join(out_dir, "input_rgb-cam_00.mp4"))
    assert os.path.isfile(os.path.join(out_dir, "input_valid_mask-cam_00.mp4"))
    assert os.path.isfile(os.path.join(out_dir, "pred_distance-cam_00.mp4"))
    assert os.path.isfile(os.path.join(out_dir, "pred_rgb-cam_00.mp4"))


@pytest.mark.dependency(depends=["test_training"])
def test_validation_lidar(output_root: Path) -> None:
    """Verifies 'val' mode executed on 'train' results of earlier 'test_training' test that outputted results to 'output_root'"""

    print("Output root:", output_root)

    config_file = output_root / "out" / "config" / "parsed.yaml"

    result = CliRunner().invoke(
        main,
        [
            f"--config-name={config_file}",
            "mode=val",
            "resume=last.ckpt",
            # reduce runtime / number of rendered frames
            "dataset.val_camera=false",
            "system.test.save_results=true",
            "system.test.save_inputs=true",
            "system.test.save_videos=true",
            "dataset.val_lidar=true",
            "dataset.val_lidar_frame_step=5",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    # Test whether the expected output directory and files have been created by the validation process.
    # Assuming the images are exported in order, any failure would prevent the process to produce the last image.
    # So only test the first and last image of each sequence of images.
    # Testing the first image gives info about whether any image have been exported.
    out_dir = os.path.join(output_root, "out", "val")
    assert os.path.isdir(out_dir)
    assert os.path.isfile(os.path.join(out_dir, "pred_pc", "000000_gt.png"))
    assert os.path.isfile(os.path.join(out_dir, "pred_pc", "000000.png"))
    assert os.path.isfile(os.path.join(out_dir, "pred_pc", "000000output_gt.ply"))
    assert os.path.isfile(os.path.join(out_dir, "pred_pc", "000000output.ply"))


@pytest.fixture(scope="module")
def grpc_add_to_server():
    return add_SensorsimServiceServicer_to_server


@pytest.fixture(scope="module")
def grpc_servicer(output_root: str) -> SensorSimService:
    # there will possibly be more than just the artifact at output_root but
    # SensorSimService should figure out the relevant subdirectories
    return SensorSimService(
        server=None,
        artifacts_glob=f"{output_root}/**/last.usdz",
        ray_chunk_size=1024,
        enable_editing_actors=True,
    )


@pytest.fixture(scope="module")
def grpc_stub_cls(grpc_channel):
    return SensorsimServiceStub


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

    # Ensure we have dynamic objects to test
    assert dynamic_objects_response.dynamic_objects, "Expected dynamic objects in response but got none"

    # Validate dynamic objects structure
    for dynamic_object in dynamic_objects_response.dynamic_objects:
        # Validate field types/values
        assert isinstance(dynamic_object.id, str), f"Expected id to be str, got {type(dynamic_object.id)}"
        assert len(dynamic_object.id) > 0, "Track ID should not be empty"
        assert isinstance(dynamic_object.semantic_class, str), (
            f"Expected semantic_class to be str, got {type(dynamic_object.semantic_class)}"
        )

        # Validate trajectory structure
        if dynamic_object.trajectory.poses:
            assert len(dynamic_object.trajectory.poses) > 0, "Trajectory should contain at least one pose"
            # Validate first pose structure
            first_pose = dynamic_object.trajectory.poses[0]
            assert first_pose.timestamp_us >= 0, "Timestamp should be non-negative"

        # Validate track dimensions (AABB)
        assert dynamic_object.object_size.size_x > 0, (
            f"Expected positive size_x, got {dynamic_object.object_size.size_x}"
        )
        assert dynamic_object.object_size.size_y > 0, (
            f"Expected positive size_y, got {dynamic_object.object_size.size_y}"
        )
        assert dynamic_object.object_size.size_z > 0, (
            f"Expected positive size_z, got {dynamic_object.object_size.size_z}"
        )


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


@pytest.mark.dependency(depends=["test_server_get_available_trajectories"])
@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
@pytest.mark.parametrize("which_timestamp", ["first", "last"])
@pytest.mark.parametrize("with_dynamic_objects", [True, False])
def test_server_render_rgb(
    grpc_stub: SensorsimServiceStub,
    image_format: Literal["PNG", "JPEG"],
    which_timestamp: Literal["first", "last"],
    with_dynamic_objects: bool,
) -> None:
    """
    Tests the main codepath rendering the first/last available timestamp in PNG/JPEG formats.

    NOTE: We put this test here so we can depend on the outputs of `test_training`.
    Otherwise we'd have to either duplicate its runtime or bundle a pre-trained model
    with the repo, which will keep getting outdated.
    """
    available_scenes: AvailableScenesReturn = grpc_stub.get_available_scenes(Empty())
    (scene_id,) = available_scenes.scene_ids

    # get available trajectories to get the first timestamp
    available_trajectories_request = AvailableTrajectoriesRequest(scene_id=scene_id)
    available_trajectories_return: AvailableTrajectoriesReturn = grpc_stub.get_available_trajectories(
        available_trajectories_request
    )
    (trajectory,) = available_trajectories_return.available_trajectories  # only one trajectory in the test dataset

    dynamic_objects: list[DynamicObject] = []
    if with_dynamic_objects:
        dynamic_objects.append(
            DynamicObject(
                # TODO: \/ get this from the dataset, currently hardcoded
                # Note: track ID is cleaned by removing @suffix, so use just "15"
                track_id="15",
                pose_pair=PosePair(
                    start_pose=se3_to_grpc_pose(np.eye(4)),
                    end_pose=se3_to_grpc_pose(np.eye(4)),
                ),
            )
        )

    if which_timestamp == "first":
        timestamp_us = trajectory.trajectory.poses[0].timestamp_us
    elif which_timestamp == "last":
        timestamp_us = trajectory.trajectory.poses[-1].timestamp_us

    request = _create_rgb_request(scene_id, grpc_stub, timestamp_us, timestamp_us + 1, image_format, dynamic_objects)

    response: RGBRenderReturn = grpc_stub.render_rgb(request)
    image = Image.open(BytesIO(response.image_bytes))

    assert image.height == 80
    # note: width will likely be larger to preserve aspect ratio
    assert image.format == image_format


@pytest.mark.dependency(depends=["test_server_get_available_trajectories"])
@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
def test_server_batch_render_rgb(
    grpc_stub: SensorsimServiceStub,
    image_format: Literal["PNG", "JPEG"],
) -> None:
    """
    Tests batch rendering of multiple cameras in a single request.

    This test verifies:
    - Multiple cameras can be rendered in a single batch call
    - Each camera result is correctly matched by camera_name
    - All renders succeed and produce valid images
    """
    available_scenes: AvailableScenesReturn = grpc_stub.get_available_scenes(Empty())
    (scene_id,) = available_scenes.scene_ids

    # Get available trajectories to get timestamp
    available_trajectories_request = AvailableTrajectoriesRequest(scene_id=scene_id)
    available_trajectories_return: AvailableTrajectoriesReturn = grpc_stub.get_available_trajectories(
        available_trajectories_request
    )
    (trajectory,) = available_trajectories_return.available_trajectories
    timestamp_us = trajectory.trajectory.poses[0].timestamp_us

    # Create two render requests for the batch (simulating different cameras or frames)
    request1 = _create_rgb_request(scene_id, grpc_stub, timestamp_us, timestamp_us + 1, image_format)
    request2 = _create_rgb_request(scene_id, grpc_stub, timestamp_us, timestamp_us + 1, image_format)

    batch_request = BatchRGBRenderRequest(
        items=[
            BatchRGBRenderRequestItem(camera_name="camera_front", request=request1),
            BatchRGBRenderRequestItem(camera_name="camera_rear", request=request2),
        ]
    )

    response: BatchRGBRenderReturn = grpc_stub.batch_render_rgb(batch_request)

    # Verify we got results for both cameras
    assert len(response.items) == 2

    # Check each result
    camera_names = {item.camera_name for item in response.items}
    assert camera_names == {"camera_front", "camera_rear"}

    for item in response.items:
        assert item.success, f"Camera {item.camera_name} failed: {item.error_message}"
        image = Image.open(BytesIO(item.result.image_bytes))
        assert image.height == 80
        assert image.format == image_format


@pytest.mark.dependency(depends=["test_server_get_available_trajectories"])
def test_server_batch_render_rgb_mismatched_scene_ids(grpc_stub: SensorsimServiceStub) -> None:
    """
    Tests that batch_render_rgb rejects requests with mismatched scene_ids.

    All items in a batch must use the same scene_id.
    """
    available_scenes: AvailableScenesReturn = grpc_stub.get_available_scenes(Empty())
    (scene_id,) = available_scenes.scene_ids

    # Get timestamp
    available_trajectories_request = AvailableTrajectoriesRequest(scene_id=scene_id)
    available_trajectories_return: AvailableTrajectoriesReturn = grpc_stub.get_available_trajectories(
        available_trajectories_request
    )
    (trajectory,) = available_trajectories_return.available_trajectories
    timestamp_us = trajectory.trajectory.poses[0].timestamp_us

    # Create two requests - one with correct scene_id, one with fake scene_id
    request1 = _create_rgb_request(scene_id, grpc_stub, timestamp_us, timestamp_us + 1, "PNG")
    request2 = _create_rgb_request(scene_id, grpc_stub, timestamp_us, timestamp_us + 1, "PNG")
    # Manually override scene_id on the second request to simulate mismatch
    request2_modified = RGBRenderRequest()
    request2_modified.CopyFrom(request2)
    request2_modified.scene_id = "different_scene_id"

    batch_request = BatchRGBRenderRequest(
        items=[
            BatchRGBRenderRequestItem(camera_name="camera_front", request=request1),
            BatchRGBRenderRequestItem(camera_name="camera_rear", request=request2_modified),
        ]
    )

    with pytest.raises(grpc.RpcError) as exc_info:
        grpc_stub.batch_render_rgb(batch_request)

    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "same scene_id" in exc_info.value.details()


@pytest.mark.dependency(depends=["test_server_get_available_trajectories"])
def test_server_batch_render_rgb_empty_batch(grpc_stub: SensorsimServiceStub) -> None:
    """
    Tests that batch_render_rgb handles an empty batch gracefully.
    """
    batch_request = BatchRGBRenderRequest(items=[])

    response: BatchRGBRenderReturn = grpc_stub.batch_render_rgb(batch_request)

    # Empty batch should return empty results
    assert len(response.items) == 0


@pytest.mark.dependency(depends=["test_server_get_available_trajectories"])
@pytest.mark.parametrize("which_timestamp", ["first", "last"])
def test_server_render_rgb_oob_timestamp(
    grpc_stub: SensorsimServiceStub, which_timestamp: Literal["first", "last"]
) -> None:
    """
    Tests the case when the requested timestamp is out of bounds (before or after the first/last timestamp).
    """
    available_scenes: AvailableScenesReturn = grpc_stub.get_available_scenes(Empty())
    (scene_id,) = available_scenes.scene_ids

    # get available trajectories to get the first timestamp
    available_trajectories_request = AvailableTrajectoriesRequest(scene_id=scene_id)
    available_trajectories_return: AvailableTrajectoriesReturn = grpc_stub.get_available_trajectories(
        available_trajectories_request
    )
    (trajectory,) = available_trajectories_return.available_trajectories  # only one trajectory in the test dataset

    if which_timestamp == "first":
        timestamp_us = trajectory.trajectory.poses[0].timestamp_us - 1
    elif which_timestamp == "last":
        timestamp_us = trajectory.trajectory.poses[-1].timestamp_us + 1

    request = _create_rgb_request(scene_id, grpc_stub, timestamp_us, timestamp_us + 1, image_format="PNG")

    with pytest.raises(grpc.RpcError, match=r"Requested time range .* is outside of scene time range .*"):
        response: RGBRenderReturn = grpc_stub.render_rgb(request)


@pytest.mark.dependency(depends=["test_server_get_available_trajectories"])
def test_server_render_rgb_linear_cde(grpc_stub: SensorsimServiceStub) -> None:
    """
    Tests rendering with different linear_cde parameters.
    """
    available_scenes: AvailableScenesReturn = grpc_stub.get_available_scenes(Empty())
    (scene_id,) = available_scenes.scene_ids

    # get available trajectories to get the first timestamp
    available_trajectories_request = AvailableTrajectoriesRequest(scene_id=scene_id)
    available_trajectories_return: AvailableTrajectoriesReturn = grpc_stub.get_available_trajectories(
        available_trajectories_request
    )
    (trajectory,) = available_trajectories_return.available_trajectories  # only one trajectory in the test dataset

    available_cameras_request = AvailableCamerasRequest(scene_id=scene_id)
    available_cameras_return: AvailableCamerasReturn = grpc_stub.get_available_cameras(available_cameras_request)

    # test if changing the linear_cde parameters to non-default (1,0,0) works
    available_cameras_return.available_cameras[0].intrinsics.ftheta_param.linear_cde.linear_c = 0.95
    available_cameras_return.available_cameras[0].intrinsics.ftheta_param.linear_cde.linear_d = 0.1
    available_cameras_return.available_cameras[0].intrinsics.ftheta_param.linear_cde.linear_e = 0.05

    rgb_render_request = RGBRenderRequest(
        scene_id=scene_id,
        resolution_h=80,
        resolution_w=128,
        camera_intrinsics=available_cameras_return.available_cameras[0].intrinsics,
        frame_start_us=trajectory.trajectory.poses[0].timestamp_us,
        frame_end_us=trajectory.trajectory.poses[0].timestamp_us + 1,
        sensor_pose=PosePair(start_pose=se3_to_grpc_pose(np.eye(4)), end_pose=se3_to_grpc_pose(np.eye(4))),
        image_format=ImageFormat.PNG,
        image_quality=95,
        dynamic_objects=[],
    )

    response: RGBRenderReturn = grpc_stub.render_rgb(rgb_render_request)
    image = Image.open(BytesIO(response.image_bytes))

    assert image.height == 80
    assert image.format == "PNG"


@pytest.mark.dependency(depends=["test_server_get_available_trajectories"])
def test_server_render_rgb_equal_timestamps(grpc_stub: SensorsimServiceStub) -> None:
    """
    Tests the case when start and end timestamps are equal, which is invalid.
    Timestamps should form a half-closed interval (start, end) where end > start.
    To render at an instant t, one should use (t, t+1) instead.
    """
    available_scenes: AvailableScenesReturn = grpc_stub.get_available_scenes(Empty())
    (scene_id,) = available_scenes.scene_ids

    # Get available trajectories to get a valid timestamp
    available_trajectories_request = AvailableTrajectoriesRequest(scene_id=scene_id)
    available_trajectories_return: AvailableTrajectoriesReturn = grpc_stub.get_available_trajectories(
        available_trajectories_request
    )
    (trajectory,) = available_trajectories_return.available_trajectories  # only one trajectory in the test dataset

    # Use a valid timestamp but set start and end to the same value
    timestamp_us = trajectory.trajectory.poses[0].timestamp_us

    request = _create_rgb_request(scene_id, grpc_stub, timestamp_us, timestamp_us, image_format="PNG")

    # Should raise an error because start and end timestamps are the same
    with pytest.raises(grpc.RpcError, match=r"Render time range .* is empty."):
        response: RGBRenderReturn = grpc_stub.render_rgb(request)


@pytest.mark.dependency(depends=["test_training"])
def test_serving_invalid_object_id(grpc_stub: SensorsimServiceStub) -> None:
    """
    Specifying an invalid dynamic object id should raise an exception.
    """
    available_scenes: AvailableScenesReturn = grpc_stub.get_available_scenes(Empty())
    (scene_id,) = available_scenes.scene_ids

    # get available trajectories to get the first timestamp
    available_trajectories_request = AvailableTrajectoriesRequest(scene_id=scene_id)
    available_trajectories_return: AvailableTrajectoriesReturn = grpc_stub.get_available_trajectories(
        available_trajectories_request
    )
    (trajectory,) = available_trajectories_return.available_trajectories  # only one trajectory in the test dataset
    timestamp_us = trajectory.trajectory.poses[0].timestamp_us  # using the first timestamp

    id_pose_pair = PosePair(start_pose=se3_to_grpc_pose(np.eye(4)), end_pose=se3_to_grpc_pose(np.eye(4)))
    dynamic_object = DynamicObject(track_id="deadbeef", pose_pair=id_pose_pair)
    request = _create_rgb_request(
        scene_id, grpc_stub, timestamp_us, timestamp_us + 1, image_format="PNG", dynamic_objects=[dynamic_object]
    )

    with pytest.raises(
        grpc.RpcError
    ):  # TODO currently errors out because we don't train a dynamic reconstruction, should still fail once we change that
        response: RGBRenderReturn = grpc_stub.render_rgb(request)


@pytest.mark.skipif(torch.cuda.get_device_capability() < (7, 5), reason="To avoid OOM on 16GB V100 CI Machine")
@pytest.mark.dependency(depends=["test_training"])
def test_server_lidar_request(grpc_stub: SensorsimServiceStub) -> None:
    """
    We put this test here so we can depend on the outputs of `test_training`.
    Otherwise we'd have to either duplicate its runtime or bundle a pre-trained model
    with the repo, which will keep getting outdated.

    #TODO pass dynamic objects
    """
    available_scenes: AvailableScenesReturn = grpc_stub.get_available_scenes(Empty())
    (scene_id,) = available_scenes.scene_ids

    # get available trajectories to get the first timestamp
    available_trajectories_request = AvailableTrajectoriesRequest(scene_id=scene_id)
    available_trajectories_return: AvailableTrajectoriesReturn = grpc_stub.get_available_trajectories(
        available_trajectories_request
    )
    first_timestamp_us = available_trajectories_return.available_trajectories[0].trajectory.poses[0].timestamp_us

    id_pose_pair = PosePair(start_pose=se3_to_grpc_pose(np.eye(4)), end_pose=se3_to_grpc_pose(np.eye(4)))

    request = LidarRenderRequest(
        scene_id=scene_id,
        lidar_config=LidarSpec(lidar_type=LidarDeviceType.AT128),
        frame_start_us=first_timestamp_us,
        frame_end_us=first_timestamp_us,
        sensor_pose=id_pose_pair,
    )

    lidar_response: LidarRenderReturn = grpc_stub.render_lidar(request)

    # AT128 rays without raydrop
    pc_xyzs = np.frombuffer(lidar_response.point_xyzs_buffer, dtype=np.float32).reshape(-1, 3)
    assert pc_xyzs.shape[0] > 0


def test_serialize_deserialize(small_clipgt_dataset_path: Path) -> None:
    PER_PIXEL_ATOL = 0.095  # very high, sometimes individual pixels differ by more than 0.05
    MEDIAN_ATOL = 1e-3  # most pixels are reproduced closely

    config_name = "configs/apps/AV/NV/3dgut_dynamic"

    # Ensure the path is a *quoted* string for Hydra compatibility with bazel's `~`-separated paths
    small_dataset_path_str = '"{0}"'.format(str(small_clipgt_dataset_path))

    def _make_system(seed: int, mode: str = "train") -> GaussiansSystem:
        hydra_args = [
            f"dataset.path={small_dataset_path_str}",
            "dataset.lidar_ids=[lidar_gt_top_p128]",
            "dataset.samplers.batch_sampler.camera_pixel_sampler.subsample=2",
            f"seed={seed}",
            f"mode={mode}",
            "out_dir=/doesnt/matter",
        ]

        config = parse_typed_config(config_name, hydra_args)
        system = make_system(config.system.name, config, load_from_checkpoint=None)
        assert isinstance(system, GaussiansSystem)
        system.setup("train")  # setup is needed to initialize model parameters from datasource
        return system

    device = torch.device("cuda")

    system_a = _make_system(42).to(device)
    system_b = _make_system(2137).to(device)

    batch: DataBatch = next(iter(system_a.datamodule.train_dataloader())).to(device)

    forward_args: dict[str, Any] = {
        "batch": batch,
    }

    renders_a = system_a.forward(**forward_args)
    renders_b = system_b.forward(**forward_args)

    rgb_a = unpack_optional(unpack_optional(renders_a.rendered_cam).rgb)
    rgb_b = unpack_optional(unpack_optional(renders_b.rendered_cam).rgb)

    # Make sure that the models are atually different
    assert not torch.allclose(rgb_a, rgb_b, atol=PER_PIXEL_ATOL)

    checkpoint_dict: dict[str, Any] = {"state_dict": system_a.state_dict()}

    # Create a new system with the same seed as system_b but in test mode so that the parameters are not initialized
    # This is to make sure that the size of the parameters doesn't mismatch
    system_c = _make_system(2137, mode="test")

    system_c.load_state_dict(checkpoint_dict["state_dict"], strict=False, assign=True)
    system_c = system_c.to(device)

    # Fix: Store the full return value first
    renders_c = system_c.forward(**forward_args)
    rgb_a_from_loaded = unpack_optional(unpack_optional(renders_c.rendered_cam).rgb)  # Then extract the rgb tensor

    assert torch.allclose(rgb_a, rgb_a_from_loaded, atol=PER_PIXEL_ATOL)
    assert (rgb_a - rgb_a_from_loaded).abs().median().item() < MEDIAN_ATOL
