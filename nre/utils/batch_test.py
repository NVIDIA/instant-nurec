# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import time

from collections import OrderedDict
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
import torch

from python.runfiles import runfiles

import ncore.impl.common.transformations as ncore_tf

from ncore.data import (
    BBox3,
    FThetaCameraModelParameters,
    ShutterType,
)
from nre.config.parse import parse_typed_config
from nre.datasets import make as make_dataset
from nre.datasets.ncore import (
    NCORESequentialDataset,
    NCORETrainDataset,
)
from nre.utils.batch import (
    CameraFreePoseViewGeometry,
    DataAndRenderingBatch,
    DataBatch,
    LidarFreePoseViewGeometry,
    RectSubsampled,
    RenderingBatch,
)
from nre.utils.misc import unpack_optional
from nre.utils.types import FrameConversion, NovelViewOverrides, RigTrajectories


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


@pytest.fixture
def fake_rig_trajectories() -> RigTrajectories:
    T1 = torch.eye(4, device="cpu", dtype=torch.float64)
    T2 = T1 * 2.0
    T4 = T1 * 4.0
    T5 = T1 * 5.0

    sequence_id = "mysequence"
    rig_trajectory = RigTrajectories.RigTrajectory(
        sequence_id=sequence_id,
        rig_bbox=BBox3(centroid=(-1.0, -2.0, -0.5), dim=(1.1, 2.2, 3.3), rot=(0.2, 0.4, 0.7)),
        cameras_linear_start_frame_indices=None,
        lidars_linear_start_frame_indices=None,
        cameras_frame_timestamps_us={
            "front_camera": torch.tensor([[10, 11], [20, 21], [30, 31]], device="cpu", dtype=torch.int64),
            "back_camera": torch.tensor([[40, 41], [50, 51]], device="cpu", dtype=torch.int64),
        },
        lidars_frame_timestamps_us={
            "lidar_top": torch.tensor([[15, 15], [25, 25], [35, 35]], device="cpu", dtype=torch.int64),
        },
        T_rig_worlds=T4.repeat(3, 1, 1),
        T_rig_world_timestamps_us=torch.tensor([5, 25, 100], device="cpu", dtype=torch.int64),
        # Note: cameras_frame_T_rig_worlds are supposed to be interpolated from the T_rig_worlds
        # based on the NCore specs but they are not.
        cameras_frame_T_rig_worlds={
            "front_camera": T1.repeat(3, 2, 1, 1),
            "back_camera": T2.repeat(3, 2, 1, 1),
        },
    )

    ftheta_params = FThetaCameraModelParameters(
        resolution=np.array([1920, 1080], dtype=np.uint64),
        shutter_type=ShutterType.GLOBAL,
        principal_point=np.array([960.0, 540.0], dtype=np.float32),
        reference_poly=FThetaCameraModelParameters.PolynomialType.ANGLE_TO_PIXELDIST,
        pixeldist_to_angle_poly=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        angle_to_pixeldist_poly=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        max_angle=1.0,
        linear_cde=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )

    lidar_calibrations = OrderedDict(
        {
            "lidar_top": RigTrajectories.LidarCalibration(
                sequence_id=sequence_id,
                logical_sensor_name="lidar_top",
                unique_sensor_idx=0,
                T_sensor_rig=T4,
                lidar_model_parameters=None,
            ),
        }
    )

    camera_calibrations = OrderedDict(
        {
            "front_camera": RigTrajectories.CameraCalibration(
                sequence_id=sequence_id,
                logical_sensor_name="front_camera",
                unique_sensor_idx=123,
                T_sensor_rig=T1,
                camera_model_parameters=ftheta_params,
            ),
            "back_camera": RigTrajectories.CameraCalibration(
                sequence_id=sequence_id,
                logical_sensor_name="back_camera",
                unique_sensor_idx=456,
                T_sensor_rig=T2,
                camera_model_parameters=ftheta_params,
            ),
        }
    )

    return RigTrajectories(
        T_world_base=T1,
        world_to_nre=FrameConversion(matrix=np.eye(4, dtype=np.float32)),
        rig_trajectories=[rig_trajectory],
        camera_calibrations=camera_calibrations,
        lidar_calibrations=lidar_calibrations,
    )


@pytest.fixture
def small_clipgt_dataset(small_clipgt_dataset_path: Path) -> tuple[NCORETrainDataset, NCORESequentialDataset]:
    config_name = "apps/Alpasim/alpasim_3dgut_speed"

    # Ensure the path is a *quoted* string for Hydra compatibility with bazel's `~`-separated paths
    small_clipgt_dataset_path_str = '"{0}"'.format(str(small_clipgt_dataset_path))

    hydra_args = [
        f"dataset.path={small_clipgt_dataset_path_str}",
        "dataset.name=ncore",
        "dataset.n_samples_per_epoch=2",  # Using a low number of iterations speeds the test up
        "dataset.camera_ids=[camera_front_wide_120fov]",
        "dataset.lidar_ids=[lidar_gt_top_p128]",
        "dataset.seek_offset_sec=0.1",  # non-trivial offset
        "dataset.samplers.batch_sampler.camera_pixel_sampler.subsample=2",  # test data is 480x270
        "dataset.show_progress_bars=False",
        "dataset.val_sensor_transl_delta_m=[0.01, 0.05, 0.1]",
        "dataset.val_sensor_rot_delta_deg=[2.0, 4.0, 6.0]",
        "dataset.val_lidar=True",
        "dataset.n_val_image_subsample=2",
        "out_dir=/doesnt/matter",  # required field
    ]

    config = parse_typed_config(config_name, hydra_args)
    train_dataset = cast(NCORETrainDataset, make_dataset(config.dataset.name, config, split="train"))
    val_dataset = cast(NCORESequentialDataset, make_dataset(config.dataset.name, config, split="val"))
    return train_dataset, val_dataset


@pytest.mark.parametrize("enable_calib", [True, False])
def test_cam_geom_from_rig_trajectories(fake_rig_trajectories: RigTrajectories, enable_calib: bool) -> None:
    cam_calib = CameraFreePoseViewGeometry.from_rig_trajectories(fake_rig_trajectories, enable_calib=enable_calib)

    assert cam_calib.T_sensor_world_startend_allviews.shape == (5, 2, 4, 4)
    assert cam_calib.timestamps_startend_us_allviews.shape == (5, 2)
    assert len(cam_calib.sensor_models) == 2
    assert str(123) in cam_calib.sensor_models
    assert str(456) in cam_calib.sensor_models
    assert cam_calib.enable_calib == enable_calib
    assert cam_calib.sensor_ids_to_frame_range == {
        "front_camera": range(0, 3),
        "back_camera": range(3, 5),
    }
    assert (
        cam_calib.timestamps_startend_us_allviews
        == torch.tensor([[10, 11], [20, 21], [30, 31], [40, 41], [50, 51]], device="cpu", dtype=torch.int64)
    ).all()


@pytest.mark.parametrize("enable_calib", [True, False])
def test_cam_geom_get_frame_ranges_per_sensor(fake_rig_trajectories: RigTrajectories, enable_calib: bool) -> None:
    cam_calib = CameraFreePoseViewGeometry.from_rig_trajectories(fake_rig_trajectories, enable_calib=enable_calib)
    assert cam_calib.get_frame_ranges_per_sensor() == {
        "front_camera": range(0, 3),
        "back_camera": range(3, 5),
    }


@pytest.mark.parametrize("enable_calib", [True, False])
def test_cam_geom_get_timestamps_all_views(fake_rig_trajectories: RigTrajectories, enable_calib: bool) -> None:
    cam_calib = CameraFreePoseViewGeometry.from_rig_trajectories(fake_rig_trajectories, enable_calib=enable_calib)
    timestamps = cam_calib.get_timestamps()
    assert timestamps.shape == (5, 2)
    assert (
        timestamps == torch.tensor([[10, 11], [20, 21], [30, 31], [40, 41], [50, 51]], device="cpu", dtype=torch.int64)
    ).all()


@pytest.mark.parametrize("enable_calib", [True, False])
def test_cam_geom_get_timestamps_single_view(fake_rig_trajectories: RigTrajectories, enable_calib: bool) -> None:
    cam_calib = CameraFreePoseViewGeometry.from_rig_trajectories(fake_rig_trajectories, enable_calib=enable_calib)
    timestamps = cam_calib.get_timestamps(3)
    assert (timestamps == torch.tensor([40, 41], device="cpu", dtype=torch.int64)).all()


@pytest.mark.parametrize("enable_calib", [True, False])
@pytest.mark.parametrize("skip_calib", [True, False])
def test_cam_geom_get_poses_all_views(
    fake_rig_trajectories: RigTrajectories, enable_calib: bool, skip_calib: bool
) -> None:
    cam_calib = CameraFreePoseViewGeometry.from_rig_trajectories(fake_rig_trajectories, enable_calib=enable_calib)
    poses = cam_calib.get_poses(None, skip_calib=skip_calib)
    assert poses.shape == (5, 2, 4, 4)
    assert poses.dtype == torch.float32
    assert poses.device == torch.device("cpu")
    assert (poses - cam_calib.T_sensor_world_startend_allviews).abs().max() < 1e-4


@pytest.mark.parametrize("enable_calib", [True, False])
@pytest.mark.parametrize("skip_calib", [True, False])
def test_cam_geom_get_poses_single_view(
    fake_rig_trajectories: RigTrajectories, enable_calib: bool, skip_calib: bool
) -> None:
    cam_calib = CameraFreePoseViewGeometry.from_rig_trajectories(fake_rig_trajectories, enable_calib=enable_calib)
    poses = cam_calib.get_poses(3, skip_calib=skip_calib)
    assert poses.shape == (2, 4, 4)
    assert poses.dtype == torch.float32
    assert poses.device == torch.device("cpu")
    assert (poses - cam_calib.T_sensor_world_startend_allviews[3]).abs().max() < 1e-4


def test_camera_data_batch(small_clipgt_dataset: tuple[NCORETrainDataset, NCORESequentialDataset]) -> None:
    train_dataset, val_dataset = small_clipgt_dataset
    full_frame_width = 480
    full_frame_height = 270

    camera_id = "camera_front_wide_120fov"
    camera_frame_idx = 1
    sampled_pixels = RectSubsampled(
        width=240, height=135, subsample_factor=2.0, original_width=full_frame_width, original_height=full_frame_height
    )
    # Create a DataBatch from the dataset.
    data_batch: DataBatch.Camera = train_dataset.get_camera_data_batch(
        sensor_id=camera_id,
        sensor_frame_idx=camera_frame_idx,
        sampled_pixels=sampled_pixels,
    )
    assert data_batch.h == sampled_pixels.height
    assert data_batch.w == sampled_pixels.width
    assert data_batch.meta[0].unique_sensor_idx == 0

    subsample = unpack_optional(data_batch.meta[0].subsample)
    rgb = unpack_optional(data_batch.labels.rgb)
    flags = unpack_optional(data_batch.labels.flags)
    assert rgb.shape == (
        1,
        round(full_frame_height / subsample.subsample_factor),
        round(full_frame_width / subsample.subsample_factor),
        3,
    )

    # Test the sequnetial validation dataset
    val_batch = val_dataset.get_item_novel_view_overrides((0, None))
    assert isinstance(val_batch, DataAndRenderingBatch)
    assert val_batch.data.lidar is None, "This should be a camera batch"
    assert unpack_optional(unpack_optional(val_batch.data.camera).meta[0].unique_sensor_idx) == 0

    rgb = unpack_optional(unpack_optional(val_batch.data.camera).labels.rgb)
    flags = unpack_optional(unpack_optional(val_batch.data.camera).labels.flags)


def test_lidar_data_batch(small_clipgt_dataset: tuple[NCORETrainDataset, NCORESequentialDataset]) -> None:
    train_dataset, val_dataset = small_clipgt_dataset
    full_frame_width = 3600
    full_frame_height = 128

    lidar_id = "lidar_gt_top_p128"
    lidar_frame_idx = 2

    # Create a DataBatch from the dataset.
    data_batch: DataBatch.Lidar = train_dataset.get_lidar_data_batch(
        sensor_id=lidar_id,
        sensor_frame_idx=lidar_frame_idx,
    )
    assert data_batch.h == full_frame_height
    assert data_batch.w == full_frame_width
    assert data_batch.meta[0].unique_sensor_idx == 0

    distance = unpack_optional(data_batch.labels.distance)
    flags = unpack_optional(data_batch.labels.flags)
    intensity = unpack_optional(data_batch.labels.intensity)
    raydrop = unpack_optional(data_batch.labels.raydrop)
    assert distance.shape == (1, full_frame_height, full_frame_width, 1)
    assert flags.shape == (1, full_frame_height, full_frame_width, 1)
    assert intensity.shape == (1, full_frame_height, full_frame_width, 1)
    assert raydrop.shape == (1, full_frame_height, full_frame_width, 1)

    # Test sequential validation dataset
    n_camera_frames = sum(len(val_dataset.camera_frames[camera_id]) for camera_id in val_dataset.camera_ids)

    val_batch = val_dataset.get_item_novel_view_overrides((n_camera_frames, None))
    assert isinstance(val_batch, DataAndRenderingBatch)
    assert val_batch.data.lidar is not None, "This should be a lidar batch"
    assert val_batch.data.camera is None, "This should be a lidar batch"

    assert val_batch.data.lidar.h == full_frame_height
    assert val_batch.data.lidar.w == full_frame_width
    assert val_batch.data.lidar.meta[0].unique_sensor_idx == 0

    distance = unpack_optional(val_batch.data.lidar.labels.distance)
    flags = unpack_optional(val_batch.data.lidar.labels.flags)
    intensity = unpack_optional(val_batch.data.lidar.labels.intensity)
    raydrop = unpack_optional(val_batch.data.lidar.labels.raydrop)
    assert distance.shape == (1, full_frame_height, full_frame_width, 1)
    assert flags.shape == (1, full_frame_height, full_frame_width, 1)
    assert intensity.shape == (1, full_frame_height, full_frame_width, 1)
    assert raydrop.shape == (1, full_frame_height, full_frame_width, 1)


def test_rendering_batch(small_clipgt_dataset: tuple[NCORETrainDataset, NCORESequentialDataset]) -> None:
    train_dataset, val_dataset = small_clipgt_dataset
    camera_id = "camera_front_wide_120fov"
    camera_frame_idx = 1
    sampled_pixels = RectSubsampled(
        width=240, height=135, subsample_factor=2.0, original_width=480, original_height=270
    )

    # Create a DataBatch from the dataset.
    camera_data_batch: DataBatch.Camera = train_dataset.get_camera_data_batch(
        sensor_id=camera_id,
        sensor_frame_idx=camera_frame_idx,
        sampled_pixels=sampled_pixels,
    )

    # Convert the DataBatch to a RenderingBatch.
    rig_trajectories = train_dataset.datasource.get_rig_trajectories()
    camera_rig_module = CameraFreePoseViewGeometry.from_rig_trajectories(rig_trajectories).cuda()
    camera_rendering_data = camera_rig_module.to_rendering_data(camera_data_batch.to("cuda"))

    # Test sequential validation dataset
    # Simulate offsets passed in the getitem function
    val_sensor_transl_delta_m = np.array([0.1, 0.2, 3.0], dtype=np.float32)
    val_sensor_rot_delta_deg = np.array([2.0, 4.0, 5.0], dtype=np.float32)
    novel_view_overrides = NovelViewOverrides(
        transl_delta_m=val_sensor_transl_delta_m,
        rot_delta_deg=val_sensor_rot_delta_deg,
    )

    val_batch = val_dataset.get_item_novel_view_overrides((0, None))
    assert isinstance(val_batch, DataAndRenderingBatch)
    assert val_batch.data.camera is not None, "This should be a camera batch"
    assert val_batch.data.lidar is None, "This should be a camera batch"

    val_batch = val_dataset.get_item_novel_view_overrides((0, novel_view_overrides))
    assert isinstance(val_batch, DataAndRenderingBatch)
    assert val_batch.data.camera is not None, "This should be a camera batch"
    assert val_batch.data.lidar is None, "This should be a camera batch"

    # Novel-view extrinsic offsets: T_offset_nre_startend is assigned after FrameMeta construction,
    # so dtype must be asserted here (float64 from numpy pose math would break matmul in sensors).
    val_camera_with_offset = unpack_optional(val_batch.data.camera)
    t_off_cam = val_camera_with_offset.meta[0].T_offset_nre_startend
    assert t_off_cam is not None
    assert t_off_cam.dtype == torch.float32
    assert t_off_cam.shape == (2, 4, 4)
    val_rig = val_dataset.datasource.get_rig_trajectories()
    camera_rig_val = CameraFreePoseViewGeometry.from_rig_trajectories(val_rig).cuda()
    _ = camera_rig_val.to_rendering_data(val_camera_with_offset.to("cuda"))

    # Test lidar dataset
    n_camera_frames = sum(len(val_dataset.camera_frames[camera_id]) for camera_id in val_dataset.camera_ids)

    val_batch = val_dataset.get_item_novel_view_overrides((n_camera_frames, None))
    assert isinstance(val_batch, DataAndRenderingBatch)
    assert val_batch.data.lidar is not None, "This should be a lidar batch"
    assert val_batch.data.camera is None, "This should be a lidar batch"

    val_batch = val_dataset.get_item_novel_view_overrides((n_camera_frames, novel_view_overrides))
    assert isinstance(val_batch, DataAndRenderingBatch)
    assert val_batch.data.lidar is not None, "This should be a lidar batch"
    assert val_batch.data.camera is None, "This should be a lidar batch"

    val_lidar_with_offset = unpack_optional(val_batch.data.lidar)
    t_off_lidar = val_lidar_with_offset.meta[0].T_offset_nre_startend
    assert t_off_lidar is not None
    assert t_off_lidar.dtype == torch.float32
    assert t_off_lidar.shape == (2, 4, 4)
    lidar_rig_val = LidarFreePoseViewGeometry.from_rig_trajectories(val_rig).cuda()
    _ = lidar_rig_val.to_rendering_data(val_lidar_with_offset.to("cuda"))


def test_datarendering(small_clipgt_dataset: tuple[NCORETrainDataset, NCORESequentialDataset]) -> None:
    train_dataset, val_dataset = small_clipgt_dataset
    sequence_id = "clipgt-9048443e-c482-4228-8326-5b3dff3be711"
    camera_id = "camera_front_wide_120fov"
    camera_frame_idx = 1
    sampled_pixels = RectSubsampled(
        width=240, height=135, subsample_factor=2.0, original_width=480, original_height=270
    )
    lidar_id = "lidar_gt_top_p128"
    lidar_frame_idx = 2

    # Create a DataBatch from the dataset.
    data_camera_batch: DataBatch.Camera = train_dataset.get_camera_data_batch(
        sensor_id=camera_id,
        sensor_frame_idx=camera_frame_idx,
        sampled_pixels=sampled_pixels,
    )
    data_lidar_batch: DataBatch.Lidar = train_dataset.get_lidar_data_batch(
        sensor_id=lidar_id,
        sensor_frame_idx=lidar_frame_idx,
    )
    data_batch = DataBatch(
        idx=0,
        sequence_id=[sequence_id],
        worker_id=[0],
        camera=data_camera_batch,
        lidar=data_lidar_batch,
    )

    # Convert the DataBatch to RenderingBatch
    rig_trajectories = train_dataset.datasource.get_rig_trajectories()
    camera_rig_module = CameraFreePoseViewGeometry.from_rig_trajectories(rig_trajectories).cuda()
    lidar_rig_module = LidarFreePoseViewGeometry.from_rig_trajectories(rig_trajectories).cuda()
    rendering_camera_data = camera_rig_module.to_rendering_data(data_camera_batch.to("cuda"))
    rendering_lidar_data = lidar_rig_module.to_rendering_data(data_lidar_batch.to("cuda"))
    rendering_batch = RenderingBatch(camera=rendering_camera_data, lidar=rendering_lidar_data)

    # convert the trainval batch to batch
    datarendering_batch = DataAndRenderingBatch(data=data_batch, rendering=rendering_batch)


def test_getitem(small_clipgt_dataset: tuple[NCORETrainDataset, NCORESequentialDataset]) -> None:
    train_dataset, _ = small_clipgt_dataset
    batch: DataAndRenderingBatch = train_dataset[0]
    assert hasattr(batch, "pin_memory") and callable(getattr(batch, "pin_memory"))


@pytest.mark.skip(reason="Used for benchmarking, not for testing")
@pytest.mark.parametrize("cache_sensor_params", [True, False], ids=["cache_sensor_params", "no_cache_sensor_params"])
def test_benchmark_camera_view_geometry(
    small_clipgt_dataset: tuple[NCORETrainDataset, NCORESequentialDataset], cache_sensor_params: bool
) -> None:
    train_dataset, val_dataset = small_clipgt_dataset
    camera_id = "camera_front_wide_120fov"
    camera_frame_idx = 1
    sampled_pixels = RectSubsampled(
        width=240, height=135, subsample_factor=2.0, original_width=480, original_height=270
    )

    # Create a DataBatch from the dataset.
    data_camera_batch: DataBatch.Camera = train_dataset.get_camera_data_batch(
        sensor_id=camera_id,
        sensor_frame_idx=camera_frame_idx,
        sampled_pixels=sampled_pixels,
    )

    # Convert the DataBatch to RenderingBatch
    rig_trajectories = train_dataset.datasource.get_rig_trajectories()
    camera_rig_module = CameraFreePoseViewGeometry.from_rig_trajectories(rig_trajectories)

    # warmup
    iters_warmup = 10
    for _ in range(iters_warmup):
        _ = camera_rig_module.to_rendering_data(data_camera_batch, cache_sensor_params=False)

    iters_profile = 100
    torch.cuda.synchronize()
    tic = time.perf_counter()
    for _ in range(iters_profile):
        rendering_camera_data = camera_rig_module.to_rendering_data(
            data_camera_batch, cache_sensor_params=cache_sensor_params
        )
    torch.cuda.synchronize()
    toc = time.perf_counter()
    print(f"Time taken: {toc - tic} seconds, {iters_profile / (toc - tic)} batches/s")


@pytest.mark.skip(reason="Used for benchmarking, not for testing")
@pytest.mark.parametrize("cache_sensor_params", [True, False], ids=["cache_sensor_params", "no_cache_sensor_params"])
def test_benchmark_lidar_view_geometry(
    small_clipgt_dataset: tuple[NCORETrainDataset, NCORESequentialDataset], cache_sensor_params: bool
) -> None:
    train_dataset, val_dataset = small_clipgt_dataset
    lidar_id = "lidar_gt_top_p128"
    lidar_frame_idx = 2

    # Create a DataBatch from the dataset.
    data_lidar_batch: DataBatch.Lidar = train_dataset.get_lidar_data_batch(
        sensor_id=lidar_id,
        sensor_frame_idx=lidar_frame_idx,
    )

    # Convert the DataBatch to RenderingBatch
    rig_trajectories = train_dataset.datasource.get_rig_trajectories()
    lidar_rig_module = LidarFreePoseViewGeometry.from_rig_trajectories(rig_trajectories)

    # warmup
    iters_warmup = 10
    for _ in range(iters_warmup):
        _ = lidar_rig_module.to_rendering_data(data_lidar_batch, cache_sensor_params=False)

    iters_profile = 100
    torch.cuda.synchronize()
    tic = time.perf_counter()
    for _ in range(iters_profile):
        rendering_lidar_data = lidar_rig_module.to_rendering_data(
            data_lidar_batch, cache_sensor_params=cache_sensor_params
        )
    torch.cuda.synchronize()
    toc = time.perf_counter()
    print(f"Time taken: {toc - tic} seconds, {iters_profile / (toc - tic)} batches/s")


def test_collate(small_clipgt_dataset: tuple[NCORETrainDataset, NCORESequentialDataset]) -> None:
    train_dataset, val_dataset = small_clipgt_dataset
    sequence_id = "clipgt-9048443e-c482-4228-8326-5b3dff3be711"
    camera_id = "camera_front_wide_120fov"
    camera_frame_idx = 1
    sampled_pixels = RectSubsampled(
        width=240, height=135, subsample_factor=2.0, original_width=480, original_height=270
    )
    lidar_id = "lidar_gt_top_p128"
    lidar_frame_idx = 2

    # Create a DataBatch from the dataset.
    data_camera_batch: DataBatch.Camera = train_dataset.get_camera_data_batch(
        sensor_id=camera_id,
        sensor_frame_idx=camera_frame_idx,
        sampled_pixels=sampled_pixels,
    )
    data_lidar_batch: DataBatch.Lidar = train_dataset.get_lidar_data_batch(
        sensor_id=lidar_id,
        sensor_frame_idx=lidar_frame_idx,
    )
    data_batch = DataBatch(
        idx=0,
        sequence_id=[sequence_id],
        worker_id=[0],
        camera=data_camera_batch,
        lidar=data_lidar_batch,
    )
    _ = DataBatch.collate_fn([data_batch, data_batch])

    # Convert the DataBatch to RenderingBatch
    rig_trajectories = train_dataset.datasource.get_rig_trajectories()
    camera_rig_module = CameraFreePoseViewGeometry.from_rig_trajectories(rig_trajectories).cuda()
    lidar_rig_module = LidarFreePoseViewGeometry.from_rig_trajectories(rig_trajectories).cuda()
    rendering_camera_data = camera_rig_module.to_rendering_data(data_camera_batch.to("cuda"))
    rendering_lidar_data = lidar_rig_module.to_rendering_data(data_lidar_batch.to("cuda"))
    rendering_batch = RenderingBatch(camera=rendering_camera_data, lidar=rendering_lidar_data)
    _ = RenderingBatch.collate_fn([rendering_batch, rendering_batch])

    # convert the trainval batch to batch
    datarendering_batch = DataAndRenderingBatch(data=data_batch, rendering=rendering_batch)
    datarendering_batch_collated = DataAndRenderingBatch.collate_fn([datarendering_batch, datarendering_batch])


def test_calibrate_camera_poses(small_clipgt_dataset: tuple[NCORETrainDataset, NCORESequentialDataset]) -> None:
    train_dataset, val_dataset = small_clipgt_dataset
    camera_id = "camera_front_wide_120fov"
    camera_frame_idx = 1
    sampled_pixels = RectSubsampled(
        width=240, height=135, subsample_factor=2.0, original_width=480, original_height=270
    )

    # Create a DataBatch from the dataset.
    data_camera_batch: DataBatch.Camera = train_dataset.get_camera_data_batch(
        sensor_id=camera_id,
        sensor_frame_idx=camera_frame_idx,
        sampled_pixels=sampled_pixels,
    )

    # Convert the DataBatch to RenderingBatch
    rig_trajectories = train_dataset.datasource.get_rig_trajectories()
    camera_rig_module = CameraFreePoseViewGeometry.from_rig_trajectories(rig_trajectories, enable_calib=True).cuda()
    assert camera_rig_module.embeds is not None
    data_camera_batch_cuda = data_camera_batch.to("cuda")
    rendering_camera_data = camera_rig_module.to_rendering_data(data_camera_batch_cuda)

    # Test the gradient of the calibrated poses
    grad = torch.autograd.grad(rendering_camera_data.poses_tquat_startend.sum(), camera_rig_module.embeds.weight)
    assert grad is not None

    # Recreate the rendering data to avoid issues with graph retention that is not compatible here with some torch.compile directives
    rendering_camera_data = camera_rig_module.to_rendering_data(data_camera_batch_cuda)
    grad = torch.autograd.grad(rendering_camera_data.rays.sum(), camera_rig_module.embeds.weight)
    assert grad is not None


def test_calibrate_lidar_poses(small_clipgt_dataset: tuple[NCORETrainDataset, NCORESequentialDataset]) -> None:
    train_dataset, val_dataset = small_clipgt_dataset
    lidar_id = "lidar_gt_top_p128"
    lidar_frame_idx = 2

    # Create a DataBatch from the dataset.
    data_lidar_batch: DataBatch.Lidar = train_dataset.get_lidar_data_batch(
        sensor_id=lidar_id,
        sensor_frame_idx=lidar_frame_idx,
    )

    # Convert the DataBatch to RenderingBatch
    rig_trajectories = train_dataset.datasource.get_rig_trajectories()
    lidar_rig_module = LidarFreePoseViewGeometry.from_rig_trajectories(rig_trajectories, enable_calib=True).cuda()
    assert lidar_rig_module.embeds is not None

    # Test the gradient of the calibrated poses
    data_lidar_batch_cuda = data_lidar_batch.to("cuda")
    rendering_lidar_data = lidar_rig_module.to_rendering_data(data_lidar_batch_cuda)
    grad = torch.autograd.grad(rendering_lidar_data.poses_tquat_startend.sum(), lidar_rig_module.embeds.weight)
    assert grad is not None

    # Recreate the rendering data to avoid issues with graph retention that is not compatible here with some torch.compile directives
    rendering_lidar_data = lidar_rig_module.to_rendering_data(data_lidar_batch_cuda)
    grad = torch.autograd.grad(rendering_lidar_data.rays.sum(), lidar_rig_module.embeds.weight)
    assert grad is not None


def test_real_lidar_rays(small_clipgt_dataset: tuple[NCORETrainDataset, NCORESequentialDataset]) -> None:
    train_dataset, val_dataset = small_clipgt_dataset
    train_dataset.use_real_lidar_rays = True

    lidar_id = "lidar_gt_top_p128"
    lidar_frame_idx = 2

    # Create a DataBatch from the dataset.
    data_lidar_batch: DataBatch.Lidar = train_dataset.get_lidar_data_batch(
        sensor_id=lidar_id,
        sensor_frame_idx=lidar_frame_idx,
    )
    assert data_lidar_batch.labels.sparse_rays is not None
    assert data_lidar_batch.labels.sparse_timestamps is not None
    assert data_lidar_batch.labels.sparse_elements is not None

    # Convert the DataBatch to RenderingBatch
    rig_trajectories = train_dataset.datasource.get_rig_trajectories()
    lidar_rig_module = LidarFreePoseViewGeometry.from_rig_trajectories(rig_trajectories, enable_calib=False).cuda()
    _ = lidar_rig_module.to_rendering_data(data_lidar_batch.to("cuda"))


def test_ncore_datasource_defaults_T_world_world_global_when_edge_absent(
    small_clipgt_dataset_path: Path,
) -> None:
    """NCore V4: missing world→world_global edge uses identity (no global CRS)."""
    path_str = '"{0}"'.format(str(small_clipgt_dataset_path))
    hydra_args = [
        f"dataset.path={path_str}",
        "dataset.name=ncore",
        "dataset.n_samples_per_epoch=1",
        "dataset.camera_ids=[camera_front_wide_120fov]",
        "dataset.lidar_ids=[lidar_gt_top_p128]",
        "dataset.seek_offset_sec=0.1",
        "dataset.samplers.batch_sampler.camera_pixel_sampler.subsample=2",
        "dataset.show_progress_bars=False",
        "dataset.val_sensor_transl_delta_m=[0.01, 0.05, 0.1]",
        "dataset.val_sensor_rot_delta_deg=[2.0, 4.0, 6.0]",
        "dataset.val_lidar=False",
        "dataset.n_val_image_subsample=2",
        "out_dir=/doesnt/matter",
    ]
    config_name = "apps/Alpasim/alpasim_3dgut_speed"

    # Must keep a reference to the real get_edge before patch.object replaces it on the class;
    # otherwise calling PoseGraphInterpolator.get_edge from inside the stub would recurse.
    real_get_edge = ncore_tf.PoseGraphInterpolator.get_edge

    def get_edge_with_missing_world_global(self, source: str, target: str):
        if source == "world" and target == "world_global":
            return None
        return real_get_edge(self, source, target)

    with patch.object(
        ncore_tf.PoseGraphInterpolator,
        "get_edge",
        get_edge_with_missing_world_global,
    ):
        config = parse_typed_config(config_name, hydra_args)
        train_dataset = cast(NCORETrainDataset, make_dataset(config.dataset.name, config, split="train"))

    ds = train_dataset.datasource
    assert ds.T_world_world_global.shape == (4, 4)
    assert ds.T_world_world_global.dtype == np.float64
    np.testing.assert_array_almost_equal(ds.T_world_world_global, np.eye(4, dtype=np.float64))

    rig = ds.get_rig_trajectories()
    assert rig.T_world_base.dtype == torch.float64
    eye = torch.eye(4, device=rig.T_world_base.device, dtype=rig.T_world_base.dtype)
    assert torch.allclose(rig.T_world_base, eye)
