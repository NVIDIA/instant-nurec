# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from pathlib import Path
from typing import cast

import pytest
import torch

from omegaconf import OmegaConf
from python.runfiles import runfiles

from nre.config.model import CalibConfig
from nre.config.nre import NREConfig
from nre.config.parse import parse_typed_config
from nre.datasets import make as make_dataset
from nre.datasets.ncore import NCORETrainDataset
from nre.datasets.summary import DataSourceSummary
from nre.models.calib import BaseCalib
from nre.utils.batch import CameraFreePoseViewGeometry, DataBatch, RectSubsampled
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


@pytest.fixture
def small_clipgt_dataset(small_clipgt_dataset_path: Path) -> tuple[NCORETrainDataset, NREConfig]:
    config_name = "apps/Alpasim/alpasim_3dgut_speed"

    # Ensure the path is a *quoted* string for Hydra compatibility with bazel's `~`-separated paths
    small_clipgt_dataset_path_str = '"{0}"'.format(str(small_clipgt_dataset_path))

    hydra_args = [
        f"dataset.path={small_clipgt_dataset_path_str}",
        "dataset.name=ncore",
        "dataset.n_samples_per_epoch=2",  # Using a low number of iterations speeds the test up
        "dataset.camera_ids=[camera_front_wide_120fov]",
        "dataset.lidar_ids=[lidar_gt_top_p128]",
        "dataset.samplers.batch_sampler.camera_pixel_sampler.subsample=2",  # test data is 480x270
        "dataset.show_progress_bars=False",
        "dataset.n_val_image_subsample=2",
        "out_dir=/doesnt/matter",  # required field
    ]

    config = parse_typed_config(config_name, hydra_args)
    train_dataset = cast(NCORETrainDataset, make_dataset(config.dataset.name, config, split="train"))
    return train_dataset, config


def test_v2_free_pose_calib(small_clipgt_dataset: tuple[NCORETrainDataset, NREConfig]) -> None:
    train_dataset, _ = small_clipgt_dataset

    # Create a DataBatch from the dataset.
    camera_id = "camera_front_wide_120fov"
    camera_frame_idx = 2
    sampled_pixels = RectSubsampled(
        width=240, height=135, subsample_factor=2.0, original_width=480, original_height=270
    )
    camera_data_batch: DataBatch.Camera = train_dataset.get_camera_data_batch(
        sensor_id=camera_id,
        sensor_frame_idx=camera_frame_idx,
        sampled_pixels=sampled_pixels,
    )
    lidar_id = "lidar_gt_top_p128"
    lidar_frame_idx = 2
    lidar_data_batch: DataBatch.Lidar = train_dataset.get_lidar_data_batch(
        sensor_id=lidar_id,
        sensor_frame_idx=lidar_frame_idx,
    )
    data_batch = DataBatch(camera=camera_data_batch, lidar=lidar_data_batch)

    # Create a config with required fields so that the calib config is happy
    config = OmegaConf.create(
        {
            "system": OmegaConf.load("configs/system/gaussians.yaml"),
            "trainer": OmegaConf.load("configs/trainer/base.yaml"),
            "dataset": OmegaConf.load("configs/dataset/ncore.yaml"),
            "calib": OmegaConf.load("configs/model/calib/free_pose.yaml"),
        }
    )
    OmegaConf.resolve(config)  # Resolve all interpolations

    # Create the calib model and run forward pass
    # Validate calib config to convert from DictConfig to typed CalibConfig
    calib_config = CalibConfig.model_validate(config.calib)
    calib = BaseCalib.factory(
        name=config.calib.name,
        config=calib_config,
        trainer_config=config.trainer,
        datasource=DataSourceSummary.from_datasource(train_dataset.datasource),
    )

    # Only CUDA version is supported: put model and data on the same device so every tensor is on GPU
    device = torch.device("cuda")
    calib = calib.to(device)
    data_batch = data_batch.to(device=device)
    rendering_batch = calib(data_batch)
    _ = calib.configure_optimizers()  # Test optimizer setup works

    # Test the autograd graph for the pose outputs
    loss = (
        unpack_optional(rendering_batch.camera).poses_tquat_startend.sum()
        + unpack_optional(rendering_batch.lidar).poses_tquat_startend.sum()
    )
    grads = torch.autograd.grad(loss, calib.params)
    for grad in grads:
        assert (grad is not None) and (grad != 0).any()

    # Re-run forward pass to make sure the graph is not retained
    rendering_batch = calib(data_batch)

    # Test the autograd graph for the ray outputs
    loss = unpack_optional(rendering_batch.camera).rays.sum() + unpack_optional(rendering_batch.lidar).rays.sum()
    grads = torch.autograd.grad(loss, calib.params)
    for grad in grads:
        assert (grad is not None) and (grad != 0).any()


def test_calib_interp_with_rig(small_clipgt_dataset: tuple[NCORETrainDataset, NREConfig]) -> None:
    train_dataset, _ = small_clipgt_dataset

    rig_trajectories = train_dataset.datasource.get_rig_trajectories()

    # For the test case (and most of the cases), we expect `interp_with_rig=True` to be the same as `interp_with_rig=False`
    # I.e. They contain the same poses.
    view_geometry1 = CameraFreePoseViewGeometry.from_rig_trajectories(rig_trajectories, interp_with_rig=True)
    view_geometry2 = CameraFreePoseViewGeometry.from_rig_trajectories(rig_trajectories, interp_with_rig=False)

    torch.testing.assert_close(
        view_geometry1.T_sensor_world_startend_allviews,
        view_geometry2.T_sensor_world_startend_allviews,
    )
