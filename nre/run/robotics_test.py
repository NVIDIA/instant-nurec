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
import zipfile

from pathlib import Path

import pytest

from click.testing import CliRunner
from python.runfiles import runfiles

from nre.run.main import main


RUNFILES = runfiles.Create()


@pytest.fixture
def small_dataset_path() -> Path:
    path = Path(
        RUNFILES.Rlocation(
            "test_robotics_data/ncore_galileofigure8b/ncore/2024-08-13_17-53-56_galileofigure8b/2024-08-13_17-53-56_galileofigure8b.json"
        )
    ).parent.parent.parent
    if not path.exists():
        raise AssertionError(
            f"Test dataset not found. This is an issue with your filesystem/test suite, not the code under test. Missing {path=}"
        )
    return path


@pytest.fixture(scope="session")
def output_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Common temporary output folder for tests that execute *both* 'train' (test_training), 'val' (test_validation), and 'test' (test_test) stages and need a common shared folder"""
    return tmp_path_factory.mktemp("nre_robotics_training_pytest_output")


@pytest.mark.dependency()
def test_training(small_dataset_path: Path, output_root: Path) -> None:
    """Test validating 'train' mode on a default config - output will be stored in 'output_root', to be consumed by subsequent test_validation and test_test tests"""

    print("Output root:", output_root)

    small_dataset_path_str = str(small_dataset_path)

    result = CliRunner().invoke(
        main,
        [
            "--config-name=apps/AV/NV/3dgut_static.yaml",
            "model/gaussians/initialization@model.layers.background.initialization=accumulated_point_cloud",
            f'model.layers.background.initialization.point_cloud_path="{small_dataset_path_str}/fused_mesh/fused_mesh.ply"',
            f'dataset.path="{small_dataset_path_str}/ncore/2024-08-13_17-53-56_galileofigure8b/2024-08-13_17-53-56_galileofigure8b.json"',
            "dataset.n_train_sample_lidar_rays=0",
            "trainer.max_epochs=1",
            "dataset.n_samples_per_epoch=10",
            "mode=train",
            f"out_dir={output_root}",
            "logger=wandb",
            "logger.offline=true",
            "logger.run_id=out",
            "checkpoint.artifact.enabled=true",
            "checkpoint.artifact.sequence_tracks.enabled=true",
            "checkpoint.artifact.rig_trajectories.enabled=true",
            "model/background=color",
            "dataset.camera_ids=[front_stereo_camera_left]",
            "dataset.max_dist_m=20.0",
            "dataset.lidar_ids=[]",
            "loss.lidar.lambda_=0.0",
            "+loss.distance.reduce.name=mean",
            "+loss.distance.fn=log_l1",
            "+loss.distance.min_distance=0.05",
            "+loss.distance.max_distance=20.0",
            "+loss.distance.normalize_by_opacity=false",
            "+loss.distance.allow_missing_supervision=false",
            "+loss.distance.lambda_=0.01",
            "+loss.distance.mask_semantic_classes=['~road',road]",
            "+loss.distance.semantic_lambdas=[1.0,3.0]",
            "checkpoint.every_n_train_steps=1000",
            "model.renderer.culling.tile_based=false",
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
def test_validation(output_root: Path) -> None:
    """Verifies 'val' mode executed on 'train' results of earlier 'test_training' test that outputted results to 'output_root'"""

    print("Output root:", output_root)

    config_file = output_root / "out" / "config" / "parsed.yaml"

    result = CliRunner().invoke(
        main,
        [
            f"--config-name={config_file}",
            "mode=val",
            "resume=last.ckpt",
            "dataset.n_val_image_subsample=4",  # >1 tests the subsampling feature and is also faster (less rays)
            # reduce runtime / number of rendered frames
            "dataset.val_camera_frame_step=20",
            "dataset.n_train_sample_lidar_rays=0",
            "system.test.save_results=true",
            "system.test.save_inputs=true",
            "system.test.save_videos=true",
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
