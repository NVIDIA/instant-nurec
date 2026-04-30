# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION
# or its affiliates is strictly prohibited.

"""Tests for ncore_aux_data module."""

from unittest.mock import MagicMock, patch

import numpy as np

from apps.aux_gen.ncore_aux_data import _run_ego_mask_for_camera, run_mask2former_segmentation


def test_imports():
    """Verify ncore_aux_data module and its external dependencies can be imported."""
    import apps.aux_gen.ncore_aux_data  # noqa: F401

    assert apps.aux_gen.ncore_aux_data.cli is not None


def test_run_ego_mask_for_camera():
    """Test per-camera ego mask logic: default all, or only listed cameras."""
    # --no-ego-mask: never run
    assert _run_ego_mask_for_camera(False, None, "cam1") is False
    assert _run_ego_mask_for_camera(False, ["cam1"], "cam1") is False

    # --ego-mask, no --ego-mask-camera-id: run for every camera
    assert _run_ego_mask_for_camera(True, None, "cam1") is True
    assert _run_ego_mask_for_camera(True, [], "cam1") is True
    assert _run_ego_mask_for_camera(True, (), "cam2") is True  # tuple from Click multiple=True

    # --ego-mask --ego-mask-camera-id cam1: run only for cam1
    assert _run_ego_mask_for_camera(True, ["cam1"], "cam1") is True
    assert _run_ego_mask_for_camera(True, ["cam1"], "cam2") is False
    assert _run_ego_mask_for_camera(True, ["cam1", "cam2"], "cam1") is True
    assert _run_ego_mask_for_camera(True, ["cam1", "cam2"], "cam3") is False


def test_run_mask2former_segmentation_does_not_call_get_egomask_when_has_ego_mask_false():
    """When has_ego_mask is False, writer.get_egomask must not be called (avoids creating empty egomask store)."""
    estimator = MagicMock()
    estimator.get_semantic_metadata.return_value = {"stuff_classes": [], "stuff_colors": []}
    writer = MagicMock()
    camera_model_parameters = MagicMock()
    camera_model_parameters.resolution = np.array([64, 64])  # .tolist() is called on it
    camera_sensor = MagicMock()

    with patch("apps.aux_gen.ncore_aux_data.get_camera_sensor_mask", return_value=None):
        run_mask2former_segmentation(
            estimator,
            writer,
            camera_id="cam1",
            camera_model_parameters=camera_model_parameters,
            camera_sensor=camera_sensor,
            timestamped_image_frame_handles=[],  # no frames so no predict loop
            visualize=False,
            has_ego_mask=False,
        )
    writer.get_egomask.assert_not_called()


def test_run_mask2former_segmentation_calls_get_egomask_when_has_ego_mask_true_and_no_sensor_mask():
    """When has_ego_mask is True and get_camera_sensor_mask returns None, writer.get_egomask is used."""
    estimator = MagicMock()
    estimator.get_semantic_metadata.return_value = {"stuff_classes": [], "stuff_colors": []}
    writer = MagicMock()
    writer.get_egomask.return_value = None
    camera_model_parameters = MagicMock()
    camera_model_parameters.resolution = np.array([64, 64])
    camera_sensor = MagicMock()

    with patch("apps.aux_gen.ncore_aux_data.get_camera_sensor_mask", return_value=None):
        run_mask2former_segmentation(
            estimator,
            writer,
            camera_id="cam1",
            camera_model_parameters=camera_model_parameters,
            camera_sensor=camera_sensor,
            timestamped_image_frame_handles=[],
            visualize=False,
            has_ego_mask=True,
        )
    writer.get_egomask.assert_called_once_with("cam1", 0)
