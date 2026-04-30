# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import contextlib
import sys
import unittest

from unittest.mock import MagicMock

import omegaconf
import pytest
import torch
import torch.nn.functional as F

# Import to trigger loss function registration
import libs.losses.models.loss_fns  # noqa: F401

from libs.losses.models.render_losses import (
    BackgroundInTrackGaussianLoss,
    DistanceLoss,
    GaussianFlattenLoss,
    NodeSemanticGaussiansLoss,
    NormalLoss,
)
from libs.losses.orchestration.config import LossItemConfig
from nre.config.trainer import TrainerConfig
from nre.datasets.ncore import NCOREDataset
from nre.models.gaussians.gaussians_model import RigidGaussianModel
from nre.utils.batch import CameraFrameLabels, DataAndRenderingBatch, DataBatch, FrameMeta
from nre.utils.types import GaussiansCompositeReturn, GaussiansRenderReturn, RayFlags


class MockTrainerConfig(TrainerConfig):
    def __init__(self):
        super().__init__(
            max_epochs=1,
            check_val_every_n_epoch=1,
            precision="32",
            log_every_n_steps=1,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
        )


# Normal Loss Tests


class TestNormalLoss(unittest.TestCase):
    def _create_test_data_batch(
        self, pred_normal: torch.Tensor, gt_normal: torch.Tensor
    ) -> tuple[GaussiansCompositeReturn, DataAndRenderingBatch]:
        n_rays = len(pred_normal)
        device = pred_normal.device

        flags = torch.full(
            (1, gt_normal.shape[1], gt_normal.shape[2], 1),
            RayFlags.VALID_NORMAL.value,
            dtype=torch.int32,
            device=device,
        )

        data_camera_batch = DataBatch.Camera(
            meta=[
                FrameMeta(
                    unique_sensor_idx=0,
                    unique_frame_idx=0,
                )
            ],
            labels=CameraFrameLabels(
                rgb=torch.zeros_like(gt_normal),
                normals=gt_normal,
                flags=flags,
            ),
        )
        rendered_cam = GaussiansRenderReturn(
            rgb=torch.zeros_like(pred_normal),
            opacity=torch.ones(n_rays, device=device),
            distance=torch.zeros(n_rays, device=device),
            normal=pred_normal,
        )
        results = GaussiansCompositeReturn(rendered_cam=rendered_cam)
        data_batch = DataBatch(
            idx=0,
            worker_id=None,
            sequence_id=["dummy"],
            lidar=None,
            camera=data_camera_batch,
        )
        batch = DataAndRenderingBatch(data=data_batch)
        return results, batch

    def test_zero_loss_when_normals_match(self):
        device = torch.device("cuda")
        h, w = 5, 4
        n_rays = h * w
        pred_normal = torch.randn(n_rays, 3, device=device)
        pred_normal = F.normalize(pred_normal, dim=1)
        gt_normal = pred_normal.clone().reshape((1, h, w, 3))

        results, batch = self._create_test_data_batch(pred_normal, gt_normal)

        config = LossItemConfig.model_validate({"fn": "l1", "lambda_": 1.0, "reduce": {"name": "mean"}})
        trainer_config = MockTrainerConfig()
        loss = NormalLoss(config, trainer_config)
        loss_ret = loss(results=results, target=batch, model=None)
        self.assertIsNotNone(loss_ret)
        self.assertTrue(torch.isclose(loss_ret.reduced_value, torch.tensor(0.0)))

    def test_positive_loss_when_normals_differ(self):
        device = torch.device("cuda")
        h, w = 5, 4
        n_rays = h * w
        gt_normal = F.normalize(torch.randn(n_rays, 3, device=device), dim=1).reshape((1, h, w, 3))
        pred_normal = F.normalize(torch.randn(n_rays, 3, device=device), dim=1)

        results, batch = self._create_test_data_batch(pred_normal, gt_normal)

        config = LossItemConfig.model_validate({"fn": "l1", "lambda_": 1.0, "reduce": {"name": "mean"}})
        trainer_config = MockTrainerConfig()
        loss = NormalLoss(config, trainer_config)
        loss_ret = loss(results=results, target=batch, model=None)
        self.assertIsNotNone(loss_ret)
        self.assertGreater(loss_ret.reduced_value.item(), 0.0)

    def test_returns_none_when_rays_cam_is_none(self):
        device = torch.device("cuda")
        h, w = 5, 4
        n_rays = h * w
        pred_normal = F.normalize(torch.randn(n_rays, 3, device=device), dim=1)
        gt_normal = pred_normal.clone().reshape((1, h, w, 3))

        results, batch = self._create_test_data_batch(pred_normal, gt_normal)
        batch.data.camera = None  # Trigger early exit in NormalLoss

        config = LossItemConfig.model_validate({"fn": "l1", "lambda_": 1.0, "reduce": {"name": "mean"}})
        trainer_config = MockTrainerConfig()
        loss = NormalLoss(config, trainer_config)
        loss_ret = loss(results=results, target=batch, model=None)
        self.assertIsNone(loss_ret)

    def test_no_difixed_flag(self):
        device = torch.device("cuda")
        h, w = 5, 4
        n_rays = h * w
        pred_normal = F.normalize(torch.randn(n_rays, 3, device=device), dim=1)
        gt_normal = pred_normal.clone().reshape((1, h, w, 3))

        results, batch = self._create_test_data_batch(pred_normal, gt_normal)
        # Ensure no DIFIXED flag is present
        assert batch.data.camera is not None
        self.assertFalse(batch.data.camera.labels.get_mask_flags_all(RayFlags.DIFIXED).any())

        config = LossItemConfig.model_validate({"fn": "l1", "lambda_": 1.0, "reduce": {"name": "mean"}})
        trainer_config = MockTrainerConfig()
        loss = NormalLoss(config, trainer_config)
        loss_ret = loss(results=results, target=batch, model=None)
        self.assertIsNotNone(loss_ret)

    def test_returns_when_normals_are_none(self):
        device = torch.device("cuda")
        h, w = 5, 4
        n_rays = h * w
        pred_normal = F.normalize(torch.randn(n_rays, 3, device=device), dim=1)
        gt_normal = pred_normal.clone().reshape((1, h, w, 3))

        results, batch = self._create_test_data_batch(pred_normal, gt_normal)
        assert batch.data.camera is not None
        # Remove normals from ground truth to emulate missing supervision
        batch.data.camera.labels.normals = None

        # When allow_missing_supervision is set, the loss should return None instead of raising.
        config = LossItemConfig.model_validate(
            {
                "fn": "l1",
                "lambda_": 1.0,
                "reduce": {"name": "mean"},
                "allow_missing_supervision": True,
            }
        )
        trainer_config = MockTrainerConfig()
        loss = NormalLoss(config, trainer_config)
        loss_ret = loss(results=results, target=batch, model=None)
        self.assertIsNone(loss_ret)

        # Without the flag we expect a ValueError to be raised.
        config_no_flag = LossItemConfig.model_validate(
            {
                "fn": "l1",
                "lambda_": 1.0,
                "reduce": {"name": "mean"},
            }
        )
        loss_no_flag = NormalLoss(config_no_flag, trainer_config)
        with self.assertRaises(ValueError):
            _ = loss_no_flag(results=results, target=batch, model=None)

    def test_no_valid_normal_flag_returns_zero_loss(self):
        """Test that when no rays have VALID_NORMAL flag, loss returns 0 (not None)."""
        device = torch.device("cuda")
        h, w = 5, 4
        n_rays = h * w
        pred_normal = F.normalize(torch.randn(n_rays, 3, device=device), dim=1)
        gt_normal = pred_normal.clone().reshape((1, h, w, 3))

        results, batch = self._create_test_data_batch(pred_normal, gt_normal)
        assert batch.data.camera is not None
        # Strip VALID_NORMAL flag from all rays
        batch.data.camera.labels.flags = torch.zeros((1, h, w, 1), dtype=torch.int32, device=device)
        self.assertFalse(batch.data.camera.labels.get_mask_flags_all(RayFlags.VALID_NORMAL).any())

        config = LossItemConfig.model_validate({"fn": "l1", "lambda_": 1.0, "reduce": {"name": "mean"}})
        trainer_config = MockTrainerConfig()
        loss = NormalLoss(config, trainer_config)
        loss_ret = loss(results=results, target=batch, model=None)
        # When all rays are masked out, reduce_mask is all zeros → reduced_value is 0
        self.assertIsNotNone(loss_ret)
        self.assertEqual(loss_ret.reduced_value.item(), 0.0)


# Distance Loss Tests


class TestDistanceLoss(unittest.TestCase):
    def _create_test_data_batch(
        self,
        pred_distance: torch.Tensor,
        gt_distance: torch.Tensor | None,
        flags: torch.Tensor,
        opacity: torch.Tensor | None = None,
    ) -> tuple[GaussiansCompositeReturn, DataAndRenderingBatch]:
        n_rays = pred_distance.numel()
        device = pred_distance.device

        h, w = flags.shape[1], flags.shape[2]

        rendered_cam = GaussiansRenderReturn(
            rgb=torch.zeros(n_rays, 3, device=device),
            opacity=(opacity if opacity is not None else torch.ones(n_rays, device=device)),
            distance=pred_distance,
        )

        labels = CameraFrameLabels(
            rgb=torch.zeros((1, h, w, 3), device=device),
            metric_distance=gt_distance,
            flags=flags,
        )

        data_camera_batch = DataBatch.Camera(
            meta=[
                FrameMeta(
                    unique_sensor_idx=0,
                    unique_frame_idx=0,
                )
            ],
            labels=labels,
        )

        results = GaussiansCompositeReturn(rendered_cam=rendered_cam)
        data_batch = DataBatch(
            idx=0,
            worker_id=None,
            sequence_id=["dummy"],
            lidar=None,
            camera=data_camera_batch,
        )
        batch = DataAndRenderingBatch(data=data_batch)
        return results, batch

    def test_basic_mask_range(self):
        device = torch.device("cuda")
        h, w = 2, 3

        # Ground truth distances (B, H, W, 1)
        gt_vals = torch.tensor([0.5, 2.0, 5.0, 9.0, 3.0, 7.5], device=device, dtype=torch.float32)
        gt = gt_vals.view(1, h, w, 1)

        # Predictions per-ray (flattened)
        # diffs for valid ones (2.0,5.0,3.0,7.5) are [1.0, -1.0, 2.0, -0.5]
        pred = gt_vals.clone()
        pred[1] += 1.0
        pred[2] -= 1.0
        pred[4] += 2.0
        pred[5] -= 0.5

        # All rays valid
        flags = torch.zeros((1, h, w, 1), dtype=torch.int32, device=device)

        results, batch = self._create_test_data_batch(pred, gt, flags)

        # Test depth_inverse_mse loss
        config = LossItemConfig.model_validate(
            {
                "fn": "depth_inverse_mse",
                "lambda_": 1.0,
                "reduce": {"name": "mean"},
                "min_distance": 0.5,
                "max_distance": 8.0,
                "normalize_by_opacity": False,
            }
        )
        trainer_config = MockTrainerConfig()
        loss = DistanceLoss(config, trainer_config)
        loss_ret = loss(results=results, target=batch, model=None)
        self.assertIsNotNone(loss_ret)

        # Valid indices according to range: 1,2,4,5
        valid_indices = torch.tensor([1, 2, 4, 5], device=device).long()
        expected = ((pred.view(-1)[valid_indices].reciprocal() - gt.view(-1)[valid_indices].reciprocal()) ** 2).mean()
        self.assertTrue(torch.isclose(loss_ret.reduced_value, expected))

        # Test log_l1 loss
        config = LossItemConfig.model_validate(
            {
                "fn": "log_l1",
                "lambda_": 1.0,
                "reduce": {"name": "mean"},
                "min_distance": 0.5,
                "max_distance": 8.0,
                "normalize_by_opacity": False,
            }
        )
        trainer_config = MockTrainerConfig()
        loss = DistanceLoss(config, trainer_config)
        loss_ret = loss(results=results, target=batch, model=None)
        self.assertIsNotNone(loss_ret)

        # Valid indices according to range: 1,2,4,5
        valid_indices = torch.tensor([1, 2, 4, 5], device=device).long()
        expected = torch.log(1.0 + (pred.view(-1)[valid_indices] - gt.view(-1)[valid_indices]).abs()).mean()
        self.assertTrue(torch.isclose(loss_ret.reduced_value, expected))

    def test_normalize_by_opacity(self):
        device = torch.device("cuda")
        h, w = 1, 4
        n_rays = h * w

        gt_vals = torch.tensor([2.0, 4.0, 6.0, 8.0], device=device)
        gt = gt_vals.view(1, h, w, 1)
        pred_vals = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        opacity = torch.full((n_rays,), 0.5, device=device)

        flags = torch.zeros((1, h, w, 1), dtype=torch.int32, device=device)
        results, batch = self._create_test_data_batch(pred_vals, gt, flags, opacity=opacity)

        config = LossItemConfig.model_validate(
            {
                "fn": "depth_inverse_mse",
                "lambda_": 1.0,
                "reduce": {"name": "mean"},
                "min_distance": 0.0,
                "max_distance": 100.0,
                "normalize_by_opacity": True,
            }
        )
        trainer_config = MockTrainerConfig()
        loss = DistanceLoss(config, trainer_config)
        loss_ret = loss(results=results, target=batch, model=None)
        self.assertIsNotNone(loss_ret)

        # pred/op = [2,4,6,8] -> diffs zero
        expected = torch.tensor(0.0, device=device)
        self.assertTrue(torch.isclose(loss_ret.reduced_value, expected))

    def test_returns_none_when_camera_is_none(self):
        device = torch.device("cuda")
        h, w = 1, 2

        # Create dummy predictions and ground truth so the batch is well-formed
        pred_vals = torch.tensor([1.0, 2.0], device=device)
        gt_vals = torch.tensor([2.0, 4.0], device=device)
        gt = gt_vals.view(1, h, w, 1)
        flags = torch.zeros((1, h, w, 1), dtype=torch.int32, device=device)

        results, batch = self._create_test_data_batch(pred_vals, gt, flags)
        # Trigger early exit in DistanceLoss.forward when camera is None
        batch.data.camera = None

        config = LossItemConfig.model_validate(
            {
                "fn": "depth_inverse_mse",
                "lambda_": 1.0,
                "reduce": {"name": "mean"},
                "min_distance": 0.0,
                "max_distance": 100.0,
                "normalize_by_opacity": False,
            }
        )
        trainer_config = MockTrainerConfig()
        loss = DistanceLoss(config, trainer_config)
        loss_ret = loss(results=results, target=batch, model=None)
        self.assertIsNone(loss_ret)

    def test_difixed_flag_excludes_rays(self):
        """Test that DIFIXED rays are excluded from loss computation (not that loss returns None)."""
        device = torch.device("cuda")
        h, w = 1, 2

        gt_vals = torch.tensor([2.0, 4.0], device=device)
        gt = gt_vals.view(1, h, w, 1)
        pred_vals = torch.tensor([1.0, 2.0], device=device)

        flags = torch.zeros((1, h, w, 1), dtype=torch.int32, device=device)
        flags[:, 0, 1, 0] = RayFlags.DIFIXED.value

        results, batch = self._create_test_data_batch(pred_vals, gt, flags)

        config = LossItemConfig.model_validate(
            {
                "fn": "depth_inverse_mse",
                "lambda_": 1.0,
                "reduce": {"name": "mean"},
                "min_distance": 0.0,
                "max_distance": 100.0,
                "normalize_by_opacity": False,
            }
        )
        trainer_config = MockTrainerConfig()
        loss = DistanceLoss(config, trainer_config)
        loss_ret = loss(results=results, target=batch, model=None)
        # DIFIXED rays are excluded, but valid rays still contribute to loss
        self.assertIsNotNone(loss_ret)
        # Only pixel [0,0] is valid (pred=1.0, gt=2.0), DIFIXED pixel [0,1] is excluded
        # depth_inverse_mse: (1/pred - 1/gt)^2 = (1/1 - 1/2)^2 = 0.25
        self.assertAlmostEqual(loss_ret.reduced_value.item(), 0.25, places=4)

    def test_allow_missing_supervision(self):
        device = torch.device("cuda")
        h, w = 1, 2

        pred_vals = torch.tensor([1.0, 2.0], device=device)
        flags = torch.zeros((1, h, w, 1), dtype=torch.int32, device=device)

        # No supervision
        results, batch = self._create_test_data_batch(pred_vals, None, flags)

        config_ok = LossItemConfig.model_validate(
            {
                "fn": "depth_inverse_mse",
                "lambda_": 1.0,
                "reduce": {"name": "mean"},
                "min_distance": 0.0,
                "max_distance": 100.0,
                "normalize_by_opacity": False,
                "allow_missing_supervision": True,
            }
        )
        trainer_config = MockTrainerConfig()
        loss_ok = DistanceLoss(config_ok, trainer_config)
        self.assertIsNone(loss_ok(results=results, target=batch, model=None))

        config_fail = LossItemConfig.model_validate(
            {
                "fn": "depth_inverse_mse",
                "lambda_": 1.0,
                "reduce": {"name": "mean"},
                "min_distance": 0.0,
                "max_distance": 100.0,
                "normalize_by_opacity": False,
            }
        )
        loss_fail = DistanceLoss(config_fail, trainer_config)
        with self.assertRaises(ValueError):
            _ = loss_fail(results=results, target=batch, model=None)

    def test_no_valid_mask_returns_zero_loss(self):
        """Test that when all pixels are outside distance range, loss returns 0 (not None)."""
        device = torch.device("cuda")
        h, w = 1, 3

        # All gt outside range
        gt_vals = torch.tensor([0.1, 100.0, 0.2], device=device)
        gt = gt_vals.view(1, h, w, 1)
        pred_vals = torch.tensor([0.1, 100.0, 0.2], device=device)
        flags = torch.zeros((1, h, w, 1), dtype=torch.int32, device=device)

        results, batch = self._create_test_data_batch(pred_vals, gt, flags)

        config = LossItemConfig.model_validate(
            {
                "fn": "depth_inverse_mse",
                "lambda_": 1.0,
                "reduce": {"name": "mean"},
                "min_distance": 0.5,
                "max_distance": 10.0,
                "normalize_by_opacity": False,
            }
        )
        trainer_config = MockTrainerConfig()
        loss = DistanceLoss(config, trainer_config)
        loss_ret = loss(results=results, target=batch, model=None)
        # When all pixels are masked out, reduce_mask is all zeros → reduced_value is 0
        self.assertIsNotNone(loss_ret)
        self.assertEqual(loss_ret.reduced_value.item(), 0.0)

    def test_semantic_masking_with_lambda(self):
        device = torch.device("cuda")
        h, w = 1, 4

        gt_vals = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
        gt = gt_vals.view(1, h, w, 1)
        pred_vals = gt_vals + torch.tensor([1.0, -1.0, 2.0, -2.0], device=device)

        # Mark all rays as VALID and ROAD
        flags = torch.zeros((1, h, w, 1), dtype=torch.int32, device=device)
        # Set ROAD_SEMANTIC on first two rays only
        flags[:, 0, 0, 0] = RayFlags.ROAD_SEMANTIC.value
        flags[:, 0, 1, 0] = RayFlags.ROAD_SEMANTIC.value

        results, batch = self._create_test_data_batch(pred_vals, gt, flags)

        config = LossItemConfig.model_validate(
            {
                "fn": "depth_inverse_mse",
                "lambda_": 1.0,
                "reduce": {"name": "mean"},
                "min_distance": 0.0,
                "max_distance": 10.0,
                "normalize_by_opacity": False,
                "mask_semantic_classes": ["road"],
                "semantic_lambdas": [2.0],
            }
        )
        trainer_config = MockTrainerConfig()
        loss = DistanceLoss(config, trainer_config)
        loss_ret = loss(results=results, target=batch, model=None)
        self.assertIsNotNone(loss_ret)

        # Only first two rays (road) contribute, lambda=2 amplifies their squared diffs
        masked = torch.tensor([True, True, False, False], device=device)
        per_ray = torch.zeros_like(pred_vals)
        inv_pred_masked = pred_vals.view(-1)[masked].reciprocal()
        inv_gt_masked = gt_vals.view(-1)[masked].reciprocal()
        per_ray[masked] = 2.0 * ((inv_pred_masked - inv_gt_masked) ** 2)
        expected = per_ray.mean()
        self.assertTrue(torch.isclose(loss_ret.reduced_value, expected))


# Gaussian Flatten Loss Tests


class _DummyGaussianNode:
    """Minimal stub mimicking a Gaussian node"""

    def __init__(self, scales: torch.Tensor):
        self._scales = scales

    def get_scales(self):
        return self._scales


class _DummyModel:
    """Minimal stub mimicking a GaussiansComposite-compatible model"""

    def __init__(self, scales: torch.Tensor):
        self.gaussians_nodes = {"layer": _DummyGaussianNode(scales)}

    def get_gaussians_node_ids(self):
        return list(self.gaussians_nodes.keys())


class TestGaussianFlattenLoss(unittest.TestCase):
    def setUp(self):
        self.trainer_config = MockTrainerConfig()
        self.max_to_median_ratio_threshold = 1.5
        self.device = torch.device("cpu")

    def _create_loss(self, axes_type: str) -> GaussianFlattenLoss:
        config = LossItemConfig.model_validate(
            {
                "fn": "abs",  # requires single argument only
                "lambda_": 1.0,
                "reduce": {"name": "mean"},
                "max_to_median_ratio_threshold": self.max_to_median_ratio_threshold,
                "axes_type": axes_type,
            }
        )
        return GaussianFlattenLoss(config, self.trainer_config)

    def _run_loss(self, loss, model):
        # Patch supported_model_types to accept the dummy model for positive tests
        loss.supported_model_types = (type(model),)
        loss_ret = loss(results=None, target=None, model=model)
        return loss_ret

    def test_fixed_axes_type(self):
        scales = torch.tensor([[1.0, 1.0, 0.2], [2.0, 1.0, 0.5]], device=self.device)
        model = _DummyModel(scales)
        loss = self._create_loss("fixed")
        loss_ret = self._run_loss(loss, model)
        self.assertIsNotNone(loss_ret)
        # Expected flatten loss per gaussian
        ratios = torch.max(scales[:, 0] / scales[:, 1], scales[:, 1] / scales[:, 0])
        expected_per_gauss = torch.relu(ratios - self.max_to_median_ratio_threshold) + scales[:, 2]
        expected_mean = expected_per_gauss.mean()
        self.assertTrue(torch.isclose(loss_ret.reduced_value, expected_mean))

    def test_free_axes_type(self):
        scales = torch.tensor([[0.8, 1.0, 3.0], [1.0, 2.0, 5.0]], device=self.device)
        model = _DummyModel(scales)
        loss = self._create_loss("free")
        loss_ret = self._run_loss(loss, model)
        self.assertIsNotNone(loss_ret)
        scales_sorted = scales.sort(dim=1, descending=False).values
        ratios = scales_sorted[:, 2] / scales_sorted[:, 1]
        expected_per_gauss = torch.relu(ratios - self.max_to_median_ratio_threshold) + scales_sorted[:, 0]
        expected_mean = expected_per_gauss.mean()
        self.assertTrue(torch.isclose(loss_ret.reduced_value, expected_mean))

    def test_invalid_axes_type_raises(self):
        loss = self._create_loss("invalid")
        scales = torch.tensor([[1.0, 1.0, 0.2]], device=self.device)
        model = _DummyModel(scales)
        # Patch supported types so we hit invalid axes_type branch
        loss.supported_model_types = (type(model),)
        with self.assertRaises(ValueError):
            loss(results=None, target=None, model=model)

    def test_model_type_assertion(self):
        # Without patching supported types -> should assert
        loss = self._create_loss("fixed")
        scales = torch.tensor([[1.0, 1.0, 0.2]], device=self.device)
        model = _DummyModel(scales)
        with self.assertRaises(AssertionError):
            loss(results=None, target=None, model=model)

    def test_empty_gaussians(self):
        class _EmptyModel(_DummyModel):
            def __init__(self):
                pass

            def get_gaussians_node_ids(self):
                return []

            def get_scales(self):
                raise RuntimeError("Should not be called")

        model = _EmptyModel()
        loss = self._create_loss("fixed")
        loss.supported_model_types = (type(model),)
        with self.assertRaises(RuntimeError):
            loss(results=None, target=None, model=model)

    def test_ratio_smaller_than_threshold(self):
        scales = torch.tensor([[1.0, 1.0, 0.3]], device=self.device)
        model = _DummyModel(scales)
        loss = self._create_loss("fixed")
        loss_ret = self._run_loss(loss, model)
        expected = torch.tensor([0.3]).mean()
        self.assertTrue(torch.isclose(loss_ret.reduced_value, expected))

    def test_ratio_larger_than_threshold(self):
        scales = torch.tensor([[3.0, 0.5, 0.3]], device=self.device)  # ratio 6
        model = _DummyModel(scales)
        loss = self._create_loss("fixed")
        loss_ret = self._run_loss(loss, model)
        ratio = torch.tensor([6.0])
        expected = torch.relu(ratio - self.max_to_median_ratio_threshold) + scales[:, 2]
        self.assertTrue(torch.isclose(loss_ret.reduced_value, expected.mean()))

    def test_zero_scales(self):
        scales = torch.tensor([[0.0, 0.0, 0.0]], device=self.device)
        model = _DummyModel(scales)
        loss = self._create_loss("fixed")
        loss_ret = self._run_loss(loss, model)
        self.assertFalse(torch.isnan(loss_ret.reduced_value))


# BackgroundInTrackGaussian forward test (only in-track gaussians should contribute)


class TestBackgroundInTrackGaussianLoss(unittest.TestCase):
    """Test BackgroundInTrackGaussianLoss behavior and numerical stability"""

    def _create_mock_data(self, positions: torch.Tensor, logits: torch.Tensor, intersection_mask: torch.Tensor):
        """Helper to create mock model, target, rigid_node and tracks_stub for testing"""

        class _BackgroundNode:
            def get_positions(self):
                return positions

            def get_densities(self, preactivation=None):
                return logits.unsqueeze(-1) if logits.dim() == 1 else logits

            def get_extra_signal_by_key(self, key):
                raise ValueError("no semantic_logits")

        background_node = _BackgroundNode()

        # Stub tracks
        class _TracksStub:
            def point_intersection(self, points, timestamps_us=None, return_dense_mask=False):
                return intersection_mask

        tracks_stub = _TracksStub()

        # Stub rigid node (only cuboid_tracks is used; isinstance is handled via module replacement below)
        class _RigidNodeStub:
            def __init__(self, tracks):
                self.cuboid_tracks = tracks

        rigid_node = _RigidNodeStub(tracks_stub)

        # Stub model
        class _ModelStub:
            def get_gaussians_node_ids(self):
                return ["rigid", "background"]

            def __init__(self, rigid, background):
                self.gaussians_nodes = {"rigid": rigid, "background": background}

        model = _ModelStub(rigid_node, background_node)

        # Target: need rendering.camera.timestamps_startend_us_cpu for median timestamp
        class _CameraStub:
            def __init__(self):
                self.timestamps_startend_us_cpu = torch.tensor([[0, 1_000_000]], dtype=torch.int64)

        class _TargetStub:
            def __init__(self, camera):
                self.rendering = type("_Rendering", (), {"camera": camera})()

        target = _TargetStub(_CameraStub())

        return model, target, rigid_node, tracks_stub

    @contextlib.contextmanager
    def _patch_render_losses(self, rigid_node_ref, tracks_stub):
        """Context manager to temporarily patch render_losses module"""

        class _RigidCheckMeta(type(RigidGaussianModel)):
            def __instancecheck__(cls, instance):
                if instance is rigid_node_ref:
                    return True
                return super().__instancecheck__(instance)

        fake_rigid_type = _RigidCheckMeta("RigidGaussianModel", (RigidGaussianModel,), {})

        class _CuboidTracksReplacement:
            class Ops:
                @staticmethod
                def concatenate(tracks_list):
                    return tracks_stub

        render_losses_mod = sys.modules["libs.losses.models.render_losses"]
        saved_rigid = render_losses_mod.RigidGaussianModel
        saved_cuboid = render_losses_mod.CuboidTracks

        render_losses_mod.RigidGaussianModel = fake_rigid_type
        render_losses_mod.CuboidTracks = _CuboidTracksReplacement

        try:
            yield
        finally:
            render_losses_mod.RigidGaussianModel = saved_rigid
            render_losses_mod.CuboidTracks = saved_cuboid

    def test_loss_unchanged_when_adding_out_of_track_gaussians(self):
        """With the fix, loss is mean over in-track only; adding out-of-track must not blow it up."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_in_track = 2
        n_out_of_track = 100
        n_total = n_in_track + n_out_of_track

        # Background: a few in-track + many out-of-track gaussians
        background_positions = torch.randn(n_total, 3, device=device)
        background_density_logits = torch.randn(n_total, 1, device=device)

        intersection_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
        intersection_mask[:n_in_track] = True

        config = LossItemConfig.model_validate(
            {
                "fn": "bce_with_logits",
                "lambda_": 0.1,
                "reduce": {"name": "mean"},
                "layer_names": ["background"],
                "density_logits_min": -20.0,
            }
        )
        trainer_config = MockTrainerConfig()
        loss = BackgroundInTrackGaussianLoss(config, trainer_config)

        model, target, rigid_node, tracks_stub = self._create_mock_data(
            background_positions, background_density_logits, intersection_mask
        )
        loss.supported_model_types = (type(model),)

        with self._patch_render_losses(rigid_node, tracks_stub):
            loss_ret = loss(results=None, target=target, model=model)

        self.assertIsNotNone(loss_ret)
        # With fix: mean over 2 in-track only -> small (BCE is typically < 1 per element)
        # Without fix: out-of-track add ~0.693 each -> 100 * 0.693 ~ 69
        self.assertLess(
            loss_ret.reduced_value.item(),
            10.0,
            "Loss should be mean over in-track gaussians only; large value suggests out-of-track are contributing.",
        )

    def test_numerical_stability_with_extreme_values(self):
        """Test forward/backward stability and gradient routing with extreme negative values"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = LossItemConfig.model_validate(
            {
                "fn": "bce_with_logits",
                "lambda_": 0.1,
                "reduce": {"name": "mean"},
                "layer_names": ["background"],
                "density_logits_min": -20.0,
            }
        )
        trainer_config = MockTrainerConfig()
        loss = BackgroundInTrackGaussianLoss(config, trainer_config)

        logits = torch.tensor(
            [
                0.5,
                -5.0,
                -80.0,
                -100.0,
                config.density_logits_min,
                config.density_logits_min - 0.1,
                config.density_logits_min + 0.1,
            ],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        positions = torch.zeros((logits.shape[0], 3), device=device)
        threshold = config.density_logits_min
        intersection_mask = torch.ones_like(logits, dtype=torch.bool, device=device)

        model, target, rigid_node, tracks_stub = self._create_mock_data(positions, logits, intersection_mask)
        loss.supported_model_types = (type(model),)

        with self._patch_render_losses(rigid_node, tracks_stub):
            loss_ret = loss(results=None, target=target, model=model)

        self.assertIsNotNone(loss_ret)

        loss_ret.reduced_value.backward()
        self.assertIsNotNone(logits.grad, "Gradients were not successfully backpropagated!")
        self.assertFalse(torch.isnan(logits.grad).any(), "Backward pass gradients produced NaN!")
        self.assertFalse(torch.isinf(logits.grad).any(), "Backward pass gradients produced Inf!")

        masked_out = logits <= threshold
        self.assertTrue(torch.all(logits.grad[masked_out] == 0.0), "Masked out points should not have gradients!")

        valid_points = ~masked_out
        self.assertTrue(
            torch.all(logits.grad[valid_points] != 0.0),
            f"Some valid points have vanishingly small gradients! Min grad: {logits.grad[valid_points].abs().min()}",
        )

    def test_all_points_transparent(self):
        """Boundary case: if all points are completely transparent (below threshold), it shouldn't crash"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = LossItemConfig.model_validate(
            {
                "fn": "bce_with_logits",
                "lambda_": 0.1,
                "reduce": {"name": "mean"},
                "layer_names": ["background"],
                "density_logits_min": -20.0,
            }
        )
        trainer_config = MockTrainerConfig()
        loss = BackgroundInTrackGaussianLoss(config, trainer_config)

        logits = torch.tensor(
            [-50.0, -100.0, config.density_logits_min, config.density_logits_min - 0.1],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        positions = torch.zeros((logits.shape[0], 3), device=device)
        intersection_mask = torch.ones_like(logits, dtype=torch.bool, device=device)

        model, target, rigid_node, tracks_stub = self._create_mock_data(positions, logits, intersection_mask)
        loss.supported_model_types = (type(model),)

        with self._patch_render_losses(rigid_node, tracks_stub):
            loss_ret = loss(results=None, target=target, model=model)

        self.assertIsNotNone(loss_ret)
        self.assertEqual(loss_ret.reduced_value.item(), 0.0)

        loss_ret.reduced_value.backward()
        self.assertTrue(torch.all(logits.grad == 0.0), "When all points are transparent, all gradients should be 0")

    def test_no_points_transparent(self):
        """Boundary case: if all points are opaque (above threshold), it should apply loss to all"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = LossItemConfig.model_validate(
            {
                "fn": "bce_with_logits",
                "lambda_": 0.1,
                "reduce": {"name": "mean"},
                "layer_names": ["background"],
                "density_logits_min": -20.0,
            }
        )
        trainer_config = MockTrainerConfig()
        loss = BackgroundInTrackGaussianLoss(config, trainer_config)

        # All values are in-track and above threshold
        logits = torch.tensor(
            [-5.0, 0.0, 20.0, config.density_logits_min + 0.1], dtype=torch.float32, device=device, requires_grad=True
        )
        positions = torch.zeros((logits.shape[0], 3), device=device)
        intersection_mask = torch.ones_like(logits, dtype=torch.bool, device=device)

        model, target, rigid_node, tracks_stub = self._create_mock_data(positions, logits, intersection_mask)
        loss.supported_model_types = (type(model),)

        with self._patch_render_losses(rigid_node, tracks_stub):
            loss_ret = loss(results=None, target=target, model=model)

        self.assertIsNotNone(loss_ret)
        self.assertGreater(loss_ret.reduced_value.item(), 0.0)

        loss_ret.reduced_value.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.all(logits.grad != 0.0), "When all points are opaque, all gradients should be non-zero")


# Semantic Label Validation Tests


class TestBackgroundInTrackGaussianLossLabelValidation(unittest.TestCase):
    """Test semantic label validation in BackgroundInTrackGaussianLoss.initialize()"""

    @staticmethod
    def _create_mock_ncore_dataset(semantic_classes_map: dict[str, int]) -> MagicMock:
        """Create a mock NCOREDataset with specified semantic_classes_map"""
        mock_dataset = MagicMock(spec=NCOREDataset)
        mock_datasource = MagicMock()
        mock_datasource.get_semantic_classes_map.return_value = semantic_classes_map
        mock_dataset.get_datasource.return_value = mock_datasource
        return mock_dataset

    def test_valid_labels_pass_validation(self):
        """Test that valid labels in layer_labels_to_use pass validation"""
        semantic_classes_map = {
            "person": 0,
            "rider": 1,
            "car": 2,
            "truck": 3,
            "bus": 4,
        }

        config = LossItemConfig.model_validate(
            {
                "fn": "bce_with_logits",
                "lambda_": 0.1,
                "reduce": {"name": "mean"},
                "layer_names": ["background"],
                "layer_labels_to_use": {
                    "background": ["person", "car", "truck"],
                },
            }
        )
        trainer_config = MockTrainerConfig()
        loss = BackgroundInTrackGaussianLoss(config, trainer_config)

        mock_dataset = self._create_mock_ncore_dataset(semantic_classes_map)

        # Should not raise
        loss.initialize(mock_dataset)

    def test_invalid_labels_raise_error(self):
        """Test that invalid labels in layer_labels_to_use raise ValueError"""
        semantic_classes_map = {
            "person": 0,
            "car": 2,
            "truck": 3,
        }

        config = LossItemConfig.model_validate(
            {
                "fn": "bce_with_logits",
                "lambda_": 0.1,
                "reduce": {"name": "mean"},
                "layer_names": ["background"],
                "layer_labels_to_use": {
                    "background": ["person", "car", "motorcycle", "bicycle"],  # motorcycle and bicycle invalid
                },
            }
        )
        trainer_config = MockTrainerConfig()
        loss = BackgroundInTrackGaussianLoss(config, trainer_config)

        mock_dataset = self._create_mock_ncore_dataset(semantic_classes_map)

        with self.assertRaises(ValueError) as cm:
            loss.initialize(mock_dataset)

        error_msg = str(cm.exception)
        self.assertIn("BackgroundInTrackGaussianLoss", error_msg)
        self.assertIn("layer_labels_to_use", error_msg)
        self.assertIn("background: motorcycle", error_msg)
        self.assertIn("background: bicycle", error_msg)
        self.assertIn("Available labels:", error_msg)

    def test_multiple_layers_with_mixed_validity(self):
        """Test validation with multiple layers where some have invalid labels"""
        semantic_classes_map = {
            "person": 0,
            "car": 2,
            "road": 10,
        }

        config = LossItemConfig.model_validate(
            {
                "fn": "bce_with_logits",
                "lambda_": 0.1,
                "reduce": {"name": "mean"},
                "layer_names": ["background", "foreground"],
                "layer_labels_to_use": {
                    "background": ["person", "car"],  # valid
                    "foreground": ["building", "tree"],  # invalid
                },
            }
        )
        trainer_config = MockTrainerConfig()
        loss = BackgroundInTrackGaussianLoss(config, trainer_config)

        mock_dataset = self._create_mock_ncore_dataset(semantic_classes_map)

        with self.assertRaises(ValueError) as cm:
            loss.initialize(mock_dataset)

        error_msg = str(cm.exception)
        self.assertIn("foreground: building", error_msg)
        self.assertIn("foreground: tree", error_msg)
        # Background layer should be validated successfully
        self.assertNotIn("background:", error_msg)

    def test_empty_layer_labels_to_use_skips_validation(self):
        """Test that empty layer_labels_to_use skips validation"""
        semantic_classes_map = {"person": 0, "car": 2}

        config = LossItemConfig.model_validate(
            {
                "fn": "bce_with_logits",
                "lambda_": 0.1,
                "reduce": {"name": "mean"},
                "layer_names": ["background"],
                "layer_labels_to_use": {},  # empty
            }
        )
        trainer_config = MockTrainerConfig()
        loss = BackgroundInTrackGaussianLoss(config, trainer_config)

        mock_dataset = self._create_mock_ncore_dataset(semantic_classes_map)

        # Should not raise even though semantic_classes_map exists
        loss.initialize(mock_dataset)

    def test_lambda_zero_skips_initialization(self):
        """Test that lambda_=0 skips initialization and validation"""
        config = LossItemConfig.model_validate(
            {
                "fn": "bce_with_logits",
                "lambda_": 0.0,  # zero lambda
                "reduce": {"name": "mean"},
                "layer_names": ["background"],
                "layer_labels_to_use": {
                    "background": ["invalid_label"],  # would be invalid but not checked
                },
            }
        )
        trainer_config = MockTrainerConfig()
        loss = BackgroundInTrackGaussianLoss(config, trainer_config)

        mock_dataset = self._create_mock_ncore_dataset({"person": 0})

        # Should not raise because lambda_=0 causes early return
        loss.initialize(mock_dataset)


class TestNodeSemanticGaussiansLossLabelValidation(unittest.TestCase):
    """Test semantic label validation in NodeSemanticGaussiansLoss.initialize()"""

    @staticmethod
    def _create_mock_ncore_dataset(semantic_classes_map: dict[str, int]) -> MagicMock:
        """Create a mock NCOREDataset with specified semantic_classes_map"""
        mock_dataset = MagicMock(spec=NCOREDataset)
        mock_datasource = MagicMock()
        mock_datasource.get_semantic_classes_map.return_value = semantic_classes_map
        mock_dataset.get_datasource.return_value = mock_datasource
        return mock_dataset

    def test_valid_labels_pass_validation(self):
        """Test that valid labels in both layer_labels_to_use and layer_labels_to_exclude pass validation"""
        semantic_classes_map = {
            "person": 0,
            "rider": 1,
            "car": 2,
            "road": 10,
        }

        config = LossItemConfig.model_validate(
            {
                "fn": "bce_with_logits",
                "lambda_": 0.1,
                "reduce": {"name": "mean"},
                "layer_labels_to_use": {
                    "dynamic": ["person", "car"],
                },
                "layer_labels_to_exclude": {
                    "static": ["rider", "road"],
                },
            }
        )
        trainer_config = MockTrainerConfig()
        loss = NodeSemanticGaussiansLoss(config, trainer_config)

        mock_dataset = self._create_mock_ncore_dataset(semantic_classes_map)

        # Should not raise
        loss.initialize(mock_dataset)

    def test_invalid_labels_raise_error(self):
        """Test that invalid labels in layer_labels_to_use or layer_labels_to_exclude raise ValueError"""
        semantic_classes_map = {
            "person": 0,
            "car": 2,
        }
        trainer_config = MockTrainerConfig()

        # Test layer_labels_to_use with invalid labels
        config_use = LossItemConfig.model_validate(
            {
                "fn": "bce_with_logits",
                "lambda_": 0.1,
                "reduce": {"name": "mean"},
                "layer_labels_to_use": {
                    "dynamic": ["person", "bicycle"],  # bicycle is invalid
                },
            }
        )
        loss_use = NodeSemanticGaussiansLoss(config_use, trainer_config)
        mock_dataset = self._create_mock_ncore_dataset(semantic_classes_map)

        with self.assertRaises(ValueError) as cm:
            loss_use.initialize(mock_dataset)

        error_msg = str(cm.exception)
        self.assertIn("NodeSemanticGaussiansLoss", error_msg)
        self.assertIn("dynamic: bicycle", error_msg)
        self.assertIn("Available labels:", error_msg)

        # Test layer_labels_to_exclude with invalid labels
        config_exclude = LossItemConfig.model_validate(
            {
                "fn": "bce_with_logits",
                "lambda_": 0.1,
                "reduce": {"name": "mean"},
                "layer_labels_to_exclude": {
                    "static": ["building", "tree"],  # both invalid
                },
            }
        )
        loss_exclude = NodeSemanticGaussiansLoss(config_exclude, trainer_config)

        with self.assertRaises(ValueError) as cm2:
            loss_exclude.initialize(mock_dataset)

        error_msg2 = str(cm2.exception)
        self.assertIn("NodeSemanticGaussiansLoss", error_msg2)
        self.assertIn("static: building", error_msg2)
        self.assertIn("static: tree", error_msg2)

    def test_mixed_valid_and_invalid_labels(self):
        """Test validation with both valid and invalid labels across use and exclude"""
        semantic_classes_map = {
            "person": 0,
            "car": 2,
            "road": 10,
        }

        config = LossItemConfig.model_validate(
            {
                "fn": "bce_with_logits",
                "lambda_": 0.1,
                "reduce": {"name": "mean"},
                "layer_labels_to_use": {
                    "dynamic": ["person", "bicycle"],  # bicycle invalid
                },
                "layer_labels_to_exclude": {
                    "static": ["car", "building"],  # building invalid
                },
            }
        )
        trainer_config = MockTrainerConfig()
        loss = NodeSemanticGaussiansLoss(config, trainer_config)

        mock_dataset = self._create_mock_ncore_dataset(semantic_classes_map)

        with self.assertRaises(ValueError) as cm:
            loss.initialize(mock_dataset)

        error_msg = str(cm.exception)
        self.assertIn("dynamic: bicycle", error_msg)
        self.assertIn("static: building", error_msg)
        # Valid labels should not appear in the invalid labels list (before "Available labels:")
        error_prefix = error_msg.split("Available labels:")[0]
        # "person" and "car" are valid so they should only appear after "Available labels:"
        self.assertNotIn("dynamic: person", error_prefix)
        self.assertNotIn("static: car", error_prefix)

    def test_lambda_zero_skips_initialization(self):
        """Test that lambda_=0 skips initialization and validation"""
        config = LossItemConfig.model_validate(
            {
                "fn": "bce_with_logits",
                "lambda_": 0.0,  # zero lambda
                "reduce": {"name": "mean"},
                "layer_labels_to_use": {
                    "dynamic": ["invalid_label"],  # would be invalid but not checked
                },
            }
        )
        trainer_config = MockTrainerConfig()
        loss = NodeSemanticGaussiansLoss(config, trainer_config)

        mock_dataset = self._create_mock_ncore_dataset({"person": 0})

        # Should not raise because lambda_=0 causes early return
        loss.initialize(mock_dataset)
