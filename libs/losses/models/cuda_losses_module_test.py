# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import unittest

import omegaconf
import torch

from libs.losses.models.render_losses import RoadGaussiansLoss
from libs.losses.orchestration.config import LossItemConfig
from nre.config.trainer import TrainerConfig


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


# Road Gaussians Tests


def create_road_gaussians_test_data(device, n_points=100, seed=42):
    """Create reproducible test data for road gaussians tests."""
    torch.manual_seed(seed)

    positions_world = torch.randn(n_points, 3, device=device, dtype=torch.float32)
    positions_world[:, 2] = torch.abs(positions_world[:, 2]) * 5  # Ensure positive z

    rotations_world = torch.randn(n_points, 4, device=device, dtype=torch.float32)

    # Identity transformation matrices
    pose_tquat = torch.randn(7, device=device, dtype=torch.float32)
    pose_tquat[3:] = pose_tquat[3:] / torch.norm(pose_tquat[3:], dim=0, keepdim=True)

    return positions_world, rotations_world, pose_tquat


def create_road_gaussians_loss_config():
    """Create standard loss configuration for road gaussians tests."""
    return LossItemConfig.model_validate(
        {
            "fn": "mse",
            "lambda_": 1.0,
            "reduce": {"name": "mean"},
            "layer_name": "road",
            "n_samples": 3,
            "grid_len": 0.5,
            "min": -5.0,
            "range": 4.0,
            "rotation_lambda": 10.0,
        }
    )


class TestRoadGaussians(unittest.TestCase):
    def test_road_gaussians_direct_vs_two_step_approach(self):
        """Test that forward_direct gives identical results to positions_and_rotations_cam_from_world + forward_direct_cam."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        device = torch.device("cuda")
        positions_world, rotations_world, pose_tquat = create_road_gaussians_test_data(device)
        config = create_road_gaussians_loss_config()
        trainer_config = MockTrainerConfig()

        loss_python = RoadGaussiansLoss(config, trainer_config, use_cuda=False)

        # Set same random seed for both approaches
        random_values = torch.randn(config.n_samples, device=device, dtype=torch.float32)

        # Approach 1: Direct method
        torch.manual_seed(12345)
        direct_result = loss_python.forward_direct(
            positions_world, rotations_world, pose_tquat, random_values=random_values
        )

        # Approach 2: Two-step method
        torch.manual_seed(12345)
        positions_cam, rotations_cam = loss_python.positions_and_rotations_cam_from_world(
            positions_world, rotations_world, pose_tquat
        )
        two_step_result = loss_python.forward_direct_cam(positions_cam, rotations_cam, random_values=random_values)

        print(f"Direct method loss: {direct_result.item():.8f}")
        print(f"Two-step method loss: {two_step_result.item():.8f}")
        print(f"Difference: {abs(direct_result.item() - two_step_result.item()):.8f}")

        self.assertTrue(
            torch.allclose(direct_result, two_step_result, rtol=1e-6, atol=1e-8),
            f"Forward pass mismatch: Direct={direct_result.item()}, Two-step={two_step_result.item()}",
        )

    def test_road_gaussians_direct_vs_two_step_backward(self):
        """Test that gradients are identical between direct and two-step approaches."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        device = torch.device("cuda")
        positions_world_1, rotations_world_1, pose_tquat = create_road_gaussians_test_data(device)
        positions_world_2, rotations_world_2 = positions_world_1.clone(), rotations_world_1.clone()
        config = create_road_gaussians_loss_config()
        trainer_config = MockTrainerConfig()

        loss_python = RoadGaussiansLoss(config, trainer_config, use_cuda=False)
        random_values = torch.randn(config.n_samples, device=device, dtype=torch.float32)

        # Enable gradients for both sets
        positions_world_1.requires_grad_(True)
        rotations_world_1.requires_grad_(True)
        positions_world_2.requires_grad_(True)
        rotations_world_2.requires_grad_(True)

        # Approach 1: Direct method
        direct_loss = loss_python.forward_direct(
            positions_world_1, rotations_world_1, pose_tquat, random_values=random_values
        )
        direct_loss.backward()
        direct_pos_grad = positions_world_1.grad.clone()
        direct_rot_grad = rotations_world_1.grad.clone()

        # Approach 2: Two-step method (need to handle chain rule manually)
        positions_cam, rotations_cam = loss_python.positions_and_rotations_cam_from_world(
            positions_world_2, rotations_world_2, pose_tquat
        )
        two_step_loss = loss_python.forward_direct_cam(positions_cam, rotations_cam, random_values=random_values)
        two_step_loss.backward()
        two_step_pos_grad = positions_world_2.grad.clone()
        two_step_rot_grad = rotations_world_2.grad.clone()

        # Compare gradients
        pos_max_rel_diff = (
            (torch.abs(direct_pos_grad - two_step_pos_grad) / (torch.abs(direct_pos_grad) + 1e-8)).max().item()
        )
        rot_max_rel_diff = (
            (torch.abs(direct_rot_grad - two_step_rot_grad) / (torch.abs(direct_rot_grad) + 1e-8)).max().item()
        )

        print(f"Position gradient max relative diff: {pos_max_rel_diff:.8f}")
        print(f"Rotation gradient max relative diff: {rot_max_rel_diff:.8f}")

        self.assertTrue(
            torch.allclose(direct_pos_grad, two_step_pos_grad, rtol=1e-4, atol=1e-6),
            f"Position gradients don't match: max_rel_diff={pos_max_rel_diff:.6f}",
        )
        self.assertTrue(
            torch.allclose(direct_rot_grad, two_step_rot_grad, rtol=1e-4, atol=1e-6),
            f"Rotation gradients don't match: max_rel_diff={rot_max_rel_diff:.6f}",
        )

    def test_road_gaussians_forward_pass(self):
        """Test that forward pass gives identical results between Python and CUDA."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        device = torch.device("cuda")
        positions_world, rotations_world, pose_tquat = create_road_gaussians_test_data(device)
        config = create_road_gaussians_loss_config()
        trainer_config = MockTrainerConfig()

        loss_python = RoadGaussiansLoss(config, trainer_config, use_cuda=False)
        loss_cuda = RoadGaussiansLoss(config, trainer_config, use_cuda=True)

        # Set same random seed for both implementations
        torch.manual_seed(12345)
        python_result = loss_python.forward_direct(positions_world, rotations_world, pose_tquat)

        torch.manual_seed(12345)
        cuda_result = loss_cuda.forward_direct(positions_world, rotations_world, pose_tquat)

        print(f"Python loss: {python_result.item():.8f}")
        print(f"CUDA loss: {cuda_result.item():.8f}")
        print(f"Difference: {abs(python_result.item() - cuda_result.item()):.8f}")

        self.assertTrue(
            torch.allclose(python_result, cuda_result, rtol=1e-6, atol=1e-8),
            f"Forward pass mismatch: Python={python_result.item()}, CUDA={cuda_result.item()}",
        )

    def test_road_gaussians_backward_pass(self):
        """Test that backward pass gives identical gradients between Python and CUDA."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        device = torch.device("cuda")
        positions_world, rotations_world, pose_tquat = create_road_gaussians_test_data(device)
        config = create_road_gaussians_loss_config()
        trainer_config = MockTrainerConfig()

        # Enable gradients
        positions_world.requires_grad_(True)
        rotations_world.requires_grad_(True)

        # Python implementation
        loss_python = RoadGaussiansLoss(config, trainer_config, use_cuda=False)
        random_values = torch.randn(config.n_samples, device=device, dtype=torch.float32)
        python_loss = loss_python.forward_direct(
            positions_world, rotations_world, pose_tquat, random_values=random_values
        )

        python_loss.backward()
        python_pos_grad = positions_world.grad.clone()
        python_rot_grad = rotations_world.grad.clone()
        positions_world.grad.zero_()
        rotations_world.grad.zero_()

        # CUDA implementation
        loss_cuda = RoadGaussiansLoss(config, trainer_config, use_cuda=True)
        cuda_loss = loss_cuda.forward_direct(positions_world, rotations_world, pose_tquat, random_values=random_values)
        cuda_loss.backward()
        cuda_pos_grad = positions_world.grad.clone()
        cuda_rot_grad = rotations_world.grad.clone()

        # Compare gradients
        pos_max_rel_diff = (
            (torch.abs(python_pos_grad - cuda_pos_grad) / (torch.abs(python_pos_grad) + 1e-8)).max().item()
        )
        rot_max_rel_diff = (
            (torch.abs(python_rot_grad - cuda_rot_grad) / (torch.abs(python_rot_grad) + 1e-8)).max().item()
        )

        print(
            f"Position gradient max relative diff: {pos_max_rel_diff:.8f}, CUDA={cuda_pos_grad.max().item():.8f}, Python={python_pos_grad.max().item():.8f}"
        )
        print(
            f"Rotation gradient max relative diff: {rot_max_rel_diff:.8f}, CUDA={cuda_rot_grad.max().item():.8f}, Python={python_rot_grad.max().item():.8f}"
        )

        self.assertTrue(
            torch.allclose(python_pos_grad, cuda_pos_grad, rtol=1e-4, atol=1e-6),
            f"Position gradients don't match: max_rel_diff={pos_max_rel_diff:.6f}",
        )
        self.assertTrue(
            torch.allclose(python_rot_grad, cuda_rot_grad, rtol=1e-4, atol=1e-6),
            f"Rotation gradients don't match: max_rel_diff={rot_max_rel_diff:.6f}",
        )
