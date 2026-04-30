# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import unittest

import torch

from omegaconf import DictConfig

from libs.gaussian_mcmc.interface import gaussian_mcmc  # type: ignore
from libs.slang_gaussians.interface import mcmc_slang  # type: ignore
from libs.slang_utils.utils import div_up  # type: ignore
from nre.config.model import MCMCStrategyConfig
from nre.models.gaussians.strategies.mcmc import MCMCStrategy
from nre.models.gaussians.strategies.test_utils import make_trainer_cfg
from nre.models.nn_extensions import TypedModuleDict


def make_gaussian_cfg() -> DictConfig:
    return DictConfig(
        {
            "name": "mcmc",
            "relocate": {"start_iteration": 1, "end_iteration": 1, "frequency": 1},
            "add": {"start_iteration": 1, "end_iteration": 1, "frequency": 1, "max_n_gaussians": 100},
            "perturb": {"start_iteration": 1, "end_iteration": 30_000, "frequency": 1, "noise_lr": {"default": 1.0}},
            "binom_n_max": 51,
            "opacity_threshold": 0.005,
            "exclude_layer_ids": [],
        }
    )


class TestMCMCStrategy(unittest.TestCase):
    def test_mcmc_strategy_instantiation(self):
        strategy = MCMCStrategy(
            config=MCMCStrategyConfig.model_validate(make_gaussian_cfg()),
            trainer_config=make_trainer_cfg(),
            init_from_datasource=False,
            gaussians_nodes=TypedModuleDict(),
        )
        self.assertIsInstance(strategy, MCMCStrategy)


def op_sigmoid(x: torch.Tensor, k: int = 100, x0: float = 0.995) -> torch.Tensor:
    """Steep sigmoid function used in MCMC perturbation."""
    return 1 / (1 + torch.exp(-k * (x - x0)))


def pytorch_perturb_gaussians(
    positions: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    densities: torch.Tensor,
    noise: torch.Tensor,
    current_lr: float,
    quaternion_format: str = "wxyz",
) -> torch.Tensor:
    """
    PyTorch reference implementation of perturb_gaussians.

    Takes PRE-ACTIVATION values (raw quats, scales before exp, densities before sigmoid)
    and computes the position perturbation.
    """
    # Apply activations
    activated_quats = torch.nn.functional.normalize(quats, dim=1)
    activated_scales = torch.exp(scales)
    activated_densities = torch.sigmoid(densities)

    # Compute covariance using the CUDA kernel
    covariance = gaussian_mcmc.quat_scale_to_covariance(activated_quats, activated_scales, quaternion_format)

    # Scale noise by steep sigmoid of (1 - density) and learning rate
    scaled_noise = noise * op_sigmoid(1 - activated_densities).unsqueeze(-1) * current_lr

    # Transform noise by covariance
    transformed_noise = torch.bmm(covariance, scaled_noise.unsqueeze(-1)).squeeze(-1)

    return positions + transformed_noise


def pytorch_perturb_gaussians_rigid(
    positions: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    densities: torch.Tensor,
    cuboid_dims: torch.Tensor,
    noise: torch.Tensor,
    current_lr: float,
    quaternion_format: str = "wxyz",
) -> torch.Tensor:
    """
    PyTorch reference implementation of perturb_gaussians with cuboid constraint.
    """
    # Apply activations
    activated_quats = torch.nn.functional.normalize(quats, dim=1)
    activated_scales = torch.exp(scales)
    activated_densities = torch.sigmoid(densities)

    # Compute covariance
    covariance = gaussian_mcmc.quat_scale_to_covariance(activated_quats, activated_scales, quaternion_format)

    # Scale noise by steep sigmoid of (1 - density) and learning rate
    scaled_noise = noise * op_sigmoid(1 - activated_densities).unsqueeze(-1) * current_lr

    # Transform noise by covariance
    transformed_noise = torch.bmm(covariance, scaled_noise.unsqueeze(-1)).squeeze(-1)

    # Apply cuboid constraint
    bounds = cuboid_dims / 2
    cur_pos_dist = positions.abs() - bounds
    pos_with_noise = positions + transformed_noise
    pos_with_noise_dist = pos_with_noise.abs() - bounds

    inside_to_outside_mask = torch.logical_and(cur_pos_dist < 0, pos_with_noise_dist >= 0)
    outside_to_more_outside_mask = torch.logical_and(cur_pos_dist >= 0, pos_with_noise_dist > cur_pos_dist)
    constrained_noise = torch.where(
        torch.logical_or(inside_to_outside_mask, outside_to_more_outside_mask),
        torch.zeros_like(transformed_noise),
        transformed_noise,
    )

    return positions + constrained_noise


class TestMCMCSlangVsPyTorch(unittest.TestCase):
    """Test that Slang MCMC kernels match PyTorch reference implementation."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda")
        self.num_gaussians = 10000

        # Generate random test data with PRE-ACTIVATION values
        self.positions = torch.randn(self.num_gaussians, 3, device=self.device, dtype=torch.float32)
        self.quats = torch.randn(self.num_gaussians, 4, device=self.device, dtype=torch.float32)
        self.scales = torch.randn(self.num_gaussians, 3, device=self.device, dtype=torch.float32)
        self.densities = torch.randn(self.num_gaussians, device=self.device, dtype=torch.float32)
        self.noise = torch.randn(self.num_gaussians, 3, device=self.device, dtype=torch.float32)
        self.cuboid_dims = torch.rand(self.num_gaussians, 3, device=self.device, dtype=torch.float32) * 2 + 0.5
        self.current_lr = 0.001

    def test_fused_perturb_gaussians_wxyz(self):
        """Test fused_perturb_gaussians with wxyz quaternion format."""
        positions_slang = self.positions.clone()

        positions_pytorch = pytorch_perturb_gaussians(
            positions=self.positions.clone(),
            quats=self.quats,
            scales=self.scales,
            densities=self.densities,
            noise=self.noise,
            current_lr=self.current_lr,
            quaternion_format="wxyz",
        )

        n_gaussians = self.num_gaussians
        threads_per_block = 256
        blocks = div_up(n_gaussians, threads_per_block)

        mcmc_slang.fused_perturb_gaussians_kernel(
            (threads_per_block, 1, 1),
            (blocks, 1, 1),
            n_gaussians,
            positions_slang.contiguous(),
            self.quats.contiguous(),
            self.scales.contiguous(),
            self.densities.contiguous(),
            self.noise.contiguous(),
            self.current_lr,
            True,  # wxyz_format
        )
        torch.cuda.synchronize()

        self.assertTrue(
            torch.allclose(positions_slang, positions_pytorch, atol=1e-5, rtol=1e-5),
            f"Max diff: {(positions_slang - positions_pytorch).abs().max().item()}",
        )

    def test_fused_perturb_gaussians_xyzw(self):
        """Test fused_perturb_gaussians with xyzw quaternion format."""
        positions_slang = self.positions.clone()

        positions_pytorch = pytorch_perturb_gaussians(
            positions=self.positions.clone(),
            quats=self.quats,
            scales=self.scales,
            densities=self.densities,
            noise=self.noise,
            current_lr=self.current_lr,
            quaternion_format="xyzw",
        )

        n_gaussians = self.num_gaussians
        threads_per_block = 256
        blocks = div_up(n_gaussians, threads_per_block)

        mcmc_slang.fused_perturb_gaussians_kernel(
            (threads_per_block, 1, 1),
            (blocks, 1, 1),
            n_gaussians,
            positions_slang.contiguous(),
            self.quats.contiguous(),
            self.scales.contiguous(),
            self.densities.contiguous(),
            self.noise.contiguous(),
            self.current_lr,
            False,  # wxyz_format = False means xyzw
        )
        torch.cuda.synchronize()

        self.assertTrue(
            torch.allclose(positions_slang, positions_pytorch, atol=1e-5, rtol=1e-5),
            f"Max diff: {(positions_slang - positions_pytorch).abs().max().item()}",
        )

    def test_fused_perturb_gaussians_rigid_wxyz(self):
        """Test fused_perturb_gaussians_rigid with wxyz quaternion format."""
        positions_slang = self.positions.clone()

        positions_pytorch = pytorch_perturb_gaussians_rigid(
            positions=self.positions.clone(),
            quats=self.quats,
            scales=self.scales,
            densities=self.densities,
            cuboid_dims=self.cuboid_dims,
            noise=self.noise,
            current_lr=self.current_lr,
            quaternion_format="wxyz",
        )

        n_gaussians = self.num_gaussians
        threads_per_block = 256
        blocks = div_up(n_gaussians, threads_per_block)

        mcmc_slang.fused_perturb_gaussians_rigid_kernel(
            (threads_per_block, 1, 1),
            (blocks, 1, 1),
            n_gaussians,
            positions_slang.contiguous(),
            self.quats.contiguous(),
            self.scales.contiguous(),
            self.densities.contiguous(),
            self.noise.contiguous(),
            self.cuboid_dims.contiguous(),
            self.current_lr,
            True,  # wxyz_format
        )
        torch.cuda.synchronize()

        self.assertTrue(
            torch.allclose(positions_slang, positions_pytorch, atol=1e-5, rtol=1e-5),
            f"Max diff: {(positions_slang - positions_pytorch).abs().max().item()}",
        )

    def test_fused_perturb_gaussians_rigid_xyzw(self):
        """Test fused_perturb_gaussians_rigid with xyzw quaternion format."""
        positions_slang = self.positions.clone()

        positions_pytorch = pytorch_perturb_gaussians_rigid(
            positions=self.positions.clone(),
            quats=self.quats,
            scales=self.scales,
            densities=self.densities,
            cuboid_dims=self.cuboid_dims,
            noise=self.noise,
            current_lr=self.current_lr,
            quaternion_format="xyzw",
        )

        n_gaussians = self.num_gaussians
        threads_per_block = 256
        blocks = div_up(n_gaussians, threads_per_block)

        mcmc_slang.fused_perturb_gaussians_rigid_kernel(
            (threads_per_block, 1, 1),
            (blocks, 1, 1),
            n_gaussians,
            positions_slang.contiguous(),
            self.quats.contiguous(),
            self.scales.contiguous(),
            self.densities.contiguous(),
            self.noise.contiguous(),
            self.cuboid_dims.contiguous(),
            self.current_lr,
            False,  # wxyz_format = False means xyzw
        )
        torch.cuda.synchronize()

        self.assertTrue(
            torch.allclose(positions_slang, positions_pytorch, atol=1e-5, rtol=1e-5),
            f"Max diff: {(positions_slang - positions_pytorch).abs().max().item()}",
        )
