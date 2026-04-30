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

import torch
import torch.nn as nn

from fused_ssim import fused_ssim
from torch.nn import functional as F

from libs.losses.models.loss_fns import total_variation_spatial
from libs.losses.models.utils import create_window, get_mask_semantic, torch_ssim
from nre.utils.batch import CameraFrameLabels
from nre.utils.types import RayFlags


class TestTotalVariation(unittest.TestCase):
    def setUp(self):
        self.batch_size = 5
        self.num_channels = 7
        self.xyz_size = 3
        self.shape = (self.batch_size, self.num_channels, self.xyz_size, self.xyz_size, self.xyz_size)

    def test_shape(self):
        x = torch.rand(self.shape)
        tv_loss = total_variation_spatial(x)
        assert tv_loss.shape == (self.batch_size,)

    def test_constant(self):
        x = torch.ones(self.shape)
        tv_loss = torch.mean(total_variation_spatial(x))
        assert tv_loss.item() == 0.0

    def test_size_one(self):
        x = torch.rand(self.batch_size, self.num_channels, 1, 1, 1)
        tv_loss = torch.mean(total_variation_spatial(x))
        assert tv_loss.item() == 0.0

    def test_random_tensor(self):
        x = torch.rand(self.shape)
        tv_loss = torch.mean(total_variation_spatial(x))
        assert tv_loss.item() >= 0.0

    def test_manual_variation(self):
        self.xyz_size = 2
        x = torch.zeros(self.batch_size, self.num_channels, self.xyz_size, self.xyz_size, self.xyz_size)
        x[:, :, 0, 0, 0] = 1.0
        x[:, :, 1, 1, 1] = 1.0

        # Manually calculate the expected total variation.
        # Since the "1"s are diagonally opposite of each other, the TV will accumulate in all three
        # dimensions, by two units in each dimension.
        # Note that both the number of channels and batch size cancel out in the normalized result.
        number_of_deltas_per_dim = (self.xyz_size - 1) * self.xyz_size * self.xyz_size
        expected_tv = torch.tensor([3 * 2 / number_of_deltas_per_dim])

        tv_loss = total_variation_spatial(x)
        print(tv_loss)
        for row in tv_loss:
            assert torch.isclose(row, expected_tv)


class TestSSIM(unittest.TestCase):
    def setUp(self):
        # Create two batches of images with the same shape over which the SSIM loss will be computed
        B, C, H, W = 10, 3, 1080, 1920
        self.img1_reference = nn.Parameter(torch.rand([B, C, H, W], device="cuda"))
        self.img2_reference = torch.rand([B, C, H, W], device="cuda")

        self.img1_fused = nn.Parameter(self.img1_reference.clone())
        self.img2_fused = self.img2_reference.clone()

    def test_fused_ssim(self):
        # Compute the fused-SSIM loss using the reference implementation
        window = create_window(11, 3)
        window = window.type_as(self.img1_reference)

        reference_ssim = torch_ssim(self.img1_reference, self.img2_reference, window=window, window_size=11, channel=3)
        # Note: only the first input to the fused_ssim function is differentiable (different to torch)
        fused_ssim_val_same = fused_ssim(self.img1_fused, self.img2_fused)

        assert torch.isclose(reference_ssim, fused_ssim_val_same, atol=1e-4, rtol=1e-4).all()
        assert torch.isclose(reference_ssim.mean(), fused_ssim_val_same.mean())

        # Compute the backward pass and compare the gradients
        reference_ssim.mean().backward()
        fused_ssim_val_same.mean().backward()

        assert torch.isclose(self.img1_reference.grad, self.img1_fused.grad).all()


class TestMaskSemantic(unittest.TestCase):
    def setUp(self):
        # Create a CameraFrameLabels object with some flags set
        BATCH_SIZE, WIDTH, HEIGHT = 2, 64, 64

        # Set some random synthetic flags
        self.is_sky = torch.rand(BATCH_SIZE, WIDTH, HEIGHT, 1) < 0.5
        self.is_road = torch.rand(BATCH_SIZE, WIDTH, HEIGHT, 1) < 0.5
        self.is_vehicle = torch.rand(BATCH_SIZE, WIDTH, HEIGHT, 1) < 0.5
        flags = torch.zeros(BATCH_SIZE, WIDTH, HEIGHT, 1, dtype=torch.int32)
        flags[self.is_sky] |= RayFlags.SKY_SEMANTIC
        flags[self.is_road] |= RayFlags.ROAD_SEMANTIC
        flags[self.is_vehicle] |= RayFlags.VEHICLE_SEMANTIC

        # Build the CameraFrameLabels object
        self.rays_meta = CameraFrameLabels(flags=flags)

    def test_single_operation(self):
        # Test that the mask semantic function returns the correct mask
        sky_semantic = get_mask_semantic(self.rays_meta, "sky")
        road_semantic = get_mask_semantic(self.rays_meta, "road")
        vehicle_semantic = get_mask_semantic(self.rays_meta, "vehicle")

        assert (sky_semantic == self.is_sky).all()
        assert (road_semantic == self.is_road).all()
        assert (vehicle_semantic == self.is_vehicle).all()

    def test_compound_operation(self):
        mask_semantic = get_mask_semantic(self.rays_meta, "sky | road")
        assert (mask_semantic == (self.is_sky | self.is_road)).all()

        mask_semantic = get_mask_semantic(self.rays_meta, "sky & road")
        assert (mask_semantic == (self.is_sky & self.is_road)).all()

        mask_semantic = get_mask_semantic(self.rays_meta, "sky & (~road | vehicle)")
        assert (mask_semantic == (self.is_sky & (~self.is_road | self.is_vehicle))).all()
