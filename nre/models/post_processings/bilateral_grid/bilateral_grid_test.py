# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import torch.nn.functional as F
import torch.optim as optim

from libs.losses.kernel.constants import GRID_NUM_CHANNELS, GRID_NUM_COLS, GRID_NUM_ROWS
from libs.losses.models.loss_fns import total_variation_spatial
from nre.models.post_processings.bilateral_grid.bilateral_grid import BilateralGrid


class TestBilateralGrid(unittest.TestCase):
    def setUp(self):
        # Create a test collection of two bilateral grids.
        self.device = "cuda"
        self.num_grids = 2
        self.width = 3
        self.height = 2
        self.depth = 2

        self.bilagrid = BilateralGrid(self.num_grids, self.width, self.height, self.depth).to(self.device)

        # Set up common test data for test_forward_* tests.
        self.batch_size = 5
        self.coords_xy = torch.rand(self.batch_size, 2)
        self.rgb = torch.rand(self.batch_size, 3)
        self.grid_idcs = torch.randint(0, self.num_grids, (self.batch_size,))

    def assertTensorClose(self, actual, expected, atol=1e-4):
        actual = actual.to(expected.device)
        self.assertTrue(torch.allclose(actual, expected, atol=atol))

    def test_setup(self):
        assert self.bilagrid.num_grids == self.num_grids
        assert self.bilagrid.width == self.width
        assert self.bilagrid.height == self.height
        assert self.bilagrid.depth == self.depth
        assert self.bilagrid.grid.shape == (self.num_grids, GRID_NUM_CHANNELS, self.depth, self.height, self.width)
        # Test initialization to identity
        self.assertTensorClose(
            self.bilagrid.grid[0, :, 0, 0, 0].view(GRID_NUM_ROWS, GRID_NUM_COLS),
            torch.eye(GRID_NUM_ROWS, GRID_NUM_COLS),
        )

    # Current guidance implementation is a simple luma conversion with these weights.
    LUMA_WEIGHTS_BT_601 = torch.tensor([0.299, 0.587, 0.114])

    def test_shape_and_rescaling(self):
        # These tests ensure interoperability with grid_sample.
        # Independent of color space.
        rgb_input = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        actual_output = self.bilagrid.guidance(rgb_input)
        self.assertTrue(actual_output.shape == (2, 1))
        expected_output = torch.tensor([[-1.0], [1.0]])
        self.assertTensorClose(actual_output, expected_output)

    def test_luma_calculation(self):
        # Independent of color space.
        rgb_input = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        expected_output = (
            torch.tensor([[self.LUMA_WEIGHTS_BT_601[0]], [self.LUMA_WEIGHTS_BT_601[1]], [self.LUMA_WEIGHTS_BT_601[2]]])
            * 2
            - 1
        )
        self.assertTensorClose(self.bilagrid.guidance(rgb_input), expected_output)

    def test_linear_conversion(self):
        rgb_input = torch.tensor([[0.5, 0.5, 0.5]])
        expected_output = torch.tensor([[0.0]])
        self.assertTensorClose(self.bilagrid.guidance(rgb_input), expected_output)

    def test_slice_shape(self):
        batch_size = 5
        coords_xy = torch.rand(batch_size, 2) * 2 - 1  # Random coordinates in [-1, 1]
        rgb = torch.rand(batch_size, 3)
        grid_idcs = torch.randint(0, self.num_grids, (batch_size,))
        output = self.bilagrid.slice_grid(coords_xy, rgb, grid_idcs)
        self.assertEqual(output.shape, (batch_size, 12))

    def test_slice_interpolation(self):
        # Set up test grid with known values
        self.bilagrid.grid.data.fill_(0.0)
        self.bilagrid.grid.data[0, :, 0, 0, 0] = torch.arange(12)
        self.bilagrid.grid.data[0, :, 0, 0, 1] = torch.arange(12, 24)
        self.bilagrid.grid.data[0, :, 0, 0, 2] = torch.arange(24, 36)
        self.bilagrid.grid.data[0, :, 0, 1, 0] = torch.arange(36, 48)
        self.bilagrid.grid.data[0, :, 0, 1, 1] = torch.arange(48, 60)
        self.bilagrid.grid.data[0, :, 0, 1, 2] = torch.arange(60, 72)
        self.bilagrid.grid.data[0, :, 1, 0, 0] = torch.arange(72, 84)
        self.bilagrid.grid.data[0, :, 1, 0, 1] = torch.arange(84, 96)
        self.bilagrid.grid.data[0, :, 1, 0, 2] = torch.arange(96, 108)
        self.bilagrid.grid.data[0, :, 1, 1, 0] = torch.arange(108, 120)
        self.bilagrid.grid.data[0, :, 1, 1, 1] = torch.arange(120, 132)
        self.bilagrid.grid.data[0, :, 1, 1, 2] = torch.arange(132, 144)

        self.bilagrid.grid.data[1, :, 0, 0, 0] = torch.arange(144, 156)
        self.bilagrid.grid.data[1, :, 0, 0, 1] = torch.arange(156, 168)
        self.bilagrid.grid.data[1, :, 0, 0, 2] = torch.arange(168, 180)
        self.bilagrid.grid.data[1, :, 0, 1, 0] = torch.arange(180, 192)
        self.bilagrid.grid.data[1, :, 0, 1, 1] = torch.arange(192, 204)
        self.bilagrid.grid.data[1, :, 0, 1, 2] = torch.arange(204, 216)
        self.bilagrid.grid.data[1, :, 1, 0, 0] = torch.arange(216, 228)
        self.bilagrid.grid.data[1, :, 1, 0, 1] = torch.arange(228, 240)
        self.bilagrid.grid.data[1, :, 1, 0, 2] = torch.arange(240, 252)
        self.bilagrid.grid.data[1, :, 1, 1, 0] = torch.arange(252, 264)
        self.bilagrid.grid.data[1, :, 1, 1, 1] = torch.arange(264, 276)
        self.bilagrid.grid.data[1, :, 1, 1, 2] = torch.arange(276, 288)

        # Test slicing at grid corners: No interpolation should happen.
        coords_xy = torch.tensor([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])
        rgb = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
        grid_idcs = torch.tensor([1, 0, 1, 0])
        # We should get back the values at the following indices, one sample per line:
        # z, y, x = [0, 0, 0], grid 1
        # z, y, x = [1, 0, 2], grid 0
        # z, y, x = [1, 1, 0], grid 1
        # z, y, x = [0, 1, 2], grid 0
        expected_output = torch.stack(
            [
                torch.arange(144, 156).float(),
                torch.arange(96, 108).float(),
                torch.arange(252, 264).float(),
                torch.arange(60, 72).float(),
            ]
        )
        self.assertTensorClose(self.bilagrid.slice_grid(coords_xy, rgb, grid_idcs), expected_output)

        # Test interpolation at grid center.
        # Since the width is 3, we expect no interpolation in x.
        coords_xy = torch.tensor([[0.0, 0.0]])
        rgb = torch.tensor([[0.5, 0.5, 0.5]])
        grid_idcs = torch.tensor([0])
        # Average of the following:
        # z, y, x = [0, 0, 1], grid 0
        # z, y, x = [0, 1, 1], grid 0
        # z, y, x = [1, 0, 1], grid 0
        # z, y, x = [1, 1, 1], grid 0
        expected_output = (
            torch.arange(12, 24) + torch.arange(48, 60) + torch.arange(84, 96) + torch.arange(120, 132)
        ) / 4.0
        self.assertTensorClose(self.bilagrid.slice_grid(coords_xy, rgb, grid_idcs), expected_output)

    def test_forward_output_shape(self):
        output = self.bilagrid.forward_coords(self.rgb, self.coords_xy, self.grid_idcs)
        self.assertEqual(output.shape, (self.batch_size, 3))

    def test_forward_identity_transformation(self):
        # The grids are initialized by default to identity transforms.
        output = self.bilagrid.forward_coords(self.rgb, self.coords_xy, self.grid_idcs)
        self.assertTensorClose(output, self.rgb)

    def test_forward_scale_transformation(self):
        # Set the grid to represent a scaling transformation
        scale_factor = 1.0 / 3.141
        self.bilagrid.grid.data.fill_(0.0)
        self.bilagrid.grid.data[:, 0, :, :, :] = scale_factor
        self.bilagrid.grid.data[:, 5, :, :, :] = scale_factor
        self.bilagrid.grid.data[:, 10, :, :, :] = scale_factor

        output = self.bilagrid.forward_coords(self.rgb, self.coords_xy, self.grid_idcs)
        expected_output = self.rgb * scale_factor
        self.assertTensorClose(output, expected_output)

    def test_forward_constant_offset(self):
        # Set the grid to represent a constant offset
        offset = torch.tensor([0.1, 0.2, 0.3])
        self.bilagrid.grid.data.fill_(0.0)
        self.bilagrid.grid.data[:, 0, :, :, :] = 1.0
        self.bilagrid.grid.data[:, 5, :, :, :] = 1.0
        self.bilagrid.grid.data[:, 10, :, :, :] = 1.0
        self.bilagrid.grid.data[:, 3, :, :, :] = offset[0]
        self.bilagrid.grid.data[:, 7, :, :, :] = offset[1]
        self.bilagrid.grid.data[:, 11, :, :, :] = offset[2]

        output = self.bilagrid.forward_coords(self.rgb, self.coords_xy, self.grid_idcs)
        expected_output = torch.clamp(self.rgb + offset, 0, 1)
        self.assertTensorClose(output, expected_output)

    def test_forward_color_permutation(self):
        # Set the grid to represent a color permutation (e.g., RGB -> BRG)
        self.bilagrid.grid.data.fill_(0.0)
        self.bilagrid.grid.data[:, 1, :, :, :] = 1.0
        self.bilagrid.grid.data[:, 6, :, :, :] = 1.0
        self.bilagrid.grid.data[:, 8, :, :, :] = 1.0

        output = self.bilagrid.forward_coords(self.rgb, self.coords_xy, self.grid_idcs)
        expected_output = self.rgb[:, [1, 2, 0]]
        self.assertTensorClose(output, expected_output)

    def test_single_grid_forward_and_backward(self):
        bilagrid_single = BilateralGrid(self.num_grids, self.width, self.height, self.depth).to(self.device)
        bilagrid_full = BilateralGrid(self.num_grids, self.width, self.height, self.depth).to(self.device)

        grid_data = torch.rand_like(self.bilagrid.grid.data)
        bilagrid_single.grid.data.copy_(grid_data)
        bilagrid_full.grid.data.copy_(grid_data)

        grid_idcs = torch.zeros(100, dtype=torch.long, device=self.bilagrid.grid.device)
        coords_xyz = torch.rand(100, 1, 1, 1, 3, device=self.bilagrid.grid.device) * 2 - 1

        selected_grids_single = bilagrid_single.grid[torch.unique(grid_idcs)]
        coords_xyz_single = coords_xyz.squeeze(1).unsqueeze(0)
        single_grid = F.grid_sample(selected_grids_single, coords_xyz_single, align_corners=True).transpose(0, 2)

        selected_grids = bilagrid_full.grid[grid_idcs]
        full_sample = F.grid_sample(selected_grids, coords_xyz, align_corners=True)
        full_sample.retain_grad()
        self.assertTensorClose(single_grid, full_sample)

        # Create dummy gradients to propagate back
        grad_output = torch.rand_like(single_grid)

        # Perform backward pass
        single_grid.backward(grad_output)
        full_sample.backward(grad_output)

        # Check if gradients are not None
        self.assertIsNotNone(bilagrid_single.grid.grad)
        self.assertIsNotNone(bilagrid_full.grid.grad)
        self.assertTensorClose(bilagrid_single.grid.grad, bilagrid_full.grid.grad)


class TestBilateralGridTraining(unittest.TestCase):
    def setUp(self):
        self.device = "cuda"
        self.num_grids = 1
        self.width = 7
        self.height = 7
        self.depth = 1
        self.bilagrid = BilateralGrid(self.num_grids, self.width, self.height, self.depth).to(self.device)

    def test_bilateral_grid_training(self):
        # Create reference RGB image. It looks like this:
        # .......
        # ...R...
        # .......
        # W..G..K
        # .......
        # ...B...
        # .......
        reference_image = torch.full((7, 7, 3), 0.5)
        reference_image[1, 3] = torch.tensor([1, 0, 0])  # Red
        reference_image[3, 0] = torch.tensor([1, 1, 1])  # White
        reference_image[3, 3] = torch.tensor([0, 1, 0])  # Green
        reference_image[3, 6] = torch.tensor([0, 0, 0])  # Black
        reference_image[5, 3] = torch.tensor([0, 0, 1])  # Blue
        reference_image = reference_image.view(-1, 3).to(self.bilagrid.device)

        # Prepare input data
        coords_xy = torch.linspace(-1, 1, 7).repeat(7, 1).unsqueeze(-1)
        coords_xy = torch.cat([coords_xy, coords_xy.transpose(0, 1)], dim=-1)
        coords_xy = coords_xy.view(-1, 2)
        coords_xy = coords_xy.to(self.bilagrid.device)

        input_rgb = torch.full((49, 3), 0.5).to(self.bilagrid.device)  # All gray input
        grid_idcs = torch.zeros(49, dtype=torch.long).to(self.bilagrid.device)

        # Loss functions
        rgb_loss_fn = nn.HuberLoss()
        rgb_lambda = 1.0
        tv_lambda = 0.1

        # Optimizer
        optimizer = optim.Adam(self.bilagrid.parameters(), lr=0.001)

        # Training loop
        n_training_steps = 1000
        for i in range(n_training_steps):
            optimizer.zero_grad()

            output = self.bilagrid.forward_coords(input_rgb, coords_xy, grid_idcs).to(self.bilagrid.device)

            rgb_loss = rgb_loss_fn(output, reference_image)
            tv_loss = total_variation_spatial(self.bilagrid.grid)
            total_loss = rgb_loss * rgb_lambda + tv_loss * tv_lambda
            total_loss.backward()

            optimizer.step()

        final_output = self.bilagrid.forward_coords(input_rgb, coords_xy, grid_idcs)

        # torch.set_printoptions(precision=2, sci_mode=False)
        # print(f"\nFinal output: \n{final_output}\n")
        # assert False

        # The expected output is similar to the reference image, but less peaky due to TV regularization.
        t = 0.88
        expected_output = t * reference_image + (1 - t) * torch.full(reference_image.shape, 0.5).to(
            self.bilagrid.device
        )
        self.assertTrue(torch.allclose(final_output, expected_output, atol=5e-2))
