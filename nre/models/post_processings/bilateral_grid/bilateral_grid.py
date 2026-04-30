# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig

from libs.losses.kernel.constants import GRID_NUM_CHANNELS, GRID_NUM_COLS, GRID_NUM_ROWS
from nre.models.base import BaseModel


class BilateralGrid(BaseModel):
    def __init__(self, num_grids: int, width: int, height: int, depth: int, *args, **kwargs):
        super().__init__(DictConfig({}))

        self.depth = depth
        self.height = height
        self.width = width

        self.num_grids = num_grids
        # Initialize grid parameters for multiple grids.
        # grid shape: (num_grids, 12, depth, height, width)
        self.grid = nn.Parameter(
            torch.eye(GRID_NUM_ROWS, GRID_NUM_COLS, device=self.device)
            .view(1, GRID_NUM_CHANNELS, 1, 1, 1)
            .repeat(
                self.num_grids,
                1,
                self.depth,
                self.height,
                self.width,
            )
        )

        # Keep guidance-related data specific to each object instead of static
        # as we may switch to learned guidance functions in the future.
        # Initialize as 2D to avoid broadcasting.
        self.luma_weights_bt_601 = nn.Buffer(torch.tensor([[0.299, 0.587, 0.114]]), persistent=False)

    def forward_coords(self, rgb: torch.Tensor, coords_xy: torch.Tensor, grid_idcs: torch.Tensor) -> torch.Tensor:
        """Applies the bilateral grid collection transformation to the input color.

        Args:
            - rgb: A tensor of shape (batch_size, 3) containing the RGB values.
            - coords_xy: A tensor of shape (batch_size, 2) containing the normalized xy coordinates in [-1, 1].
            - grid_idcs: A tensor of shape (batch_size) containing the grid indices.

        Returns:
            A tensor of shape (batch_size, 3) containing the transformed colors.
        """
        assert coords_xy.shape[0] == rgb.shape[0] and coords_xy.shape[0] == grid_idcs.shape[0], (
            "Bilateral grid forward requires matching metadata sizes"
        )
        rgb = rgb.to(self.device)

        grid_idcs = grid_idcs.to(dtype=torch.int32)

        assert grid_idcs.max() < self.num_grids and grid_idcs.min() >= -1, (
            "grid_idcs must be in the range [-1, num_grids)"
        )

        # Create mask for pixels to skip (invalid sensor_id)
        skip_mask = grid_idcs == -1

        # Initialize output with original RGB values
        output = rgb.clone()

        # Only process pixels with valid sensor_id (between [0; num_grids-1])
        if skip_mask.all():
            return output

        valid_mask = ~skip_mask
        valid_rgb = rgb[valid_mask]
        valid_coords = coords_xy[valid_mask]
        valid_grid_idcs = grid_idcs[valid_mask]

        affine_tf = self.slice_grid(coords_xy=valid_coords, rgb=valid_rgb, grid_idcs=valid_grid_idcs)

        # Reshape affine_tf and rgb for matrix multiplication
        # affine_tf shape: (valid_batch_size, 3, 4)
        affine_tf = affine_tf.view(-1, 3, 4)

        # valid_rgb shape: (valid_batch_size, 3, 1)
        valid_rgb = valid_rgb.unsqueeze(-1)

        # Apply the affine transformation to the rgb color
        # transformed_rgb shape: (valid_batch_size, 3)
        transformed_rgb = torch.matmul(affine_tf[..., :3], valid_rgb).squeeze(-1) + affine_tf[..., 3]

        # Clamp to SDR colors
        transformed_rgb = torch.clamp(transformed_rgb, 0, 1)

        # Update output for valid pixels
        output[valid_mask] = transformed_rgb

        return output

    def forward(
        self,
        rgb: torch.Tensor,
        coords_xy: torch.Tensor,
        grid_idcs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            rgb: A tensor of shape (batch_size, 3) containing the RGB values.
            coords_xy: A tensor of shape (batch_size, 2) containing the normalized xy coordinates in [0, 1].
            grid_idcs: A tensor of shape (batch_size) containing the grid indices.
        """
        return self.forward_coords(rgb, coords_xy * 2.0 - 1.0, grid_idcs)

    def slice_grid(self, coords_xy: torch.Tensor, rgb: torch.Tensor, grid_idcs: torch.Tensor) -> torch.Tensor:
        """Slices the bilateral grid collection at given coordinates using trilinear interpolation.

        Args:
            - coords_xy: A tensor of shape (batch_size, 2) containing the xy coordinates in [-1, 1].
            - rgb: A tensor of shape (batch_size, 3) containing the RGB values in [0, 1].
            - grid_idcs: A tensor of shape (batch_size) containing the grid indices in [0, num_grids-1].

        Returns:
            A tensor of shape (batch_size, 12) containing the interpolated grid parameters.
        """
        coords_xy = coords_xy.to(self.device)
        rgb = rgb.to(self.device)
        batch_size = rgb.shape[0]

        # Generate slicing coordinates.
        # coords_xy shape: (batch_size, 2)
        # coords_z shape: (batch_size, 1)
        coords_z = self.guidance(rgb)
        # coords_xyz shape: (batch_size, 3)
        coords_xyz = torch.cat([coords_xy, coords_z], dim=-1)

        # Reshape coordinates for grid_sample.
        # coords_xyz shape: (batch_size, 1, 1, 1, 3)
        coords_xyz = coords_xyz.view(batch_size, 1, 1, 1, 3)

        # Select the appropriate grids based on grid_idcs.
        # selected_grids shape: (batch_size, 12, depth, height, width)
        # If all grid indices are the same, we can optimize the memory usage by selecting the grid directly.
        # Then, perform trilinear interpolation using grid_sample.
        # Since the grid is a downsampled version of the input image, using align_corners = True
        # ensures that sampling at the input image coordinates behaves as expected.
        # affine_tf shape: (batch_size, 12, 1, 1, 1)
        if len(unique_grid_idx := torch.unique(grid_idcs)) == 1:
            selected_grids = self.grid[unique_grid_idx]
            coords_xyz = coords_xyz.squeeze(1).unsqueeze(0)
            affine_tf = F.grid_sample(selected_grids, coords_xyz, align_corners=True).transpose(0, 2)
        else:
            selected_grids = self.grid[grid_idcs]
            affine_tf = F.grid_sample(selected_grids, coords_xyz, align_corners=True)

        # Reshape the interpolated parameters.
        # affine_tf shape: (batch_size, 12)
        affine_tf = affine_tf.view(batch_size, 12)

        return affine_tf

    def guidance(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb = rgb.to(self.device)
        # Convert RGB to luma.
        # rgb shape: (batch_size, 3)
        # luma shape: (batch_size, 1)
        luma = torch.mm(rgb, self.luma_weights_bt_601.t())
        # Rescale to [-1, 1] for grid_sample
        luma = luma * 2 - 1
        return luma
