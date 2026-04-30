# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from omegaconf import DictConfig

from libs.losses.kernel.constants import GRID_NUM_CHANNELS, GRID_NUM_COLS, GRID_NUM_ROWS
from nre.models.base import BaseModel
from nre.utils.profiling import ScopedTimer


# Import the CUDA implementation
try:
    from nre.models.post_processings.bilateral_grid.cuda import bilateral_grid_cuda  # type: ignore  # pycena: skip
except ImportError:
    import bilateral_grid_cuda  # type: ignore  # pycena: skip


class BilateralGridFunction(torch.autograd.Function):
    """Autograd function for the bilateral grid forward operation."""

    @ScopedTimer("BilateralGridFunction.forward")
    @staticmethod
    def forward(
        ctx: Any,
        grid: torch.Tensor,
        coords_xy: torch.Tensor,
        rgb: torch.Tensor,
        grid_idcs: torch.Tensor,
        enable_gridsize1_optimization: bool,
    ) -> torch.Tensor:
        # Save inputs for backward pass
        assert rgb.is_contiguous(), "rgb must be contiguous"
        ctx.save_for_backward(grid, coords_xy, rgb, grid_idcs)
        ctx.enable_gridsize1_optimization = enable_gridsize1_optimization

        # Forward pass using CUDA implementation
        output = torch.empty_like(rgb)
        bilateral_grid_cuda.bilateral_grid_forward(
            grid, coords_xy, rgb, grid_idcs, output, enable_gridsize1_optimization
        )
        return output

    @ScopedTimer("BilateralGridFunction.backward")
    @staticmethod
    def backward(ctx: Any, *grad_output: Any) -> Any:
        grid, coords_xy, rgb, grid_idcs = ctx.saved_tensors

        grad_grid = torch.zeros_like(grid)
        grad_rgb = torch.empty_like(rgb)

        # Call CUDA implementation for backward pass
        bilateral_grid_cuda.bilateral_grid_backward(
            grid,
            coords_xy,
            rgb,
            grid_idcs,
            grad_output[0],
            grad_grid,
            grad_rgb,
            ctx.enable_gridsize1_optimization,
        )

        # Return gradients for all inputs (grid, coords_xy, rgb, grid_idcs, enable_gridsize1_optimization)
        # Note: grid_idcs and enable_gridsize1_optimization don't need gradient
        return grad_grid, None, grad_rgb, None, None


class BilateralGridCUDA(BaseModel):
    """CUDA-accelerated implementation of the bilateral grid color transform."""

    def __init__(
        self,
        num_grids: int,
        width: int,
        height: int,
        depth: int,
        enable_gridsize1_optimization: bool = True,
    ):
        super().__init__(DictConfig({}))

        self.enable_gridsize1_optimization = enable_gridsize1_optimization

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

    def forward(
        self,
        rgb: torch.Tensor,
        coords_xy: torch.Tensor,
        grid_idcs: torch.Tensor,
    ) -> torch.Tensor:
        """Applies the bilateral grid forward operation.

        Args:
            - rgb: A tensor of shape (batch_size, 3) containing the RGB values.
            - coords_xy: A tensor of shape (batch_size, 2) containing the normalized xy coordinates in [0, 1].
            - grid_idcs: A tensor of shape (batch_size,) containing the grid indices.

        Returns:
            A tensor of shape (batch_size, 3) containing the transformed colors.
        """

        return BilateralGridFunction.apply(
            # Making sure image_res can be 1D or 2D
            self.grid,
            coords_xy,
            rgb,
            grid_idcs,
            self.enable_gridsize1_optimization,
        )
