# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Layer 2: CUDA losses model layer - Neural network modules for losses."""

from typing import Optional

import torch

from libs.losses.functional.cuda_losses_function import RoadGaussiansFunction


class RoadGaussiansLossCUDA(torch.nn.Module):
    """
    CUDA-accelerated version of RoadGaussiansLoss.

    This class provides the same interface as the original RoadGaussiansLoss
    but uses CUDA kernels for computation with all transformations done in CUDA.
    Random values are generated on the Python side and scaled in CUDA.
    """

    def __init__(
        self, layer_name: str, n_samples: int, grid_len: float, min_val: float, range_val: float, rotation_lambda: float
    ):
        super().__init__()
        self.layer_name = layer_name
        self.n_samples = n_samples
        self.grid_len = grid_len
        self.min = min_val
        self.range = range_val
        self.rotation_lambda = rotation_lambda

    def forward(
        self,
        positions_world: torch.Tensor,
        rotations_world: torch.Tensor,
        pose_tquat: torch.Tensor,
        random_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the road gaussians loss with full CUDA implementation.

        Args:
            positions_world: World-space positions of gaussians [N, 3]
            rotations_world: World-space rotations of gaussians [N, 4] (quaternion xyzw format)
            pose_tquat: [1, 7] - 7 values: [tx, ty, tz, qx, qy, qz, qw]

        Returns:
            loss: Scalar loss value
        """
        # Call the CUDA function with all transformations done in CUDA
        loss = RoadGaussiansFunction.apply(
            positions_world,
            rotations_world,
            pose_tquat,
            self.n_samples,
            self.min,
            self.range,
            self.grid_len,
            self.rotation_lambda,
            random_values,
        )

        return loss
