# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from typing import Literal

import torch

from libs.slang_gaussians.interface import mcmc_slang  # type: ignore
from libs.slang_utils.utils import div_up


def fused_perturb_gaussians(
    positions: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    densities: torch.Tensor,
    current_lr: float,
    quaternion_format: Literal["xyzw", "wxyz"] = "wxyz",
) -> None:
    n_gaussians = positions.size(0)
    assert positions.dim() == 2 and positions.size(1) == 3, f"positions must be [N, 3], got {positions.shape}"
    assert quats.dim() == 2 and quats.size(1) == 4, f"quats must be [N, 4], got {quats.shape}"
    assert scales.dim() == 2 and scales.size(1) == 3, f"scales must be [N, 3], got {scales.shape}"
    assert densities.dim() == 1, f"densities must be [N], got {densities.shape}"
    assert quats.size(0) == n_gaussians, "quats and positions must have same batch size"
    assert scales.size(0) == n_gaussians, "scales and positions must have same batch size"
    assert densities.size(0) == n_gaussians, "densities and positions must have same batch size"
    assert quaternion_format in ("xyzw", "wxyz"), f"quaternion_format must be 'xyzw' or 'wxyz'"
    assert positions.is_cuda, "Tensors must be on CUDA device"
    assert positions.dtype == torch.float32, "Tensors must be float32"

    if n_gaussians == 0:
        return

    positions = positions.contiguous()
    quats = quats.contiguous()
    scales = scales.contiguous()
    densities = densities.contiguous()

    # Generate random noise
    noise = torch.randn_like(positions)

    threads_per_block = 256
    blocks = div_up(n_gaussians, threads_per_block)
    wxyz_format = quaternion_format == "wxyz"

    mcmc_slang.fused_perturb_gaussians_kernel(
        (threads_per_block, 1, 1),
        (blocks, 1, 1),
        n_gaussians,
        positions,
        quats,
        scales,
        densities,
        noise,
        current_lr,
        wxyz_format,
    )


def fused_perturb_gaussians_rigid(
    positions: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    densities: torch.Tensor,
    cuboid_dims: torch.Tensor,
    current_lr: float,
    quaternion_format: Literal["xyzw", "wxyz"] = "wxyz",
) -> None:
    n_gaussians = positions.size(0)
    assert positions.dim() == 2 and positions.size(1) == 3, f"positions must be [N, 3], got {positions.shape}"
    assert quats.dim() == 2 and quats.size(1) == 4, f"quats must be [N, 4], got {quats.shape}"
    assert scales.dim() == 2 and scales.size(1) == 3, f"scales must be [N, 3], got {scales.shape}"
    assert densities.dim() == 1, f"densities must be [N], got {densities.shape}"
    assert cuboid_dims.dim() == 2 and cuboid_dims.size(1) == 3, f"cuboid_dims must be [N, 3], got {cuboid_dims.shape}"
    assert quats.size(0) == n_gaussians, "quats and positions must have same batch size"
    assert scales.size(0) == n_gaussians, "scales and positions must have same batch size"
    assert densities.size(0) == n_gaussians, "densities and positions must have same batch size"
    assert cuboid_dims.size(0) == n_gaussians, "cuboid_dims and positions must have same batch size"
    assert quaternion_format in ("xyzw", "wxyz"), f"quaternion_format must be 'xyzw' or 'wxyz'"
    assert positions.is_cuda, "Tensors must be on CUDA device"
    assert positions.dtype == torch.float32, "Tensors must be float32"

    if n_gaussians == 0:
        return

    # Ensure contiguous
    positions = positions.contiguous()
    quats = quats.contiguous()
    scales = scales.contiguous()
    densities = densities.contiguous()
    cuboid_dims = cuboid_dims.contiguous()

    # Generate random noise
    noise = torch.randn_like(positions)

    threads_per_block = 256
    blocks = div_up(n_gaussians, threads_per_block)
    wxyz_format = quaternion_format == "wxyz"

    mcmc_slang.fused_perturb_gaussians_rigid_kernel(
        (threads_per_block, 1, 1),
        (blocks, 1, 1),
        n_gaussians,
        positions,
        quats,
        scales,
        densities,
        noise,
        cuboid_dims,
        current_lr,
        wxyz_format,
    )
