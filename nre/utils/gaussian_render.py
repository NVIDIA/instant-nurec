# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generic rendering utilities for Gaussian splatting models."""

import os

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import imageio
import numpy as np
import torch

from kiui.cam import orbit_camera

from apps.asset_harvester.rendering import render
from nre.models.gaussians.utils import PLYGaussianLoader
from nre.utils.geometry import quat_to_so3_matrix, so3_matrix_to_quat


format_type = Literal["image", "video", "image+video"]


@dataclass
class RenderConfig:
    """Simple config for rendering parameters."""

    output_size: int = 512
    znear: float = 0.1
    zfar: float = 500.0
    fov: float = 70
    fps: int = 30
    elevation: float = 0
    dist: float = 1.5
    num_views: int = 32
    gaussian_type: str = "3dgut"


def render_ply_orbit(
    ply_path: Path,
    output_dir: Path,
    format: format_type,
    compatible: bool,
    opt: RenderConfig,
):
    """Render PLY file to images/video from orbit camera views.

    Args:
        ply_path: Path to the Gaussian PLY file
        output_dir: Directory to save rendered images/video
        format: Output format - "image", "video", or "image+video"
        compatible: If True, assume PLY has pre-activated values
        opt: RenderConfig with rendering settings
    """
    loaded_ply = PLYGaussianLoader(ply_path)

    # transform gaussians to Y-up coordinate system
    transform_tensor = torch.tensor(
        [
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=loaded_ply.positions.dtype,
        device=loaded_ply.positions.device,
    )
    loaded_ply.transform(transform_tensor)

    positions = loaded_ply.positions
    densities = loaded_ply.densities
    scales = loaded_ply.scales
    rotations = loaded_ply.rotations
    features_albedo = loaded_ply.features_albedo
    features_specular = loaded_ply.features_specular
    if features_specular is not None:
        raise NotImplementedError("features_specular is not supported")

    # invert activation to make it compatible with the original ply format
    if not compatible:
        densities = torch.sigmoid(densities)
        scales = torch.exp(scales)
        features_albedo = features_albedo * 0.28209479177387814 + 0.5

    gaussians = torch.cat(
        [
            positions,
            densities,
            scales,
            rotations,
            features_albedo,
        ],
        dim=1,
    ).unsqueeze(0)  # [B, N, 14]

    tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fov))

    res = opt.output_size
    f = res / (2 * tan_half_fov)
    intrinsics_real = torch.tensor([f, f, res / 2.0, res / 2.0], device=loaded_ply.positions.device)[None, ...]

    azimuth = np.linspace(0, 360, opt.num_views + 1)
    cam_views = []
    intrinsics = []
    for azi in azimuth:
        cam_poses = (
            torch.from_numpy(orbit_camera(opt.elevation, azi, radius=opt.dist))
            .unsqueeze(0)
            .to(loaded_ply.positions.device)
        )
        cam_poses[:, :3, 1:3] *= -1  # invert up & forward direction

        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2)  # [V, 4, 4]
        cam_views.append(cam_view)
        intrinsics.append(intrinsics_real)
    cam_views_tensor = torch.stack(cam_views, dim=1)
    intrinsics_tensor = torch.stack(intrinsics, dim=1)
    images_dict = render(opt, gaussians, cam_views_tensor, intrinsics=intrinsics_tensor)
    images_tensor = images_dict["image"]
    images_np = (images_tensor.permute(0, 1, 3, 4, 2).float().cpu().numpy() * 255).astype(np.uint8)  # [B, V, H, W, 3]
    images_list = [images_np[i] for i in range(len(images_np))]

    os.makedirs(output_dir, exist_ok=True)

    for b, images in enumerate(images_list):
        for i in range(images.shape[0]):
            imageio.imwrite(output_dir / Path(f"{b:06d}_{i:06d}.png"), images[i])

        if "video" in format:
            imageio.mimsave(output_dir / Path(f"{b:06d}.mp4"), images, fps=opt.fps)


def render_gaussians_single_view(gaussian_params, opt: RenderConfig, elevation: float, azimuth: float, distance: float):
    """Render Gaussians from a single camera view.

    Args:
        gaussian_params: Dictionary containing Gaussian parameters:
            - positions: [1, N, 3] vertex positions
            - scales: [1, N, 3] scale values (log space)
            - rotations: [1, N, 4] quaternions in wxyz format
            - opacities: [1, N, 1] opacity values (logit space)
            - shs: [1, N, N_sh] spherical harmonics coefficients
        opt: RenderConfig with rendering settings
        elevation: Camera elevation angle in degrees
        azimuth: Camera azimuth angle in degrees
        distance: Camera distance from subject

    Returns:
        Rendered image as numpy array [H, W, 3] in uint8 format
    """
    # Extract individual parameters
    positions = gaussian_params["positions"][0]  # [N, 3]
    scales_raw = gaussian_params["scales"][0]  # [N, 3]
    rotations = gaussian_params["rotations"][0]  # [N, 4] - quaternions in wxyz format
    densities_raw = gaussian_params["opacities"][0]  # [N, 1]
    features_albedo_raw = gaussian_params["shs"][0]  # [N, N_sh]

    # Transform gaussians to Y-up coordinate system
    transform_tensor = torch.tensor(
        [
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=positions.dtype,
        device=positions.device,
    )

    # Apply transformation to positions
    positions = (positions @ transform_tensor[:3, :3].T) + transform_tensor[:3, 3]

    # Apply transformation to rotations (quaternions)
    # Convert from wxyz to xyzw for quat_to_so3_matrix
    rotations_xyzw = rotations[:, [1, 2, 3, 0]]
    rotations_matrix = quat_to_so3_matrix(rotations_xyzw, unbatch=False)
    # Apply rotation transform
    rotations_transformed = transform_tensor[:3, :3] @ rotations_matrix
    # Convert back to quaternion wxyz format
    rotations = so3_matrix_to_quat(rotations_transformed, unbatch=False)[:, [3, 0, 1, 2]]

    # Apply activations
    densities = torch.sigmoid(densities_raw)
    scales = torch.exp(scales_raw)
    features_albedo = features_albedo_raw * 0.28209479177387814 + 0.5

    # Create gaussians tensor
    gaussians = torch.cat(
        [
            positions,
            densities,
            scales,
            rotations,
            features_albedo,
        ],
        dim=-1,
    ).unsqueeze(0)  # [1, N, 11+N_sh]

    # Setup camera parameters
    tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fov))
    res = opt.output_size
    f = res / (2 * tan_half_fov)
    intrinsics_real = torch.tensor([f, f, res / 2.0, res / 2.0])[None, ...].cuda()

    # Single camera view with specified parameters
    cam_poses = torch.from_numpy(orbit_camera(elevation, azimuth, radius=distance)).unsqueeze(0).to("cuda")
    cam_poses[:, :3, 1:3] *= -1  # invert up & forward direction

    # cameras needed by gaussian rasterizer
    cam_view = torch.inverse(cam_poses).transpose(1, 2)  # [V, 4, 4]

    cam_views_tensor = cam_view.unsqueeze(1)  # [1, 1, 4, 4]
    intrinsics_tensor = intrinsics_real.unsqueeze(1)  # [1, 1, 4]

    # Render the gaussians
    images_dict = render(opt, gaussians, cam_views_tensor, intrinsics=intrinsics_tensor)
    images_tensor = images_dict["image"]
    images_np = (images_tensor.permute(0, 1, 3, 4, 2).float().cpu().numpy() * 255).astype(np.uint8)

    # Single image from single camera view
    image = images_np[0, 0]  # [H, W, 3]

    return image
