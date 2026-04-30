# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import nvdiffrast.torch as dr
import torch

from einops import rearrange

from libs.vren.interface import camera_rays_to_image_points
from ncore.impl.data.types import CameraModelParameters
from nre.utils.profiling import ScopedTimer


def cubemap_ray_directions(size: int, device: torch.device) -> torch.Tensor:
    """
    Compute (6, size, size, 3) ray directions corresponding to the sky texture.
    """
    # Corresponds to pixel centers (not corners)
    px = (torch.arange(size, device=device) + 0.5) / size * 2 - 1
    uu, vv = torch.meshgrid(px, px, indexing="xy")
    front_dirs = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1)
    front_dirs = front_dirs / front_dirs.norm(dim=-1, keepdim=True)

    xx, yy, zz = front_dirs.unbind(-1)
    right_dirs = torch.stack([zz, yy, -xx], dim=-1)
    left_dirs = torch.stack([-zz, yy, xx], dim=-1)
    top_dirs = torch.stack([xx, -zz, yy], dim=-1)
    bottom_dirs = torch.stack([xx, zz, -yy], dim=-1)
    back_dirs = torch.stack([-xx, yy, -zz], dim=-1)

    return torch.stack([right_dirs, left_dirs, top_dirs, bottom_dirs, front_dirs, back_dirs], dim=0)


@ScopedTimer("nrm.utils.unproject_to_sky_cubemap")
@torch.compile
def unproject_to_sky_cubemap(
    sky_cubemap_size: int,
    R_camera_world: torch.Tensor,
    camera_model_parameters: list[CameraModelParameters],
    feature: torch.Tensor,
    feature_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Unproject the RGB image to the sky cubemap.
    Args:
        R_camera_world: (N, 3, 3) world rotation matrix of the camera
        camera_model_parameters: The camera model parameters. [N,]
        feature: The feature image. [N, H, W, C]
        feature_mask: The mask of the feature image. [N, H, W, 1]
    Returns:
        The sky cubemap feature image. [6, self.sky_cubemap_size, self.sky_cubemap_size, C]
        The mask corresponding to the cubemap. [6, self.sky_cubemap_size, self.sky_cubemap_size, 1]
    """
    # TODO: This function is very slow. Consider downsample the feature image or use batched implementation
    # of camera_rays_to_image_points.

    sky_rays_d = cubemap_ray_directions(sky_cubemap_size, feature.device)
    feature_dim = feature.shape[-1]
    sky_cubemap_shape = (6, sky_cubemap_size, sky_cubemap_size)
    sky_cubemap_feature = torch.zeros((*sky_cubemap_shape, feature_dim), device=feature.device)
    sky_cubemap_valid_counts = torch.zeros(sky_cubemap_shape, device=feature.device, dtype=torch.int32)
    for vidx in range(feature.shape[0]):
        resolution = torch.from_numpy(camera_model_parameters[vidx].resolution).to(feature.device)
        with torch.autocast("cuda", enabled=False):
            # Fastest implementation we found. Don't change.
            image_points_return = camera_rays_to_image_points(
                camera_model_parameters[vidx], (sky_rays_d @ R_camera_world[vidx, :3, :3].float()).reshape(-1, 3)
            )
        image_points_valid_inds: torch.Tensor = torch.where(image_points_return.valid_flag)[0]
        valid_samples_uv = (image_points_return.image_points[image_points_valid_inds] / resolution) * 2 - 1
        if feature_mask is not None:
            valid_samples_mask = (
                torch.nn.functional.grid_sample(
                    rearrange(feature_mask[vidx].float(), "H W 1 -> 1 1 H W"),
                    valid_samples_uv[None, None],
                    padding_mode="border",
                    align_corners=False,
                ).reshape(-1)
                > 0.9
            )
            valid_samples_uv = valid_samples_uv[valid_samples_mask]
            image_points_valid_inds = image_points_valid_inds[valid_samples_mask]

        sky_cubemap_feature.view(-1, feature_dim)[image_points_valid_inds] += torch.nn.functional.grid_sample(
            rearrange(feature[vidx], "H W C -> 1 C H W"),
            valid_samples_uv[None, None],
            padding_mode="border",
            align_corners=False,
        )[0, :, 0].T
        sky_cubemap_valid_counts.view(-1)[image_points_valid_inds] += 1

    # Average the features from multiple views
    sky_cubemap_feature /= torch.clamp(sky_cubemap_valid_counts[..., None].float(), min=1e-3)
    sky_cubemap_valid_mask = sky_cubemap_valid_counts > 0

    return sky_cubemap_feature, sky_cubemap_valid_mask[..., None]


def rotate_sky_cubemap(cubemap: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """
    Rotate the cubemap by the given rotation matrix.
    Note that due to aliasing, rotating the cubemap first and then back is not the same as the original.
    Args:
        cubemap: (6, cubemap_size, cubemap_size, 3)
        rotation: (3, 3)
    Returns:
        (6, cubemap_size, cubemap_size, 3)
    """
    cubemap_size = cubemap.shape[1]
    query_rays = cubemap_ray_directions(cubemap_size, device=cubemap.device) @ rotation.float()
    query_rays = query_rays.reshape(-1, 3)
    opengl_rays_d = torch.stack([query_rays[:, 0], -query_rays[:, 1], query_rays[:, 2]], dim=-1)
    sky_color = dr.texture(
        cubemap[None],
        opengl_rays_d[None, None],
        filter_mode="linear",
        boundary_mode="cube",
    )
    assert isinstance(sky_color, torch.Tensor)
    sky_color = sky_color.reshape(6, cubemap_size, cubemap_size, 3)
    return sky_color


def layout_sky_cubemap(cubemap: torch.Tensor) -> torch.Tensor:
    """
    Layout the cubemap into a 2x3 grid, suitable for visualization. Grid has the following layout:
    [
        [left, front, right],
        [back, bottom, top],
    ]
    Args:
        cubemap: (6, cubemap_size, cubemap_size, 3)
    Returns:
        (2 * cubemap_size, 3 * cubemap_size, 3)
    """
    right, left, top, bottom, front, back = cubemap.unbind(0)
    return torch.cat(
        [
            torch.cat([left, front, right], dim=1),
            torch.cat([back, bottom, top], dim=1),
        ],
        dim=0,
    )
