# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from einops import rearrange

from ncore.impl.data.types import CameraModelParameters

from instant_nurec.utils.sensors.ray_gen import (
    camera_rays_to_image_points,
)


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


@torch.compile
def unproject_to_sky_cubemap(
    sky_cubemap_size: int,
    R_camera_world: torch.Tensor,
    camera_model_parameters: list[CameraModelParameters],
    feature: torch.Tensor,
    feature_mask: torch.Tensor,
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
    sky_rays_d = cubemap_ray_directions(sky_cubemap_size, feature.device)
    feature_dim = feature.shape[-1]
    sky_cubemap_shape = (6, sky_cubemap_size, sky_cubemap_size)
    sky_cubemap_feature = torch.zeros((*sky_cubemap_shape, feature_dim), device=feature.device)
    sky_cubemap_valid_counts = torch.zeros(sky_cubemap_shape, device=feature.device, dtype=torch.int32)
    for vidx in range(feature.shape[0]):
        resolution = torch.from_numpy(camera_model_parameters[vidx].resolution).to(feature.device)
        with torch.autocast("cuda", enabled=False):
            image_points_return = camera_rays_to_image_points(
                camera_model_parameters[vidx], (sky_rays_d @ R_camera_world[vidx, :3, :3].float()).reshape(-1, 3)
            )
        image_points_valid_inds: torch.Tensor = torch.where(image_points_return.valid_flag)[0]
        valid_samples_uv = (image_points_return.image_points[image_points_valid_inds] / resolution) * 2 - 1
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

    sky_cubemap_feature /= torch.clamp(sky_cubemap_valid_counts[..., None].float(), min=1e-3)
    sky_cubemap_valid_mask = sky_cubemap_valid_counts > 0

    return sky_cubemap_feature, sky_cubemap_valid_mask[..., None]


def rotate_sky_cubemap(cubemap: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Rotate the cubemap by the given rotation matrix.

    Per-face (u, v) projection follows the conventions established by
    ``cubemap_ray_directions`` (face order +X, -X, -Y, +Y, +Z, -Z; note
    that indices 2/3 are swapped relative to OpenGL).

    Each face is sampled independently with ``padding_mode="border"``;
    seam blending is local to a face rather than cross-face. Note that
    due to aliasing, rotating the cubemap first and then back is not
    the same as the original.

    Args:
        cubemap: (6, cubemap_size, cubemap_size, C)
        rotation: (3, 3)
    Returns:
        (6, cubemap_size, cubemap_size, C)
    """
    H = cubemap.shape[1]
    C = cubemap.shape[-1]
    device = cubemap.device

    query_rays = cubemap_ray_directions(H, device=device) @ rotation.float()
    query_rays = query_rays.reshape(-1, 3)  # (6*H*H, 3)

    abs_xyz = query_rays.abs()
    dominant_axis = abs_xyz.argmax(dim=-1)  # 0=x, 1=y, 2=z
    a = abs_xyz.gather(-1, dominant_axis.unsqueeze(-1)).squeeze(-1)  # (N,)
    pos = query_rays.gather(-1, dominant_axis.unsqueeze(-1)).squeeze(-1) > 0

    face_idx = torch.where(
        dominant_axis == 0,
        torch.where(pos, torch.zeros_like(dominant_axis), torch.ones_like(dominant_axis)),
        torch.where(
            dominant_axis == 1,
            torch.where(pos, torch.full_like(dominant_axis, 3), torch.full_like(dominant_axis, 2)),
            torch.where(pos, torch.full_like(dominant_axis, 4), torch.full_like(dominant_axis, 5)),
        ),
    )

    x, y, z = query_rays.unbind(-1)
    inv_a = 1.0 / a.clamp(min=1e-12)
    # u/v formulas per face:
    # 0 (+X): u=-z/a, v= y/a
    # 1 (-X): u= z/a, v= y/a
    # 2 (-Y): u= x/a, v= z/a
    # 3 (+Y): u= x/a, v=-z/a
    # 4 (+Z): u= x/a, v= y/a
    # 5 (-Z): u=-x/a, v= y/a
    u_per_face = torch.stack(
        [-z * inv_a, z * inv_a, x * inv_a, x * inv_a, x * inv_a, -x * inv_a], dim=-1
    )
    v_per_face = torch.stack(
        [y * inv_a, y * inv_a, z * inv_a, -z * inv_a, y * inv_a, y * inv_a], dim=-1
    )
    u = u_per_face.gather(-1, face_idx.unsqueeze(-1)).squeeze(-1)
    v = v_per_face.gather(-1, face_idx.unsqueeze(-1)).squeeze(-1)

    cube_4d = cubemap.permute(0, 3, 1, 2).contiguous()  # (6, C, H, H)
    out = torch.zeros((6 * H * H, C), dtype=cubemap.dtype, device=device)
    for f in range(6):
        mask = face_idx == f
        if mask.any():
            uv = torch.stack([u[mask], v[mask]], dim=-1)  # (M, 2)
            grid = uv.unsqueeze(0).unsqueeze(0)  # (1, 1, M, 2)
            sampled = torch.nn.functional.grid_sample(
                cube_4d[f : f + 1],
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            out[mask] = sampled[0, :, 0].T

    return out.reshape(6, H, H, C)




