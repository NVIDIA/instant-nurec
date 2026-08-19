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

import math

import torch
import torch.nn.functional as F

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


def directions_to_cubemap_face_uv(directions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Map directions to the Instant NuRec cubemap face and ``(u, v)``.

    The returned UV coordinates use the ``[-1, 1]`` convention consumed by
    :func:`torch.nn.functional.grid_sample`. Face order matches
    :func:`cubemap_ray_directions`: ``+X, -X, -Y, +Y, +Z, -Z``.
    """

    if directions.shape[-1] != 3:
        raise ValueError(f"Cubemap directions must end in three channels, got {tuple(directions.shape)}")
    flat = directions.reshape(-1, 3).float()
    abs_xyz = flat.abs()
    dominant_axis = abs_xyz.argmax(dim=-1)
    dominant = abs_xyz.gather(-1, dominant_axis.unsqueeze(-1)).squeeze(-1)
    positive = flat.gather(-1, dominant_axis.unsqueeze(-1)).squeeze(-1) > 0
    face = torch.where(
        dominant_axis == 0,
        torch.where(positive, torch.zeros_like(dominant_axis), torch.ones_like(dominant_axis)),
        torch.where(
            dominant_axis == 1,
            torch.where(positive, torch.full_like(dominant_axis, 3), torch.full_like(dominant_axis, 2)),
            torch.where(positive, torch.full_like(dominant_axis, 4), torch.full_like(dominant_axis, 5)),
        ),
    )

    x, y, z = flat.unbind(-1)
    inv_dominant = dominant.clamp_min(1e-12).reciprocal()
    u_by_face = torch.stack(
        [
            -z * inv_dominant,
            z * inv_dominant,
            x * inv_dominant,
            x * inv_dominant,
            x * inv_dominant,
            -x * inv_dominant,
        ],
        dim=-1,
    )
    v_by_face = torch.stack(
        [
            y * inv_dominant,
            y * inv_dominant,
            z * inv_dominant,
            -z * inv_dominant,
            y * inv_dominant,
            y * inv_dominant,
        ],
        dim=-1,
    )
    uv = torch.stack(
        [
            u_by_face.gather(-1, face.unsqueeze(-1)).squeeze(-1),
            v_by_face.gather(-1, face.unsqueeze(-1)).squeeze(-1),
        ],
        dim=-1,
    )
    return face.reshape(directions.shape[:-1]), uv.reshape(*directions.shape[:-1], 2)


def sample_sky_cubemap(cubemap: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample an Instant NuRec cubemap for arbitrary directions."""

    if cubemap.ndim != 4 or cubemap.shape[0] != 6 or cubemap.shape[1] != cubemap.shape[2]:
        raise ValueError(f"Expected cubemap shape (6, H, H, C), got {tuple(cubemap.shape)}")
    face, uv = directions_to_cubemap_face_uv(directions)
    flat_face = face.reshape(-1)
    flat_uv = uv.reshape(-1, 2)
    channels = cubemap.shape[-1]
    textures = cubemap.permute(0, 3, 1, 2).contiguous()
    sampled = torch.empty((len(flat_face), channels), dtype=cubemap.dtype, device=cubemap.device)
    for face_index in range(6):
        mask = flat_face == face_index
        if mask.any():
            grid = flat_uv[mask].unsqueeze(0).unsqueeze(0)
            values = F.grid_sample(
                textures[face_index : face_index + 1],
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            sampled[mask] = values[0, :, 0].T
    return sampled.reshape(*directions.shape[:-1], channels)


def soften_cubemap_mask(
    mask: torch.Tensor,
    *,
    dilation_kernel: int = 31,
    gaussian_sigma: float = 8.0,
) -> torch.Tensor:
    """Dilate and feather a per-face cubemap mask.

    The operation intentionally treats the six faces as an image batch. This
    matches Kelvin predict preprocessing: a broad dilation first covers small
    projection gaps, then a separable Gaussian produces the sky/road blend.
    """

    if mask.ndim != 4 or mask.shape[0] != 6 or mask.shape[-1] != 1:
        raise ValueError(f"Expected cubemap mask shape (6, H, W, 1), got {tuple(mask.shape)}")
    if mask.shape[1] != mask.shape[2]:
        raise ValueError(f"Expected square cubemap faces, got {tuple(mask.shape[1:3])}")
    if dilation_kernel <= 0 or dilation_kernel % 2 == 0:
        raise ValueError("dilation_kernel must be a positive odd integer")
    if gaussian_sigma <= 0:
        raise ValueError("gaussian_sigma must be positive")

    face_mask = mask.permute(0, 3, 1, 2).float()
    face_mask = F.max_pool2d(
        face_mask,
        kernel_size=dilation_kernel,
        stride=1,
        padding=dilation_kernel // 2,
    )

    radius = max(1, math.ceil(3.0 * gaussian_sigma))
    coordinates = torch.arange(-radius, radius + 1, device=mask.device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (coordinates / gaussian_sigma).square())
    kernel /= kernel.sum()
    horizontal = kernel.reshape(1, 1, 1, -1)
    vertical = kernel.reshape(1, 1, -1, 1)
    face_mask = F.conv2d(
        F.pad(face_mask, (radius, radius, 0, 0), mode="replicate"),
        horizontal,
    )
    face_mask = F.conv2d(
        F.pad(face_mask, (0, 0, radius, radius), mode="replicate"),
        vertical,
    )
    return face_mask.permute(0, 2, 3, 1).clamp_(0.0, 1.0).contiguous()


@torch.no_grad()
def build_observed_sky_cubemap_with_mask(
    size: int,
    directions: torch.Tensor,
    rgb: torch.Tensor,
    sky_mask: torch.Tensor,
    road_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a self-contained sky texture from decoder observations.

    Released Kelvin Front/Varying weights do not contain the learned sky head,
    but the dense decoder still predicts both canonical RGB and a SKY semantic
    mask for every source ray. Those observations are accumulated into the
    cubemap once during reconstruction. The returned softened mask is retained
    on the primitive so export preprocessing can apply Kelvin's ROAD-Gaussian
    infill after density pruning.
    """

    if size <= 0:
        raise ValueError("Cubemap size must be positive")
    if directions.shape != rgb.shape or directions.shape[-1] != 3:
        raise ValueError(
            f"Directions and RGB must have identical (..., 3) shape, got {directions.shape} and {rgb.shape}"
        )
    if sky_mask.shape != directions.shape[:-1]:
        raise ValueError(f"Sky mask shape {sky_mask.shape} does not match rays {directions.shape[:-1]}")
    if road_mask is not None and road_mask.shape != sky_mask.shape:
        raise ValueError(f"Road mask shape {road_mask.shape} does not match sky mask {sky_mask.shape}")

    flat_directions = F.normalize(directions.reshape(-1, 3).float(), dim=-1)
    flat_rgb = rgb.reshape(-1, 3).float().clamp(0.0, 1.0)
    flat_sky_mask = sky_mask.reshape(-1).bool()
    finite = torch.isfinite(flat_directions).all(dim=-1) & torch.isfinite(flat_rgb).all(dim=-1)
    flat_sky_mask &= finite

    default_road = torch.tensor([0.16, 0.17, 0.18], device=flat_rgb.device)
    road_color = default_road
    if road_mask is not None:
        road_colors = flat_rgb[road_mask.reshape(-1).bool() & finite]
        if len(road_colors):
            road_colors = road_colors[:: max(1, len(road_colors) // 100_000)]
            median = road_colors.median(dim=0).values
            road_color = road_colors[(road_colors - median).square().sum(dim=-1).argmin()]

    cube_directions = cubemap_ray_directions(size, device=flat_rgb.device).float()
    cube_up = cube_directions[..., 2:3]
    if flat_sky_mask.any():
        observed_colors = flat_rgb[flat_sky_mask]
        observed_up = flat_directions[flat_sky_mask, 2:3]
        mean_color = observed_colors.mean(dim=0)
        mean_up = observed_up.mean()
        centered_up = observed_up - mean_up
        variance = centered_up.square().mean().clamp_min(1e-4)
        slope = (centered_up * (observed_colors - mean_color)).mean(dim=0) / variance
        slope = slope.clamp(-0.5, 0.5)
        fitted_sky = (mean_color + (cube_up - mean_up) * slope).clamp(0.0, 1.0)
    else:
        horizon = torch.tensor([0.48, 0.58, 0.70], device=flat_rgb.device)
        zenith = torch.tensor([0.20, 0.38, 0.62], device=flat_rgb.device)
        fitted_sky = torch.lerp(horizon, zenith, cube_up.clamp(0.0, 1.0))

    # The base cubemap remains sky-like in every direction. Kelvin export
    # preprocessing is responsible for replacing non-sky texels with the
    # representative color selected from the *pruned Gaussian ROAD class*.
    fallback = fitted_sky
    if not flat_sky_mask.any():
        fallback = torch.where(
            cube_up >= 0,
            fallback,
            road_color.view(1, 1, 1, 3),
        )
        empty_mask = torch.zeros((6, size, size, 1), dtype=torch.float32, device=flat_rgb.device)
        return fallback.contiguous(), empty_mask

    sky_directions = flat_directions[flat_sky_mask]
    sky_colors = flat_rgb[flat_sky_mask]
    face, uv = directions_to_cubemap_face_uv(sky_directions)
    texel = (((uv + 1.0) * (0.5 * size)).floor()).long().clamp(0, size - 1)
    linear_index = face.reshape(-1).long() * (size * size) + texel[:, 1] * size + texel[:, 0]
    sums = torch.zeros((6 * size * size, 3), dtype=torch.float32, device=flat_rgb.device)
    counts = torch.zeros((6 * size * size, 1), dtype=torch.float32, device=flat_rgb.device)
    sums.scatter_add_(0, linear_index[:, None].expand(-1, 3), sky_colors)
    counts.scatter_add_(0, linear_index[:, None], torch.ones_like(linear_index, dtype=torch.float32)[:, None])
    observed = (sums / counts.clamp_min(1.0)).reshape(6, size, size, 3)
    observed_mask = (counts > 0).reshape(6, size, size, 1)
    cubemap = torch.where(observed_mask, observed, fallback).contiguous()
    return cubemap, soften_cubemap_mask(observed_mask)


@torch.no_grad()
def build_observed_sky_cubemap(
    size: int,
    directions: torch.Tensor,
    rgb: torch.Tensor,
    sky_mask: torch.Tensor,
    road_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compatibility wrapper returning only the observation-derived cubemap."""

    cubemap, _ = build_observed_sky_cubemap_with_mask(
        size,
        directions,
        rgb,
        sky_mask,
        road_mask,
    )
    return cubemap


def layout_sky_cubemap(cubemap: torch.Tensor) -> torch.Tensor:
    """Arrange the six faces as ``[[left, front, right], [back, bottom, top]]``."""

    if cubemap.ndim != 4 or cubemap.shape[0] != 6 or cubemap.shape[1] != cubemap.shape[2]:
        raise ValueError(f"Expected cubemap shape (6, H, H, C), got {tuple(cubemap.shape)}")
    right, left, top, bottom, front, back = cubemap.unbind(0)
    return torch.cat(
        [
            torch.cat([left, front, right], dim=1),
            torch.cat([back, bottom, top], dim=1),
        ],
        dim=0,
    )


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


