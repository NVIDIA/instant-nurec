# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference-view Gaussian rendering with cubemap sky compositing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from PIL import Image

from instant_nurec.primitives.kelvin_primitive import KelvinInstantNuRecPrimitive
from instant_nurec.utils.batch import DataAndRenderingBatch
from instant_nurec.utils.cubemap import sample_sky_cubemap
from instant_nurec.utils.geometry import tquat_to_se3_matrix
from instant_nurec.utils.misc import unpack_optional
from instant_nurec.utils.sensor import to_simple_pinhole_model_parameters


@dataclass(frozen=True, slots=True)
class RenderPreviewStats:
    path: Path
    width: int
    height: int
    background_fraction: float
    sky_contribution_mean: float


def require_gsplat():
    """Import and return gsplat's rasterizer before reconstruction starts."""

    try:
        from gsplat import rasterization
    except (ImportError, OSError) as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "Sky preview rendering requires gsplat. Install it with "
            "`uv sync --extra render`, then rerun with --render-preview."
        ) from exc
    return rasterization


def composite_sky_and_affine(
    foreground_rgb: torch.Tensor,
    opacity: torch.Tensor,
    sky_rgb: torch.Tensor,
    affine_matrix: torch.Tensor,
) -> torch.Tensor:
    """Match Kelvin rendering: alpha-composite sky, then apply camera ISP."""

    if foreground_rgb.shape != sky_rgb.shape or foreground_rgb.shape[-1] != 3:
        raise ValueError(
            "foreground_rgb and sky_rgb must have identical (..., 3) shapes, "
            f"got {tuple(foreground_rgb.shape)} and {tuple(sky_rgb.shape)}"
        )
    if opacity.shape not in (foreground_rgb.shape[:-1], (*foreground_rgb.shape[:-1], 1)):
        raise ValueError(f"Opacity shape {tuple(opacity.shape)} does not match RGB {tuple(foreground_rgb.shape)}")
    if affine_matrix.shape != (3, 4):
        raise ValueError(f"Expected affine matrix (3, 4), got {tuple(affine_matrix.shape)}")

    alpha = opacity.unsqueeze(-1) if opacity.ndim == foreground_rgb.ndim - 1 else opacity
    composed = foreground_rgb + (1.0 - alpha) * sky_rgb
    composed = torch.einsum("...p,qp->...q", composed, affine_matrix[:, :3])
    return (composed + affine_matrix[:, 3]).clamp(0.0, 1.0)


def _camera_affine_index(context: DataAndRenderingBatch, frame_index: int) -> int:
    camera = unpack_optional(context.data.camera)
    ordered_sensor_indices: list[int] = []
    for meta in camera.meta:
        if meta.unique_sensor_idx not in ordered_sensor_indices:
            ordered_sensor_indices.append(meta.unique_sensor_idx)
    return ordered_sensor_indices.index(camera.meta[frame_index].unique_sensor_idx)


def render_reference_preview(
    primitive: KelvinInstantNuRecPrimitive,
    context: DataAndRenderingBatch,
    path: Path,
    *,
    frame_index: int = 0,
) -> RenderPreviewStats:
    """Render one context frame and composite the primitive's sky cubemap.

    ``gsplat`` is imported lazily so the default reconstruction/PLY workflow
    keeps its original dependency footprint. Install the ``render`` extra to
    enable this function.
    """

    rasterization = require_gsplat()

    rendering = unpack_optional(unpack_optional(context.rendering).camera)
    camera_data = unpack_optional(context.data.camera)
    if not 0 <= frame_index < camera_data.b:
        raise IndexError(f"frame_index {frame_index} is outside [0, {camera_data.b})")

    device = primitive.device()
    rays = rendering.rays[frame_index]
    height, width = rays.shape[:2]
    pinhole = to_simple_pinhole_model_parameters(rendering.sensor_model_parameters[frame_index])
    fx, fy = (float(value) for value in pinhole.focal_length)
    cx, cy = (float(value) for value in pinhole.principal_point)
    intrinsics = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    # Match public inference, which encodes the exposure-end pose. The
    # foreground projection remains a simple-pinhole approximation; sky rays
    # themselves retain the NCore camera model and rolling-shutter geometry.
    camera_to_world = tquat_to_se3_matrix(
        rendering.poses_tquat_startend[frame_index, 1],
        unbatch=True,
    ).to(device=device, dtype=torch.float32)
    world_to_camera = torch.linalg.inv(camera_to_world)

    static = primitive.static_layer
    rendered_rgb, rendered_alpha, _ = rasterization(
        means=static.positions.float(),
        quats=static.rotations.float(),
        scales=static.scales.float(),
        opacities=static.densities[:, 0].float(),
        colors=static.rgb.float(),
        viewmats=world_to_camera.unsqueeze(0),
        Ks=intrinsics.unsqueeze(0),
        width=width,
        height=height,
        near_plane=0.01,
        far_plane=1000.0,
        sh_degree=None,
        render_mode="RGB",
        camera_model="pinhole",
        packed=True,
    )
    foreground = rendered_rgb[0, ..., :3]
    opacity = rendered_alpha[0, ..., 0]
    sky = sample_sky_cubemap(
        primitive.sky_cubemap,
        rays[..., 3:].to(device=device, dtype=torch.float32),
    )
    affine_index = _camera_affine_index(context, frame_index)
    affine = primitive.affine_matrix[affine_index].float()
    composed = composite_sky_and_affine(foreground, opacity, sky, affine)

    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = (composed.detach().cpu() * 255.0).round().to(torch.uint8).numpy()
    Image.fromarray(pixels, mode="RGB").save(path)
    sky_contribution = ((1.0 - opacity[..., None]) * sky).mean().item()
    return RenderPreviewStats(
        path=path,
        width=width,
        height=height,
        background_fraction=(1.0 - opacity).mean().item(),
        sky_contribution_mean=sky_contribution,
    )
