# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference-view Gaussian rendering with cubemap sky compositing."""

from __future__ import annotations

import inspect

from dataclasses import dataclass
from pathlib import Path

import torch

from PIL import Image

from instant_nurec.primitives.kelvin_primitive import KelvinInstantNuRecPrimitive
from instant_nurec.utils.batch import DataAndRenderingBatch
from instant_nurec.utils.cubemap import sample_sky_cubemap
from instant_nurec.utils.geometry import tquat_to_se3_matrix
from instant_nurec.utils.misc import unpack_optional


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
            "Calibrated rendering requires gsplat. Install it with "
            "`uv sync --extra render`, then rerun the render command."
        ) from exc
    parameters = inspect.signature(rasterization).parameters.values()
    if "rays" not in inspect.signature(rasterization).parameters and not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
    ):
        raise RuntimeError(
            "The installed gsplat lacks calibrated world-ray rendering. Run "
            "`uv sync --extra render --reinstall-package gsplat` to install the pinned upstream fix."
        )
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


def _looks_like_ftheta(parameters: object) -> bool:
    return all(
        hasattr(parameters, name)
        for name in (
            "reference_poly",
            "pixeldist_to_angle_poly",
            "angle_to_pixeldist_poly",
            "max_angle",
            "linear_cde",
            "principal_point",
            "shutter_type",
        )
    )


def _ftheta_gsplat_parameters(parameters: object) -> tuple[object, object, object]:
    """Convert NCore F-theta parameters to public gsplat's host types."""

    external_distortion = getattr(parameters, "external_distortion_parameters", None)
    if external_distortion is None:
        external_distortion = getattr(parameters, "external_distortion", None)
    if external_distortion is not None:
        raise NotImplementedError(
            "F-theta rendering with external windshield distortion is not "
            "supported by this public calibrated-render path."
        )

    try:
        from gsplat.rendering import (
            FThetaCameraDistortionParameters,
            FThetaPolynomialType,
            RollingShutterType,
        )
    except (ImportError, OSError) as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "Sky preview rendering requires gsplat. Install it with "
            "`uv sync --extra render`, then rerun with --render-preview."
        ) from exc

    reference_name = getattr(
        parameters.reference_poly,
        "name",
        str(parameters.reference_poly).rsplit(".", maxsplit=1)[-1],
    )
    shutter_name = getattr(
        parameters.shutter_type,
        "name",
        str(parameters.shutter_type).rsplit(".", maxsplit=1)[-1],
    )
    try:
        reference_poly = FThetaPolynomialType[reference_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported F-theta reference polynomial: {parameters.reference_poly!r}") from exc
    try:
        rolling_shutter = RollingShutterType[shutter_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported camera shutter type: {parameters.shutter_type!r}") from exc

    ftheta_coeffs = FThetaCameraDistortionParameters(
        reference_poly=reference_poly,
        pixeldist_to_angle_poly=tuple(float(value) for value in parameters.pixeldist_to_angle_poly),
        angle_to_pixeldist_poly=tuple(float(value) for value in parameters.angle_to_pixeldist_poly),
        max_angle=float(parameters.max_angle),
        linear_cde=tuple(float(value) for value in parameters.linear_cde),
    )
    return ftheta_coeffs, rolling_shutter, RollingShutterType.GLOBAL


@torch.inference_mode()
def render_composited_frame(
    primitive: KelvinInstantNuRecPrimitive,
    context: DataAndRenderingBatch,
    *,
    frame_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render and composite one context frame without writing it to disk.

    ``gsplat`` is imported lazily so the default reconstruction/PLY workflow
    keeps its original dependency footprint. Install the ``render`` extra to
    enable this function.

    Returns the composed RGB image, foreground opacity, and sampled sky image
    as device-resident tensors.
    """

    rendering = unpack_optional(unpack_optional(context.rendering).camera)
    camera_data = unpack_optional(context.data.camera)
    if not 0 <= frame_index < camera_data.b:
        raise IndexError(f"frame_index {frame_index} is outside [0, {camera_data.b})")

    device = primitive.device()
    rays = rendering.rays[frame_index]
    height, width = rays.shape[:2]
    world_rays = rays.to(device=device, dtype=torch.float32).contiguous()
    sensor_parameters = rendering.sensor_model_parameters[frame_index]
    if not _looks_like_ftheta(sensor_parameters):
        raise NotImplementedError(
            "Calibrated rendering currently supports NCore F-theta cameras only; "
            f"got {type(sensor_parameters).__name__}."
        )

    rasterization = require_gsplat()
    ftheta_coeffs, rolling_shutter, global_shutter = _ftheta_gsplat_parameters(sensor_parameters)
    focal = float(sensor_parameters.angle_to_pixeldist_poly[1])
    cx, cy = (float(value) for value in sensor_parameters.principal_point)
    intrinsics = torch.tensor(
        [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    camera_to_world_start = tquat_to_se3_matrix(
        rendering.poses_tquat_startend[frame_index, 0],
        unbatch=True,
    ).to(device=device, dtype=torch.float32)
    camera_to_world_end = tquat_to_se3_matrix(
        rendering.poses_tquat_startend[frame_index, 1],
        unbatch=True,
    ).to(device=device, dtype=torch.float32)
    world_to_camera = torch.linalg.inv(camera_to_world_start)
    world_to_camera_end = torch.linalg.inv(camera_to_world_end)

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
        near_plane=0.2,
        far_plane=torch.finfo(torch.float32).max,
        sh_degree=None,
        render_mode="RGB",
        camera_model="ftheta",
        packed=False,
        with_ut=True,
        with_eval3d=True,
        global_z_order=False,
        rays=world_rays.unsqueeze(0),
        ftheta_coeffs=ftheta_coeffs,
        rolling_shutter=rolling_shutter,
        viewmats_rs=(world_to_camera_end.unsqueeze(0) if rolling_shutter != global_shutter else None),
    )
    foreground = rendered_rgb[0, ..., :3]
    opacity = rendered_alpha[0, ..., 0]
    sky = sample_sky_cubemap(
        primitive.sky_cubemap,
        world_rays[..., 3:],
    )
    affine_index = _camera_affine_index(context, frame_index)
    affine = primitive.affine_matrix[affine_index].float()
    composed = composite_sky_and_affine(foreground, opacity, sky, affine)
    return composed, opacity, sky


def render_reference_preview(
    primitive: KelvinInstantNuRecPrimitive,
    context: DataAndRenderingBatch,
    path: Path,
    *,
    frame_index: int = 0,
) -> RenderPreviewStats:
    """Render one composited context frame and save it as a PNG."""

    composed, opacity, sky = render_composited_frame(
        primitive,
        context,
        frame_index=frame_index,
    )
    height, width = composed.shape[:2]

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
