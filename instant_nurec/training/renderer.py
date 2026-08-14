# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch

from instant_nurec.predict.render_preview import (
    _configure_gsplat_build,
    _external_distortion_gsplat_parameters,
    _ftheta_gsplat_parameters,
    _kelvin_renderer_config,
    _kelvin_ut_parameters,
    _looks_like_ftheta,
    require_gsplat,
)
from instant_nurec.primitives.kelvin_primitive import KelvinInstantNuRecPrimitive
from instant_nurec.utils.batch import DataAndRenderingBatch
from instant_nurec.utils.cubemap import sample_sky_cubemap
from instant_nurec.utils.geometry import tquat_to_se3_matrix
from instant_nurec.utils.misc import unpack_optional
from instant_nurec.utils.types import RayFlags


@dataclass(kw_only=True, slots=True)
class KelvinRenderOutput:
    rgb: torch.Tensor
    opacity: torch.Tensor
    distance: torch.Tensor
    sky_rgb: torch.Tensor


def _gather_gaussians(primitive: KelvinInstantNuRecPrimitive, timestamp_us: int):
    static = primitive.static_layer
    positions = [static.positions]
    rotations = [static.rotations]
    scales = [static.scales]
    densities = [static.densities]
    rgbs = [static.rgb]
    for dynamic in primitive.dynamic_layers:
        if len(dynamic) == 0:
            continue
        interpolated = dynamic.interpolate(timestamp_us)
        positions.append(interpolated.positions)
        rotations.append(interpolated.rotations)
        scales.append(interpolated.scales)
        densities.append(interpolated.densities)
        rgbs.append(interpolated.rgb)
    return tuple(torch.cat(values, dim=0).float() for values in (positions, rotations, scales, densities, rgbs))


def _camera_kwargs(parameters: object, device: torch.device) -> tuple[torch.Tensor, dict]:
    _configure_gsplat_build()
    external_distortion = _external_distortion_gsplat_parameters(parameters)
    if _looks_like_ftheta(parameters):
        coeffs, rolling, global_shutter = _ftheta_gsplat_parameters(parameters)
        focal = float(parameters.angle_to_pixeldist_poly[1])
        cx, cy = (float(value) for value in parameters.principal_point)
        K = torch.tensor([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], device=device, dtype=torch.float32)
        return K, {
            "camera_model": "ftheta",
            "ftheta_coeffs": coeffs,
            "rolling_shutter": rolling,
            "external_distortion_coeffs": external_distortion,
            "_global_shutter": global_shutter,
        }
    required = ("focal_length", "principal_point", "radial_coeffs")
    if not all(hasattr(parameters, name) for name in required):
        raise TypeError(f"Unsupported training camera model: {type(parameters).__name__}")
    fx, fy = (float(value) for value in parameters.focal_length)
    cx, cy = (float(value) for value in parameters.principal_point)
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device, dtype=torch.float32)
    from gsplat.rendering import RollingShutterType

    shutter_name = getattr(parameters.shutter_type, "name", str(parameters.shutter_type).rsplit(".", 1)[-1])
    rolling = RollingShutterType[shutter_name]
    is_fisheye = not hasattr(parameters, "tangential_coeffs")
    camera_kwargs = {
        "camera_model": "fisheye" if is_fisheye else "pinhole",
        "radial_coeffs": torch.as_tensor(parameters.radial_coeffs, device=device, dtype=torch.float32)[None],
        "external_distortion_coeffs": external_distortion,
        "rolling_shutter": rolling,
        "_global_shutter": RollingShutterType.GLOBAL,
    }
    if not is_fisheye:
        camera_kwargs["tangential_coeffs"] = torch.as_tensor(
            parameters.tangential_coeffs,
            device=device,
            dtype=torch.float32,
        )[None]
        camera_kwargs["thin_prism_coeffs"] = torch.as_tensor(
            parameters.thin_prism_coeffs,
            device=device,
            dtype=torch.float32,
        )[None]
    return K, camera_kwargs


def render_kelvin_supervision(
    primitive: KelvinInstantNuRecPrimitive,
    supervision: DataAndRenderingBatch,
) -> KelvinRenderOutput:
    """Differentiably render all supervision frames with calibrated world rays."""

    rasterization = require_gsplat()
    rendering = unpack_optional(unpack_optional(supervision.rendering).camera)
    camera_data = unpack_optional(supervision.data.camera)
    outputs_rgb, outputs_opacity, outputs_distance, outputs_sky = [], [], [], []

    for frame_index, meta in enumerate(camera_data.meta):
        rays = rendering.rays[frame_index].float().contiguous()
        height, width = rays.shape[:2]
        parameters = rendering.sensor_model_parameters[frame_index]
        K, camera_kwargs = _camera_kwargs(parameters, rays.device)
        global_shutter = camera_kwargs.pop("_global_shutter")
        c2w_start = tquat_to_se3_matrix(rendering.poses_tquat_startend[frame_index, 0], unbatch=True).float()
        c2w_end = tquat_to_se3_matrix(rendering.poses_tquat_startend[frame_index, 1], unbatch=True).float()
        w2c_start, w2c_end = torch.linalg.inv(c2w_start), torch.linalg.inv(c2w_end)
        center_timestamp = int(rendering.timestamps_startend_us_cpu[frame_index].sum().item() // 2)
        means, quats, scales, densities, colors = _gather_gaussians(primitive, center_timestamp)
        rendered, alpha, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=densities[:, 0],
            colors=colors,
            viewmats=w2c_start[None],
            Ks=K[None],
            width=width,
            height=height,
            near_plane=0.2,
            far_plane=torch.finfo(torch.float32).max,
            sh_degree=None,
            # Match Bazel's 3DGUT `RGB-d` contract: return opacity-weighted,
            # along-ray hit distance here and normalize it exactly once in the
            # render distance loss.
            render_mode="RGB-d",
            packed=False,
            with_ut=True,
            ut_params=_kelvin_ut_parameters(),
            renderer_config=_kelvin_renderer_config(),
            with_eval3d=True,
            global_z_order=False,
            rays=rays[None],
            viewmats_rs=(w2c_end[None] if camera_kwargs["rolling_shutter"] != global_shutter else None),
            **camera_kwargs,
        )
        foreground = rendered[0, ..., :3]
        distance = rendered[0, ..., 3:4]
        opacity = alpha[0]
        sky = sample_sky_cubemap(primitive.sky_cubemap, rays[..., 3:])
        # Official rule: only labeled GT-sky rays update the cubemap through
        # render losses; foreground rays see a detached sky value.
        sky_mask = camera_data.labels.get_mask_flags_all(RayFlags.SKY_SEMANTIC)[frame_index].float()
        sky_for_composite = sky * sky_mask + sky.detach() * (1.0 - sky_mask)
        composed = foreground + (1.0 - opacity) * sky_for_composite
        # The affine token axis is keyed by unique_sensor_idx in the official
        # model.  This also lets external supervision cameras intentionally
        # share the context camera's affine transform.
        if not 0 <= meta.unique_sensor_idx < len(primitive.affine_matrix):
            raise IndexError(
                f"Camera unique_sensor_idx={meta.unique_sensor_idx} has no affine token "
                f"(available: 0..{len(primitive.affine_matrix) - 1})"
            )
        affine = primitive.affine_matrix[meta.unique_sensor_idx].float()
        composed = torch.einsum("...p,qp->...q", composed, affine[:, :3]) + affine[:, 3]
        outputs_rgb.append(composed.clamp(0.0, 1.0))
        outputs_opacity.append(opacity)
        outputs_distance.append(distance)
        outputs_sky.append(sky)
    return KelvinRenderOutput(
        rgb=torch.stack(outputs_rgb),
        opacity=torch.stack(outputs_opacity),
        distance=torch.stack(outputs_distance),
        sky_rgb=torch.stack(outputs_sky),
    )
