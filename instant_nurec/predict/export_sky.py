# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the non-Gaussian sky state that cannot be stored in a 3DGS PLY."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from PIL import Image

from instant_nurec.primitives.kelvin_primitive import KelvinInstantNuRecPrimitive
from instant_nurec.utils.cubemap import layout_sky_cubemap, rotate_sky_cubemap
from instant_nurec.utils.types import RigTrajectories


SKY_SIDECAR_VERSION = 1
SKY_FACE_ORDER = ("right", "left", "top", "bottom", "front", "back")
SKY_FACE_AXES = ("+X", "-X", "-Y", "+Y", "+Z", "-Z")


def sky_sidecar_path(ply_path: Path) -> Path:
    return ply_path.with_suffix(".sky.npz")


def sky_preview_path(ply_path: Path) -> Path:
    return ply_path.with_suffix(".sky.png")


def export_sky(
    primitives: KelvinInstantNuRecPrimitive,
    rig_trajectories: RigTrajectories,
    ply_path: Path,
) -> tuple[Path, Path]:
    """Write a world-aligned cubemap sidecar and a human-readable preview.

    Standard Gaussian-splatting PLY files have no environment-map or camera
    ISP fields. The ``.sky.npz`` file therefore travels next to the PLY and
    stores both, without pickle objects, for the renderer.
    """

    world_rotation = rig_trajectories.T_world_base[:3, :3].to(
        device=primitives.device(), dtype=torch.float32
    )
    cubemap = rotate_sky_cubemap(primitives.sky_cubemap, world_rotation)
    cubemap = cubemap.detach().float().cpu()
    mask = primitives.sky_cubemap_mask
    if mask is None:
        mask = torch.zeros_like(cubemap[..., :1])
    else:
        mask = rotate_sky_cubemap(mask, world_rotation)
        mask = mask.detach().float().cpu().clamp(0.0, 1.0)
    affine = primitives.affine_matrix.detach().float().cpu()
    affine_sensor_ids = tuple(rig_trajectories.camera_calibrations.keys())
    affine_sensor_indices = tuple(
        calibration.unique_sensor_idx
        for calibration in rig_trajectories.camera_calibrations.values()
    )
    if len(affine_sensor_ids) != len(affine):
        raise ValueError(
            "Affine matrix rows must match ordered camera calibrations: "
            f"got {len(affine)} rows and {len(affine_sensor_ids)} calibrations"
        )
    observed_fraction = float(mask.mean())
    if not bool(mask.max() > 0):
        sky_source = "synthetic_fallback"
    elif bool(mask.min() >= 1):
        sky_source = "observed_rgb_semantics"
    else:
        sky_source = "observed_rgb_semantics_plus_fallback"

    sidecar_path = sky_sidecar_path(ply_path)
    preview_path = sky_preview_path(ply_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sidecar_path,
        format_version=np.asarray(SKY_SIDECAR_VERSION, dtype=np.int32),
        sky_cubemap=cubemap.numpy().astype(np.float16),
        sky_cubemap_mask=mask.numpy().astype(np.float16),
        affine_matrix=affine.numpy().astype(np.float32),
        affine_sensor_indices=np.asarray(affine_sensor_indices, dtype=np.int64),
        affine_sensor_ids=np.asarray(affine_sensor_ids),
        face_order=np.asarray(SKY_FACE_ORDER),
        face_axes=np.asarray(SKY_FACE_AXES),
        uv_convention=np.asarray("u_left_to_right_v_top_to_bottom"),
        coordinate_frame=np.asarray("ncore_world"),
        sky_source=np.asarray(sky_source),
        sky_observed_fraction=np.asarray(observed_fraction, dtype=np.float32),
    )

    preview = layout_sky_cubemap(cubemap.clamp(0.0, 1.0))
    preview_u8 = (preview * 255.0).round().to(torch.uint8).numpy()
    Image.fromarray(preview_u8, mode="RGB").save(preview_path)
    return sidecar_path, preview_path
