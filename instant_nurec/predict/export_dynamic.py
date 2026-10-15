# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export time-dependent Gaussians beside the static PLY, without pickle."""

from pathlib import Path

import numpy as np
import torch

from instant_nurec.primitives.kelvin_primitive import KelvinInstantNuRecPrimitive
from instant_nurec.utils.types import RigTrajectories


def export_dynamic(
    primitive: KelvinInstantNuRecPrimitive,
    rig: RigTrajectories,
    ply_path: Path,
) -> Path:
    world = rig.T_world_base.to(device=primitive.device(), dtype=torch.float32)
    payload = {
        "format_version": np.asarray(1, dtype=np.int32),
        "coordinate_frame": np.asarray("ncore_world"),
        "quaternion_format": np.asarray("wxyz"),
        "n_layers": np.asarray(len(primitive.dynamic_layers), dtype=np.int32),
    }
    for index, layer in enumerate(primitive.dynamic_layers):
        layer = layer.rigid_transform(world)
        for name in (
            "keyframe_positions", "keyframe_timestamps_us", "rotations", "scales", "rgb", "max_densities",
        ):
            tensor = getattr(layer, name).detach().cpu()
            if tensor.is_floating_point():
                tensor = tensor.float()
            payload[f"layer_{index}_{name}"] = tensor.numpy()
        if layer.motion_time_bounds_us is not None:
            payload[f"layer_{index}_motion_time_bounds_us"] = layer.motion_time_bounds_us.detach().cpu().numpy()
        payload[f"layer_{index}_falloff"] = np.asarray(layer.falloff)
        payload[f"layer_{index}_falloff_gaussian_exp"] = np.asarray(layer.FALLOFF_GAUSSIAN_EXP, dtype=np.float32)
    path = ply_path.with_suffix(".dynamic.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path
