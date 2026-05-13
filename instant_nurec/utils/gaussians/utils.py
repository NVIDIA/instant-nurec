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

from __future__ import annotations

import logging

from pathlib import Path
from typing import TypeVar

import numpy as np
import point_cloud_utils as pcu
import torch


log = logging.getLogger(__name__)

# SH degree-0 coefficient (RGB <-> SH band-0 round-trip in export_ply).
C0 = 0.28209479177387814

T = TypeVar("T", np.ndarray, torch.Tensor)


def RGB2SH(rgb: T) -> T:
    return (rgb - 0.5) / C0


def write_ply_3dgs(
    path: Path,
    positions: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    densities: torch.Tensor,
    features_albedo: torch.Tensor,
    color: torch.Tensor | None = None,
    normals: torch.Tensor | None = None,
    custom_attributes: dict[str, torch.Tensor] = {},
) -> None:
    """
    Writes a PLY file from the given tensors in the original 3DGS format.

    Note that the format should be compatible with the original 3DGS implementation but differences
    between 3DGS/3DGUT/3DGRT rendering will cause slight differences when rendered with
    3rd-party 3DGS viewers.
    Note2: The given tensors should be the raw Gaussian parameters, not the activated ones (e.g., sigmoid, exp, relu, etc.).
    """
    mesh = pcu.TriangleMesh()
    mesh.vertex_data.positions = positions.cpu().numpy()

    if color is not None:
        mesh.vertex_data.colors = color.cpu().numpy()

    if normals is not None:
        assert normals.shape == positions.shape, "normals must have the same shape as positions"
        mesh.vertex_data.normals = normals.cpu().numpy()

    rotations_numpy = rotations.cpu().numpy()
    for attr_i in range(4):
        mesh.vertex_data.custom_attributes[f"rot_{attr_i}"] = rotations_numpy[..., attr_i]

    scales_numpy = scales.cpu().numpy()
    for attr_i in range(3):
        mesh.vertex_data.custom_attributes[f"scale_{attr_i}"] = scales_numpy[..., attr_i]

    mesh.vertex_data.custom_attributes["opacity"] = densities.cpu().numpy()

    features_albedo_numpy = features_albedo.cpu().numpy()
    for attr_i in range(3):
        mesh.vertex_data.custom_attributes[f"f_dc_{attr_i}"] = features_albedo_numpy[..., attr_i]

    for key, value in custom_attributes.items():
        mesh.vertex_data.custom_attributes[key] = value.cpu().numpy()

    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(str(path))
    log.info(f"Wrote {path.suffix}-file: {path.absolute()}")
