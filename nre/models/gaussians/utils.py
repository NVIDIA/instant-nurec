# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import logging
import math

from pathlib import Path
from typing import List, Optional, TypeAlias, TypeVar

import numpy as np
import torch


import point_cloud_utils as pcu

from torch import nn

from nre.utils.geometry import quat_to_so3_matrix, so3_matrix_to_quat
from nre.utils.types import PointCloud


log = logging.getLogger(__name__)

## NOTE: SPH code from gaussian-splatting, from plenoctree, from ???
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

T = TypeVar("T", np.ndarray, torch.Tensor)


def RGB2SH(rgb: T) -> T:
    return (rgb - 0.5) / C0


def sh_degree_to_specular_dim(degree: int) -> int:
    """Number of dimensions used by SH of deg [1..degree], inclusive"""
    return 3 * ((degree + 1) ** 2 - 1)


def sh_degree_to_num_features(degree: int) -> int:
    """Number of dimensions used by SH of deg [0..degree], inclusive"""
    return sh_degree_to_specular_dim(degree) + 3


def num_features_to_sh_degree(num_features: int) -> int:
    """
    Given num_features from sh_degree_to_num_features(d) = 3 * (d + 1)^2, compute the integer degree d.
    """
    # 1) Check that num_features is a multiple of 3
    assert num_features % 3 == 0, (
        f"num_features = {num_features} is not a multiple of 3, so it cannot match 3*(d+1)^2 for integer d"
    )

    # 2) Divide by 3
    squared_part = num_features // 3

    # 3) Check that squared_part is a perfect square:
    candidate = math.isqrt(squared_part)
    assert candidate * candidate == squared_part, (
        f"num_features = {num_features} implies {squared_part} is not a perfect square, so it cannot match (d+1)^2 for integer d"
    )

    # 4) Subtract 1 to get the degree
    degree = candidate - 1

    # 5) Optional: check for negative degree (if candidate == 0, that means no valid degree).
    assert degree >= 0, f"num_features = {num_features} is too small to represent degree 0 or higher"

    return degree


def write_ply_3dgs(
    path: Path,
    positions: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    densities: torch.Tensor,
    features_albedo: torch.Tensor,
    features_specular: torch.Tensor | None = None,
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

    num_gaussians = positions.shape[0]
    if features_specular is not None:
        num_speculars = features_specular.shape[-1] // 3
        features_specular_numpy = (
            features_specular.reshape((num_gaussians, num_speculars, 3))
            .transpose(2, 1)
            .reshape((num_gaussians, num_speculars * 3))
            .cpu()
            .numpy()
        )
        for attr_i in range(features_specular.shape[-1]):
            mesh.vertex_data.custom_attributes[f"f_rest_{attr_i}"] = features_specular_numpy[..., attr_i]

    for key, value in custom_attributes.items():
        mesh.vertex_data.custom_attributes[key] = value.cpu().numpy()

    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(str(path))
    log.info(f"Wrote {path.suffix}-file: {path.absolute()}")


