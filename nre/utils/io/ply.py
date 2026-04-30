# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging

from typing import Optional

import numpy as np
import point_cloud_utils as pcu

from numpy.typing import NDArray


def save_ply(
    filename: str,
    vertices: NDArray[np.float32],
    triangles: Optional[NDArray[np.int32]] = None,
    normals: Optional[NDArray[np.float32]] = None,
    colors: Optional[NDArray[np.uint8]] = None,
    logger: Optional[logging.Logger] = None,
):
    assert vertices.shape[1] == 3
    mesh = pcu.TriangleMesh()
    mesh.vertex_data.positions = vertices
    data_description = f"{len(vertices)} vertices"
    if triangles is not None:
        assert triangles.shape[1] == 3
        assert np.max(triangles) < len(vertices)
        mesh.face_data.vertex_ids = triangles
        data_description = f"{data_description}, {len(triangles)} triangles"
    if normals is not None:
        assert normals.shape == (vertices.shape[0], 3)
        mesh.vertex_data.normals = normals
        data_description = f"{data_description}, {len(normals)} normals"
    if colors is not None:
        assert colors.shape == (vertices.shape[0], 3)
        mesh.vertex_data.colors = colors
        data_description = f"{data_description}, {len(colors)} colors"
    if logger:
        logger.info(f"Saving {data_description} to {filename}")
    mesh.save(filename)
