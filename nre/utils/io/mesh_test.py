# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import tempfile

import numpy as np
import point_cloud_utils as pcu

from pxr import UsdGeom

from nre.utils.io.mesh import Mesh, serialize_mesh, serialize_mesh_usd
from nre.utils.types import NamedSerialized


def test_mesh_colors_default_none() -> None:
    """Mesh.colors defaults to None when not provided."""
    mesh = Mesh(vertices=np.zeros((3, 3), dtype=np.float32), faces=np.array([[0, 1, 2]], dtype=np.int32))
    assert mesh.colors is None


def test_serialize_mesh_ply_includes_colors() -> None:
    """Colors survive the PLY serialize -> load round-trip."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    mesh = Mesh(vertices=vertices, faces=faces, colors=colors)

    result = serialize_mesh(mesh, export_disjoint_meshes=False, formats=["ply"])
    assert len(result) == 1
    item = result[0]
    assert isinstance(item, NamedSerialized)
    assert item.filename == "mesh.ply"
    assert len(item.serialized) > 0

    # Round-trip: write serialized bytes to disk and load back
    assert isinstance(item.serialized, bytes)
    with tempfile.NamedTemporaryFile(suffix=".ply") as tmp:
        tmp.write(item.serialized)
        tmp.flush()
        loaded = pcu.TriangleMesh()
        loaded.load(tmp.name)

    np.testing.assert_allclose(loaded.vertex_data.positions, vertices, atol=1e-5)
    assert loaded.vertex_data.colors is not None
    # PCU may load colors as RGBA (with alpha=255); compare RGB channels only.
    loaded_rgb = loaded.vertex_data.colors[:, :3]
    np.testing.assert_array_equal(loaded_rgb, colors)


def test_serialize_mesh_usd_includes_display_colors() -> None:
    """serialize_mesh_usd writes per-vertex displayColor primvar when colors are present."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    mesh = Mesh(vertices=vertices, faces=faces, colors=colors)

    stage = serialize_mesh_usd(mesh)

    mesh_prim = stage.GetPrimAtPath("/World/mesh")
    primvars_api = UsdGeom.PrimvarsAPI(mesh_prim)
    display_color = primvars_api.GetPrimvar("displayColor")
    assert display_color.IsDefined()
    assert display_color.GetInterpolation() == UsdGeom.Tokens.vertex

    color_values = np.array(display_color.Get())
    expected = colors.astype(np.float32) / 255.0
    np.testing.assert_allclose(color_values, expected, atol=1e-5)


def test_serialize_mesh_usd_no_vertex_display_colors_without_colors() -> None:
    """Without colors, displayColor primvar has no per-vertex interpolation."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = Mesh(vertices=vertices, faces=faces)

    stage = serialize_mesh_usd(mesh)

    mesh_prim = stage.GetPrimAtPath("/World/mesh")
    primvars_api = UsdGeom.PrimvarsAPI(mesh_prim)
    display_color = primvars_api.GetPrimvar("displayColor")
    # UsdGeom.Mesh may auto-create displayColor, but it should not have per-vertex data
    if display_color.IsDefined():
        assert display_color.GetInterpolation() != UsdGeom.Tokens.vertex
