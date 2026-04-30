# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import gzip
import io
import logging

from pathlib import Path
from typing import Any, Optional

import msgpack
import numpy as np
import numpy.typing as npt

from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdUtils, UsdVol

from nre.datasets.utils import nre_matrix_to_nerf
from nre.models.base import BaseModel
from nre.utils.io.utils import USDReferences, initialize_usd_stage
from nre.utils.misc import unpack_optional
from nre.utils.types import AABB3D, ArtifactContents, NamedSerialized, NamedUSDStage


def aabb_center(aabb_min: np.ndarray, aabb_max: np.ndarray) -> np.ndarray:
    return 0.5 * (aabb_min + aabb_max)


def aabb_diag(aabb_min: np.ndarray, aabb_max: np.ndarray) -> np.ndarray:
    return aabb_max - aabb_min


def get_crop_box_matrix(aabb_min: np.ndarray, aabb_max: np.ndarray, offset: np.ndarray, scale: float) -> np.ndarray:
    cen = aabb_center(aabb_min, aabb_max)
    radius = aabb_diag(aabb_min, aabb_max) * 0.5

    rv = np.zeros((4, 3))
    rv[0] = radius[0]
    rv[1] = radius[1]
    rv[2] = radius[2]
    rv[3] = cen
    rv = nre_matrix_to_nerf(rv, offset)

    return rv


def get_crop_box_corners(
    aabb_min: np.ndarray, aabb_max: np.ndarray, offset: np.ndarray, scale: float
) -> list[np.ndarray]:
    m = get_crop_box_matrix(aabb_min, aabb_max, offset, scale)
    rv = []
    for i in range(8):
        rv.append(np.dot(m.T, np.array([1.0 if i & 1 else -1.0, 1.0 if i & 2 else -1.0, 1.0 if i & 4 else -1.0, 1.0])))
    return rv


def add_default_matteobject_domelight(stage, prim_path: str):
    """Add a default DomeLight for matte object rendering.

    Args:
        stage: USD stage to add the light to
        prim_path: Path to the prim
    """
    dome_light_path = prim_path
    dome_light = UsdLux.DomeLight.Define(stage, dome_light_path)
    dome_light_prim = dome_light.GetPrim()

    # Set texture format
    dome_light_prim.CreateAttribute("inputs:texture:format", Sdf.ValueTypeNames.Token).Set("latlong")

    # Set transform attributes
    dome_light_prim.CreateAttribute("xformOp:rotateXYZ", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0, 0, 0))
    dome_light_prim.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(1, 1, 1))
    dome_light_prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0, 0, 0))
    dome_light_prim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray).Set(
        ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    )


def serialize_nrend_usd(
    model: BaseModel,
    aabb: AABB3D,
    offset: npt.NDArray[np.float32],
    proxy_mesh_paths: Optional[USDReferences],
    sequence_track_paths: Optional[USDReferences],
    serialize_to_legacy_nrend_dict: bool,
    bounded_volume: bool = False,
) -> ArtifactContents:
    res: ArtifactContents = []

    name = "volume"  # For compatibility with artifact, name must be 'volume'

    # Serialize msgpack data
    if serialize_to_legacy_nrend_dict:
        model_json_dict = model.serialize_to_legacy_nrend_dict(1.0, offset)
        nerf_offset = [0.0, 0.0, 0.0]
    else:
        model_json_dict = model.serialize_to_json_dict()
        nerf_offset = offset.tolist()

    # patches for backward compatibility
    # - missing appearance_embedding config
    # TODO: remove this when dropping support for Kit 107.3 since later versions are backward compatible
    if "appearance_embedding" not in model_json_dict["nre_data"]["config"]:
        model_json_dict["nre_data"]["config"]["appearance_embedding"] = {
            "name": "skip-appearance",
            "embedding_dim": 0,
            "device": "cuda",
        }

    buffer = io.BytesIO()
    # disable compression since it takes from 10 seconds (level=1) to 60 seconds (level=9)
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=0) as f:
        packed = msgpack.packb(model_json_dict)
        f.write(packed)
    res.append(NamedSerialized(filename=name + ".nurec", serialized=buffer.getvalue()))

    # Create NeRF USD stage
    stage = initialize_usd_stage()

    # Render settings
    render_settings: dict[str, Any] = {}
    render_settings["rtx:rendermode"] = "RaytracedLighting"  # for matte-object support
    render_settings["rtx:directLighting:sampledLighting:samplesPerPixel"] = 8  # for shadows accuracy
    render_settings["rtx:post:histogram:enabled"] = False  # disabling auto-exposure
    render_settings["rtx:post:registeredCompositing:invertToneMap"] = True
    render_settings["rtx:post:registeredCompositing:invertColorCorrection"] = True
    render_settings["rtx:material:enableRefraction"] = False  # for translucency support
    render_settings["rtx:post:tonemap:op"] = 2  # Reinhard operator
    render_settings["rtx:raytracing:fractionalCutoutOpacity"] = False  # OM-117111
    render_settings["rtx:matteObject:visibility:secondaryRays"] = True
    stage.SetMetadataByDictKey("customLayerData", "renderSettings", render_settings)

    # Define UsdVol::Volume.
    nurec_path = "/World/" + name
    nurec_volume = UsdVol.Volume.Define(stage, nurec_path)
    nurec_prim = nurec_volume.GetPrim()

    # Define nurec volume properties
    nurec_prim.CreateAttribute("omni:nurec:isNuRecVolume", Sdf.ValueTypeNames.Bool).Set(True)

    # Do not use proxy transform (deprecated, used for backward compatibility)
    nurec_prim.CreateAttribute("omni:nurec:useProxyTransform", Sdf.ValueTypeNames.Bool).Set(False)

    # Define nrend field assets and link to NeRF prim.
    density_field_path = nurec_path + "/density_field"
    density_field = stage.DefinePrim(density_field_path, "OmniNuRecFieldAsset")
    nurec_volume.CreateFieldRelationship("density", density_field_path)
    emissive_color_field_path = nurec_path + "/emissive_color_field"
    emissive_color_field = stage.DefinePrim(emissive_color_field_path, "OmniNuRecFieldAsset")
    nurec_volume.CreateFieldRelationship("emissiveColor", emissive_color_field_path)

    # Fill out field asset properties.
    # Assume all files are saved in same directory, so filenames correspond to relative paths trivially.
    nurec_relative_path = "./" + name + ".nurec"
    density_field.CreateAttribute("filePath", Sdf.ValueTypeNames.Asset).Set(nurec_relative_path)
    density_field.CreateAttribute("fieldName", Sdf.ValueTypeNames.Token).Set("density")
    density_field.CreateAttribute("fieldDataType", Sdf.ValueTypeNames.Token).Set("float")
    density_field.CreateAttribute("fieldRole", Sdf.ValueTypeNames.Token).Set("density")
    emissive_color_field.CreateAttribute("filePath", Sdf.ValueTypeNames.Asset).Set(nurec_relative_path)
    emissive_color_field.CreateAttribute("fieldName", Sdf.ValueTypeNames.Token).Set("emissiveColor")
    emissive_color_field.CreateAttribute("fieldDataType", Sdf.ValueTypeNames.Token).Set("float3")
    emissive_color_field.CreateAttribute("fieldRole", Sdf.ValueTypeNames.Token).Set("emissiveColor")
    # Add identity color correction matrix
    emissive_color_field.CreateAttribute("omni:nurec:ccmR", Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f([1.0, 0.0, 0.0, 0.0])
    )
    emissive_color_field.CreateAttribute("omni:nurec:ccmG", Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f([0.0, 1.0, 0.0, 0.0])
    )
    emissive_color_field.CreateAttribute("omni:nurec:ccmB", Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f([0.0, 0.0, 1.0, 0.0])
    )

    if bounded_volume:
        # Assume "render AABB to local" is always eye(3)
        aabb_corners = get_crop_box_corners(aabb.blb.squeeze().numpy(), aabb.trf.squeeze().numpy(), offset, scale=1.0)
        min_coord = aabb_corners[0]
        max_coord = aabb_corners[0]
        for c in aabb_corners:
            min_coord = np.minimum(min_coord, c)
            max_coord = np.maximum(max_coord, c)
    else:
        min_coord = np.array([-np.inf, -np.inf, -np.inf])
        max_coord = np.array([np.inf, np.inf, np.inf])

    nurec_prim.GetAttribute("extent").Set([min_coord.tolist(), max_coord.tolist()])

    # Set offset
    nurec_prim.CreateAttribute("omni:nurec:offset", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(nerf_offset))

    # Set crop as two float3 instead of one float[], so that it is user-configurable in the UI.
    # Assume the property doesn't exist yet.
    nurec_prim.CreateAttribute("omni:nurec:crop:minBounds", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(min_coord.tolist()))
    nurec_prim.CreateAttribute("omni:nurec:crop:maxBounds", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(max_coord.tolist()))

    def get_composed_prim_path(default_prim_path: str, prim_path: str, reference_path: str):
        # Ensure that all incoming parameters are strings, not USD Paths
        prim_path = str(prim_path)
        default_prim_path = str(default_prim_path)
        reference_path = str(reference_path)

        # Process the prim path to remove the default prim path, mirroring the behavior of USD layer composition.
        prim_path = prim_path[len(default_prim_path) :]
        return reference_path + prim_path

    if proxy_mesh_paths:
        # The delegate captures all errors about dangling references, effectively silencing them.
        delegate = UsdUtils.CoalescingDiagnosticDelegate()
        # Set the proxy mesh(es)
        targets = []
        for named_mesh_stage, proxy_mesh_prim_path in unpack_optional(proxy_mesh_paths):
            override_prim_path = f"/World/{Path(named_mesh_stage.filename).stem}"
            proxy_mesh_world_prim = stage.OverridePrim(override_prim_path)

            # Assume all files are saved in same directory, so filenames correspond to relative paths trivially.
            # Note: Do not refer only to the mesh prim as a reference, but rather, to the whole scene
            # ("world" xform prim). This makes it so the matte mesh materials are also referenced properly.
            proxy_mesh_world_prim.GetReferences().AddReference(named_mesh_stage.filename)

            targets.append(get_composed_prim_path("/World", proxy_mesh_prim_path, override_prim_path))

            # Check if the referenced prim has subsets. If it does, they are also targets for the proxy mesh.
            # N.B. that we append the mesh prim itself in either case to cover all rendering configs.
            mesh = UsdGeom.Mesh(named_mesh_stage.stage.GetPrimAtPath(proxy_mesh_prim_path))
            subsets = UsdGeom.Subset.GetAllGeomSubsets(mesh)
            for subset in subsets:
                targets.append(get_composed_prim_path("/World", subset.GetPath(), override_prim_path))
        nurec_prim.CreateRelationship("proxy").SetTargets(targets)

        # Add default DomeLight for matte object rendering
        add_default_matteobject_domelight(stage, "/World/MatteObjectDomeLight")

    if sequence_track_paths:
        # The delegate captures all errors about dangling references, effectively silencing them.
        delegate = UsdUtils.CoalescingDiagnosticDelegate()
        # Set the tracks
        targets = []
        for named_tracks_stage, track_prim_path in unpack_optional(sequence_track_paths):
            override_prim_path = f"/World/{Path(named_tracks_stage.filename).stem}"
            track_world_prim = stage.OverridePrim(override_prim_path)

            # Assume all files are saved in same directory, so filenames correspond to relative paths trivially.
            track_world_prim.GetReferences().AddReference(named_tracks_stage.filename)

            targets.append(get_composed_prim_path("/World", track_prim_path, override_prim_path))

        nurec_prim.CreateRelationship("tracks").SetTargets(targets)

    res.append(NamedUSDStage(filename=name + ".usda", stage=stage))

    return res
