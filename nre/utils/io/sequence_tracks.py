# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
import logging

from typing import Optional

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom

from nre.datasets.summary import DataSourceSummary
from nre.datasets.tracks import CuboidTracks
from nre.utils.geometry import tquat_to_se3_matrix
from nre.utils.io.utils import initialize_usd_stage, nre_tf_to_usd_tf, sanitize_usd_path
from nre.utils.types import ArtifactContents, NamedSerialized, NamedUSDStage


def serialize_sequence_tracks_usd(
    cuboid_tracks: CuboidTracks, excluded_tracks: Optional[CuboidTracks], usd_timestamp_offset_us: int
) -> Usd.Stage:
    logger = logging.getLogger(__name__)

    stage = initialize_usd_stage()
    usd_time_code_per_second = stage.GetTimeCodesPerSecond()
    usd_timestamp_scale = usd_time_code_per_second * 1e-06  # from microseconds to usd timecodes
    usd_start_time_code = np.inf
    usd_end_time_code = 0.0

    cuboid_tracks_processed: dict[str, dict] = {}
    if cuboid_tracks is not None:
        excluded_tracks_in_chunk = frozenset(excluded_tracks.tracks_id) if excluded_tracks else set()
        n_tracks = len(cuboid_tracks.tracks_id)
        for t in filter(lambda x: cuboid_tracks.tracks_id[x] not in excluded_tracks_in_chunk, range(n_tracks)):
            track_start_idx, N_track_poses = cuboid_tracks.tracks_packinfo[t, :]
            track_timestamps_us = (
                cuboid_tracks.tracks_timestamps_us[track_start_idx : track_start_idx + N_track_poses].cpu().numpy()
            )
            usd_time_code = usd_timestamp_scale * (track_timestamps_us - usd_timestamp_offset_us)
            usd_start_time_code = min(usd_start_time_code, np.min(usd_time_code))
            usd_end_time_code = max(usd_end_time_code, np.max(usd_time_code))
            track_poses = (
                cuboid_tracks.tracks_poses[track_start_idx : track_start_idx + N_track_poses, :]
                .data.cpu()
                .numpy()
                .astype(np.double)
            )
            track_poses_se3 = tquat_to_se3_matrix(track_poses).cpu().numpy()
            poses = []
            for i in range(usd_time_code.shape[0]):
                poses.append(nre_tf_to_usd_tf(track_poses_se3[i, :, :]))
            # Clean the track ID by removing @ suffixes for serialization
            cleaned_track_id = DataSourceSummary._clean_track_id_str(cuboid_tracks.tracks_id[t])
            cuboid_tracks_processed[cleaned_track_id] = {
                "poses": poses,
                "usd_timecodes": usd_time_code,
                "cuboid_dims": cuboid_tracks.cuboids_dims[t].cpu().numpy().astype(np.double),
                "label_class": cuboid_tracks.tracks_label_class[t],
            }
        logger.info(f"Collected params for {len(cuboid_tracks_processed.keys())} bounding boxes.")

    if usd_start_time_code <= usd_end_time_code:
        stage.SetMetadata("startTimeCode", usd_start_time_code)
        stage.SetMetadata("endTimeCode", usd_end_time_code)
    stage.SetMetadataByDictKey("customLayerData", "absoluteTimeOffsetMicroSec", usd_timestamp_offset_us)

    # Define xform containing all the bounding boxes
    bboxes_path = "/World/bounding_boxes"
    bboxes_prim = UsdGeom.Xform.Define(stage, bboxes_path)

    for i, (track_name, params) in enumerate(cuboid_tracks_processed.items()):
        # Create editing cuboid XForm
        cuboid_path = sanitize_usd_path(f"/World/bounding_boxes/track_{i:05d}_{track_name}")
        cuboid_xform = UsdGeom.Xform.Define(stage, cuboid_path)
        # Create cuboid
        cuboid_mesh = UsdGeom.Mesh.Define(stage, f"{cuboid_path}/cuboid")
        cuboid_mesh_xform = UsdGeom.Xformable(cuboid_mesh)
        cuboid_prim = cuboid_mesh.GetPrim()

        # Apply poses
        cuboid_transform_op = cuboid_mesh_xform.AddTransformOp()
        for t in range(len(params["usd_timecodes"])):
            cuboid_transform_op.Set(params["poses"][t], params["usd_timecodes"][t])

        # Define mesh
        width, height, depth = params["cuboid_dims"]
        half_width, half_height, half_depth = width / 2, height / 2, depth / 2
        points = [
            Gf.Vec3f(-half_width, -half_height, half_depth),
            Gf.Vec3f(half_width, -half_height, half_depth),
            Gf.Vec3f(half_width, half_height, half_depth),
            Gf.Vec3f(-half_width, half_height, half_depth),
            Gf.Vec3f(-half_width, -half_height, -half_depth),
            Gf.Vec3f(half_width, -half_height, -half_depth),
            Gf.Vec3f(half_width, half_height, -half_depth),
            Gf.Vec3f(-half_width, half_height, -half_depth),
        ]
        cuboid_mesh.CreatePointsAttr(points)

        faceVertexIndices = [
            0,
            1,
            2,
            3,  # Front face
            4,
            5,
            6,
            7,  # Back face
            0,
            1,
            5,
            4,  # Bottom face
            2,
            3,
            7,
            6,  # Top face
            0,
            3,
            7,
            4,  # Left face
            1,
            2,
            6,
            5,  # Right face
        ]
        cuboid_mesh.CreateFaceVertexIndicesAttr(faceVertexIndices)
        cuboid_mesh.CreateFaceVertexCountsAttr([4, 4, 4, 4, 4, 4])

        # Set visibility
        cuboid_primvars_api = UsdGeom.PrimvarsAPI(cuboid_prim)
        cuboid_primvars_api.CreatePrimvar("hideForCamera", Sdf.ValueTypeNames.Bool, UsdGeom.Tokens.constant).Set(True)
        cuboid_prim.CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool, True).Set(True)

        # Add semantics
        cuboid_prim.AddAppliedSchema("SemanticsAPI:Semantics")
        cuboid_prim.CreateAttribute("semantic:Semantics:params:semanticType", Sdf.ValueTypeNames.String, False).Set(
            "class"
        )
        cuboid_prim.CreateAttribute("semantic:Semantics:params:semanticData", Sdf.ValueTypeNames.String, False).Set(
            params["label_class"]
        )

        # Add visibility temporal range : min/max timestamps
        cuboid_primvars_api.CreatePrimvar("startTimeCode", Sdf.ValueTypeNames.Double, UsdGeom.Tokens.constant).Set(
            np.min(params["usd_timecodes"])
        )
        cuboid_primvars_api.CreatePrimvar("endTimeCode", Sdf.ValueTypeNames.Double, UsdGeom.Tokens.constant).Set(
            np.max(params["usd_timecodes"])
        )

        # Add track id (use cleaned track name)
        cuboid_primvars_api.CreatePrimvar("trackUId", Sdf.ValueTypeNames.Token, UsdGeom.Tokens.constant).Set(track_name)

    return stage


def serialize_sequence_tracks(
    sequence_id: str,
    cuboid_tracks: CuboidTracks,
    excluded_tracks: Optional[CuboidTracks],
    filename: str = "sequence_tracks",
    formats: list[str] = ["json", "usda"],
    usd_timestamp_offset_us: int = 0,
) -> ArtifactContents:
    res: ArtifactContents = []
    for file_format in formats:
        filename_with_suffix = filename + "." + file_format
        match file_format:
            case "json":
                # Get the serialized data and clean track IDs in the dict
                sequence_tracks_dict = {sequence_id: cuboid_tracks.to_dict()}
                # Clean track IDs in the serialized data
                DataSourceSummary._clean_track_ids_in_serialized_sequence_tracks_dict(sequence_tracks_dict)
                res.append(
                    NamedSerialized(
                        filename=filename_with_suffix, serialized=json.dumps(sequence_tracks_dict, indent=4)
                    )
                )
            case "usd" | "usda":
                res.append(
                    NamedUSDStage(
                        filename=filename_with_suffix,
                        stage=serialize_sequence_tracks_usd(cuboid_tracks, excluded_tracks, usd_timestamp_offset_us),
                    )
                )
            case _:
                raise ValueError(f"The following sequence tracks format is not supported: {file_format}")
    return res
