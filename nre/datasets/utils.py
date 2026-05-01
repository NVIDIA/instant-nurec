# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import dataclasses

from functools import lru_cache
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import tqdm

import ncore.data
import ncore.impl.common.transformations as ncore_transformations
import ncore.sensors
import ncore_internal.data.v3
import nre.utils.ncore_utils as ncore_utils

from libs.vren.interface import (  # type: ignore
    world_points_to_image_points_shutter_pose,
)
from nre.datasets.tracks import CuboidTracks, TrackFlags
from nre.utils.batch import RectSubsampled
from nre.utils.geometry import (
    se3_matrix_to_tquat,
)
from nre.utils.misc import (
    to_torch,
    unpack_optional,
)
from nre.utils.types import CuboidTracksData, HalfClosedInterval, TracksData
def compute_cuboid_df(
    sequence_loader: ncore.data.SequenceLoaderProtocol,
    time_range_us: HalfClosedInterval,
) -> pd.DataFrame:
    """Extracts cuboid observations from the dataset in a given time range.

    Predict-only standalone always uses `vars(observation)` directly; the
    NRE-side `serialize_observation=True` path that produced JSON dicts via
    `observation.to_dict()` was unused at predict time and significantly
    slower (Phase 1 step 4.3). The progress bar is also always disabled.
    """

    cuboid_observations = sequence_loader.get_cuboid_track_observations(
        timestamp_interval_us=ncore_transformations.HalfClosedInterval(
            start=time_range_us.start,
            # Note: ncore's HalfClosedInterval type correctly calls this non-exclusive boundary `stop`,
            # whereas the notion within NRE's HalfClosedInterval with `end` (which is inconsistently used as
            # inclusive / exclusive) is messy
            stop=time_range_us.end,
        ),
    )

    # Load all cuboid observations into dataframe for easy querying
    cuboid_dicts = [vars(observation) for observation in cuboid_observations]

    if len(cuboid_dicts):
        # load all observations into dataframe, deducing dynamic types from structure
        cuboids_df = pd.DataFrame.from_records(cuboid_dicts)
    else:
        # initialize empty cuboids dataframe, inheriting all top-level fields from CuboidTrackObservation type
        cuboids_df = pd.DataFrame(
            {
                field.name: pd.Series(dtype="object" if field.type not in ("int", "float", "bool") else field.type)
                for field in dataclasses.fields(ncore.data.CuboidTrackObservation)
            }
        )

    # Make sure observations are ordered in time (so per-track poses are ordered as well)
    return cuboids_df.sort_values(by=["track_id", "timestamp_us"], ascending=[True, True])


def consolidate_cuboid_tracks(
    cuboids_df: pd.DataFrame,
    sequence_loader: ncore.data.SequenceLoaderProtocol,
    track_label_sources: list[str] | None,
    track_min_centroid_rig_dist_m: float,
    T_world_world_base: np.ndarray | None,
    tqdm_disabled: bool,
) -> dict[str, dict]:
    """
    Gather the cuboid track observations into a set of named tracks with timestamped poses relative to the world frame.

    Args:
        cuboids_df (pd.DataFrame): Dataframe of all cuboid observations to consolidate into tracks
        sequence_loader (ncore.data.SequenceLoaderProtocol): The sequence loader to load the data from
        track_label_sources (list[str] | None): List of label sources to consider for track consolidation
        track_min_centroid_rig_dist_m (float): Minimum distance of the track centroid to the rig frame to consider the observation
        T_world_world_base (np.ndarray | None): Optional base transformation to apply to all world poses
        tqdm_disabled (bool): When true, no progress bars are showed

    Returns:
        dict[str, dict]: Consolidated cuboid tracks (indexed by track-id) with elements:
            - dimension: (np.ndarray) The (l,w,h) dimensions of the cuboid
            - label_class: (int) The class id of the cuboid
            - poses: (list[np.ndarray]) List of cuboid poses in world frame
            - timestamps_us: (list[int]) List of timestamps corresponding to the poses
    """

    # Container to keep track of all observed label sources (@-concatenation of label source names and optional versions or 'any')
    all_label_sources_set: set[str] = set()

    # Setup the set of valid label sources set (@-concatenation of label source names and optional versions or 'any')
    valid_label_sources: set[str] | None = None
    if track_label_sources is not None:
        valid_label_sources = set()
        for track_label_source in track_label_sources:
            # check if specified config label source is versioned
            if len(track_label_source.split("@", 1)) > 1:
                # if version is specified by the config, require this specific version for valid label sources
                valid_label_sources.add(track_label_source)
            else:
                # otherwise any version of this label source type is considered valid
                valid_label_sources.add(track_label_source + "@any")

    # Cache evaluated transformations
    @lru_cache(maxsize=(n_expected_poses := 20 * 10 * 60))  # cache up to 20 minutes of poses at 10Hz
    def get_T_reference_world(reference_frame_id: str, reference_frame_timestamp_us: int) -> np.ndarray:
        T_reference_world = sequence_loader.pose_graph.evaluate_poses(
            reference_frame_id, "world", np.array(reference_frame_timestamp_us, dtype=np.uint64)
        )

        if T_world_world_base is not None:
            T_reference_world = T_world_world_base @ T_reference_world

        return T_reference_world

    @lru_cache(maxsize=n_expected_poses)
    def get_T_reference_rig(reference_frame_id: str, reference_frame_timestamp_us: int) -> Optional[np.ndarray]:
        try:
            return sequence_loader.pose_graph.evaluate_poses(
                reference_frame_id, "rig", np.array(reference_frame_timestamp_us, dtype=np.uint64)
            )
        except KeyError:
            return None

    # Extract all tracks for the given data range
    all_tracks: dict[str, dict] = {}
    for _, row in tqdm.tqdm(cuboids_df.iterrows(), desc="Associate Cuboid Labels -> Tracks", disable=tqdm_disabled):
        observation = (
            ncore.data.CuboidTrackObservation.from_dict(row.to_dict())
            if isinstance(
                row.bbox3, dict
            )  # Check by the bbox3 field to see if the observation is serialized (otherwise it should be BBox3)
            else ncore.data.CuboidTrackObservation(**row.to_dict())
        )

        # evaluate transformations
        T_reference_world = get_T_reference_world(
            observation.reference_frame_id, observation.reference_frame_timestamp_us
        )
        T_reference_rig = get_T_reference_rig(observation.reference_frame_id, observation.reference_frame_timestamp_us)

        # skip self-classifications if rig frame is available
        if T_reference_rig is not None:
            bbox_rig = ncore_transformations.transform_bbox(observation.bbox3.to_array(), T_reference_rig)
            if (
                np.linalg.norm(bbox_rig[:3]) < track_min_centroid_rig_dist_m
            ):  # skip observations that are too close to the rig center
                continue

        # collect observation source (@-concatenation of label source names and optional versions or 'any')
        all_label_sources_set.add(
            versioned_label_source := observation.source.name + f"@{unpack_optional(observation.source_version, 'any')}"
        )

        # handle label source filtering, if enabled
        if valid_label_sources is not None:
            if not (
                (
                    # check versioned source type
                    versioned_label_source in valid_label_sources
                )
                or (
                    # check unversioned source type
                    observation.source.name + "@any" in valid_label_sources
                )
            ):
                # skip label if not in enabled set of label sources
                continue

        if observation.track_id not in all_tracks:
            # instantiate new track
            all_tracks[observation.track_id] = {
                # track-constants:
                "dimension": observation.bbox3.to_array()[3:6],
                "label_class": observation.class_id,
                # per track instance data:
                "poses": [],
                "timestamps_us": [],
            }

        # track to update with this instance's pose / speed data
        track = all_tracks[observation.track_id]

        # Skip if it is a duplicated timestamp
        if observation.timestamp_us in track["timestamps_us"][-2:]:
            continue

        track["timestamps_us"].append(observation.timestamp_us)
        track["poses"].append(
            ncore_transformations.bbox_pose(
                ncore_transformations.transform_bbox(observation.bbox3.to_array(), T_reference_world)
            )
        )

    # Error out if there is more than a single label source in the input datasets and an explicit selection is *not* provided
    if len(all_label_sources_set) > 1 and (track_label_sources is None):
        raise ValueError(
            f"compute_cuboid_tracks: Observed {all_label_sources_set} track label sources in input data but no explicit "
            "track label source specified - please specify *explicit* list of track sources with "
            "the dataset.cuboid_tracks_params.track_label_sources config"
        )

    return all_tracks


