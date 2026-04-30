# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, cast

import numpy as np
import torch

from omegaconf import DictConfig, ListConfig

from nre.datasets.tracks import CuboidTracks, TrackFlags
from nre.utils.misc import unpack_optional


log = logging.getLogger(__name__)


@dataclass(kw_only=True)
class LayerTrackIds:
    """
    Helper class that manages the object id mapping from the cuboid tracks loaded from dataset and the user's specification.
    """

    config: DictConfig

    # (persistent) track ids for the object layer
    track_ids: List[str] | None = None

    # Mapping from this layer's unique_id to cuboid_track's unique_id
    global_unique_track_idx: torch.Tensor | None = None

    # Whether this is a stand-alone layer that does not depend on tracks
    is_standalone: bool = False

    @dataclass(kw_only=True)
    class Edit:
        """
        Manages the edited cuboid tracks.
        """

        # Edited cuboid tracks
        cuboid_tracks: CuboidTracks

        # Maps from this layer's original unique_id to the edited unique_id
        mapping: torch.Tensor

    tracks_edit: Edit | None = None

    def initialize_from_tracks(self, cuboid_tracks: CuboidTracks) -> None:
        log.info(f"LayerTrackIds: Initializing track filtering from {len(cuboid_tracks.tracks_id)} available tracks")

        layer_mask: np.ndarray = np.ones(len(cuboid_tracks.tracks_id), dtype=bool)

        filtering_reasons = {}  # Track ID -> reason for inclusion/exclusion

        # Validate mutually exclusive options
        if "ids" in self.config and "exclude_ids" in self.config:
            raise ValueError(
                "Cannot specify both 'ids' and 'exclude_ids' in the same layer configuration. "
                "Use 'ids' to explicitly list tracks to include, OR use 'exclude_ids' with other filters (like 'is_dynamic') to exclude specific tracks."
            )

        if "ids" in self.config:
            self.is_standalone = False
            # If ids is provided, then ignore other arguments. This allows user control more easily
            assert isinstance(self.config.ids, ListConfig), "ids should be a list"

            for i, track_id in enumerate(cuboid_tracks.tracks_id):
                if track_id not in self.config.ids:
                    layer_mask[i] = False

        else:
            # Filter based on label classes
            if "label_classes" in self.config:
                self.is_standalone = False
                assert isinstance(self.config.label_classes, ListConfig), "label_classes should be a list"

                for i in range(len(cuboid_tracks.tracks_id)):
                    if cuboid_tracks.tracks_label_class[i] not in self.config.label_classes:
                        layer_mask[i] = False

            # Filter based on dynamic mask
            if "is_dynamic" in self.config:
                self.is_standalone = False
                assert isinstance(self.config.is_dynamic, bool), "is_dynamic should be a boolean"

                if self.config.is_dynamic:
                    filter_mask = cuboid_tracks.get_mask_flags_all(TrackFlags.DYNAMIC)
                else:
                    filter_mask = cuboid_tracks.get_mask_flags_none(TrackFlags.DYNAMIC)
                layer_mask = np.logical_and(layer_mask, filter_mask.cpu().numpy())

            # Stand-alone models does not depend on any tracks
            if "is_standalone" in self.config:
                assert isinstance(self.config.is_standalone, bool), "is_standalone should be a boolean"
                self.is_standalone = self.config.is_standalone
                if self.config.is_standalone:
                    layer_mask[:] = False

        # Handle exclude_ids configuration
        if "exclude_ids" in self.config:
            assert isinstance(self.config.exclude_ids, ListConfig), "exclude_ids should be a list"
            exclude_ids_list = list(self.config.exclude_ids)

            for i, track_id in enumerate(cuboid_tracks.tracks_id):
                if layer_mask[i] and track_id in exclude_ids_list:
                    layer_mask[i] = False

        # Note if self.config is empty, all tracks are included
        if not self.config:
            log.info("No filtering config provided - including all tracks by default")
            for track_id in cuboid_tracks.tracks_id:
                filtering_reasons[track_id] = "included - no filtering criteria"

        if np.any(layer_mask):
            self.is_standalone = False

        self.track_ids = [cuboid_tracks.tracks_id[i] for i in range(len(cuboid_tracks.tracks_id)) if layer_mask[i]]

        # These two are special cases, so add some extra logging
        if "ids" in self.config or "exclude_ids" in self.config:
            log.info(f"LayerTrackIds: track_ids: {self.track_ids}")

            # Log detailed filtering results
            included_tracks = [track_id for i, track_id in enumerate(cuboid_tracks.tracks_id) if layer_mask[i]]
            excluded_tracks = [track_id for i, track_id in enumerate(cuboid_tracks.tracks_id) if not layer_mask[i]]

            log.info(f"Track filtering completed:")
            log.info(f"  - Included: {len(included_tracks)} tracks")
            log.info(f"  - Excluded: {len(excluded_tracks)} tracks")
            log.info(f"  - Standalone layer: {self.is_standalone}")

            if included_tracks:
                log.info("Included tracks:")
                for track_id in included_tracks:
                    reason = filtering_reasons.get(track_id, "unknown reason")
                    log.info(f"  ✓ {track_id}: {reason}")

            if excluded_tracks:
                log.info("Excluded tracks:")
                for track_id in excluded_tracks:
                    reason = filtering_reasons.get(track_id, "unknown reason")
                    log.info(f"  ✗ {track_id}: {reason}")

        self.global_unique_track_idx = torch.from_numpy(np.where(layer_mask)[0]).to(cuboid_tracks.device, torch.long)

    def get_layer_tracks(self, cuboid_tracks: CuboidTracks) -> CuboidTracks:
        return CuboidTracks.Ops.subset_from_indices(cuboid_tracks, unpack_optional(self.global_unique_track_idx))

    def inv_map_unique_track_idx(self, unique_track_idx: torch.Tensor) -> torch.Tensor:
        """
        Maps global unique track idx to local unique track idx in this layer.
        Tracks not found are assigned -1.
        """
        min_track_idx: int = cast(int, torch.min(unique_track_idx).item())
        max_track_idx: int = cast(int, torch.max(unique_track_idx).item())
        assert min_track_idx >= -1, "Invalid unique track idx"

        track_idx_mapping = unpack_optional(self.global_unique_track_idx)
        mapping_tensor = torch.full((max_track_idx + 2,), -1, dtype=torch.int32, device=unique_track_idx.device)
        mapping_tensor[track_idx_mapping + 1] = torch.arange(
            len(track_idx_mapping), dtype=torch.int32, device=unique_track_idx.device
        )

        return mapping_tensor[unique_track_idx + 1]

    def is_empty(self) -> bool:
        assert self.track_ids is not None, "track_ids should be initialized before checking empty"
        return (len(self.track_ids) == 0) if not self.is_standalone else False


class CompositeModel(ABC):
    @abstractmethod
    def get_updated_cuboid_tracks(self) -> CuboidTracks:
        pass
