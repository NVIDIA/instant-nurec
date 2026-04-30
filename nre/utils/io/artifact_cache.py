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
import zipfile

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import List

from nre.config.checkpoint import ArtifactConfig
from nre.utils.io.usd_default_layer import serialize_usd_default_layer
from nre.utils.io.utils import USDReferences
from nre.utils.misc import dataclass_items
from nre.utils.types import ArtifactContents, NamedSerialized, NamedUSDStage


logger = logging.getLogger(__name__)


@dataclass
class ArtifactCache:
    artifact_config: ArtifactConfig

    checkpoint: NamedSerialized | None = None
    parsed_config: NamedSerialized | None = None

    data_info: NamedSerialized | None = None
    metadata: NamedSerialized | None = None
    datasource_summary: NamedSerialized | None = None

    meshes: ArtifactContents | None = None
    # Proxy meshes are all USD meshes in `meshes`, augmented with USD-specific information
    proxy_meshes: USDReferences | None = None

    rig_trajectories: ArtifactContents | None = None
    sequence_tracks: ArtifactContents | None = None
    # USD references of the sequence tracks to be associated to the NeRF prim
    usd_sequence_tracks: USDReferences | None = None

    nrend_checkpoint: ArtifactContents | None = None

    def write_to_usdz(self, file_path: Path) -> None:
        def is_artifact_contents(obj) -> bool:
            if not isinstance(obj, Sequence):
                return False
            return all(isinstance(item, (NamedSerialized, NamedUSDStage)) for item in obj)

        # filter from the root stage the meshes and sequences_tracks which are included under the NeRF XForm
        def include_usd_stage(element: NamedUSDStage):
            return (
                isinstance(element, NamedUSDStage)
                and not (element in [stage for stage, _ in self.proxy_meshes or []])
                and not (element in [stage for stage, _ in self.usd_sequence_tracks or []])
            )

        # Discover USD files in dataclass items.
        # Assume that all files are stored in the same directory, hence the path is equal to the filename.
        refs: List[NamedUSDStage] = []
        for _, member in dataclass_items(self):
            if include_usd_stage(member):
                refs.append(member)
            elif is_artifact_contents(member):
                for member_list_elem in member:
                    if include_usd_stage(member_list_elem):
                        refs.append(member_list_elem)

        usd_default_layer = NamedUSDStage(filename="default.usda", stage=serialize_usd_default_layer(refs))

        # Make sure path to usdz-file exists, as zipfile.Zipfile(...) can only create files, not directories
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(file_path, "w", compression=zipfile.ZIP_STORED) as zip_file:
            usd_default_layer.save_to_zip(zip_file)

            for _, member in dataclass_items(self):
                if member is not None:
                    if isinstance(member, NamedSerialized) or isinstance(member, NamedUSDStage):
                        member.save_to_zip(zip_file)
                    elif is_artifact_contents(member):
                        for member_list_elem in member:
                            member_list_elem.save_to_zip(zip_file)

        logger.info(f"File {file_path} created successfully.")
