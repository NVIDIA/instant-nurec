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

import contextlib
import glob
import json
import logging
import zipfile

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Generator, List

import yaml


log = logging.getLogger(__name__)


@dataclass
class Artifact:
    """
    Represents an NRE artifact: a .usdz file which includes the state dict.

    The artifact fields are lazy-loaded and cached. Use `with artifact.temporary_cache(): ...` to automatically clear the cache after the block.
    """

    source: Path

    # for caching
    _checkpoint: BytesIO | None = field(default=None, init=False, repr=False)
    _data_info: Dict[str, Any] | None = field(default=None, init=False, repr=False)
    _datasource_summary: Dict[str, Any] | None = field(default=None, init=False, repr=False)
    _mesh_ply: bytes | None = field(default=None, init=False, repr=False)
    _ground_mesh_ply: bytes | None = field(default=None, init=False, repr=False)
    _metadata: Dict[str, Any] | None = field(default=None, init=False, repr=False)
    _nrend_checkpoint: bytes | None = field(default=None, init=False, repr=False)
    _parsed_config: Dict[str, Any] | None = field(default=None, init=False, repr=False)
    _rig_trajectories: Dict[str, Any] | None = field(default=None, init=False, repr=False)
    _custom_rig_trajectories: Dict[str, Any] | None = field(default=None, init=False, repr=False)
    _sequence_tracks: Dict[str, Any] | None = field(default=None, init=False, repr=False)
    _external_assets: Dict[str, Dict[str, Any]] | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        assert self.source.suffix == ".usdz"

    @staticmethod
    def discover_from_glob(glob_query: str, recursive: bool = True) -> List[Artifact]:
        """
        A factory method to create artifact instances
        """
        assert glob_query.endswith(".usdz"), (
            f"glob query needs to end in .usdz to find valid artifacts (got {glob_query=})."
        )
        return [Artifact(Path(path)) for path in glob.glob(glob_query, recursive=recursive)]

    def clear_cache(self) -> None:
        """
        Clears the cache for the artifact. Use `with artifact.temporary_cache(): ...` to automatically clear the cache after the block.
        """
        self._checkpoint = None
        self._data_info = None
        self._datasource_summary = None
        self._mesh_ply = None
        self._ground_mesh_ply = None
        self._metadata = None
        self._nrend_checkpoint = None
        self._parsed_config = None
        self._rig_trajectories = None
        self._custom_rig_trajectories = None
        self._sequence_tracks = None
        self._external_assets = None

    @contextlib.contextmanager
    def temporary_cache(self) -> Generator[Artifact, None, None]:
        """
        Enters a context after which the artifact cache is automatically cleared.
        """

        try:
            yield self
        finally:
            self.clear_cache()

    @property
    def checkpoint(self) -> BytesIO:
        if self._checkpoint is None:
            with zipfile.ZipFile(self.source, "r") as zip_file:
                self._checkpoint = BytesIO(zip_file.read("checkpoint.ckpt"))

        self._checkpoint.seek(0)  # reset the stream to the beginning if reusing the cache
        return self._checkpoint

    @property
    def data_info(self) -> Dict[str, Any]:
        if self._data_info is None:
            with zipfile.ZipFile(self.source, "r") as zip_file:
                self._data_info = json.load(zip_file.open("data_info.json"))

        return self._data_info

    @property
    def datasource_summary(self) -> Dict[str, Any]:
        if self._datasource_summary is None:
            with zipfile.ZipFile(self.source, "r") as zip_file:
                self._datasource_summary = json.load(zip_file.open("datasource_summary.json"))

        return self._datasource_summary

    @property
    def mesh_ply(self) -> bytes:
        if self._mesh_ply is None:
            with zipfile.ZipFile(self.source, "r") as zip_file:
                self._mesh_ply = zip_file.read("mesh.ply")

        return self._mesh_ply

    @property
    def ground_mesh_ply(self) -> bytes:
        if self._ground_mesh_ply is None:
            with zipfile.ZipFile(self.source, "r") as zip_file:
                self._ground_mesh_ply = zip_file.read("mesh_ground.ply")

        return self._ground_mesh_ply

    @property
    def metadata(self) -> Dict[str, Any]:
        if self._metadata is None:
            with zipfile.ZipFile(self.source, "r") as zip_file:
                self._metadata = yaml.safe_load(zip_file.open("metadata.yaml"))

        return self._metadata

    @property
    def nrend_checkpoint(self) -> bytes:
        if self._nrend_checkpoint is None:
            with zipfile.ZipFile(self.source, "r") as zip_file:
                self._nrend_checkpoint = zip_file.read("volume.nurec")

        return self._nrend_checkpoint

    @property
    def parsed_config(self) -> Dict[str, Any]:
        # It happens frequently that an old artifact is used, so the config returned here is a parsed, versioned config
        # from an earlier version of the program. Once obtained, you need to upgrade this config to the current version
        # by applying upgrade_config() to it. We do not apply this prior to returning the config because you may also
        # want to apply upgrade_model() which requires the pre-upgrade config returned here.
        if self._parsed_config is None:
            with zipfile.ZipFile(self.source, "r") as zip_file:
                self._parsed_config = yaml.safe_load(zip_file.open("parsed_config.yaml"))

        return self._parsed_config

    @property
    def rig_trajectories(self) -> Dict[str, Any]:
        if self._rig_trajectories is None:
            with zipfile.ZipFile(self.source, "r") as zip_file:
                self._rig_trajectories = json.load(zip_file.open("rig_trajectories.json"))

        return self._rig_trajectories

    def load_custom_rig_trajectories(self, custom_rig_trajectories_path: str) -> None:
        """
        Load custom rig trajectories from an external file without overwriting the original artifact trajectories.

        Args:
            custom_rig_trajectories_path: Path to a JSON file containing custom rig trajectories
        """
        with open(custom_rig_trajectories_path, "r") as file:
            self._custom_rig_trajectories = json.load(file)

    @property
    def custom_rig_trajectories(self) -> Dict[str, Any]:
        """
        Get the loaded custom rig trajectories. Must call load_custom_rig_trajectories() first.

        Returns:
            The custom rig trajectories as a dictionary

        Raises:
            ValueError: If custom rig trajectories haven't been loaded yet
        """
        if self._custom_rig_trajectories is None:
            raise ValueError("Custom rig trajectories not loaded. Call load_custom_rig_trajectories(path) first.")
        return self._custom_rig_trajectories

    @property
    def sequence_tracks(self) -> Dict[str, Any]:
        if self._sequence_tracks is None:
            with zipfile.ZipFile(self.source, "r") as zip_file:
                self._sequence_tracks = json.load(zip_file.open("sequence_tracks.json"))

        return self._sequence_tracks

    @property
    def external_assets(self) -> Dict[str, Dict[str, Any]]:
        """
        Lazy-loaded external assets mapping track_id to PLY bytes.
        Expected structure in artifact: external_assets/{track_id}/gs.ply
        """
        if self._external_assets is None:
            self._external_assets = {}
            with zipfile.ZipFile(self.source, "r") as zip_file:
                metadata_dict = {}
                if "external_assets/metadata.yaml" in zip_file.namelist():
                    metadata_content = yaml.safe_load(zip_file.open("external_assets/metadata.yaml"))
                    if "assets" in metadata_content:
                        for asset_info in metadata_content["assets"]:
                            track_id = asset_info["track_id"]
                            metadata_dict[track_id] = {
                                "label_class": asset_info.get("label_class"),
                                "cuboids_dims": asset_info.get("cuboids_dims"),
                            }

                for file_name in zip_file.namelist():
                    path = PurePosixPath(file_name)

                    # Expected pattern: external_assets / {track_id} / {track_id}.ply
                    if path.parts[0] != "external_assets":
                        continue
                    if len(path.parts) != 3:
                        log.warning(f"Skipping external_assets zip entry with unexpected depth: '{file_name}'")
                        continue

                    track_id = path.parts[1]

                    if path.parts[2] != f"{track_id}.ply":
                        log.warning(
                            f"Skipping zip entry with unexpected filename: '{file_name}' (expected '{track_id}.ply')"
                        )
                        continue

                    # Read the PLY file bytes
                    self._external_assets[track_id] = {
                        "ply_bytes": zip_file.read(file_name),
                        "metadata": metadata_dict.get(track_id, {}),
                    }

        return self._external_assets

    @property
    def scene_id(self) -> str:
        """
        The name used to identify the scene via `scene_id=` when requesting
        """
        return self.metadata["scene_id"]
