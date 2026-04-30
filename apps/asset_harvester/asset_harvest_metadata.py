# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import base64
import io
import logging
import os

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

import numpy as np

from asset_harvester.ncore_parser.mvdata import MVData  # pycena: skip
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from PIL import Image

from apps.asset_harvester.utils import strip_track_id_suffix
from nre.config.asset_harvest import Asset, AssetHarvestingConfig, AssetHarvestingMetadata, MultiViewData, MVDataView
from nre.config.parse import parse_untyped_config
from nre.utils.types import NamedSerialized


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def save_image_to_file(image_array: np.ndarray, output_path: Path, format: str = "JPEG") -> str:
    """Save numpy image array to file and return relative path from output directory"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.fromarray(image_array.astype(np.uint8))
    img.save(output_path, format=format, quality=95 if format == "JPEG" else None)

    # Return relative path - we'll make it relative to the metadata file location
    return str(output_path)


def load_image_from_file(filepath: str, base_path: Optional[Path] = None) -> np.ndarray:
    """Load image from file path (relative to base_path if provided)"""
    if base_path:
        full_path = base_path / filepath
    else:
        full_path = Path(filepath)

    if not full_path.exists():
        raise FileNotFoundError(f"Image file not found: {full_path}")

    img = Image.open(full_path)
    return np.array(img)


def encode_image_to_base64(image_array: np.ndarray, format: str = "JPEG") -> str:
    """Convert numpy image array to base64 string - DEPRECATED, kept for reference"""
    img = Image.fromarray(image_array.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format=format, quality=95 if format == "JPEG" else None)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def decode_base64_to_image(base64_str: str) -> np.ndarray:
    """Decode base64 string back to numpy image array - DEPRECATED, kept for reference"""
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    return np.array(img)


# Type variables for generic converter
TSourceFormat = TypeVar("TSourceFormat")  # Source format (e.g., MVData)
TTargetFormat = TypeVar("TTargetFormat")  # Target format (e.g., Asset)


class MetadataConverter(ABC, Generic[TSourceFormat, TTargetFormat]):
    """Abstract base class for converting between different metadata formats."""

    @abstractmethod
    def convert_to_metadata(self, source_objects: Dict[str, TSourceFormat]) -> Dict[str, TTargetFormat]:
        """Convert from source format to target format.

        Args:
            source_objects: Dictionary of id -> source objects

        Returns:
            Dictionary of id -> target objects
        """
        pass

    @abstractmethod
    def convert_from_metadata(self, source_objects: Dict[str, TTargetFormat]) -> Dict[str, TSourceFormat]:
        """Convert from target format back to source format.

        Args:
            source_objects: Dictionary of id -> source objects

        Returns:
            Dictionary of id -> target objects
        """
        pass


class MultiViewMetadataConverter(MetadataConverter[MVData, Asset]):
    """Converter between MVData and Asset formats - used to convert ViewsExtractor output / Multiview Generator input"""

    def __init__(self, output_dir: Path, base_path: Optional[Path] = None):
        """Initialize converter with directories for saving/loading images.

        Args:
            output_dir: Base directory where images will be saved during conversion to metadata
            base_path: Base path for resolving relative paths when converting from metadata.
                      If None, paths are treated as absolute.
        """
        super().__init__()
        self.output_dir = Path(output_dir)
        self.base_path = base_path

    def convert_to_metadata(self, source_objects: Dict[str, MVData]) -> Dict[str, Asset]:
        """Convert MVData objects to Asset format, saving images to disk.

        Args:
            source_objects: Dictionary of track_id -> MVData objects

        Returns:
            Dictionary of clean_track_id -> Asset with file paths (track_id suffix stripped)
        """
        mvdata_assets = {}

        for track_id, mvdata in source_objects.items():
            clean_track_id = strip_track_id_suffix(track_id)
            mvdata_assets[clean_track_id] = self._convert_single_to_metadata(mvdata, track_id)

        return mvdata_assets

    def convert_from_metadata(self, source_objects: Dict[str, Asset]) -> Dict[str, MVData]:
        """Convert Asset objects back to MVData, loading images from disk.

        Args:
            source_objects: Dictionary of track_id -> Asset objects

        Returns:
            Dictionary of track_id -> MVData objects
        """
        mvdata_objects = {}

        for track_id, asset in source_objects.items():
            mvdata = self._convert_single_from_metadata(asset)
            mvdata_objects[track_id] = mvdata

        return mvdata_objects

    def set_base_path(self, base_path: Path) -> None:
        """Update the base path for resolving relative paths.

        Args:
            base_path: New base path for image loading
        """
        self.base_path = base_path

    # Private helper methods
    def _convert_single_to_metadata(self, mvdata: MVData, track_id: str) -> Asset:
        """Convert a single MVData instance to Asset."""
        views = []
        num_frames = mvdata.frames.shape[0]
        clean_track_id = strip_track_id_suffix(track_id)

        # Create track-specific directory with views_extractor_output subdirectory
        track_dir = self.output_dir / clean_track_id / "views_extractor_output"
        track_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_frames):
            frame_path = track_dir / f"frame_{i:03d}.jpg"
            frame_rel_path = save_image_to_file(mvdata.frames[i], frame_path, format="JPEG")
            frame_rel_path = os.path.relpath(frame_path, self.output_dir)

            mask_rel_path = None
            if mvdata.masks_instance and i < len(mvdata.masks_instance):
                mask_path = track_dir / f"mask_{i:03d}.png"
                # We scale up by 255.0 since mask2former saves masks in [0,1.0] range
                mask_rel_path = save_image_to_file(mvdata.masks_instance[i] * 255.0, mask_path, format="PNG")
                mask_rel_path = os.path.relpath(mask_path, self.output_dir)

            view = MVDataView(
                frame=frame_rel_path,
                instance_mask=mask_rel_path,
                cam_pose=mvdata.cam_poses[i].tolist(),
                dist=float(mvdata.dists[i]),
                fov=float(mvdata.fov[i]),
                sensor_id=mvdata.sensor_id[i] if i < len(mvdata.sensor_id) else "",
            )
            views.append(view)

        multiview_data = MultiViewData(
            bbox_pos=mvdata.bbox_pos.tolist() if hasattr(mvdata.bbox_pos, "tolist") else list(mvdata.bbox_pos),
            views=views,
        )

        return Asset(
            clip_id=mvdata.clip_id,
            track_id=clean_track_id,
            label_class=mvdata.npct,
            cuboids_dims=mvdata.lwh.tolist() if mvdata.lwh is not None else [],
            ply_file="",
            multiview_data=multiview_data,
        )

    def _convert_single_from_metadata(self, asset: Asset) -> MVData:
        """Convert a single Asset back to MVData."""
        if self.base_path is None:
            raise ValueError("base_path must be set before converting from metadata")

        frames = []
        masks_instance = []
        cam_poses = []
        dists = []
        fovs = []
        sensor_ids = []

        for view in asset.multiview_data.views:
            frames.append(load_image_from_file(view.frame, self.base_path))

            if view.instance_mask:
                # We scale down by 255.0 since we saved the masks in [0,255.0] range
                masks_instance.append(load_image_from_file(view.instance_mask, self.base_path) / 255.0)

            cam_poses.append(view.cam_pose)
            dists.append(view.dist)
            fovs.append(view.fov)
            sensor_ids.append(view.sensor_id)

        mvdata = MVData(
            clip_id=asset.clip_id,
            obj_id=asset.track_id,
            frames=np.array(frames),
            cam_poses=np.array(cam_poses),
            dists=np.array(dists),
            fov=np.array(fovs),
            npct=asset.label_class,
            bbox_pos=np.array(asset.multiview_data.bbox_pos),
            bbox_pix=[],
            lwh=np.array(asset.cuboids_dims) if asset.cuboids_dims else None,
            sensor_id=sensor_ids,
            masks_instance=masks_instance if masks_instance else None,
        )

        return mvdata


class AssetMetadataManager:
    """Manages metadata persistence and conversion for asset harvesting pipeline."""

    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.metadata: Optional[AssetHarvestingMetadata] = None
        self.converter = MultiViewMetadataConverter(output_dir=self.save_dir)

    @property
    def has_metadata(self) -> bool:
        """Check if metadata is loaded/created."""
        return self.metadata is not None

    def load_from_file(self, metadata_file: str, hydra_args: Optional[List[str]] = None) -> AssetHarvestingMetadata:
        """Load metadata from YAML file."""
        if not Path(metadata_file).exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        self.metadata = self._parse_metadata_file(metadata_file, hydra_args)
        return self.metadata

    def _parse_metadata_file(
        self, metadata_file: str, hydra_args: Optional[List[str]] = None, config_dir: str = "."
    ) -> AssetHarvestingMetadata:
        """Parse and validate metadata from file."""
        # Clear GlobalHydra if already initialized to prevent conflicts (this is a bugfix)
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        logger.info(f"Parsing metadata file: {metadata_file}")

        if hydra_args is None:
            hydra_args = []

        # Process hydra args for metadata mode: prefix AssetHarvestingConfig-related args with 'config.'
        processed_hydra_args = []
        metadata_direct_fields = set(AssetHarvestingMetadata.model_fields.keys())

        for arg in hydra_args:
            # Check if this is an override argument (contains '=')
            if "=" in arg:
                key_part = arg.split("=")[0]
                # Check if this targets a top-level field that belongs directly to AssetHarvestingMetadata
                if any(key_part.startswith(field) for field in metadata_direct_fields):
                    # This targets AssetHarvestingMetadata directly
                    processed_hydra_args.append(arg)
                elif key_part.startswith("config."):
                    # Already has config prefix, use as-is
                    processed_hydra_args.append(arg)
                else:
                    # This likely targets AssetHarvestingConfig, so prefix with 'config.'
                    processed_hydra_args.append(f"config.{arg}")
            else:
                # Non-override arguments (like mode changes) pass through unchanged
                processed_hydra_args.append(arg)

        untyped_config = parse_untyped_config(
            config_name=metadata_file, hydra_args=processed_hydra_args, config_dir=config_dir
        )
        typed_config = AssetHarvestingMetadata.model_validate(untyped_config, context={"config_name": metadata_file})

        return typed_config

    def save(self) -> None:
        """Save current metadata to save_dir."""
        if not self.metadata:
            raise ValueError("No metadata to save. Create or load metadata first.")

        metadata_path = self.save_dir / "metadata.yaml"
        self._save_metadata_file(metadata_path)

    def _save_metadata_file(self, metadata_path: Path) -> None:
        """Save metadata to specified path."""
        if not self.metadata:
            raise ValueError("No metadata to save")

        metadata_dict = self.metadata.model_dump()
        # Save using OmegaConf to ensure Hydra compatibility with the @package _global_ directive
        metadata_yaml = "# @package _global_ \n\n" + OmegaConf.to_yaml(OmegaConf.create(metadata_dict))

        serialized = NamedSerialized(
            filename=f"metadata.yaml",
            serialized=metadata_yaml,
        )
        serialized.save(Path(metadata_path.parent))

    # Creation/Loading methods
    def convert_mvdata_to_assets_metadata(self, mvdata: Dict[str, MVData]) -> Dict[str, Asset]:
        """Convert MVData objects from ViewsExtractor to Asset format.

        Args:
            mvdata: Dictionary of track_id -> MVData objects

        Returns:
            Dictionary of track_id -> Asset with file paths
        """
        # Use converter to transform MVData to Asset
        return self.converter.convert_to_metadata(mvdata)

    def set_metadata(self, metadata: AssetHarvestingMetadata) -> None:
        """Set the metadata object.

        Args:
            metadata: AssetHarvestingMetadata object to store
        """
        self.metadata = metadata

    # Conversion methods
    def get_mvdata(self, metadata_file_path: Path, track_ids: Optional[List[str]] = None) -> Dict[str, MVData]:
        """Get MVData objects for MultiviewGenerator from current metadata.

        Args:
            metadata_file_path: Path to the metadata file (needed to resolve relative image paths)
            track_ids: Optional list of specific track IDs to extract
        """
        if not self.metadata:
            raise ValueError("No metadata available. Create or load metadata first.")

        # Filter assets if track_ids provided
        if track_ids:
            available_tracks = set(self.metadata.assets.keys()) if self.metadata.assets else set()
            requested_tracks = set(track_ids)
            missing_tracks = requested_tracks - available_tracks

            if missing_tracks:
                raise ValueError(
                    f"Requested tracks not found in metadata: {missing_tracks}. Available tracks: {available_tracks}"
                )

            # Filter to requested tracks
            filtered_assets = (
                {track_id: self.metadata.assets[track_id] for track_id in track_ids} if self.metadata.assets else {}
            )
            logger.info(f"Converting requested tracks: {track_ids}")
        else:
            # Use all tracks
            filtered_assets = self.metadata.assets or {}
            logger.info(f"Converting all tracks: {list(filtered_assets.keys())}")

        # Set base path for the converter as we set relative filepaths to metadata file
        self.converter.set_base_path(metadata_file_path.parent)

        # Convert using the converter
        return self.converter.convert_from_metadata(filtered_assets)

    def get_runtime_config(self) -> AssetHarvestingConfig:
        """Get config from current metadata."""
        if not self.metadata:
            raise ValueError("No metadata available. Create or load metadata first.")

        return self.metadata.config
