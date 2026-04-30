# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import shutil
import tempfile
import zipfile

from pathlib import Path
from typing import Any, Dict, List

import click
import yaml

from nre.artifact import Artifact


log = logging.getLogger(__name__)


def _strip_track_id_suffix(track_id: str) -> str:
    """Strip '@...' suffix from track_id for clean directory/file names."""
    return track_id.split("@")[0] if "@" in track_id else track_id


def load_external_assets_metadata(external_assets_dir: Path) -> Dict[str, Any]:
    """
    Load the complete metadata from the asset harvester metadata.yaml file.

    Args:
        external_assets_dir: Path to the asset harvester output directory

    Returns:
        Complete metadata dictionary from the metadata.yaml file
    """
    metadata_path = external_assets_dir / "metadata.yaml"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        metadata = yaml.safe_load(f)

    if "assets" not in metadata:
        raise ValueError("No 'assets' field found in metadata.yaml")

    return metadata


def copy_external_assets_to_temp(temp_dir: Path, external_assets_dir: Path, external_metadata: Dict[str, Any]) -> None:
    """
    Copy external assets from asset harvester output to temp directory with correct structure.
    Also creates a metadata.yaml file in the external_assets folder.

    Args:
        temp_dir: Temporary directory where artifact is unzipped
        external_assets_dir: Path to asset harvester output directory
        external_metadata: Complete metadata dictionary from external assets
    """
    assets_metadata_list = []

    for track_id, asset_info in external_metadata["assets"].items():
        # Strip '@...' suffix for backward compatibility with old metadata format
        clean_track_id = _strip_track_id_suffix(track_id)

        # Get the relative PLY file path from metadata
        ply_relative_path = asset_info["ply_file"]
        src_ply = external_assets_dir / ply_relative_path

        if not src_ply.exists():
            raise FileNotFoundError(f"PLY file not found for track {track_id}: {src_ply}")

        # Copy PLY file to external_assets/{clean_track_id}/{clean_track_id}.ply
        dst_dir = temp_dir / "external_assets" / clean_track_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_ply = dst_dir / f"{clean_track_id}.ply"
        shutil.copy2(src_ply, dst_ply)

        # Collect metadata for this asset
        asset_metadata = {
            "track_id": clean_track_id,
            "label_class": asset_info["label_class"],
            "cuboids_dims": asset_info["cuboids_dims"],
        }
        assets_metadata_list.append(asset_metadata)

    # Write metadata.yaml to external_assets folder
    external_assets_metadata_path = temp_dir / "external_assets" / "metadata.yaml"
    external_assets_metadata = {"assets": assets_metadata_list}

    with open(external_assets_metadata_path, "w") as f:
        yaml.dump(external_assets_metadata, f, default_flow_style=False)


def unzip_artifact_to_temp(artifact_path: Path, temp_dir: Path) -> None:
    """
    Unzip the artifact to a temporary directory.

    Args:
        artifact_path: Path to the .usdz artifact file
        temp_dir: Temporary directory to extract to
    """
    with zipfile.ZipFile(artifact_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)


def create_artifact_from_temp(temp_dir: Path, output_artifact_path: Path) -> None:
    """
    Create a new artifact .usdz file from the temporary directory.

    Args:
        temp_dir: Temporary directory containing artifact contents
        output_artifact_path: Path for the output .usdz file
    """
    with zipfile.ZipFile(output_artifact_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(temp_dir)
                zipf.write(file_path, arcname)


def create_edit_assets_config(
    output_artifact_path: Path, external_asset_ids: List[str], external_assets_dir: Path, output_edit_file: Path
) -> None:
    """
    Create the edit-assets.json configuration file.

    Args:
        output_artifact_path: Path to the output artifact
        external_asset_ids: List of external asset track IDs
        external_assets_dir: Path to the asset harvester output directory
        output_edit_file: Path to save the JSON configuration
    """
    assets_metadata_list = []
    external_metadata = load_external_assets_metadata(external_assets_dir)
    for track_id in external_asset_ids:
        # Strip '@...' suffix for backward compatibility with old metadata format
        clean_track_id = _strip_track_id_suffix(track_id)
        asset_info = external_metadata["assets"].get(track_id, {})
        assets_metadata_list.append(
            {
                "track_id": clean_track_id,
                "label_class": asset_info.get("label_class"),
                "cuboid_dims": asset_info.get("cuboids_dims"),
            }
        )

    edit_config = {
        "metadata": {
            "output_artifact_path": str(output_artifact_path),
            "external_assets_metadata": assets_metadata_list,
        },
        "replace": [],
        "remove": [],
        "insert": {
            "asset_ids": [],
            "data": {},
        },
    }

    with open(output_edit_file, "w") as f:
        json.dump(edit_config, f, indent=2)


@click.command("export-external-assets")
@click.option(
    "--artifact-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the input .usdz artifact file",
)
@click.option(
    "--external-assets-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the asset harvester output directory",
)
@click.option(
    "--output-artifact-path",
    type=click.Path(path_type=Path),
    required=False,
    default=None,
    help="Path for the output .usdz artifact file (optional - if not provided, only generates edit-assets.json)",
)
@click.option(
    "--output-edit-file",
    type=click.Path(path_type=Path),
    required=True,
    help="Path for the edit-assets.json output file",
)
def export_external_assets(
    artifact_path: Path, external_assets_dir: Path, output_artifact_path: Path | None, output_edit_file: Path
) -> None:
    """
    Add external assets from asset harvester to an NRE artifact and generate edit-assets.json.

    This script:
    1. Loads an existing NRE artifact (.usdz file)
    2. Loads assets + metdata.yaml from asset harvester output (see apps/asset_harvester/README.md)
        - note: the metadata.yaml is the source of truth for determining
                the track_ids packaged into the output artifact
    3. Merges external assets into the artifact with expected structure (see nre/artifact/artifact.py external_assets property)
    4. Generates a template edit-assets.json file for use with render_grpc.py

    Edit-assets.json description:
    The generated JSON file enables dynamic scene editing during rendering via render_grpc.py (--edit-assets flag).
    The metadata field is informational only and not required for functionality.
    It supports three operations:

    - "replace": List of ReplaceAssetAction objects mapping original_id (track from artifact) to replacement_id (asset in external_assets).
                 original_id must exist in the artifact's sequence_tracks, replacement_id must exist in USDZ'sexternal_assets.
                 Each replacement action has an "object_size" field: a list of 3 floats [size_x, size_y, size_z] representing AABB dimensions.
                 If "object_size" is missing or empty, render_grpc.py will fall back to "cuboid_dims" from the metadata for that replacement_id.
                 Example: [{"original_id": "8", "replacement_id": "13", "object_size": [4.5, 2.0, 1.8]},
                           {"original_id": "18", "replacement_id": "22", "object_size": []},
                           {"original_id": "6", "replacement_id": "7"}]
                 This will (1) replace track_id '8' with asset '13' using specified dimensions
                           (2) replace track_id '18' with asset '22', use 22's cuboid_dims
                           (3) replace track_id '6' with asset '7', use 7's cuboid_dims

    - "remove": List of track IDs (strings) to remove from rendering. IDs must be from artifact's sequence_tracks.
                Tracks are filtered out during render request creation.
                Example: ["13","8"]

    - "insert": Dictionary containing CuboidTracks data for new dynamic objects to add to the scene.
                Requires a "data" field containing the full CuboidTracks.to_dict() output.
                Requires an "asset_ids" field that decouples track identity from asset selection, enabling asset swapping.
                Track IDs can be any string that doesn't conflict with existing id's in USDZ's sequence_tracks.json.
                The asset_ids must exist in the USDZ's external_assets folder, must be same length as tracks_id, correspond 1:1 in order.

                Example: {
                    "asset_ids": ["8", "13"],  # Enables using different assets
                    "data": {
                        "tracks_data": {
                            "tracks_id": ["car_1", "car_2"], # car_1 will use asset_id "8" and car_2 will use asset_id "13"
                            "tracks_poses": [...],
                            ...
                        },
                        "cuboidtracks_data": {
                            "cuboids_dims": [[4.5, 2.0, 1.8], [3.0, 1.5, 1.5]]
                        }
                    }
                }

    Template:
        {
            "metadata": {
                // metadata is informational only and not required for functionality
                "output_artifact_path": "/path/to/output.usdz",
                "external_asset_ids": ["8", "13", "18"]
            },
            "replace": [
                {"original_id": "8", "replacement_id": "13", "object_size": [4.5, 2.0, 1.8]},
                {"original_id": "18", "replacement_id": "22", "object_size": []},
                {"original_id": "6", "replacement_id": "7"}
            ],
            "remove": ["13"],
            "insert": {
                "asset_ids": ["8", "13"],  // Optional: 1:1 indexing with tracks_id for asset swapping
                "data": {
                    // CuboidTracks.to_dict() output
                }
            }
        }
    """
    if artifact_path.suffix != ".usdz":
        raise ValueError(f"Artifact must be a .usdz file, got: {artifact_path}")

    if output_artifact_path is not None:
        if output_artifact_path.suffix != ".usdz":
            raise ValueError(f"Output artifact must be a .usdz file, got: {output_artifact_path}")
        output_artifact_path.parent.mkdir(parents=True, exist_ok=True)

    output_edit_file.parent.mkdir(parents=True, exist_ok=True)

    artifact = Artifact(artifact_path)

    try:
        # Load external assets metadata
        log.info("Loading metadata from external assets")
        external_metadata = load_external_assets_metadata(external_assets_dir)
        external_asset_ids = list(external_metadata["assets"].keys())
        log.info(f"External asset IDs: {sorted(external_asset_ids)}")

        # Create modified artifact with external assets (if output path is provided)
        if output_artifact_path is not None:
            log.info(f"Creating modified artifact with external assets: {output_artifact_path}")
            temp_dir = Path(tempfile.mkdtemp(prefix="artifact_external_assets_"))
            try:
                unzip_artifact_to_temp(artifact_path, temp_dir)
                copy_external_assets_to_temp(temp_dir, external_assets_dir, external_metadata)
                create_artifact_from_temp(temp_dir, output_artifact_path)

            finally:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
        else:
            log.info("Skipping new artifact creation (no output path provided)")

        # Generate edit-assets.json
        # Use output_artifact_path if provided, otherwise use the input artifact_path
        artifact_path_for_config = output_artifact_path if output_artifact_path is not None else artifact_path
        create_edit_assets_config(artifact_path_for_config, external_asset_ids, external_assets_dir, output_edit_file)
        log.info(f"Generated edit-assets configuration: {output_edit_file}")

    finally:
        artifact.clear_cache()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    export_external_assets()
