# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
import tempfile
import zipfile

from pathlib import Path

import lietorch as lt
import numpy as np
import torch

from nre.artifact.artifact import Artifact
from nre.datasets.tracks import CuboidTracks, TracksData
from nre.utils.io.sequence_tracks import serialize_sequence_tracks
from nre.utils.types import CuboidTracksData


def create_test_cuboid_tracks_with_suffixes() -> CuboidTracks:
    """Create a test CuboidTracks object with track IDs that have @suffixes."""
    # Create test data with track IDs that have @suffixes
    tracks_id = [
        "15@scene:obstacles:autolabels:v2",
        "349@scene:obstacles:autolabels:v2",
        "42@scene:obstacles:autolabels:v2",
        "100@scene:obstacles:autolabels:v2",
    ]

    n_tracks = len(tracks_id)
    n_poses_per_track = 3

    # Create poses data (tquat format: [tx, ty, tz, qx, qy, qz, qw])
    tracks_poses_tquat = torch.zeros(n_tracks * n_poses_per_track, 7, dtype=torch.float32)
    tracks_timestamps_us = torch.zeros(n_tracks * n_poses_per_track, dtype=torch.int64)
    tracks_packinfo = torch.zeros(n_tracks, 2, dtype=torch.int64)

    # Fill in the data
    for i, track_id in enumerate(tracks_id):
        start_idx = i * n_poses_per_track
        end_idx = start_idx + n_poses_per_track

        # Set poses (identity poses for simplicity)
        tracks_poses_tquat[start_idx:end_idx, 6] = 1.0  # qw = 1 for identity quaternion

        # Set timestamps (increasing over time)
        tracks_timestamps_us[start_idx:end_idx] = torch.arange(i * 1000, i * 1000 + n_poses_per_track * 100, 100)

        # Set packinfo
        tracks_packinfo[i, 0] = start_idx
        tracks_packinfo[i, 1] = n_poses_per_track

    # Convert to SE3 format
    tracks_poses = lt.SE3(tracks_poses_tquat)

    # Create dimensions (cuboid sizes)
    cuboids_dims = torch.ones(n_tracks, 3, dtype=torch.float32) * 2.0  # 2x2x2 meter cubes

    tracks_data = TracksData(
        tracks_id=tracks_id,
        tracks_poses=tracks_poses,
        tracks_timestamps_us=tracks_timestamps_us,
        tracks_packinfo=tracks_packinfo,
        max_track_n_poses=n_poses_per_track,
        tracks_label_class=["obstacle"] * n_tracks,
        tracks_flags=torch.zeros(n_tracks, dtype=torch.int32),
    )

    cuboidtracks_data = CuboidTracksData(cuboids_dims=cuboids_dims)

    return CuboidTracks(tracks_data=tracks_data, cuboidtracks_data=cuboidtracks_data)


def test_sequence_tracks_json_track_id_cleaning() -> None:
    """Test that JSON serialization cleans track IDs by removing @suffixes."""
    # Create test cuboid tracks with @suffixes
    cuboid_tracks = create_test_cuboid_tracks_with_suffixes()

    # Serialize to JSON format
    sequence_tracks_artifacts = serialize_sequence_tracks(
        sequence_id="test_sequence",
        cuboid_tracks=cuboid_tracks,
        excluded_tracks=None,
        filename="sequence_tracks",
        formats=["json"],
        usd_timestamp_offset_us=0,
    )

    # Parse the JSON to check cleaned track IDs
    json_artifact = sequence_tracks_artifacts[0]
    assert hasattr(json_artifact, "serialized"), "Expected NamedSerialized artifact"
    sequence_tracks_data = json.loads(json_artifact.serialized)
    cleaned_track_ids = sequence_tracks_data["test_sequence"]["tracks_data"]["tracks_id"]

    # Verify all track IDs are cleaned (no @suffixes)
    expected_cleaned_ids = ["15", "349", "42", "100"]

    assert len(cleaned_track_ids) == len(expected_cleaned_ids), (
        f"Expected {len(expected_cleaned_ids)} track IDs, got {len(cleaned_track_ids)}"
    )

    for i, (actual_id, expected_id) in enumerate(zip(cleaned_track_ids, expected_cleaned_ids)):
        assert actual_id == expected_id, f"Track ID {i}: expected '{expected_id}', got '{actual_id}'"
        assert "@" not in actual_id, f"Track ID '{actual_id}' still contains '@' suffix"


def test_sequence_tracks_usda_track_id_cleaning() -> None:
    """Test that USDA serialization cleans track IDs by removing @suffixes."""
    # Create test cuboid tracks with @suffixes
    cuboid_tracks = create_test_cuboid_tracks_with_suffixes()

    # Serialize to USDA format
    sequence_tracks_artifacts = serialize_sequence_tracks(
        sequence_id="test_sequence",
        cuboid_tracks=cuboid_tracks,
        excluded_tracks=None,
        filename="sequence_tracks",
        formats=["usda"],
        usd_timestamp_offset_us=0,
    )

    # Get the USDA content
    usda_artifact = sequence_tracks_artifacts[0]
    assert hasattr(usda_artifact, "stage"), "Expected NamedUSDStage artifact"
    usda_content = usda_artifact.stage.ExportToString()

    # Check that the USDA content doesn't contain @suffixes
    for original_id in ["15@scene:obstacles:autolabels:v2", "349@scene:obstacles:autolabels:v2"]:
        assert original_id not in usda_content, (
            f"USDA file still contains original track ID with @suffix: {original_id}"
        )

    # Check that cleaned IDs are present
    for cleaned_id in ["15", "349", "42", "100"]:
        assert cleaned_id in usda_content, f"USDA file missing cleaned track ID: {cleaned_id}"


def test_usdz_track_id_cleaning() -> None:
    """Test that USDZ serialized items have cleaned track IDs (no @suffixes)."""
    # Create test cuboid tracks with @suffixes
    cuboid_tracks = create_test_cuboid_tracks_with_suffixes()

    # Serialize to USDZ format (which includes sequence_tracks.json)
    sequence_tracks_artifacts = serialize_sequence_tracks(
        sequence_id="test_sequence",
        cuboid_tracks=cuboid_tracks,
        excluded_tracks=None,
        filename="sequence_tracks",
        formats=["json", "usda"],
        usd_timestamp_offset_us=0,
    )

    # Create a temporary USDZ file
    with tempfile.NamedTemporaryFile(suffix=".usdz", delete=False) as tmp_file:
        usdz_path = Path(tmp_file.name)

    try:
        # Create a simple USDZ by zipping the artifacts
        with zipfile.ZipFile(usdz_path, "w", compression=zipfile.ZIP_STORED) as zip_file:
            for artifact in sequence_tracks_artifacts:
                if hasattr(artifact, "serialized") and artifact.serialized:
                    # Add JSON file
                    zip_file.writestr(artifact.filename, artifact.serialized)
                elif hasattr(artifact, "stage") and artifact.stage:
                    # Add USDA file
                    usda_content = artifact.stage.ExportToString()
                    zip_file.writestr(artifact.filename, usda_content)

        # Load the USDZ artifact
        usdz_artifact = Artifact(usdz_path)

        # Get the sequence tracks from the USDZ
        sequence_tracks = usdz_artifact.sequence_tracks

        # Verify the structure
        assert "test_sequence" in sequence_tracks, "Expected 'test_sequence' key in sequence tracks"
        tracks_data = sequence_tracks["test_sequence"]["tracks_data"]
        assert "tracks_id" in tracks_data, "Expected 'tracks_id' in tracks_data"

        # Get the track IDs from the USDZ
        track_ids_from_usdz = tracks_data["tracks_id"]

        # Verify that all track IDs are cleaned (no @suffixes)
        expected_cleaned_ids = ["15", "349", "42", "100"]

        assert len(track_ids_from_usdz) == len(expected_cleaned_ids), (
            f"Expected {len(expected_cleaned_ids)} track IDs, got {len(track_ids_from_usdz)}"
        )

        for i, (actual_id, expected_id) in enumerate(zip(track_ids_from_usdz, expected_cleaned_ids)):
            assert actual_id == expected_id, f"Track ID {i}: expected '{expected_id}', got '{actual_id}'"
            assert "@" not in actual_id, f"Track ID '{actual_id}' still contains '@' suffix"

        # Also verify the USDA file if it exists
        with zipfile.ZipFile(usdz_path, "r") as zip_file:
            if "sequence_tracks.usda" in zip_file.namelist():
                usda_content = zip_file.read("sequence_tracks.usda").decode("utf-8")

                # Check that the USDA content doesn't contain @suffixes
                for original_id in ["15@scene:obstacles:autolabels:v2", "349@scene:obstacles:autolabels:v2"]:
                    assert original_id not in usda_content, (
                        f"USDA file still contains original track ID with @suffix: {original_id}"
                    )

                # Check that cleaned IDs are present
                for cleaned_id in ["15", "349", "42", "100"]:
                    assert cleaned_id in usda_content, f"USDA file missing cleaned track ID: {cleaned_id}"

    finally:
        # Clean up the temporary file
        usdz_path.unlink()


def test_original_vs_cleaned_track_ids() -> None:
    """Test that we can verify the difference between original and cleaned track IDs."""
    # Create test data
    cuboid_tracks = create_test_cuboid_tracks_with_suffixes()

    # Get original track IDs
    original_track_ids = cuboid_tracks.tracks_id

    # Serialize and check what gets written to JSON
    sequence_tracks_artifacts = serialize_sequence_tracks(
        sequence_id="test_sequence",
        cuboid_tracks=cuboid_tracks,
        excluded_tracks=None,
        filename="sequence_tracks",
        formats=["json"],
        usd_timestamp_offset_us=0,
    )

    # Parse the JSON to see the cleaned track IDs
    json_artifact = sequence_tracks_artifacts[0]
    assert hasattr(json_artifact, "serialized"), "Expected NamedSerialized artifact"
    sequence_tracks_data = json.loads(json_artifact.serialized)
    cleaned_track_ids = sequence_tracks_data["test_sequence"]["tracks_data"]["tracks_id"]

    # Verify cleaning worked
    for original, cleaned in zip(original_track_ids, cleaned_track_ids):
        expected_cleaned = original.split("@")[0]
        assert cleaned == expected_cleaned, f"Expected '{expected_cleaned}', got '{cleaned}' for original '{original}'"
