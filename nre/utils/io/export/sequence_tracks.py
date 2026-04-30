# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging

from pathlib import Path
from typing import Optional

import click

from nre.config.parse import parse_typed_config
from nre.datasets import make as make_dataset
from nre.datasets.ncore import NCOREDataSource
from nre.datasets.tracks import CuboidTracks
from nre.systems import make as make_system
from nre.utils.io.rig_trajectories import rig_trajectories_time_range
from nre.utils.io.sequence_tracks import serialize_sequence_tracks


@click.command("export-sequence-tracks")
@click.option(
    "--config-name",
    type=str,
    help="Hydra config to load - has to contain a dataset specification",
    default="tests/ncore_ds",
    required=True,
)
@click.option(
    "--checkpoint-path",
    type=str,
    help="Checkpoint file path to load",
    required=False,
    default=None,
)
@click.option(
    "--output-dir",
    type=str,
    help="Path to the output target directory",
    required=True,
)
@click.option(
    "--dynamic-only/--all-tracks",
    type=bool,
    help="Return dynamic object tracks only or return all tracks (default behavior).",
    default=False,
)
@click.option(
    "--world-frame/--nre-frame",
    type=bool,
    help="Return tracks in world frame (default behavior) or in NRE frame.",
    default=True,
)
@click.option(
    "--format",
    "formats",
    multiple=True,
    type=click.Choice(["json", "usda"], case_sensitive=False),
    help="Mesh formats to be exported",
    default=["json", "usda"],
)
@click.option(
    "--controllable-only/--all-tracks",
    type=bool,
    help="Return controllable object tracks only.",
    default=False,
)
@click.argument("hydra-args", nargs=-1)
def export_sequence_tracks(
    config_name: str,
    checkpoint_path: str,
    output_dir: str,
    dynamic_only: bool,
    world_frame: bool,
    formats: tuple[str, ...],
    controllable_only: bool,
    hydra_args: list[str],
) -> None:
    """Extracts and exports the sequence tracks for a given dataset"""
    # Initialize logger
    logging.basicConfig(level=logging.INFO)

    if dynamic_only and ("usd" in formats or "usda" in formats):
        raise ValueError("Exporting only dynamic sequence tracks is not supported for USD format!")

    config = parse_typed_config(config_name=config_name, hydra_args=hydra_args)

    exported_formats = list(formats)
    exported_data = []

    has_cuboid_tracks = False
    if checkpoint_path is not None:
        if config.resume is None:
            # FIXME: Configs should not be mutable
            config.resume = checkpoint_path
        # create the system to have access to the model
        system = make_system(config.system.name, config, load_from_checkpoint=checkpoint_path)
        has_cuboid_tracks = hasattr(system.model, "get_updated_cuboid_tracks")

    if dynamic_only and has_cuboid_tracks:
        raise ValueError("Exporting only dynamic sequence tracks is not supported for model with cuboid tracks!")

    # get the time range from datasource
    datasource = make_dataset(config.dataset.name, config, split="train").get_datasource()
    if not isinstance(datasource, NCOREDataSource):
        raise NotImplementedError("Sequence tracks export is only implemented for NCOREDataSource")
    rig_time_range = rig_trajectories_time_range(datasource.get_rig_trajectories())

    # export the raw datasource tracks
    cuboid_tracks = datasource.get_cuboid_tracks(dynamic_only=dynamic_only, world_frame=world_frame)
    # exclude dynamic tracks only for static models (without cuboid tracks)
    dynamic_tracks = datasource.cuboidtracks_dynamic

    # if available merge the updated tracks from the model with the datasource tracks
    if has_cuboid_tracks:
        assert hasattr(system.model, "get_updated_cuboid_tracks")

        model_cuboid_tracks = CuboidTracks.Ops.transform_with_frame_conversion(
            getattr(system.model, "get_updated_cuboid_tracks")(),
            datasource.world_to_nre.inverse(),
            None,
        )
        if controllable_only:
            cuboid_tracks = model_cuboid_tracks
        else:
            tracks_in_both = list(set(cuboid_tracks.tracks_id) & set(model_cuboid_tracks.tracks_id))
            tracks_datasource_only = list(set(cuboid_tracks.tracks_id) - set(model_cuboid_tracks.tracks_id))

            cuboid_tracks = CuboidTracks.Ops.concatenate(
                [
                    CuboidTracks.Ops.subset_from_tracks_id(cuboid_tracks, tracks_datasource_only),
                    CuboidTracks.Ops.subset_from_tracks_id(model_cuboid_tracks, tracks_in_both),
                ]
            )

    # Alpasim depends on the sequence tracks being keyed with `dummy_chunk_id`
    exported_data = serialize_sequence_tracks(
        sequence_id="dummy_chunk_id",
        cuboid_tracks=cuboid_tracks,
        excluded_tracks=dynamic_tracks,
        formats=exported_formats,
        usd_timestamp_offset_us=rig_time_range.start,
    )

    # Write the data
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    for item in exported_data:
        item.save(output_dir_path)
