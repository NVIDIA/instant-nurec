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
from typing import List, Literal, cast

import click
import pytorch_lightning as pl
import torch

from nre.config.parse import parse_untyped_config
from nre.datasets import make as make_dataset
from nre.datasets.ncore import NCOREDataSource
from nre.utils.io.mesh import mesh_from_point_cloud, serialize_mesh, smooth_mesh
from nre.utils.types import PointCloud


GAUSSIAN_MESH_RANDOM_SEED = 123  # Seed used for deterministic results throughout the script


@click.command("export-mesh")
@click.option(
    "--config-name",
    type=str,
    help="Hydra config to load - has to contain a dataset specification",
    default="tests/ncore_ds",
    required=True,
)
@click.option(
    "--output-dir",
    type=str,
    help="Path to the output target directory",
    required=True,
)
@click.option(
    "--mesh-basename",
    type=str,
    help="Base filename of output file without suffix",
    default="mesh",
)
@click.option(
    "--camera-id",
    "camera_ids",
    multiple=True,
    type=str,
    help="Cameras to be used (multiple value option, all if not specified)",
    default=None,
)
@click.option(
    "--lidar-id",
    "lidar_ids",
    multiple=True,
    type=str,
    help="Lidars to be used (multiple value option, all if not specified)",
    default=None,
)
@click.option(
    "--format",
    "formats",
    multiple=True,
    type=click.Choice(["ply", "usd"], case_sensitive=False),
    help="Mesh formats to be exported",
    default=["ply", "usd"],
)
@click.option(
    "--smooth/--no-smooth",
    type=bool,
    help="Enable or disable mesh smoothing",
    default=False,
)
@click.option(
    "--n-neighbors",
    type=int,
    help="Number of neighbors used in the k-nn search for the normal estimation",
    default=200,
)
@click.option(
    "--step-frame",
    type=click.IntRange(min=1, max_open=True),
    help="Step used to downsample the number of frames",
    default=1,
)
@click.option(
    "--trim-distance",
    type=float,
    help="Trimming distance to trimm unwanted parts of the mesh (everything that is further away from the input points will be removed)",
    default=0.225,
)
@click.option(
    "--apply-road-segmentation/--no-apply-road-segmentation",
    type=bool,
    help="Enable or disable mesh segmentation into road and non-road faces",
    default=False,
)
@click.option(
    "--export-disjoint-meshes/--disable-disjoint-mesh-export",
    type=bool,
    help="Enable or disable export of disjoing segmented meshes",
    default=False,
)
@click.option(
    "--coord-space",
    type=click.Choice(["nre", "world"]),
    help="Coordinate space of exported mesh.",
    default="world",
)
@click.argument("hydra-args", nargs=-1)
def export_mesh(
    config_name: str,
    output_dir: str,
    mesh_basename: str,
    camera_ids: tuple[str, ...],
    lidar_ids: tuple[str, ...],
    formats: tuple[str, ...],
    smooth: bool,
    n_neighbors: int,
    step_frame: int,
    trim_distance: float,
    hydra_args: list[str],
    apply_road_segmentation: bool,
    export_disjoint_meshes: bool,
    coord_space: str,
):
    """Extracts and exports a triangular mesh for a given dataset"""
    # Initialize logger
    logging.basicConfig(level=logging.INFO)

    pl.seed_everything(GAUSSIAN_MESH_RANDOM_SEED)

    config = parse_untyped_config(config_name=config_name, hydra_args=hydra_args)

    datasource = make_dataset(config.dataset.name, config, split="train").get_datasource()
    if not isinstance(datasource, NCOREDataSource):
        raise NotImplementedError("Mesh export is only implemented for NCOREDataSource")

    # Get point cloud in NRE coordinate space
    point_cloud = PointCloud.collate_fn(
        [
            pc
            for pc in datasource.get_point_clouds(
                device=torch.device("cpu"),
                lidar_ids=list(lidar_ids) if len(lidar_ids) else None,
                camera_ids=list(camera_ids) if len(camera_ids) else None,
                valid_points_only=True,
                non_dynamic_points_only=True,
                color_type=None,
                step_frame=step_frame,
            )
        ],
        device=torch.device("cpu"),
    )

    mesh = mesh_from_point_cloud(
        point_cloud,
        n_neighbors=n_neighbors,
        trim_distance=trim_distance,
        apply_road_segmentation=apply_road_segmentation,
        source_to_target=datasource.world_to_nre.inverse() if coord_space == "world" else None,
    )

    data = serialize_mesh(
        mesh=mesh,
        export_disjoint_meshes=export_disjoint_meshes,
        filename=mesh_basename,
        formats=cast(List[Literal["ply", "usd"]], list(formats)),
    )

    if smooth:
        rig_trajectories = datasource.get_rig_trajectories()
        smoothed_mesh = smooth_mesh(mesh, rig_trajectories)
        data.extend(
            serialize_mesh(
                mesh=smoothed_mesh,
                export_disjoint_meshes=export_disjoint_meshes,
                filename=mesh_basename + "_smoothed",
                formats=cast(List[Literal["ply", "usd"]], list(formats)),
            )
        )

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    for item in data:
        item.save(output_dir_path)
