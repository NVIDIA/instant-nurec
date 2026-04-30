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

import click
import numpy as np
import torch

from nre.config.parse import parse_untyped_config
from nre.datasets import make as make_dataset
from nre.datasets.ncore import NCOREDataSource
from nre.utils.io.ground_mesh import get_nominal_ground_point_under_lidar, reconstruct_ground_mesh_from_points
from nre.utils.io.ply import save_ply


logger = logging.getLogger(__name__)


@click.command("export-ground-mesh")
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
    "--step-frame",
    type=click.IntRange(min=1, max_open=True),
    help="Frame step used to skip poses when fusing point clouds (>1 skips poses)",
    default=1,
)
@click.option(
    "--export-per-frame-diagnostics",
    is_flag=True,
    help="Export intermediate processing results per Lidar frame",
    default=False,
)
@click.option(
    "--export-meshing-diagnostics",
    is_flag=True,
    help="Export extra files (incl. input segmented points, mesh before smoothing) to PLY to help diagnose meshing",
    default=False,
)
@click.option(
    "--num-plane-hypotheses",
    type=int,
    help="Number of plane hypotheses to evaluate (>0)",
    default=100,
)
@click.option(
    "--min-ray-length",
    type=float,
    help="Minimum length of rays to filter unwanted points near the sensor (>0)",
    default=0.1,
)
@click.option(
    "--ground-compat-max-distance",
    type=float,
    help="Max distance between valid ground plane hypotheses and the ground control point under the ego vehicle",
    default=0.5,
)
@click.option(
    "--ground-compat-max-angle-deg",
    type=float,
    help="Max angle between valid ground plane hypotheses and the up direction",
    default=60.0,
)
@click.option(
    "--plane-max-distance",
    type=float,
    help="Max distance of a point from a plane hypothesis to be considered inlier",
    default=0.3,
)
@click.option(
    "--plane-max-angle-deg",
    type=float,
    help="Max angle between a point normal and the normal of a plane hypothesis for the point to be considered inlier",
    default=30.0,
)
@click.option(
    "--plane-min-eval-points",
    type=int,
    help="Min required number of points to evaluate plane hypotheses (>0), skipping segmentation if not reached",
    default=1000,
)
@click.option(
    "--plane-max-eval-points",
    type=int,
    help="Max number of points to evaluate plane hypotheses (>0), subsampling if exceeded",
    default=10000,
)
@click.option(
    "--plane-min-inliers",
    type=int,
    help="Min number of inlier points to accept a segmentation result (>0)",
    default=1000,
)
@click.option(
    "--enable-plane-extension",
    is_flag=True,
    help="Allow all points of a spin that are compatible with the dominant plane to be marked as ground point",
    default=False,
)
@click.option(
    "--voxel-size",
    type=float,
    help="Voxel size (in world units) for point cloud downsampling to be applied before meshing",
    default=0.1,
)
@click.option(
    "--min-points-per-voxel",
    type=int,
    help="Minimum number of points per voxel",
    default=1,
)
@click.option(
    "--smoothing-passes",
    type=int,
    help="Number of mesh smoothing passes to apply (>=0)",
    default=10,
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Verbose mode",
    default=False,
)
@click.option(
    "--valid-points-only",
    is_flag=True,
    help="Only use valid points for ground mesh reconstruction",
    default=False,  # BUG:5195915 WAR
)
@click.option(
    "--colored",
    is_flag=True,
    help="Color mesh vertices by projecting camera RGB onto LiDAR points",
    default=False,
)
@click.argument("hydra-args", nargs=-1)
def export_ground_mesh(
    config_name: str,
    output_dir: str,
    step_frame: int,
    export_per_frame_diagnostics: bool,
    export_meshing_diagnostics: bool,
    num_plane_hypotheses: int,
    min_ray_length: float,
    ground_compat_max_distance: float,
    ground_compat_max_angle_deg: float,
    plane_max_distance: float,
    plane_max_angle_deg: float,
    plane_min_eval_points: int,
    plane_max_eval_points: int,
    plane_min_inliers: int,
    enable_plane_extension: bool,
    voxel_size: float,
    min_points_per_voxel: int,
    smoothing_passes: int,
    verbose: bool,
    valid_points_only: bool,
    colored: bool,
    hydra_args: list[str],
):
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s")
    config = parse_untyped_config(config_name=config_name, hydra_args=hydra_args)

    # Check parameters of the algorithm
    if num_plane_hypotheses < 1:
        raise ValueError("num_plane_hypotheses must be positive")
    if min_ray_length <= 0.0:
        raise ValueError("min_ray_length must be positive")
    if ground_compat_max_distance <= 0.0:
        raise ValueError("ground_compat_max_distance must be positive")
    if ground_compat_max_angle_deg <= 0.0:
        raise ValueError("ground_compat_max_angle_deg must be positive")
    if plane_max_distance <= 0.0:
        raise ValueError("plane_max_distance must be positive")
    if plane_max_angle_deg <= 0.0:
        raise ValueError("plane_max_angle_deg must be positive")
    if num_plane_hypotheses < 1:
        raise ValueError("num_plane_hypotheses must be positive")
    if num_plane_hypotheses < 1:
        raise ValueError("num_plane_hypotheses must be positive")
    if num_plane_hypotheses < 1:
        raise ValueError("num_plane_hypotheses must be positive")
    if plane_min_eval_points < 1:
        raise ValueError("plane_min_eval_points must be positive")
    if plane_max_eval_points < 1:
        raise ValueError("plane_max_eval_points must be positive")
    if plane_min_inliers < 1:
        raise ValueError("plane_min_inliers must be positive")
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")
    if min_points_per_voxel < 1:
        raise ValueError("min_points_per_voxel must be positive")
    if smoothing_passes <= 0.0:
        raise ValueError("voxel_size must be positive")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    datasource = make_dataset(config.dataset.name, config, split="train").get_datasource()
    if not isinstance(datasource, NCOREDataSource):
        raise NotImplementedError("Only NCORE datasets are supported")

    # Yields a point cloud per Lidar spin.
    point_cloud_generator = datasource.get_point_clouds(
        device=torch.device("cpu"),
        lidar_ids=None,
        camera_ids=None,
        valid_points_only=valid_points_only,
        non_dynamic_points_only=True,
        color_type="camera-rgb" if colored else None,
        step_frame=step_frame,
    )

    smooth_vertices, triangles, vertices, points, normals, road_mask, initial_road_mask, vertex_colors = (
        reconstruct_ground_mesh_from_points(
            point_cloud_generator=point_cloud_generator,
            min_ray_length=min_ray_length,
            ground_control_point=get_nominal_ground_point_under_lidar(datasource),
            ground_compat_max_distance=ground_compat_max_distance,
            ground_compat_max_angle_deg=ground_compat_max_angle_deg,
            num_plane_hypotheses=num_plane_hypotheses,
            plane_max_distance=plane_max_distance,
            plane_max_angle_deg=plane_max_angle_deg,
            plane_min_eval_points=plane_min_eval_points,
            plane_max_eval_points=plane_max_eval_points,
            plane_min_inliers=plane_min_inliers,
            enable_plane_extension=enable_plane_extension,
            voxel_size=voxel_size,
            min_points_per_voxel=min_points_per_voxel,
            smoothing_passes=smoothing_passes,
            export_per_frame_diagnostics=export_per_frame_diagnostics,
            output_path=output_path,
            points_to_world_transf=datasource.world_to_nre.inverse(),
            verbose=verbose,
        )
    )

    save_ply(str(output_path / "mesh_ground.ply"), smooth_vertices, triangles, colors=vertex_colors, logger=logger)

    if export_meshing_diagnostics:
        save_ply(str(output_path / "mesh_ground_before_smoothing.ply"), vertices, triangles, logger=logger)
        save_ply(
            str(output_path / "input_points_road_prior.ply"),
            vertices=points[initial_road_mask],
            normals=normals[initial_road_mask],
            logger=logger,
        )
        save_ply(
            str(output_path / "input_points_road.ply"),
            vertices=points[road_mask],
            normals=normals[road_mask],
            logger=logger,
        )
        nonroad_mask = np.logical_not(road_mask)
        save_ply(
            str(output_path / "input_points_nonroad.ply"),
            vertices=points[nonroad_mask],
            normals=normals[nonroad_mask],
            logger=logger,
        )
