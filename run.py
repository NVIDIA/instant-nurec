# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import click
import click_default_group

from nre import __version__
from nre.benchmark.eval_ground_mesh import eval_ground_mesh
from nre.benchmark.eval_rendering_metrics import eval_rendering_metrics
from nre.benchmark.gt_from_ncore import export_benchmark_gt_from_ncore
from nre.grpc.serve import serve_grpc
from nre.metrics.compute_metrics import compute_metrics
from nre.render.novel_trajectory.render_novel_trajectory import render_novel_trajectory
from nre.run import main, profile_dataloader, run_script
from nre.utils.io.asset_harvester_tools.generate_asset_harvester_training_yaml import (
    generate_asset_harvester_training_yaml_cli,
)
from nre.utils.io.export.artifact_structure import export_artifact_structure
from nre.utils.io.export.custom_rig_trajectory import export_custom_rig_trajectory
from nre.utils.io.export.depth import export_depth
from nre.utils.io.export.ego_mask import export_ego_mask
from nre.utils.io.export.external_assets import export_external_assets
from nre.utils.io.export.gaussian_plys import export_gaussian_plys
from nre.utils.io.export.gaussian_statistics import gaussianStatistics
from nre.utils.io.export.gaussian_usd_asset import export_gaussian_usd_asset
from nre.utils.io.export.ground_mesh import export_ground_mesh
from nre.utils.io.export.mask_overlay import export_mask_overlay
from nre.utils.io.export.mesh import export_mesh
from nre.utils.io.export.ncore_diagnostic import export_ncore_diagnostic
from nre.utils.io.export.ncore_tracks import export_ncore_tracks
from nre.utils.io.export.parsed_config import export_parsed_config
from nre.utils.io.export.point_cloud import export_point_cloud
from nre.utils.io.export.render import render
from nre.utils.io.export.render_grpc import render_grpc
from nre.utils.io.export.rig_trajectories import export_rig_trajectories
from nre.utils.io.export.sequence_tracks import export_sequence_tracks
from nre.utils.io.export.usdz_artifact import export_usdz_artifact
from nre.utils.upgrade.cli.artifact import upgrade_artifact
from nre.utils.upgrade.cli.config import upgrade_config
from nre.viewer.ply_viewer import run_ply_viewer
from nre.viewer.usdz_viewer import run_viewer


@click.group(cls=click_default_group.DefaultGroup, default="main", default_if_no_args=True)
@click.version_option(version=str(__version__))
def cli():
    pass


cli.add_command(main)
cli.add_command(run_script)
cli.add_command(export_parsed_config)
cli.add_command(export_artifact_structure)
cli.add_command(upgrade_artifact)
cli.add_command(upgrade_config)
cli.add_command(export_depth)
cli.add_command(export_ego_mask)
cli.add_command(export_external_assets)
cli.add_command(export_ncore_tracks)
cli.add_command(export_mesh)
cli.add_command(export_gaussian_plys)
cli.add_command(export_gaussian_usd_asset)
cli.add_command(gaussianStatistics)
cli.add_command(export_ground_mesh)
cli.add_command(export_mask_overlay)
cli.add_command(export_point_cloud)
cli.add_command(export_rig_trajectories)
cli.add_command(export_sequence_tracks)
cli.add_command(export_usdz_artifact)
cli.add_command(export_ncore_diagnostic)
cli.add_command(export_custom_rig_trajectory)
cli.add_command(render)
cli.add_command(render_grpc)
cli.add_command(render_novel_trajectory)
cli.add_command(serve_grpc)
cli.add_command(profile_dataloader)
cli.add_command(run_viewer)
cli.add_command(run_ply_viewer)
cli.add_command(eval_ground_mesh)
cli.add_command(export_benchmark_gt_from_ncore)
cli.add_command(eval_rendering_metrics)
cli.add_command(compute_metrics)
cli.add_command(generate_asset_harvester_training_yaml_cli)

if __name__ == "__main__":
    cli(show_default=True, max_content_width=100)
