# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Optional

import click

from nre.utils.io.export.ncore_diagnostic import export_ncore_diagnostic_func


@click.command("export-ncore-benchmark-gt")
@click.option(
    "--shard-file-pattern",
    type=str,
    help=(
        "Path to the input .zarr.itar file(s). The extension can be omitted. "
        "Expands /path/file-[1-3] to [/path/file-1, /path/file-2, /path/file-3]"
    ),
    required=False,
    deprecated="Please use --dataset-path instead",
)
@click.option(
    "--dataset-path",
    type=str,
    help="Path to a NCore V3/V4 sequence meta-file",
    required=False,
)
@click.option("--poses-component-group", type=str, help="V4 component group for 'poses'", default="default")
@click.option("--intrinsics-component-group", type=str, help="V4 component group for 'intrinsics'", default="default")
@click.option("--masks-component-group", type=str, help="V4 component group for 'masks'", default="default")
@click.option("--cuboids-component-group", type=str, help="V4 component group for 'cuboids'", default="default")
@click.option(
    "--output-dir",
    type=str,
    help="Path to an (preferably new) output directory to export the requested data to",
    required=False,
)
@click.option(
    "--frame-step-camera",
    type=int,
    help="Frame step to use when exporting camera RGB frames / images (>1 means frame skipping)",
    default=50,
)
@click.option(
    "--frame-step-lidar",
    type=int,
    help="Frame step to use when exporting Lidar frames / spins (>1 means frame skipping)",
    default=50,
)
def export_benchmark_gt_from_ncore(
    shard_file_pattern: Optional[str],
    dataset_path: Optional[str],
    poses_component_group: str,
    intrinsics_component_group: str,
    masks_component_group: str,
    cuboids_component_group: str,
    output_dir: str,
    frame_step_camera: int,
    frame_step_lidar: int,
) -> None:
    """Exports ground-truth data needed for benchmarking novel-view prediction"""
    export_ncore_diagnostic_func(
        shard_file_pattern=shard_file_pattern,
        dataset_path=dataset_path,
        poses_component_group=poses_component_group,
        intrinsics_component_group=intrinsics_component_group,
        masks_component_group=masks_component_group,
        cuboids_component_group=cuboids_component_group,
        output_dir=output_dir,
        frame_step_camera=frame_step_camera,
        frame_step_lidar=frame_step_lidar,
        frame_naming="timestamp",
        video_fps=15,
        export_meta=False,
        export_camera_images=True,
        export_lidar_points=False,
        export_lidar_points_fused=False,
        export_semantic_labelmaps=False,
        export_semantic_overlays=False,
        export_ego_masks=True,
        export_ego_mask_overlays=False,
        invert_mask=False,
        format="image",
        export_all=False,
    )
