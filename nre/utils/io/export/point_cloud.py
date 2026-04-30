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
from typing import Optional, Protocol

import click
import numpy as np
import point_cloud_utils as pcu
import torch

from nre.config.parse import parse_typed_config
from nre.datasets import make as make_dataset
from nre.datasets.ncore import NCOREDataSource
from nre.utils.misc import unpack_optional
from nre.utils.types import PointCloud, PointCloudColorType, RayFlags


def get_road_flags(point_cloud: PointCloud) -> np.ndarray:
    is_road = (
        torch.bitwise_and(unpack_optional(point_cloud.flags), RayFlags.ROAD_SEMANTIC.value).cpu().numpy().astype(bool)
    )
    return is_road


def save_points_ply(filename: str, positions: np.ndarray, colors: Optional[np.ndarray]) -> None:
    mesh = pcu.TriangleMesh()
    mesh.vertex_data.positions = positions
    if colors is not None:
        mesh.vertex_data.colors = colors
    print(f"Saving {positions.shape[0]} points to {filename}")
    mesh.save(filename)


class Colorizer(Protocol):
    def color_type(self) -> PointCloudColorType: ...

    def __call__(self, point_cloud: PointCloud) -> np.ndarray: ...


class RgbColorizer:
    def color_type(self) -> PointCloudColorType:
        return "camera-rgb"

    def __call__(self, point_cloud: PointCloud) -> np.ndarray:
        return unpack_optional(point_cloud.color).cpu().numpy()


class SemanticColorizer:
    def color_type(self) -> PointCloudColorType:
        return "semantics"

    def __call__(self, point_cloud: PointCloud) -> np.ndarray:
        return unpack_optional(point_cloud.color).cpu().numpy()


class RoadColorizer:
    def __init__(
        self,
        road_color: np.ndarray = np.array([100, 150, 50], dtype=np.uint8),
        background_color: np.ndarray = np.zeros((1, 3), dtype=np.uint8),
    ):
        self.road_color = road_color.reshape(1, 3).astype(np.uint8)
        self.background_color = background_color.reshape(1, 3).astype(np.uint8)

    def color_type(self) -> PointCloudColorType:
        return None

    def __call__(self, point_cloud: PointCloud) -> np.ndarray:
        is_road = get_road_flags(point_cloud)
        num_points = point_cloud.xyz_end.size(0)
        colors = np.repeat(self.background_color, num_points, axis=0)
        assert colors.shape == (num_points, 3)
        assert colors.dtype == np.uint8
        colors[is_road, :] = self.road_color
        return colors


def save_point_cloud(
    output_path: Path,
    point_cloud: PointCloud,
    datasource: NCOREDataSource,
    filename: str = "point_cloud",
    colorizer: Optional[Colorizer] = None,
    per_class: bool = False,
) -> None:
    positions = point_cloud.xyz_end.cpu().numpy()
    colors = colorizer(point_cloud) if colorizer else None

    if per_class:
        if isinstance(colorizer, (RgbColorizer, SemanticColorizer)):
            if point_cloud.semantic_class_id is None:
                raise ValueError("Export per semantic class not possible, point cloud has no semantic labels")

            semantic_class_ids = point_cloud.semantic_class_id.cpu().numpy().astype(int)
            semantic_classes_map = datasource.get_semantic_classes_map(camera_semantics=False, lidar_semantics=True)
            assert semantic_classes_map is not None
            for class_name, class_id in semantic_classes_map.items():
                is_point_in_class = semantic_class_ids == class_id
                num_points_in_class = np.sum(is_point_in_class)
                if num_points_in_class > 0:
                    save_points_ply(
                        str(output_path / filename) + "_" + "_".join(class_name.split()) + ".ply",
                        positions[is_point_in_class, :],
                        colors[is_point_in_class, :] if colors is not None else None,
                    )

        elif isinstance(colorizer, RoadColorizer):
            is_road = get_road_flags(point_cloud)
            is_nonroad = np.logical_not(is_road)

            save_points_ply(
                str(output_path / filename) + "_road.ply",
                positions[is_road, :],
                colors[is_road, :] if colors is not None else None,
            )

            save_points_ply(
                str(output_path / filename) + "_nonroad.ply",
                positions[is_nonroad, :],
                colors[is_nonroad, :] if colors is not None else None,
            )

    else:
        filename = str(output_path / filename) + ".ply"
        save_points_ply(filename, positions, colors)


@click.command("export-point-cloud")
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
    "--colorizer",
    "colorization",
    type=click.Choice(["none", "rgb", "semantic", "road"], case_sensitive=True),
    default="rgb",
    help=(
        "Specifies how to colorize (and filter in relation to colorization) the exported points. "
        "'none': points remain uncolored, "
        "'rgb': colorize by camera projection (only points visible in the selected camera(s) are exported), "
        "'semantic': colorize points based on their semantic class id by using the color map stored in the dataset, "
        "'road': colorize points based on the value of their binary road/nonroad flag."
    ),
)
@click.option(
    "--per-frame",
    is_flag=True,
    help="Export scans per Lidar frame instead of a single fused point cloud",
)
@click.option(
    "--per-class",
    is_flag=True,
    help="Split the point cloud per semantic class (according to --colorizer 'semantic' or 'road')",
)
@click.option(
    "--frame-step",
    type=int,
    default=50,
    help="Frame step size to use for exporting unfused (--per-frame) Lidar points (>1 means skipping).",
)
@click.option(
    "--valid-points-only",
    is_flag=True,
    help="Only use valid points for ground mesh reconstruction",
    default=False,  # BUG:5195915 WAR
)
@click.argument("hydra-args", nargs=-1)
def export_point_cloud(
    config_name: str,
    output_dir: str,
    colorization: str,
    per_frame: bool,
    per_class: bool,
    frame_step: int,
    valid_points_only: bool,
    hydra_args: list[str],
) -> None:
    """Various ways to colorize, group and export the Lidar point clouds of an NCORE dataset"""

    # Initialize logger
    logging.basicConfig(level=logging.INFO)
    config = parse_typed_config(config_name=config_name, hydra_args=hydra_args)

    colorizer: Optional[Colorizer]
    match colorization:
        case "none":
            filename = "point_cloud"
            colorizer = None
        case "rgb":
            filename = "colored_point_cloud"
            colorizer = RgbColorizer()
        case "semantic":
            filename = "semantic_point_cloud"
            colorizer = SemanticColorizer()
        case "road":
            filename = "segmented_point_cloud"
            colorizer = RoadColorizer()
        case _:
            raise ValueError(f"Unsupported colorizer '{colorization}'")

    datasource = make_dataset(config.dataset.name, config, split="train").get_datasource()

    if not isinstance(datasource, NCOREDataSource):
        raise TypeError(f"Unsupported dataset type: {type(datasource)}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    color_type: PointCloudColorType = colorizer.color_type() if colorizer is not None else None
    point_clouds = datasource.get_point_clouds(
        device=torch.device("cpu"),
        color_type=color_type,
        valid_points_only=valid_points_only,
    )

    if per_frame:
        for idx, point_cloud in enumerate(point_clouds):
            if idx % frame_step == 0:
                save_point_cloud(output_path, point_cloud, datasource, filename + f"_{idx:04d}", colorizer, per_class)
    else:
        print("Fusing point clouds")
        fused_point_cloud: PointCloud = PointCloud.collate_fn([pc for pc in point_clouds], device=torch.device("cpu"))
        save_point_cloud(output_path, fused_point_cloud, datasource, filename, colorizer, per_class)


if __name__ == "__main__":
    export_point_cloud(show_default=True)
