# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import glob
import json
import logging
import os

from pathlib import Path
from typing import Literal, Optional, Tuple

# NOTE(qi): let's explicitly import this to explicitly trigger any import issues
import click
import imageio
import imageio_ffmpeg  # type: ignore
import numpy as np
import point_cloud_utils as pcu
import polyscope as ps


DEFAULT_VIEW_JSON_STR = """
{
    "farClipRatio": 1000.0,
    "nearClipRatio": 0.005,
    "fov": 45.0,
    "projectionMode": "Perspective",
    "viewMat": [0.0, -1.0, 0.0, -0.0, -0.0, 0.0, 1.0, -0.0, -1.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    "windowHeight": 1080,
    "windowWidth": 1920
}"""


def load_points_ply(filename: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    mesh = pcu.TriangleMesh(filename, dtype=np.float32)
    return mesh.vertex_data.positions


logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(os.path.basename(__file__))

format_type = Literal["image", "video", "image+video"]


@click.command()
@click.option("--pc-dir", type=click.Path(exists=True))
@click.option("--output-dir", type=click.Path())
@click.option("--view-json-path", type=str, default=None)
@click.option(
    "--format",
    type=click.Choice(["image", "video", "image+video"]),
    help="Controls the type of export file produced",
    default="image+video",
)
@click.option("--radius", type=float, default=0.04, help="Radius of the points")
@click.option("--fps", type=int, default=10, help="Frames per second")
@click.option("--camera-pos", type=click.Tuple([float, float, float]), default=(0, 0, 0), help="Camera position")
@click.option("--camera-lookdir", type=click.Tuple([float, float, float]), default=(1, 0, 0), help="Camera lookdir")
@click.option("--camera-updir", type=click.Tuple([float, float, float]), default=(0, 0, 1), help="Camera updir")
@click.option("--show", is_flag=True, help="Show the point clouds in Polyscope, defaults to False")
def render_point_clouds_from_pc_dir(
    pc_dir: str,
    output_dir: str,
    view_json_path: str | None,
    format: format_type,
    fps: int,
    radius: float,
    camera_pos: Tuple[float, float, float],
    camera_lookdir: Tuple[float, float, float],
    camera_updir: Tuple[float, float, float],
    show: bool,
):
    """Takes a directory including stored point clouds and renders a video using Polyscope."""
    ps.set_allow_headless_backends(not show)
    ps.init()
    ps.set_front_dir("neg_x_front")
    ps.set_up_dir("z_up")
    ps.set_automatically_compute_scene_extents(False)
    ps.set_length_scale(1.0)
    ps.set_ground_plane_mode("none")

    if view_json_path is not None:
        with open(view_json_path, "r") as f:
            view_json = f.read()
        ps.set_view_from_json(view_json)
    else:
        view_json = DEFAULT_VIEW_JSON_STR
        ps.set_view_from_json(view_json)
        intrinsics = ps.get_view_camera_parameters().get_intrinsics()
        camera_params = ps.CameraParameters(
            extrinsics=ps.CameraExtrinsics(root=camera_pos, look_dir=camera_lookdir, up_dir=camera_updir),
            intrinsics=intrinsics,
        )
        ps.set_view_camera_parameters(camera_params)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir / Path("pred"), exist_ok=True)
    os.makedirs(output_dir / Path("gt"), exist_ok=True)

    pred_images = []
    gt_images = []
    for filename in sorted(glob.glob(f"{pc_dir}/*_gt.ply")):
        id = int(Path(filename).stem[:6])
        logger.info(f"Processing pc {id}")

        gt = load_points_ply(f"{pc_dir}/{id:06d}output_gt.ply")
        pred = load_points_ply(f"{pc_dir}/{id:06d}output.ply")

        ps.register_point_cloud(
            f"pred_{id}", pred, enabled=True, color=[0, 0, 1], point_render_mode="sphere", radius=radius
        )
        pred_images.append(image := ps.screenshot_to_buffer())
        imageio.imwrite(output_dir / Path(f"pred/{id:06d}.png"), image)

        if show:
            ps.show()
        ps.remove_all_structures()

        ps.register_point_cloud(
            f"gt_{id}", gt, enabled=True, color=[1, 0, 0], point_render_mode="sphere", radius=radius
        )
        gt_images.append(image := ps.screenshot_to_buffer())
        imageio.imwrite(output_dir / Path(f"gt/{id:06d}.png"), image)

        if show:
            ps.show()
        ps.remove_all_structures()

    if "video" in format:
        imageio.mimsave(output_dir / Path("pred.mp4"), pred_images, fps=fps)
        imageio.mimsave(output_dir / Path("gt.mp4"), gt_images, fps=fps)


if __name__ == "__main__":
    render_point_clouds_from_pc_dir()
