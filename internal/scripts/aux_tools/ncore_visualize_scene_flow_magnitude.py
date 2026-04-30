# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import os
import time

from pathlib import Path
from typing import Tuple

import click
import matplotlib.cm
import numpy as np
import tqdm

from PIL import Image as PILImage

from ncore_internal.data.v3 import CameraSensor, ShardDataLoader
from nre.utils.ncore_utils import AuxShardDataLoader


def visualize_scene_flow_magnitude(
    aux_loader: AuxShardDataLoader,
    camera_id: str,
    frame_timestamps_us: int,
    mask_erode_radius: int = 0,  # radius of mask erosion (before median vote)
    instance_dist_threshold: float = 100,  # instances with distance to ego car larger than the threshold will be labeled as dynamic
) -> np.ndarray:
    dynamic_mag = aux_loader.get_scene_flow_magnitude(
        camera_id, frame_timestamps_us, mask_erode_radius, instance_dist_threshold
    )

    MAX_VELOCITY = 10.0
    # TODO ZG: I don't know how to properly type the colormaps un matplotlib. check if easily solvable
    dynamic_vis = matplotlib.cm.jet(dynamic_mag / MAX_VELOCITY) * 255  # type: ignore
    dynamic_vis = dynamic_vis * (np.expand_dims(dynamic_mag, axis=2) > 0)  # mask out background
    return dynamic_vis.astype(np.uint8)[:, :, :3]


@click.command()
@click.option(
    "--shard-file-pattern", type=str, help="Data shard pattern to load (supports range expansion)", required=True
)
@click.option("--output-dir", type=str, help="Path to the output folder", required=True)
@click.option(
    "--camera-id",
    "-c",
    "camera_ids",
    multiple=True,
    type=str,
    help="Cameras to be used (multiple value option, all if not specified)",
    default=None,
)
@click.option(
    "--start-frame", type=click.IntRange(min=0, max_open=True), help="Initial frame to be exported", default=None
)
@click.option(
    "--stop-frame", type=click.IntRange(min=0, max_open=True), help="Past-the-end frame to be exported", default=None
)
@click.option(
    "--step-frame",
    type=click.IntRange(min=1, max_open=True),
    help="Step used to downsample the number of frames",
    default=None,
)
@click.option(
    "--open-consolidated/--no-open-consolidated", is_flag=True, default=True, help="Open shards consolidated meta-data"
)
@click.option("--verbose", is_flag=True, default=False, help="Enable verbose logging outputs")
def ncore_visualize_scene_flow_magnitude(
    shard_file_pattern: str,
    output_dir: str,
    camera_ids: list[str],
    start_frame: int | None,
    stop_frame: int | None,
    step_frame: int | None,
    open_consolidated: bool,
    verbose: bool,
):
    # Initialize the logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    if verbose:
        logger.setLevel(logging.DEBUG)

    # Determine all data shards to process
    shard_input_paths = ShardDataLoader.evaluate_shard_file_pattern(shard_file_pattern)

    # Set up output path
    (output_path := Path(output_dir)).mkdir(parents=True, exist_ok=True)

    # Process each input shard individually
    for shard_input_path in shard_input_paths:
        loader = ShardDataLoader([shard_input_path], open_consolidated=open_consolidated)

        if not camera_ids:
            camera_ids = loader.get_camera_ids()

        # loader for the optical flow and segmentation
        aux_loader = AuxShardDataLoader.from_shard_data_loader(loader)

        for camera_id in camera_ids:
            # Prepare the output folder
            os.makedirs(output_path / "scene_flow" / camera_id, exist_ok=True)

            assert isinstance(camera_sensor := loader.get_sensor(camera_id), CameraSensor), (
                "only camera sensors supported"
            )

            offset = aux_loader.get_optical_flow_meta(camera_id)["frame_offset"]
            interval = aux_loader.get_optical_flow_meta(camera_id)["snapped_interval"]

            # Collect all image handles with timestamps
            timestamped_image_frame_handles: list[Tuple[int, CameraSensor.EncodedImageDataHandle]] = list(
                zip(
                    camera_sensor.get_frames_timestamps_us(),
                    [camera_sensor.get_frame_handle(i) for i in range(camera_sensor.get_frames_count())],
                )
            )

            timestamped_image_frame_handles = timestamped_image_frame_handles[start_frame:stop_frame:step_frame]

            n_images = len(timestamped_image_frame_handles)

            # create viz folder
            shard_name = os.path.basename(str(loader.get_shard_paths()[0])).replace(".zarr.itar", "")
            viz_folder = os.path.join(output_path, "scene_flow", camera_id, shard_name + "-scene_flow_visualize")
            os.makedirs(viz_folder, exist_ok=True)

            for run_idx, (frame_timestamp_us, _) in tqdm.tqdm(enumerate(timestamped_image_frame_handles)):
                img_source = timestamped_image_frame_handles[run_idx][1].get_data().get_decoded_image()

                time_start = time.perf_counter()
                dynamic_vis_0 = visualize_scene_flow_magnitude(
                    aux_loader, camera_id, frame_timestamp_us, mask_erode_radius=0
                )
                logger.debug(
                    "get_scene_flow_magnitude time with ( mask_erode_radius=0 ): %5.2f"
                    % (time.perf_counter() - time_start)
                )

                time_start = time.perf_counter()
                dynamic_vis_20 = visualize_scene_flow_magnitude(
                    aux_loader, camera_id, frame_timestamp_us, mask_erode_radius=20
                )
                logger.debug(
                    "get_scene_flow_magnitude time with ( mask_erode_radius=20 ): %5.2f"
                    % (time.perf_counter() - time_start)
                )

                scene_flow_vis = np.concatenate([dynamic_vis_20, dynamic_vis_0], axis=1)
                PILImage.fromarray(scene_flow_vis).resize((img_source.size[0] * 2 // 4, img_source.size[1] // 4)).save(
                    os.path.join(viz_folder, f"{run_idx:06d}.jpeg")
                )

                if (run_idx % 100 == 0 and run_idx > 0) or run_idx == n_images - 1:
                    assert interval > 0 and offset > 0
                    fps = round(offset / interval)
                    video_cmd = (
                        f"ffmpeg -y -r {fps}  -i  {viz_folder}/%6d.jpeg"
                        + f" -c:v libx264 -vf fps={fps} -pix_fmt yuv420p  {viz_folder}.mp4"
                    )
                    os.system(video_cmd)


if __name__ == "__main__":
    ncore_visualize_scene_flow_magnitude(show_default=True)
