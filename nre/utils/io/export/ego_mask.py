# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import logging
import os

import click
import numpy as np

from PIL import Image

from ncore_internal.data.v3 import ShardDataLoader


logger = logging.getLogger(__name__)


# TODO: remove this script in favor of nre/benchmark/gt_from_ncore.py or nre/utils/io/export/ncore_diagnostic.py
@click.command("export-ego-mask")
@click.option(
    "--shard-file-pattern",
    "shard_file_pattern",
    type=str,
    required=True,
    help="Path to ncore zarr file",
)
@click.option(
    "--output-dir",
    "output_dir",
    type=str,
    required=True,
    help="Path to the output images directory. It should be the same directory of the checkpoint for grpc usage",
)
@click.option(
    "--camera-ids", "camera_ids", type=str, multiple=True, help="List of camera ids. [default: all available cameras]"
)
@click.option(
    "--invert-mask/--no-invert-mask",
    default=False,
    help="Whether to show the masked region and blank the unmasked region (--no-invert-mask) or show the unmasked region and blank the masked region (--invert-mask)",
)
@click.option(
    "--camera-frame-idx",
    "camera_frame_idx",
    type=int,
    default=50,
    help="Index of the camera frame to export the ego mask for",
)
def export_ego_mask(
    shard_file_pattern: str, output_dir: str, camera_ids: list[str], invert_mask: bool, camera_frame_idx: int
) -> None:
    shards = ShardDataLoader.evaluate_shard_file_pattern(shard_file_pattern)
    loader = ShardDataLoader(shards)

    output_dir = os.path.join(output_dir, "ego-hoods")
    os.makedirs(output_dir, exist_ok=True)
    resized_h = 540
    resized_w = 960

    # Save all ego-hoods if no camera_ids are provided
    if not camera_ids:
        camera_ids = loader.get_camera_ids()

    for camera_id in camera_ids:
        camera_sensor = loader.get_camera_sensor(camera_id)
        camera_mask_image = camera_sensor.get_camera_mask_image()

        if camera_mask_image is None:
            logger.warning(f"Camera {camera_id} does not have a mask image")
            continue

        camera_mask_array = np.asarray(camera_mask_image.convert("L").resize((resized_w, resized_h))) != 0
        if not invert_mask:
            camera_mask_array = ~camera_mask_array

        alpha_array = np.expand_dims(np.ones_like(camera_mask_array), axis=-1).astype(np.uint8) * 255
        assert camera_frame_idx < camera_sensor.get_frames_count(), (
            f"Camera {camera_id} only has {camera_sensor.get_frames_count()} frames"
        )
        camera_image_array = np.asarray(
            camera_sensor.get_frame_image(camera_frame_idx).resize((resized_w, resized_h))
        ).copy()
        camera_image_array[camera_mask_array] = 0
        alpha_array[camera_mask_array] = 0
        camera_image_array = np.concatenate((camera_image_array, alpha_array), axis=-1)
        ego_mask = Image.fromarray(camera_image_array, mode="RGBA")
        ego_mask.save(os.path.join(output_dir, f"{camera_id}.png"))
        logging.info(f"Wrote: {os.path.abspath(os.path.join(output_dir, f'{camera_id}.png'))}")
