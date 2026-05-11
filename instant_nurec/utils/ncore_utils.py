# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import logging

import numpy as np
import PIL.Image as PILImage

from upath import UPath

import ncore
import ncore.data
import ncore.data.v4

from instant_nurec.utils.types import HalfClosedInterval


def get_mask_image(
    mask_image: PILImage.Image | None, target_mask_size: tuple[int, int]
) -> np.ndarray | None:
    """
    Returns a boolean mask for, e.g., a camera sensor, scaled to the target resolution if required.

    The mask image is converted to grayscale and resized to match the camera sensor's resolution if their aspect ratios are sufficiently close.
    The resulting mask is returned as a NumPy boolean array, where `True` indicates masked-out regions.

    Args:
        mask_image (PILImage.Image | None): The mask image to be processed.
        target_mask_size (tuple[int, int]): The target size (width, height) to resize the mask image to.

    Returns:
        np.ndarray | None: A boolean NumPy array representing the mask, or None if no mask image is available.

    Raises:
        AssertionError: If the aspect ratio of the mask image does not match the camera sensor's resolution within a tolerance.
    """

    camera_mask: np.ndarray | None = None
    if mask_image is not None:
        # some external data-sources falsely provide masks as multi-channel
        # images -> force them to be gray-scale for our purposes
        mask_image = mask_image.convert("L")

        # Camera mask image might not have the same resolution as target camera.
        # Resize it to the target resolution if aspect ratios match
        if (camera_mask_size := mask_image.size) != target_mask_size:
            assert np.isclose(
                camera_mask_aspect := camera_mask_size[0] / camera_mask_size[1],
                target_mask_aspect := target_mask_size[0] / target_mask_size[1],
                atol=1e-2,
            ), (
                f"Camera mask aspect ratio {camera_mask_aspect:.4f} does not match camera "
                f"resolution aspect ratio {target_mask_aspect:.4f} - mask is not compatible with camera"
            )

            logging.info(
                f"Resizing camera mask {camera_mask_size} to target resolution {target_mask_size} [matching aspect ratios]"
            )
            mask_image = mask_image.resize(
                (target_mask_size[0], target_mask_size[1]),
                # bicubic is default for L / grayscale images - set it explicitly,
                # as this is sufficient for the subsequent binarization
                resample=PILImage.Resampling.BICUBIC,
            )

        # True for parts that we want to mask out
        camera_mask = np.asarray(mask_image) != 0

    return camera_mask


def get_camera_sensor_mask(
    camera_sensor: ncore.data.CameraSensorProtocol,
) -> np.ndarray | None:
    """
    Returns a boolean mask for a NCore V4 camera sensor, scaled to the sensor's resolution if required.

    The mask image is converted to grayscale and resized to match the camera sensor's resolution if their aspect ratios are sufficiently close.
    The resulting mask is returned as a NumPy boolean array, where `True` indicates masked-out regions.

    Predict-only reads ncorev4 only; the V3 native sensor branch
    was dropped together with the V3 sequence loader

    Returns:
        np.ndarray | None: A boolean NumPy array representing the mask, or None if no mask image is available.

    Raises:
        AssertionError: If the aspect ratio of the mask image does not match the camera sensor's resolution within a tolerance.
    """

    # V4 potentially provides more than a single mask, use 'ego' mask if available
    camera_mask_image: PILImage.Image | None = camera_sensor.get_mask_images().get("ego")
    resolution = camera_sensor.model_parameters.resolution

    return get_mask_image(camera_mask_image, tuple(resolution))


def parse_sequence_meta_file(sequence_meta_file: UPath) -> tuple[str, HalfClosedInterval, list[UPath]]:
    """Parse a NCore V4 single-sequence meta JSON; return ``(sequence_id, time_range_us, component_store_paths)``."""

    assert sequence_meta_file.is_file(), f"{__name__} provided path {sequence_meta_file} not a file"

    with sequence_meta_file.open("r") as fp:
        try:
            dataset_meta = json.load(fp)
        except ValueError as e:
            raise ValueError(f"{__name__} provided file {sequence_meta_file} not a json file") from e

    version = dataset_meta.get("version")
    assert version is not None and version.startswith("v4"), (
        f"{__name__} provided json file {sequence_meta_file} is not a NCore V4 single-sequence file (version={version!r})"
    )
    assert all(
        key in dataset_meta
        for key in ("sequence_id", "sequence_timestamp_interval_us", "version", "component_stores")
    ), f"{__name__} provided json file {sequence_meta_file} not a NCore V4 single-sequence file"

    time_range_us = HalfClosedInterval(
        dataset_meta["sequence_timestamp_interval_us"]["start"],
        dataset_meta["sequence_timestamp_interval_us"]["stop"],
    )
    dataset_paths = [
        sequence_meta_file.parent / component_store["path"] for component_store in dataset_meta["component_stores"]
    ]

    return dataset_meta["sequence_id"], time_range_us, dataset_paths


def create_sequence_loader(
    dataset_paths: list[UPath],
    open_consolidated: bool,
    v4_poses_component_group: str,
    v4_intrinsics_component_group: str,
    v4_masks_component_group: str,
    v4_cuboids_component_group: str,
) -> ncore.data.SequenceLoaderProtocol:
    """Create a NCore V4 sequence loader."""
    return ncore.data.v4.SequenceLoaderV4(
        ncore.data.v4.SequenceComponentGroupsReader(dataset_paths, open_consolidated=open_consolidated),
        poses_component_group_name=v4_poses_component_group,
        intrinsics_component_group_name=v4_intrinsics_component_group,
        masks_component_group_name=v4_masks_component_group,
        cuboids_component_group_name=v4_cuboids_component_group,
    )
