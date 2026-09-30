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

import concurrent.futures
import io
import json
import logging

from collections import defaultdict
from pathlib import Path
from typing import DefaultDict

import numpy as np
import PIL.Image as PILImage
import zarr
import zarr.storage

from upath import UPath

import ncore
import ncore.data
import ncore.data.v4
import ncore.impl.data.stores as ncore_data_stores

from instant_nurec.utils.types import HalfClosedInterval


SEMANTIC_SEG_BASE_GROUP = "semantic_segmentation"
DEPTH_BASE_GROUP = "depth"
EGO_MASK_BASE_GROUP = "egomask"


class AuxShardDataLoader:
    """Read NCore camera labels from adjacent sharded auxiliary stores.

    The naming convention is ``<data-shard>.aux.<signal>.zarr[.itar]``.
    A signal-specific override replaces matching adjacent stores rather than
    being loaded alongside them.
    Stores are optional: an empty loader simply reports every signal absent.
    """

    def __init__(
        self,
        sequence_id: str,
        dataset_paths: list[Path] | list[UPath],
        open_consolidated: bool = True,
        signal_override_paths: dict[str, UPath] | None = None,
    ) -> None:
        store_paths: set[UPath] = set()
        signal_override_paths = signal_override_paths or {}
        matched_override_keys: set[str] = set()
        for raw_dataset_path in dataset_paths:
            dataset_path = UPath(raw_dataset_path).absolute()
            dataset_base_name = dataset_path.stem.split(".")[0]
            for path in dataset_path.parent.iterdir():
                path_signal_name: str | None = None
                if path.is_file():
                    supported = path.name.endswith(".zarr.itar")
                    matches = path.name.startswith(dataset_base_name + ".aux.") or path.name.startswith(
                        dataset_base_name + "-annotations"
                    )
                    if supported and matches:
                        path_signal_name = path.name.split(".")[-3]
                        store_paths.add(path)
                elif path.is_dir() and path.name.endswith(".zarr") and path.name.startswith(
                    dataset_base_name + ".aux."
                ):
                    path_signal_name = path.name.split(".")[-2]
                    store_paths.add(path)

                if path_signal_name is not None and path_signal_name in signal_override_paths:
                    store_paths.discard(path)
                    store_paths.add(signal_override_paths[path_signal_name])
                    matched_override_keys.add(path_signal_name)

        for unmatched_key in sorted(set(signal_override_paths) - matched_override_keys):
            logging.warning(
                "signal_override_paths entry %r -> %s had no effect: no native aux store matches "
                "this signal name for sequence %r.",
                unmatched_key,
                signal_override_paths[unmatched_key],
                sequence_id,
            )

        self.aux_shard_stores: list[zarr.storage.Store] = []
        self.base_groups: DefaultDict[str, list[zarr.Group]] = defaultdict(list)

        def open_store(store_path: UPath):
            store = (
                ncore_data_stores.IndexedTarStore(store_path, mode="r")
                if store_path.is_file()
                else zarr.storage.DirectoryStore(store_path)
            )
            root = (
                ncore_data_stores.open_compressed_consolidated(store=store, mode="r")
                if open_consolidated
                else zarr.open(store=store, mode="r")
            )
            return root, store

        loaded_groups: set[tuple[str, int]] = set()
        loaded_sequence_id: str | None = None
        loaded_shard_count: int | None = None
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(open_store, path) for path in store_paths]
            for future in concurrent.futures.as_completed(futures):
                root, store = future.result()
                aux_sequence_id = root.attrs.get("sequence_id")
                shard_id = root.attrs.get("shard_id")
                shard_count = root.attrs.get("shard_count")
                root_group_name = root.attrs.get("aux_root_group_name", "annotations")
                if loaded_sequence_id is None:
                    loaded_sequence_id = aux_sequence_id
                    loaded_shard_count = shard_count
                if loaded_sequence_id != aux_sequence_id or sequence_id != aux_sequence_id:
                    raise ValueError(
                        f"Aux sequence {aux_sequence_id!r} does not match source sequence {sequence_id!r}"
                    )
                if loaded_shard_count != shard_count:
                    raise ValueError("Cannot combine auxiliary stores with different shard counts")
                for group_name, group in root[root_group_name].items():
                    if not isinstance(group, zarr.Group):
                        continue
                    key = (group_name, shard_id)
                    if key in loaded_groups:
                        raise ValueError(f"Aux group {group_name!r} is duplicated for shard {shard_id}")
                    loaded_groups.add(key)
                    self.base_groups[group_name].append(group)
                self.aux_shard_stores.append(store)

    def _has_base_group(self, group_name: str, sensor_id: str | None = None) -> bool:
        if group_name not in self.base_groups:
            return False
        return sensor_id is None or any(sensor_id in group for group in self.base_groups[group_name])

    def has_semantic_segmentation(self, camera_id: str | None = None) -> bool:
        return self._has_base_group(SEMANTIC_SEG_BASE_GROUP, camera_id)

    def get_semantic_segmentation_meta(self, camera_id: str) -> dict:
        if not self.has_semantic_segmentation(camera_id):
            raise KeyError(f"No semantic segmentation found for {camera_id}")
        return dict(self.base_groups[SEMANTIC_SEG_BASE_GROUP][0][camera_id].attrs)

    def get_semantic_segmentation(self, camera_id: str, timestamp_us: int) -> PILImage.Image:
        for group in self.base_groups[SEMANTIC_SEG_BASE_GROUP]:
            try:
                dataset = group[camera_id][str(timestamp_us)]
            except KeyError:
                continue
            return PILImage.open(io.BytesIO(dataset[()]), formats=[dataset.attrs["format"]])
        raise KeyError(f"No semantic segmentation found for {camera_id} at {timestamp_us}")

    def has_depth(self, camera_id: str | None = None) -> bool:
        return self._has_base_group(DEPTH_BASE_GROUP, camera_id)

    def get_depth_meta(self, camera_id: str) -> dict:
        if not self.has_depth(camera_id):
            raise KeyError(f"No depth found for {camera_id}")
        return dict(self.base_groups[DEPTH_BASE_GROUP][0][camera_id].attrs)

    def get_depth(self, camera_id: str, timestamp_us: int) -> np.ndarray:
        metadata = self.get_depth_meta(camera_id)
        for group in self.base_groups[DEPTH_BASE_GROUP]:
            try:
                dataset = group[camera_id][str(timestamp_us)]
            except KeyError:
                continue
            if metadata.get("store_depth_as_png", False):
                image = PILImage.open(io.BytesIO(dataset[()]), formats=["png"])
                return np.asarray(image, dtype=np.float32) * float(metadata["max_depth_m"]) / 65535.0
            return np.asarray(dataset, dtype=np.float32)
        raise KeyError(f"No depth found for {camera_id} at {timestamp_us}")

    def has_egomask(self, camera_id: str | None = None) -> bool:
        return self._has_base_group(EGO_MASK_BASE_GROUP, camera_id)

    def get_egomask(self, camera_id: str, timestamp_us: int = 0) -> np.ndarray:
        for group in self.base_groups[EGO_MASK_BASE_GROUP]:
            if camera_id not in group:
                continue
            frame_keys = list(group[camera_id].keys())
            if not frame_keys:
                continue
            frame_key = "0" if timestamp_us == 0 and "0" in frame_keys else min(
                frame_keys, key=lambda key: abs(int(key) - timestamp_us)
            )
            dataset = group[camera_id][frame_key]
            image = PILImage.open(io.BytesIO(dataset[()]), formats=[dataset.attrs["format"]]).convert("L")
            return np.asarray(image) > 0
        raise KeyError(f"No ego mask found for {camera_id}")


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
