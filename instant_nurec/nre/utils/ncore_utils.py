# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import concurrent.futures
import io
import json
import logging

from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Tuple

import numpy as np
import PIL.Image as PILImage
import torch
import torchvision
import torchvision.transforms.functional
import zarr
import zarr.storage

from upath import UPath

import ncore
import ncore.data
import ncore.data.v4
import ncore.impl.data.stores as ncore_data_stores

from instant_nurec.nre.utils.types import HalfClosedInterval


# Common aux base-group names. Predict-only standalone uses just three;
# instance_segmentation, optical_flow, scene_flow, normal,
# lidar_semantic_segmentation, lidar_camera_visibility, semantic_logits,
# and dinov2 were defined but unreferenced (Phase 1 step 4.3).
SEMANTIC_SEG_BASE_GROUP = "semantic_segmentation"
DEPTH_BASE_GROUP = "depth"
EGO_MASK_BASE_GROUP = "egomask"


class AuxShardDataLoader:
    """Very simple (~dumb / linear) annotation data loader for NCore multi-shard-associated aux data"""

    def __init__(
        self,
        sequence_id: str,
        dataset_paths: list[Path] | list[UPath],
        open_consolidated=True,
    ) -> None:
        ## collect store candidate paths for a given sequence
        store_paths: set[UPath] = set()

        # inferred store paths from data shard paths
        for dataset_path in dataset_paths:
            # Make sure paths are absolute at this point
            dataset_path = UPath(dataset_path).absolute()

            # Map data shard file name to corresponding annotation file-name (this is fragile)
            dataset_base_name = dataset_path.stem.split(".")[0]

            # find matching stores paths
            for path in dataset_path.parent.iterdir():
                if path.is_file():
                    if not path.name.endswith(".zarr.itar"):
                        # not a supported file-based store format
                        continue

                    # check for matching base names
                    if (
                        # new-style aux data <session-shard>.aux.<signal>.zarr.itar
                        path.name.startswith(dataset_base_name + ".aux.")
                        or
                        # backwards-compatibility <session-shard>-annotations.zarr.itar
                        path.name.startswith(dataset_base_name + "-annotations")
                    ):
                        store_paths.add(path)
                elif path.is_dir():
                    if not path.name.endswith(".zarr"):
                        # not a supported directory store format
                        continue

                    if path.name.startswith(
                        dataset_base_name + ".aux."
                    ):  # new-style aux data <session-shard>.aux.<signal>.zarr
                        store_paths.add(path)

        ## load stores concurrently
        self.aux_shard_stores: list[zarr.storage.Store] = []
        self.base_groups: DefaultDict[str, list[zarr.Group]] = defaultdict(
            list
        )  # maps from base-group-name to *unordered* list of groups per shard

        with concurrent.futures.ThreadPoolExecutor() as executor:

            def thread_load_aux_store(store_path: UPath):
                """Thread-executed shard opening"""

                if store_path.is_file():
                    # load itar store
                    aux_shard_store = ncore_data_stores.IndexedTarStore(store_path, mode="r")
                else:
                    # load directory store
                    aux_shard_store = zarr.storage.DirectoryStore(store_path)

                aux_shard_root = (
                    ncore_data_stores.open_compressed_consolidated(store=aux_shard_store, mode="r")
                    if open_consolidated
                    else zarr.open(store=aux_shard_store, mode="r")
                )

                return aux_shard_root, aux_shard_store

            loaded_base_groups: set[Tuple[str, int]] = set()  # sanity check loaded data for consistency
            for future in concurrent.futures.as_completed(
                [executor.submit(thread_load_aux_store, store_path) for store_path in store_paths]
            ):
                # Note: thread completion order is not relevant here
                aux_shard_root, aux_shard_store = future.result()

                aux_sequence_id = aux_shard_root.attrs.get("sequence_id")
                aux_shard_id = aux_shard_root.attrs.get("shard_id")
                aux_shard_count = aux_shard_root.attrs.get("shard_count")
                aux_root_group_name = aux_shard_root.attrs.get(
                    "aux_root_group_name",
                    # backwards-compatibility <session-shard>-annotations.zarr.itar fallback
                    "annotations",
                )

                # setup consistency checks
                if not len(self.base_groups):
                    self._sequence_id: str = aux_sequence_id
                    self._shard_count: int = aux_shard_count

                if not self._sequence_id == aux_sequence_id:
                    raise ValueError("Can't load aux data for different sequences")
                if sequence_id != aux_sequence_id:
                    raise ValueError(
                        f"Loaded aux data for sequence {aux_sequence_id} not compatible with source sequence {sequence_id}"
                    )
                if not self._shard_count == aux_shard_count:
                    raise ValueError("Can't load aux data from different sequence subdivisions")

                # register loaded groups within this store per shard
                for base_group_name, base_group in aux_shard_root[aux_root_group_name].items():
                    # only register groups, not datasets
                    if not isinstance(base_group, zarr.Group):
                        continue

                    if (base_group_key := (base_group_name, aux_shard_id)) in loaded_base_groups:
                        raise ValueError(f"Group {base_group_name} loaded multiple times for shard ID {aux_shard_id}")
                    loaded_base_groups.add(base_group_key)

                    self.base_groups[base_group_name].append(base_group)

                self.aux_shard_stores.append(aux_shard_store)

    def _has_base_group(self, base_group_id: str, sensor_id: str | None = None) -> bool:
        """Check if base_group_id-typed aux group / signal exists. If sensor_id is provided, additionally check if the requested signal is available for the given sensor."""
        has_base_group = base_group_id in self.base_groups
        # check if base_group aux_data available at all
        if not has_base_group or sensor_id is None:
            return has_base_group
        # additionally check if the signal is available for the given sensor
        return any(sensor_id in base_group for base_group in self.base_groups[base_group_id])

    def has_semantic_segmentation(self, camera_id: str | None = None) -> bool:
        """Check if semantic segmentation data exists. If camera_id is provided, check if it is available for the given camera ID."""
        return self._has_base_group(SEMANTIC_SEG_BASE_GROUP, camera_id)

    def get_semantic_segmentation_meta(self, camera_id: str) -> dict:
        if not self.has_semantic_segmentation(camera_id):
            raise KeyError(f"No semantic segmentation data found for {camera_id}")

        # Take meta from first shard
        return dict(self.base_groups[SEMANTIC_SEG_BASE_GROUP][0][camera_id].attrs)

    def get_semantic_segmentation(self, camera_id: str, frame_timestamps_us: int) -> PILImage.Image:
        if SEMANTIC_SEG_BASE_GROUP not in self.base_groups:
            raise KeyError(f"no semantic segmentation data loaded")

        # find sample by linearly going through available shards samples
        # TODO(@janickm): this can be done much more efficiently and will be slow for a lot of shards
        for base_group in self.base_groups[SEMANTIC_SEG_BASE_GROUP]:
            try:
                ds = base_group[camera_id][str(frame_timestamps_us)]
            except KeyError:
                # it's ok if the key isn't in the current shard - continue look in next shard
                continue

            return PILImage.open(io.BytesIO(ds[()]), formats=[ds.attrs["format"]])

        raise KeyError(f"semantic segmentation not found for {camera_id} and timestamp {frame_timestamps_us}")

    def has_depth(self, camera_id: str | None = None) -> bool:
        """Check if depth data exists. If camera_id is provided, check if it is available for the given camera ID."""
        return self._has_base_group(DEPTH_BASE_GROUP, camera_id)

    def get_depth(
        self, camera_id: str, frame_timestamps_us: int, target_width_height: tuple[int, int] | None = None
    ) -> np.ndarray:
        if not self.has_depth():
            raise KeyError("no depth data loaded")

        for base_group in self.base_groups[DEPTH_BASE_GROUP]:
            try:
                depth_meta = self.get_depth_meta(camera_id)
                ds = base_group[camera_id][str(frame_timestamps_us)]
                store_as_png = depth_meta["store_depth_as_png"]

                if store_as_png:
                    max_depth_m = depth_meta["max_depth_m"]
                    image = PILImage.open(io.BytesIO(ds[()]), formats=["png"])
                    depth = np.array(image).astype(np.float32) * max_depth_m / 65535
                else:
                    depth = np.asarray(ds).astype(np.float32)

                if target_width_height:
                    depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)

                    depth = (
                        torchvision.transforms.functional.resize(
                            depth_tensor, [target_width_height[1], target_width_height[0]], antialias=True
                        )
                        .squeeze(0)
                        .squeeze(0)
                        .numpy()
                    )

                return depth

            except:
                continue

        raise KeyError(f"depth not found for {camera_id} and timestamp {frame_timestamps_us}")

    def has_egomask(self, camera_id: str | None = None) -> bool:
        """Check if ego-mask data exists. If camera_id is provided, check availability for that camera."""
        return self._has_base_group(EGO_MASK_BASE_GROUP, camera_id)

    def get_egomask(self, camera_id: str, frame_timestamps_us: int) -> np.ndarray:
        """Retrieve the ego-mask (binary numpy array) for a given camera using the first available timestamp."""
        # camera_id is the camera to retrieve the ego-mask for
        # aggregated mask is stored at special frame_timestamp '0'
        # if frame_timestamps_us is 0, the aggregated super ego-mask is returned
        # if frame_timestamps_us is not 0, the closest frame timestamp is returned

        if EGO_MASK_BASE_GROUP not in self.base_groups:
            raise KeyError("no ego-mask data loaded")

        # search through shards for the requested camera
        for base_group in self.base_groups[EGO_MASK_BASE_GROUP]:
            # skip if camera not present in this shard
            if camera_id not in base_group:
                continue

            if frame_timestamps_us == 0:
                # get the super ego-mask (frame_key = 0)
                frame_key = 0
            else:
                # get closest frame timestamp
                frame_keys = list(base_group[camera_id].keys())
                if not frame_keys:
                    raise KeyError(f"No ego-masks found for {camera_id}")
                frame_key = min(frame_keys, key=lambda x: abs(int(x) - frame_timestamps_us))

            try:
                ds = base_group[camera_id][frame_key]
            except KeyError:
                continue

            # decode PNG to binary mask
            image = PILImage.open(io.BytesIO(ds[()]), formats=[ds.attrs["format"]]).convert("L")
            return np.asarray(image) > 0

        # no mask found
        raise KeyError(f"No ego-mask found for {camera_id}")


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

    Predict-only standalone reads ncorev4 only; the V3 native sensor branch
    was dropped together with the V3 sequence loader (Phase 1 step 4.3).

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
    """Parse a NCore V4 single-sequence meta JSON; return ``(sequence_id, time_range_us, component_store_paths)``.

    Predict-only standalone consumes ncorev4 archives only; the NRE-side V3
    branch was dropped (Phase 1 step 4.3).
    """

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

