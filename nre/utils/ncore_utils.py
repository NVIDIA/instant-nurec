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
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import DefaultDict, Optional, Protocol, Tuple, cast

import numpy as np
import PIL.Image as PILImage
import torch
import torchvision
import torchvision.transforms.functional
import zarr
import zarr.storage

from numcodecs import Blosc
from upath import UPath

import ncore
import ncore.data
import ncore.data.v4
import ncore.impl.data.stores as ncore_data_stores
import ncore_internal.data
import ncore_internal.data.v3

from nre.utils.morph import MorphOp
from nre.utils.types import HalfClosedInterval


# Common aux base-group names used by both the aux data writer and loader
SEMANTIC_SEG_BASE_GROUP = "semantic_segmentation"
INSTANCE_SEG_BASE_GROUP = "instance_segmentation"
OPTICAL_FLOW_BASE_GROUP = "optical_flow"
SCENE_FLOW_BASE_GROUP = "scene_flow"
NORMAL_BASE_GROUP = "normal"
DEPTH_BASE_GROUP = "depth"
LIDAR_SEMANTIC_SEG_BASE_GROUP = "lidar_semantic_segmentation"
LIDAR_CAMERA_VISIBILITY_BASE_GROUP = "lidar_camera_visibility"
SEG_LOGIT_BASE_GROUP = "semantic_logits"
DINOV2_BASE_GROUP = "dinov2"
EGO_MASK_BASE_GROUP = "egomask"


class AuxShardDataLoader:
    """Very simple (~dumb / linear) annotation data loader for NCore multi-shard-associated aux data"""

    @staticmethod
    def from_shard_data_loader(
        loader: ncore_internal.data.v3.ShardDataLoader, open_consolidated=True
    ) -> AuxShardDataLoader:
        """Factory method for V3 NCore sequence loaders"""
        return AuxShardDataLoader(
            sequence_id=loader.get_sequence_id(),
            dataset_paths=loader.get_shard_paths(),
            open_consolidated=open_consolidated,
        )

    @staticmethod
    def from_sequence_loader(
        sequence_loader: ncore.data.SequenceLoaderProtocol, open_consolidated=True
    ) -> AuxShardDataLoader:
        """Factory method for V4 NCore compat sequence loaders"""
        return AuxShardDataLoader(
            sequence_id=sequence_loader.sequence_id,
            dataset_paths=sequence_loader.sequence_paths,
            open_consolidated=open_consolidated,
        )

    def __init__(
        self,
        sequence_id: str,
        dataset_paths: list[Path] | list[UPath],
        open_consolidated=True,
        signal_override_paths: dict[str, UPath] | None = None,
    ) -> None:
        ## collect store candidate paths for a given sequence
        store_paths: set[UPath] = set()
        signal_override_paths = signal_override_paths or {}
        matched_override_keys: set[str] = set()

        # inferred store paths from data shard paths
        for dataset_path in dataset_paths:
            # Make sure paths are absolute at this point
            dataset_path = UPath(dataset_path).absolute()

            # Map data shard file name to corresponding annotation file-name (this is fragile)
            dataset_base_name = dataset_path.stem.split(".")[0]

            # find matching stores paths
            for path in dataset_path.parent.iterdir():
                path_signal_name: str | None = None
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
                        path_signal_name = path.name.split(".")[-3] if path.name.endswith(".zarr.itar") else None
                        store_paths.add(path)
                elif path.is_dir():
                    if not path.name.endswith(".zarr"):
                        # not a supported directory store format
                        continue

                    if path.name.startswith(
                        dataset_base_name + ".aux."
                    ):  # new-style aux data <session-shard>.aux.<signal>.zarr
                        path_signal_name = path.name.split(".")[-2]
                        store_paths.add(path)

                # Replace store path with the overridden path if it exists
                if path_signal_name is not None and path_signal_name in signal_override_paths:
                    if path in store_paths:
                        store_paths.remove(path)
                    store_paths.add(signal_override_paths[path_signal_name])
                    matched_override_keys.add(path_signal_name)

        # Warn about any signal_override_paths entries that did not match a native aux store.
        # These overrides are silently dropped (they have no effect), so surface them to the user
        # without promoting the silent drop into a hard failure.
        for unmatched_key in sorted(set(signal_override_paths.keys()) - matched_override_keys):
            logging.warning(
                "signal_override_paths entry %r -> %s had no effect: no native aux store matches this signal name for sequence %r.",
                unmatched_key,
                signal_override_paths[unmatched_key],
                sequence_id,
            )

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

    def reload_store_resources(self) -> None:
        """Trigger a reload of the resources of each shard store - useful to, e.g., re-open file objects in multi-process settings"""
        for aux_shard_store in self.aux_shard_stores:
            # only need to reload itar-based stores
            if isinstance(aux_shard_store, ncore_data_stores.IndexedTarStore):
                aux_shard_store.reload_resources()

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

    def has_instance_segmentation(self, camera_id: str | None = None) -> bool:
        """Check if instance segmentation data exists. If camera_id is provided, check if it is available for the given camera ID."""
        return self._has_base_group(INSTANCE_SEG_BASE_GROUP, camera_id)

    def get_instance_segmentation_meta(self, camera_id: str) -> dict:
        if not self.has_instance_segmentation(camera_id):
            raise KeyError(f"No instance segmentation data found for {camera_id}")

        # Take meta from first shard
        return dict(self.base_groups[INSTANCE_SEG_BASE_GROUP][0][camera_id].attrs)

    def get_instance_segmentation(self, camera_id: str, frame_timestamps_us: int) -> dict:
        if INSTANCE_SEG_BASE_GROUP not in self.base_groups:
            raise KeyError(f"no instance segmentation data loaded")

        for base_group in self.base_groups[INSTANCE_SEG_BASE_GROUP]:
            try:
                ds = base_group[camera_id][str(frame_timestamps_us)]
                w, h = base_group[camera_id].attrs["resolution"]
            except KeyError:
                continue

            instance_masks = np.unpackbits(ds["instance_masks"]).reshape(-1, h, w)

            return {
                "instance_masks": instance_masks,
                "scores": np.array(ds["scores"]),
                "classes": np.array(ds["classes"][...]),
            }

        raise KeyError(f"instance segmentation not found for {camera_id} and timestamp {frame_timestamps_us}")

    def has_semantic_logits(self, camera_id: str | None = None) -> bool:
        """Check if semantic logits data exists. If camera_id is provided, check if it is available for the given camera ID."""
        return self._has_base_group(SEG_LOGIT_BASE_GROUP, camera_id)

    def get_semantic_logits_meta(self, camera_id: str) -> dict:
        if not self.has_semantic_logits(camera_id):
            raise KeyError(f"No semantic logits data found for {camera_id}")

        # Take meta from first shard
        return dict(self.base_groups[SEG_LOGIT_BASE_GROUP][0][camera_id].attrs)

    def get_semantic_logits(self, camera_id: str, frame_timestamps_us: int) -> np.ndarray:
        if SEG_LOGIT_BASE_GROUP not in self.base_groups:
            raise KeyError(f"no semantic segmentation logit data loaded")

        for base_group in self.base_groups[SEG_LOGIT_BASE_GROUP]:
            try:
                ds = base_group[camera_id][str(frame_timestamps_us)]
            except KeyError:
                continue
            return np.array(ds)
        raise KeyError(f"semantic logits not found for {camera_id} and timestamp {frame_timestamps_us}")

    def has_dinov2(self, camera_id: str | None = None) -> bool:
        """Check if dinov2 data exists. If camera_id is provided, check if it is available for the given camera ID."""
        return self._has_base_group(DINOV2_BASE_GROUP, camera_id)

    def get_dinov2_meta(self, camera_id: str) -> tuple[dict, dict, dict]:
        if not self.has_dinov2(camera_id):
            raise KeyError(f"No dinov2 data found for {camera_id}")

        base_group = self.base_groups[DINOV2_BASE_GROUP][0][camera_id]
        extractor_meta_dict = dict(base_group.attrs)

        color_transform_dict = {}
        for key in (color_transform_group := base_group["color_transform"]):
            if isinstance(color_transform_group[key], zarr.Array):
                color_transform_dict[key] = np.array(color_transform_group[key])
            else:
                color_transform_dict[key] = color_transform_group.attrs[key]

        pca_dict = {}
        if "pca" in base_group:
            for key in (pca_group := base_group["pca"]):
                if isinstance(pca_group[key], zarr.Array):
                    pca_dict[key] = np.array(pca_group[key])
                else:
                    pca_dict[key] = pca_group.attrs[key]

        return extractor_meta_dict, color_transform_dict, pca_dict

    def get_dinov2(self, camera_id: str, frame_timestamps_us: int) -> tuple[np.ndarray, np.ndarray | None]:
        if DINOV2_BASE_GROUP not in self.base_groups:
            raise KeyError(f"no dinov2 data loaded")

        for base_group in self.base_groups[DINOV2_BASE_GROUP]:
            try:
                ds = base_group[camera_id][str(frame_timestamps_us)]
            except KeyError:
                continue

            dinov2_features = np.array(ds["features"])
            dinov2_valid = None
            if "valid" in ds:
                dinov2_valid_bits = np.array(ds["valid"])
                h, w, _ = dinov2_features.shape
                dinov2_valid = np.unpackbits(dinov2_valid_bits)[: h * w].reshape(h, w).astype(bool)

            return dinov2_features, dinov2_valid

        raise KeyError(f"dinov2 not found for {camera_id} and timestamp {frame_timestamps_us}")

    def has_optical_flow(self, camera_id: str | None = None) -> bool:
        """Check if optical flow data exists. If camera_id is provided, check if it is available for the given camera ID."""
        return self._has_base_group(OPTICAL_FLOW_BASE_GROUP, camera_id)

    def get_optical_flow_meta(self, camera_id: str) -> dict:
        if not self.has_optical_flow(camera_id):
            raise KeyError(f"No optical flow data found for {camera_id}")

        # Take meta from first shard
        return dict(self.base_groups[OPTICAL_FLOW_BASE_GROUP][0][camera_id].attrs)

    def get_optical_flow(self, camera_id: str, frame_timestamps_us: int) -> np.ndarray:
        if not self.has_optical_flow():
            raise KeyError(f"no optical flow data loaded")

        for base_group in self.base_groups[OPTICAL_FLOW_BASE_GROUP]:
            try:
                ds = base_group[camera_id][str(frame_timestamps_us)]
                w, h = base_group[camera_id].attrs["resolution"]

                if base_group[camera_id].attrs["store_as_png"]:
                    shift_to_positive = base_group[camera_id].attrs["shift_to_positive"]

                    image_data = np.asarray(
                        PILImage.open(io.BytesIO(ds[()]), formats=[ds.attrs["format"]]).convert("RGBA")
                    ).astype(np.float32)

                    flow_x = image_data[:, :, 0:1] * 256.0 + image_data[:, :, 1:2] - shift_to_positive
                    flow_y = image_data[:, :, 2:3] * 256.0 + image_data[:, :, 3:4] - shift_to_positive

                    flow = np.concatenate([flow_x, flow_y], axis=2)
                else:
                    flow = np.asarray(ds)

                backward_flow = torch.nn.functional.interpolate(
                    torch.from_numpy(flow.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).cuda(),
                    size=(h, w),
                    mode="bilinear",
                )
                backward_flow = backward_flow[0].permute(1, 2, 0).cpu().numpy()

            except KeyError:
                continue

            return backward_flow  # the optical flow is from frame t to frame t-offset

        raise KeyError(f"optical flow not found for {camera_id} and timestamp {frame_timestamps_us}")

    def has_scene_flow(self, camera_id: str | None = None) -> bool:
        """Check if scene flow data exists. If camera_id is provided, check if it is available for the given camera ID."""
        return self._has_base_group(SCENE_FLOW_BASE_GROUP, camera_id)

    def get_scene_flow_meta(self, camera_id: str) -> dict:
        if not self.has_scene_flow(camera_id):
            raise KeyError(f"No scene flow data found for {camera_id}")

        # Take meta from first shard
        return dict(self.base_groups[SCENE_FLOW_BASE_GROUP][0][camera_id].attrs)

    def get_scene_flow(self, camera_id: str, frame_timestamps_us: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self.has_scene_flow():
            raise KeyError(f"no scene flow data loaded")

        for base_group in self.base_groups[SCENE_FLOW_BASE_GROUP]:
            try:
                meta_data = self.get_scene_flow_meta(camera_id)
                scale_to_int = meta_data[
                    "scale_to_int"
                ]  # "scale" value timed to the float flow before converting to int16
                w, h = meta_data["resolution"]

                data = base_group[camera_id][str(frame_timestamps_us)]

                if meta_data["store_as_png"]:
                    shift_to_positive = meta_data["shift_to_positive"]

                    image_data = np.asarray(
                        PILImage.open(io.BytesIO(data[()]), formats=[data.attrs["format"]]).convert("RGBA")
                    ).astype(np.float32)
                    image_h = image_data.shape[0]

                    image_data_1 = image_data[: image_h // 2, :, :]
                    image_data_2 = image_data[image_h // 2 :, :, :]

                    flow_x = image_data_1[:, :, 0:1] * 256.0 + image_data_1[:, :, 1:2] - shift_to_positive
                    flow_y = image_data_1[:, :, 2:3] * 256.0 + image_data_1[:, :, 3:4] - shift_to_positive
                    flow_z = image_data_2[:, :, 0:1] * 256.0 + image_data_2[:, :, 1:2] - shift_to_positive
                    lidar_d = image_data_2[:, :, 2:3] * 256.0 + image_data_2[:, :, 3:4] - shift_to_positive

                    scene_flow_data = (
                        np.concatenate([flow_x, flow_y, flow_z, lidar_d], axis=2).astype(np.float32) / scale_to_int
                    )

                    # upsample to original size
                    scene_flow_data = (
                        torch.nn.functional.interpolate(
                            torch.from_numpy(scene_flow_data).permute(2, 0, 1).unsqueeze(0).cuda(),
                            size=(h, w),
                            mode="bilinear",
                        )[0]
                        .permute(1, 2, 0)
                        .cpu()
                        .numpy()
                    )

                else:
                    scene_flow_data = np.zeros([h, w, 4], dtype=np.float32)
                    scene_flow_data[data[:, 0], data[:, 1], :] = np.array(data[:, 2:]).astype(np.float32) / scale_to_int

            except KeyError:
                continue

            scene_flow = scene_flow_data[:, :, :3]  # the scene flow is forward, from frame t-offset to frame t
            lidar_dist = scene_flow_data[:, :, 3:]

            return scene_flow, lidar_dist

        raise KeyError(f"scene flow not found for {camera_id} and timestamp {frame_timestamps_us}")

    def get_scene_flow_magnitude(
        self,
        camera_id: str,
        frame_timestamps_us: int,
        mask_erode_radius: int = 0,  # radius of mask erosion (before median vote)
        instance_dist_threshold: float = 100,  # instances with distance to ego car larger than the threshold will be labeled as dynamic
    ) -> np.ndarray:
        instances_meta = self.get_instance_segmentation_meta(camera_id)
        instances = self.get_instance_segmentation(camera_id, frame_timestamps_us)
        scene_flow_data = self.get_scene_flow(camera_id, frame_timestamps_us)
        scene_flow = torch.from_numpy(scene_flow_data[0]).cuda()
        lidar_dist = torch.from_numpy(scene_flow_data[1]).cuda()

        w: int = instances_meta["resolution"][0]
        h: int = instances_meta["resolution"][1]

        dynamic_mag = np.zeros([h, w], np.float32)

        # erode the mask of all instances together
        mask_all = torch.sum(torch.from_numpy(instances["instance_masks"]).cuda(), dim=0) > 0.5

        if mask_erode_radius > 0:
            downsample_scale = 2  # downsample the mask before erode for memory saving

            eroder = MorphOp(
                c_out=1,
                type_str="erosion2d",
                device=torch.device("cuda"),
                kernel_size=mask_erode_radius // downsample_scale + 1,
                use_soft_max=False,
            )

            tensor_to_erode = mask_all.unsqueeze(0).unsqueeze(0).float()
            tensor_to_erode_sq = torch.cat([tensor_to_erode, torch.zeros_like(tensor_to_erode)], dim=2)[
                :, :, :w, :w
            ]  # pad it to a square image
            resizer1 = torchvision.transforms.Resize(
                size=(w // downsample_scale, w // downsample_scale),
                interpolation=torchvision.transforms.InterpolationMode.NEAREST,
            )  # for speeding up
            resizer2 = torchvision.transforms.Resize(
                size=(w, w), interpolation=torchvision.transforms.InterpolationMode.NEAREST
            )
            tensor_eroded = resizer2(eroder(resizer1(tensor_to_erode_sq)))[
                :, :, :h, :w
            ]  # eroding and remove the padded zeros

            mask_all_eroded = tensor_eroded.squeeze() > 0.5
        else:
            mask_all_eroded = mask_all

        # get instance masks
        for id in range(instances["instance_masks"].shape[0]):
            MAX_VELOCITY = 10.0

            instance_mask = torch.from_numpy(instances["instance_masks"][id]).cuda() * mask_all_eroded
            instance_pixels = instance_mask.nonzero().cpu().numpy()

            # undo the erosion if number of instance_pixels is smaller than 100
            if instance_pixels.shape[0] < 100:
                instance_mask = torch.from_numpy(instances["instance_masks"][id]).cuda()
                instance_pixels = instance_mask.nonzero().cpu().numpy()

            class_id = instances["classes"][id]
            is_vehicle = instances_meta["thing_classes"][class_id] in ["car", "truck", "bus", "train"]

            if instance_pixels.shape[0] > 1 and is_vehicle:
                distance_on_mask = lidar_dist[
                    instance_pixels[:, 0], instance_pixels[:, 1], :
                ]  # distance of each pixel to ego car
                if (
                    torch.median(distance_on_mask).item() < instance_dist_threshold
                ):  # if the median distance is smaller than threshold, we compute the dynamic using scene flow
                    scene_flow_on_mask = scene_flow[instance_pixels[:, 0], instance_pixels[:, 1], :]
                    magn = torch.median(torch.norm(scene_flow_on_mask, dim=1)).item()
                else:
                    magn = MAX_VELOCITY  # if the median distance is larger than threshold, the instance is treated as dynamic
            else:
                magn = MAX_VELOCITY  # instances that are not vehicles are always dynamic

            dynamic_mag[instances["instance_masks"][id].astype(bool)] = magn

        return dynamic_mag

    def has_depth(self, camera_id: str | None = None) -> bool:
        """Check if depth data exists. If camera_id is provided, check if it is available for the given camera ID."""
        return self._has_base_group(DEPTH_BASE_GROUP, camera_id)

    def get_depth_meta(self, camera_id: str) -> dict:
        if not self.has_depth(camera_id):
            raise KeyError("No depth data found for {camera_id}")

        # Take meta from first shard
        return dict(self.base_groups[DEPTH_BASE_GROUP][0][camera_id].attrs)

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

    def has_normal(self, camera_id: str | None = None) -> bool:
        """Check if normals exist. If camera_id is provided, check if they are available for the given camera ID."""
        return self._has_base_group(NORMAL_BASE_GROUP, camera_id)

    def get_normal_meta(self, camera_id: str) -> dict:
        if not self.has_normal(camera_id):
            raise KeyError("No normal labels found for {camera_id}")

        # Take meta from first shard
        return dict(self.base_groups[NORMAL_BASE_GROUP][0][camera_id].attrs)

    def get_normal(
        self, camera_id: str, frame_timestamps_us: int, target_width_height: tuple[int, int] | None = None
    ) -> np.ndarray:
        if not self.has_normal():
            raise KeyError("no normal data loaded")

        for base_group in self.base_groups[NORMAL_BASE_GROUP]:
            try:
                ds = base_group[camera_id][str(frame_timestamps_us)]
                image: PILImage.Image = PILImage.open(io.BytesIO(ds[()]), formats=["png"])
                if target_width_height:
                    image = image.resize(target_width_height, PILImage.Resampling.LANCZOS)
                image_data = np.asarray(image).astype(np.float32)
                image_data = image_data / 127.5 - 1

                return image_data

            except:
                continue

        raise KeyError(f"normal labels not found for {camera_id} and timestamp {frame_timestamps_us}")

    def has_lidar_semantic_segmentation(self, lidar_id: str | None = None) -> bool:
        """Check if lidar semantic segmentation data exists. If camera_id is provided, check if it is available for the given camera ID."""
        return self._has_base_group(LIDAR_SEMANTIC_SEG_BASE_GROUP, lidar_id)

    def get_lidar_semantic_segmentation_meta(self, lidar_id: str) -> dict:
        if not self.has_lidar_semantic_segmentation(lidar_id):
            raise KeyError(f"No lidar semantic segmentation data found for {lidar_id}")

        # Take meta from first shard
        return dict(self.base_groups[LIDAR_SEMANTIC_SEG_BASE_GROUP][0][lidar_id].attrs)

    def get_lidar_semantic_segmentation(self, lidar_id: str, frame_timestamps_us: int) -> np.ndarray:
        if LIDAR_SEMANTIC_SEG_BASE_GROUP not in self.base_groups:
            raise KeyError(f"no lidar semantic segmentation data loaded")

        # find sample by linearly going through available shards samples
        for base_group in self.base_groups[LIDAR_SEMANTIC_SEG_BASE_GROUP]:
            try:
                ds = base_group[lidar_id][str(frame_timestamps_us)]
            except KeyError:
                # it's ok if the key isn't in the current shard - continue look in next shard
                continue

            img = PILImage.open(io.BytesIO(ds[()]), formats=[ds.attrs["format"]])
            return np.array(img).reshape(-1)

        raise KeyError(f"semantic segmentation not found for {lidar_id} and timestamp {frame_timestamps_us}")

    def has_lidar_camera_visibility(self, lidar_id: str | None = None) -> bool:
        """Check if lidar camera visibility data exists. If lidar_id is provided, check if it is available for the given lidar_id ID."""
        return self._has_base_group(LIDAR_CAMERA_VISIBILITY_BASE_GROUP, lidar_id)

    def get_lidar_camera_visibility_meta(self, lidar_id: str) -> dict:
        if not self.has_lidar_camera_visibility(lidar_id):
            raise KeyError(f"No lidar camera visibility data found for sensor {lidar_id}")

        # Take meta from first shard
        return dict(self.base_groups[LIDAR_CAMERA_VISIBILITY_BASE_GROUP][0][lidar_id].attrs)

    def get_lidar_camera_visibility(self, lidar_id: str, frame_timestamps_us: int, camera_ids: list[str] = []) -> dict:
        if LIDAR_CAMERA_VISIBILITY_BASE_GROUP not in self.base_groups:
            raise KeyError(f"no lidar camera visibility data loaded")

        # find sample by linearly going through available shards samples
        for base_group in self.base_groups[LIDAR_CAMERA_VISIBILITY_BASE_GROUP]:
            try:
                ds = base_group[lidar_id][str(frame_timestamps_us)]
            except KeyError:
                # it's ok if the key isn't in the current shard - continue look in next shard
                continue

            packed_visibility_mask = np.asarray(ds).astype(np.uint8)
            visibility_mask = np.unpackbits(packed_visibility_mask, axis=1)

            visibility_mask_dict = {}
            meta = self.get_lidar_camera_visibility_meta(lidar_id)["camera_visibility_order"]
            num_valid_cameras = len(meta)
            camera_ids = list(meta.keys()) if len(camera_ids) == 0 else camera_ids
            for camera_id in camera_ids:
                assert camera_id in meta and meta[camera_id] < num_valid_cameras, (
                    "get_lidar_camera_visibility: invalid camera id"
                )
                visibility_mask_dict[camera_id] = visibility_mask[:, meta[camera_id]]
            return visibility_mask_dict

        raise KeyError(f"semantic segmentation not found for {lidar_id} and timestamp {frame_timestamps_us}")

    def has_egomask(self, camera_id: str | None = None) -> bool:
        """Check if ego-mask data exists. If camera_id is provided, check availability for that camera."""
        return self._has_base_group(EGO_MASK_BASE_GROUP, camera_id)

    def get_egomask_meta(self, camera_id: str) -> dict:
        """Return metadata for ego-mask estimation for a given camera."""
        if not self.has_egomask(camera_id):
            raise KeyError(f"No ego-mask data found for {camera_id}")
        # Take meta from first shard
        return dict(self.base_groups[EGO_MASK_BASE_GROUP][0][camera_id].attrs)

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

    def get_aggregated_egomask(self, camera_id: str) -> np.ndarray:
        """Retrieve the aggregated super ego-mask (binary numpy array) for a given camera."""
        return self.get_egomask(camera_id, 0)  # aggregated mask is stored at special timestamp '0'


def get_mask_image(
    mask_image: PILImage.Image | None, target_mask_size: tuple[int, int], mask_override_path: str | None = None
) -> np.ndarray | None:
    """
    Returns a boolean mask for, e.g., a camera sensor, scaled to the target resolution if required.

    This function retrieves a mask image, either from the provided image itself or from an optional override path.
    The mask image is converted to grayscale and resized to match the camera sensor's resolution if their aspect ratios are sufficiently close.
    The resulting mask is returned as a NumPy boolean array, where `True` indicates masked-out regions.

    Args:
        mask_image (PILImage.Image | None): The mask image to be processed.
        target_mask_size (tuple[int, int]): The target size (width, height) to resize the mask image to.
        mask_override_path (str | None, optional): Path to an external mask image to override the sensor's default mask. Defaults to None.

    Returns:
        np.ndarray | None: A boolean NumPy array representing the mask, or None if no mask image is available.

    Raises:
        AssertionError: If the aspect ratio of the mask image does not match the camera sensor's resolution within a tolerance.
    """

    if mask_override_path is not None:
        mask_image = cast(PILImage.Image, PILImage.open(mask_override_path))

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
    camera_sensor: ncore_internal.data.v3.CameraSensor | ncore.data.CameraSensorProtocol,
    mask_override_path: str | None = None,
) -> np.ndarray | None:
    """
    Returns a boolean mask for a camera sensor, scaled to the sensor's resolution if required.

    This function retrieves a mask image for the given camera sensor, either from the sensor itself or from an optional override path.
    The mask image is converted to grayscale and resized to match the camera sensor's resolution if their aspect ratios are sufficiently close.
    The resulting mask is returned as a NumPy boolean array, where `True` indicates masked-out regions.

    Args:
        camera_sensor (ncore_data.CameraSensor | ncore_data4_compat.CameraSensorProtocol): The camera sensor  / sensor protocol providing model parameters and default mask image.
        mask_override_path (str | None, optional): Path to an external mask image to override the sensor's default mask. Defaults to None.

    Returns:
        np.ndarray | None: A boolean NumPy array representing the mask, or None if no mask image is available.

    Raises:
        AssertionError: If the aspect ratio of the mask image does not match the camera sensor's resolution within a tolerance.
    """

    camera_mask_image: PILImage.Image | None
    match camera_sensor:
        case ncore_internal.data.v3.CameraSensor():
            camera_mask_image = camera_sensor.get_camera_mask_image()
            resolution = camera_sensor.get_camera_model_parameters().resolution
        case ncore.data.CameraSensorProtocol():
            # V4 potentially provides more than a single mask, use 'ego' mask if available
            camera_mask_image = camera_sensor.get_mask_images().get("ego")
            resolution = camera_sensor.model_parameters.resolution
        case _:
            raise ValueError(f"{__name__} get_camera_sensor_mask: unsupported camera sensor type {type(camera_sensor)}")

    return get_mask_image(
        camera_mask_image,
        tuple(resolution),
        mask_override_path,
    )


# Common V3 / V4 sequence data loading
NCoreDataFormat = Enum("NCoreDataFormat", "V3 V4")


def parse_sequence_meta_file(sequence_meta_file: UPath) -> tuple[NCoreDataFormat, str, HalfClosedInterval, list[UPath]]:
    """
    Parse sequence meta file to extract data format, time interval, and dataset file paths.

    Args:
        sequence_meta_file (UPath): Path to the sequence meta file.
    """

    assert sequence_meta_file.is_file(), f"{__name__} provided path {sequence_meta_file} not a file"

    with sequence_meta_file.open("r") as fp:
        try:
            dataset_meta = json.load(fp)
        except ValueError as e:
            raise ValueError(f"{__name__} provided file {sequence_meta_file} not a json file") from e

    if not ((version := dataset_meta.get("version")) is not None and version.startswith("v4")):
        # V3 single-sequence meta file

        # sanity check schema
        assert all((key in dataset_meta for key in ("sequence_id", "pose-range", "shards", "shard-ids"))), (
            f"{__name__} provided json file {sequence_meta_file} not a NCore V3 single-sequence file"
        )

        # this is V3
        data_format = NCoreDataFormat.V3

        # determine time-range
        time_range_us = HalfClosedInterval(
            dataset_meta["pose-range"]["start-timestamp_us"],
            # make sure that the final value is included in the half-closed interval
            dataset_meta["pose-range"]["end-timestamp_us"] + 1,
        )

        # collect shards paths relative to meta file
        dataset_paths = [sequence_meta_file.parent / shard["path"] for shard in dataset_meta["shards"]]
    else:
        # V4 single-sequence meta file

        # sanity check schema
        assert all(
            (
                key in dataset_meta
                for key in ("sequence_id", "sequence_timestamp_interval_us", "version", "component_stores")
            )
        ), f"{__name__} provided json file {sequence_meta_file} not a NCore V4 single-sequence file"

        # this is V4
        data_format = NCoreDataFormat.V4

        # determine time-range of subsection
        time_range_us = HalfClosedInterval(
            dataset_meta["sequence_timestamp_interval_us"]["start"],
            dataset_meta["sequence_timestamp_interval_us"]["stop"],
        )

        # collect component store paths relative to meta file
        dataset_paths = [
            sequence_meta_file.parent / component_store["path"] for component_store in dataset_meta["component_stores"]
        ]

    return data_format, dataset_meta["sequence_id"], time_range_us, dataset_paths


def create_sequence_loader(
    # Common attributes
    data_format: NCoreDataFormat,
    dataset_paths: list[UPath],
    open_consolidated: bool,
    # V3-specific attributes
    v3_cuboid_loading_max_workers: Optional[int],
    # V4-specific attributes
    v4_poses_component_group: str,
    v4_intrinsics_component_group: str,
    v4_masks_component_group: str,
    v4_cuboids_component_group: str,
) -> ncore.data.SequenceLoaderProtocol:
    """
    Create a sequence loader for the specified NCore data format (V3 or V4).
    """
    match data_format:
        case NCoreDataFormat.V3:
            return ncore_internal.data.v3.SequenceLoaderV3(
                ncore_internal.data.v3.ShardDataLoader(shard_paths=dataset_paths, open_consolidated=open_consolidated),
                cuboid_loading_max_workers=v3_cuboid_loading_max_workers,
            )
        case NCoreDataFormat.V4:
            return ncore.data.v4.SequenceLoaderV4(
                ncore.data.v4.SequenceComponentGroupsReader(dataset_paths, open_consolidated=open_consolidated),
                poses_component_group_name=v4_poses_component_group,
                intrinsics_component_group_name=v4_intrinsics_component_group,
                masks_component_group_name=v4_masks_component_group,
                cuboids_component_group_name=v4_cuboids_component_group,
            )
        case _:
            raise ValueError(f"{__name__} create_sequence_loader: unsupported data format {data_format}")

