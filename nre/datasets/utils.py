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
import dataclasses
import io
import lzma
import math
import struct

from collections import defaultdict
from functools import lru_cache
from typing import Generator, List, Optional, Tuple, TypeVar, Union

import cv2
import lietorch as lt
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torchvision
import tqdm

from scipy import ndimage

import ncore.data
import ncore.impl.common.transformations as ncore_transformations
import ncore.sensors
import ncore_internal.data.v3
import nre.utils.ncore_utils as ncore_utils

from libs.vren.interface import (  # type: ignore
    world_points_to_image_points_shutter_pose,
)
from nre.config import (
    CuboidTracksConfig,
    ValidPixelsCuboidTrackConfig,
    ValidPixelsFrameMaskConfig,
    ValidPixelsSceneFlowConfig,
    ValidPixelsTrafficLightConfig,
)
from nre.datasets.tracks import CuboidTracks, TrackFlags
from nre.utils.batch import RectSubsampled
from nre.utils.geometry import (
    PoseLinearVelocityInterpolator,
    quat_to_euler,
    se3_matrix_to_se3,
    se3_matrix_to_tquat,
)
from nre.utils.misc import (
    assert_default_device_on_local_rank,
    set_default_device,
    to_torch,
    unpack_optional,
)
from nre.utils.types import AABB3D, CuboidTracksData, HalfClosedInterval, PointCloud, TracksData
def get_visualdebugger():
    class _Null:
        def __getattr__(self, _n):
            return lambda *a, **k: None
    return _Null()


Tensor = TypeVar("Tensor", np.ndarray, torch.Tensor)

# Re-export MorphOp for backwards compatibility
from nre.utils.morph import MorphOp


def nerf_matrix_to_nre(pose: np.ndarray, offset: Union[np.ndarray, list] = [0, 0, 0]) -> np.ndarray:
    """
    Convert c2w matrices from OpenGL convention to OpenCV convention

    < OpenGL convention >
    facing [-z] direction, y upwards, x right
            y
            ^
            |
            o------> x
           /
         z

    < OpenCV convention >
    facing [+z] direction, y downwards, x right
              z
             /
            o------> x
            |
            ↓
            y
    """

    pose[:3, 1] *= -1.0
    pose[:3, 2] *= -1.0
    pose[:3, 3] += np.array(offset)

    return pose


def nre_matrix_to_nerf(nre_matrix: np.ndarray, offset: Union[np.ndarray, list] = [0, 0, 0]) -> np.ndarray:
    nre_matrix[1] *= -1.0
    nre_matrix[2] *= -1.0
    nre_matrix[3] -= np.array(offset)

    return nre_matrix


def fov_to_focal_length(resolution: int, fov_radians: float) -> float:
    return 0.5 * resolution / np.tan(0.5 * fov_radians)


def load_pc_dat(file_path: str, allow_lookup_fallback: bool = True) -> np.ndarray:
    """
    Loads binary .dat / .dat.xz files representing a 2D single-precision array.
    Serialized 2D arrays usually represent a point-clouds with columns defined as

    [x_s, y_s, z_s, x_e, y_e, z_e, dist, intensity, dynamic_flag]

    - xys_s / xyz_e: the start / end point of world rays
    - dist: the norm of the ray
    - intensity: lidar intensity response value for this point
    - dynamic_flag:
      - -1: if the information is not available,
      -  0: static
      -  1: = dynamic

    Args:
        file_path: path to .dat / .dat.xz file to load.
        allow_lookup_fallback: If enabled, will fall back to .dat.xz/.dat, resp., in case loading .dat/.dat.xz fails (for backwards-compatibility).
    Return:
        lidar_data: loaded 2D single-precision array
    """

    def load(file: Union[io.BufferedReader, lzma.LZMAFile]) -> np.ndarray:
        # The first number denotes the number of points
        n_rows, n_columns = struct.unpack("<ii", file.read(8))
        # The remaining data are floats saved in little endian
        # Columns usually contain: x_s, y_s, z_s, x_e, y_e, z_e, d, intensity, dynamic_flag
        # Dynamic flag is set to -1 if the information is not available, 0 static, 1 = dynamic
        return np.array(struct.unpack("<%sf" % (n_rows * n_columns), file.read()), dtype=np.float32).reshape(
            n_rows, n_columns
        )

    if file_path.endswith(".dat"):
        try:
            with open(file_path, "rb") as file:
                lidar_data = load(file)
        except FileNotFoundError as e:
            if allow_lookup_fallback:
                with lzma.open(file_path + ".xz", "rb") as lzma_file:
                    lidar_data = load(lzma_file)
            else:
                raise e
    elif file_path.endswith(".dat.xz"):
        try:
            with lzma.open(file_path, "rb") as lzma_file:
                lidar_data = load(lzma_file)
        except FileNotFoundError as e:
            if allow_lookup_fallback:
                with open(file_path.replace(".dat.xz", ".dat"), "rb") as file:
                    lidar_data = load(file)
            else:
                raise e
    else:
        raise ValueError("invalid file format provided, supporting .dat / .dat.xz files only")

    return lidar_data


def get_indices_of_points_visible_in_image(
    points: npt.NDArray | torch.Tensor,
    nre_to_sensor_startend: npt.NDArray,
    camera_params: ncore.data.ConcreteCameraModelParametersUnion,
    device: torch.device,
    camera_timestep_start: int,
    camera_timestep_end: int,
    point_filter: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Returns the indices of points that are visible in the image

    Args:
        points: (N, 3) array of points in NRE frame [npt.NDArray or torch.Tensor]
        nre_to_sensor_startend: (2, 4, 4) array of start/end poses of the sensor in NRE frame [npt.NDArray]
        camera_params: camera parameters [ncore.data.ConcreteCameraModelParametersUnion]
        device: device to run the operation on [torch.device]
        camera_timestep_start: start timestamp of the camera frame [int]
        camera_timestep_end: end timestamp of the camera frame [int]
        point_filter: optional additional (H, W) valid point filter [torch.Tensor]
    Returns:
        valid_indices: (N,) tensor of indices of points that are visible in the image [torch.Tensor]
    """
    if not points.ndim == 2 and points.shape[1] == 3:
        raise ValueError("Points must be a 2D array with 3 columns")

    if isinstance(points, np.ndarray):
        points = to_torch(points, device=device)

    camera_model = ncore.sensors.CameraModel.from_parameters(camera_params, device=str(device))
    if str(device) == "cpu":
        image_points = camera_model.world_points_to_image_points_shutter_pose(
            points,
            nre_to_sensor_startend[0],
            nre_to_sensor_startend[1],
            return_valid_indices=True,
        )
    else:
        image_points = world_points_to_image_points_shutter_pose(
            camera_params,
            points,
            se3_matrix_to_tquat(nre_to_sensor_startend),
            torch.LongTensor([camera_timestep_start, camera_timestep_end]),
        )

    valid_indices = unpack_optional(image_points.valid_indices)
    if point_filter is not None:
        pixels = image_points.image_points.int()
        valid_indices = valid_indices[point_filter[pixels[..., 1], pixels[..., 0]]]

    return valid_indices


def color_pc_semantics(
    semantic_class_id: torch.Tensor,
    semantic_colormap: torch.Tensor,
    ignore_label_class: int,
) -> torch.Tensor:
    """Converts class ids to colors from a color map"""
    color_for_ignore_label = torch.zeros(3, dtype=torch.uint8, device=semantic_colormap.device).unsqueeze(0)
    semantic_colormap = torch.concat([semantic_colormap, color_for_ignore_label])
    colors = semantic_colormap[
        torch.where(
            semantic_class_id == ignore_label_class,
            len(semantic_colormap) - 1,
            semantic_class_id.long(),
        )
    ]
    return colors


def visualise_point_cloud(points: np.ndarray, col: np.ndarray) -> None:
    """Debugging functionality to visualize point clouds"""
    visualdebugger = get_visualdebugger()
    visualdebugger.clear()
    visualdebugger.add_point_cloud(
        "Lidar points",
        points,
        colors_quantities={"All points": col},
        enabled=True,
        radius=0.0004,
    )
    visualdebugger.show()


def get_sampled_pixels(
    sampler_sampled_pixels: np.ndarray | RectSubsampled,
    camera_resolution: tuple[int, int],
    camera_all_pixels: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Determine the pixel indices to sample rays / image signals from depending on the pixel-samplers results

    Also returns a flag indicating whether the rays should be sampled at half-pixel offsets
    """

    # Get full resolution
    image_width, image_height = camera_resolution

    # extract pixel indices to sample rays / image signals from
    match sampler_sampled_pixels:
        case np.ndarray():
            # Constructing camera batch for free / unstructured pixels
            pixels = sampler_sampled_pixels

            # Always stick to rays associated with centers pixel
            halfpixoffset_rays = False
        case RectSubsampled(
            subsample_factor=subsample_factor,
            i=i,
            j=j,
            width=width,
            height=height,
            original_width=original_width,
            original_height=original_height,
        ):
            # Constructing camera batch for a structured rectangular pixel region
            n_subsample = int(subsample_factor)
            # Assert that subsample_factor is integer
            assert n_subsample == subsample_factor, (
                f"Subsample factor {subsample_factor} in the sampler_sampled_pixels must be an integer for now."
            )

            # Subsample pixels from all pixels according to the subsample factor
            frame_all_pixels_subsampled = camera_all_pixels.reshape(image_height, image_width, 2)[
                ::n_subsample, ::n_subsample
            ]

            # Offset the subsampled pixel coordinates to minimize the correlation errors of downsampled
            # pixel domains relative to the full pixel domain. This is to bound the correspondence of subsampled
            # pixel centers relative to corresponding full image pixel centers to 0.5px (even subsample factors)
            # or 0px (odd subsample factors) [see visualizations in original MR]
            #
            # The _remaining_ 0.5px offset is accounted for ray sampling below.
            pixels_subsampled_offset = (n_subsample - 1) // 2

            # Select the pixels according to ROI, applying offset
            pixels = (
                frame_all_pixels_subsampled[j : (j + height), i : (i + width)].reshape(-1, 2) + pixels_subsampled_offset
            )

            # If we guarantee that the image is contiguous _and_ the subsample factor is _even_, we need to
            # select rays at half-pixel offsets to be consistent with the centers of the downsampled pixels.
            # If subsampling the factor is odd, the subselected pixels are aligned to the image-point
            # centers of the downsampled pixels, so default rays from original pixel centers are used in
            # that case also
            halfpixoffset_rays = n_subsample % 2 == 0
        case _:
            raise ValueError(f"Unexpected sampled pixels type {type(sampler_sampled_pixels)}")

    return pixels, halfpixoffset_rays


class PackedMask:
    """Wrapper for memory-efficient bit-packed binary masks (immutable)"""

    mask_shape: Tuple[int, ...]  # source-resolution of the mask
    mask_packed: npt.NDArray[np.uint8]  # the uint8-packed and flattened (potentially padded) mask

    def __init__(self, mask: np.ndarray):
        assert mask.dtype == np.bool_, f"PackedMask: expecting binary mask of type np.bool_, received {mask.dtype}"
        self.mask_shape = mask.shape
        self.mask_packed = np.packbits(mask.flatten())

    def unpacked(self) -> npt.NDArray[np.bool_]:
        """Returns the unpacked mask in the original resolution"""
        return (np.unpackbits(self.mask_packed, count=math.prod(self.mask_shape)).astype(np.bool_)).reshape(
            self.mask_shape
        )


def chunk_sizes(n: int, size: int) -> Generator[int, None, None]:
    """Divides size into n approximately equally sized subdivisions such that they sum to size

    Example: list(chunk_sizes(3, 4096)) == [1365, 1365, 1366]
    """
    start = 0
    for i in range(1, n):
        stop = i * size // n  # evenly sized initial elements
        yield stop - start
        start = stop
    yield size - start  # collect overlap in last element


def test_chunk_size() -> None:
    l = list(chunk_sizes(1, 4096))
    assert sum(l) == 4096

    l = list(chunk_sizes(2, 4096))
    assert sum(l) == 4096

    l = list(chunk_sizes(3, 4096))
    assert sum(l) == 4096


def compute_valid_pixels_ego(
    camera_frame_ranges: dict[str, range], cameras_valid_pixels_ego_masks: dict[str, PackedMask]
) -> dict[str, dict[int, PackedMask]]:
    """
    Initializes per camera frame valid-pixels masks to the static ego camera masks
    (initialized masks might be refined subsequently).

    Args:
        camera_frame_ranges (dict[str, range]): The range for each camera on which to compute the masks.
        cameras_valid_pixels_ego_masks (dict[str, PackedMask]): The static ego camera masks for each camera.

    Returns:
        dict[str, dict[int, PackedMask]]: Valid pixel masks for each camera and each frame within the range.
    """
    cameras_frame_valid_pixels_masks: dict[str, dict[int, PackedMask]] = defaultdict(dict)
    for camera_id, camera_frame_range in camera_frame_ranges.items():
        for camera_frame_idx in camera_frame_range:
            cameras_frame_valid_pixels_masks[camera_id] |= {
                camera_frame_idx:
                # store per-frame ego mask as reference only as packed masks are immutable (reducing memory requirements)
                cameras_valid_pixels_ego_masks[camera_id]
            }
    return cameras_frame_valid_pixels_masks


def lidar_frame_dynamic_flag(lidar_sensor: ncore.data.LidarSensorProtocol, frame_idx: int) -> Optional[np.ndarray]:
    """Loads the (deprecated) dynamic flag for a given lidar sensor and frame index."""

    dynamic_flag: np.ndarray | None = None

    # 'dynamic_flag' is deprecated in NCore and moved to non-mandatory generic frame
    # data for new datasets. However, old V3 datasets still exposes this as mandatory frame
    # data, so check this first and fall-back to secondary APIs otherwise
    match lidar_sensor:
        case ncore_internal.data.v3.SequenceLoaderV3.LidarSensor():
            # compat sensor on V3 data
            if (lidar_sensor_v3 := lidar_sensor.lidar_sensor).has_frame_data(frame_idx, "dynamic_flag"):
                # load from deprecated mandatory frame data
                dynamic_flag = lidar_sensor_v3.get_frame_data(frame_idx, "dynamic_flag")

    # fall back to non-mandatory generic frame data
    if dynamic_flag is None and lidar_sensor.has_frame_generic_data(frame_idx, "dynamic_flag"):
        dynamic_flag = lidar_sensor.get_frame_generic_data(frame_idx, "dynamic_flag")

    return dynamic_flag


def compute_cameras_valid_pixels_frame_mask(
    camera_sensors: dict[str, ncore.data.CameraSensorProtocol],
    camera_frame_ranges: dict[str, range],
    valid_pixels_frame_mask_params: ValidPixelsFrameMaskConfig,
    cameras_frame_valid_pixels_masks: dict[str, dict[int, PackedMask]],
    tqdm_disabled: bool,
) -> dict[str, dict[int, PackedMask]]:
    """
    Uses per-frame masks from frame data to construct per-frame masks by loading each key via `get_frame_generic_data`.
    Includes pixels marked as true in any of the per-key masks. Updates cameras_frame_valid_pixels_masks.

    Args:
        camera_sensors (dict[str, CameraSensorProtocol]): Contains the mask data.
        camera_frame_ranges (dict[str, range]): The per-camera range of data.
        valid_pixels_frame_mask_params (ValidPixelsFrameMaskConfig): Contains the classes which will be combined.
        cameras_frame_valid_pixels_masks (dict[str, dict[int, PackedMask]]): The masks to be updated. This input data will be modified.
        tqdm_disabled (bool): If True, will not show progress bars.

    Returns:
        dict[str, dict[int, PackedMask]]. Updated cameras_frame_valid_pixels_masks.
    """

    frame_mask_classes = valid_pixels_frame_mask_params.classes
    assert frame_mask_classes is not None and len(frame_mask_classes) > 0, (
        f"compute_cameras_valid_pixels_frame_mask: need at least one frame mask class"
    )

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for camera_id, camera_frame_range in tqdm.tqdm(
            camera_frame_ranges.items(), desc="Frame Masks [cameras]", disable=tqdm_disabled
        ):
            camera_sensor = camera_sensors[camera_id]

            frame_valid_pixels_masks = cameras_frame_valid_pixels_masks[camera_id]

            def thread_camera_frame_mask(camera_frame_idx):
                # this has been initialized before
                frame_valid_pixel_mask = frame_valid_pixels_masks[camera_frame_idx].unpacked()

                # constructed mask for current frame by combining all masks from the enabled classes
                include_mask = camera_sensor.get_frame_generic_data(camera_frame_idx, frame_mask_classes[0]).astype(
                    bool
                )
                for key in frame_mask_classes[1:]:
                    include_mask = np.logical_or(
                        include_mask,
                        camera_sensor.get_frame_generic_data(camera_frame_idx, key).astype(bool),
                    )

                # dilate the mask
                exclude_mask = ndimage.binary_dilation(
                    np.logical_not(include_mask),
                    iterations=valid_pixels_frame_mask_params.n_dilation_iterations,
                )

                include_mask = np.logical_not(exclude_mask)

                # mask out all pixels from all masks of this frame
                frame_valid_pixel_mask &= include_mask

                frame_valid_pixels_masks[camera_frame_idx] = PackedMask(frame_valid_pixel_mask)

            for _ in tqdm.tqdm(
                concurrent.futures.as_completed(
                    [
                        executor.submit(thread_camera_frame_mask, camera_frame_idx)
                        for camera_frame_idx in camera_frame_range
                    ]
                ),
                desc="Frame Masks [cameras-frames]",
                total=len(camera_frame_range),
                disable=tqdm_disabled,
            ):
                pass
    return cameras_frame_valid_pixels_masks


def compute_valid_lidarpoints_all(
    lidar_frame_ranges: dict[str, range],
    lidar_sensors: dict[str, ncore.data.LidarSensorProtocol],
) -> dict[str, dict[int, PackedMask]]:
    """Initializes per lidar frame valid-point masks to be unconditionally valid for all points (initialized masks might be refined subsequently).

    Note that certain invalid points (like self-measurements of the ego vehicle) have already been pre-filtered as part of
    the NCore data preprocessing and all remaining lidar points can therefore be considered to be valid up to subsequent restrictions.

    Args:
        lidar_frame_ranges (dict[str, range]): Per lidar frame ranges in which to compute masks.
        lidar_sensors: The lidar sensor data which contains the number of points per frame.
        lidar_model_parameters: The lidar model parameters.

    Returns:
        dict[str, dict[int, PackedMask]]: The initialized (unconditionally valid) point masks to be refined later.
    """
    lidars_frame_valid_points_masks: dict[str, dict[int, PackedMask]] = defaultdict(dict)
    for lidar_id, lidar_frame_range in lidar_frame_ranges.items():
        lidar_sensor = lidar_sensors[lidar_id]
        lidars_frame_valid_points_masks[lidar_id] = {}
        for lidar_frame_idx in lidar_frame_range:
            n_frame_points = lidar_sensor.get_frame_ray_bundle_count(lidar_frame_idx)
            lidars_frame_valid_points_masks[lidar_id] |= {
                lidar_frame_idx: PackedMask(np.full((n_frame_points,), True, dtype=np.bool_))
            }
    return lidars_frame_valid_points_masks


def compute_cuboid_df(
    sequence_loader: ncore.data.SequenceLoaderProtocol,
    time_range_us: HalfClosedInterval,
    tqdm_disabled: bool = True,
    serialize_observation: bool = True,
) -> pd.DataFrame:
    """Extracts cuboid observations from the dataset in a given time range

    Args:
        sequence_loader (ncore.data.SequenceLoaderProtocol): The sequence loader to load the data from
        time_range_us (HalfClosedInterval): The time range in which to compute the tracks for
        tqdm_disabled (bool): When true, no progress bars are showed
        serialize_observation (bool): When true, serialize the cuboid observation into json dicts (very slow)

    Returns:
        pd.DataFrame: Dataframe of all cuboid observations in the given time range
    """

    cuboid_observations = sequence_loader.get_cuboid_track_observations(
        timestamp_interval_us=ncore_transformations.HalfClosedInterval(
            start=time_range_us.start,
            # Note: ncore's HalfClosedInterval type correctly calls this non-exclusive boundary `stop`,
            # whereas the notion within NRE's HalfClosedInterval with `end` (which is inconsistently used as
            # inclusive / exclusive) is messy
            stop=time_range_us.end,
        ),
    )

    # Load all cuboid observations into dataframe for easy querying
    cuboid_dicts = []
    for observation in tqdm.tqdm(
        cuboid_observations,
        desc="Load all Cuboid Labels",
        disable=tqdm_disabled,
    ):
        cuboid_dicts.append(observation.to_dict() if serialize_observation else vars(observation))

    if len(cuboid_dicts):
        # load all observations into dataframe, deducing dynamic types from structure
        cuboids_df = pd.DataFrame.from_records(cuboid_dicts)
    else:
        # initialize empty cuboids dataframe, inheriting all top-level fields from CuboidTrackObservation type
        cuboids_df = pd.DataFrame(
            {
                field.name: pd.Series(dtype="object" if field.type not in ("int", "float", "bool") else field.type)
                for field in dataclasses.fields(ncore.data.CuboidTrackObservation)
            }
        )

    # Make sure observations are ordered in time (so per-track poses are ordered as well)
    return cuboids_df.sort_values(by=["track_id", "timestamp_us"], ascending=[True, True])


def consolidate_cuboid_tracks(
    cuboids_df: pd.DataFrame,
    sequence_loader: ncore.data.SequenceLoaderProtocol,
    track_label_sources: list[str] | None,
    track_min_centroid_rig_dist_m: float,
    T_world_world_base: np.ndarray | None,
    tqdm_disabled: bool,
) -> dict[str, dict]:
    """
    Gather the cuboid track observations into a set of named tracks with timestamped poses relative to the world frame.

    Args:
        cuboids_df (pd.DataFrame): Dataframe of all cuboid observations to consolidate into tracks
        sequence_loader (ncore.data.SequenceLoaderProtocol): The sequence loader to load the data from
        track_label_sources (list[str] | None): List of label sources to consider for track consolidation
        track_min_centroid_rig_dist_m (float): Minimum distance of the track centroid to the rig frame to consider the observation
        T_world_world_base (np.ndarray | None): Optional base transformation to apply to all world poses
        tqdm_disabled (bool): When true, no progress bars are showed

    Returns:
        dict[str, dict]: Consolidated cuboid tracks (indexed by track-id) with elements:
            - dimension: (np.ndarray) The (l,w,h) dimensions of the cuboid
            - label_class: (int) The class id of the cuboid
            - poses: (list[np.ndarray]) List of cuboid poses in world frame
            - timestamps_us: (list[int]) List of timestamps corresponding to the poses
    """

    # Container to keep track of all observed label sources (@-concatenation of label source names and optional versions or 'any')
    all_label_sources_set: set[str] = set()

    # Setup the set of valid label sources set (@-concatenation of label source names and optional versions or 'any')
    valid_label_sources: set[str] | None = None
    if track_label_sources is not None:
        valid_label_sources = set()
        for track_label_source in track_label_sources:
            # check if specified config label source is versioned
            if len(track_label_source.split("@", 1)) > 1:
                # if version is specified by the config, require this specific version for valid label sources
                valid_label_sources.add(track_label_source)
            else:
                # otherwise any version of this label source type is considered valid
                valid_label_sources.add(track_label_source + "@any")

    # Cache evaluated transformations
    @lru_cache(maxsize=(n_expected_poses := 20 * 10 * 60))  # cache up to 20 minutes of poses at 10Hz
    def get_T_reference_world(reference_frame_id: str, reference_frame_timestamp_us: int) -> np.ndarray:
        T_reference_world = sequence_loader.pose_graph.evaluate_poses(
            reference_frame_id, "world", np.array(reference_frame_timestamp_us, dtype=np.uint64)
        )

        if T_world_world_base is not None:
            T_reference_world = T_world_world_base @ T_reference_world

        return T_reference_world

    @lru_cache(maxsize=n_expected_poses)
    def get_T_reference_rig(reference_frame_id: str, reference_frame_timestamp_us: int) -> Optional[np.ndarray]:
        try:
            return sequence_loader.pose_graph.evaluate_poses(
                reference_frame_id, "rig", np.array(reference_frame_timestamp_us, dtype=np.uint64)
            )
        except KeyError:
            return None

    # Extract all tracks for the given data range
    all_tracks: dict[str, dict] = {}
    for _, row in tqdm.tqdm(cuboids_df.iterrows(), desc="Associate Cuboid Labels -> Tracks", disable=tqdm_disabled):
        observation = (
            ncore.data.CuboidTrackObservation.from_dict(row.to_dict())
            if isinstance(
                row.bbox3, dict
            )  # Check by the bbox3 field to see if the observation is serialized (otherwise it should be BBox3)
            else ncore.data.CuboidTrackObservation(**row.to_dict())
        )

        # evaluate transformations
        T_reference_world = get_T_reference_world(
            observation.reference_frame_id, observation.reference_frame_timestamp_us
        )
        T_reference_rig = get_T_reference_rig(observation.reference_frame_id, observation.reference_frame_timestamp_us)

        # skip self-classifications if rig frame is available
        if T_reference_rig is not None:
            bbox_rig = ncore_transformations.transform_bbox(observation.bbox3.to_array(), T_reference_rig)
            if (
                np.linalg.norm(bbox_rig[:3]) < track_min_centroid_rig_dist_m
            ):  # skip observations that are too close to the rig center
                continue

        # collect observation source (@-concatenation of label source names and optional versions or 'any')
        all_label_sources_set.add(
            versioned_label_source := observation.source.name + f"@{unpack_optional(observation.source_version, 'any')}"
        )

        # handle label source filtering, if enabled
        if valid_label_sources is not None:
            if not (
                (
                    # check versioned source type
                    versioned_label_source in valid_label_sources
                )
                or (
                    # check unversioned source type
                    observation.source.name + "@any" in valid_label_sources
                )
            ):
                # skip label if not in enabled set of label sources
                continue

        if observation.track_id not in all_tracks:
            # instantiate new track
            all_tracks[observation.track_id] = {
                # track-constants:
                "dimension": observation.bbox3.to_array()[3:6],
                "label_class": observation.class_id,
                # per track instance data:
                "poses": [],
                "timestamps_us": [],
            }

        # track to update with this instance's pose / speed data
        track = all_tracks[observation.track_id]

        # Skip if it is a duplicated timestamp
        if observation.timestamp_us in track["timestamps_us"][-2:]:
            continue

        track["timestamps_us"].append(observation.timestamp_us)
        track["poses"].append(
            ncore_transformations.bbox_pose(
                ncore_transformations.transform_bbox(observation.bbox3.to_array(), T_reference_world)
            )
        )

    # Error out if there is more than a single label source in the input datasets and an explicit selection is *not* provided
    if len(all_label_sources_set) > 1 and (track_label_sources is None):
        raise ValueError(
            f"compute_cuboid_tracks: Observed {all_label_sources_set} track label sources in input data but no explicit "
            "track label source specified - please specify *explicit* list of track sources with "
            "the dataset.cuboid_tracks_params.track_label_sources config"
        )

    return all_tracks


def compute_cuboid_tracks(
    sequence_loader: ncore.data.SequenceLoaderProtocol,
    time_range_us: HalfClosedInterval,
    cuboid_tracks_params: CuboidTracksConfig,
    tqdm_disabled: bool,
) -> tuple[CuboidTracks, pd.DataFrame]:
    """
    Extracts all cuboid tracks from the dataset in a given frame range

    Args:
        sequence_loader (ncore.data.SequenceLoaderProtocol): The sequence loader to load the data from
        time_range_us (HalfClosedInterval): The time range in which to compute the tracks for
        cuboid_tracks_params (CuboidTracksConfig): Contains configurable parameters for cuboid tracks
        tqdm_disabled (bool): When true, no progress bars are showed

    Returns:
        CuboidTracks: All extracted cuboid tracks, as well as a dataframe of all cuboid observations
    """
    cuboids_df = compute_cuboid_df(sequence_loader, time_range_us, tqdm_disabled)
    all_tracks = consolidate_cuboid_tracks(
        cuboids_df=cuboids_df,
        sequence_loader=sequence_loader,
        track_label_sources=cuboid_tracks_params.track_label_sources,
        track_min_centroid_rig_dist_m=cuboid_tracks_params.track_min_centroid_rig_dist_m,
        T_world_world_base=None,
        tqdm_disabled=tqdm_disabled,
    )

    # Store all tracks and classify dynamic trajectories based on the speed threshold / unconditional dynamic flag
    all_track_ids = []
    all_tracks_poses = []
    all_tracks_timestamps_us = []
    all_tracks_label_class = []
    all_tracks_flags = []
    all_cuboid_dims = []

    for track_id, track in tqdm.tqdm(all_tracks.items(), desc="Assemble Tracks [tracks]", disable=tqdm_disabled):
        if len(track["timestamps_us"]) <= 1:
            continue

        # initialize track-associated pose-interpolator
        poses_list: list[np.ndarray] = track["poses"]
        timestamps_us_list: list[int] = track["timestamps_us"]
        track_flags = TrackFlags.NONE

        # perform pose extrapolation by one instance into the past / future if enabled to
        # improve interpolation coverage for frames that were measured before / after the first / last
        # available track pose
        if cuboid_tracks_params.track_extrapolate:
            # extrapolate first pose to the past
            poses_list.insert(
                0,
                # extrapolate into pre-time P = (P_1 @ P_0^-1)^-1 @ P_0 = (P_0 @ P_1^-1) @ P_0
                (poses_list[0] @ ncore_transformations.se3_inverse(poses_list[1])) @ poses_list[0],
            )
            timestamps_us_list.insert(0, timestamps_us_list[0] - (timestamps_us_list[1] - timestamps_us_list[0]))

            # extrapolate last pose to the future
            poses_list.append(
                # extrapolate into post-time P = (P_N @ P_{N-1}^-1) @ P_N
                (poses_list[-1] @ ncore_transformations.se3_inverse(poses_list[-2])) @ poses_list[-1],
            )
            timestamps_us_list.append(timestamps_us_list[-1] + (timestamps_us_list[-1] - timestamps_us_list[-2]))

        poses = np.stack(poses_list, dtype=np.float32)
        timestamps_us = np.stack(timestamps_us_list)

        # store all tracks unconditionally
        all_track_ids.append(track_id)
        all_tracks_poses.append(poses)
        all_tracks_timestamps_us.append(timestamps_us)
        all_tracks_label_class.append(track["label_class"])
        all_tracks_flags.append(track_flags)
        all_cuboid_dims.append(track["dimension"])

    # Map to member structs
    cuboidtracks_all = CuboidTracks.Factory.from_numpy(
        all_track_ids,
        all_tracks_poses,
        all_tracks_timestamps_us,
        all_tracks_label_class,
        all_tracks_flags,
        cuboids_dims=all_cuboid_dims,
    )

    return cuboidtracks_all, cuboids_df


def classify_dynamic_cuboid_tracks(
    cuboidtracks: CuboidTracks,
    cuboid_tracks_params: CuboidTracksConfig,
    unconditionally_dynamic_classes: set[str],
    tqdm_disabled: bool,
    cuboidtracks_visible_intervals: Optional[list[np.ndarray]] = None,
) -> tuple[CuboidTracks, CuboidTracks]:
    """
    Classify dynamic tracks from `cuboidtracks` using configured metrics.

    If `cuboid_tracks_params.camera_visibility` is enabled, only timestamps within
    `cuboidtracks_visible_intervals` are used for the metrics (needs to be provided in this case).

    Args:
        cuboidtracks (CuboidTracks): The cuboid tracks to classify
        cuboid_tracks_params (CuboidTracksConfig): Contains configurable parameters for cuboid tracks
        unconditionally_dynamic_classes (set[str]): The set of classes that are unconditionally dynamic
        tqdm_disabled (bool): When true, no progress bars are showed.
        cuboidtracks_visible_intervals (Optional[list[np.ndarray]]): The visible intervals for the cuboid tracks, N_tracks x (N_intervals_i x 2) [int64]

    Returns:
        cuboidtracks_all: same tracks, but with DYNAMIC flags updated
        cuboidtracks_dynamic: subset containing only dynamic tracks
    """

    if cuboidtracks.n_tracks == 0:
        return cuboidtracks, CuboidTracks.Factory.empty(device=cuboidtracks.device)

    if cuboid_tracks_params.camera_visibility:
        assert cuboidtracks_visible_intervals is not None, (
            "cuboidtracks_visible_intervals must be provided when cuboid_tracks_params.camera_visibility is enabled"
        )
        assert len(cuboidtracks_visible_intervals) == cuboidtracks.n_tracks, (
            f"cuboidtracks_visible_intervals length ({len(cuboidtracks_visible_intervals)}) does not match n_tracks ({cuboidtracks.n_tracks})"
        )

    # Pull packed data to CPU for the interpolator (numpy-based)
    tracks_packinfo = cuboidtracks.tracks_packinfo.cpu().numpy()
    tracks_timestamps_us = cuboidtracks.tracks_timestamps_us.cpu().numpy()
    tracks_poses = cuboidtracks.tracks_poses.matrix().cpu().numpy()
    tracks_label_class = cuboidtracks.tracks_label_class

    new_track_flags = cuboidtracks.tracks_flags.cpu().numpy()

    for track_idx in tqdm.tqdm(range(cuboidtracks.n_tracks), desc="Dynamic Tracks [tracks]", disable=tqdm_disabled):
        start_idx, n_poses = tracks_packinfo[track_idx].tolist()

        if n_poses < 2:
            # cannot infer velocity/distance reliably from single track pose
            new_track_flags[track_idx] &= ~TrackFlags.DYNAMIC
            continue

        poses = tracks_poses[start_idx : start_idx + n_poses]
        timestamps_us = tracks_timestamps_us[start_idx : start_idx + n_poses]

        # Use pose sample timestamps for evaluation by default
        eval_timestamps_us = timestamps_us.copy()

        # Determine evaluation timestamps from each camera-visible interval instead if enabled
        if cuboid_tracks_params.camera_visibility:
            intervals_us = unpack_optional(cuboidtracks_visible_intervals)[track_idx]
            assert intervals_us.dtype == np.int64, "interval timestamps must be int64"
            assert intervals_us.shape[1] == 2, "intervals must have shape (N, 2)"

            eval_timestamps_us_list: list[np.ndarray] = []
            for interval_start_us, interval_end_us in intervals_us:
                in_any = (timestamps_us >= interval_start_us) & (timestamps_us <= interval_end_us)
                # Sorted timestamps
                eval_timestamps_us_list.append(
                    np.unique(
                        np.concatenate(
                            (
                                np.asarray([interval_start_us, interval_end_us]),
                                timestamps_us[in_any],
                            )
                        ).astype(np.int64)
                    )
                )
            eval_timestamps_us = (
                np.concatenate(eval_timestamps_us_list)
                if len(eval_timestamps_us_list) > 0
                else np.empty(0, dtype=np.int64)
            )

        # Fallback: if camera_visibility enabled but no intervals found (e.g., object behind ego),
        # fall back to all track timestamps so motion can still be evaluated from the ncore bag.
        # This is to avoid making all objects static if they are behind ego.
        if cuboid_tracks_params.camera_visibility and len(eval_timestamps_us) == 0:
            eval_timestamps_us = timestamps_us.copy()

        # Initialize dynamic classification from unconditional dynamic label classes
        track_is_dynamic: bool = tracks_label_class[track_idx] in unconditionally_dynamic_classes

        # Evaluate dynamic classification at evaluation timestamps
        if len(eval_timestamps_us) > 0:
            # Create velocity interpolator at all pose samples
            interpolator = PoseLinearVelocityInterpolator(poses, timestamps_us)

            # Check if displacement and distance thresholds or speed threshold is used and classify accordingly
            if cuboid_tracks_params.use_displacement_and_distance:
                track_displacement_m = interpolator.get_displacement_m(eval_timestamps_us)
                track_distance_m = interpolator.get_distance_m(eval_timestamps_us)
                track_is_dynamic |= track_displacement_m > cuboid_tracks_params.track_min_displacement_m
                track_is_dynamic |= track_distance_m > cuboid_tracks_params.track_min_distance_m
            else:
                track_speeds_m_s = interpolator.get_speeds_m_s(eval_timestamps_us)
                match cuboid_tracks_params.track_speed_reduction_op:
                    case "median":
                        track_is_dynamic |= float(np.median(track_speeds_m_s)) > cuboid_tracks_params.track_min_speed_ms
                    case "max":
                        track_is_dynamic |= float(np.max(track_speeds_m_s)) > cuboid_tracks_params.track_min_speed_ms

        if track_is_dynamic:
            new_track_flags[track_idx] |= TrackFlags.DYNAMIC
        else:
            new_track_flags[track_idx] &= ~TrackFlags.DYNAMIC

    cuboidtracks_all = CuboidTracks(
        tracks_data=TracksData(
            tracks_id=cuboidtracks.tracks_id[:],
            tracks_packinfo=cuboidtracks.tracks_packinfo.clone(),
            tracks_poses=lt.SE3(cuboidtracks.tracks_poses.data.clone()),
            tracks_timestamps_us=cuboidtracks.tracks_timestamps_us.clone(),
            tracks_flags=to_torch(new_track_flags, device=cuboidtracks.device, dtype=torch.int32),
            max_track_n_poses=cuboidtracks.max_track_n_poses,
            tracks_label_class=cuboidtracks.tracks_label_class[:],
        ),
        cuboidtracks_data=CuboidTracksData(
            cuboids_dims=cuboidtracks.cuboids_dims.clone(),
        ),
    )
    dynamic_mask = (cuboidtracks_all.tracks_flags & TrackFlags.DYNAMIC).bool()
    cuboidtracks_dynamic = CuboidTracks.Ops.subset_from_mask(cuboidtracks_all, dynamic_mask)

    return cuboidtracks_all, cuboidtracks_dynamic


def compute_camera_visible_intervals_cuboid_tracks(
    cuboidtracks: CuboidTracks,
    camera_sensors: dict[str, ncore.data.CameraSensorProtocol],
    camera_models: dict[str, ncore.sensors.CameraModel],
    camera_frame_ranges: dict[str, range],
    cameras_all_pixels: dict[str, np.ndarray],
    cameras_all_rays: dict[str, np.ndarray],
    cameras_frame_valid_pixels_masks: Optional[dict[str, dict[int, PackedMask]]] = None,
    cameras_frame_track_idxs: Optional[dict[str, dict[int, np.ndarray]]] = None,
    valid_pixels_cuboid_tracks_params: Optional[ValidPixelsCuboidTrackConfig] = None,
    subsample: int = 1,
    max_intersections_per_ray: int = 32,
    tqdm_disabled: bool = False,
) -> list[np.ndarray]:
    """
    Computes per-track visible time-intervals from camera rolling-shutter ray intersections:
    - A track is considered "visible" at a ray timestamp if any camera ray intersects its cuboid at that timestamp.
    - For each (camera frame, track), we record [min_hit_time_us, max_hit_time_us] over all hit rays in that frame.
    - Across frames/cameras, intervals are merged if overlapping or adjacent.

    Returns:
    - cuboidtracks_visible_intervals: list[np.ndarray] of per-track [start_us, end_us] visible intervals, N_tracks x (N_intervals_i x 2) [int64]
    """

    if cuboidtracks.n_tracks == 0:
        return []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        cuboids_dims_padding_cuda = torch.tensor(
            (
                valid_pixels_cuboid_tracks_params.track_padding_m
                if valid_pixels_cuboid_tracks_params is not None
                else [0.0, 0.0, 0.0]
            ),
            dtype=torch.float32,
            device=torch.device("cuda"),
        )

        # Precompute per-track timestamp bounds for clipping hit timestamps
        tracks_packinfo = cuboidtracks.tracks_packinfo
        tracks_timestamps_us = cuboidtracks.tracks_timestamps_us
        tracks_min_timestamp = tracks_timestamps_us[tracks_packinfo[:, 0]]
        tracks_max_timestamp = tracks_timestamps_us[tracks_packinfo[:, 0] + tracks_packinfo[:, 1] - 1]

        intervals_per_track: list[list[tuple[int, int]]] = [[] for _ in range(cuboidtracks.n_tracks)]

        for camera_id, camera_frame_range in camera_frame_ranges.items():
            camera_sensor = camera_sensors[camera_id]
            camera_model = camera_models[camera_id]

            orig_width = int(camera_model.resolution[0].item())
            orig_height = int(camera_model.resolution[1].item())
            width, height = orig_width // subsample, orig_height // subsample

            # Optional: per-frame valid pixel masks for this camera
            frame_valid_pixels_masks = (
                cameras_frame_valid_pixels_masks[camera_id] if cameras_frame_valid_pixels_masks is not None else None
            )
            frame_track_idxs = cameras_frame_track_idxs[camera_id] if cameras_frame_track_idxs is not None else None

            if subsample != 1 and frame_valid_pixels_masks is not None:
                raise ValueError(
                    "subsample > 1 is not supported when cameras_frame_valid_pixels_masks is provided "
                    "(mask update assumes full-resolution rays/pixels)."
                )

            # initialize cuda buffers
            camera_all_pixels_cuda = to_torch(
                cameras_all_pixels[camera_id]
                .reshape(orig_height, orig_width, 2)[::subsample, ::subsample]
                .reshape((-1, 2)),
                device="cuda",
            )
            camera_all_rays_cuda = to_torch(
                cameras_all_rays[camera_id][::subsample, ::subsample].reshape((-1, 3)), device="cuda"
            )
            camera_rays_relative_frame_times_cuda = camera_model.image_points_relative_frame_times(
                camera_model.pixels_to_image_points(camera_all_pixels_cuda.cpu())
            ).to("cuda")
            assert_default_device_on_local_rank()

            # process each frame in a separate cuda stream to maximize throughput of independent computations
            cuda_streams = [torch.cuda.Stream() for _ in range(16)]

            def thread_camera_frame_intervals(
                camera_frame_idx: int,
                *,
                camera_sensor=camera_sensor,
                camera_model=camera_model,
                camera_all_pixels_cuda=camera_all_pixels_cuda,
                camera_all_rays_cuda=camera_all_rays_cuda,
                camera_rays_relative_frame_times_cuda=camera_rays_relative_frame_times_cuda,
                frame_valid_pixels_masks=frame_valid_pixels_masks,
                frame_track_idxs=frame_track_idxs,
                w=width,
                h=height,
                cuda_streams=cuda_streams,
            ) -> list[tuple[int, int, int]]:
                """
                Computes a list of (track_idx, min_hit_time_us, max_hit_time_us) for one frame.
                """
                set_default_device()

                # this has been initialized before if not None
                frame_valid_pixel_mask = (
                    frame_valid_pixels_masks[camera_frame_idx].unpacked()
                    if frame_valid_pixels_masks is not None
                    else None
                )

                if frame_valid_pixel_mask is not None and frame_track_idxs is not None:
                    frame_track_idxs |= {camera_frame_idx: np.full_like(frame_valid_pixel_mask, -1, dtype=np.int32)}

                frame_start_timestamp_us = int(
                    camera_sensor.get_frame_timestamp_us(camera_frame_idx, ncore.data.FrameTimepoint.START)
                )
                frame_end_timestamp_us = int(
                    camera_sensor.get_frame_timestamp_us(camera_frame_idx, ncore.data.FrameTimepoint.END)
                )

                T_sensor_world_start = camera_sensor.get_frames_T_sensor_target(
                    "world", camera_frame_idx, ncore.data.FrameTimepoint.START
                )
                T_sensor_world_end = camera_sensor.get_frames_T_sensor_target(
                    "world", camera_frame_idx, ncore.data.FrameTimepoint.END
                )

                # run kernel and to GPU / from GPU copy in dedicated cuda streams
                with torch.cuda.stream(current_stream := cuda_streams[camera_frame_idx % len(cuda_streams)]):
                    camera_poses_cuda = (
                        se3_matrix_to_tquat(np.array([T_sensor_world_start, T_sensor_world_end], dtype=np.float32))
                        .pin_memory()
                        .to(device="cuda", dtype=torch.float32, non_blocking=True)
                    )

                    ray_rolling_shutter_intersection_result = cuboidtracks.ray_rolling_shutter_intersection(
                        pixel_idxs=camera_all_pixels_cuda,
                        camera_rays=camera_all_rays_cuda,
                        camera_poses=camera_poses_cuda,
                        camera_timestamp_start_us=frame_start_timestamp_us,
                        camera_timestamp_end_us=frame_end_timestamp_us,
                        w=w,
                        h=h,
                        shutter_type=camera_model.shutter_type.value,
                        cuboids_dims_padding=cuboids_dims_padding_cuda,
                        max_intersections_per_ray=max_intersections_per_ray,
                        with_intersections_ts=False,
                    )

                    hits_cnt = ray_rolling_shutter_intersection_result.intersections_cnt
                    intersections_tracks_idx = ray_rolling_shutter_intersection_result.intersections_tracks_idx

                    intersections_tracks_idx_np = intersections_tracks_idx.to("cpu", non_blocking=True).numpy()

                    hitmask_np = (
                        (hits_cnt.reshape(frame_valid_pixel_mask.shape) > 0).to("cpu", non_blocking=True).numpy()
                        if frame_valid_pixel_mask is not None
                        else None
                    )

                    # Per-ray timestamp
                    rays_timestamps_us = frame_start_timestamp_us + (
                        camera_rays_relative_frame_times_cuda * float(frame_end_timestamp_us - frame_start_timestamp_us)
                    ).to(torch.int64)

                    # Flatten (ray, slot) hits
                    mask = intersections_tracks_idx >= 0
                    hit_idx_cpu: torch.Tensor = torch.empty((0,), dtype=torch.int32, device="cpu")
                    min_timestamps_cpu: torch.Tensor = torch.empty((0,), dtype=torch.int64, device="cpu")
                    max_timestamps_cpu: torch.Tensor = torch.empty((0,), dtype=torch.int64, device="cpu")
                    if mask.any():
                        track_idx_hits = intersections_tracks_idx[mask].to(torch.int64)
                        timestamps_hits = rays_timestamps_us[:, None].expand(-1, intersections_tracks_idx.shape[1])[
                            mask
                        ]

                        # Reduce min/max timestamp per track
                        inf = torch.iinfo(torch.int64).max
                        min_timestamps = torch.full((cuboidtracks.n_tracks,), inf, dtype=torch.int64, device="cuda")
                        max_timestamps = torch.full((cuboidtracks.n_tracks,), -1, dtype=torch.int64, device="cuda")

                        min_timestamps.scatter_reduce_(
                            0, track_idx_hits, timestamps_hits, reduce="amin", include_self=True
                        )
                        max_timestamps.scatter_reduce_(
                            0, track_idx_hits, timestamps_hits, reduce="amax", include_self=True
                        )

                        hit_tracks = (min_timestamps != inf) & (max_timestamps >= 0)
                        if hit_tracks.any():
                            hit_idx_cpu = (
                                torch.nonzero(hit_tracks, as_tuple=False).squeeze(1).to("cpu", non_blocking=True)
                            )
                            # Clip hit timestamps to within each track's own timestamp span.
                            hit_tracks_min_timestamp = tracks_min_timestamp[hit_tracks]
                            hit_tracks_max_timestamp = tracks_max_timestamp[hit_tracks]
                            min_timestamps_cpu = (
                                min_timestamps[hit_tracks]
                                .clamp(min=hit_tracks_min_timestamp, max=hit_tracks_max_timestamp)
                                .to("cpu", non_blocking=True)
                            )
                            max_timestamps_cpu = (
                                max_timestamps[hit_tracks]
                                .clamp(min=hit_tracks_min_timestamp, max=hit_tracks_max_timestamp)
                                .to("cpu", non_blocking=True)
                            )
                    current_stream.synchronize()

                    # Update optional masks
                    if (
                        hitmask_np is not None
                        and frame_valid_pixel_mask is not None
                        and frame_valid_pixels_masks is not None
                    ):
                        frame_valid_pixel_mask[hitmask_np] = False
                        frame_valid_pixels_masks[camera_frame_idx] = PackedMask(frame_valid_pixel_mask)

                        if frame_track_idxs is not None:
                            frame_track_idxs[camera_frame_idx][hitmask_np] = intersections_tracks_idx_np[
                                hitmask_np.flatten()
                            ][:, 0]

                    return [
                        (int(track_idx), int(min_timestamp), int(max_timestamp))
                        for track_idx, min_timestamp, max_timestamp in zip(
                            hit_idx_cpu, min_timestamps_cpu, max_timestamps_cpu
                        )
                    ]

            futures = [
                executor.submit(thread_camera_frame_intervals, camera_frame_idx)
                for camera_frame_idx in camera_frame_range
            ]
            for fut in tqdm.tqdm(
                concurrent.futures.as_completed(futures),
                desc=f"Tracks Visibility [{camera_id}->frames]",
                total=len(camera_frame_range),
                disable=tqdm_disabled,
            ):
                for track_idx, start_us, end_us in fut.result():
                    intervals_per_track[track_idx].append((start_us, end_us))

    def _merge_intervals_us(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Merge overlapping/adjacent [start_us, end_us] intervals."""
        if not intervals:
            return []
        intervals_sorted = sorted(intervals, key=lambda x: (x[0], x[1]))
        merged: list[list[int]] = [[intervals_sorted[0][0], intervals_sorted[0][1]]]
        for start_timestamp_us, end_timestamp_us in intervals_sorted[1:]:
            if start_timestamp_us <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], end_timestamp_us)
            else:
                merged.append([start_timestamp_us, end_timestamp_us])
        return [(start_timestamp_us, end_timestamp_us) for start_timestamp_us, end_timestamp_us in merged]

    # Merge intervals per track
    merged_per_track: list[list[tuple[int, int]]] = [_merge_intervals_us(iv) for iv in intervals_per_track]
    return [
        (np.asarray(iv, dtype=np.int64).reshape(-1, 2) if len(iv) else np.empty((0, 2), dtype=np.int64))
        for iv in merged_per_track
    ]


def compute_valid_lidarpoints_trafficlight_cameravisible(
    aux_loader: ncore_utils.AuxShardDataLoader,
    lidar_sensors: dict[str, ncore.data.LidarSensorProtocol],
    all_camera_ids: list[str],
    time_range_us: HalfClosedInterval,
    sensor_trafficlight_class_ids: dict[str, int],
    lidars_frame_valid_points_masks: dict[str, dict[int, PackedMask]],
    tqdm_disabled: bool,
) -> dict[str, dict[int, PackedMask]]:
    """
    Initializes or extends valid lidar point masks based on lidar segmentation of traffic light points and visibility in active cameras

    Args:
        aux_loader (AuxShardDataLoader): Used for the semantic segmentation and lidar-camera visibility
        lidar_sensors (dict[str, LidarSensorProtocol]): Used for the frame timestamps
        all_camera_ids (list[str]): A list of camera ids to use for the lidar-camera visibility masks
        time_range_us (HalfClosedInterval): Specifies the range of lidar frames with timestamps
        valid_pixels_traffic_light_params (ValidPixelsTrafficLightConfig): Specifies the kernel size for dilation
        sensor_trafficlight_class_ids (dict[str, int]): Class IDs to compare to the semantic segmentation.
        lidars_frame_valid_points_masks (dict[str, dict[int, PackedMask]]): The valid point masks. This input will be updated in place.
        tqdm_disabled (bool): If true, will not show progress bars

    Returns:
        dict[str, dict[int, PackedMask]]. Updated lidars_frame_valid_points_masks
    """

    for lidar_id in lidar_sensors.keys():
        lidar_frame_valid_points_masks = lidars_frame_valid_points_masks[lidar_id]

        lidar_sensor = lidar_sensors[lidar_id]

        for lidar_frame_idx in tqdm.tqdm(
            time_range_us.cover_range(lidar_sensor.get_frames_timestamps_us()),
            desc="Traffic Light Masks [lidars->frames]",
            disable=tqdm_disabled,
        ):
            lidar_timestamp = lidar_sensor.get_frame_timestamp_us(lidar_frame_idx)

            # load semantic from annotation shard
            semantic_segmentation = aux_loader.get_lidar_semantic_segmentation(lidar_id, lidar_timestamp)
            traffic_light_mask = semantic_segmentation == sensor_trafficlight_class_ids[lidar_id]

            # determine combined camera visibility mask
            visibility_mask = np.full(len(semantic_segmentation), False)
            load_visibility_mask_dict = aux_loader.get_lidar_camera_visibility(
                lidar_id, lidar_timestamp, camera_ids=all_camera_ids
            )
            for camera_id in all_camera_ids:
                assert camera_id in load_visibility_mask_dict, f"no lidar-camera-visibility data loaded for {camera_id}"
                visibility_mask = np.logical_or(load_visibility_mask_dict[camera_id], visibility_mask)

            # mark camera observed lidar points *not* on traffic-lights as *valid*
            camera_visible_non_traffic_light_points_mask = np.logical_not(
                np.logical_or(traffic_light_mask, np.logical_not(visibility_mask))
            )

            if (frame_valid_points_mask := lidar_frame_valid_points_masks.get(lidar_frame_idx)) is not None:
                # extend existing mask
                lidar_frame_valid_points_masks[lidar_frame_idx] = PackedMask(
                    frame_valid_points_mask & camera_visible_non_traffic_light_points_mask
                )
            else:
                # initialize new mask
                lidar_frame_valid_points_masks[lidar_frame_idx] = PackedMask(
                    camera_visible_non_traffic_light_points_mask
                )
    return lidars_frame_valid_points_masks


def compute_valid_pixels_trafficlight(
    aux_loader: ncore_utils.AuxShardDataLoader,
    camera_sensors: dict[str, ncore.data.CameraSensorProtocol],
    camera_frame_ranges: dict[str, range],
    valid_pixels_traffic_light_params: ValidPixelsTrafficLightConfig,
    sensor_trafficlight_class_ids: dict[str, int],
    cameras_frame_valid_pixels_masks: dict[str, dict[int, PackedMask]],
    tqdm_disabled: bool,
) -> dict[str, dict[int, PackedMask]]:
    """
    Initializes or extends valid pixel masks using semantic segmentation based traffic light masks

    Args:
        aux_loader (AuxShardDataLoader): Used for the semantic segmentation
        camera_sensors (dict[str, CameraSensorProtocol]): Used for frame timestamps
        camera_frame_ranges (dict[str, range]): Specifies frame range to apply the update
        valid_pixels_traffic_light_params (ValidPixelsTrafficLightConfig): Specifies the kernel size for dilation
        sensor_trafficlight_class_ids (dict[str, int]): Class IDs to compare to the semantic segmentation.
        cameras_frame_valid_pixels_masks (dict[str, dict[int, PackedMask]]): The valid pixel masks. This will be updated in place.
        tqdm_disabled (bool): If true, will not show progress bars

    Returns:
        dict[str, dict[int, PackedMask]]. Updated cameras_frame_valid_pixels_masks with the dynamic pixel masks
    """

    kernel_size = valid_pixels_traffic_light_params.seg_dilate_radius
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # camera mask
    for camera_id, camera_frame_range in tqdm.tqdm(
        camera_frame_ranges.items(), desc="Traffic Light Masks [cameras]", disable=tqdm_disabled
    ):
        camera_sensor = camera_sensors[camera_id]

        frame_valid_pixels_masks = cameras_frame_valid_pixels_masks[camera_id]

        for camera_frame_idx in tqdm.tqdm(
            camera_frame_range, desc="Traffic Light Masks [cameras-frames]", disable=tqdm_disabled
        ):
            # this has been initialized before
            frame_valid_pixel_mask = frame_valid_pixels_masks[camera_frame_idx].unpacked()

            # load semantic from annotation shard
            semantic_segmentation = np.asarray(
                aux_loader.get_semantic_segmentation(camera_id, camera_sensor.get_frame_timestamp_us(camera_frame_idx))
            )

            traffic_light_pixels_mask = semantic_segmentation == sensor_trafficlight_class_ids[camera_id]

            # mask dilation, cv2
            light_mask = (traffic_light_pixels_mask * 255).astype(np.uint8)
            light_mask = cv2.dilate(light_mask, kernel, iterations=1)
            traffic_light_pixels_mask = light_mask == 255

            # set traffic light pixels as invalid
            frame_valid_pixel_mask[traffic_light_pixels_mask] = False

            frame_valid_pixels_masks[camera_frame_idx] = PackedMask(frame_valid_pixel_mask)
    return cameras_frame_valid_pixels_masks


def compute_valid_pixels_sceneflow(
    aux_loader: ncore_utils.AuxShardDataLoader,
    camera_sensors: dict[str, ncore.data.CameraSensorProtocol],
    camera_models: dict[str, ncore.sensors.CameraModel],
    camera_frame_ranges: dict[str, range],
    valid_pixels_scene_flow_config: ValidPixelsSceneFlowConfig,
    cameras_frame_valid_pixels_masks: dict[str, dict[int, PackedMask]],
) -> dict[str, dict[int, PackedMask]]:
    """
    Initializes or extends valid pixel masks using scene flow based dynamic masks

    Args:
        aux_loader (AuxShardDataLoader): Used for the scene flow magnitude
        camera_sensors (dict[str, CameraSensorProtocol]): Used for frame timestamps
        camera_models (dict[str, CameraModel]): Used for image resolution
        camera_frame_ranges (dict[str, range]): Specifies frame range to apply the update
        valid_pixels_scene_flow_config (ValidPixelsSceneFlowConfig): Contains config params which effect the dynamic mask
        cameras_frame_valid_pixels_masks (dict[str, dict[int, PackedMask]]): The valid pixel masks. This will be updated in place.

    Returns:
        dict[str, dict[int, PackedMask]]. Updated cameras_frame_valid_pixels_masks with the dynamic pixel masks
    """
    dilator = MorphOp(
        c_out=1,
        type_str="dilation2d",
        device=torch.device("cuda"),
        kernel_size=valid_pixels_scene_flow_config.flow_dilate_radius,
        use_soft_max=False,
    )

    downsample_scale = (
        valid_pixels_scene_flow_config.flow_downsample_scale
    )  # downsample the mask before dilating for memory saving

    for camera_id, camera_frame_range in camera_frame_ranges.items():
        camera_sensor = camera_sensors[camera_id]
        camera_model = camera_models[camera_id]

        w, h = camera_model.resolution.cpu().numpy()[:]

        resizer1 = torchvision.transforms.Resize(
            size=(w // downsample_scale, w // downsample_scale),
            interpolation=torchvision.transforms.InterpolationMode.NEAREST,
        )  # for speeding up
        resizer2 = torchvision.transforms.Resize(
            size=(w, w), interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )

        frame_valid_pixels_masks = cameras_frame_valid_pixels_masks[camera_id]

        for camera_frame_idx in camera_frame_range:
            # this has been initialized before
            frame_valid_pixel_mask = frame_valid_pixels_masks[camera_frame_idx].unpacked()

            # load scene flow magnitude from annotation shard
            scene_flow_magnitude = np.asarray(
                aux_loader.get_scene_flow_magnitude(
                    camera_id, int(camera_sensor.get_frame_timestamp_us(camera_frame_idx))
                )
            )

            dynamic_pixels_mask = np.greater(scene_flow_magnitude, valid_pixels_scene_flow_config.flow_min_speed_ms)

            # mask dilation
            tensor_to_dilate = to_torch(dynamic_pixels_mask, device="cuda").unsqueeze(0).unsqueeze(0).float()
            tensor_to_dilate_sq = torch.cat([tensor_to_dilate, torch.zeros_like(tensor_to_dilate)], 2)[
                :, :, :w, :w
            ]  # pad it to a square image

            tensor_dilated = resizer2(dilator(resizer1(tensor_to_dilate_sq)))[
                :, :, :h, :w
            ]  # dilating and remove the padded region

            dynamic_pixels_mask = (tensor_dilated.squeeze() > 0.5).cpu().detach().numpy()  # dilated mask

            # set dynamic pixels as invalid
            frame_valid_pixel_mask[dynamic_pixels_mask] = False

            frame_valid_pixels_masks[camera_frame_idx] = PackedMask(frame_valid_pixel_mask)
    return cameras_frame_valid_pixels_masks


def transform_cuboid_track_observations(
    observations: List[ncore.data.CuboidTrackObservation],
    pose_graph: ncore_transformations.PoseGraphInterpolator,
    target_frame_id: str,
    target_frame_timestamp_us: int,
    tqdm_disabled: bool,
) -> List[ncore.data.CuboidTrackObservation]:
    """Transforms label observation to a new reference frame"""

    ret = []

    for observation in tqdm.tqdm(
        observations,
        desc=f"Transform cuboid track observations",
        disable=tqdm_disabled,
    ):
        ret.append(
            observation.transform(
                target_frame_id=target_frame_id,
                target_frame_timestamp_us=target_frame_timestamp_us,
                pose_graph=pose_graph,
            )
        )

    return ret


def compute_points_outside_tracks(
    points: npt.NDArray | torch.Tensor,
    bbox_labels: List[ncore_internal.data.v3.FrameLabel3]
    | List[ncore.data.BBox3]
    | List[ncore.data.CuboidTrackObservation],
    track_padding_m: list[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Determine which points lie outside all provided track-aligned cuboid volumes.

    Args:
        points: (N, 3) array or tensor of point positions in the same world frame
            as the cuboid poses. If a NumPy array is provided it will be moved to CUDA.
        bbox_labels: List of cuboid labels convertable to cuboid poses and dimensions
            (`bbox3`) for the points to test against.
        track_padding_m: Length-3 list [dx, dy, dz] of padding (meters) added to
            the cuboid dimensions before testing containment.

    Returns:
        tuple:
            - outside_point_mask: (N,) boolean tensor, True where the point is
              outside all provided (padded) track cuboids.
            - cuboid_transforms: (T, 4, 4) float32 tensor of SE(3) transforms,
              mapping track-local coordinates to world coordinates for each track.
            - track_dims_padded_m: (T, 3) float32 tensor of padded cuboid
              dimensions [length, width, height] used for testing.
    """

    # collect all posed bboxes
    bboxes: List[ncore.data.BBox3] = []
    for bbox_label in bbox_labels:
        match bbox_label:
            case ncore_internal.data.v3.FrameLabel3() | ncore.data.CuboidTrackObservation():
                bboxes.append(bbox_label.bbox3)
            case ncore.data.BBox3():
                bboxes.append(bbox_label)
            case _:
                raise TypeError(f"compute_points_outside_tracks: unsupported bbox_label type {type(bbox_label)}")

    cuboid_transforms = torch.from_numpy(
        np.stack([ncore_transformations.bbox_pose(bbox.to_array()) for bbox in bboxes])
    ).to(dtype=torch.float32, device="cuda")

    track_dim = torch.stack([torch.tensor(bbox.dim) + torch.tensor(track_padding_m) for bbox in bboxes]).to(
        dtype=torch.float32, device="cuda"
    )

    # Create the track AABBs centered at [0,0,0]
    track_abbs = AABB3D.from_center_extent(center=torch.zeros_like(track_dim), extent=track_dim)

    if isinstance(points, np.ndarray):
        points = to_torch(points, device="cuda")

    # TODO: extend ncore_transformations.se3_inverse to support batched inputs and torch
    point_in_track_coords = ncore_transformations.transform_point_cloud(
        points.unsqueeze(0),
        se3_matrix_to_se3(cuboid_transforms, unbatch=False).inv().matrix(),
    )

    # Check if point is in any of the tracks
    outside_point_mask = ~track_abbs.points_within_aabbs(point_in_track_coords, in_all=False)

    return outside_point_mask, cuboid_transforms, track_dim


def compute_point_cloud_inside_tracks_mask(
    point_cloud: PointCloud,
    time_range_us: HalfClosedInterval,
    cuboidtracks_dynamic: CuboidTracks,
    track_padding_m: list[float],
    batch_size: int,
    tqdm_disabled: bool,
    restrict_to_class_ids: Optional[list[int]] = None,
) -> torch.Tensor:
    """
    Compute a boolean mask for points that are INSIDE dynamic tracks over a time range.

    Args:
        point_cloud: Point cloud in the nre frame; uses `xyz_end` (N, 3) and `n_points`.
        time_range_us: Half-closed interval of timestamps to consider (microseconds).
        cuboidtracks_dynamic: Dynamic tracks in the same reference frame as the point cloud.
        track_padding_m: Length-3 list [dx, dy, dz] of padding (meters) to expand
            each track cuboid before testing containment.
        batch_size: Batch size for the cuboids used to filter the point cloud.
        tqdm_disabled: If True, disables progress bars.
        restrict_to_class_ids: If provided, only points with semantic_class_id in this
            list are checked for track intersection; others always return False. Typically
            set to dynamic classes (person, car, bicycle) to skip static infrastructure
            (building, road, pole). If None, all points are checked.

    Returns:
        (N,) boolean tensor where True indicates a point is INSIDE a dynamic track cuboid.
    """

    device = torch.device("cuda")
    if point_cloud.xyz_end.device != device:
        point_cloud = point_cloud.to(device=device)

    inside_tracks_mask = torch.full((point_cloud.n_points,), False, device=device, dtype=torch.bool)
    batch_size = max(1, int(batch_size))

    check_indices = torch.arange(point_cloud.n_points, device=device)
    points_xyz = point_cloud.xyz_end

    if restrict_to_class_ids is not None and point_cloud.semantic_class_id is not None:
        restrict_to_class_ids_tensor = torch.tensor(
            restrict_to_class_ids, dtype=point_cloud.semantic_class_id.dtype, device=device
        )
        restrict_mask = torch.isin(point_cloud.semantic_class_id, restrict_to_class_ids_tensor)
        check_indices = torch.where(restrict_mask)[0]
        points_xyz = point_cloud.xyz_end[check_indices]

    labels = []
    for i, (start_idx, n_poses) in enumerate(cuboidtracks_dynamic.tracks_packinfo):
        track_timestamps_us = cuboidtracks_dynamic.tracks_timestamps_us[start_idx : start_idx + n_poses]
        object_dim = cuboidtracks_dynamic.cuboids_dims[i]
        in_time_range_idxs = torch.logical_and(
            track_timestamps_us >= time_range_us.start,
            track_timestamps_us < time_range_us.end,
        )
        track_poses_in_time_range = cuboidtracks_dynamic.tracks_poses[start_idx : start_idx + n_poses][
            in_time_range_idxs
        ].vec()
        labels.extend(
            [
                ncore.data.BBox3(
                    centroid=tuple(track_pose[:3].cpu().numpy().astype(float)),
                    dim=tuple(object_dim.cpu().numpy().astype(float)),
                    rot=tuple(quat_to_euler(track_pose[3:]).cpu().numpy().astype(float)),
                )
                for track_pose in track_poses_in_time_range
            ]
        )

    for start_idx in tqdm.tqdm(
        range(0, len(labels), batch_size),
        desc=f"Inside Track Points [label-batches]",
        disable=tqdm_disabled,
    ):
        outside_mask = compute_points_outside_tracks(
            points_xyz, labels[start_idx : start_idx + batch_size], track_padding_m
        )[0]
        inside_tracks_mask[check_indices] = torch.logical_or(
            inside_tracks_mask[check_indices],
            ~outside_mask,
        )

    return inside_tracks_mask
