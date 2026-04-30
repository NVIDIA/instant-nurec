# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import contextlib
import functools
import gc
import io
import logging
import os
import tempfile
import threading
import time
import traceback

from concurrent import futures
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple
from urllib.parse import urlparse

import click
import grpc
import numpy as np
import nvidia.nvimgcodec as nvimgcodec
import torch

from einops import rearrange
from grpc_health.v1 import health, health_pb2, health_pb2_grpc  # type: ignore
from omegaconf import OmegaConf
from PIL import Image
from scipy.spatial.transform import Rotation as R

import ncore.data
import ncore.impl.common.transformations as ncore_transformations
import nre.grpc.protos.common_pb2 as grpc_types
import nre.grpc.protos.sensorsim_pb2 as sensorsim_types
import nre.utils.cli as cli
import nre.utils.profiling as profile

from nre.artifact import Artifact
from nre.config.model import RendererBackend
from nre.config.version import get_version
from nre.datasets.summary import DataSourceSummary
from nre.datasets.tracks import CuboidTracks, TrackFlags
from nre.difix.model import DifixModelFactory
from nre.grpc.downloader import Downloader, check_safe_scene_id
from nre.grpc.ego_hood import EgocarRigBank
from nre.grpc.grpc_server_config import GrpcServerConfig, grpc_cli_options
from nre.grpc.protos.sensorsim_pb2 import (
    AvailableCamerasRequest,
    AvailableCamerasReturn,
    AvailableDynamicObjectsRequest,
    AvailableDynamicObjectsReturn,
    AvailableEgoMasksReturn,
    AvailableTrajectoriesRequest,
    AvailableTrajectoriesReturn,
    BatchRGBRenderRequest,
    BatchRGBRenderRequestItem,
    BatchRGBRenderReturn,
    BatchRGBRenderReturnItem,
    BivariateWindshieldModelParameters,
    CameraSpec,
    EditAssetsRequest,
    EditAssetsResponse,
    ExternalAssetObjectsRequest,
    ExternalAssetObjectsReturn,
    FthetaCameraParam,
    ImageFormat,
    LidarRenderFilter,
    LidarRenderRequest,
    LidarRenderReturn,
    OpenCVFisheyeCameraParam,
    OpenCVPinholeCameraParam,
    RestoreModelParametersRequest,
    RGBRenderRequest,
    RGBRenderReturn,
)
from nre.grpc.protos.sensorsim_pb2_grpc import SensorsimServiceServicer, add_SensorsimServiceServicer_to_server
from nre.grpc.scene_cache import SceneCache
from nre.grpc.trajectory import QVec, Trajectory
from nre.models.gaussians.utils import Asset
from nre.render import (
    ActorsSnapshot,
    LidarRayBundle,
    PoseRange,
    RayBundle,
    RenderableModel,
)
from nre.render.actors import ActorTracks
from nre.utils.cli import SettingsCollector
from nre.utils.geometry import tquat_to_se3_matrix
from nre.utils.lidar_model import LidarModelBundle
from nre.utils.metrics import MetricsCollector, create_metric_sample
from nre.utils.misc import unpack_optional
from nre.utils.profiling import ScopedTimer
from nre.utils.types import (
    FrameConversion,
    HalfClosedInterval,
    RigTrajectories,
)


log = logging.getLogger("nre.grpc.serve")


def grpc_pose_to_tquat(pose: grpc_types.Pose) -> torch.Tensor:
    """
    Converts a single gRPC Pose into a 7d tquat (translation + quaternion) defined as [x,y,z,q_x,q_y,q_z,q_w]

    Args:
        single gRPC Pose [grpc_types.Pose]

    Returns:
        single 7d tquat List[float]
    """
    return torch.tensor([pose.vec.x, pose.vec.y, pose.vec.z, pose.quat.x, pose.quat.y, pose.quat.z, pose.quat.w])


def tquat_to_grpc_pose(tquat: np.ndarray | torch.Tensor) -> grpc_types.Pose:
    """
    Converts a single 7d tquat consisting of format [x,y,z,q_x,q_y,q_z,q_w] into a gRPC Pose

    Args:
        single tquat [np.ndarray | torch.Tensor]

    Returns:
        single gRPC Pose [grpc_types.Pose]
    """
    assert tquat.shape == (7,)
    return grpc_types.Pose(
        vec=grpc_types.Vec3(
            x=tquat[0].item(),
            y=tquat[1].item(),
            z=tquat[2].item(),
        ),
        quat=grpc_types.Quat(
            x=tquat[3].item(),
            y=tquat[4].item(),
            z=tquat[5].item(),
            w=tquat[6].item(),
        ),
    )


def grpc_pose_to_torch_se3(grpc_pose: grpc_types.Pose, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Converts a single gRPC Pose into a SE3 4x4 torch Tensor

    Args:
        single gRPC Pose [grpc_types.Pose]

    Returns:
        single torch Tensor [torch.Tensor]
    """
    return tquat_to_se3_matrix(torch.tensor(grpc_pose_to_tquat(grpc_pose), dtype=dtype))


def se3_to_grpc_pose(se3: np.ndarray) -> grpc_types.Pose:
    """
    Converts a single SE3 4x4 matrix into a gRPC Pose

    Args:
        single SE3 matrix [np.ndarray]

    Returns:
        single gRPC Pose [grpc_types.Pose]
    """
    quat = R.from_matrix(se3[..., :3, :3]).as_quat(canonical=False)
    vec3 = se3[..., :3, 3]

    return grpc_types.Pose(
        vec=grpc_types.Vec3(
            x=vec3[0],
            y=vec3[1],
            z=vec3[2],
        ),
        quat=grpc_types.Quat(
            x=quat[0],
            y=quat[1],
            z=quat[2],
            w=quat[3],
        ),
    )


def actors_snapshot_from_render_request(request: RGBRenderRequest | LidarRenderRequest) -> ActorsSnapshot:
    track_ids = []
    track_poses = []

    for dynamic_object in request.dynamic_objects:
        track_ids.append(DataSourceSummary._clean_track_id_str(dynamic_object.track_id))

        track_poses.append(
            torch.stack(
                [
                    grpc_pose_to_tquat(dynamic_object.pose_pair.start_pose),
                    grpc_pose_to_tquat(dynamic_object.pose_pair.end_pose),
                ]
            )
        )

    if len(track_ids) == 0:
        return ActorsSnapshot.empty()

    return ActorsSnapshot(actor_ids=track_ids, actor_poses=torch.stack(track_poses, dim=0))


def actor_tracks_from_grpc(grpc_tracks: List[sensorsim_types.DynamicObjectTrack]) -> ActorTracks:
    """Convert gRPC DynamicObjectTrack list to ActorTracks."""
    if not grpc_tracks:
        return ActorTracks()

    cuboid_tracks_list = []
    interpolator_list = []

    for grpc_track in grpc_tracks:
        trajectory = Trajectory.from_grpc(grpc_track.trajectory)
        poses_se3 = trajectory.poses.as_se3()  # [N_poses, 4, 4] np.ndarray
        timestamps_int64 = trajectory.timestamps_us.astype(np.int64)
        object_dims = np.array(
            [grpc_track.object_size.size_x, grpc_track.object_size.size_y, grpc_track.object_size.size_z],
            dtype=np.float32,
        )

        cuboid_track = CuboidTracks.Factory.from_numpy(
            tracks_id=[grpc_track.id],
            tracks_poses=[poses_se3.astype(np.float32)],
            tracks_timestamps_us=[timestamps_int64],
            tracks_label_class=[grpc_track.semantic_class],
            tracks_flags=[TrackFlags.DYNAMIC | TrackFlags.CONTROLLABLE],
            cuboids_dims=[object_dims],
        )

        interpolator = ncore_transformations.PoseInterpolator(
            poses=cuboid_track.tracks_poses.matrix().cpu(), timestamps=cuboid_track.tracks_timestamps_us.cpu()
        )

        cuboid_tracks_list.append(cuboid_track)
        interpolator_list.append(interpolator)

    return ActorTracks(_cuboid_tracks_list=cuboid_tracks_list, _interpolator_list=interpolator_list)


def actor_tracks_to_grpc(
    actor_tracks: ActorTracks, track_to_asset: Optional[Dict[str, str]] = None
) -> List[sensorsim_types.DynamicObjectTrack]:
    """Convert ActorTracks to gRPC DynamicObjectTrack format.

    Args:
        actor_tracks: ActorTracks to convert
        track_to_asset: Optional mapping from track_id to asset_id.
                        The track_id is derived from the actor_tracks, the asset_id is derived from the assets in the AssetBank.
                        Use get_external_asset_objects() to get the list of available asset_ids.
    """
    grpc_objects = []

    for cuboid_track in actor_tracks._cuboid_tracks_list:
        (track_id,) = cuboid_track.tracks_id  # Validates exactly one track
        (track_label_class,) = cuboid_track.tracks_label_class  # Validates exactly one class
        poses_se3 = cuboid_track.tracks_poses
        timestamps_us = cuboid_track.tracks_timestamps_us

        trajectory = Trajectory(
            timestamps_us=timestamps_us.cpu().numpy().astype(np.uint64),
            poses=QVec.from_se3(poses_se3.matrix().cpu().numpy()),
        )

        if len(cuboid_track.cuboids_dims) == 0:
            raise ValueError(f"No cuboid dimensions found for track {track_id}")
        track_dim = cuboid_track.cuboids_dims[0].cpu().numpy().astype(np.float32)  # [3] array

        # Decouple track identity from asset selection
        asset_id = track_to_asset[track_id] if track_to_asset else "none"

        grpc_object = sensorsim_types.DynamicObjectTrack(
            id=track_id,
            semantic_class=track_label_class,
            trajectory=trajectory.to_grpc(),
            object_size=grpc_types.AABB(size_x=track_dim[0], size_y=track_dim[1], size_z=track_dim[2]),
            asset_id=asset_id,
        )
        grpc_objects.append(grpc_object)

    return grpc_objects


@dataclass
class TorchCamera:
    camera_model_parameters: ncore.data.ConcreteCameraModelParametersUnion
    logical_camera_id: str
    unique_sensor_idx: int
    time_range_us: HalfClosedInterval
    world_to_nre: FrameConversion
    T_camera_rig: torch.Tensor

    def __repr__(self):
        return (
            f"TorchCamera(logical_camera_id={self.logical_camera_id}, "
            f"camera_model_parameters={self.camera_model_parameters}, "
            f"time_range_us={self.time_range_us})"
        )

    @ScopedTimer()
    def build_ray_bundle(self, request: RGBRenderRequest) -> RayBundle:
        start_timestamp_us = request.frame_start_us
        end_timestamp_us = request.frame_end_us
        resolution_hw = (request.resolution_h, request.resolution_w)
        start_pose = grpc_pose_to_tquat(request.sensor_pose.start_pose)
        end_pose = grpc_pose_to_tquat(request.sensor_pose.end_pose)

        request_interval = HalfClosedInterval(start_timestamp_us, end_timestamp_us)

        if request_interval not in self.time_range_us:
            raise TimestampOutOfRangeError(
                f"Requested time range {request_interval} is outside of scene time range {self.time_range_us}."
            )

        original_w, original_h = self.camera_model_parameters.resolution.tolist()
        aspect_ratio = original_w / original_h

        req_resolution_h, req_resolution_w = resolution_hw
        if (req_resolution_w / original_w) > (req_resolution_h / original_h):
            new_w = req_resolution_w
            new_h = int(new_w / aspect_ratio)
            image_domain_scale = new_w / original_w
        else:
            new_h = req_resolution_h
            new_w = int(new_h * aspect_ratio)
            image_domain_scale = new_h / original_h

        with ScopedTimer("camera_model_parameters.transform"):
            camera_params = self.camera_model_parameters.transform(
                image_domain_scale=image_domain_scale,
                new_resolution=(new_w, new_h),
            )

        # TODO: remove this half-closed-interval convention from gRPC API because it is rather unintuitive to add 1 us
        # to the exact timestamp of the end of frame.
        if end_timestamp_us <= start_timestamp_us:
            raise TimestampOutOfRangeError(
                f"Render time range [{start_timestamp_us=}, {end_timestamp_us=}) is empty. "
                "To render at an instant t, use [t, t+1) as the interval."
            )

        return RayBundle.build(
            camera_model_parameters=camera_params,
            camera_to_world=PoseRange(start_pose, end_pose, start_timestamp_us, end_timestamp_us),
            world_to_nre=self.world_to_nre,
            unique_sensor_idx=self.unique_sensor_idx,
        )


def _rig_trajectory_to_grpc_trajectory(trajectory: RigTrajectories.RigTrajectory) -> grpc_types.Trajectory:
    poses = QVec.from_se3(trajectory.T_rig_worlds.numpy())
    timestamps_us = trajectory.T_rig_world_timestamps_us.numpy().astype(np.uint64)

    return Trajectory(
        timestamps_us=timestamps_us,
        poses=poses,
    ).to_grpc()


class CacheFullError(Exception):
    """Raised when the cache is full and cannot accept new backends."""

    pass


class NoSpareBackendsError(Exception):
    """Raised when OOM occurs and no spare backends are available to evict."""

    pass


class TimestampOutOfRangeError(ValueError):
    """Raised when a render request's time range is outside the scene range."""

    pass


class ModelIncompatibilityError(RuntimeError):
    """Raised when a model/artifact is incompatible with the server."""

    pass


def _render_exception_message(exc: Exception) -> str:
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    return str(exc)


def _classify_render_exception(exc: Exception) -> tuple[grpc.StatusCode, str]:
    if isinstance(exc, TimestampOutOfRangeError):
        return grpc.StatusCode.OUT_OF_RANGE, _render_exception_message(exc)
    if isinstance(exc, ModelIncompatibilityError):
        return grpc.StatusCode.FAILED_PRECONDITION, _render_exception_message(exc)
    if isinstance(exc, (CacheFullError, NoSpareBackendsError, torch.OutOfMemoryError, torch.cuda.OutOfMemoryError)):
        message = _render_exception_message(exc)
        if isinstance(exc, (torch.OutOfMemoryError, torch.cuda.OutOfMemoryError)):
            extra = "This can occur when running an unconverged checkpoint, or when volumetric effects are present (e.g fog)."
            message = f"{message}\n{extra}" if message else extra
        return grpc.StatusCode.RESOURCE_EXHAUSTED, message
    if isinstance(exc, KeyError):
        return grpc.StatusCode.NOT_FOUND, _render_exception_message(exc)
    if isinstance(exc, (ValueError, TypeError)):
        return grpc.StatusCode.INVALID_ARGUMENT, _render_exception_message(exc)
    return grpc.StatusCode.UNKNOWN, _render_exception_message(exc)


class BackendCache:
    """
    Count-based LRU cache with automatic OOM handling for GPU model backends.

    The cache maintains a fixed maximum number of backends (configurable via maxsize). When the limit
    is reached, spare (unused) backends are evicted in LRU order. If an OOM error occurs during model
    loading, spare backends are automatically evicted and the load is retried until successful or all
    spares are exhausted.

    Eviction strategy:
    - Maintains up to `maxsize` total backends (in-use + spare)
    - Evicts oldest spare backends when count limit is exceeded
    - Never evicts in-use (active) backends
    - On GPU OOM: reactively evicts spares one at a time and retries load

    Usage pattern:
    1. Call `checkout(key)` to get a backend for a request (marks as in-use)
    2. If `checkout` returns None, create a new backend and add via `put(key, backend)`
    3. After request completes, call `checkin(backend)` to mark as spare (available for reuse)
    4. Call `evict_one_spare()` to manually free GPU memory (used by OOM retry logic)

    This is a thread-safe class.
    """

    def __init__(self, maxsize: int = 10):
        """
        Initialize the BackendCache with count-based LRU eviction and OOM retry capability.

        Args:
            maxsize: Maximum number of models to cache (default: 10). When this limit is reached,
                    oldest spare backends are evicted. If OOM occurs during model loading,
                    spare backends are automatically evicted and load is retried.
        """
        if maxsize <= 0:
            raise ValueError(f"BackendCache maxsize must be positive, got {maxsize}")
        self.lock = threading.RLock()

        # List of spare backends (LRU order - oldest first)
        self.spares: List[Backend] = list()

        # Set of backends that are currently in use
        self.in_use: set[Backend] = set()

        self.maxsize = maxsize
        log.info(f"BackendCache initialized with LRU eviction: maxsize={maxsize}, OOM retry enabled")

    def checkout(self, key: str) -> Backend | None:
        """
        Get an available backend for a key and mark as in-use.
        If no available backend is found, return None to indicate caller should create one.
        """

        with self.lock:
            # First try to find an available backend for this key
            for backend in self.spares:
                if backend.cache_key == key:
                    self.spares.remove(backend)
                    self.in_use.add(backend)
                    return backend

            return None

    def checkin(self, backend: Backend):
        """Return a backend to the pool, marking it as available."""

        assert backend.cache_key is not None

        with self.lock:
            assert backend in self.in_use
            self.in_use.remove(backend)
            self.spares.append(backend)

    def put(self, key: str, backend: Backend):
        """
        Add a new backend for the key, initially marked as in-use.
        Automatically evicts spare backends if cache size limit is reached.

        Raises:
            CacheFullError: If maxsize would be exceeded because all backends are in-use
        """

        # The backend must be a brand new instance
        assert backend.cache_key is None

        evicted_any = False
        with self.lock:
            # Check if cache is full BEFORE modifying any state
            # If all maxsize slots are in-use, we cannot add another backend
            if len(self.in_use) >= self.maxsize:
                # All cache slots are occupied by in-use backends, cannot add more
                # Exit lock before expensive GPU cleanup
                pass  # Will raise after lock is released
            else:
                # We can add the backend - proceed with state modifications
                backend.cache_key = key
                self.in_use.add(backend)

                # Enforce count-based limit (returns True if eviction happened)
                evicted_any = self._enforce_count_limit()

        # If cache was full, raise error after releasing lock
        if len(self.in_use) >= self.maxsize and backend.cache_key is None:
            # Free GPU memory from the rejected backend before raising exception
            del backend
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            raise CacheFullError(
                f"Cannot add backend '{key}': all {self.maxsize} cache slots are in-use. "
                f"No spare backends available to evict. "
                f"Consider increasing --cache-size or waiting for requests to complete."
            )

        if evicted_any and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def put_with_retries(
        self,
        key: str,
        backend_factory: Callable[[], Backend],
        max_retries: int = 5,
        enable_eviction: bool = True,
    ) -> Backend:
        """
        Add a new backend by calling backend_factory, with automatic OOM retry logic.

        If GPU OOM occurs during backend creation, optionally evicts spare backends
        and retries. This encapsulates the common pattern of OOM-reactive cache management.

        Args:
            key: Cache key for the backend
            backend_factory: Callable that creates and returns a Backend instance
            max_retries: Maximum number of OOM retries (default: 5)
            enable_eviction: If True, evict spare backends on OOM and retry (default: True)

        Returns:
            The created Backend instance (now cached and marked as in-use)

        Raises:
            NoSpareBackendsError: If OOM occurs and no spare backends available to evict
            CacheFullError: If cache is full (all slots in-use)
            torch.cuda.OutOfMemoryError: If OOM persists after all retries
            Exception: Any other exception from backend_factory
        """
        attempt = 0

        while attempt < max_retries:
            try:
                backend = backend_factory()
                self.put(key, backend)
                return backend

            except torch.cuda.OutOfMemoryError as e:
                attempt += 1
                log.warning(
                    f"OOM error loading '{key}' (attempt {attempt}/{max_retries}). "
                    f"{'Trying to evict a spare backend...' if enable_eviction else 'Eviction disabled, will retry.'}"
                )

                if enable_eviction:
                    # Try to evict a spare backend to free memory
                    if self.evict_one_spare():
                        log.info("Evicted a spare backend, retrying...")
                        continue
                    else:
                        # No more spares to evict
                        log.exception(
                            f"Cannot load '{key}': out of memory and no spare backends to evict. "
                            f"All {len(self.in_use)} cached backends are currently in use."
                        )
                        raise NoSpareBackendsError(
                            f"Out of GPU memory loading '{key}'. No spare backends available to evict."
                        ) from e
                else:
                    # Eviction disabled, just retry
                    if attempt >= max_retries:
                        log.exception(f"OOM loading '{key}' after {max_retries} attempts (eviction disabled)")
                        raise
                    continue

        # Should not reach here, but just in case
        raise torch.cuda.OutOfMemoryError(f"Failed to load '{key}' after {max_retries} attempts")

    def _enforce_count_limit(self) -> bool:
        """
        Enforce count-based cache limit by removing unused backends in LRU order.

        Note: must be called with the lock held

        Returns:
            True if any backends were evicted (caller should do CUDA/GC cleanup)
        """
        assert self.maxsize is not None
        # Count-based eviction: evict spares when total count exceeds maxsize
        evicted_any = False
        while len(self.in_use) + len(self.spares) > self.maxsize and len(self.spares) > 0:
            evicted_backend = self.spares.pop(0)
            cache_key = evicted_backend.cache_key
            log.info(f"Evicted spare backend '{cache_key}' (LRU, maxsize={self.maxsize})")
            # Delete immediately to free references for GC
            del evicted_backend
            evicted_any = True

        # Return whether eviction happened; caller will do CUDA/GC cleanup after releasing lock
        return evicted_any

    def evict_one_spare(self) -> bool:
        """
        Evict the oldest spare backend to free up memory.

        Returns:
            True if a backend was evicted, False if no spares available.
        """
        # Extract backend info while holding lock
        evicted_backend = None
        with self.lock:
            if not self.spares:
                return False

            evicted_backend = self.spares.pop(0)
            cache_key = evicted_backend.cache_key

            log.info(f"Evicting spare backend '{cache_key}' to free GPU memory")

        # Do expensive CUDA/GC cleanup outside the lock to avoid blocking other cache operations
        del evicted_backend
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()  # Force immediate cleanup of Python references and CPU tensors

        return True


class CameraBank:
    def __init__(self, artifact: Artifact, world_to_nre: FrameConversion):
        super().__init__()

        rig_trajectories = RigTrajectories.from_dict(artifact.rig_trajectories)

        # TODO can this be the following? self.world_to_nre = rig_trajectories.world_to_nre
        self.world_to_nre = world_to_nre

        self.cameras: dict[str, TorchCamera] = {}

        unique_camera_id_to_camera_id = {
            uci: rig_trajectories.camera_calibrations[uci].logical_sensor_name
            for uci in rig_trajectories.camera_calibrations.keys()
        }

        # Assumes there is only a single rig_trajectory
        assert len(rig_trajectories.rig_trajectories) == 1, (
            f"{self.__class__.__name__}: expected a single rig_trajectory, got {len(rig_trajectories.rig_trajectories)}"
        )
        rig_trajectory = rig_trajectories.rig_trajectories[0]
        self.trajectory = _rig_trajectory_to_grpc_trajectory(rig_trajectory)

        rig_timestamps_us = rig_trajectory.T_rig_world_timestamps_us
        time_range_us = HalfClosedInterval(
            int(rig_timestamps_us[0].item()),
            int(rig_timestamps_us[-1].item()) + 1,  # +1 to make the interval half-closed
        )

        for unique_camera_id in rig_trajectory.cameras_frame_timestamps_us.keys():
            camera_calibration = rig_trajectories.camera_calibrations[unique_camera_id]
            logical_camera_id = unique_camera_id_to_camera_id[unique_camera_id]

            camera = TorchCamera(
                time_range_us=time_range_us,
                camera_model_parameters=camera_calibration.camera_model_parameters,
                logical_camera_id=logical_camera_id,
                unique_sensor_idx=camera_calibration.unique_sensor_idx,
                world_to_nre=self.world_to_nre,
                T_camera_rig=camera_calibration.T_sensor_rig,
            )

            self.cameras[logical_camera_id] = camera

    @staticmethod
    def _get_camera_model_parameters(
        camera_intrinsics: CameraSpec,
    ) -> ncore.data.ConcreteCameraModelParametersUnion:
        """Get the camera model parameters from the camera intrinsics."""
        # Get the external distortion parameters from the camera intrinsics if provided
        external_distortion_parameters: Optional[ncore.data.ConcreteExternalDistortionParametersUnion] = None
        if camera_intrinsics.WhichOneof("external_distortion") == "bivariate_windshield_model_param":
            windshield_param = camera_intrinsics.bivariate_windshield_model_param

            # Secure workaround for mismatch between integer enum values assigned in the
            # enum ncore.data.ReferencePolynomial (FORWARD:1, BACKWARD:2) (using auto()) and the
            # enum in the .proto file declaration of ReferencePolynomial (FORWARD:0, BACKWARD:1)
            reference_poly: ncore.data.ReferencePolynomial
            match windshield_param.reference_poly:
                case BivariateWindshieldModelParameters.ReferencePolynomial.FORWARD:
                    reference_poly = ncore.data.ReferencePolynomial.FORWARD
                case BivariateWindshieldModelParameters.ReferencePolynomial.BACKWARD:
                    reference_poly = ncore.data.ReferencePolynomial.BACKWARD
                case _:
                    raise ValueError(f"Unsupported reference polynomial type: {windshield_param.reference_poly}")

            external_distortion_parameters = ncore.data.BivariateWindshieldModelParameters(
                reference_poly=reference_poly,
                horizontal_poly=np.array(windshield_param.horizontal_poly, dtype=np.float32),
                vertical_poly=np.array(windshield_param.vertical_poly, dtype=np.float32),
                horizontal_poly_inverse=np.array(windshield_param.horizontal_poly_inverse, dtype=np.float32),
                vertical_poly_inverse=np.array(windshield_param.vertical_poly_inverse, dtype=np.float32),
            )
        elif camera_intrinsics.HasField("external_distortion"):  # Allows for providing no external distortion
            raise ValueError(
                f"Unsupported external distortion type {camera_intrinsics.WhichOneof('external_distortion')}"
            )

        camera_model_parameters: Optional[ncore.data.ConcreteCameraModelParametersUnion] = None
        # For now, only FThetaCameraModelParameters are supported
        if camera_intrinsics.WhichOneof("camera_param") == "ftheta_param":
            ftheta_param = camera_intrinsics.ftheta_param
            camera_model_parameters = ncore.data.FThetaCameraModelParameters(
                resolution=np.array([camera_intrinsics.resolution_w, camera_intrinsics.resolution_h], dtype=np.uint64),
                # shutter type enums are synchronized manually between gRPC and NCORE
                shutter_type=camera_intrinsics.shutter_type,  # type: ignore
                principal_point=np.array(
                    [ftheta_param.principal_point_x, ftheta_param.principal_point_y], dtype=np.float32
                ),
                # PolynomialType enums are synchronized manually between gRPC and NCORE
                reference_poly=ftheta_param.reference_poly,  # type: ignore
                pixeldist_to_angle_poly=np.array(ftheta_param.pixeldist_to_angle_poly, dtype=np.float32),
                angle_to_pixeldist_poly=np.array(ftheta_param.angle_to_pixeldist_poly, dtype=np.float32),
                max_angle=ftheta_param.max_angle,
                external_distortion_parameters=external_distortion_parameters,
                linear_cde=np.array(
                    [
                        ftheta_param.linear_cde.linear_c,
                        ftheta_param.linear_cde.linear_d,
                        ftheta_param.linear_cde.linear_e,
                    ],
                    dtype=np.float32,
                ),
            )
        elif camera_intrinsics.WhichOneof("camera_param") == "opencv_fisheye_param":
            opencv_fisheye_param = camera_intrinsics.opencv_fisheye_param
            camera_model_parameters = ncore.data.OpenCVFisheyeCameraModelParameters(
                resolution=np.array([camera_intrinsics.resolution_w, camera_intrinsics.resolution_h], dtype=np.uint64),
                principal_point=np.array(
                    [opencv_fisheye_param.principal_point_x, opencv_fisheye_param.principal_point_y],
                    dtype=np.float32,
                ),
                focal_length=np.array(
                    [opencv_fisheye_param.focal_length_x, opencv_fisheye_param.focal_length_y], dtype=np.float32
                ),
                radial_coeffs=np.array(opencv_fisheye_param.radial_coeffs, dtype=np.float32),
                max_angle=opencv_fisheye_param.max_angle,
                shutter_type=camera_intrinsics.shutter_type,  # type: ignore
                external_distortion_parameters=external_distortion_parameters,
            )
        elif camera_intrinsics.WhichOneof("camera_param") == "opencv_pinhole_param":
            opencv_pinhole_param = camera_intrinsics.opencv_pinhole_param
            camera_model_parameters = ncore.data.OpenCVPinholeCameraModelParameters(
                resolution=np.array([camera_intrinsics.resolution_w, camera_intrinsics.resolution_h], dtype=np.uint64),
                principal_point=np.array(
                    [opencv_pinhole_param.principal_point_x, opencv_pinhole_param.principal_point_y],
                    dtype=np.float32,
                ),
                focal_length=np.array(
                    [opencv_pinhole_param.focal_length_x, opencv_pinhole_param.focal_length_y], dtype=np.float32
                ),
                radial_coeffs=np.array(opencv_pinhole_param.radial_coeffs, dtype=np.float32),
                tangential_coeffs=np.array(opencv_pinhole_param.tangential_coeffs, dtype=np.float32),
                thin_prism_coeffs=np.array(opencv_pinhole_param.thin_prism_coeffs, dtype=np.float32),
                shutter_type=camera_intrinsics.shutter_type,  # type: ignore
                external_distortion_parameters=external_distortion_parameters,
            )
        # TODO: Add support for other camera model parameters

        if camera_model_parameters is None:
            raise TypeError("Unsupported camera model parameters type.")

        return camera_model_parameters

    def _get_camera(self, camera_intrinsics: CameraSpec) -> TorchCamera:
        """Get an existing TorchCamera from CameraBank or construct a new TorchCamera if parameters are provided"""
        key = camera_intrinsics.logical_id
        # If camera_intrinsics does not have camera_param, return the camera model from the camera bank
        if camera_intrinsics.WhichOneof("camera_param") is None:
            return self.cameras[key]
        # If the request has a assigned camera param, it indicates that user customized the camera parameters
        else:
            with ScopedTimer("get_camera_model_parameters"):
                camera_model_parameters = self._get_camera_model_parameters(camera_intrinsics)
            with ScopedTimer("TorchCamera"):
                # Return a new TorchCamera with the customized camera model parameters, and the rest of the attributes from the original camera
                return TorchCamera(
                    time_range_us=self.cameras[key].time_range_us,
                    camera_model_parameters=camera_model_parameters,
                    logical_camera_id=self.cameras[key].logical_camera_id,
                    unique_sensor_idx=self.cameras[key].unique_sensor_idx,
                    world_to_nre=self.cameras[key].world_to_nre,
                    T_camera_rig=self.cameras[key].T_camera_rig,
                )

    def _populate_ftheta_camera_param(
        self, camera_model_parameters: ncore.data.FThetaCameraModelParameters
    ) -> FthetaCameraParam:
        fthetaCameraParam = FthetaCameraParam()
        fthetaCameraParam.principal_point_x = camera_model_parameters.principal_point[0]
        fthetaCameraParam.principal_point_y = camera_model_parameters.principal_point[1]
        fthetaCameraParam.reference_poly = camera_model_parameters.reference_poly.value  # type: ignore
        fthetaCameraParam.pixeldist_to_angle_poly.extend(camera_model_parameters.pixeldist_to_angle_poly.tolist())
        fthetaCameraParam.angle_to_pixeldist_poly.extend(camera_model_parameters.angle_to_pixeldist_poly.tolist())
        fthetaCameraParam.max_angle = camera_model_parameters.max_angle
        fthetaCameraParam.linear_cde.linear_c = camera_model_parameters.linear_cde[0]
        fthetaCameraParam.linear_cde.linear_d = camera_model_parameters.linear_cde[1]
        fthetaCameraParam.linear_cde.linear_e = camera_model_parameters.linear_cde[2]
        return fthetaCameraParam

    def _populate_opencv_pinhole_camera_param(
        self, camera_model_parameters: ncore.data.OpenCVPinholeCameraModelParameters
    ) -> OpenCVPinholeCameraParam:
        opencvPinholeCameraParam = OpenCVPinholeCameraParam()
        opencvPinholeCameraParam.principal_point_x = camera_model_parameters.principal_point[0]
        opencvPinholeCameraParam.principal_point_y = camera_model_parameters.principal_point[1]
        opencvPinholeCameraParam.focal_length_x = camera_model_parameters.focal_length[0]
        opencvPinholeCameraParam.focal_length_y = camera_model_parameters.focal_length[1]
        opencvPinholeCameraParam.radial_coeffs.extend(camera_model_parameters.radial_coeffs.tolist())
        opencvPinholeCameraParam.tangential_coeffs.extend(camera_model_parameters.tangential_coeffs.tolist())
        opencvPinholeCameraParam.thin_prism_coeffs.extend(camera_model_parameters.thin_prism_coeffs.tolist())
        return opencvPinholeCameraParam

    def _populate_opencv_fisheye_camera_param(
        self, camera_model_parameters: ncore.data.OpenCVFisheyeCameraModelParameters
    ) -> OpenCVFisheyeCameraParam:
        opencvFisheyeCameraParam = OpenCVFisheyeCameraParam()
        opencvFisheyeCameraParam.principal_point_x = camera_model_parameters.principal_point[0]
        opencvFisheyeCameraParam.principal_point_y = camera_model_parameters.principal_point[1]
        opencvFisheyeCameraParam.focal_length_x = camera_model_parameters.focal_length[0]
        opencvFisheyeCameraParam.focal_length_y = camera_model_parameters.focal_length[1]
        opencvFisheyeCameraParam.radial_coeffs.extend(camera_model_parameters.radial_coeffs.tolist())
        opencvFisheyeCameraParam.max_angle = camera_model_parameters.max_angle
        return opencvFisheyeCameraParam

    @staticmethod
    def _populate_external_distortion_parameters(
        intrinsics_proto: CameraSpec,
        distortion_params: Optional[ncore.data.ConcreteExternalDistortionParametersUnion],
    ) -> None:
        """Populates the right external distortion parameters field in a CameraSpec proto class
        according to the concrete type of the external distortion parameters passed as argument
        """
        if distortion_params is None:
            return
        if isinstance(distortion_params, ncore.data.BivariateWindshieldModelParameters):
            distortion_params_proto = BivariateWindshieldModelParameters()
            # Secure workaround for mismatch between integer enum values assigned in the
            # enum ncore.data.ReferencePolynomial (FORWARD:1, BACKWARD:2) (using auto()) and the
            # enum in the .proto file declaration of ReferencePolynomial (FORWARD:0, BACKWARD:1)
            match distortion_params.reference_poly:
                case ncore.data.ReferencePolynomial.FORWARD:
                    distortion_params_proto.reference_poly = (
                        BivariateWindshieldModelParameters.ReferencePolynomial.FORWARD
                    )
                case ncore.data.ReferencePolynomial.BACKWARD:
                    distortion_params_proto.reference_poly = (
                        BivariateWindshieldModelParameters.ReferencePolynomial.BACKWARD
                    )
                case _:
                    raise ValueError(f"Unsupported reference polynomial type: {distortion_params.reference_poly}")
            distortion_params_proto.horizontal_poly.extend(distortion_params.horizontal_poly.tolist())
            distortion_params_proto.vertical_poly.extend(distortion_params.vertical_poly.tolist())
            distortion_params_proto.horizontal_poly_inverse.extend(distortion_params.horizontal_poly_inverse.tolist())
            distortion_params_proto.vertical_poly_inverse.extend(distortion_params.vertical_poly_inverse.tolist())
            intrinsics_proto.bivariate_windshield_model_param.CopyFrom(distortion_params_proto)
        else:
            raise TypeError(f"Unsupported external distortion parameters type {type(distortion_params).__name__}.")

    def get_available_cameras(self) -> Generator[AvailableCamerasReturn.AvailableCamera, None, None]:
        for logical_cam_id, camera in self.cameras.items():
            camera_model_parameters = camera.camera_model_parameters

            # For now, only FThetaCameraModelParameters and OpenCVPinholeCameraModelParameters are supported
            if isinstance(camera_model_parameters, ncore.data.FThetaCameraModelParameters):
                intrinsics_proto = CameraSpec(
                    logical_id=logical_cam_id,
                    ftheta_param=self._populate_ftheta_camera_param(camera_model_parameters),
                    resolution_w=camera_model_parameters.resolution[0],
                    resolution_h=camera_model_parameters.resolution[1],
                    shutter_type=camera_model_parameters.shutter_type.value,  # type: ignore
                )
                CameraBank._populate_external_distortion_parameters(
                    intrinsics_proto, camera_model_parameters.external_distortion_parameters
                )
                yield AvailableCamerasReturn.AvailableCamera(
                    intrinsics=intrinsics_proto,
                    logical_id=logical_cam_id,
                    rig_to_camera=se3_to_grpc_pose(camera.T_camera_rig.numpy()),
                )
            elif isinstance(camera_model_parameters, ncore.data.OpenCVPinholeCameraModelParameters):
                intrinsics_proto = CameraSpec(
                    logical_id=logical_cam_id,
                    opencv_pinhole_param=self._populate_opencv_pinhole_camera_param(camera_model_parameters),
                    resolution_w=camera_model_parameters.resolution[0],
                    resolution_h=camera_model_parameters.resolution[1],
                    shutter_type=camera_model_parameters.shutter_type.value,  # type: ignore
                )
                CameraBank._populate_external_distortion_parameters(
                    intrinsics_proto, camera_model_parameters.external_distortion_parameters
                )
                yield AvailableCamerasReturn.AvailableCamera(
                    intrinsics=intrinsics_proto,
                    logical_id=logical_cam_id,
                    rig_to_camera=se3_to_grpc_pose(camera.T_camera_rig.numpy()),
                )
            elif isinstance(camera_model_parameters, ncore.data.OpenCVFisheyeCameraModelParameters):
                intrinsics_proto = CameraSpec(
                    logical_id=logical_cam_id,
                    opencv_fisheye_param=self._populate_opencv_fisheye_camera_param(camera_model_parameters),
                    resolution_w=camera_model_parameters.resolution[0],
                    resolution_h=camera_model_parameters.resolution[1],
                    shutter_type=camera_model_parameters.shutter_type.value,  # type: ignore
                )
                CameraBank._populate_external_distortion_parameters(
                    intrinsics_proto, camera_model_parameters.external_distortion_parameters
                )
                yield AvailableCamerasReturn.AvailableCamera(
                    intrinsics=intrinsics_proto,
                    logical_id=logical_cam_id,
                    rig_to_camera=se3_to_grpc_pose(camera.T_camera_rig.numpy()),
                )
            else:
                raise TypeError(f"Unsupported camera model parameters type.")


class LidarBank:
    def __init__(self, artifact: Artifact):
        super().__init__()

        rig_trajectories = RigTrajectories.from_dict(artifact.rig_trajectories)
        lidar_calibrations = rig_trajectories.lidar_calibrations
        assert len(lidar_calibrations) <= 1, "Up to one lidar is supported only"
        self.lidar: LidarModelBundle | None = None
        if len(lidar_calibrations) == 1:
            lidar_calibration = list(lidar_calibrations.values())[0]
            lidar_model_parameters = lidar_calibration.lidar_model_parameters
            if lidar_model_parameters is not None:
                self.lidar = LidarModelBundle.load_from_config(lidar_model_parameters.to_dict())

    def get_lidar(self) -> LidarModelBundle | None:
        return self.lidar


class AssetBank:
    """
    Manages external track assets (PLY files) from artifact.
    Follows the same pattern as CameraBank and LidarBank.
    """

    def __init__(self, artifact: Artifact):
        super().__init__()
        self.external_assets = artifact.external_assets  # Dict[track_id, ply_bytes]
        self._dims_offset = OmegaConf.select(
            OmegaConf.create(artifact.parsed_config),
            "dataset.valid_pixels_cuboid_track_params.track_padding_m",
            default=[0.0, 0.0, 0.0],
        )

        if self.external_assets:
            log.info(
                f"AssetBank initialized with {len(self.external_assets)} track assets: "
                f"{list(self.external_assets.keys())}"
            )
        else:
            log.info("AssetBank initialized with no external assets")

    @property
    def dims_offset(self) -> torch.Tensor:
        return torch.tensor(list(self._dims_offset), dtype=torch.float32, device="cuda")

    def has_track(self, track_id: str) -> bool:
        """Check if a track ID exists in external assets."""
        return track_id in self.external_assets

    def get_track_ids(self) -> List[str]:
        """Get all available track IDs."""
        return list(self.external_assets.keys())

    def get_track_asset(self, track_id: str, cuboid_dims: torch.Tensor, device: str = "cuda") -> Asset:
        """
        Get parsed Asset for a track with specified cuboid dimensions.

        Args:
            track_id: Track ID
            cuboid_dims: Cuboid dimensions tensor
            device: Device to load tensors onto

        Raises:
            KeyError: If track_id not found
        """
        if not self.has_track(track_id):
            raise KeyError(f"Track ID '{track_id}' not found. Available: {self.get_track_ids()}")

        asset_data = self.external_assets[track_id]
        ply_bytes = asset_data["ply_bytes"]

        return Asset.from_ply_bytes(ply_bytes, device=device, cuboids_dims=cuboid_dims)

    def get_track_cuboid_dims(self, track_id: str, device: str = "cuda") -> torch.Tensor:
        """
        Get the cuboid dimensions for a track ID.
        """
        if not self.has_track(track_id):
            raise KeyError(f"Track ID '{track_id}' not found. Available: {self.get_track_ids()}")

        asset_data = self.external_assets[track_id]
        return torch.tensor(asset_data["metadata"]["cuboids_dims"], dtype=torch.float32, device=device)


def with_render_lock():
    """
    Decorator to ensure that Backend render function is thread-safe.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(backend: Backend, *args, **kwargs):
            with backend._render_lock:
                return func(backend, *args, **kwargs)

        return wrapper

    return decorator


@dataclass
class Backend:
    renderable_model: RenderableModel
    camera_bank: CameraBank
    world_to_nre: FrameConversion
    lidar_bank: LidarBank
    asset_bank: AssetBank
    # for thread-safe access (see with_render_lock)
    _render_lock: threading.Lock = field(default_factory=threading.Lock)

    # Caching key like scene_id. See BackendCache for more details.
    cache_key: str | None = None

    class RendererType(Enum):
        NREND = auto()
        GSPLAT = auto()
        DEFAULT = auto()

    def __hash__(self):
        # Use object identity for hashing since cache_key is mutable
        # This ensures hash stability even before cache_key is set
        return hash(id(self))

    def __eq__(self, other):
        # Use object identity for equality to match __hash__
        # This maintains the hash/eq contract: if a == b, then hash(a) == hash(b)
        return self is other

    @staticmethod
    def from_artifact(
        artifact: Artifact,
        renderer: Backend.RendererType,
    ) -> Backend:
        try:
            config_overrides: tuple[str, ...] = ()
            enable_nrend = False

            if renderer == Backend.RendererType.GSPLAT:
                # gRPC serves many artifacts; prefer an explicit flag over arbitrary overrides.
                config_overrides = ("model.renderer.name=3dgut-gsplat",)
            elif renderer == Backend.RendererType.NREND:
                enable_nrend = True

            renderable_model = RenderableModel.load_from_artifact(
                artifact, enable_nrend=enable_nrend, config_overrides=config_overrides
            )
        except (TypeError, ValueError) as exc:
            raise ModelIncompatibilityError(str(exc)) from exc

        world_to_nre = RigTrajectories.from_dict(artifact.rig_trajectories).world_to_nre

        backend = Backend(
            renderable_model=renderable_model,
            camera_bank=CameraBank(artifact, world_to_nre),
            world_to_nre=world_to_nre,
            lidar_bank=LidarBank(artifact),
            asset_bank=AssetBank(artifact),
        )

        torch.cuda.empty_cache()
        gc.collect()

        return backend

    def _validate_dynamic_object_updates_request(self, num_dynamic_objects: int, enable_editing_actors: bool) -> None:
        """Validate whether dynamic object updates are allowed for the current request."""
        if num_dynamic_objects == 0:
            return

        if not enable_editing_actors:
            raise ValueError(
                "Got render request with DynamicObject updates but actor editing is disabled on the server. "
                "Restart serve-grpc with --enable-editing-actors."
            )

        if not self.renderable_model.supports_edit_actors():
            raise ValueError("Got render request with DynamicObject updates but model does not support actor updates")

    @with_render_lock()
    @ScopedTimer()
    def render_camera_request(
        self, request: RGBRenderRequest, enable_editing_actors: bool = False
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        self._validate_dynamic_object_updates_request(
            num_dynamic_objects=len(request.dynamic_objects),
            enable_editing_actors=enable_editing_actors,
        )

        camera = self.camera_bank._get_camera(request.camera_intrinsics)
        ray_bundle = camera.build_ray_bundle(request)

        if enable_editing_actors and self.renderable_model.supports_edit_actors():
            actors_snapshot = actors_snapshot_from_render_request(request)
        else:
            actors_snapshot = None

        rendered = self.renderable_model.render_camera_frame_from_ray_bundle(
            ray_bundle,
            actors_snapshot=actors_snapshot,
            frame_start_us=request.frame_start_us,
            frame_end_us=request.frame_end_us,
        )
        w, h = ray_bundle.rendering_data.w, ray_bundle.rendering_data.h

        assert rendered.color_image is not None, "Renderer did not return a color image"

        return rendered.color_image, (h, w)

    def get_available_cameras(self) -> AvailableCamerasReturn:
        return AvailableCamerasReturn(available_cameras=list(self.camera_bank.get_available_cameras()))

    def get_available_trajectories(self) -> AvailableTrajectoriesReturn:
        return AvailableTrajectoriesReturn(
            available_trajectories=[
                AvailableTrajectoriesReturn.AvailableTrajectory(
                    trajectory_idx=0, trajectory=self.camera_bank.trajectory
                )
            ]
        )

    def get_external_asset_objects(self) -> ExternalAssetObjectsReturn:
        """
        Get list of track IDs for external asset objects.

        Returns:
            ExternalAssetObjectsReturn containing list of available track IDs
        """
        return ExternalAssetObjectsReturn(track_ids=self.asset_bank.get_track_ids())

    def get_dynamic_objects(self) -> List[sensorsim_types.DynamicObjectTrack]:
        """Return DynamicObjectTrack for all controllable actors in gRPC format."""
        if not self.renderable_model.supports_edit_actors():
            return []  # Return empty list instead of raising error

        actor_tracks = self.renderable_model.get_actor_tracks()
        return actor_tracks_to_grpc(actor_tracks)

    def replace_assets(self, replace_list: List[sensorsim_types.ReplaceAssetAction]) -> None:
        def is_empty_aabb(aabb: grpc_types.AABB) -> bool:
            """Check if AABB is all zeros."""
            return aabb.size_x == 0.0 and aabb.size_y == 0.0 and aabb.size_z == 0.0

        for replacement in replace_list:
            original_id = replacement.original_id
            replacement_id = replacement.replacement_id

            if not replacement_id:
                raise ValueError(
                    f"Replacement_id is empty for original_id '{original_id}', please provide a valid replacement_id"
                )

            device = str(self.renderable_model.device)

            # Check if object_size is provided and valid, otherwise use fallback
            if not replacement.HasField("object_size") or is_empty_aabb(replacement.object_size):
                if not self.asset_bank.has_track(replacement_id):
                    raise ValueError(f"object_size required for filesystem path replacement_id '{replacement_id}' ")
                log.info(
                    f"object_size not provided or empty for original_id '{original_id}', "
                    f"using fallback from replacement asset '{replacement_id}'"
                )
                cuboid_dims = self.asset_bank.get_track_cuboid_dims(replacement_id, device=device)
            else:
                cuboid_dims = torch.tensor(
                    [replacement.object_size.size_x, replacement.object_size.size_y, replacement.object_size.size_z],
                    dtype=torch.float32,
                    device=device,
                )

            if self.asset_bank.has_track(replacement_id):
                asset = self.asset_bank.get_track_asset(replacement_id, cuboid_dims=cuboid_dims, device=device)
            else:
                asset = Asset(Path(replacement_id), device=device)
                asset.cuboids_dims = cuboid_dims
                log.info(
                    f"Loaded PLY from filesystem for replacement '{original_id}': {replacement_id} "
                    f"({asset.positions.shape[0]} gaussians)"
                )
            replaced = self.renderable_model.replace_asset(
                original_id, asset=asset, dims_offset=self.asset_bank.dims_offset
            )
            if not replaced:
                raise RuntimeError(
                    f"Failed to replace original_id='{original_id}' with replacement_id='{replacement_id}'"
                )

    def insert_assets(self, dynamic_objects: List[sensorsim_types.DynamicObjectTrack]) -> None:
        """Insert dynamic object assets with proper coordinate frame conversion.

        Supports both AssetBank track IDs and filesystem PLY paths as asset_id.
        AssetBank is checked first; if the ID is not found, it is treated as a filesystem path.
        """
        actor_tracks = actor_tracks_from_grpc(dynamic_objects)

        track_to_asset = {obj.id: obj.asset_id for obj in dynamic_objects}

        for cuboid_track in actor_tracks._cuboid_tracks_list:
            (track_id,) = cuboid_track.tracks_id

            asset_id = track_to_asset[track_id]

            nre_cuboid_track = CuboidTracks.Ops.transform_with_frame_conversion(cuboid_track, self.world_to_nre, None)

            cuboid_dims = torch.tensor(
                nre_cuboid_track.cuboids_dims, dtype=torch.float32, device=str(self.renderable_model.device)
            )

            device = str(self.renderable_model.device)
            if self.asset_bank.has_track(asset_id):
                asset = self.asset_bank.get_track_asset(asset_id, cuboid_dims=cuboid_dims, device=device)
            else:
                asset = Asset(Path(asset_id), device=device)
                asset.cuboids_dims = cuboid_dims
                log.info(
                    f"Loaded PLY from filesystem for track '{track_id}': {asset_id} "
                    f"({asset.positions.shape[0]} gaussians)"
                )

            inserted = self.renderable_model.insert_asset(asset, nre_cuboid_track, self.asset_bank.dims_offset)
            if not inserted:
                raise RuntimeError(f"Failed to insert track_id='{track_id}' with asset_id='{asset_id}'")

    @with_render_lock()
    def edit_assets(
        self,
        replace_assets: List[sensorsim_types.ReplaceAssetAction],
        insert_assets: List[sensorsim_types.DynamicObjectTrack],
    ) -> None:
        """
        Edit assets by replacing and inserting assets in the renderable_model.
        The client should send a one-time edit_assets request with the desired asset list to be edited before sending their RGBRenderRequests.
        Each edit operation will change the internal state of the renderable_model

        Args:
            replace_assets: List of ReplaceAssetAction specifying original_id (track from artifact) and replacement_id (asset from AssetBank)
            insert_assets: List of DynamicObjectTrack to insert, the ply data of the inserted assets should be present in the AssetBank
        """
        if not self.renderable_model.supports_edit_actors():
            raise RuntimeError("Trying to edit assets but model does not support actor updates")

        # Save the original training parameters of the model before applying asset edit operations, user can restore the original parameters
        self.renderable_model.save_training_parameters()

        try:
            if replace_assets:
                self.replace_assets(replace_assets)

            if insert_assets:
                self.insert_assets(insert_assets)
        except Exception:
            self.renderable_model.restore_training_parameters()
            raise

    @with_render_lock()
    def restore_model_parameters(self) -> None:
        """
        Restore the original training parameters of the model.
        This undoes any asset editing operations.
        """
        if not self.renderable_model.supports_edit_actors():
            raise RuntimeError("Model does not support actor updates")

        self.renderable_model.restore_training_parameters()

    @with_render_lock()
    @ScopedTimer()
    def render_lidar_request(
        self, request: LidarRenderRequest, enable_editing_actors: bool = False
    ) -> LidarRenderReturn:
        """
        Go-to gRPC lidar rendering API.
        Currently missing
            * proper serialization of Lidar spec
            * proper trajectories for ego and traffic objects: currently only start and end pose are passed
        """

        self._validate_dynamic_object_updates_request(
            num_dynamic_objects=len(request.dynamic_objects),
            enable_editing_actors=enable_editing_actors,
        )

        # Update dynamic tracks
        actors_snapshot: Optional[ActorsSnapshot] = None
        if enable_editing_actors and self.renderable_model.supports_edit_actors():
            actors_snapshot = actors_snapshot_from_render_request(request)

        # generate lidar rays from a nominal Lidar model
        with ScopedTimer("LidarRayBundle.build"):
            tquat_sensor_world_start = torch.tensor(grpc_pose_to_tquat(request.sensor_pose.start_pose))
            tquat_sensor_world_end = torch.tensor(grpc_pose_to_tquat(request.sensor_pose.end_pose))

            lidar_to_world = PoseRange(
                tquat_sensor_world_start, tquat_sensor_world_end, request.frame_start_us, request.frame_end_us
            )
            lidar_ray_bundle = LidarRayBundle.build(
                lidar_model=unpack_optional(
                    self.lidar_bank.get_lidar(), msg="Lidar rendering is not supported: artifact has no lidar sensor"
                ),
                lidar_to_world=lidar_to_world,
                world_to_nre=self.world_to_nre,
            )

        # Extract filter parameters from request (use None to signal "use default")
        raydrop_threshold = None
        opacity_threshold = None
        enable_distance_filter = None
        distance_filter_threshold = None

        if request.HasField("render_filter"):
            render_filter: LidarRenderFilter = request.render_filter
            if render_filter.HasField("raydrop_threshold"):
                raydrop_threshold = render_filter.raydrop_threshold
            if render_filter.HasField("opacity_threshold"):
                opacity_threshold = render_filter.opacity_threshold
            if render_filter.HasField("enable_distance_filter"):
                enable_distance_filter = render_filter.enable_distance_filter
            if render_filter.HasField("distance_filter_threshold"):
                distance_filter_threshold = render_filter.distance_filter_threshold

        with ScopedTimer("render_lidar_frame_from_ray_bundle"):
            lidar_frame = self.renderable_model.render_lidar_frame_from_ray_bundle(
                lidar_ray_bundle,
                raydrop_threshold=raydrop_threshold,
                opacity_threshold=opacity_threshold,
                enable_distance_filter=enable_distance_filter,
                distance_filter_threshold=distance_filter_threshold,
                actors_snapshot=actors_snapshot,
                frame_start_us=request.frame_start_us,
                frame_end_us=request.frame_end_us,
            )

        with ScopedTimer("unpack_results"):
            pc_tensor = lidar_frame.point_positions.cpu().flatten().numpy().astype(np.float32)
            if lidar_frame.point_intensities is not None:
                intensity_tensor = lidar_frame.point_intensities.cpu().flatten().numpy().astype(np.float32)
            else:
                intensity_tensor = None

        with ScopedTimer("protobuf encoding"):
            # Create protobuf message directly from tensors
            final_result = LidarRenderReturn()

            if pc_tensor is not None:
                # Use numpy array view for zero-copy conversion
                final_result.point_xyzs_buffer = pc_tensor.tobytes()
                final_result.num_points = len(pc_tensor) // 3

            if intensity_tensor is not None:
                final_result.point_intensities_buffer = intensity_tensor.tobytes()
            else:
                log.info(f"The model loaded has no intensity prediction")

        return final_result


class ServerMetricsManager:
    """Manages metrics collection for the gRPC server."""

    def __init__(self, metrics_output_dir: Optional[str], artifacts: Optional[dict[str, Artifact]] = None):
        self._metrics_output_dir = metrics_output_dir
        self._metrics_collector: Optional[MetricsCollector] = None
        self._warmup_rgb_counter = 0
        self._warmup_batch_rgb_counter = 0
        self._warmup_lidar_counter = 0
        self._warmup_skip = 3
        self._save_timer: Optional[threading.Timer] = None
        self._timer_lock = threading.Lock()

        # Initialize metrics collection if output directory is specified
        if self._metrics_output_dir:
            self._initialize_metrics_collection(artifacts)

    def _initialize_metrics_collection(self, artifacts: Optional[dict[str, Artifact]] = None) -> None:
        """Initialize metrics collection with run_id detection."""
        # Create output directory if it doesn't exist
        if self._metrics_output_dir is None:
            raise ValueError("Metrics output directory is None")
        Path(self._metrics_output_dir).mkdir(parents=True, exist_ok=True)

        # Try to get run_id from artifact metadata, fallback to directory-based deduction
        run_id = "grpc_server"
        if artifacts:
            # Use the first artifact's metadata to get run_id
            first_artifact = next(iter(artifacts.values()))
            try:
                metadata = first_artifact.metadata
                if metadata and "logger" in metadata and "run_id" in metadata["logger"]:
                    artifact_run_id = metadata["logger"]["run_id"]
                    if artifact_run_id:  # Check if it's not None or empty
                        run_id = artifact_run_id
                        log.info(f"Using run_id from artifact metadata: {run_id}")
                    else:
                        log.warning("run_id in artifact metadata is empty, falling back to directory-based deduction")
                        run_id = self._deduce_run_id_from_path(first_artifact)
                else:
                    log.warning("No run_id found in artifact metadata, falling back to directory-based deduction")
                    run_id = self._deduce_run_id_from_path(first_artifact)
            except Exception as e:
                log.warning(
                    f"Failed to read run_id from artifact metadata: {e}, falling back to directory-based deduction"
                )
                run_id = self._deduce_run_id_from_path(first_artifact)

        # Create metrics collector with individual parameters
        self._metrics_collector = MetricsCollector(mode="grpc_server", run_id=run_id)
        log.info(f"Metrics collection enabled, output directory: {self._metrics_output_dir}")

    def _deduce_run_id_from_path(self, artifact: Artifact) -> str:
        """
        Fallback method to deduce run_id from artifact file path.
        This is used when run_id is not available in artifact metadata.
        """
        artifact_path = Path(artifact.source)
        # Try to extract a meaningful run identifier from the artifact path
        # Look for common patterns like timestamps, experiment names, etc.
        if artifact_path.parent.name != "." and artifact_path.parent.name:
            if (
                artifact_path.parent.stem == "artifacts"
                and artifact_path.parent.parent.name != "."
                and artifact_path.parent.parent.name
            ):
                return artifact_path.parent.parent.name
            else:
                return artifact_path.parent.name
        else:
            return artifact_path.stem  # filename without extension

    def record_rgb_render_time(self, render_time_ms: float) -> None:
        """Record RGB render time if metrics are enabled and warmup is complete."""
        if self._metrics_collector is None:
            return

        if self._warmup_rgb_counter >= self._warmup_skip:
            self._metrics_collector.save_metric(create_metric_sample("render_time_ms", render_time_ms))
            self._save_metrics_deferred()

        self._warmup_rgb_counter += 1

    def record_lidar_render_time(self, render_time_ms: float) -> None:
        """Record LiDAR render time if metrics are enabled and warmup is complete."""
        if self._metrics_collector is None:
            return

        if self._warmup_lidar_counter >= self._warmup_skip:
            metric_sample = create_metric_sample("lidar_render_time_ms", render_time_ms)
            self._metrics_collector.save_metric(metric_sample)
            self._save_metrics_deferred()

        self._warmup_lidar_counter += 1

    def record_batch_rgb_render_time(self, render_time_ms: float, num_cameras: int) -> None:
        """Record batch RGB render time if metrics are enabled and warmup is complete.

        Args:
            render_time_ms: Total time to render the batch.
            num_cameras: Number of cameras in the batch (for per-camera avg calculation).
        """
        if self._metrics_collector is None:
            return

        if self._warmup_batch_rgb_counter >= self._warmup_skip:
            self._metrics_collector.save_metric(create_metric_sample("batch_render_time_ms", render_time_ms))
            if num_cameras > 0:
                self._metrics_collector.save_metric(
                    create_metric_sample("batch_render_time_per_camera_ms", render_time_ms / num_cameras)
                )
            self._save_metrics_deferred()

        self._warmup_batch_rgb_counter += 1

    def shutdown(self) -> None:
        """Cancel any pending timer and save final metrics before shutdown."""
        if self._metrics_collector is None:
            return

        with self._timer_lock:
            if self._save_timer is not None:
                self._save_timer.cancel()
                self._save_timer = None
        self._save_metrics()

    def record_duration(self, name: str, duration_ms: float) -> None:
        """Record a generic duration metric (e.g., gRPC ser/de) if metrics are enabled."""
        if self._metrics_collector is None:
            return

        self._metrics_collector.save_metric(create_metric_sample(name, duration_ms))
        self._save_metrics_deferred()

    def _save_metrics(self) -> None:
        """Save collected metrics to the output directory."""
        if self._metrics_collector is None or self._metrics_output_dir is None:
            return

        try:
            # Write metrics to YAML file (the method creates metrics.yaml in the directory)
            self._metrics_collector.write_metrics_to_yaml(self._metrics_output_dir)
        except Exception:
            log.exception("Failed to save metrics")

    def _save_metrics_deferred(self) -> None:
        """Schedule a deferred save of metrics that gets delayed by 1 second each time it's called."""
        if self._metrics_collector is None or self._metrics_output_dir is None:
            return

        with self._timer_lock:
            # Cancel any existing timer
            if self._save_timer is not None:
                self._save_timer.cancel()

            # Start a new timer that will save metrics after 1 second of inactivity
            self._save_timer = threading.Timer(1.0, self._save_metrics)
            self._save_timer.start()


def _is_local_scene_uri(uri: str) -> bool:
    """Return True if uri is a file:// URI or absolute local path (for on-the-fly scene registration)."""
    stripped = uri.strip()
    return stripped.lower().startswith("file://") or Path(stripped).is_absolute()


class SceneDownloadInterceptor(grpc.ServerInterceptor):
    """Interceptor that checks for x-nre-scene-url header and downloads an external scene (http(s)://) or registers from local filesystem (file:// or local path)."""

    def __init__(self):
        self.service = None

    def set_service(self, sensor_sim_service: SensorSimService):
        self.service = sensor_sim_service

    def _terminate(self, code: grpc.StatusCode, details: str) -> grpc.UnaryUnaryRpcCallback:
        def _terminator(ignored_request, context):
            context.abort(code, details)

        return grpc.unary_unary_rpc_method_handler(_terminator)

    def intercept_service(
        self,
        continuation: Callable[[grpc.HandlerCallDetails], grpc.RpcMethodHandler],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        # Check for our custom header
        metadata = dict(handler_call_details.invocation_metadata)

        if "x-nre-scene-url" in metadata:
            assert self.service is not None, "Service not set"

            scene_url = metadata["x-nre-scene-url"]
            scene_id = metadata["x-nre-scene-id"]

            if not check_safe_scene_id(scene_id):
                return self._terminate(grpc.StatusCode.INVALID_ARGUMENT, f"Invalid scene ID: {scene_id}")

            is_local = _is_local_scene_uri(scene_url)
            try:
                if is_local:
                    self.service._register_local_scene(scene_id, scene_url)
                else:
                    self.service._download_scene(scene_url, scene_id)
            except Exception as e:
                if is_local:
                    log.exception(f"Failed to register local scene {scene_id} from {scene_url}")
                    return self._terminate(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        f"Failed to register local scene from {scene_url}: {e}",
                    )
                else:
                    log.exception(f"Failed to download scene {scene_id} from {scene_url}")
                    return self._terminate(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        f"Failed to download scene from {scene_url}: {e}",
                    )

        # Continue with normal processing
        return continuation(handler_call_details)


class MetricsInterceptor(grpc.ServerInterceptor):
    """Interceptor to measure decode/encode and end-to-end durations per RPC."""

    def __init__(self) -> None:
        self.metrics_manager: Optional[ServerMetricsManager] = None

    def set_metrics_manager(self, manager: ServerMetricsManager) -> None:
        self.metrics_manager = manager

    def intercept_service(
        self,
        continuation: Callable[[grpc.HandlerCallDetails], grpc.RpcMethodHandler],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        handler = continuation(handler_call_details)
        # Only wrap unary-unary for our service methods
        if handler is None or handler.unary_unary is None:
            return handler

        # Determine method-specific metric names
        method_name = handler_call_details.method.split("/")[-1]
        decode_metric = f"grpc.decode.{method_name}_ms"
        encode_metric = f"grpc.encode.{method_name}_ms"
        e2e_sent_metric = f"grpc.e2e_sent.{method_name}_ms"

        metrics = self.metrics_manager

        # Per-RPC E2E start captured in deserializer and shared with close callback.
        # Using a closure keeps state per request without thread-local/global storage.
        e2e_start = 0.0

        def request_deserializer(raw: bytes):
            nonlocal e2e_start
            if metrics is not None:
                # Capture earliest per-RPC start (before deserialization).
                e2e_start = time.perf_counter()

            obj = handler.request_deserializer(raw)
            if metrics is not None:
                metrics.record_duration(decode_metric, (time.perf_counter() - e2e_start) * 1000.0)
            return obj

        def response_serializer(resp):
            nonlocal e2e_start
            t0 = 0.0
            if metrics is not None:
                t0 = time.perf_counter()
            out = handler.response_serializer(resp)
            if metrics is not None:
                t1 = time.perf_counter()
                if e2e_start:
                    metrics.record_duration(e2e_sent_metric, (t1 - e2e_start) * 1000.0)

                metrics.record_duration(encode_metric, (t1 - t0) * 1000.0)

            return out

        return grpc.unary_unary_rpc_method_handler(
            handler.unary_unary,
            request_deserializer=request_deserializer,
            response_serializer=response_serializer,
        )


class SensorSimService(SensorsimServiceServicer):
    def __init__(
        self,
        server: Optional[grpc.Server],  # None during tests
        artifacts_glob: Optional[str],
        ray_chunk_size: int,
        egocar_hoods_dir: str | None = None,
        downloader: Optional[Downloader] = None,  # None during tests
        scene_cache: Optional[SceneCache] = None,  # None during tests
        cache_size: int = 10,
        metrics_output_dir: Optional[str] = None,
        enable_editing_actors: Optional[bool] = False,
        server_config: Optional[GrpcServerConfig] = None,
    ) -> None:
        self.server = server
        self.downloader = downloader
        self.scene_cache = scene_cache
        self.ray_chunk_size = ray_chunk_size
        self.enable_editing_actors = enable_editing_actors
        self.version_id = self._create_version_id()
        self.server_config_map = self._build_server_config_map(server_config)
        self.server_config = server_config
        log.info(f"Identifying as {self.version_id=}")

        if artifacts_glob is None:
            artifacts_list = []
        else:
            artifacts_list = Artifact.discover_from_glob(artifacts_glob)

        # Check for duplicates -- the desired abstraction is to index by scene_id
        scene_id_to_artifact_paths: dict[str, list[Path]] = {}
        for artifact in artifacts_list:
            scene_id_to_artifact_paths.setdefault(artifact.scene_id, []).append(artifact.source)
        duplicates = {artifact: paths for artifact, paths in scene_id_to_artifact_paths.items() if len(paths) >= 2}
        if duplicates:
            raise AssertionError(f"Duplicate scene IDs found. Duplicates (scene_id: artifact paths): {duplicates}.")

        self.artifacts = {artifact.scene_id: artifact for artifact in artifacts_list}
        self._local_scene_ids: set[str] = set()
        self._local_registration_lock = threading.Lock()

        log.info(f"Available scenes: {list(self.artifacts.keys())}.")

        self.difix_enabled = False

        self.egocar_rig_bank = (
            EgocarRigBank.load_from_dir(egocar_hoods_dir) if egocar_hoods_dir else EgocarRigBank.empty()
        )
        log.info(f"Available egocar masks: {self.egocar_rig_bank}.")

        self.nvjpeg_encoder = nvimgcodec.Encoder()
        self.nvjpeg_encoder_lock = threading.Lock()

        # Initialize backend cache with LRU eviction and OOM retry capability
        self.backend_cache = BackendCache(maxsize=cache_size)

        # Initialize metrics manager
        self._metrics_manager = ServerMetricsManager(metrics_output_dir, self.artifacts)

    @staticmethod
    def _build_server_config_map(server_config: Optional[GrpcServerConfig]) -> dict[str, str]:
        if server_config is None:
            return {}

        raw_config_dict = server_config.model_dump()

        def stringify_value(value: Any) -> str:
            if value is None:
                return "null"
            if isinstance(value, Enum):
                return str(value.value)
            if isinstance(value, (str, int, float, bool, list, dict, tuple)):
                return str(value)
            else:
                raise ValueError(f"Unsupported config type: {type(value)}")

        return {key: stringify_value(value) for key, value in raw_config_dict.items()}

    @ScopedTimer()
    def encode_image(self, image_u8: torch.Tensor, request: RGBRenderRequest) -> RGBRenderReturn:
        match request.image_format:
            case ImageFormat.UNDEFINED:
                raise TypeError("Request did not specify image format (default value found).")
            case ImageFormat.PNG:
                image = Image.fromarray(image_u8.cpu().numpy())
                img_io = io.BytesIO()
                image.save(img_io, "PNG")
                img_io.seek(0)
                return RGBRenderReturn(image_bytes=img_io.getvalue())
            case ImageFormat.JPEG:
                if request.image_quality == 0:
                    raise ValueError("Request did not specify image quality (found default of 0).")

                with self.nvjpeg_encoder_lock:
                    return RGBRenderReturn(
                        image_bytes=self.nvjpeg_encoder.encode(
                            image_u8.contiguous(),  # nvjpeg requires contiguous tensors and e.g. difix can return non-contiguous tensors
                            codec=".jpg",
                            params=nvimgcodec.EncodeParams(
                                quality=request.image_quality,
                            ),
                        ),
                    )
            case ImageFormat.JPEG2000:
                raise TypeError("JPEG2k is not currently supported.")
            case ImageFormat.RGB_UINT8_PLANAR:
                image_u8_np = image_u8.cpu().numpy()
                planar_image_np = image_u8_np.transpose(2, 0, 1)
                return RGBRenderReturn(image_bytes=planar_image_np.tobytes())
            case _:
                raise TypeError(f"Unknown image format requested ({request.image_format=}).")

    def set_difix(
        self,
        difix_enabled: bool,
        difix_url: str,
        difix_cache_dir: str,
        model_filename: str,
        difix_resolution: tuple[int, int],
    ) -> None:
        """
        Optionally enable Difix for render postprocessing.
        """
        self.difix_enabled = difix_enabled
        if difix_enabled:
            self.difix_model = DifixModelFactory.get(difix_url, difix_cache_dir, model_filename, difix_resolution)

    def test_all_scenes(self) -> None:
        """
        Attempts to load each available scene to fail fast
        """
        log.info("Testing gathered scenes...")
        for scene_id in self.artifacts:
            with self.get_backend(scene_id) as _:
                pass
        log.info("...done testing gathered scenes.")

    def _create_version_id(self) -> grpc_types.VersionId:
        nre_version = unpack_optional(
            get_version(
                # Use an empty default if version is not available in current env (e.g. during sandboxed unit test execution)
                allow_empty=True
            )
        )
        return grpc_types.VersionId(
            version_id=nre_version.semantic_string(),
            git_hash=nre_version.git_commit_sha_short,
            grpc_api_version=grpc_types.VersionId.APIVersion(major=0, minor=0, patch=1),
        )

    def get_version(self, request: grpc_types.Empty, context: grpc.ServicerContext) -> grpc_types.VersionId:
        log.info(f"get_version")
        return self.version_id

    def get_server_config(
        self, request: grpc_types.Empty, context: grpc.ServicerContext
    ) -> sensorsim_types.ServerConfig:
        log.info("get_server_config")
        return sensorsim_types.ServerConfig(
            server_config=self.server_config_map,
        )

    def get_available_scenes(
        self, request: grpc_types.Empty, context: grpc.ServicerContext
    ) -> grpc_types.AvailableScenesReturn:
        log.info(f"get_available_scenes")
        return grpc_types.AvailableScenesReturn(scene_ids=self.artifacts)

    def get_available_cameras(
        self, request: AvailableCamerasRequest, context: grpc.ServicerContext
    ) -> AvailableCamerasReturn:
        log.info(f"get_available_cameras")
        try:
            with self.get_backend(request.scene_id) as backend:
                return backend.get_available_cameras()
        except (CacheFullError, NoSpareBackendsError) as e:
            log.warning(f"Cannot serve scene {request.scene_id}: {e}")
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))
            raise  # unreachable, but needed for mypy
        except Exception:
            log.error(traceback.format_exc())
            raise
        return  # unreachable, but needed for mypy

    def get_available_trajectories(
        self, request: AvailableTrajectoriesRequest, context: grpc.ServicerContext
    ) -> AvailableTrajectoriesReturn:
        log.info(f"get_available_trajectories")
        try:
            with self.get_backend(request.scene_id) as backend:
                return backend.get_available_trajectories()
        except (CacheFullError, NoSpareBackendsError) as e:
            log.warning(f"Cannot serve scene {request.scene_id}: {e}")
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))
            raise  # unreachable, but needed for mypy
        except Exception:
            log.error(traceback.format_exc())
            raise

    def get_available_ego_masks(
        self, request: grpc_types.Empty, context: grpc.ServicerContext
    ) -> AvailableEgoMasksReturn:
        log.info(f"get_available_ego_masks")
        return AvailableEgoMasksReturn(ego_mask_metadata=self.egocar_rig_bank.available_metadata())

    def get_external_asset_objects(
        self, request: ExternalAssetObjectsRequest, context: grpc.ServicerContext
    ) -> ExternalAssetObjectsReturn:
        """
        Get list of track IDs for external asset objects in a scene.

        Args:
            request: Contains scene_id to query
            context: gRPC context

        Returns:
            ExternalAssetObjectsReturn with list of track IDs
        """
        log.info(f"get_external_asset_objects")
        try:
            with self.get_backend(request.scene_id) as backend:
                return backend.get_external_asset_objects()
        except Exception:
            log.error(traceback.format_exc())
            raise

    def get_dynamic_objects(
        self, request: AvailableDynamicObjectsRequest, context: grpc.ServicerContext
    ) -> AvailableDynamicObjectsReturn:
        log.info(f"get_dynamic_objects for scene {request.scene_id}")
        try:
            with self.get_backend(request.scene_id) as backend:
                dynamic_objects = backend.get_dynamic_objects()
                return AvailableDynamicObjectsReturn(dynamic_objects=dynamic_objects)
        except (CacheFullError, NoSpareBackendsError) as e:
            log.warning(f"Cannot serve scene {request.scene_id}: {e}")
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))
            raise  # unreachable, but needed for mypy
        except Exception:
            log.error(traceback.format_exc())
            raise

    def edit_assets(self, request: EditAssetsRequest, context: grpc.ServicerContext) -> EditAssetsResponse:
        log.info(f"edit_assets for scene {request.scene_id}")
        try:
            with self.get_backend(request.scene_id) as backend:
                backend.edit_assets(replace_assets=list(request.replace), insert_assets=list(request.insert))
                return EditAssetsResponse(success=True, message="Success")
        except Exception as e:
            log.error(traceback.format_exc())
            return EditAssetsResponse(success=False, message=f"Failed to edit assets: {str(e)}")

    def restore_model_parameters(
        self, request: RestoreModelParametersRequest, context: grpc.ServicerContext
    ) -> grpc_types.Empty:
        log.info(f"restore_model_parameters for scene: {request.scene_id}")

        try:
            with self.get_backend(request.scene_id) as backend:
                backend.restore_model_parameters()
                log.info(f"Successfully restored model parameters for scene: {request.scene_id}")
        except Exception:
            log.error(traceback.format_exc())
            raise

        return grpc_types.Empty()

    @ScopedTimer()
    def render_rgb(self, request: RGBRenderRequest, context: grpc.ServicerContext) -> RGBRenderReturn:
        log.info("render_rgb")

        # Start timing for metrics collection
        render_start_time = time.perf_counter()

        try:
            with self.get_backend(request.scene_id) as backend:
                raster_f32, img_size = backend.render_camera_request(request, self.enable_editing_actors)

            if self.difix_enabled:
                # Flatten for difix processing
                flat_raster = rearrange(raster_f32, "h w c -> (h w) c")
                flat_raster = self.difix_model(flat_raster, img_size, False)
                # Unflatten difix output
                raster_f32 = rearrange(flat_raster, "(h w) c -> h w c", h=img_size[0])

            # inpaint the egocar mask
            if (egocar_mask := self.egocar_rig_bank.select_from_request(request)) is not None:
                raster_f32 = egocar_mask.overlay_on_image(raster_f32)

            # Calculate render time and collect metrics
            render_time_ms = (time.perf_counter() - render_start_time) * 1000.0
            self._metrics_manager.record_rgb_render_time(render_time_ms)

            # Mark frame boundary and send plots
            profile.mark_frame_boundary("rgb_render")

            return self.encode_image((raster_f32 * 255).clamp(0, 255).to(torch.uint8), request)
        except Exception as exc:
            status, message = _classify_render_exception(exc)
            if status == grpc.StatusCode.UNKNOWN:
                log.error(traceback.format_exc())
                raise
            log.warning(f"render_rgb failed with {status.name}: {message}")
            context.abort(status, message)
            raise  # unreachable, but needed for mypy

    @ScopedTimer()
    def batch_render_rgb(self, request: BatchRGBRenderRequest, context: grpc.ServicerContext) -> BatchRGBRenderReturn:
        """Render multiple cameras in a single call.

        This endpoint batches multiple render requests to reduce network overhead and optimize
        common setup operations. Each item in the batch can represent a different camera, a
        different frame/timestamp for the same camera, or any combination - as long as all
        items share the same scene_id.

        Architecture:
        - Per-camera rendering - sequential loop (designed for future parallelization)

        Args:
            request: Batch request containing a list of BatchRGBRenderRequestItem, each with:
                - camera_name: Identifier to match request with response
                - request: Full RGBRenderRequest (camera pose, intrinsics, frame timing, etc.)
                All items must use the same scene_id.
            context: gRPC service context for error handling and metadata.

        Returns:
            BatchRGBRenderReturn containing a list of BatchRGBRenderReturnItem, each with:
                - camera_name: Matches the request's camera_name
                - result: RGBRenderReturn with the rendered image (if success=True)
                - success: Whether this specific camera render succeeded
                - error_message: Error details if success=False

        Raises:
            grpc.StatusCode.INVALID_ARGUMENT: If items have different scene_ids.
            grpc.StatusCode.FAILED_PRECONDITION: If scene cannot be served (cache full, no backends).
        """
        log.info(f"batch_render_rgb: {len(request.items)} cameras")

        if not request.items:
            return BatchRGBRenderReturn(items=[])

        # All items must use the same scene
        scene_id = request.items[0].request.scene_id

        # Validate all items use the same scene_id
        mismatched_scenes = [
            (i, item.camera_name, item.request.scene_id)
            for i, item in enumerate(request.items)
            if item.request.scene_id != scene_id
        ]
        if mismatched_scenes:
            error_msg = (
                f"All cameras in batch must use the same scene_id. "
                f"Expected '{scene_id}', but found mismatches: "
                + ", ".join(f"item[{i}] '{name}' has '{sid}'" for i, name, sid in mismatched_scenes)
            )
            log.error(f"batch_render_rgb: {error_msg}")
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, error_msg)
            raise ValueError(error_msg)  # unreachable, but needed for mypy

        # Start timing for metrics collection
        render_start_time = time.perf_counter()

        results = []
        try:
            with self.get_backend(scene_id) as backend:
                # Per-camera rendering (sequential)
                for item in request.items:
                    result = self._render_single_camera_for_batch(backend, item, self.enable_editing_actors)
                    results.append(result)

            # Calculate render time and collect metrics
            render_time_ms = (time.perf_counter() - render_start_time) * 1000.0
            self._metrics_manager.record_batch_rgb_render_time(render_time_ms, len(results))

            # Mark frame boundary
            profile.mark_frame_boundary("batch_rgb_render")

            return BatchRGBRenderReturn(items=results)

        except Exception as exc:
            status, message = _classify_render_exception(exc)
            if status == grpc.StatusCode.UNKNOWN:
                log.error(traceback.format_exc())
                raise
            log.warning(f"batch_render_rgb failed with {status.name}: {message}")
            context.abort(status, message)
            raise  # unreachable, but needed for mypy

    def _render_single_camera_for_batch(
        self, backend: Backend, item: BatchRGBRenderRequestItem, enable_editing_actors: Optional[bool] = None
    ) -> BatchRGBRenderReturnItem:
        """Render a single camera within a batch context.

        Isolated method to enable future parallelization.
        """
        try:
            # Call existing render logic
            raster_f32, img_size = backend.render_camera_request(
                item.request, enable_editing_actors=enable_editing_actors
            )

            # Apply difix if enabled
            if self.difix_enabled:
                # Flatten for difix processing
                flat_raster = rearrange(raster_f32, "h w c -> (h w) c")
                flat_raster = self.difix_model(flat_raster, img_size, False)
                # Unflatten difix output
                raster_f32 = rearrange(flat_raster, "(h w) c -> h w c", h=img_size[0])

            # Apply ego mask if enabled
            if (egocar_mask := self.egocar_rig_bank.select_from_request(item.request)) is not None:
                raster_f32 = egocar_mask.overlay_on_image(raster_f32)

            # Encode the image
            encoded = self.encode_image((raster_f32 * 255).clamp(0, 255).to(torch.uint8), item.request)

            return BatchRGBRenderReturnItem(
                camera_name=item.camera_name,
                result=encoded,
                success=True,
            )
        except Exception as exc:
            status, message = _classify_render_exception(exc)
            error_message = f"{status.name}: {message}" if message else status.name
            log.error(f"Failed to render camera {item.camera_name}: {error_message}")
            return BatchRGBRenderReturnItem(
                camera_name=item.camera_name,
                success=False,
                error_message=error_message,
            )

    def render_lidar(self, request: LidarRenderRequest, context: grpc.ServicerContext) -> LidarRenderReturn:
        log.info(f"render_lidar")

        # Start timing for metrics collection
        render_start_time = time.perf_counter()

        try:
            with self.get_backend(request.scene_id) as backend:
                result = backend.render_lidar_request(request, self.enable_editing_actors)

            # Calculate render time and collect metrics
            render_time_ms = (time.perf_counter() - render_start_time) * 1000.0
            self._metrics_manager.record_lidar_render_time(render_time_ms)

            return result
        except (CacheFullError, NoSpareBackendsError) as e:
            log.warning(f"Cannot serve scene {request.scene_id}: {e}")
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))
            raise  # unreachable, but needed for mypy
        except Exception as exc:
            status, message = _classify_render_exception(exc)
            if status == grpc.StatusCode.UNKNOWN:
                log.error(traceback.format_exc())
                raise
            log.warning(f"render_lidar failed with {status.name}: {message}")
            context.abort(status, message)
            raise  # unreachable, but needed for mypy

    def shut_down(self, request: grpc_types.Empty, context: grpc.ServicerContext) -> grpc_types.Empty:
        log.info(f"shut_down")

        # Handle metrics shutdown
        self._metrics_manager.shutdown()

        context.add_callback(self._shut_down)
        return grpc_types.Empty()

    def _shut_down(self) -> None:
        if self.server is not None:
            self.server.stop(0)
        # self.server is None during tests as we can't shut down a pytest fixture

    def _register_local_scene(self, scene_id: str, local_uri: str) -> None:
        """Register a scene from a file:// URI or local path; make it available for rendering without download."""
        u = local_uri.strip()
        if u.lower().startswith("file://"):
            parsed = urlparse(u)
            path = Path(parsed.path)
        else:
            path = Path(u)
        resolved = path.resolve()
        if not resolved.exists():
            raise ValueError(f"{resolved} does not exist")
        if not resolved.is_file():
            raise ValueError(f"{resolved} is not a file")

        # Fast path: already registered with same path → idempotent, no lock or log
        if scene_id in self.artifacts:
            existing = self.artifacts[scene_id].source.resolve()
            if resolved == existing:
                return
            raise ValueError(
                f"Scene ID {scene_id!r} is already registered with a different artifact "
                f"(existing: {existing}, requested: {resolved})"
            )

        log.info(f"Registering local scene {scene_id} at {resolved}")

        def _do_register() -> None:
            if scene_id in self.artifacts:
                existing = self.artifacts[scene_id].source.resolve()
                if resolved != existing:
                    raise ValueError(
                        f"Scene ID {scene_id!r} is already registered with a different artifact "
                        f"(existing: {existing}, requested: {resolved})"
                    )
                return
            self._local_scene_ids.add(scene_id)
            try:
                self._load_scene_from_file(scene_id, resolved)
            except Exception:
                self._local_scene_ids.discard(scene_id)
                raise

        if self.scene_cache is not None:
            with self.scene_cache.lock_scene_path(scene_id):
                _do_register()
        else:
            with self._local_registration_lock:
                _do_register()

        log.info(f"Successfully registered local scene {scene_id}")

    def _download_scene(self, scene_url: str, scene_id: str) -> None:
        """Download a scene from a URI and make it available for rendering."""

        # Check if we already have this scene in artifacts
        if scene_id in self.artifacts:
            return

        try:
            log.info(f"Downloading scene {scene_id} from {scene_url}")

            # Delegate to the SceneCache to handle the download with proper locking
            # This will use our new concurrent download but keep the original locking pattern
            scene_path = self._download_with_cache(scene_id, scene_url)

            # Now load the scene from the downloaded file
            self._load_scene_from_file(scene_id, scene_path)

            # Cleanup artifacts that are no longer in the scene cache
            self._cleanup_artifacts()

            log.info(f"Successfully downloaded and loaded scene {scene_id}")
            return

        except Exception:
            log.exception(f"Failed to download scene {scene_id} from {scene_url}")
            raise
        return  # unreachable, but needed for mypy

    def _download_with_cache(self, scene_id: str, uri: str) -> Path:
        """Download a scene with proper cache locking."""

        assert self.scene_cache is not None
        assert self.downloader is not None

        # Fast (i.e no file lock) check if scene is already in
        if self.scene_cache.has_scene(scene_id):
            return self.scene_cache.get_scene_path(scene_id)

        with self.scene_cache.lock_scene_path(scene_id):
            # Check again in case the scene was added while we was waiting for the lock
            if self.scene_cache.has_scene(scene_id):
                return self.scene_cache.get_scene_path(scene_id)

            # Generate a unique temporary filename
            temp_dir = Path(tempfile.gettempdir())
            temp_path = temp_dir / f"nre_download_{scene_id}_{int(time.time())}_{os.getpid()}.tmp"

            try:
                # Download concurrently to the temporary file
                download_start = time.time()
                total_bytes_downloaded = self.downloader.download(uri, temp_path)
                download_time = time.time() - download_start
                download_speed = total_bytes_downloaded / (1024 * 1024 * download_time) if download_time > 0 else 0
                log.info(
                    f"Downloaded {total_bytes_downloaded / (1024 * 1024):.2f} MB in {download_time:.2f}s ({download_speed:.2f} MB/s)"
                )

                # Now use SceneCache to safely add the file to cache with proper locking
                cache_path = self.scene_cache.add_scene(scene_id, temp_path)
                return cache_path
            except Exception:
                # Clean up temp file on error
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise

        # unreachable, but needed for mypy
        assert False, "All paths in _download_with_cache should return within the with block"

    def _load_scene_from_file(self, scene_id: str, file_path: Path) -> None:
        """Load a scene from a downloaded file."""
        # We need to create an Artifact instance from the file
        artifact = Artifact(file_path)
        self.artifacts[scene_id] = artifact
        return

    def _cleanup_artifacts(self) -> None:
        """
        Remove artifacts that are no longer in the scene cache
        assuming cached scenes are a subset of artifacts.
        Local scenes (registered via x-nre-scene-url with a local path) are never removed.
        """

        assert self.scene_cache is not None

        # Gather artifacts to delete (exclude local scenes, which are not in the cache).
        # Read _local_scene_ids under lock to avoid race with _register_local_scene.
        with self._local_registration_lock:
            artifacts_to_delete = [
                scene_id
                for scene_id in self.artifacts.keys()
                if scene_id not in self._local_scene_ids and not self.scene_cache.has_scene(scene_id)
            ]

        log.info(f"Deleting {len(artifacts_to_delete)} artifacts: {artifacts_to_delete}")

        for scene_id in artifacts_to_delete:
            del self.artifacts[scene_id]

        return

    def _get_renderer(self) -> Backend.RendererType:
        if self.server_config is None:
            return Backend.RendererType.DEFAULT
        match self.server_config.renderer:
            case RendererBackend.NREND:
                return Backend.RendererType.NREND
            case RendererBackend.GSPLAT:
                return Backend.RendererType.GSPLAT
            case RendererBackend.DEFAULT:
                return Backend.RendererType.DEFAULT

    @contextlib.contextmanager
    def get_backend(self, artifact_id: str) -> Generator[Backend, None, None]:
        """
        A context manager that gets an available backend for the given artifact and parameters.

        It creates a new backend if one doesn't exist. If creation fails due to OOM,
        automatically evicts spare backends and retries until successful or no spares remain.
        """

        # Try to get an available backend
        backend = self.backend_cache.checkout(artifact_id)

        if backend is None:
            # No available backends, create a new one with OOM retry
            if artifact_id not in self.artifacts:
                raise KeyError(f"Scene {artifact_id=} not available.")

            with self.artifacts[artifact_id].temporary_cache() as artifact:
                log.info(f"Creating new backend for {artifact_id} from {artifact.source}")

                # Use the cache's put_with_retries for OOM-reactive backend creation
                try:
                    backend = self.backend_cache.put_with_retries(
                        key=artifact_id,
                        backend_factory=lambda: Backend.from_artifact(artifact, self._get_renderer()),
                        max_retries=5,
                        enable_eviction=True,
                    )
                except Exception:
                    # Log and re-raise any exceptions (OOM, CacheFullError, NoSpareBackendsError, etc.)
                    log.exception(f"Failed to create backend for {artifact_id}")
                    raise

        # At this point, backend must be set (either from checkout or from retry loop)
        assert backend is not None, "backend should never be None here"

        try:
            # Yield the backend to the caller (as a context manager)
            yield backend
        except torch.cuda.OutOfMemoryError:
            # On OOM during backend operation, try to free memory before re-raising
            # This helps the server recover from transient OOM conditions
            log.warning(f"OOM during operation on {artifact_id}, attempting memory cleanup")

            # First, try basic cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Additionally, evict spare backends to free GPU memory for future requests
            # This helps the server recover from OOM by reducing memory pressure
            evicted_count = 0
            while self.backend_cache.evict_one_spare():
                evicted_count += 1
            if evicted_count > 0:
                log.info(f"Evicted {evicted_count} spare backend(s) to free GPU memory after OOM")

            raise
        finally:
            # Always return the backend to the cache, even if an exception occurred
            # This prevents backends from getting stuck in the in_use set forever
            self.backend_cache.checkin(backend)


@click.command("serve-grpc")
@grpc_cli_options()
@cli.scopedtimer_cli_options(print_func=log.info)
@click.version_option(version=str(unpack_optional(get_version(), default="version-not-available")))
@click.pass_context
def serve_grpc(ctx: click.Context, **grpc_config_options) -> None:
    """Neural Rendering gRPC server"""

    # Instantiate config (this will trigger model_post_init and update CLI context if needed)
    grpc_config = GrpcServerConfig(**grpc_config_options)

    # Capture and log CLI settings
    collector = SettingsCollector.from_click_context(ctx, "serve-grpc")
    collector.log_settings(log)

    log.info(f"Renderer backend: {grpc_config.renderer.value}")

    log.info(
        f"Starting grpc server with {grpc_config.max_workers=} and cache_size={grpc_config.cache_size} (LRU with OOM retry)"
    )

    with ScopedTimer("serve_grpc"):
        scene_download_interceptor = SceneDownloadInterceptor()
        metrics_interceptor = MetricsInterceptor()

        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=grpc_config.max_workers),
            interceptors=[metrics_interceptor, scene_download_interceptor],
            # for lidar output
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),  # 50MB
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),  # 50MB
            ],
        )

        scene_cache = SceneCache(
            cache_dir=Path(grpc_config.download_cache_dir).expanduser(), max_size=grpc_config.download_cache_size
        )

        downloader = Downloader()

        service = SensorSimService(
            server,
            grpc_config.artifact_glob,
            ray_chunk_size=grpc_config.ray_chunk_size,
            egocar_hoods_dir=grpc_config.egocar_hood_dir,
            downloader=downloader,
            scene_cache=scene_cache,
            cache_size=grpc_config.cache_size,
            metrics_output_dir=grpc_config.metrics_output_dir,
            enable_editing_actors=grpc_config.enable_editing_actors,
            server_config=grpc_config,
        )

        scene_download_interceptor.set_service(service)
        metrics_interceptor.set_metrics_manager(service._metrics_manager)

        if grpc_config.test_scenes_are_valid:
            service.test_all_scenes()

        service.set_difix(
            grpc_config.enable_difix,
            grpc_config.difix_url,
            grpc_config.difix_cache,
            grpc_config.difix_model_filename,
            grpc_config.difix_resolution,
        )

        address = f"{grpc_config.host}:{grpc_config.port}"
        server.add_insecure_port(address)

        add_SensorsimServiceServicer_to_server(service, server)

        health_servicer = health.HealthServicer()
        health_server: Optional[grpc.Server] = None

        # Health check serving:
        # - Default/legacy: register health service on the main server.
        # - If --health-port is set and differs from --port: start a dedicated health-only gRPC server.
        if grpc_config.health_port is None or grpc_config.health_port == grpc_config.port:
            health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
            log.info(f"Serving on {address} (health on same port)")
        else:
            health_address = f"{grpc_config.host}:{grpc_config.health_port}"
            health_server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
            health_pb2_grpc.add_HealthServicer_to_server(health_servicer, health_server)
            health_server.add_insecure_port(health_address)
            log.info(f"Serving main gRPC on {address} and health on {health_address}")
            health_server.start()

        # Start serving
        health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
        server.start()

        try:
            server.wait_for_termination()
        finally:
            # Cleanup: mark health as down and stop health server if separate
            if health_server is not None:
                try:
                    health_servicer.set("", health_pb2.HealthCheckResponse.NOT_SERVING)
                except Exception:
                    log.exception("Failed to update health status to NOT_SERVING during shutdown")
                health_server.stop(0)

    ScopedTimer.print_summary()
