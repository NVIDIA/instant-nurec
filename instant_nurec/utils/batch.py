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

import queue

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, List, Self, Sequence, Tuple, TypeAlias, TypeVar, Union, cast

import torch

from instant_nurec.utils.geometry import se3pose_from_matrix
from instant_nurec.utils.sensors.ray_gen import (
    image_points_to_world_rays_shutter_pose,
)
from ncore.data import ConcreteCameraModelParametersUnion
from ncore.impl.common.transformations import PoseInterpolator
from ncore.sensors import (
    CameraModel,
    FThetaCameraModel,
)
from instant_nurec.utils.misc import assert_same_type, collate_fn, unpack_optional
from instant_nurec.utils.sensors import SensorModelComputations
from instant_nurec.utils.sensors.ncore_sensors_converters import (
    CameraModelConverter,
    DynamicPose,
    Pose,
)
from instant_nurec.utils.types import (
    CuboidTracksDataPack,
    RigTrajectories,
)


ConcreteCameraModelsUnion: TypeAlias = FThetaCameraModel
ConcreteSensorModelParametersUnion: TypeAlias = ConcreteCameraModelParametersUnion


def generate_grid_2d_indices(
    resolution: Tuple[int, int], device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Computes (x, y) pixel coordinates for all pixels in the sensor frame.

    Args:
        resolution: (w, h) sensor width and height.

    Returns:
        torch.Tensor: A (N, 2) tensor with N = width * height, zero-based
            indices x in [0, w-1], y in [0, h-1]. Predict only ever needs
            "xy" order; the "yx" branch was dropped.
    """
    w, h = resolution
    sensor_pixels_x, sensor_pixels_y = torch.meshgrid(
        torch.arange(w, dtype=torch.int16, device=device),
        torch.arange(h, dtype=torch.int16, device=device),
        indexing="xy",
    )
    return torch.stack([sensor_pixels_x.flatten(), sensor_pixels_y.flatten()], dim=1)


@dataclass(kw_only=True, slots=True)
class RenderingData:
    """Camera data for rendering.

    Note: The `rays`, `poses_tquat_startend` and the underlying scene representation (e.g. 3D Gaussians)
    should be in the same coordinate system. For example, the most common case is that it carries `rays`
    in the scene space, the `poses_tquat_startend` is the transform from sensor to scene space, and the
    underlying scene representation (e.g. 3D Gaussians) is also in the scene space.

    The fields are:

    - rays: Ray origins and directions for the camera. [Tensor[float32]]. (B, height, width, 6).
    - sensor_model_parameters: List of camera model parameters. List of length B.
    - poses_tquat_startend: Start and end poses of the frame. [Tensor[float32]]. (B, 2, 7)
    - timestamps_startend_us: Start and end timestamps of the frame in microseconds. [Tensor[int64]]. (B, 2)
    """

    rays: torch.Tensor
    sensor_model_parameters: list[ConcreteSensorModelParametersUnion]
    poses_tquat_startend: torch.Tensor  # (B, 2, 7)
    timestamps_startend_us: torch.Tensor  # (B, 2) - kept on GPU for GPU operations
    rays_timestamps_us: torch.Tensor | None = None  # (B, height, width, 1)
    timestamps_startend_us_cpu: torch.Tensor  # (B, 2) - cpu copy to avoid .item() calls
    _distance_to_depth_scale: torch.Tensor | None = None  # [Tensor[float32]] (B, height, width, 1)

    def __post_init__(self):
        B = self.rays.shape[0]
        assert self.rays.ndim == 4 and self.rays.shape[3] == 6, "Rays must be a 4D tensor (B, height, width, 6)"
        assert len(self.sensor_model_parameters) == B, "Model parameters must be a list of length B"
        assert self.poses_tquat_startend.shape == (B, 2, 7), "Poses must be a 3D tensor (B, 2, 7)"
        assert self.timestamps_startend_us.shape == (B, 2), "Timestamps must be a 2D tensor (B, 2)"
        assert self.timestamps_startend_us_cpu.shape == (B, 2), "CPU timestamps must be a 2D tensor (B, 2)"
        if self.rays_timestamps_us is not None:
            assert self.rays_timestamps_us.ndim == 4 and self.rays_timestamps_us.shape[3] == 1, (
                f"Rays timestamps must be a 4D tensor (B, height, width, 1), but got {self.rays_timestamps_us.shape}"
            )
        if self._distance_to_depth_scale is not None:
            assert (
                self._distance_to_depth_scale.ndim == 4
                and self._distance_to_depth_scale.shape[:3] == self.rays.shape[:3]
            ), (
                f"Depth to distance scale must be a 4D tensor (B, height, width, 1) and match the rays shape (B, height, width, 6), but got {self._distance_to_depth_scale.shape}"
            )

    @property
    def b(self) -> int:
        return self.rays.shape[0]

    @property
    @torch.autocast(device_type="cuda", enabled=False)
    def distance_to_depth_scale(self) -> torch.Tensor:
        """Compute the multiplication factor to convert depth to distance.
        Shape: (B, height, width, 1)"""
        if self._distance_to_depth_scale is None:
            scales: list[torch.Tensor] = []
            for bidx in range(self.b):
                sensor_model = CameraModel.from_parameters(
                    cast(ConcreteCameraModelParametersUnion, self.sensor_model_parameters[bidx]),
                    device=self.rays.device,
                )
                width, height = sensor_model.resolution.tolist()
                elements = generate_grid_2d_indices((width, height), device=self.rays.device)
                sensor_rays = sensor_model.pixels_to_camera_rays(elements).reshape(height, width, 3)
                scales.append(sensor_rays[..., 2].reshape(height, width))
            self._distance_to_depth_scale = torch.stack(scales, dim=0).unsqueeze(-1)
        return self._distance_to_depth_scale

    @classmethod
    def collate_fn(
        cls,
        seq: List[Self],
        device: torch.device = torch.device("cpu"),
    ) -> Self:
        if any(item.rays_timestamps_us is None for item in seq):
            rays_timestamps_us = None
        else:
            rays_timestamps_us = collate_fn([item.rays_timestamps_us for item in seq], device)
        # Keep GPU version on target device, CPU copy on CPU
        timestamps_startend_us = collate_fn([item.timestamps_startend_us for item in seq], device)
        timestamps_startend_us_cpu = collate_fn([item.timestamps_startend_us_cpu for item in seq], torch.device("cpu"))
        if any(item._distance_to_depth_scale is None for item in seq):
            _distance_to_depth_scale = None
        else:
            _distance_to_depth_scale = collate_fn([item._distance_to_depth_scale for item in seq], device)
        return cls(
            rays=collate_fn([item.rays for item in seq], device),
            sensor_model_parameters=[p for item in seq for p in item.sensor_model_parameters],
            poses_tquat_startend=collate_fn([item.poses_tquat_startend for item in seq], device),
            timestamps_startend_us=timestamps_startend_us,
            rays_timestamps_us=rays_timestamps_us,
            timestamps_startend_us_cpu=timestamps_startend_us_cpu,
            _distance_to_depth_scale=_distance_to_depth_scale,
        )

    def to(self, *args, **kwargs) -> Self:
        return self.__class__(
            rays=self.rays.to(*args, **kwargs),
            sensor_model_parameters=self.sensor_model_parameters,
            poses_tquat_startend=self.poses_tquat_startend.to(*args, **kwargs),
            timestamps_startend_us=self.timestamps_startend_us.to(*args, **kwargs),
            rays_timestamps_us=self.rays_timestamps_us.to(*args, **kwargs)
            if self.rays_timestamps_us is not None
            else None,
            # CPU copy stays on CPU - don't move it
            timestamps_startend_us_cpu=self.timestamps_startend_us_cpu,
            _distance_to_depth_scale=self._distance_to_depth_scale.to(*args, **kwargs)
            if self._distance_to_depth_scale is not None
            else None,
        )

    def __getitem__(self, item: Union[int, slice]) -> Self:
        """Allows indexing into the dataclass to get a subset of the data."""
        if isinstance(item, int):
            item = slice(item, item + 1)

        return self.__class__(
            rays=self.rays[item],
            sensor_model_parameters=self.sensor_model_parameters[item],
            poses_tquat_startend=self.poses_tquat_startend[item],
            timestamps_startend_us=self.timestamps_startend_us[item],
            rays_timestamps_us=self.rays_timestamps_us[item] if self.rays_timestamps_us is not None else None,
            timestamps_startend_us_cpu=self.timestamps_startend_us_cpu[item],
            _distance_to_depth_scale=self._distance_to_depth_scale[item]
            if self._distance_to_depth_scale is not None
            else None,
        )


@dataclass(kw_only=True, slots=True)
class FrameMeta:
    """Metadata for a camera frame.

    The fields are:
    - unique_sensor_idx: Index of the sensor that captured the frame. int32
    - unique_frame_idx: Unique index (among all sensors) for the frame. int32
    """

    unique_sensor_idx: int
    unique_frame_idx: int

    # Tensor version that will be used for cuda, populated automatically
    unique_frame_idx_tensor: torch.Tensor | None = None

    # Stringified version of the unique sensor index, used to index sensor_models as a nn.ModuleDict.
    unique_sensor_idx_str: str | None = None

    def __post_init__(self):
        if self.unique_frame_idx_tensor is None:
            self.unique_frame_idx_tensor = (
                torch.tensor([self.unique_frame_idx], dtype=torch.int32) if self.unique_frame_idx != -1 else None
            )
        if self.unique_sensor_idx_str is None:
            self.unique_sensor_idx_str = str(self.unique_sensor_idx)

    @classmethod
    def collate_fn(
        cls,
        seq: List[Self],
        device: torch.device = torch.device("cpu"),
    ) -> List[Self]:
        return [s.to(device) for s in seq]

    def to(self, *args, **kwargs) -> Self:
        return self.__class__(
            unique_sensor_idx=self.unique_sensor_idx,
            unique_frame_idx=self.unique_frame_idx,
            unique_frame_idx_tensor=self.unique_frame_idx_tensor.to(*args, **kwargs)
            if self.unique_frame_idx_tensor is not None
            else None,
            unique_sensor_idx_str=self.unique_sensor_idx_str,
        )


@dataclass(kw_only=True, slots=True)
class CameraFrameLabels:
    """Labels for a camera frame.

    The fields are:
    - rgb: Optional. RGB value within [0, 1]. Default is None. [Tensor[float32]]. (B, height, width, 3).
    """

    rgb: torch.Tensor | None = None

    def __post_init__(self):
        if self.rgb is not None:
            assert self.rgb.ndim == 4 and self.rgb.shape[3] == 3, "RGB must be a 4D tensor (B, height, width, 3)"
            assert self.rgb.dtype == torch.float32, "RGB must be a float32 tensor"

    @classmethod
    def collate_fn(
        cls,
        seq: List[Self],
        device: torch.device = torch.device("cpu"),
    ) -> Self:
        return cls(
            rgb=collate_fn([item.rgb for item in seq], device),
        )

    def to(self, *args, **kwargs) -> Self:
        return self.__class__(
            rgb=self.rgb.to(*args, **kwargs) if self.rgb is not None else None,
        )

    def __getitem__(self, item: Union[int, slice, torch.Tensor]) -> Self:
        """Allows indexing into the dataclass to get a subset of the data."""
        if isinstance(item, int):
            item = slice(item, item + 1)

        return self.__class__(
            rgb=self.rgb[item] if self.rgb is not None else None,
        )


@dataclass(kw_only=True, slots=True)
class DataBatch:
    """Data for camera frames."""

    @dataclass(kw_only=True, slots=True)
    class Camera:
        """Data for a camera frame. Includes the frame meta and labels."""

        meta: List[FrameMeta]
        labels: CameraFrameLabels

        @property
        def b(self) -> int:
            return len(self.meta)

        @classmethod
        def collate_fn(
            cls,
            seq: List[DataBatch.Camera],
            device: torch.device = torch.device("cpu"),
        ) -> DataBatch.Camera:
            return cls(
                meta=FrameMeta.collate_fn([meta for item in seq for meta in item.meta], device),
                labels=CameraFrameLabels.collate_fn([item.labels for item in seq], device),
            )

        def to(self, *args, **kwargs) -> Self:
            return self.__class__(
                meta=[meta.to(*args, **kwargs) for meta in self.meta],
                labels=self.labels.to(*args, **kwargs),
            )

        def __getitem__(self, item: Union[int, slice]) -> Self:
            """Allows indexing into the dataclass to get a subset of the data."""
            if isinstance(item, int):
                item = slice(item, item + 1)

            return self.__class__(meta=self.meta[item], labels=self.labels[item])

    camera: Camera | None = None

    @classmethod
    def collate_fn(
        cls,
        seq: List[DataBatch],
        device: torch.device = torch.device("cpu"),
    ) -> DataBatch:
        if any(item.camera is None for item in seq):
            camera = None
        else:
            camera = DataBatch.Camera.collate_fn([unpack_optional(item.camera) for item in seq], device)

        return cls(camera=camera)

    def to(self, *args, **kwargs) -> Self:
        return self.__class__(
            camera=self.camera.to(*args, **kwargs) if self.camera is not None else None,
        )


@dataclass(kw_only=True, slots=True)
class RenderingBatch:
    """
    A RenderingBatch is a collection of camera RenderingData.
    """

    camera: RenderingData | None = None

    @classmethod
    def collate_fn(
        cls,
        seq: List[RenderingBatch],
        device: torch.device = torch.device("cpu"),
    ) -> RenderingBatch:
        if any(item.camera is None for item in seq):
            camera = None
        else:
            camera = RenderingData.collate_fn([unpack_optional(item.camera) for item in seq], device)

        return cls(camera=camera)

    def to(self, *args, **kwargs) -> Self:
        return self.__class__(
            camera=self.camera.to(*args, **kwargs) if self.camera is not None else None,
        )


@dataclass(kw_only=True, slots=True)
class DataAndRenderingBatch:
    """
    A DataAndRenderingBatch is a compounding type used for training and validation.

    The fields are:
    - data: The information from the dataset that are required for training and validation. [DataBatch]
    - rendering: The information that will be fed into the renderer. [RenderingBatch]

    Note the design of the workflow for training and validation is:

    1. None -> [dataloader] -> DataAndRenderingBatch(data: DataBatch, rendering: RenderingBatch | None)
    2. DataAndRenderingBatch -> [pre-processing] -> DataAndRenderingBatch(data: DataBatch, rendering: RenderingBatch)
    3. RenderingBatch  -> [renderer] -> RenderingOutput
    4. (DataAndRenderingBatch, RenderingOutput) -> [loss] -> loss

    For inference, the workflow is:

    rendering: RenderingBatch -> [renderer] -> RenderingOutput
    """

    # DataBatch is always provided by the dataloader, whereas the RenderingBatch is optional.
    # In some cases we would like to compute the rendering data in the dataloader.
    # For example, for the Difix sampler, or to hide the latency when not performing camera pose optimization.
    data: DataBatch
    rendering: RenderingBatch | None = None

    @classmethod
    def collate_fn(
        cls,
        seq: List[DataAndRenderingBatch],
        device: torch.device = torch.device("cpu"),
    ) -> DataAndRenderingBatch:
        if all(seq_isnone := [item.rendering is None for item in seq]):
            rendering = None
        elif not any(seq_isnone):
            rendering = RenderingBatch.collate_fn(cast(list[RenderingBatch], [item.rendering for item in seq]), device)
        else:
            raise ValueError("DataAndRenderingBatch.collate_fn: All items must have either a rendering or no rendering")

        return cls(
            data=DataBatch.collate_fn([item.data for item in seq], device),
            rendering=rendering,
        )

    def to(self, *args, **kwargs) -> Self:
        return self.__class__(
            data=self.data.to(*args, **kwargs),
            rendering=self.rendering.to(*args, **kwargs) if self.rendering is not None else None,
        )

    def pin_memory(self):
        """
        Enable pinned memory for async data transfer in PyTorch.

        When using a DataLoader with `pin_memory=True`, PyTorch calls
        `pin_memory()` on each object it retrieves from the dataset.
        Implementing this method ensures that the returned object is moved
        into pinned (page-locked) memory.
        """
        q = queue.Queue()
        q.put(self)
        while not q.empty():
            dataclass_obj = q.get()
            for field in fields(dataclass_obj):
                f = getattr(dataclass_obj, field.name)
                pin_memory_attr = getattr(f, "pin_memory", None)
                if callable(pin_memory_attr):
                    setattr(dataclass_obj, field.name, pin_memory_attr())
                elif is_dataclass(f):
                    q.put(f)
        return self


class CameraFreePoseViewGeometry(torch.nn.Module):
    """
    FreePoseViewGeometry for camera sensors. It stores raw (un-subsampled) camera extrinsics and intrinsics for all frames & views. [Exist for the new batch format]
    """

    def __init__(
        self,
        T_sensor_world_startend_allviews: torch.Tensor,  # (n_frames, 2, 4, 4)
        timestamps_startend_us_allviews: torch.Tensor,  # (n_frames, 2)
        sensor_models: dict[str, CameraModel],  # mapping from unique_sensor_idx to CameraModel
        # Maps a sensor id to a range of unique frame indices that can be used to recover the slices of
        # T_sensor_world_startend_allviews and timestamps_startend_us_allviews belonging to a specific sensor.
        sensor_ids_to_frame_range: dict[str, range],
    ):
        super().__init__()
        assert T_sensor_world_startend_allviews.shape[0] == timestamps_startend_us_allviews.shape[0], (
            "T_sensor_world_startend_allviews and timestamps_startend_us_allviews must have the same number of frames"
        )
        self.T_sensor_world_startend_allviews = torch.nn.Buffer(T_sensor_world_startend_allviews, persistent=False)
        self.timestamps_startend_us_allviews = torch.nn.Buffer(timestamps_startend_us_allviews, persistent=False)
        self.timestamps_startend_us_allviews_cpu = timestamps_startend_us_allviews.clone().cpu()
        self.sensor_models = torch.nn.ModuleDict(sensor_models)
        self.sensor_ids_to_frame_range = sensor_ids_to_frame_range

        # Cache per-sensor data for `to_rendering_data`: ncore `parameters`
        # and `sensorlib_parameters` from `CameraModelConverter` (world rays
        # use these with `image_points_to_world_rays_shutter_pose`).
        self.cached_sensor_params: dict[str, dict] = {
            k: {"parameters": None, "sensorlib_parameters": None} for k in sensor_models.keys()
        }
        self.cached_sensor_subsample: dict[int, ConcreteCameraModelsUnion] = {}

    @staticmethod
    def from_rig_trajectories(rig_trajectories: RigTrajectories) -> CameraFreePoseViewGeometry:
        """
        Initialize a `CameraFreePoseViewGeometry` from a `NCOREDataSource`.
        """
        camera_calibrations = rig_trajectories.camera_calibrations
        world_to_scene = rig_trajectories.world_to_scene

        # collect all extrinsics and intrinsics
        T_sensor_world_startend_allviews = []
        timestamps_startend_us_allviews = []
        sensor_models: dict[str, CameraModel] = {}
        sensor_ids_to_frame_range: dict[str, range] = {}  # camera_id -> range of unique frame indices
        unique_frame_start_index = 0

        for sensor_id, camera_calibration in camera_calibrations.items():
            unique_sensor_idx = camera_calibration.unique_sensor_idx if camera_calibration.unique_sensor_idx >= 0 else 0

            sensor_model = CameraModel.from_parameters(
                camera_calibration.camera_model_parameters, device=torch.device("cpu"), dtype=torch.float32
            )
            sensor_models[str(unique_sensor_idx)] = sensor_model

            candidate_trajectories = [
                r for r in rig_trajectories.rig_trajectories if r.sequence_id == camera_calibration.sequence_id
            ]
            assert len(candidate_trajectories) == 1, (
                f"Expected exactly one rig trajectory to match the sequence with name {camera_calibration.sequence_id}"
            )
            rig_trajectory = candidate_trajectories[0]

            pose_interpolator = PoseInterpolator(
                rig_trajectory.T_rig_worlds.cpu(), rig_trajectory.T_rig_world_timestamps_us.cpu()
            )

            timestamps_us = rig_trajectory.cameras_frame_timestamps_us[sensor_id]
            assert timestamps_us.ndim == 2 and timestamps_us.shape[1] == 2, (
                "timestamps_us is expected to be a 2D tensor with shape (n_frames, 2)"
            )
            timestamps_startend_us_allviews.append(timestamps_us)

            T_sensor_rig_np = camera_calibration.T_sensor_rig.cpu().numpy()
            for timestamp_us in timestamps_us:
                T_rig_world_startend = pose_interpolator.interpolate_to_timestamps(timestamp_us.cpu())
                T_sensor_world_startend = world_to_scene.transform_poses(T_rig_world_startend @ T_sensor_rig_np)
                T_sensor_world_startend_allviews.append(torch.from_numpy(T_sensor_world_startend).to(torch.float32))

            # Build map from sensor id to a range of unique frame indices that can be used to recover the slices of
            # T_sensor_world_startend_allviews and timestamps_startend_us_allviews belonging to a specific sensor.
            num_frames = timestamps_us.shape[0]
            sensor_ids_to_frame_range[sensor_id] = range(
                unique_frame_start_index, unique_frame_start_index + num_frames
            )
            unique_frame_start_index += num_frames

        return CameraFreePoseViewGeometry(
            T_sensor_world_startend_allviews=torch.stack(T_sensor_world_startend_allviews, dim=0),
            timestamps_startend_us_allviews=torch.cat(timestamps_startend_us_allviews, dim=0),
            sensor_models=sensor_models,
            sensor_ids_to_frame_range=sensor_ids_to_frame_range,
        )

    def get_sensor_model(self, frame_meta: FrameMeta) -> CameraModel:
        """
        Getter to request sensor model for a given sensor index.

        Args:
            frame_meta: The frame metadata. See `FrameMeta` for more details.

        Returns:
            The sensor model.
        """
        unique_sensor_idx = frame_meta.unique_sensor_idx if frame_meta.unique_sensor_idx >= 0 else 0
        return cast(ConcreteCameraModelsUnion, self.sensor_models[str(unique_sensor_idx)])

    def get_poses_and_timestamps_startend(
        self,
        meta: FrameMeta,
    ) -> SensorModelComputations.PosesAndTimestampsStartendReturn:
        """
        Get the poses and timestamps for a given frame.
        """
        return SensorModelComputations.get_poses_and_timestamps_startend(
            T_sensor_world_startend_allviews=self.T_sensor_world_startend_allviews,
            timestamps_startend_us_allviews=self.timestamps_startend_us_allviews,
            timestamps_startend_us_allviews_cpu=self.timestamps_startend_us_allviews_cpu,
            unique_frame_idx=meta.unique_frame_idx,
            unique_frame_idx_tensor=unpack_optional(meta.unique_frame_idx_tensor),
        )

    def to_rendering_data(self, data_batch: DataBatch.Camera) -> RenderingData:
        """
        Convert a `DataBatch.Camera` to a `RenderingData`.

        This function will compute the followings and store them in the `RenderingData` object:
        * rays in world space via sensorlib (`sensorlib_parameters` from `CameraModelConverter` and `image_points_to_world_rays_shutter_pose`).
        * startend poses and timestamps for the given frame in the subsampled domain.
        * sensor model in the subsampled domain.
        """
        if data_batch.b == 1:
            return self._to_rendering_data_single_batch(data_batch)
        rendering_data_list: list[RenderingData] = []
        for bidx in range(data_batch.b):
            rendering_data_list.append(self._to_rendering_data_single_batch(data_batch[bidx]))
        return RenderingData.collate_fn(rendering_data_list, device=rendering_data_list[0].rays.device)

    def _to_rendering_data_single_batch(self, data_batch: DataBatch.Camera) -> RenderingData:
        """
        Internal single batch version of `to_rendering_data`.
        """
        assert data_batch.b == 1, "Only one frame is supported"
        meta = data_batch.meta[0]

        pose_and_timestamps_startend_return = self.get_poses_and_timestamps_startend(meta)

        T_sensor_world_startend = pose_and_timestamps_startend_return.T_sensor_world_startend
        timestamps_startend_us_gpu = pose_and_timestamps_startend_return.timestamps_startend_us_gpu
        timestamps_startend_us_cpu = pose_and_timestamps_startend_return.timestamps_startend_us_cpu

        if meta.unique_sensor_idx in self.cached_sensor_subsample:
            sensor_model = self.cached_sensor_subsample[meta.unique_sensor_idx]
        else:
            sensor_model = cast(ConcreteCameraModelsUnion, self.get_sensor_model(meta))
            self.cached_sensor_subsample[meta.unique_sensor_idx] = sensor_model

        unique_sensor_idx = meta.unique_sensor_idx if meta.unique_sensor_idx >= 0 else 0
        cached_sensor_params = self.cached_sensor_params[str(unique_sensor_idx)]
        if cached_sensor_params["sensorlib_parameters"] is not None:
            sensor_model_parameters = cached_sensor_params["parameters"]
            sensorlib_parameters = cached_sensor_params["sensorlib_parameters"]
        else:
            sensor_model_parameters = sensor_model.get_parameters()
            sensorlib_parameters = CameraModelConverter.convert(sensor_model, device=T_sensor_world_startend.device)
            self.cached_sensor_params[str(unique_sensor_idx)] = {
                "parameters": sensor_model_parameters,
                "sensorlib_parameters": sensorlib_parameters,
            }

        translations, rotations = se3pose_from_matrix(T_sensor_world_startend)

        poses_tquat_startend = torch.cat([translations, rotations], dim=1)
        poses_tquat_startend = poses_tquat_startend.unsqueeze(0)

        timestamps_cpu = timestamps_startend_us_cpu.flatten()

        (world_rays, timestamps_us, _, _) = image_points_to_world_rays_shutter_pose(
            image_points=None,
            projection=sensorlib_parameters.projection,
            external_distortion=sensorlib_parameters.external_distortion,
            resolution=sensorlib_parameters.resolution,
            shutter_type=sensorlib_parameters.shutter_type,
            dynamic_pose=DynamicPose(
                start_pose=Pose(translation=translations[0], rotation=rotations[0]),
                end_pose=Pose(translation=translations[1], rotation=rotations[1]),
            ),
            start_timestamp_us=int(timestamps_cpu[0].item()),
            end_timestamp_us=int(timestamps_cpu[1].item()),
            return_timestamps=True,
        )

        rays = unpack_optional(world_rays)  # camera rays are in scene space
        rays = rays.reshape(sensorlib_parameters.resolution[1], sensorlib_parameters.resolution[0], 6)
        timestamps = unpack_optional(timestamps_us)
        timestamps = timestamps.reshape(sensorlib_parameters.resolution[1], sensorlib_parameters.resolution[0], 1)

        return RenderingData(
            rays=rays.unsqueeze(0),
            # FIXME: This will be on cpu in numpy array.
            sensor_model_parameters=[sensor_model_parameters],
            poses_tquat_startend=poses_tquat_startend,
            timestamps_startend_us=timestamps_startend_us_gpu,
            rays_timestamps_us=timestamps.unsqueeze(0),
            timestamps_startend_us_cpu=timestamps_startend_us_cpu,
        )


@dataclass(slots=False, kw_only=True)
class InstantNuRecDataBatch:
    """
    A batch that contains (B,) groups of context DataBatch(es).
    We also precompute RenderingBatch altogether to hide latency for data preprocessing.

    Contains
        - context: list of context images (A DataAndRenderingBatch)
        - cuboid_tracks: list of cuboid tracks
        - context_rig: list of rig trajectories for context (intrinsics subsampled already to context images)
        - meta: list of dictionaries of metadata for each batch, e.g. sequence_id, ncore_json_path, etc.
    """

    context: list[DataAndRenderingBatch]
    cuboid_tracks: list[CuboidTracksDataPack] | None = None
    context_rig: list[RigTrajectories] | None = None
    meta: list[dict[str, Any]] | None = None

    def __post_init__(self):
        if self.cuboid_tracks is not None:
            assert len(self.context) == len(self.cuboid_tracks), "Number of context and cuboid tracks must match"
        if self.context_rig is not None:
            assert len(self.context) == len(self.context_rig), "Number of context and context_rig must match"

    def __getitem__(self, item: Union[int, slice]) -> Self:
        """Allows indexing into the dataclass to get a subset of the data."""
        if isinstance(item, int):
            item = slice(item, item + 1)

        return self.__class__(
            context=self.context[item],
            cuboid_tracks=self.cuboid_tracks[item] if self.cuboid_tracks is not None else None,
            context_rig=self.context_rig[item] if self.context_rig is not None else None,
            meta=self.meta[item] if self.meta is not None else None,
        )

    def __len__(self) -> int:
        return len(self.context)

    def to(self, device: torch.device, **kwargs) -> Self:
        """Move all dataclass-aware fields to ``device``."""

        def _move_list(items, attr: str | None = None):
            if items is None:
                return None
            out = []
            for item in items:
                if hasattr(item, "to_device"):
                    out.append(item.to_device(device))
                elif hasattr(item, "to"):
                    out.append(item.to(device, **kwargs))
                else:
                    out.append(item)
            return out

        return self.__class__(
            context=_move_list(self.context),
            cuboid_tracks=_move_list(self.cuboid_tracks),
            context_rig=_move_list(self.context_rig),
            meta=self.meta,
        )

    @torch.autocast(device_type="cuda", enabled=False)
    def maybe_compute_rendering_data(self, device: torch.device):
        """Populates self.context[...].data.rendering unless already present."""

        if self.context_rig is None:
            return
        for data, rig in zip(self.context, self.context_rig):
            # Do not re-compute if rendering data already exists.
            if data.rendering is not None:
                continue
            camera_rendering_data = (
                CameraFreePoseViewGeometry.from_rig_trajectories(rig)
                .to(device=device)
                .to_rendering_data(data.data.camera.to(device))
                if data.data.camera is not None
                else None
            )
            data.rendering = RenderingBatch(camera=camera_rendering_data)

    @classmethod
    def collate_fn(
        cls,
        seq: Sequence[InstantNuRecDataBatch],
        device: torch.device = torch.device("cpu"),
    ) -> Self:
        T = TypeVar("T")

        def _collate_vals(*vals: list[T] | None) -> list[T] | None:
            assert len(vals) > 0, "At least one value must be provided"
            if any(val is None for val in vals):
                return None

            vals_arr = sum([unpack_optional(val) for val in vals], [])
            assert_same_type(vals_arr)

            for vi, v in enumerate(vals_arr):
                if isinstance(v, DataBatch):
                    vals_arr[vi] = cast(T, v.to(device))

                elif hasattr(v, "to_device"):
                    vals_arr[vi] = cast(T, v.to_device(device))

            return vals_arr

        return cls(
            context=unpack_optional(_collate_vals(*[batch.context for batch in seq])),
            cuboid_tracks=_collate_vals(*[batch.cuboid_tracks for batch in seq]),
            context_rig=_collate_vals(*[batch.context_rig for batch in seq]),
            meta=_collate_vals(*[batch.meta for batch in seq]),
        )
