# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import gzip
import pickle

from typing import Any, Optional, cast

import msgpack
import numpy as np
import torch

from libs.nrend.renderer import Renderer  # type: ignore
from libs.vren.interface import to_vren  # type: ignore
from ncore.data import (
    ConcreteCameraModelParametersUnion,
    ConcreteLidarModelParametersUnion,
)
from nre.utils.misc import get_union_types


def _prepare_sensor_model(
    frames_sensor_model: Optional[ConcreteCameraModelParametersUnion | ConcreteLidarModelParametersUnion],
    renderer_settings: dict,
    device: torch.device,
) -> Any:
    """Helper function to prepare sensor model (camera or lidar) for rendering.

    Args:
        frames_sensor_model: The sensor model parameters (camera or lidar)
        renderer_settings: The renderer settings dictionary
        device: The torch device to use

    Returns:
        Prepared sensor model or None if frames_sensor_model is None
    """
    if frames_sensor_model is None:
        return None

    if isinstance(frames_sensor_model, get_union_types(ConcreteCameraModelParametersUnion)):
        camera_model = cast(ConcreteCameraModelParametersUnion, frames_sensor_model)
        return Renderer.prepare_and_cache_camera_model(camera_model, device)
    elif isinstance(frames_sensor_model, get_union_types(ConcreteLidarModelParametersUnion)):
        lidar_model = cast(ConcreteLidarModelParametersUnion, frames_sensor_model)
        lidar_tiling = renderer_settings.get("tiling", {}).get("lidar", {})
        return Renderer.prepare_and_cache_lidar_model(
            lidar_model,
            n_bins_elevation=lidar_tiling.get("n_bins_elevation", 16),
            max_pts_per_tile=lidar_tiling.get("tile_size_elevation", 16) * lidar_tiling.get("tile_size_azimuth", 16),
            device=device,
        ).parameters
    else:
        raise ValueError(f"Unsupported sensor model: {type(frames_sensor_model).__name__}")


class RendererTestCase:
    """
    A class to create a test case : a model and instances of the inputs / outputs of the render function

    Members:
        model : dictionary containing the serialized model data
        renderer : dictionnary containing the renderer settings
        frame_id : unique frame identifier
        frame_width : width of the frame
        frame_height : height of the frame
        frame_start_timestamp: timestamp of the frame at the begining of the capture
        frame_end_timestamp: timestamp of the frame at the end of the capture
        rays_origin : contiguous float tensor containing the 3d position of the rays origin [HxWx3]
        rays_direction : contiguous float tensor containing the 3d normalized rays direction [HxWx3]
        rays_timestamp : contiguous float tensor containing the rays timestamp (in [frame_start_timestamp, frame_end_timestamp]) [HxWx1]
        frames_sensor_model : variant describing the sensor model
        frames_sensor_ids : contiguous int tensor containing the frame sensor ids (sensor id, sensor start frame id, sensor end frame id) [2]
        frames_sensor_start_pose : contiguous float tensor containing  position and rotation (quaternion) of the frame sensor at frame_start_timestamp [7]
        frames_sensor_end_pose : contiguous float tensor containing  position and rotation (quaternion) of the frame sensor at frame_end_timestamp [7]
        num_active_track_instances : number of active instances for the current frame
        active_track_instances_ids : contiguous int tensor containing the num_active_tracks map idx (into the initialized track_ids) and instance ids of the active tracks [num_active_tracksx2]
        active_track_instances_start_pose : contiguous float tensor containing  position and rotation (quaternion) of the active tracks at frame_start_timestamp [num_active_tracksx7]
        active_track_instances_end_pose : contiguous float tensor containing position and rotation (quaternion) of the active tracks at frame_end_timestamp [num_active_tracksx7]
    """

    def __init__(
        self,
        model: dict,
        renderer: dict,
        track_instances_uid: list[str],
        frame_id: int,
        frame_width: int,
        frame_height: int,
        frame_start_timestamp: int,
        frame_end_timestamp: int,
        rays_origin: torch.Tensor,
        rays_direction: torch.Tensor,
        rays_timestamp: Optional[torch.Tensor],
        frames_sensor_model: Optional[ConcreteCameraModelParametersUnion],
        frames_sensor_ids: Optional[torch.Tensor],
        frames_sensor_start_pose: Optional[torch.Tensor],
        frames_sensor_end_pose: Optional[torch.Tensor],
        num_active_track_instances: int,
        active_track_instances_ids: torch.Tensor,
        active_track_instances_start_pose: torch.Tensor,
        active_track_instances_end_pose: torch.Tensor,
        rays_radiance_density: torch.Tensor,
        rays_hit_distance: torch.Tensor,
        device: torch.device,
        rays_hit_normal: Optional[torch.Tensor] = None,
        extra_ray_signals: Optional[torch.Tensor] = None,
    ):
        self.model = model
        self.renderer = renderer
        self.track_instances_uid = track_instances_uid
        self.frame_id = frame_id
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_start_timestamp = frame_start_timestamp
        self.frame_end_timestamp = frame_end_timestamp
        self.rays_origin = rays_origin.reshape((frame_height, frame_width, 3))
        self.rays_direction = rays_direction.reshape((frame_height, frame_width, 3))
        self.rays_timestamp = (
            torch.zeros((frame_height, frame_width, 1), dtype=torch.int64)
            if rays_timestamp is None
            else rays_timestamp.reshape((frame_height, frame_width, 1))
        )
        self.frames_sensor_model = frames_sensor_model
        self.frames_sensor_ids = frames_sensor_ids
        self.frames_sensor_start_pose = frames_sensor_start_pose
        self.frames_sensor_end_pose = frames_sensor_end_pose
        self.num_active_track_instances = num_active_track_instances
        self.active_track_instances_ids = active_track_instances_ids
        self.active_track_instances_start_pose = active_track_instances_start_pose
        self.active_track_instances_end_pose = active_track_instances_end_pose
        self.rays_radiance_density = rays_radiance_density.cpu().numpy()
        self.rays_hit_distance = rays_hit_distance.cpu().numpy()
        self.device = device
        self.rays_hit_normal = rays_hit_normal.cpu().numpy() if rays_hit_normal is not None else None
        self.extra_ray_signals = extra_ray_signals.cpu().numpy() if extra_ray_signals is not None else None

    @classmethod
    def from_file(cls, file_path: str, device: torch.device):
        # log required to easily identified failing test case
        print(f"NRend::RendererTestCase ... loading test case {file_path}.")

        def load_tensor(bytes, dtype=np.float32) -> torch.Tensor:
            return torch.from_numpy(np.frombuffer(bytes, dtype).copy()).to(device)

        with gzip.open(file_path, "rb") as f:
            packed_test_case_dict = f.read()
        test_case_dict = msgpack.unpackb(packed_test_case_dict)

        frame_id = test_case_dict["frame_id"]
        frame_start_timestamp = test_case_dict["frame_start_timestamp"]
        frame_end_timestamp = test_case_dict["frame_end_timestamp"]
        frame_width = test_case_dict["frame_width"]
        frame_height = test_case_dict["frame_height"]
        rays_origin = test_case_dict["rays_origin"]
        rays_direction = test_case_dict["rays_direction"]
        rays_timestamp = test_case_dict["rays_timestamp"]
        num_active_track_instances = test_case_dict["num_active_track_instances"]
        active_track_instances_ids = test_case_dict["active_track_instances_ids"]
        active_track_instances_start_pose = test_case_dict["active_track_instances_start_pose"]
        active_track_instances_end_pose = test_case_dict["active_track_instances_end_pose"]
        rays_radiance_density = test_case_dict["rays_radiance_density"]
        rays_hit_distance = test_case_dict["rays_hit_distance"]
        rays_hit_normal = test_case_dict["rays_hit_normal"] if "rays_hit_normal" in test_case_dict else None
        extra_ray_signals = test_case_dict["extra_ray_signals"] if "extra_ray_signals" in test_case_dict else None

        if "frames_sensor_model" in test_case_dict:
            frames_sensor_model = pickle.loads(test_case_dict["frames_sensor_model"])
            frames_sensor_ids = load_tensor(test_case_dict["frames_sensor_ids"], dtype=np.int32)
            frames_sensor_start_pose = load_tensor(test_case_dict["frames_sensor_start_pose"])
            frames_sensor_end_pose = load_tensor(test_case_dict["frames_sensor_end_pose"])
        else:
            frames_sensor_model = None
            frames_sensor_start_pose = None
            frames_sensor_end_pose = None
            frames_sensor_ids = None

        return cls(
            model=test_case_dict["model"],
            renderer=test_case_dict["renderer"] if "renderer" in test_case_dict else {},
            track_instances_uid=test_case_dict["track_instances_uid"],
            frame_id=frame_id,
            frame_width=frame_width,
            frame_height=frame_height,
            frame_start_timestamp=frame_start_timestamp,
            frame_end_timestamp=frame_end_timestamp,
            rays_origin=load_tensor(rays_origin).reshape((frame_height, frame_width, 3)),
            rays_direction=load_tensor(rays_direction).reshape((frame_height, frame_width, 3)),
            rays_timestamp=load_tensor(rays_timestamp, dtype=np.int64).reshape((frame_height, frame_width, 1)),
            frames_sensor_model=frames_sensor_model,
            frames_sensor_ids=frames_sensor_ids,
            frames_sensor_start_pose=frames_sensor_start_pose,
            frames_sensor_end_pose=frames_sensor_end_pose,
            num_active_track_instances=num_active_track_instances,
            active_track_instances_ids=load_tensor(active_track_instances_ids, dtype=np.int32),
            active_track_instances_start_pose=load_tensor(active_track_instances_start_pose),
            active_track_instances_end_pose=load_tensor(active_track_instances_end_pose),
            rays_radiance_density=load_tensor(rays_radiance_density),
            rays_hit_distance=load_tensor(rays_hit_distance),
            device=device,
            rays_hit_normal=load_tensor(rays_hit_normal) if rays_hit_normal is not None else None,
            extra_ray_signals=load_tensor(extra_ray_signals) if extra_ray_signals is not None else None,
        )

    def run(self, decimal: int = 4):
        torch.cuda.synchronize()  # catch any async errors which may have happened before the test is launched
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]

        if len(devices) > 1:
            # we're going to ensure that all allocations happened on `self.device`. We start by emptying the caches and checking allocation states of all devices.
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            peak_memory_allocated_before_test_by_device = {
                device: torch.cuda.max_memory_allocated(device) for device in devices
            }

        renderer = Renderer(
            model=self.model,
            render_settings=self.renderer,
            log_level=Renderer.LogLevel.DEBUG,
            track_instances_uid_map=self.track_instances_uid,
        )
        assert renderer.valid(), "RendererTestCase : invalid model"

        frames_sensor_model = _prepare_sensor_model(self.frames_sensor_model, self.renderer, self.device)

        rays_radiance_density, rays_hit_distance, rays_hit_normal, extra_ray_signals, _ = renderer.render(
            frame_id=self.frame_id,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            frame_start_timestamp=self.frame_start_timestamp,
            frame_end_timestamp=self.frame_end_timestamp,
            rays_origin=self.rays_origin,
            rays_direction=self.rays_direction,
            rays_timestamp=self.rays_timestamp,
            frames_sensor_model=frames_sensor_model,
            frames_sensor_ids=self.frames_sensor_ids,
            frames_sensor_start_pose=self.frames_sensor_start_pose,
            frames_sensor_end_pose=self.frames_sensor_end_pose,
            num_active_track_instances=self.num_active_track_instances,
            active_track_instances_ids=self.active_track_instances_ids,
            active_track_instances_start_pose=self.active_track_instances_start_pose,
            active_track_instances_end_pose=self.active_track_instances_end_pose,
        )

        def assert_device(tensor: torch.Tensor) -> None:
            assert tensor.device == self.device, f"RendererTestCase : device mismatch {tensor.device} != {self.device}"

        assert_device(rays_radiance_density)
        assert_device(rays_hit_distance)
        assert_device(rays_hit_normal)
        assert_device(extra_ray_signals)

        if len(devices) > 1:
            peak_memory_allocated_after_test_by_device = {
                device: torch.cuda.max_memory_allocated(device) for device in devices
            }
            for device in devices:
                peak_delta = (
                    peak_memory_allocated_after_test_by_device[device]
                    - peak_memory_allocated_before_test_by_device[device]
                )
                if device != self.device:
                    assert peak_delta == 0, (
                        f"RendererTestCase : device {device} has {peak_delta:,} bytes allocated, expected 0"
                    )

        np.testing.assert_array_almost_equal(
            rays_radiance_density.cpu().numpy(),
            self.rays_radiance_density.reshape(rays_radiance_density.shape),
            decimal=decimal,
        )

        np.testing.assert_array_almost_equal(
            rays_hit_distance.cpu().numpy(),
            self.rays_hit_distance.reshape(rays_hit_distance.shape),
            decimal=decimal,
        )

        if self.rays_hit_normal is not None:
            np.testing.assert_array_almost_equal(
                rays_hit_normal.cpu().numpy(),
                self.rays_hit_normal.reshape(rays_hit_normal.shape),
                decimal=decimal,
            )

        if self.extra_ray_signals is not None:
            np.testing.assert_array_almost_equal(
                extra_ray_signals.cpu().numpy(),
                self.extra_ray_signals.reshape(extra_ray_signals.shape),
                decimal=decimal,
            )

        torch.cuda.synchronize()  # catch any async errors which may have happend during the test

    def write(self, file_path: str):
        test_case_dict: dict[str, Any] = {}
        test_case_dict["model"] = self.model
        test_case_dict["renderer"] = self.renderer
        test_case_dict["track_instances_uid"] = self.track_instances_uid
        test_case_dict["frame_id"] = self.frame_id
        test_case_dict["frame_width"] = self.frame_width
        test_case_dict["frame_height"] = self.frame_height
        test_case_dict["frame_start_timestamp"] = self.frame_start_timestamp
        test_case_dict["frame_end_timestamp"] = self.frame_end_timestamp
        test_case_dict["rays_origin"] = self.rays_origin.cpu().numpy().tobytes()
        test_case_dict["rays_direction"] = self.rays_direction.cpu().numpy().tobytes()
        test_case_dict["rays_timestamp"] = self.rays_timestamp.cpu().numpy().tobytes()
        if (
            (self.frames_sensor_model is not None)
            and (self.frames_sensor_ids is not None)
            and (self.frames_sensor_start_pose is not None)
            and (self.frames_sensor_end_pose is not None)
        ):
            test_case_dict["frames_sensor_model"] = pickle.dumps(self.frames_sensor_model)
            test_case_dict["frames_sensor_ids"] = self.frames_sensor_ids.cpu().numpy().tobytes()
            test_case_dict["frames_sensor_start_pose"] = self.frames_sensor_start_pose.cpu().numpy().tobytes()
            test_case_dict["frames_sensor_end_pose"] = self.frames_sensor_end_pose.cpu().numpy().tobytes()
        test_case_dict["num_active_track_instances"] = self.num_active_track_instances
        test_case_dict["active_track_instances_ids"] = self.active_track_instances_ids.cpu().numpy().tobytes()
        test_case_dict["active_track_instances_start_pose"] = (
            self.active_track_instances_start_pose.cpu().numpy().tobytes()
        )
        test_case_dict["active_track_instances_end_pose"] = self.active_track_instances_end_pose.cpu().numpy().tobytes()
        test_case_dict["rays_radiance_density"] = self.rays_radiance_density.tobytes()
        test_case_dict["rays_hit_distance"] = self.rays_hit_distance.tobytes()
        test_case_dict["rays_hit_normal"] = self.rays_hit_normal.tobytes() if self.rays_hit_normal is not None else None
        test_case_dict["extra_ray_signals"] = (
            self.extra_ray_signals.tobytes() if self.extra_ray_signals is not None else None
        )
        packed_test_case_dict = msgpack.packb(test_case_dict)

        with gzip.open(file_path, "wb") as f:
            f.write(packed_test_case_dict)

    def update(self, file_path: str):
        renderer = Renderer(
            model=self.model,
            render_settings=self.renderer,
            log_level=Renderer.LogLevel.DEBUG,
            track_instances_uid_map=self.track_instances_uid,
        )
        assert renderer.valid(), "RendererTestCase : invalid model"

        frames_sensor_model = _prepare_sensor_model(self.frames_sensor_model, self.renderer, self.device)

        rays_radiance_density, rays_hit_distance, rays_hit_normal, extra_ray_signals, _ = renderer.render(
            frame_id=self.frame_id,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            frame_start_timestamp=self.frame_start_timestamp,
            frame_end_timestamp=self.frame_end_timestamp,
            rays_origin=self.rays_origin,
            rays_direction=self.rays_direction,
            rays_timestamp=self.rays_timestamp,
            frames_sensor_model=frames_sensor_model,
            frames_sensor_ids=self.frames_sensor_ids,
            frames_sensor_start_pose=self.frames_sensor_start_pose,
            frames_sensor_end_pose=self.frames_sensor_end_pose,
            num_active_track_instances=self.num_active_track_instances,
            active_track_instances_ids=self.active_track_instances_ids,
            active_track_instances_start_pose=self.active_track_instances_start_pose,
            active_track_instances_end_pose=self.active_track_instances_end_pose,
        )

        self.rays_radiance_density = rays_radiance_density.cpu().numpy()
        self.rays_hit_distance = rays_hit_distance.cpu().numpy()
        self.rays_hit_normal = rays_hit_normal.cpu().numpy()
        self.extra_ray_signals = extra_ray_signals.cpu().numpy()
        self.write(file_path)
