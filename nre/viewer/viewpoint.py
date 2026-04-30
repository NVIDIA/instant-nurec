# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import torch
import viser
import viser.transforms as vtf

from kornia.geometry.linalg import inverse_transformation

from ncore.data import (
    OpenCVPinholeCameraModelParameters,
)
from ncore.sensors import CameraModel, FThetaCameraModel, OpenCVFisheyeCameraModel, OpenCVPinholeCameraModel
from nre.render import PoseRange, RayBundle, camera_model_to_parameters
from nre.utils.types import FrameConversion, HalfClosedInterval
from nre.viewer.dataset_interface import CameraTrajectoryData


@dataclass(kw_only=True)
class LookAtPose:
    """
    Represents pose as (up_direction, look_at, position) triplet.
    This is the parametrization used in viser frontend.
    """

    up: np.ndarray  # [3]
    look_at: np.ndarray  # [3]
    position: np.ndarray  # [3]

    def to_se3(self) -> vtf.SE3:
        forward = self.look_at - self.position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, self.up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.array([right, -up, forward])  # Note the up vector for OpenCV convention
        t = self.position

        return vtf.SE3.from_rotation_and_translation(vtf.SO3.from_matrix(R), t)

    @property
    def look_at_distance(self) -> float:
        return np.linalg.norm(self.look_at - self.position).item()

    @staticmethod
    def from_se3(se3: vtf.SE3, look_at_distance: float) -> LookAtPose:
        Q_mat = se3.rotation().as_matrix()
        direction = Q_mat[:, 2]  # Get the third column (forward direction)
        position = se3.translation()
        look_at = position + direction * look_at_distance
        up = -Q_mat[:, 1]  # Get the second column (up direction), negate for OpenCV convention

        return LookAtPose(up=up, look_at=look_at, position=position)


def ftheta_to_simple_pinhole(ftheta_camera: FThetaCameraModel) -> OpenCVPinholeCameraModel:
    """
    Finds a simple (no distortion) pinhole camera which attempts to closely match the FTheta model
    """
    fx = ftheta_camera.fw_poly[1].item()
    fy = ftheta_camera.fw_poly[1].item()
    cx = ftheta_camera.resolution[0].item() / 2
    cy = ftheta_camera.resolution[1].item() / 2

    pinhole_params = OpenCVPinholeCameraModelParameters(
        resolution=ftheta_camera.resolution.cpu().numpy().astype(np.uint64),
        shutter_type=ftheta_camera.shutter_type,
        principal_point=np.array([cx, cy], dtype=np.float32),
        focal_length=np.array([fx, fy], dtype=np.float32),
        radial_coeffs=np.zeros(6, dtype=np.float32),
        tangential_coeffs=np.zeros(2, dtype=np.float32),
        thin_prism_coeffs=np.zeros(4, dtype=np.float32),
    )

    return OpenCVPinholeCameraModel(pinhole_params, device="cpu")


def pinhole_to_simple_pinhole(pinhole_camera: OpenCVPinholeCameraModel) -> OpenCVPinholeCameraModel:
    """
    Strips away distortion parameters to make the pinhole camera compatible with viser model
    """
    cx = pinhole_camera.resolution[0].item() / 2
    cy = pinhole_camera.resolution[1].item() / 2

    pinhole_params = OpenCVPinholeCameraModelParameters(
        resolution=pinhole_camera.resolution.cpu().numpy().astype(np.uint64),
        shutter_type=pinhole_camera.shutter_type,
        principal_point=np.array([cx, cy], dtype=np.float32),
        focal_length=pinhole_camera.focal_length.cpu().numpy(),
        radial_coeffs=np.zeros(6, dtype=np.float32),
        tangential_coeffs=np.zeros(2, dtype=np.float32),
        thin_prism_coeffs=np.zeros(4, dtype=np.float32),
    )

    return OpenCVPinholeCameraModel(pinhole_params, device="cpu")


def fisheye_to_simple_pinhole(fisheye_camera: OpenCVFisheyeCameraModel) -> OpenCVPinholeCameraModel:
    """
    Finds a simple (no distortion) pinhole camera which attempts to closely match the fisheye model
    """
    cx = fisheye_camera.resolution[0].item() / 2
    cy = fisheye_camera.resolution[1].item() / 2

    pinhole_params = OpenCVPinholeCameraModelParameters(
        resolution=fisheye_camera.resolution.cpu().numpy().astype(np.uint64),
        shutter_type=fisheye_camera.shutter_type,
        principal_point=np.array([cx, cy], dtype=np.float32),
        focal_length=fisheye_camera.focal_length.cpu().numpy().astype(np.float32),
        radial_coeffs=np.zeros(6, dtype=np.float32),
        tangential_coeffs=np.zeros(2, dtype=np.float32),
        thin_prism_coeffs=np.zeros(4, dtype=np.float32),
    )

    return OpenCVPinholeCameraModel(pinhole_params, device="cpu")


def to_simple_pinhole(camera: CameraModel) -> OpenCVPinholeCameraModel:
    """
    Dispatches on specific camera type to find a matching simple pinhole model
    Here the simple pinhole model's principal point is moved to the image center to make viser's 3D elements align with rendering.
    """
    if isinstance(camera, OpenCVPinholeCameraModel):
        return pinhole_to_simple_pinhole(camera)
    elif isinstance(camera, FThetaCameraModel):
        return ftheta_to_simple_pinhole(camera)
    elif isinstance(camera, OpenCVFisheyeCameraModel):
        return fisheye_to_simple_pinhole(camera)
    else:
        raise TypeError(f"Camera {type(camera)=} not (yet) supported.")


def pinhole_to_fov(pinhole_camera: OpenCVPinholeCameraModel, viewer_aspect_ratio: float) -> float:
    """
    Find a `fov` parameter for viser to match the display of a given NCORE pinhole camera.
    This does not make sense for other camera models because they will not align with
    viser pinhole anyway. In fact, there will still be discrepancy if our
    OpenCVPinholeCameraModel has non-zero distortion parameters (perhaps to be checked
    and asserted against).

    - pinhole_camera: the NCORE camera
    - viewer_aspect_ratio: the aspect of viser viewport
    """
    w, h = pinhole_camera.resolution
    pinhole_aspect_ratio = w / h
    if viewer_aspect_ratio < pinhole_aspect_ratio:
        # the view is being padded on top and bottom so we need to adjust
        # the height to account for that
        h = h * pinhole_aspect_ratio / viewer_aspect_ratio

    _fx, fy = pinhole_camera.focal_length
    return float(2 * np.arctan2(h / 2, fy))


def vtf_se3_to_torch_tquat(se3: vtf.SE3) -> torch.Tensor:
    """NRE uses xyz_xyzw format for quaternions"""
    return torch.from_numpy(se3.wxyz_xyz)[[4, 5, 6, 1, 2, 3, 0]].float()


ViewerCameraChoice = Literal["original", "pinhole"]


@dataclass
class Viewpoint:
    """
    Represents a viewpoint, at a specified location/pose in time and space
    and the original and simplified camera models.
    """

    end_timestamp_us: int
    exposure_time_us: int
    trajectory_data: CameraTrajectoryData
    se3_world: vtf.SE3
    look_at_distance: float
    original_camera: CameraModel
    pinhole_camera: OpenCVPinholeCameraModel

    _ray_bundle_cache: dict[tuple[int, ViewerCameraChoice], RayBundle] = field(default_factory=dict)

    @staticmethod
    def create(trajectory: CameraTrajectoryData, end_frame_timestamp_us: int, look_at_distance: float) -> Viewpoint:
        exposure_time_range_us = HalfClosedInterval(
            start=end_frame_timestamp_us - trajectory.average_exposure_time_us,
            end=end_frame_timestamp_us + 1,
        )
        if exposure_time_range_us not in trajectory.time_range_us:
            raise AssertionError(f"Viewpoint {exposure_time_range_us=} outside {trajectory.time_range_us=}.")

        pose_world_mat = trajectory.get_poses_world(
            torch.tensor([end_frame_timestamp_us - trajectory.average_exposure_time_us]),
        ).squeeze(0)
        pose_world_qt = vtf.SE3.from_matrix(pose_world_mat.numpy())

        return Viewpoint(
            end_timestamp_us=end_frame_timestamp_us,
            exposure_time_us=trajectory.average_exposure_time_us,
            trajectory_data=trajectory,
            se3_world=pose_world_qt,
            look_at_distance=look_at_distance,
            original_camera=trajectory.camera_model,
            pinhole_camera=to_simple_pinhole(trajectory.camera_model),
        )

    @property
    def lookat_world(self) -> LookAtPose:
        return LookAtPose.from_se3(self.se3_world, self.look_at_distance)

    def viser_matching_fov(self, viewer_aspect_ratio: float) -> float:
        """
        Returns the field of view needed for viser to match self.pinhole_camera's projection

        Args:
            - viewer_aspect_ratio: the aspect ratio of viser viewport.
        """
        return pinhole_to_fov(self.pinhole_camera, viewer_aspect_ratio)

    def get_ray_bundle(
        self,
        resolution_step: float,
        which_camera: ViewerCameraChoice,
        world_to_nre: FrameConversion,
        camera_unique_idx: int | None = None,
    ) -> RayBundle:
        """
        Get a ray bundle from specified camera in nre coordinates.

        Args:
            - resolution_step: float subsampling step for rays, i.e. 4.0 means 1/4th original resolution
            - which_camera: whether the rays reflect the GT camera or the simplified pinhole model
            - world_to_nre: a conversion function to map from world coordinates to nre coordinates
            - camera_unique_idx: unique index of the camera
        """
        T_sensor_world_start_qt = self.se3_world
        T_sensor_world_start_matrix = torch.from_numpy(self.se3_world.as_matrix()).float()

        if which_camera == "original":
            camera_model = self.original_camera
            start_timestamp = self.end_timestamp_us - self.exposure_time_us
            end_timestamp = self.end_timestamp_us

            # find the pose delta between start and end of the frame at current timestamp
            T_sensor_world_start_gt, T_sensor_world_end_gt = self.trajectory_data.get_poses_world(
                torch.tensor(
                    [
                        self.end_timestamp_us - self.exposure_time_us,
                        self.end_timestamp_us,
                    ],
                )
            )
            T_sensor_world_delta = T_sensor_world_end_gt @ inverse_transformation(T_sensor_world_start_gt)

            # end pose is current pose adjusted by delta
            T_sensor_world_end_matrix = T_sensor_world_delta @ T_sensor_world_start_matrix

        elif which_camera == "pinhole":
            camera_model = self.pinhole_camera
            start_timestamp = self.end_timestamp_us - self.exposure_time_us
            end_timestamp = self.end_timestamp_us

            T_sensor_world_end_matrix = T_sensor_world_start_matrix

        T_sensor_world_end_qt = vtf.SE3.from_matrix(T_sensor_world_end_matrix.numpy())
        w, h = camera_model.resolution.tolist()

        camera_params = camera_model_to_parameters(camera_model)
        camera_params = camera_params.transform(
            image_domain_scale=1.0 / resolution_step,
            new_resolution=(round(w / resolution_step), round(h / resolution_step)),
        )

        camera_to_world = PoseRange(
            start_pose_tquat_sensor_world=vtf_se3_to_torch_tquat(T_sensor_world_start_qt),
            end_pose_tquat_sensor_world=vtf_se3_to_torch_tquat(T_sensor_world_end_qt),
            start_timestamp_us=start_timestamp,
            end_timestamp_us=end_timestamp,
        )

        return RayBundle.build(camera_params, camera_to_world, world_to_nre, unique_sensor_idx=camera_unique_idx)

    def update_to_client_pose(self, client: viser.ClientHandle) -> Viewpoint:
        """
        Updates (out of place) the camera pose according to client's camera pose.
        Camera intrinsics are unaffected.

        Args:
            - client: viser client handle to grab pose from.
        """
        look_at_distance = np.linalg.norm(client.camera.look_at - client.camera.position).item()
        return replace(
            self,
            se3_world=vtf.SE3.from_rotation_and_translation(
                vtf.SO3(wxyz=client.camera.wxyz),
                client.camera.position,
            ),
            look_at_distance=look_at_distance,
        )
