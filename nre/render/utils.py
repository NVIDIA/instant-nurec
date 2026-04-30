# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Tuple

import torch

from libs.geometry.kernels.pose import frame_transform_poses_tquat
from ncore.data import ConcreteCameraModelParametersUnion
from ncore.sensors import CameraModel, FThetaCameraModel, OpenCVFisheyeCameraModel, OpenCVPinholeCameraModel
from nre.utils.profiling import ScopedTimer
from nre.utils.types import FrameConversion


@ScopedTimer()
def camera_model_to_parameters(camera: CameraModel) -> ConcreteCameraModelParametersUnion:
    """
    Dispatches on specific camera type to find the corresponding parameters
    """
    match camera:
        case OpenCVPinholeCameraModel():
            return camera.get_parameters()
        case OpenCVFisheyeCameraModel():
            return camera.get_parameters()
        case FThetaCameraModel():
            return camera.get_parameters()
        case _:
            raise TypeError(f"Camera {type(camera)=} not (yet) supported.")


def transform_intrinsics_to_resolution(
    camera_intrinsics: ConcreteCameraModelParametersUnion,
    resolution: Tuple[int, int],  # width, height
) -> ConcreteCameraModelParametersUnion:
    """
    Transform intrinsics such that the image fits within the requested resolution while maintaining the aspect ratio
    up to an int-rounding error.
    """

    cam_width, cam_height = camera_intrinsics.resolution.astype(int).tolist()
    req_width, req_height = resolution
    aspect_ratio = cam_width / cam_height
    if (req_width / cam_width) > (req_height / cam_height):
        render_width = req_width
        render_height = int(render_width / aspect_ratio)
        image_scale = render_width / cam_width
    else:
        render_height = req_height
        render_width = int(render_height * aspect_ratio)
        image_scale = render_height / cam_height

    # Scale the camera parameters to match the rendering output resolution.
    camera_intrinsics = camera_intrinsics.transform(
        image_domain_scale=image_scale,
        new_resolution=(render_width, render_height),
    )
    assert camera_intrinsics.resolution.astype(int).tolist() == [render_width, render_height]
    return camera_intrinsics


def frame_transform_poses(coord_frames: FrameConversion, T_poses: torch.Tensor, is_tquat: bool) -> torch.Tensor:
    """
    Transforms poses in the source frame to corresponding poses in the target frame.

    Supports both singular (4,4) and batched (N,4,4) input poses 'T_poses_source'

    NOTE: The implementation is based on FrameConversion::transform_poses

    Args:
        coord_frames: frame object [FrameConversion]
        T_poses: poses in source frame units [torch.Tensor]
        is_tquat: flag to indicate if the input poses are in tquat format

    Returns:
        poses have target frame units [torch.Tensor]
    """
    if is_tquat:
        assert T_poses.shape[-1] == 7
        T_poses = frame_transform_poses_tquat(
            T_poses,
            coord_frames.rotation_quat_tuple,
            coord_frames.translation_tuple,
            coord_frames.target_scale,
        )
    else:
        T_poses = torch.from_numpy(coord_frames.transform_poses(T_poses.cpu().numpy())).to(
            device=T_poses.device, dtype=T_poses.dtype
        )
    return T_poses
