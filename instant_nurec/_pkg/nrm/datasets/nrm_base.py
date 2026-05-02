# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import logging

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

import ncore.data

from instant_nurec._pkg.nrm.config.dataset import CameraSubsamplerConfig
from instant_nurec._pkg.utils.sensors import RectSubsampledSensor


logger = logging.getLogger(__name__)


class CameraSubsampler:
    """
    Dedicated class to subsample camera parameters or the images (currently center crop is used).
    This could be later extended to include more complex subsampling strategies (e.g. for progressive training).
    """

    def __init__(self, config: CameraSubsamplerConfig):
        self.frame_width = config.frame_width
        self.frame_height = config.frame_height

    def _compute_pixel_rect(
        self, original_width: int, original_height: int
    ) -> Tuple[RectSubsampledSensor, Tuple[int, int]]:
        scale_factor = max(self.frame_width / original_width, self.frame_height / original_height)
        scaled_w = round(original_width * scale_factor)
        scaled_h = round(original_height * scale_factor)
        offset_w = (scaled_w - self.frame_width) // 2
        offset_h = (scaled_h - self.frame_height) // 2
        return RectSubsampledSensor(
            subsample_factor=1.0 / scale_factor,
            i=offset_w,
            j=offset_h,
            width=self.frame_width,
            height=self.frame_height,
        ), (scaled_w, scaled_h)

    def apply_camera_parameters(
        self, camera_parameters: ncore.data.ConcreteCameraModelParametersUnion
    ) -> ncore.data.ConcreteCameraModelParametersUnion:
        original_width = camera_parameters.resolution[0].item()
        original_height = camera_parameters.resolution[1].item()
        pixel_rect, _ = self._compute_pixel_rect(original_width, original_height)
        return camera_parameters.transform(
            image_domain_scale=1.0 / pixel_rect.subsample_factor,
            image_domain_offset=(pixel_rect.i, pixel_rect.j),
            new_resolution=(pixel_rect.width, pixel_rect.height),
        )

    def apply_depth_data(self, depth_data: np.ndarray) -> np.ndarray:
        """
        Apply reshape where depth_data is (H, W). Uses nearest-foreground downsampling
        via adaptive max-pool on negated depths to avoid sky bleed-in.
        """
        original_height, original_width = depth_data.shape[:2]
        pixel_rect, (scaled_w, scaled_h) = self._compute_pixel_rect(original_width, original_height)

        depth_data_pth = torch.from_numpy(depth_data).float().clone()
        depth_data_pth[depth_data_pth == 0] = float("inf")
        depth_data_pth = -F.adaptive_max_pool2d(-depth_data_pth[None, None], (scaled_h, scaled_w))
        depth_data_pth[depth_data_pth == float("inf")] = 0.0

        depth_data = depth_data_pth[0, 0].numpy()
        return depth_data[
            pixel_rect.j : pixel_rect.j + pixel_rect.height, pixel_rect.i : pixel_rect.i + pixel_rect.width
        ]

    def apply_frame_data(self, frame_data: np.ndarray) -> np.ndarray:
        """
        Apply reshape where frame_data is (H, W, C) or (H, W)
        If frame_data is float then we do bilinear interpolation, otherwise we do nearest neighbor.
        """
        if not frame_data.flags.writeable:
            frame_data = frame_data.copy()

        if frame_data.ndim == 2:
            frame_data = frame_data[..., None]
            batch_dim = False
        else:
            batch_dim = True

        assert frame_data.ndim == 3
        is_floating_point = frame_data.dtype in [np.float32, np.float64]

        original_height, original_width = frame_data.shape[:2]
        pixel_rect, (scaled_w, scaled_h) = self._compute_pixel_rect(original_width, original_height)
        frame_data_pth = torch.from_numpy(frame_data).moveaxis(-1, 0)

        if is_floating_point:
            frame_data_pth = torch.nn.functional.interpolate(
                frame_data_pth[None],
                size=(scaled_h, scaled_w),
                mode="bilinear",
                align_corners=True,
                antialias=True,
            )
        else:
            frame_data_pth = torch.nn.functional.interpolate(
                frame_data_pth[None].float(),
                size=(scaled_h, scaled_w),
                mode="nearest",
            )

        frame_data = frame_data_pth[0].moveaxis(0, -1).numpy().astype(frame_data.dtype)
        frame_data = frame_data[
            pixel_rect.j : pixel_rect.j + pixel_rect.height, pixel_rect.i : pixel_rect.i + pixel_rect.width
        ]
        return frame_data if batch_dim else frame_data[..., 0]


class NRMDataError(Exception):
    """
    Exception raised when an error occurs while loading NRM data.
    Predict-only standalone propagates this directly: the NRE retry-on-error
    loop made sense for training over millions of samples but not for a
    single inference run where a bad sample should fail loud.
    """

    def __init__(self, message: str = "An error occurred while loading NRM data"):
        super().__init__(message)
        self.message = message
