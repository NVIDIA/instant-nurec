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

import numpy as np
import numpy.typing as npt

from nre.datasets.samplers.base import (
    BaseCameraPixelSampler,
    BaseFrameSampler,
    BaseLidarPointSampler,
    BaseSensorSampler,
    CameraPixelSamplerReturn,
    FrameSamplerReturn,
    LidarPointSamplerReturn,
    SensorSamplerReturn,
)
from nre.utils.profiling import ScopedTimer, TimingTag


class UniformFrameSampler(BaseFrameSampler):
    """Implements uniformly sampling of sensor frames"""

    @ScopedTimer("UniformFrameSampler.sample_frame", TimingTag.DATALOADER)
    def sample_frame(
        self,
        rng: np.random.Generator,
        batch_idx: int,
        frame_range: range,
        unique_sensor_id: str,
    ) -> FrameSamplerReturn:
        """Implementation of 'FrameSampler' protocol uniformly sampling a valid frame"""
        return FrameSamplerReturn(
            sampled_frame_idx=rng.choice(frame_range, size=1, replace=False, shuffle=False).item()
        )


class UniformSensorSampler(BaseSensorSampler):
    """Implements uniform sensor sampling"""

    def sample_sensor(
        self,
        rng: np.random.Generator,
        batch_idx: int,
        sensor_ids: list[str],
    ) -> SensorSamplerReturn:
        """Implementation of 'SensorSampler' protocol uniformly sampling a sensor"""

        if self.sample_all_sensors:
            return SensorSamplerReturn(sampled_sensor_ids=sensor_ids)
        else:
            return SensorSamplerReturn(
                sampled_sensor_ids=rng.choice(sensor_ids, size=1, replace=False, shuffle=False).tolist()
            )


def sample_elements_uniform(rng: np.random.Generator, n_samples: int, elements: npt.NDArray) -> npt.NDArray:
    # the number of valid elements is a lower bound for the number of independent elements we can produce
    # (this supports zero valid elements in the limit)
    n_samples = min(len(elements), n_samples)

    # sample elements uniformly from the domain of elements
    return rng.choice(elements, size=n_samples, replace=False, shuffle=False)


class UniformCameraPixelSampler(BaseCameraPixelSampler):
    """Implements uniform sampling of camera pixels"""

    @ScopedTimer("UniformCameraPixelSampler.sample_camera_pixels", TimingTag.DATALOADER)
    def sample_camera_pixels(
        self,
        rng: np.random.Generator,
        batch_idx: int,
        frame_range: range,
        n_frame_pixel_samples: int,
        frame_all_pixels: npt.NDArray,
        frame_valid_pixels_mask: npt.NDArray,
        unique_camera_id: str,
        camera_frame_idx: int,
    ) -> CameraPixelSamplerReturn:
        """Implementation of 'CameraPixelSampler' protocol via uniform sampling for valid pixels"""
        frame_valid_pixels = frame_all_pixels[frame_valid_pixels_mask.flatten()]

        return CameraPixelSamplerReturn(
            sampled_pixels=sample_elements_uniform(rng, n_frame_pixel_samples, frame_valid_pixels)
        )


class UniformLidarPointSampler(BaseLidarPointSampler):
    """Implements uniform sampling of Lidar points"""

    @ScopedTimer("UniformLidarPointSampler.sample_lidar_points", TimingTag.DATALOADER)
    def sample_lidar_points(
        self,
        rng: np.random.Generator,
        batch_idx: int,
        frame_range: range,
        n_frame_point_samples: int,
        frame_valid_points_mask: npt.NDArray,
        unique_lidar_id: str,
        lidar_frame_idx: int,
    ) -> LidarPointSamplerReturn:
        """Implementation of 'LidarPointSampler' protocol via uniform sampling for valid points"""

        # Sample from *valid* points only
        frame_valid_point_idxs = np.where(frame_valid_points_mask)[0]

        return LidarPointSamplerReturn(
            sampled_point_idxs=sample_elements_uniform(rng, n_frame_point_samples, frame_valid_point_idxs)
        )


BaseSensorSampler.register_to_sensor_sampler_factory("uniform", UniformSensorSampler)
BaseFrameSampler.register_to_frame_sampler_factory("uniform", UniformFrameSampler)
BaseCameraPixelSampler.register_to_camera_pixel_sampler_factory("uniform", UniformCameraPixelSampler)
BaseLidarPointSampler.register_to_lidar_point_sampler_factory("uniform", UniformLidarPointSampler)
