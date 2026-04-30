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

import logging

import numpy as np

from omegaconf import DictConfig

from nre.datasets.samplers.base import BaseSensorSampler, SensorSamplerReturn


class WeightedSensorSampler(BaseSensorSampler):
    """Implements non-uniform sensor sampling using a per-camera probability distribution.

    Config field:
        per_camera_probabilities: list of floats, one per camera, summing to 1.0.
            Entry i is the sampling probability for the i-th camera in the dataset's
            camera enumeration. The order matches `dataset.camera_ids` in the active
            sensor config (e.g. configs/apps/prod/<RIG>/options/sensors/<rig>.yaml).
    """

    def __init__(self, config: DictConfig, dataset):
        super().__init__(config, dataset)
        self.logger = logging.getLogger(__name__)
        self._logged_once = False
        self.per_camera_probabilities = np.asarray(config.per_camera_probabilities, dtype=float)
        if not np.all(np.isfinite(self.per_camera_probabilities)) or np.any(self.per_camera_probabilities < 0):
            raise ValueError(f"Invalid per_camera_probabilities: {self.per_camera_probabilities.tolist()}")
        if not np.isclose(self.per_camera_probabilities.sum(), 1.0, atol=1e-6):
            raise ValueError(f"per_camera_probabilities must sum to 1.0, got {self.per_camera_probabilities.sum():.8f}")

    def sample_sensor(
        self,
        rng: np.random.Generator,
        batch_idx: int,
        sensor_ids: list[str],
    ) -> SensorSamplerReturn:
        """Implementation of 'SensorSampler' protocol sampling a sensor using configured weights."""

        if self.sample_all_sensors:
            return SensorSamplerReturn(sampled_sensor_ids=sensor_ids)

        if len(self.per_camera_probabilities) != len(sensor_ids):
            raise ValueError(
                f"per_camera_probabilities length ({len(self.per_camera_probabilities)}) does not match "
                f"number of sensors ({len(sensor_ids)}). "
                f"per_camera_probabilities[i] applies to sensor_ids[i], in the order the dataset "
                f"enumerates camera sensors. "
                f"sensor_ids={sensor_ids}"
            )

        if not self._logged_once:
            self.logger.info(
                "Weighted sensor sampling: %s",
                {sid: f"{p:.3f}" for sid, p in zip(sensor_ids, self.per_camera_probabilities.tolist())},
            )
            self._logged_once = True

        return SensorSamplerReturn(
            sampled_sensor_ids=rng.choice(sensor_ids, size=1, replace=False, p=self.per_camera_probabilities).tolist()
        )


BaseSensorSampler.register_to_sensor_sampler_factory("weighted", WeightedSensorSampler)
