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

from typing import TYPE_CHECKING

import numpy as np
import torch

from omegaconf import DictConfig

from nre.datasets.samplers.base import BaseFrameSampler, FrameSamplerReturn


if TYPE_CHECKING:
    from nre.datasets.ncore import NCORETrainDataset

from ncore.impl.data.util import closest_index_sorted


class TimestampFrameSampler(BaseFrameSampler):
    """Sampler for batches based on timestamp related sampling. All sensors in a batch will sample the frame that is closest to a randomly sampled timestamp"""

    def __init__(
        self,
        config: DictConfig,
        dataset: NCORETrainDataset,
    ) -> None:
        # Assumes there is only a single rig_trajectory
        rt = dataset.get_datasource().get_rig_trajectories()
        assert len(rt.rig_trajectories) == 1, (
            f"{self.__class__.__name__}: expected a single rig_trajectory, got {len(rt.rig_trajectories)}"
        )
        rig_trajectory = rt.rig_trajectories[0]
        self.unique_sensor_id_to_timestamps: dict[str, torch.Tensor] = (
            rig_trajectory.cameras_frame_timestamps_us | rig_trajectory.lidars_frame_timestamps_us
        )
        self.batch_idx_to_timestamp: dict[int, int] = {}

    def sample_frame(
        self,
        rng: np.random.Generator,
        batch_idx: int,
        frame_range: range,
        unique_sensor_id: str,
    ) -> FrameSamplerReturn:
        """Sample a single frame based on the given batch_idx which defines the unique timestamp for all the sensors in the batch"""

        # Take the end timestamps of all frames for the given sensor
        sensor_timestamps = self.unique_sensor_id_to_timestamps[unique_sensor_id][:, 1].numpy()
        assert len(frame_range) == len(sensor_timestamps), (
            f"{self.__class__.__name__}: the number of timestamps and the number of frames should be the same"
        )

        # If the timestamp of this batch was already sampled then fetch it from the map, otherwise sample a new one and store it
        # This ensures that different sensors in the same batched will be sampled around the same timestamp. We override the
        # mapping after each batch such that we maintain random sampling of timestamps across epochs
        if batch_idx in self.batch_idx_to_timestamp:
            sampled_timestamp = self.batch_idx_to_timestamp[batch_idx]
        else:
            sampled_timestamp = rng.choice(sensor_timestamps, size=1, replace=False, shuffle=False).item()
            self.batch_idx_to_timestamp = {batch_idx: sampled_timestamp}

        frame_idx = closest_index_sorted(sensor_timestamps, sampled_timestamp)

        return FrameSamplerReturn(sampled_frame_idx=frame_range[frame_idx])


BaseFrameSampler.register_to_frame_sampler_factory("timestamp", TimestampFrameSampler)
