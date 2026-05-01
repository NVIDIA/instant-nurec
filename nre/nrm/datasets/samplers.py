# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from dataclasses import dataclass

import numpy as np

from nre.nrm.config.dataset import (
    AdaptiveSequentialFrameBatchSamplerConfig,
    LidarFrameBatchParamsConfig,
)
from nre.nrm.datasets.nrm_base import NRMDataError
from nre.utils.types import HalfClosedInterval


logger = logging.getLogger(__name__)


@dataclass(slots=True, kw_only=True)
class RigPoses:
    T_rig_worlds: np.ndarray
    T_rig_world_timestamps_us: np.ndarray

    def __post_init__(self):
        n_poses = len(self.T_rig_worlds)
        assert self.T_rig_worlds.shape == (n_poses, 4, 4) and self.T_rig_worlds.dtype in [np.float32, np.float64]
        assert self.T_rig_world_timestamps_us.shape == (n_poses,) and self.T_rig_world_timestamps_us.dtype == np.uint64


def get_closest_frame_index(frame_timestamps_us: np.ndarray, target_timestamp_us: int) -> int:
    """
    Find the index of the frame whose timestamp is closest to the target timestamp.

    Args:
        frame_timestamps_us (np.ndarray): Array of frame timestamps in microseconds.
        target_timestamp_us (int): Target timestamp in microseconds.

    Returns:
        int: Index of the closest frame.
    """
    return int(np.abs(frame_timestamps_us.astype(np.int64) - target_timestamp_us).argmin())


def sample_lidar_frame_batch(
    config: LidarFrameBatchParamsConfig,
    frame_batch: FrameBatchSamplerReturn,
    sensor_frame_timestamps_us: dict[str, np.ndarray],
    lidar_id: str,
) -> FrameBatchSamplerReturn:
    """
    Sample lidar frame indices based on a previously sampled camera frame batch.
    """
    if config.gap_from_image_us == 0:
        return frame_batch

    sampled_lidar_frame_idxs: list[int] = []
    lidar_frame_timestamps_us = sensor_frame_timestamps_us[lidar_id]
    for sensor_id, frame_idxs in frame_batch.sampled_sensor_frame_idxs.items():
        assert sensor_id != lidar_id, "Lidar frame batch cannot be sampled from the same sensor"
        sensor_timestamps = sensor_frame_timestamps_us[sensor_id]
        for frame_idx in frame_idxs:
            timestamps_diff = np.abs(lidar_frame_timestamps_us.astype(np.int64) - sensor_timestamps[frame_idx].item())
            sampled_lidar_frame_idxs.extend(np.where(timestamps_diff <= config.gap_from_image_us)[0].tolist())

    sampled_lidar_frame_idxs = sorted(list(set(sampled_lidar_frame_idxs)))
    return FrameBatchSamplerReturn(
        sampled_sensor_frame_idxs={lidar_id: sampled_lidar_frame_idxs} | frame_batch.sampled_sensor_frame_idxs
    )


@dataclass(kw_only=True)
class FrameBatchSamplerReturn:
    """Return type of FrameBatchSampler"""

    # Mapping from sensor id to sampled frame indices (could also contain lidar sensors)
    sampled_sensor_frame_idxs: dict[str, list[int]]


class AdaptiveSequentialFrameBatchSampler:
    """
    Sequentially samples enough chunks to cover the sequence while keeping frame gaps below a configured maximum.

    The sequence is split into the minimum number of equal-sized chunks such that each chunk can be represented by
    n_frames_per_sample frame slots with max_frame_gap_timestamp_us spacing. Each sample index returns one chunk.
    """

    def __init__(self, config: AdaptiveSequentialFrameBatchSamplerConfig):
        self.n_frames_per_sample: int = config.n_frames_per_sample
        self.n_samples_per_sequence: int = config.n_samples_per_sequence
        self.max_frame_gap_timestamp_us: int = config.max_frame_gap_timestamp_us
        assert self.n_frames_per_sample > 0, "n_frames_per_sample must be positive"
        assert self.n_samples_per_sequence > 0, "n_samples_per_sequence must be positive"
        assert self.max_frame_gap_timestamp_us > 0, "max_frame_gap_timestamp_us must be positive"

    @property
    def num_samples_per_sequence(self) -> int:
        return self.n_samples_per_sequence

    def sample_frame_batch(
        self,
        rng: np.random.Generator,
        sample_idx: int,
        camera_frame_timestamps_us: dict[str, np.ndarray],
        time_intervals: list[HalfClosedInterval],
        rig_poses: RigPoses | None,
    ) -> FrameBatchSamplerReturn:
        assert len(camera_frame_timestamps_us) > 0, "No camera timestamps is provided to the frame batch sampler"
        assert 0 <= sample_idx < self.n_samples_per_sequence, "Sample index out of bounds"
        assert len(time_intervals) > 0, "No time intervals to sample from"

        sequence_start_timestamp = min(interval.start for interval in time_intervals)
        sequence_end_timestamp = max(interval.end for interval in time_intervals)
        sequence_total_timespan = sequence_end_timestamp - sequence_start_timestamp
        max_chunk_timespan = self.max_frame_gap_timestamp_us * self.n_frames_per_sample
        n_chunks = max(1, int(np.ceil(sequence_total_timespan / max_chunk_timespan)))

        if sample_idx >= n_chunks:
            return FrameBatchSamplerReturn(sampled_sensor_frame_idxs={})

        frame_gap_timestamp_us = sequence_total_timespan / (n_chunks * self.n_frames_per_sample)
        first_frame_idx = sample_idx * self.n_frames_per_sample
        ref_frame_timestamps_us = [
            int(sequence_start_timestamp + (first_frame_idx + frame_idx) * frame_gap_timestamp_us)
            for frame_idx in range(self.n_frames_per_sample)
        ]

        sampled_sensor_frame_idxs: dict[str, list[int]] = {}
        for camera_id, camera_timestamps_us in camera_frame_timestamps_us.items():
            sampled_sensor_frame_idxs[camera_id] = []
            for frame_timestamp_us in ref_frame_timestamps_us:
                sampled_sensor_frame_idxs[camera_id].append(
                    get_closest_frame_index(camera_timestamps_us, int(frame_timestamp_us))
                )

        return FrameBatchSamplerReturn(sampled_sensor_frame_idxs=sampled_sensor_frame_idxs)


