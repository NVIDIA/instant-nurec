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

from dataclasses import dataclass

import numpy as np

from nre.nrm.config.dataset import AdaptiveSequentialFrameBatchSamplerConfig
from nre.utils.types import HalfClosedInterval


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

    def sample_frame_batch(
        self,
        sample_idx: int,
        camera_frame_timestamps_us: dict[str, np.ndarray],
        time_intervals: list[HalfClosedInterval],
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


