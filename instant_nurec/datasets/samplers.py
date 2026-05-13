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

import logging

import numpy as np

from instant_nurec.config_schema.dataset import AdaptiveSequentialFrameBatchSamplerConfig
from instant_nurec.utils.types import HalfClosedInterval


logger = logging.getLogger(__name__)


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


# `sampled_sensor_frame_idxs` mapping: camera id → frame indices.
SampledSensorFrameIdxs = dict[str, list[int]]


class AdaptiveSequentialFrameBatchSampler:
    """
    Sequentially samples enough chunks to cover the sequence while keeping frame gaps below a configured maximum.

    The sequence is split into the minimum number of equal-sized chunks such that each chunk can be represented by
    n_frames_per_sample frame slots with max_frame_gap_timestamp_us spacing. Each sample index returns one chunk.
    """

    def __init__(
        self,
        config: AdaptiveSequentialFrameBatchSamplerConfig,
        n_frames_per_sample: int,
    ):
        # ``n_frames_per_sample`` is passed in by the dataset rather than
        # read from the config.
        self.n_frames_per_sample: int = n_frames_per_sample
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
    ) -> SampledSensorFrameIdxs:
        assert len(camera_frame_timestamps_us) > 0, "No camera timestamps is provided to the frame batch sampler"
        assert 0 <= sample_idx < self.n_samples_per_sequence, "Sample index out of bounds"
        assert len(time_intervals) > 0, "No time intervals to sample from"

        sequence_start_timestamp = min(interval.start for interval in time_intervals)
        sequence_end_timestamp = max(interval.end for interval in time_intervals)
        sequence_total_timespan = sequence_end_timestamp - sequence_start_timestamp
        max_chunk_timespan = self.max_frame_gap_timestamp_us * self.n_frames_per_sample
        n_chunks = max(1, int(np.ceil(sequence_total_timespan / max_chunk_timespan)))

        if sample_idx == 0 and n_chunks > self.n_samples_per_sequence:
            chunk_seconds = max_chunk_timespan / 1e6
            clip_seconds = sequence_total_timespan / 1e6
            covered_seconds = self.n_samples_per_sequence * chunk_seconds
            dropped = n_chunks - self.n_samples_per_sequence
            logger.warning(
                "Clip spans %.1fs (~%d chunks of %.1fs) but n_samples_per_sequence=%d "
                "(`--max-chunks`) caps processing to the first ~%.1fs; "
                "%d chunk(s) will be silently dropped. "
                "Set --max-chunks to %d (= ceil(clip_seconds / %.1f)) to process the full clip.",
                clip_seconds,
                n_chunks,
                chunk_seconds,
                self.n_samples_per_sequence,
                covered_seconds,
                dropped,
                n_chunks,
                chunk_seconds,
            )

        if sample_idx >= n_chunks:
            return {}

        frame_gap_timestamp_us = sequence_total_timespan / (n_chunks * self.n_frames_per_sample)
        first_frame_idx = sample_idx * self.n_frames_per_sample
        ref_frame_timestamps_us = [
            int(sequence_start_timestamp + (first_frame_idx + frame_idx) * frame_gap_timestamp_us)
            for frame_idx in range(self.n_frames_per_sample)
        ]

        sampled_sensor_frame_idxs: SampledSensorFrameIdxs = {}
        for camera_id, camera_timestamps_us in camera_frame_timestamps_us.items():
            sampled_sensor_frame_idxs[camera_id] = []
            for frame_timestamp_us in ref_frame_timestamps_us:
                sampled_sensor_frame_idxs[camera_id].append(
                    get_closest_frame_index(camera_timestamps_us, int(frame_timestamp_us))
                )

        return sampled_sensor_frame_idxs

