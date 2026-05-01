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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np

from nre.nrm.config.dataset import (
    AdaptiveSequentialFrameBatchSamplerConfig,
    BaseFrameBatchSamplerConfig,
    LidarFrameBatchParamsConfig,
    SupervisionFrameBatchParamsConfig,
    UniformFrameBatchSamplerConfig,
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


def random_valid_start_time_from_intervals(
    rng: np.random.Generator,
    intervals: list[HalfClosedInterval],
    total_time_gap: int,
) -> int:
    """
    Args:
        rng (np.random.Generator): Random number generator.
        intervals (list[HalfClosedInterval]): List of half-closed intervals to sample from.
        total_time_gap (int): Total time gap to ensure the selected value is valid.

    Returns:
        int: A valid start timestamp where [ans, ans + total_time_gap) is within one of the intervals.
    """

    @dataclass(slots=True)
    class ClosedInterval:
        """Represents a closed interval [start, end]"""

        start: int
        end: int

        @property
        def valid_integers(self) -> int:
            return self.end - self.start + 1

    # Obtain valid start time intervals
    start_time_intervals: list[ClosedInterval] = []
    for time_interval in intervals:
        if time_interval.length >= total_time_gap:
            start_time_intervals.append(ClosedInterval(time_interval.start, time_interval.end - total_time_gap))
    if len(start_time_intervals) == 0:
        if len(intervals) == 0:
            raise NRMDataError(
                "No time intervals to sample from (empty list). Rig, auxiliary loaders, and context cameras "
                "may not overlap in time for this sequence."
            )
        max_len = max(ti.length for ti in intervals)
        raise NRMDataError(
            f"Uniform frame batch needs a contiguous time span of at least {total_time_gap} us "
            f"(frame_gap * (n_frames_per_sample - 1)), but after intersecting sensors the longest span is "
            f"{max_len} us. Reduce n_frames_per_sample or frame_gap_timestamp_us, or use a longer subrange."
        )

    # Pick a random start time from the valid intervals
    total_length = sum(interval.valid_integers for interval in start_time_intervals)
    sample_time = rng.integers(total_length)
    for interval in start_time_intervals:
        if sample_time < interval.valid_integers:
            return interval.start + sample_time
        sample_time -= interval.valid_integers
    raise RuntimeError("Should not reach here")

def sample_supervision_frame_batch(
    config: SupervisionFrameBatchParamsConfig,
    rng: np.random.Generator,
    context_frame_batch: FrameBatchSamplerReturn,
    sensor_frame_timestamps_us: dict[str, np.ndarray],
    supervision_sensor_ids: list[str],
    sensor_sample_ratios: list[float],
) -> FrameBatchSamplerReturn:
    context_timestamps_us_min = min(
        sensor_frame_timestamps_us[sensor_id][min(frame_idxs)]
        for sensor_id, frame_idxs in context_frame_batch.sampled_sensor_frame_idxs.items()
    )
    context_timestamps_us_max = max(
        sensor_frame_timestamps_us[sensor_id][max(frame_idxs)]
        for sensor_id, frame_idxs in context_frame_batch.sampled_sensor_frame_idxs.items()
    )
    n_samples: int = config.n_frames_per_sample
    supervision_sensor_frame_idxs: dict[str, list[int]] = {}
    for sensor_id, sample_ratio in zip(supervision_sensor_ids, sensor_sample_ratios, strict=True):
        sensor_timestamps = sensor_frame_timestamps_us[sensor_id]
        min_frame_idx = max(
            get_closest_frame_index(sensor_timestamps, context_timestamps_us_min - config.prepend_timestamps_us), 0
        )
        max_frame_idx = min(
            get_closest_frame_index(sensor_timestamps, context_timestamps_us_max + config.append_timestamps_us),
            len(sensor_timestamps) - 1,
        )
        frame_idxs = list(range(min_frame_idx, max_frame_idx + 1))
        sampled_indices: list[int] = []
        match config.sample_strategy:
            case "random":
                sampled_indices = np.sort(rng.choice(frame_idxs, size=n_samples, replace=True)).tolist()
            case "stratified":
                bins = np.linspace(0, len(frame_idxs), n_samples + 1, dtype=int)
                for i in range(n_samples):
                    bin_start, bin_end = bins[i], bins[i + 1]
                    if bin_start == bin_end:
                        sampled_indices.append(frame_idxs[bin_start])
                    else:
                        sampled_indices.append(rng.choice(frame_idxs[bin_start:bin_end]))
        if sample_ratio != 1.0:
            sampled_size = round(len(sampled_indices) * sample_ratio)
            sampled_indices = np.sort(rng.choice(sampled_indices, size=sampled_size, replace=False)).tolist()
        if len(sampled_indices) > 0:
            supervision_sensor_frame_idxs[sensor_id] = sampled_indices

    if config.include_context_frames:
        supervision_sensor_ids_set = set(supervision_sensor_ids)
        for sensor_id, ctx_idxs in context_frame_batch.sampled_sensor_frame_idxs.items():
            if sensor_id not in supervision_sensor_ids_set:
                continue
            existing = supervision_sensor_frame_idxs.get(sensor_id, [])
            merged = sorted(set(existing) | set(ctx_idxs))
            if merged:
                supervision_sensor_frame_idxs[sensor_id] = merged

    return FrameBatchSamplerReturn(sampled_sensor_frame_idxs=supervision_sensor_frame_idxs)


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


class BaseFrameBatchSampler(ABC):
    """Base sensor frame sampler used to sample a batch of frames from which a given batch is formed.
    These can be either sampled according to timestamps or trajectories"""

    @property
    @abstractmethod
    def num_samples_per_sequence(self) -> int: ...

    @abstractmethod
    def sample_frame_batch(
        self,
        rng: np.random.Generator,
        sample_idx: int,
        camera_frame_timestamps_us: dict[str, np.ndarray],
        time_intervals: list[HalfClosedInterval],
        rig_poses: RigPoses | None,
    ) -> FrameBatchSamplerReturn:
        """The return value only contains camera sensors."""
        ...


frame_batch_samplers: dict[str, Callable[..., BaseFrameBatchSampler]] = {}


def register(name):
    def decorator(cls):
        frame_batch_samplers[name] = cls
        return cls

    return decorator


def make(name: str, config: BaseFrameBatchSamplerConfig) -> BaseFrameBatchSampler:
    return frame_batch_samplers[name](config)


@register("uniform")
class UniformFrameBatchSampler(BaseFrameBatchSampler):
    """
    Randomly first select a frame, and then sample n_frames_per_sample frames with a fixed time gap.
    For each sequence this will spawn n_samples_per_sequence samples.

    +--------+--------+--------+
    | 0      | 1      | 2      |3
    +---C---------C--------C---+
    where C = frame_gap_timestamp_us
    """

    def __init__(self, config: UniformFrameBatchSamplerConfig):
        self.n_frames_per_sample: int = config.n_frames_per_sample
        self.n_samples_per_sequence: int = config.n_samples_per_sequence
        self.frame_gap_timestamp_us: int = config.frame_gap_timestamp_us

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

        # Calculate total time gap based on timestamp gap directly
        total_time_gap = self.frame_gap_timestamp_us * (self.n_frames_per_sample - 1)
        first_frame_timestamp = random_valid_start_time_from_intervals(rng, time_intervals, total_time_gap)

        # Generate timestamps with fixed time gap
        ref_frame_timestamps_us = [
            first_frame_timestamp + i * self.frame_gap_timestamp_us for i in range(self.n_frames_per_sample)
        ]

        sampled_sensor_frame_idxs: dict[str, list[int]] = {}
        for camera_id, camera_timestamps_us in camera_frame_timestamps_us.items():
            sampled_sensor_frame_idxs[camera_id] = []
            for frame_timestamp_us in ref_frame_timestamps_us:
                sampled_sensor_frame_idxs[camera_id].append(
                    get_closest_frame_index(camera_timestamps_us, int(frame_timestamp_us))
                )

        return FrameBatchSamplerReturn(sampled_sensor_frame_idxs=sampled_sensor_frame_idxs)


@register("adaptive_sequential")
class AdaptiveSequentialFrameBatchSampler(BaseFrameBatchSampler):
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


