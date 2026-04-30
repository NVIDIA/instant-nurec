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
    LinearWithIndexFrameBatchSamplerConfig,
    SequentialFrameBatchSamplerConfig,
    SequentialLengthFrameBatchSamplerConfig,
    SupervisionFrameBatchParamsConfig,
    UniformFrameBatchSamplerConfig,
    UniformLengthFrameBatchSamplerConfig,
    VaryingIntervalFrameBatchSamplerConfig,
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


def random_valid_start_value_from_intervals(
    rng: np.random.Generator,
    intervals: list[tuple[float, float]],
    total_value_gap: float,
) -> float:
    """
    Args:
        rng (np.random.Generator): Random number generator.
        intervals (list[tuple[float, float]]): List of intervals to sample from (inclusive).
        total_value_gap (float): Total value gap to ensure the selected value is valid.

    Returns:
        float: A valid start value where [ans, ans + total_value_gap] is within one of the intervals.
    """

    valid_start_intervals: list[tuple[float, float]] = [
        (st, ed - total_value_gap) for (st, ed) in intervals if (ed - st) >= total_value_gap
    ]
    assert len(valid_start_intervals) > 0, "No valid intervals to sample from"

    # Pick a random start time from the valid intervals
    total_sampled_value = sum((ed - st) for (st, ed) in valid_start_intervals)
    sample_value = rng.uniform(0, total_sampled_value)
    for st, ed in valid_start_intervals:
        if sample_value < (ed - st):
            return st + sample_value
        sample_value -= ed - st
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


@register("varying_interval")
class VaryingIntervalFrameBatchSampler(BaseFrameBatchSampler):
    """
    First randomly pick a sequence_gap ranging from sequence_gap_timestamp_us_min to sequence_gap_timestamp_us_max.
    Then randomly select a start frame, and *randomly* sample n_frames_per_sample frames with the above time gap.
    For each sequence this will spawn n_samples_per_sequence samples.

    +----+-----------+---------+
    | 0  | 1         | 2       |3
    +---------- P -------------+
    where P = sequence_gap_timestamp_us_min to sequence_gap_timestamp_us_max
    """

    def __init__(self, config: VaryingIntervalFrameBatchSamplerConfig):
        self.n_frames_per_sample: int = config.n_frames_per_sample
        self.n_samples_per_sequence: int = config.n_samples_per_sequence
        self.sequence_gap_timestamp_us_min: int = config.sequence_gap_timestamp_us_min
        self.sequence_gap_timestamp_us_max: int = config.sequence_gap_timestamp_us_max
        assert self.n_frames_per_sample >= 2, "n_frames_per_sample must be at least 2 to sample a sequence gap"

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

        # Determine total time gap (i.e. sequence_gap) not exceeding the max interval length
        total_time_gap = rng.integers(self.sequence_gap_timestamp_us_min, self.sequence_gap_timestamp_us_max + 1)
        max_interval_length = max(interval.length for interval in time_intervals)
        total_time_gap = min(total_time_gap, max_interval_length)

        first_frame_timestamp = random_valid_start_time_from_intervals(rng, time_intervals, total_time_gap)

        # Generate timestamps with randomized time gap
        ref_frame_alphas = rng.random(self.n_frames_per_sample - 2).tolist() + [0.0, 1.0]
        ref_frame_timestamps_us = [first_frame_timestamp + int(alpha * total_time_gap) for alpha in ref_frame_alphas]
        ref_frame_timestamps_us.sort()

        sampled_sensor_frame_idxs: dict[str, list[int]] = {}
        for camera_id, camera_timestamps_us in camera_frame_timestamps_us.items():
            sampled_sensor_frame_idxs[camera_id] = []
            for frame_timestamp_us in ref_frame_timestamps_us:
                sampled_sensor_frame_idxs[camera_id].append(
                    get_closest_frame_index(camera_timestamps_us, int(frame_timestamp_us))
                )

        return FrameBatchSamplerReturn(sampled_sensor_frame_idxs=sampled_sensor_frame_idxs)


@register("linear_with_index")
class LinearWithIndexFrameBatchSampler(BaseFrameBatchSampler):
    """
    Sample n-1 frames linearly across a time gap, and 1 frame based on sample_idx.
    For each sequence this will spawn n_samples_per_sequence samples.

    The sampling strategy:
    1. Sample n-1 frames linearly distributed between first and last frame (first + total_time_gap)
    2. Sample 1 frame at a position determined by sample_idx (normalized to [0, 1]) within the total_time_gap range
    """

    def __init__(self, config: LinearWithIndexFrameBatchSamplerConfig):
        self.n_frames_per_sample: int = config.n_frames_per_sample
        self.n_samples_per_sequence: int = config.n_samples_per_sequence
        self.total_time_gap: int = config.total_time_gap
        self.first_frame_timestamp: int = config.first_frame_timestamp

        # Validate n_samples_per_sequence to prevent division logic inconsistencies
        if self.n_samples_per_sequence < 1:
            raise ValueError(
                f"n_samples_per_sequence must be at least 1, got {self.n_samples_per_sequence}. "
                "Zero or negative values lead to inconsistent division logic in sampling calculations."
            )

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

        first_frame_timestamp = self.first_frame_timestamp + min(interval.start for interval in time_intervals)

        # Calculate last frame timestamp based on total_time_gap
        last_frame_timestamp = first_frame_timestamp + self.total_time_gap

        # Sample n-1 frames linearly between first and last frame
        linear_timestamps = np.linspace(
            first_frame_timestamp, last_frame_timestamp, self.n_frames_per_sample - 1, dtype=int
        )

        # Sample 1 frame based on sample_idx (normalized to [0, 1]) within the total_time_gap range
        sample_alpha = sample_idx / max(1, self.n_samples_per_sequence - 1)
        min_gap = self.total_time_gap / (self.n_samples_per_sequence + 1)
        index_timestamp = int(first_frame_timestamp + min_gap + sample_alpha * (self.total_time_gap - min_gap * 2))

        # Combine all timestamps and sort them
        all_timestamps = np.concatenate([linear_timestamps, [index_timestamp]])
        all_timestamps = np.sort(all_timestamps)

        # Remove duplicates while preserving order
        unique_timestamps: list[float] = []
        for timestamp in all_timestamps:
            if not unique_timestamps or timestamp != unique_timestamps[-1]:
                unique_timestamps.append(timestamp)

        # If we have fewer unique timestamps than needed, pad with the last timestamp
        while len(unique_timestamps) < self.n_frames_per_sample:
            unique_timestamps.append(unique_timestamps[-1])

        # Take only the required number of frames
        ref_frame_timestamps_us = unique_timestamps[: self.n_frames_per_sample]

        sampled_sensor_frame_idxs: dict[str, list[int]] = {}
        for camera_id, camera_timestamps_us in camera_frame_timestamps_us.items():
            sampled_sensor_frame_idxs[camera_id] = []
            for frame_timestamp_us in ref_frame_timestamps_us:
                sampled_sensor_frame_idxs[camera_id].append(
                    get_closest_frame_index(camera_timestamps_us, int(frame_timestamp_us))
                )

        return FrameBatchSamplerReturn(sampled_sensor_frame_idxs=sampled_sensor_frame_idxs)


@register("sequential")
class SequentialFrameBatchSampler(BaseFrameBatchSampler):
    """
    Sequentially samples video frames in chunks, e.g., (0, t-1), (t, 2t-1), ...
    For each sequence this will spawn n_samples_per_sequence samples.

    The sampling strategy:
    1. Calculate the total time span needed: frame_gap_timestamp_us * (n_frames_per_sample-1) * n_samples_per_sequence
    2. For each sample, select n_frames_per_sample consecutive frames starting from sample_idx * chunk_size
    3. Frames are spaced by frame_gap_timestamp_us within each sample
    4. The reconstructed video spans from t_start to t_start + total_time_span
    """

    def __init__(self, config: SequentialFrameBatchSamplerConfig):
        self.n_frames_per_sample: int = config.n_frames_per_sample
        self.n_samples_per_sequence: int = config.n_samples_per_sequence
        self.first_frame_timestamp: int = config.first_frame_timestamp
        self.frame_gap_timestamp_us: int = config.frame_gap_timestamp_us
        self.allow_out_of_bounds: bool = config.allow_out_of_bounds

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

        # Calculate the time span for one chunk
        chunk_time_span = self.frame_gap_timestamp_us * self.n_frames_per_sample

        # Use the first_frame_timestamp from config, similar to LinearWithIndexFrameBatchSampler
        first_frame_timestamp = self.first_frame_timestamp + min(interval.start for interval in time_intervals)

        # Calculate the start time for this specific sample
        sample_start_time = first_frame_timestamp + sample_idx * chunk_time_span

        # Generate timestamps with fixed time gap for this sample
        ref_frame_timestamps_us = [
            sample_start_time + i * self.frame_gap_timestamp_us for i in range(self.n_frames_per_sample)
        ]

        max_timestamp_us = max(interval.end for interval in time_intervals)
        if self.allow_out_of_bounds:
            # If the entire range is out of bounds, return empty batch.
            if ref_frame_timestamps_us[0] > max_timestamp_us:
                return FrameBatchSamplerReturn(sampled_sensor_frame_idxs={})

            # If part of the range is out of bounds, recompute reference timestamps so it include last frame.
            if ref_frame_timestamps_us[-1] > max_timestamp_us:
                ref_frame_timestamps_us = np.linspace(
                    sample_start_time, max_timestamp_us, self.n_frames_per_sample, dtype=int
                ).tolist()

        else:
            # check if the last frame is within the time intervals
            if ref_frame_timestamps_us[-1] > max_timestamp_us:
                raise ValueError(
                    f"The last frame is out of the time intervals: {ref_frame_timestamps_us[-1]} > {max_timestamp_us}"
                )

        sampled_sensor_frame_idxs: dict[str, list[int]] = {}
        for camera_id, camera_timestamps_us in camera_frame_timestamps_us.items():
            sampled_sensor_frame_idxs[camera_id] = []
            for frame_timestamp_us in ref_frame_timestamps_us:
                sampled_sensor_frame_idxs[camera_id].append(
                    get_closest_frame_index(camera_timestamps_us, int(frame_timestamp_us))
                )

        return FrameBatchSamplerReturn(sampled_sensor_frame_idxs=sampled_sensor_frame_idxs)


def interpolate_timestamps(values: list[float], value_array: np.ndarray, timestamps_us: np.ndarray) -> np.ndarray:
    """
    Equivalent to np.interp, but handles the uint64 dtype of timestamps_us correctly.
    """
    timestamps_us_min = np.min(timestamps_us)
    return (
        np.interp(values, value_array, (timestamps_us - timestamps_us_min).astype(float)).astype(timestamps_us.dtype)
        + timestamps_us_min
    )


@register("uniform_length")
class UniformLengthFrameBatchSampler(BaseFrameBatchSampler):
    """
    First calculate running length (cumsum of the norm of the difference between consecutive pose positions).
    Then use the given time_intervals to crop the valid ranges of running length (all inclusive).
    Sample a random length starting point from the valid ranges, and then linearly interpolate the corresponding timestamps.
    Sensor timestamps are NN to the interpolated timestamps.
    """

    def __init__(self, config: UniformLengthFrameBatchSamplerConfig):
        self.length_gap_min: float = config.length_gap_min
        self.length_gap_max: float = config.length_gap_max
        self.n_frames_per_sample: int = config.n_frames_per_sample
        self.n_samples_per_sequence: int = config.n_samples_per_sequence
        assert self.n_frames_per_sample > 1, "n_frames_per_sample must be at least 2 to sample a length gap"

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
        assert rig_poses is not None, "Rig poses are required to sample a frame batch"
        assert len(camera_frame_timestamps_us) > 0, "No camera timestamps is provided to the frame batch sampler"
        assert 0 <= sample_idx < self.n_samples_per_sequence, "Sample index out of bounds"

        # Compute running length, rl[i] - rl[j] is length between pose i and j
        running_length = np.cumsum(
            np.linalg.norm(rig_poses.T_rig_worlds[1:, :3, 3] - rig_poses.T_rig_worlds[:-1, :3, 3], axis=-1), axis=0
        )
        running_length = np.concatenate([[0], running_length])

        # Convert from time intervals to length intervals (all inclusive)
        valid_length_intervals: list[tuple[float, float]] = []
        for interval in time_intervals:
            i_left = np.searchsorted(rig_poses.T_rig_world_timestamps_us, interval.start)
            i_right = np.searchsorted(rig_poses.T_rig_world_timestamps_us, interval.end) - 1
            valid_length_intervals.append((running_length[i_left], running_length[i_right]))

        # Sample a length starting point from the intervals (also truncate gap if valid ranges are too short)
        max_interval_length = max((ed - st) for (st, ed) in valid_length_intervals)
        target_interval_length = min(
            rng.uniform(self.length_gap_min, self.length_gap_max) * (self.n_frames_per_sample - 1),
            max_interval_length,
        )
        target_interval_gap = target_interval_length / (self.n_frames_per_sample - 1)

        if not (self.length_gap_min <= target_interval_gap <= self.length_gap_max):
            logger.warning(
                f"{self.__class__.__name__}: Truncated interval gap {target_interval_gap} is "
                f"out of bounds defined between {self.length_gap_min} and {self.length_gap_max}."
            )

        sampled_start_length = random_valid_start_value_from_intervals(
            rng, valid_length_intervals, target_interval_length
        )

        # Note here we can safely make sure that the sampled length is WITHIN given time intervals,
        # now we convert them into the actual timestamp
        # (linear interpolating instead of NN since rig_poses might be lower-frequency)
        ref_frame_timestamps_us = interpolate_timestamps(
            [sampled_start_length + i * target_interval_gap for i in range(self.n_frames_per_sample)],
            running_length,
            rig_poses.T_rig_world_timestamps_us,
        )

        sampled_sensor_frame_idxs: dict[str, list[int]] = {}
        for camera_id, camera_timestamps_us in camera_frame_timestamps_us.items():
            sampled_sensor_frame_idxs[camera_id] = []
            for frame_timestamp_us in ref_frame_timestamps_us:
                sampled_sensor_frame_idxs[camera_id].append(
                    get_closest_frame_index(camera_timestamps_us, int(frame_timestamp_us))
                )

        return FrameBatchSamplerReturn(sampled_sensor_frame_idxs=sampled_sensor_frame_idxs)


@register("sequential_length")
class SequentialLengthFrameBatchSampler(BaseFrameBatchSampler):
    """
    Sample n_frames_per_sample frames sequentially from the given time_intervals, with a fixed length gap.
    """

    def __init__(self, config: SequentialLengthFrameBatchSamplerConfig):
        self.length_gap: float = config.length_gap
        self.allow_out_of_bounds: bool = config.allow_out_of_bounds
        self.n_frames_per_sample: int = config.n_frames_per_sample
        self.n_samples_per_sequence: int = config.n_samples_per_sequence

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
        assert rig_poses is not None, "Rig poses are required to sample a frame batch"
        assert len(camera_frame_timestamps_us) > 0, "No camera timestamps is provided to the frame batch sampler"
        assert 0 <= sample_idx < self.n_samples_per_sequence, "Sample index out of bounds"

        # Compute running length, rl[i] - rl[j] is length between pose i and j
        running_length = np.cumsum(
            np.linalg.norm(rig_poses.T_rig_worlds[1:, :3, 3] - rig_poses.T_rig_worlds[:-1, :3, 3], axis=-1), axis=0
        )
        running_length = np.concatenate([[0], running_length])

        first_frame_timestamp = min(interval.start for interval in time_intervals)
        i_left = np.searchsorted(rig_poses.T_rig_world_timestamps_us, first_frame_timestamp).item()
        first_frame_length: float = running_length[i_left].item()
        last_frame_timestamp = max(interval.end for interval in time_intervals)
        i_right = np.searchsorted(rig_poses.T_rig_world_timestamps_us, last_frame_timestamp).item() - 1
        last_frame_length: float = running_length[i_right].item()

        # Advance based on sample_idx
        first_frame_length += sample_idx * self.length_gap * self.n_frames_per_sample
        ref_lengths: list[float] = [first_frame_length + i * self.length_gap for i in range(self.n_frames_per_sample)]

        if self.allow_out_of_bounds:
            # If the entire range is out of bounds, return empty batch.
            if ref_lengths[0] > last_frame_length:
                return FrameBatchSamplerReturn(sampled_sensor_frame_idxs={})

            # If part of the range is out of bounds, recompute reference lengths so it include last frame.
            if ref_lengths[-1] > last_frame_length:
                ref_lengths = np.linspace(first_frame_length, last_frame_length, self.n_frames_per_sample).tolist()

        else:
            # check if the last frame is within the length intervals
            if ref_lengths[-1] > last_frame_length:
                raise ValueError(
                    f"The last frame is out of the length intervals: {ref_lengths[-1]} > {last_frame_length}"
                )

        ref_frame_timestamps_us = interpolate_timestamps(
            ref_lengths, running_length, rig_poses.T_rig_world_timestamps_us
        )

        sampled_sensor_frame_idxs: dict[str, list[int]] = {}
        for camera_id, camera_timestamps_us in camera_frame_timestamps_us.items():
            sampled_sensor_frame_idxs[camera_id] = []
            for frame_timestamp_us in ref_frame_timestamps_us:
                sampled_sensor_frame_idxs[camera_id].append(
                    get_closest_frame_index(camera_timestamps_us, int(frame_timestamp_us))
                )

        return FrameBatchSamplerReturn(sampled_sensor_frame_idxs=sampled_sensor_frame_idxs)
