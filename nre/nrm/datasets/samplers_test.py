# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np
import pytest

from nre.nrm.config.dataset import (
    AdaptiveSequentialFrameBatchSamplerConfig,
    SupervisionFrameBatchParamsConfig,
    UniformFrameBatchSamplerConfig,
    VaryingIntervalFrameBatchSamplerConfig,
)
from nre.nrm.datasets.samplers import (
    AdaptiveSequentialFrameBatchSampler,
    BaseFrameBatchSampler,
    FrameBatchSamplerReturn,
    UniformFrameBatchSampler,
    VaryingIntervalFrameBatchSampler,
    sample_supervision_frame_batch,
)
from nre.utils.types import HalfClosedInterval


@pytest.mark.parametrize(
    "context_sampler_config, sampler_class",
    [
        (
            VaryingIntervalFrameBatchSamplerConfig(
                n_frames_per_sample=3,
                n_samples_per_sequence=1,
                sequence_gap_timestamp_us_min=0,
                sequence_gap_timestamp_us_max=10,
            ),
            VaryingIntervalFrameBatchSampler,
        ),
        (
            UniformFrameBatchSamplerConfig(
                n_frames_per_sample=3,
                n_samples_per_sequence=1,
                frame_gap_timestamp_us=1,
            ),
            UniformFrameBatchSampler,
        ),
    ],
)
def test_varying_interval_frame_batch_sampler(context_sampler_config, sampler_class) -> None:
    camera_id = "test_camera"
    camera_frame_timestamps = {camera_id: np.arange(100)}

    context_sampler: BaseFrameBatchSampler = sampler_class(context_sampler_config)
    rng = np.random.default_rng(seed=0)

    context_samples = context_sampler.sample_frame_batch(
        rng, 0, camera_frame_timestamps, [HalfClosedInterval(0, 20)], None
    )
    supervision_samples = sample_supervision_frame_batch(
        config=SupervisionFrameBatchParamsConfig(n_frames_per_sample=3, sample_strategy="random"),
        rng=rng,
        context_frame_batch=context_samples,
        sensor_frame_timestamps_us=camera_frame_timestamps,
        supervision_sensor_ids=[camera_id],
        sensor_sample_ratios=[1.0],
    )

    context_range = range(
        min(context_samples.sampled_sensor_frame_idxs[camera_id]),
        max(context_samples.sampled_sensor_frame_idxs[camera_id]) + 1,
    )
    assert all(frame_idx in context_range for frame_idx in supervision_samples.sampled_sensor_frame_idxs[camera_id])


def test_adaptive_sequential_frame_batch_sampler() -> None:
    camera_id = "test_camera"
    camera_frame_timestamps = {camera_id: np.arange(100)}
    context_sampler = AdaptiveSequentialFrameBatchSampler(
        AdaptiveSequentialFrameBatchSamplerConfig(
            n_frames_per_sample=5,
            n_samples_per_sequence=3,
            max_frame_gap_timestamp_us=10,
        )
    )
    rng = np.random.default_rng(seed=0)

    context_samples_0 = context_sampler.sample_frame_batch(
        rng, 0, camera_frame_timestamps, [HalfClosedInterval(0, 100)], None
    )
    context_samples_1 = context_sampler.sample_frame_batch(
        rng, 1, camera_frame_timestamps, [HalfClosedInterval(0, 100)], None
    )
    context_samples_2 = context_sampler.sample_frame_batch(
        rng, 2, camera_frame_timestamps, [HalfClosedInterval(0, 100)], None
    )

    assert context_sampler.num_samples_per_sequence == 3
    sampled_frame_idxs_0 = context_samples_0.sampled_sensor_frame_idxs[camera_id]
    sampled_frame_idxs_1 = context_samples_1.sampled_sensor_frame_idxs[camera_id]
    assert sampled_frame_idxs_0 == [0, 10, 20, 30, 40]
    assert sampled_frame_idxs_1 == [50, 60, 70, 80, 90]
    assert sampled_frame_idxs_1[0] - sampled_frame_idxs_0[-1] == 10
    assert context_samples_2.sampled_sensor_frame_idxs == {}


def test_sample_supervision_frame_batch_include_context_frames() -> None:
    camera_id = "cam"
    camera_frame_timestamps = {camera_id: np.arange(100, dtype=np.int64)}
    rng = np.random.default_rng(seed=42)
    context = FrameBatchSamplerReturn(sampled_sensor_frame_idxs={camera_id: [10, 25, 40]})

    supervision = sample_supervision_frame_batch(
        config=SupervisionFrameBatchParamsConfig(
            n_frames_per_sample=2,
            sample_strategy="random",
            include_context_frames=True,
        ),
        rng=rng,
        context_frame_batch=context,
        sensor_frame_timestamps_us=camera_frame_timestamps,
        supervision_sensor_ids=[camera_id],
        sensor_sample_ratios=[1.0],
    )

    sup_idxs = set(supervision.sampled_sensor_frame_idxs[camera_id])
    assert {10, 25, 40}.issubset(sup_idxs)
