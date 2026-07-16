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

from typing import Literal

from instant_nurec.config_schema.base_schema import BaseConfigSchema, Field


logger = logging.getLogger(__name__)


class NCoreInstantNuRecCuboidTracksParamsConfig(BaseConfigSchema):
    track_min_travel_distance_m: float = Field(default=1.5, ge=0.0)
    track_min_centroid_rig_dist_m: float = Field(
        default=3.0,
        ge=0.0,
        description="Distance threshold for cubic tracks to be considered self-classifications to skip [m]",
    )
    track_extrapolate_timestamps_us: int = Field(
        default=int(1e6),
        description="Extrapolate the track by this many timestamps in the past and future (to improve interpolation coverage)",
    )
    track_label_source: Literal["AUTOLABEL", "EXTERNAL", "GT_SYNTHETIC", "GT_ANNOTATION"] = Field(default="AUTOLABEL")


class AdaptiveSequentialFrameBatchSamplerConfig(BaseConfigSchema):
    """Adaptive sequential frame-batch sampler config.

    The released checkpoint was trained with 18 input frames per sample.
    """

    n_frames_per_sample: int = Field(
        default=18,
        gt=0,
        description="Number of frames in each model input sample.",
    )

    n_samples_per_sequence: int = Field(
        default=8,
        description=(
            "Maximum number of time-chunks processed per sequence. Clips longer than "
            "n_samples_per_sequence * n_frames_per_sample * max_frame_gap_timestamp_us "
            "(approx 108 s with defaults) are truncated and the sampler logs a WARNING "
            "naming the dropped chunk count and the value needed to cover the full clip -- "
            "bump this for long clips."
        ),
    )
    max_frame_gap_timestamp_us: int = Field(
        default=750_000,
        description=(
            "Max spacing (us) between adjacent frames inside one sample. Multiplied by "
            "n_frames_per_sample, this sets each chunk's max timespan; tighter values "
            "mean more chunks needed to cover a clip."
        ),
    )


class CameraSubsamplerConfig(BaseConfigSchema):
    """Image dimensions expected by the released model."""

    frame_width: int = Field(
        default=784,
        gt=0,
        description="Width after aspect-preserving resize and center crop.",
    )
    frame_height: int = Field(
        default=448,
        gt=0,
        description="Height after aspect-preserving resize and center crop.",
    )


class NCoreInstantNuRecDatasetConfig(BaseConfigSchema):
    """Predict-side config for the NCorev4 dataset loader.

    Required field: ``ncore_json_paths`` — an explicit list of absolute
    sequence-metadata JSON paths. The CLI's ``resolve_ncore_paths``
    helper resolves a ``--ncore-path`` (single ``.json`` or ``.lst``)
    into this list before constructing the config.
    """

    ncore_json_paths: list[str] = Field(
        description="Absolute paths to ncorev4 sequence metadata JSON files.",
        min_length=1,
    )
    open_consolidated: bool = Field(default=True)
    camera_max_fov_deg: float = Field(
        default=190.0,
        description="For FTheta and OpenCVFishEye camera models, this is used to control the max camera angle, such that "
        "max_angle = min(max_fov / 2, camera_model.max_angle). This will make boundary pixels classified as invalid",
    )
    n_camera_mask_dilation_iterations: int = Field(default=10)

    camera_subsampler: CameraSubsamplerConfig = Field(
        default_factory=CameraSubsamplerConfig,
        description="Image resize and crop applied before model inference.",
    )

    context_camera_ids: list[str] = Field(
        default_factory=lambda: ["camera_front_wide_120fov"],
        description="A list of camera ids, such as `camera_front_wide_120fov`",
    )

    frame_batch_sampler: AdaptiveSequentialFrameBatchSamplerConfig = Field(
        default_factory=AdaptiveSequentialFrameBatchSamplerConfig,
    )
    supervision_camera_ids: list[str] = Field(
        default_factory=lambda: ["camera_front_wide_120fov"],
        description="A list of camera ids, such as `camera_front_wide_120fov`. This is also used to determine the canonical order of cameras in unique sensor idx",
    )

    cuboid_tracks_params: NCoreInstantNuRecCuboidTracksParamsConfig = Field(
        default_factory=NCoreInstantNuRecCuboidTracksParamsConfig,
    )




class InstantNuRecSplitsConfig(BaseConfigSchema):
    """Splits configuration. Predict-only keeps just the predict
    split; pydantic ``extras="ignore"`` drops the train/val/test entries
    that the pretrained ``parsed.yaml`` still carries."""

    predict: NCoreInstantNuRecDatasetConfig | None = Field(default=None, description="Dataset to use in prediction mode")
