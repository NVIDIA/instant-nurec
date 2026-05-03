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


class NCoreNRMCuboidTracksParamsConfig(BaseConfigSchema):
    lidar_id: str
    track_min_travel_distance_m: float = Field(ge=0.0)
    track_min_centroid_rig_dist_m: float = Field(
        ge=0.0,
        description="Distance threshold for cubic tracks to be considered self-classifications to skip [m]",
    )
    track_extrapolate_timestamps_us: int = Field(
        default=int(1e6),
        description="Extrapolate the track by this many timestamps in the past and future (to improve interpolation coverage)",
    )
    track_label_source: Literal["AUTOLABEL", "EXTERNAL", "GT_SYNTHETIC", "GT_ANNOTATION"]


class AdaptiveSequentialFrameBatchSamplerConfig(BaseConfigSchema):
    n_frames_per_sample: int = Field(
        description="Number of frames in each dataset sample (i.e. one return from get_item of the dataset)"
    )
    n_samples_per_sequence: int = Field(
        description="Number of samples to return for each sequence (i.e. one recording from the full dataset)"
    )
    max_frame_gap_timestamp_us: int = Field(
        description="Maximum gap between adjacent sampled frames in the batch, in microseconds."
    )


class CameraSubsamplerConfig(BaseConfigSchema):
    frame_width: int = Field(description="Width of the image to subsample (aspect-preserving center crop)")
    frame_height: int = Field(description="Height of the image to subsample (aspect-preserving center crop)")


class NCoreNRMDatasetConfig(BaseConfigSchema):
    """Predict-side config for the NCorev4 dataset loader."""

    ncore_json_list_path: str = Field(
        description="The path to a file that contains the list of sequence meta json files to load.",
    )
    ncore_json_base_path: str = Field(
        description="Base path used to resolve relative paths in the ncore_json_list_path.",
    )
    open_consolidated: bool = Field(default=True)
    camera_max_fov_deg: float = Field(
        default=190.0,
        description="For FTheta and OpenCVFishEye camera models, this is used to control the max camera angle, such that "
        "max_angle = min(max_fov / 2, camera_model.max_angle). This will make boundary pixels classified as invalid",
    )
    n_camera_mask_dilation_iterations: int = Field(default=10)

    camera_subsampler: CameraSubsamplerConfig = Field(description="Image resize to a given height/width.")

    context_camera_ids: list[str] = Field(
        description="A list of camera ids, such as `camera_front_wide_120fov`",
    )

    frame_batch_sampler: AdaptiveSequentialFrameBatchSamplerConfig
    supervision_camera_ids: list[str] = Field(
        description="A list of camera ids, such as `camera_front_wide_120fov`. This is also used to determine the canonical order of cameras in unique sensor idx",
    )

    # Note: this is not the same as the other cuboid tracks params
    cuboid_tracks_params: NCoreNRMCuboidTracksParamsConfig




class NRMSplitsConfig(BaseConfigSchema):
    """NRM splits configuration. Predict-only standalone keeps just the predict
    split; pydantic extras="ignore" drops the train/val/test entries that the
    pretrained parsed.yaml still carries."""

    predict: NCoreNRMDatasetConfig | None = Field(default=None, description="Dataset to use in prediction mode")
