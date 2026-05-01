# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Annotated, Literal, Optional, Self

import numpy as np

from pydantic import model_validator

from nre.config.base_schema import BaseConfigSchema, Field
from nre.utils.misc import unpack_optional


logger = logging.getLogger(__name__)


class NCoreNRMAuxDataConfig(BaseConfigSchema):
    enabled: bool
    enabled_context: bool
    semantic_segmentation: bool
    depth: bool | str = Field(
        default=False,
        description=(
            "True to load depth from the ncore aux data, False to disable, "
            "or a path to load depth from another aux file store. "
            "Supported template variable: {{clip_id}}."
        ),
    )
    egomask: bool


class NCoreNRMCuboidTracksParamsConfig(BaseConfigSchema):
    lidar_id: Optional[str]
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


class LidarFrameBatchParamsConfig(BaseConfigSchema):
    gap_from_image_us: int = Field(description="Max gap from image in microseconds", default=0)


class AdaptiveSequentialFrameBatchSamplerConfig(BaseConfigSchema):
    name: Literal["adaptive_sequential"] = "adaptive_sequential"
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


class BaseNCoreNRMDatasetConfig(BaseConfigSchema):
    """Base config for NCore-based datasets. Subclasses must define `name`."""

    class ExternalSupervisionCameraIdConfig(BaseConfigSchema):
        """Supervision cameras that are loaded from external ncore files"""

        ncore_path: str = Field(
            description="Path to the external ncore file, relative to the current sequence json file."
        )
        camera_id: str
        unique_sensor_idx: int = Field(
            description="Unique sensor index for the camera, used to determine the canonical order of cameras.",
        )
        sample_ratio: float = Field(
            default=1.0,
            description="Sample ratio of the camera on top of the supervision frame batch sampling.",
        )

    # V3 sequence loader parameters
    cuboid_loading_max_workers: Optional[int] = Field(
        description="If specified, set the max_workers for thread parallel cuboid loading in the V3 sequence loader"
        " (automatically infered number for 'null' - serial execution with '0')",
        default=0,
    )

    # V4 component group names
    poses_component_group: str = Field(
        description="Name of the V4 component group for poses",
        default="default",
    )
    intrinsics_component_group: str = Field(
        description="Name of the V4 component group for intrinsics",
        default="default",
    )
    masks_component_group: str = Field(
        description="Name of the V4 component group for masks",
        default="default",
    )
    cuboids_component_group: str = Field(
        description="Name of the V4 component group for cuboids",
        default="default",
    )

    # Deprecated support for old configs -- TODO [JH]: to be removed in the future.
    ncore_json_list_path: str = Field(
        description="The path to a file that contains the list of sequence meta json files to load.",
    )
    ncore_json_base_path: Optional[str] = Field(
        default=None,
        description="If specified, will be used as a base path to resolve relative paths in the ncore_json_list_path",
    )
    ncore_json_paths: Optional[list[str]] = Field(
        default=None,
        description="If specified, will be the list of data files to load. Used to quickly run test on smaller number of datapoints.",
    )
    s3_block_size_mb: int = Field(default=50, description="The block size in MB for S3 universal paths.")
    s3_cache_type: str = Field(default="readahead", description="The cache type for S3 universal paths.")

    subrange_json_path: Optional[str] = Field(
        default=None,
        description="If specified, will only load the *intersection* of the sequences specified in the json_list_path and "
        "the ones in the keys of this subrange. The json file is a dictionary with format {sequence_name: [subrange_1, "
        "subrange_2, ...]}, where each subrange is a tuple of start/end time points normalized to 0~1 of the sequence length.",
    )
    open_consolidated: bool = Field(default=True)  # Does not appear to be used
    camera_max_fov_deg: float = Field(
        default=190.0,
        description="For FTheta and OpenCVFishEye camera models, this is used to control the max camera angle, such that "
        "max_angle = min(max_fov / 2, camera_model.max_angle). This will make boundary pixels classified as invalid",
    )
    n_camera_mask_dilation_iterations: int = Field(default=10)

    aux_data: NCoreNRMAuxDataConfig

    camera_subsampler: CameraSubsamplerConfig = Field(description="Image resize to a given height/width.")

    context_camera_ids: list[str | ExternalSupervisionCameraIdConfig] = Field(
        description="A list of camera ids, such as `camera_front_wide_120fov`",
    )

    frame_batch_sampler: AdaptiveSequentialFrameBatchSamplerConfig
    supervision_camera_ids: list[str | ExternalSupervisionCameraIdConfig] = Field(
        description="A list of camera ids, such as `camera_front_wide_120fov`. This is also used to determine the canonical order of cameras in unique sensor idx",
    )

    # Note: this is not the same as the other cuboid tracks params
    cuboid_tracks_params: NCoreNRMCuboidTracksParamsConfig

    lidar_frame_batch: LidarFrameBatchParamsConfig = Field(
        description="Parameters for the lidar frame batch", default_factory=LidarFrameBatchParamsConfig
    )

    compute_rendering_data: bool = Field(
        default=True,
        description="Whether to pre-compute the rendering data (e.g. rays) in the dataloader (CPU). Turning this off will significantly reduce data loader memory consumption.",
    )

    cache_loaders_and_sensors: bool = Field(
        default=False,
        description="If True, cache the result of _get_loaders_and_sensors (one entry keyed by ncore_json_path). "
        "This helps cut loading time during prediction when multiple consecutive batches share the same sequence loader instance.",
    )

    camera_id_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Map logical camera ids to ids used inside ncore archives if they are different.",
    )
    lidar_id_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Map logical lidar ids to ids used inside ncore archives if they are different.",
    )

    def model_post_init(self, __context) -> None:
        assert self.ncore_json_list_path is not None or self.ncore_json_paths is not None, (
            "Either ncore_json_list_path or ncore_json_paths must be provided"
        )

    def concretize(self, epoch: int, rng: np.random.Generator) -> Self:
        """Predict-only standalone has no augmentations; concretize is a no-op
        passthrough. Self-invented: NRE used this hook to sample augmentation
        tiers per epoch."""
        del epoch, rng
        return self


class NCoreNRMDatasetConfig(BaseNCoreNRMDatasetConfig):
    """Standard NCore dataset config."""

    name: Literal["nrm-ncore"]


NRMSplitConfig = Annotated[NCoreNRMDatasetConfig, Field(discriminator="name")]


class NRMSplitsConfig(BaseConfigSchema):
    """NRM splits configuration. Predict-only standalone keeps just the predict
    split; pydantic extras="ignore" drops the train/val/test entries that the
    pretrained parsed.yaml still carries."""

    name: Literal["nrm"]

    predict: NRMSplitConfig | None = Field(default=None, description="Dataset to use in prediction mode")
