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
import re

from typing import Annotated, Any, Dict, List, Literal, Optional, Self, Union

import numpy as np

from pydantic import RootModel, model_validator

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


class BaseFrameBatchSamplerConfig(BaseConfigSchema):
    enabled: bool = Field(default=True, description="Whether to enable this frame batch sampler")
    n_frames_per_sample: int = Field(
        description="Number of frames in each dataset sample (i.e. one return from get_item of the dataset)"
    )
    n_samples_per_sequence: int = Field(
        description="Number of samples to return for each sequence (i.e. one recording from the full dataset)"
    )


class UniformFrameBatchSamplerConfig(BaseFrameBatchSamplerConfig):
    name: Literal["uniform"] = "uniform"

    frame_gap_timestamp_us: int = Field(description="Gap between frames in the batch, in microseconds.")


class AdaptiveSequentialFrameBatchSamplerConfig(BaseFrameBatchSamplerConfig):
    name: Literal["adaptive_sequential"] = "adaptive_sequential"
    max_frame_gap_timestamp_us: int = Field(
        description="Maximum gap between adjacent sampled frames in the batch, in microseconds."
    )


class VaryingIntervalFrameBatchSamplerConfig(BaseFrameBatchSamplerConfig):
    name: Literal["varying_interval"] = "varying_interval"

    sequence_gap_timestamp_us_min: int = Field(
        description="Minimum gap between sequences in the batch, in microseconds."
    )
    sequence_gap_timestamp_us_max: int = Field(
        description="Maximum gap between sequences in the batch, in microseconds."
    )


class UniformLengthFrameBatchSamplerConfig(BaseFrameBatchSamplerConfig):
    name: Literal["uniform_length"] = "uniform_length"
    length_gap_min: float = Field(description="Minimum length gap between adjacent frames in the batch, in meters.")
    length_gap_max: float = Field(description="Maximum length gap between adjacent frames in the batch, in meters.")


class LinearWithIndexFrameBatchSamplerConfig(BaseFrameBatchSamplerConfig):
    name: Literal["linear_with_index"] = "linear_with_index"
    first_frame_timestamp: int = Field(description="Timestamp of the first frame in the batch, in microseconds.")
    total_time_gap: int = Field(
        description="Total time gap between first and last frame in the batch, in microseconds."
    )


class SequentialFrameBatchSamplerConfig(BaseFrameBatchSamplerConfig):
    name: Literal["sequential"] = "sequential"
    first_frame_timestamp: int = Field(description="Timestamp of the first frame in the batch, in microseconds.")
    frame_gap_timestamp_us: int = Field(description="Gap between consecutive frames in the batch, in microseconds.")
    allow_out_of_bounds: bool = Field(description="Whether to allow out of bounds sampling.", default=False)


class SequentialLengthFrameBatchSamplerConfig(BaseFrameBatchSamplerConfig):
    name: Literal["sequential_length"] = "sequential_length"
    length_gap: float = Field(description="Gap between consecutive frames in the batch, in meters.")
    allow_out_of_bounds: bool = Field(description="Whether to allow out of bounds sampling.", default=False)


BatchSamplerConfigType = Annotated[
    Union[
        UniformFrameBatchSamplerConfig,
        AdaptiveSequentialFrameBatchSamplerConfig,
        VaryingIntervalFrameBatchSamplerConfig,
        LinearWithIndexFrameBatchSamplerConfig,
        SequentialFrameBatchSamplerConfig,
        UniformLengthFrameBatchSamplerConfig,
        SequentialLengthFrameBatchSamplerConfig,
    ],
    Field(discriminator="name"),
]


class CameraSubsamplerConfig(BaseConfigSchema):
    frame_width: int = Field(description="Width of the image to subsample (aspect-preserving center crop)")
    frame_height: int = Field(description="Height of the image to subsample (aspect-preserving center crop)")


class SupervisionFrameBatchParamsConfig(BaseConfigSchema):
    n_frames_per_sample: int = 12
    prepend_timestamps_us: int = 0
    append_timestamps_us: int = 0
    sample_strategy: Literal["random", "stratified"] = "random"
    include_context_frames: bool = Field(
        default=False,
        description="If True, every context frame index is unioned into the supervision batch per sensor "
        "(for sensors listed in supervision_sensor_ids that also appear in the context batch).",
    )
    camera_subsampler: CameraSubsamplerConfig | None = Field(
        default=None, description="Image resize to a given height/width."
    )


class AugmentationItemConfig(BaseConfigSchema):
    values: list[Any] = Field(description="List of values to sample from")
    weights: list[float] = Field(description="Weights for the values")
    start_epoch: int = Field(description="Epoch from which this augmentation tier is active (inclusive).")

    @model_validator(mode="after")
    def _validate_values_weights_lengths(self) -> Self:
        if len(self.values) != len(self.weights):
            raise ValueError(
                f"AugmentationItemConfig requires len(values) == len(weights), "
                f"got len(values)={len(self.values)} and len(weights)={len(self.weights)}."
            )
        return self

    def sample_value(self, rng: np.random.Generator) -> Any:
        """Sample one value from this item according to its weights."""
        # Avoid advancing RNG state if there's only one value.
        if len(self.values) == 1:
            return self.values[0]
        weights = np.asarray(self.weights, dtype=np.float64)
        weights = weights / weights.sum()
        idx = rng.choice(len(self.values), p=weights)
        return self.values[idx]


class AugmentationsConfig(BaseConfigSchema):
    """Only includes implemented augmentation features: camera_subsampler and supervision_frame_resolution."""

    camera_subsampler: Dict[str, AugmentationItemConfig] | None = Field(
        default=None,
        description="Tier name (e.g. 'low_res', 'high_res') -> augmentation item for the camera subsampler.",
    )
    context_sensor_idxs: Dict[str, AugmentationItemConfig] | None = Field(
        default=None,
        description="Tier name -> unique sensor index to be sub-selected from the context cameras.",
    )
    context_n_frames_per_sample: Dict[str, AugmentationItemConfig] | None = Field(
        default=None,
        description="Tier name -> number of frames to sample from the context cameras.",
    )
    supervision_frame_resolution: Dict[str, AugmentationItemConfig] | None = Field(
        default=None,
        description="Tier name -> augmentation item for the supervision frame resolution.",
    )
    max_context_pixels: int | None = Field(
        default=None,
        description="Maximum number of context pixels to sample from the context cameras. This is checked during the concretization step, and if "
        "the total number of pixels exceeds this value, will re-sample another configuration from the tier.",
    )

    @model_validator(mode="after")
    def _validate_tiers_cover_epoch_zero(self) -> Self:
        """Every tier group must include at least one tier with start_epoch <= 0, so
        `pick_active_tier` has a defined result at the beginning of training."""
        for field_name, tiers in (
            ("camera_subsampler", self.camera_subsampler),
            ("supervision_frame_resolution", self.supervision_frame_resolution),
        ):
            if tiers is None or not tiers:
                continue
            min_start = min(tier.start_epoch for tier in tiers.values())
            if min_start > 0:
                raise ValueError(
                    f"AugmentationsConfig.{field_name} must contain a tier with start_epoch <= 0 "
                    f"(so it is active at epoch 0), but the minimum start_epoch among the "
                    f"provided tiers is {min_start}. Lower the start_epoch of one tier (typically "
                    f"the 'baseline' tier) to 0 or a negative value."
                )
        return self

    @property
    def is_empty(self) -> bool:
        return (
            self.camera_subsampler is None
            and self.supervision_frame_resolution is None
            and self.context_sensor_idxs is None
            and self.context_n_frames_per_sample is None
        )

    def pick_active_tier(
        self, tiers: Dict[str, AugmentationItemConfig] | None, epoch: int
    ) -> tuple[str, AugmentationItemConfig] | None:
        """Pick the tier active at this epoch: max(start_epoch) among tiers with start_epoch <= epoch. Returns None if tiers is None or no tier is active."""
        if tiers is None:
            return None
        active = [(k, v) for k, v in tiers.items() if v.start_epoch <= epoch]
        if not active:
            return None
        return max(active, key=lambda kv: kv[1].start_epoch)


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

    frame_batch_samplers: Dict[str, BatchSamplerConfigType]
    supervision_camera_ids: list[str | ExternalSupervisionCameraIdConfig] = Field(
        description="A list of camera ids, such as `camera_front_wide_120fov`. This is also used to determine the canonical order of cameras in unique sensor idx",
    )

    supervision_frame_batch: SupervisionFrameBatchParamsConfig = Field(
        description="As our supervision always enforces stratified sampling, so no sampler needed here"
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

    augmentations: AugmentationsConfig | None = Field(
        default=None, description="Augmentations to apply to the dataset."
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
        if self.supervision_frame_batch.camera_subsampler is None:
            self.supervision_frame_batch.camera_subsampler = self.camera_subsampler

    @property
    def is_concrete(self) -> bool:
        return self.augmentations is None or self.augmentations.is_empty

    def concretize(self, epoch: int, rng: np.random.Generator) -> Self:
        if self.is_concrete:
            return self
        aug = unpack_optional(self.augmentations)

        max_retries = 16
        for _ in range(max_retries):
            updates: dict[str, Any] = {"augmentations": None}

            tier_result = aug.pick_active_tier(aug.camera_subsampler, epoch)
            if tier_result is not None:
                _tier_name, tier = tier_result
                updates["camera_subsampler"] = CameraSubsamplerConfig.model_validate(tier.sample_value(rng))

            tier_result = aug.pick_active_tier(aug.supervision_frame_resolution, epoch)
            if tier_result is not None:
                _tier_name, tier = tier_result
                subsampler = CameraSubsamplerConfig.model_validate(tier.sample_value(rng))
                updates["supervision_frame_batch"] = self.supervision_frame_batch.model_copy(
                    update={"camera_subsampler": subsampler}
                )

            tier_result = aug.pick_active_tier(aug.context_sensor_idxs, epoch)
            if tier_result is not None:
                _tier_name, tier = tier_result
                selected_idxs: list[int] = tier.sample_value(rng)

                def _unique_sensor_idx(cam: str | BaseNCoreNRMDatasetConfig.ExternalSupervisionCameraIdConfig) -> int:
                    if isinstance(cam, BaseNCoreNRMDatasetConfig.ExternalSupervisionCameraIdConfig):
                        return cam.unique_sensor_idx
                    return self.supervision_camera_ids.index(cam)

                updates["context_camera_ids"] = [
                    c for c in self.context_camera_ids if _unique_sensor_idx(c) in selected_idxs
                ]
                updates["supervision_camera_ids"] = [
                    c for c in self.supervision_camera_ids if _unique_sensor_idx(c) in selected_idxs
                ]

            tier_result = aug.pick_active_tier(aug.context_n_frames_per_sample, epoch)
            if tier_result is not None:
                _tier_name, tier = tier_result
                n_frames = int(tier.sample_value(rng))
                updates["frame_batch_samplers"] = {
                    name: sampler.model_copy(update={"n_frames_per_sample": n_frames})
                    for name, sampler in self.frame_batch_samplers.items()
                }

            updated = self.model_copy(update=updates)

            if aug.max_context_pixels is not None:
                # height x width x num_cameras x num_frames
                n_pixels = (
                    len(updated.context_camera_ids)
                    * updated.camera_subsampler.frame_height
                    * updated.camera_subsampler.frame_width
                )
                samplers = list(updated.frame_batch_samplers.values())
                assert samplers, "frame_batch_samplers must be non-empty"
                n_frames_set = {bs.n_frames_per_sample for bs in samplers}
                if len(n_frames_set) != 1:
                    raise ValueError(
                        f"All frame_batch_samplers must share the same n_frames_per_sample; got {n_frames_set}"
                    )
                n_pixels *= n_frames_set.pop()
                if n_pixels > aug.max_context_pixels:
                    logger.warning(
                        f"Max context pixels exceeded, re-sampling. Current: {n_pixels / 1e6:.2f}M, Max: {aug.max_context_pixels / 1e6:.2f}M"
                    )
                    continue

            return updated

        raise RuntimeError(
            f"concretize: exhausted {max_retries} retries finding a sample within "
            f"max_context_pixels={aug.max_context_pixels}"
        )


class NCoreNRMDatasetConfig(BaseNCoreNRMDatasetConfig):
    """Standard NCore dataset config."""

    name: Literal["nrm-ncore"]


class DummyNRMDatasetConfig(BaseConfigSchema):
    """
    Dummy dataset as a placeholder for testing-only configs, or to override a mixture to empty.
    """

    name: Literal["nrm-dummy"]


class TestIndexNRMDatasetConfig(BaseConfigSchema):
    """
    TestIndexNRMDataset is a dataset class that returns a batch of data at the given index.
    """

    name: Literal["nrm-test-index"]
    size: int = Field(description="Size of the dataset")


SingleNRMDatasetConfig = Union[
    NCoreNRMDatasetConfig, DummyNRMDatasetConfig, TestIndexNRMDatasetConfig
]


class NRMMixedDatasetConfig(BaseConfigSchema):
    class MixtureComponentConfig(BaseConfigSchema):
        sample_ratio: float = Field(
            description="Sample ratio of the mixture component. Will be renormalized to sum to 1."
        )
        config: SingleNRMDatasetConfig = Field(
            discriminator="name",
            description="Config of the mixture component",
        )

    name: Literal["nrm-mixed"]
    mixture: Dict[str, MixtureComponentConfig] = Field(description="List of mixture components")


NRMSplitConfig = Annotated[
    Union[
        NCoreNRMDatasetConfig,
        NRMMixedDatasetConfig,
        DummyNRMDatasetConfig,
        TestIndexNRMDatasetConfig,
    ],
    Field(discriminator="name"),
]


EPOCH_KEY_REGEX = re.compile(r"^epoch_(\d+)$")


class NRMEpochSplitConfig(RootModel[Dict[str, NRMSplitConfig]]):
    """
    A dict of NRMSplitConfig, where the keys are the epoch numbers: epoch_{idx}.
    For example, we will start to use "epoch_10" config after the 10th epoch.
    """

    root: Dict[str, NRMSplitConfig]

    @model_validator(mode="before")
    @classmethod
    def validate_epoch_keys(cls, v: Any) -> Any:
        """Validate that all keys start with 'epoch_' for epoch-based config switching"""
        for key in v.keys():
            match = EPOCH_KEY_REGEX.match(key)
            if match is None:
                raise ValueError(f"Epoch split config keys must be 'epoch_{{epoch_number}}', got {key}")

        return v

    def milestones(self) -> dict[int, NRMSplitConfig]:
        """Returns `self` with keys parsed to integers."""
        milestones = {}
        for k, v in self.root.items():
            match = EPOCH_KEY_REGEX.match(k)
            assert match is not None, "This should've been caught in validate_epoch_keys"

            milestones[int(match.group(1))] = v

        return milestones

    def last_milestone(self) -> NRMSplitConfig:
        """Returns the last milestone config."""
        return self.root[max(self.root.keys())]


class NRMSplitsConfig(BaseConfigSchema):
    """NRM splits configuration into three splits: train, val, test"""

    name: Literal["nrm"]

    train: NRMSplitConfig | NRMEpochSplitConfig
    val: NRMSplitConfig | NRMEpochSplitConfig
    test: NRMSplitConfig | NRMEpochSplitConfig | None = Field(default=None, description="Dataset to use in test mode")
    predict: NRMSplitConfig | None = Field(default=None, description="Dataset to use in prediction mode")
