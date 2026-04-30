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

from typing import Literal

from nre.config.base_schema import BaseConfigSchema, Field
from nre.config.checkpoint import ArtifactConfig


class PrimitivePLYExportConfig(BaseConfigSchema):
    enabled: bool = Field(
        default=False,
        description="Whether to enable primitive ply export. The exported ply file contains the preactived density "
        "and scale, and colors represented in SH.",
    )
    density_activation: Literal["sigmoid", "exp"] | None = Field(
        default="sigmoid",
        description="The inv activation function to apply to the densities before exporting.",
    )
    scale_activation: Literal["sigmoid", "exp"] | None = Field(
        default="exp",
        description="The inv activation function to apply to the scales before exporting.",
    )
    color_mode: Literal["rgb", "sh"] = Field(default="sh", description="The color representation to export.")
    apply_affine_mtx: bool = Field(
        default=False,
        description="Whether to apply the affine matrix to the rgb values before exporting. "
        "Only works when exporting a single camera.",
    )
    falloff_sigma_timestamp_us: int | None = Field(
        default=None,
        description="The relative timestamp to apply the falloff sigma, used to filter the gaussians before export. "
        "A value of 0 will use the minimum timestamp, -1 will use the maximum, and None applies no filtering. "
        "Other int values will be added to the minimum timestamp (in us).",
    )
    minimum_density: float | None = Field(
        ge=0.0,
        default=0.0,
        description="Primitives with strictly lower activated density than this will be filtered out.",
    )
    minimum_scale: float | None = Field(
        ge=0.0,
        default=0.0,
        description="Primitives with strictly lower activated scale (in any dimension) than this will be filtered out.",
    )
    minimum_surface_area: float | None = Field(
        ge=0.0,
        default=0.0,
        description="Primitives with strictly lower surface area (scale x^2 + y^2 + z^2) than this will be filtered out.",
    )
    maximum_velocity: float | None = Field(
        default=0.01,
        description="Primitives with strictly higher velocity (m/s) than this will be filtered out.",
    )
    maximum_sky_mask: float | None = Field(
        default=0.5, description="Primitives with a strictly higher sky_mask value will be filtered out."
    )


class PrimitiveMergeConfig(BaseConfigSchema):
    """
    Configuration for primitive merging. It typically contains the following stages:
    1. Transform each primitive to a reference frame (defined by the first chunk); filtering is done per-chunk via model.export_preprocess.
    2. Merge primitives into a single primitive. Optionally de-overlap so that GS from one chunk do not interfere with others.
    3. Postprocess the merged primitive (such as voxelization)
    """

    enabled: bool = Field(default=False, description="Whether to enable primitive merging")

    # Stage 2 options:
    overlap_strategy: Literal["none", "depth_truncation", "two_sigma", "frustum_ownership"] = Field(
        default="two_sigma", description="Strategy for conflicts of gaussians from different chunks"
    )
    dynamic_sigma_threshold: float = Field(
        default=1.0,
        description="For Celsius model, we need to extend the existing length for static scenes (except for two-sigma strategy).",
    )
    depth_truncation_threshold: float = Field(
        default=50.0, description="Z-depth threshold (in meters) for truncating Gaussians to avoid overlap"
    )
    frustum_ownership_max_diff_m: float = Field(
        default=0.0,
        description="Maximum distance in meters between the distances from one GS to non-owned chunks and owned chunks",
    )
    enable_sky_mask: bool = Field(
        default=True,
        description="Whether to enable sky mask (to render sky based on sky token) for the merged primitive. When False, Celsius typically uses model.export_preprocess.keep_sky_gaussians.",
    )

    # Stage 3 options:
    enable_voxelization: bool = Field(
        default=False, description="Whether to apply voxelization to merge nearby Gaussians"
    )
    voxel_size: float = Field(default=0.1, description="Size of voxels for voxelization (in meters)")
    voxel_fusion_mode: Literal["average", "kl_optimal"] = Field(
        default="average",
        description=(
            "How to fuse Gaussians within a voxel. "
            "'average' uses weighted averaging of all attributes (existing behavior). "
            "'kl_optimal' uses moment-matching for position/rotation/scale (correct covariance merge) "
            "while averaging other attributes."
        ),
    )

    def __post_init__(self):
        # When merge is enabled, ensure merged output has sky: either enable_sky_mask (render via token) or
        # model.export_preprocess.keep_sky_gaussians (Celsius) must be True; validated at runtime if needed.
        assert self.frustum_ownership_max_diff_m >= 0.0, (
            f"{self.__class__.__name__}: 'frustum_ownership_max_diff_m' must be non-negative"
        )


class SensorOverrideConfig(BaseConfigSchema):
    """
    Configuration for overriding sensor parameters for rendered video export.
    """

    sensor_id: str
    height: int | None = Field(
        default=None, description="Height of the rendered frame in pixels. None means stays unchanged."
    )
    translation_offset: tuple[float, float, float] | None = Field(
        default=None, description="Translation offset in meters. None means stays unchanged."
    )
    rotation_offset: tuple[float, float, float, bool] | None = Field(
        default=None,
        description="Rotation offset in degrees, last element is rotation first. None means stays unchanged.",
    )
    force_pinhole: bool = Field(default=False, description="Whether to force pinhole camera model.")


class RenderVideoConfig(BaseConfigSchema):
    """
    Configuration for rendered video export.
    """

    enabled: bool = Field(default=False, description="Whether to enable rendered video export.")
    fps: int = Field(default=30, description="Frames per second for the rendered video.")

    depth: bool = Field(default=True, description="Whether to render the depth map.")
    override_sensors: list[SensorOverrideConfig] | None = Field(
        default=None, description="List of sensors to override. None means use original sensors."
    )


class PredictConfig(BaseConfigSchema):
    """
    Configuration for inference functionality typically used only in "predict" mode.
    """

    chunk_size: int = Field(
        default=1, description="Size of the chunk/mini-batch for reconstruction for large batch sizes."
    )
    primitive_merge: PrimitiveMergeConfig = Field(
        default_factory=PrimitiveMergeConfig, description="Configuration for primitive merging"
    )
    export_ply: PrimitivePLYExportConfig = Field(
        default_factory=PrimitivePLYExportConfig, description="Configuration for primitive ply export"
    )
    artifact: ArtifactConfig = Field(
        default_factory=ArtifactConfig, description="Configuration for artifact (usdz) generation and export"
    )
    render_video: RenderVideoConfig = Field(
        default_factory=RenderVideoConfig, description="Configuration for rendered video export"
    )
