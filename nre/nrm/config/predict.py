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
    overlap_strategy: Literal["none", "frustum_ownership"] = Field(
        default="frustum_ownership", description="Strategy for conflicts of gaussians from different chunks"
    )
    frustum_ownership_max_diff_m: float = Field(
        default=0.0,
        description="Maximum distance in meters between the distances from one GS to non-owned chunks and owned chunks",
    )
    enable_sky_mask: bool = Field(
        default=True,
        description="Whether to enable sky mask (to render sky based on sky token) for the merged primitive. When False, Celsius typically uses model.export_preprocess.keep_sky_gaussians.",
    )

    def __post_init__(self):
        # When merge is enabled, ensure merged output has sky: either enable_sky_mask (render via token) or
        # model.export_preprocess.keep_sky_gaussians (Celsius) must be True; validated at runtime if needed.
        assert self.frustum_ownership_max_diff_m >= 0.0, (
            f"{self.__class__.__name__}: 'frustum_ownership_max_diff_m' must be non-negative"
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
