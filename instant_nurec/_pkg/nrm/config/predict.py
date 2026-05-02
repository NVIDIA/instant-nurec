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

from instant_nurec._pkg.config.base_schema import BaseConfigSchema, Field


class PrimitiveMergeConfig(BaseConfigSchema):
    """
    Configuration for primitive merging. It typically contains the following stages:
    1. Transform each primitive to a reference frame (defined by the first chunk); filtering is done per-chunk via model.export_preprocess.
    2. Merge primitives into a single primitive with frustum-ownership de-overlap so GS from one chunk do not interfere with others.
    """

    enabled: bool = Field(default=False, description="Whether to enable primitive merging")
    frustum_ownership_max_diff_m: float = Field(
        default=0.0,
        description="Maximum distance in meters between the distances from one GS to non-owned chunks and owned chunks",
        ge=0.0,
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
