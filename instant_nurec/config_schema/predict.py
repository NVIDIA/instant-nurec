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

from instant_nurec.config_schema.base_schema import BaseConfigSchema, Field


class PrimitiveMergeConfig(BaseConfigSchema):
    """
    Configuration for primitive merging. It typically contains the following stages:
    1. Transform each primitive to a reference frame (defined by the first chunk); filtering is done per-chunk via model.export_preprocess.
    2. Merge primitives into a single primitive with frustum-ownership de-overlap so GS from one chunk do not interfere with others.
    3. (Optional) Apply KL-optimal voxelization to collapse co-located Gaussians, iterating until the static-layer count lands in [0.9 * target_n_gaussians, target_n_gaussians].
    """

    enabled: bool = Field(default=False, description="Whether to enable primitive merging")
    frustum_ownership_max_diff_m: float = Field(
        default=5.0,
        description="Maximum distance in meters between the distances from one GS to non-owned chunks and owned chunks",
        ge=0.0,
    )
    enable_voxelization: bool = Field(
        default=False,
        description="Whether to apply KL-optimal voxelization to merge nearby Gaussians post-merge",
    )
    voxel_size: float = Field(
        default=0.1,
        description="Initial voxel edge length (scene units) for the iterative voxelization search; doubled when the result exceeds target_n_gaussians, halved when below 0.9 * target_n_gaussians.",
        gt=0.0,
    )
    target_n_gaussians: int = Field(
        default=2_000_000,
        description="Target post-voxelization static-Gaussian count. The iterative search returns when the count lands in [0.9 * target, target].",
        gt=0,
    )
    max_voxelization_iterations: int = Field(
        default=20,
        description="Cap for the iterative voxel-size search. When exceeded, the latest voxelization is returned with a WARNING log.",
        gt=0,
    )


class PredictConfig(BaseConfigSchema):
    """
    Configuration for inference functionality typically used only in "predict" mode.
    """

    primitive_merge: PrimitiveMergeConfig = Field(
        default_factory=PrimitiveMergeConfig, description="Configuration for primitive merging"
    )
