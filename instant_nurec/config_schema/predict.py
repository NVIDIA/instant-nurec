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
