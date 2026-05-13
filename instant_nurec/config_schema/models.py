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

"""Public model configs."""

from __future__ import annotations

from instant_nurec.config_schema.base_schema import BaseConfigSchema, Field


class PrimitiveExportPreprocessConfig(BaseConfigSchema):
    """Per-chunk primitive preprocessing applied before export and
    (optionally) chunk merge."""

    density_prune_threshold: float = Field(
        default=0.01, description="Density threshold for pruning Gaussians in each chunk."
    )


class KelvinModelConfig(BaseConfigSchema):
    """Slim runtime config; only ``export_preprocess`` is user-exposed."""

    export_preprocess: PrimitiveExportPreprocessConfig = Field(
        default_factory=PrimitiveExportPreprocessConfig,
        description="Per-chunk preprocess options for predict/export (filtering before merge or export).",
    )
