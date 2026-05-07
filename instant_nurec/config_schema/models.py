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

"""Public model configs.

Only the post-JIT runtime knobs live here: architecture parameters
(encoder dims, decoder dims, activation shifts, scene_rescale,
patch_shape, track_padding_m, ...) are baked into the JIT artifact at
trace time and therefore moved to
``instant_nurec_internal.config_schema.models``. The runtime adapter
reads ``scene_rescale`` and ``cuboids_dims_padding`` directly off the
loaded JIT module's buffers; users have no reason to override either.
"""

from __future__ import annotations

from instant_nurec.config_schema.base_schema import BaseConfigSchema, Field


class PrimitiveExportPreprocessConfig(BaseConfigSchema):
    """Per-chunk primitive preprocessing applied *after* the JIT call,
    before export and (optionally) chunk merge."""

    density_prune_threshold: float = Field(
        default=0.01, description="Density threshold for pruning Gaussians in each chunk."
    )


class KelvinModelConfig(BaseConfigSchema):
    """Slim runtime config for the Kelvin model.

    Architecture-side fields (encoder/decoder/sky/activations/patch_shape/
    scene_rescale/track_padding_m) are baked into the JIT artifact and
    live in ``instant_nurec_internal.config_schema.models.KelvinFullModelConfig``;
    only the post-JIT preprocess knob is user-exposed here.
    """

    export_preprocess: PrimitiveExportPreprocessConfig = Field(
        default_factory=PrimitiveExportPreprocessConfig,
        description="Per-chunk preprocess options for predict/export (filtering before merge or export).",
    )
