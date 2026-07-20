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

from typing import List, Literal, Tuple

from instant_nurec.config_schema.base_schema import BaseConfigSchema, Field


class PrimitiveExportPreprocessConfig(BaseConfigSchema):
    """
    Config for per-chunk primitive preprocessing before export (and before merge).
    Used by preprocess_for_export(); not part of merge logic.
    """

    density_prune_threshold: float = Field(
        default=0.01, description="Density threshold for pruning Gaussians in each chunk."
    )


class GaussiansActivationConfig(BaseConfigSchema):
    """
    Configuration for activation functions used in neural reconstruction models.
    """

    # Opacity activation parameters
    opacity_shift: float = Field(default=-2.0, description="Shift parameter for opacity sigmoid activation")

    # Scale activation parameters
    scale_shift_log_ratio: float = Field(default=-2.9, description="Shift parameter for scale activation")
    scale_max: float = Field(default=0.045, description="Maximum scale value")
    scale_min: float = Field(
        default=0.0,
        description="Minimum scale value (clamp applied after exp). Use 0.01 when using 3DGUT renderer to avoid NaN gradients.",
    )




class KelvinDAv3EncoderConfig(BaseConfigSchema):
    depth: int = Field(default=12)
    n_heads: int = Field(default=12)
    embed_dim: int = Field(default=1536)
    take_block_indices: List[int] = Field(default_factory=lambda: [5, 7, 9, 11])
    aa_start_block_idx: int = Field(default=4)
    checkpointing: Literal["all", "local", "none"] = Field(
        default="all", description="Whether to checkpoint the encoder"
    )


class KelvinDPTDecoderConfig(BaseConfigSchema):
    dpt_dim: int = Field(default=128)
    dpt_reassemble_hidden_dims: List[int] = Field(default_factory=lambda: [96, 192, 384, 768])

    checkpointing: bool = Field(default=True, description="Whether to use checkpointing for the DPT decoder")
    dpt_chunk_size: int = Field(
        default=4, description="Chunk size for the DPT decoder. Used for saving memory. -1 to disable."
    )

    # Motion-related:
    time_encoding_dim: int = Field(default=256, description="Dimension of the time sinusoidal encoding")
    motion_depth: int = Field(default=1, description="Depth of the motion head (V-DPM setup is equivalent to 8)")

    def model_post_init(self, __context) -> None:
        assert self.dpt_dim > 0, "DPT dimension must be positive"


class KelvinSkyCubemapDecoderConfig(BaseConfigSchema):
    cubemap_size: int = Field(default=448)
    embed_dim: int = Field(default=384)
    depth: int = Field(default=1)
    checkpointing: bool = Field(default=True, description="Whether to use checkpointing for the cubemap decoder")


class KelvinModelConfig(BaseConfigSchema):
    """
    Configuration for the Kelvin model.
    """

    track_padding_m: List[float] = Field(
        default_factory=lambda: [1.0, 1.0, 1.0],
        description=(
            "Padding in meters for cuboid track bounding boxes when warping world points for motion supervision "
            "(x, y, z)."
        ),
        min_length=3,
        max_length=3,
    )

    scene_rescale: float = Field(default=0.15, description="Rescale scenes for model input and output")
    sky: KelvinSkyCubemapDecoderConfig = Field(default_factory=KelvinSkyCubemapDecoderConfig)

    patch_shape: Tuple[int, int] = Field(default=(14, 14))

    encoder: KelvinDAv3EncoderConfig = Field(default_factory=KelvinDAv3EncoderConfig)
    decoder: KelvinDPTDecoderConfig = Field(default_factory=KelvinDPTDecoderConfig)
    activations: GaussiansActivationConfig = Field(
        default_factory=GaussiansActivationConfig, description="Activation functions configuration."
    )

    export_preprocess: PrimitiveExportPreprocessConfig = Field(
        default_factory=PrimitiveExportPreprocessConfig,
        description="Per-chunk preprocess options for predict/export (filtering before merge or export).",
    )
