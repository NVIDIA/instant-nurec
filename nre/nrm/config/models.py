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

from typing import Any, List, Literal, Optional, Tuple

from nre.config.base_schema import BaseConfigSchema, Field


class BaseModelConfig(BaseConfigSchema):
    pass


class PrimitiveExportPreprocessConfig(BaseConfigSchema):
    """
    Config for per-chunk primitive preprocessing before export (and before merge).
    Used by preprocess_for_export(); not part of merge logic.
    """

    enabled: bool = Field(
        default=True,
        description="Whether to run per-chunk preprocess (density/sky/road filtering) before export or merge.",
    )
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
    scale_type: Literal["world", "pixel"] = Field(default="world", description="Type of scale activation")
    scale_shift_log_ratio: float = Field(default=-1.0, description="Shift parameter for scale activation")
    scale_max: float = Field(default=0.3, description="Maximum scale value")
    scale_min: float = Field(
        default=0.0,
        description="Minimum scale value (clamp applied after exp). Use 0.01 when using 3DGUT renderer to avoid NaN gradients.",
    )

    # Distance activation parameters
    distance_type: Literal["sigmoid", "none"] = Field(default="sigmoid", description="Type of distance activation")
    distance_min: float = Field(default=0.1, description="Minimum distance value")
    distance_max: float = Field(default=500.0, description="Maximum distance value")
    distance_shift: float = Field(default=-1.65, description="Shift parameter for distance sigmoid")

    # XYZ activation parameters
    xyz_type: Literal["exp", "none"] = Field(default="exp", description="Type of XYZ activation")
    z_offset: float = Field(default=1.0, description="Offset for XYZ activation")




class KelvinDAv3EncoderConfig(BaseConfigSchema):
    name: Literal["dav3-encoder"]
    depth: int
    n_heads: int
    embed_dim: int
    take_block_indices: List[int]
    aa_start_block_idx: int
    ffn_type: Literal["mlp", "swiglu"]
    checkpointing: Literal["all", "local", "none"] = Field(
        default="none", description="Whether to checkpoint the encoder"
    )


class KelvinDPTDecoderConfig(BaseConfigSchema):
    name: Literal["dpt-decoder"]
    dpt_dim: int
    dpt_reassemble_hidden_dims: List[int]

    checkpointing: bool = Field(default=False, description="Whether to use checkpointing for the DPT decoder")
    dpt_chunk_size: int = Field(
        default=-1, description="Chunk size for the DPT decoder. Used for saving memory. -1 to disable."
    )

    # Motion-related:
    time_encoding_dim: int = Field(default=256, description="Dimension of the time sinusoidal encoding")
    motion_depth: int = Field(default=4, description="Depth of the motion head (V-DPM setup is equivalent to 8)")

    def model_post_init(self, __context) -> None:
        assert self.dpt_dim > 0, "DPT dimension must be positive"


class KelvinSkyCubemapDecoderConfig(BaseConfigSchema):
    name: Literal["cubemap-decoder"]
    cubemap_size: int
    embed_dim: int
    depth: int
    checkpointing: bool = Field(default=False, description="Whether to use checkpointing for the cubemap decoder")


class KelvinPostProcessingConfig(BaseConfigSchema):
    enabled: bool = Field(
        default=True,
        description="If False, skip the per-camera affine RGB module and use an identity transform at render time.",
    )


class KelvinModelConfig(BaseModelConfig):
    """
    Configuration for the Kelvin model.
    """

    name: Literal["kelvin"]

    track_padding_m: List[float] = Field(
        default=[1.0, 1.0, 1.0],
        description=(
            "Padding in meters for cuboid track bounding boxes when warping world points for motion supervision "
            "(x, y, z)."
        ),
        min_length=3,
        max_length=3,
    )

    scene_rescale: float = Field(default=1.0, description="Rescale scenes for model input and output")
    prepare_normal_supervision: bool = Field(default=True, description="Whether to prepare normal supervision")
    use_2dgs: bool = Field(default=False, description="Whether to use 2DGS rendering")
    freeze_encoder: bool = Field(
        default=False,
        description="If True, freeze encoder parameters (requires_grad=False); train decoder/sky/post-processing only.",
    )

    sky: KelvinSkyCubemapDecoderConfig

    patch_shape: Tuple[int, int] = Field(default=(8, 8))

    encoder: KelvinDAv3EncoderConfig
    decoder: KelvinDPTDecoderConfig
    post_processing: KelvinPostProcessingConfig = Field(default_factory=KelvinPostProcessingConfig)
    activations: GaussiansActivationConfig = Field(
        default_factory=GaussiansActivationConfig, description="Activation functions configuration."
    )

    export_preprocess: PrimitiveExportPreprocessConfig = Field(
        default_factory=PrimitiveExportPreprocessConfig,
        description="Per-chunk preprocess options for predict/export (filtering before merge or export).",
    )
