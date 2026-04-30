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
from nre.config.difix import DifixModelConfig
from nre.config.model import BaseRendererConfig, RendererConfigType


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


class KelvinExportPreprocessConfig(PrimitiveExportPreprocessConfig):
    """Kelvin export preprocess; only uses base density_prune_threshold."""


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

    # Sky mask activation parameters
    sky_mask_clamp_min: float = Field(default=-10.0, description="Minimum clamp value for sky mask")
    sky_mask_clamp_max: float = Field(default=10.0, description="Maximum clamp value for sky mask")

    # Forward flow activation parameters
    forward_flow_scale: float = Field(default=30.0, description="Scale factor for forward flow")

    # Falloff sigma activation parameters
    falloff_sigma_min: float = Field(
        default=0.25, description="Minimum falloff sigma value. If set to -1, it will be inferred from frame gap."
    )
    falloff_sigma_max: float = Field(default=2.0, description="Maximum falloff sigma value")
    falloff_sigma_clamp_min: float = Field(default=-5.0, description="Minimum clamp value for falloff sigma")
    falloff_sigma_clamp_max: float = Field(default=5.0, description="Maximum clamp value for falloff sigma")



class KelvinTokenGSEncoderConfig(BaseConfigSchema):
    name: Literal["token-gs-encoder"]
    depth: int
    n_heads: int
    embed_dim: int
    use_qk_norm: bool = Field(default=True)
    layer_scale_init_values: Optional[float] = Field(
        default=1e-4, description="The initial values for the layer scale. If None, no layer scale is used."
    )


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

    depth_offset: bool = Field(default=False, description="Whether to predict a depth offset (in world space)")
    uv_offset: bool = Field(default=False, description="Whether to predict a UV offset (in pixel space)")

    checkpointing: bool = Field(default=False, description="Whether to use checkpointing for the DPT decoder")
    dpt_chunk_size: int = Field(
        default=-1, description="Chunk size for the DPT decoder. Used for saving memory. -1 to disable."
    )

    fusion_for_gs_motion: bool = Field(
        default=False, description="Whether to use fusion for the GS head and the motion head"
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
    fusion_dim: int | None = Field(default=None, description="Dimension of the fusion layer")
    checkpointing: bool = Field(default=False, description="Whether to use checkpointing for the cubemap decoder")


class KelvinSkySolidColorConfig(BaseConfigSchema):
    name: Literal["solid-color"]
    color: tuple[float, float, float]
    cubemap_size: int


class KelvinPostProcessingConfig(BaseConfigSchema):
    enabled: bool = Field(
        default=True,
        description="If False, skip the per-camera affine RGB module and use an identity transform at render time.",
    )
    optimization_start_global_step: int = Field(
        default=0, description="Start optimizing the post processing module from this global step."
    )


class KelvinModelConfig(BaseModelConfig):
    """
    Configuration for the Kelvin model.
    """

    name: Literal["kelvin"]
    renderer: RendererConfigType

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

    sky: KelvinSkyCubemapDecoderConfig | KelvinSkySolidColorConfig = Field(discriminator="name")

    patch_shape: Tuple[int, int] = Field(default=(8, 8))

    encoder: KelvinTokenGSEncoderConfig | KelvinDAv3EncoderConfig = Field(discriminator="name")
    decoder: KelvinDPTDecoderConfig = Field(discriminator="name")
    post_processing: KelvinPostProcessingConfig = Field(default_factory=KelvinPostProcessingConfig)
    init_weights_paths: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Paths to initial weights by component or full model. "
            'Single entry with key "tokengs" or "full" loads a full-model checkpoint. '
            "Multiple entries are passed to encoder/decoder/sky initialize_weights by name."
        ),
    )

    activations: GaussiansActivationConfig = Field(
        default_factory=GaussiansActivationConfig, description="Activation functions configuration."
    )

    # Augmentations
    voxel_size: float | None = Field(default=None, description="Voxel size for the voxelization")

    export_preprocess: KelvinExportPreprocessConfig = Field(
        default_factory=KelvinExportPreprocessConfig,
        description="Per-chunk preprocess options for predict/export (filtering before merge or export).",
    )
