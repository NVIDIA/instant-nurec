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


class CelsiusExportPreprocessConfig(PrimitiveExportPreprocessConfig):
    """Celsius-specific export preprocess options (e.g. sky handling)."""

    keep_sky_gaussians: bool = Field(
        default=False,
        description="Whether to keep sky Gaussians when filtering. If False, sky Gaussians are removed per chunk.",
    )
    project_to_z_offset: bool = Field(
        default=False,
        description="If True, project each Gaussian onto the road plane (z_offset in rig space) along the ray "
        "that spawned it (from context). Requires context_rig to be passed to preprocess_for_export.",
    )
    z_offset: float = Field(
        default=0.0,
        description="Z height of the road plane in rig space; used when project_to_z_offset is True.",
    )
    projection: Literal["ray"] = Field(
        default="ray",
        description="Projection method: 'ray' projects along the spawning ray.",
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


class _CelsiusModelEncoderConfig(BaseConfigSchema):
    depth: int
    embed_dim: int
    n_heads: int
    mlp_ratio: float
    layer_scale_init_values: Optional[float] = Field(
        default=None, description="The initial values for the layer scale. If None, no layer scale is used."
    )
    qk_norm: bool = Field(
        default=False, description="Whether to use layer normalization after the query and key projections"
    )
    block_pattern: str = Field(
        default="T", description="Pattern of blocks in the encoder. 'T' for Transformer, 'M' for Mamba2Block."
    )

    def model_post_init(self, __context) -> None:
        # Expand block pattern to full length, for example if depth == 8, then
        # - block_pattern = "T" -> "TTTTTTTT"
        # - block_pattern = "TMMT" -> "TMMTTMMT"
        assert self.depth % len(self.block_pattern) == 0, "Block pattern must be divisible by depth"
        self.block_pattern = self.block_pattern * (self.depth // len(self.block_pattern))


class _CelsiusModelSkyModuleConfig(BaseConfigSchema):
    enabled: bool


class _CelsiusModelMotionModuleConfig(BaseConfigSchema):
    enabled: bool
    n_motion_tokens: int = Field(
        description="Set to 0 to disable motion basis (velocity will just be predicted from upscaled backbone feature)"
    )
    unpatch_dim: int = Field(default=128, description="Dimension of the progressive unpatching (very memory intensive)")
    motion_qkv_dim: int
    falloff: bool

    bidirectional_flow: bool = Field(
        default=False, description="Whether to use bidirectional flow (i.e. velocity will be 6-D)"
    )

    # [-1, 0] means disabled, [100000, 200000] means always enabled
    context_replace_start_global_step: int = Field(
        default=-1, description="Before this global step, the context will be replaced with the predicted velocity."
    )
    context_replace_end_global_step: int = Field(
        default=0, description="After this global step, the context will not be replaced with the predicted velocity."
    )

    def model_post_init(self, __context) -> None:
        assert self.context_replace_start_global_step < self.context_replace_end_global_step


class _CelsiusModelAffineModuleConfig(BaseConfigSchema):
    enabled: bool
    n_affine_tokens: int = Field(
        default=-1, description="Number of affine tokens. Set to -1 to allow dynamic number of tokens."
    )
    cross_attend: bool = Field(default=True, description="How the affine token interact with the other tokens")
    optimization_start_global_step: int = Field(
        default=0, description="Start optimizing the affine module from this global step."
    )

    def model_post_init(self, __context) -> None:
        if self.n_affine_tokens == -1:
            assert self.cross_attend, "Dynamic number of affine tokens requires cross_attend=True."


class _CelsiusModelVelocityFromLidarConfig(BaseConfigSchema):
    enabled: bool = Field(default=False, description="Whether to use velocity from lidar if available")
    gap_from_image_us: int = Field(default=0, description="Max gap from image in microseconds")
    near_mask_threshold_m: float = Field(default=3.0, description="Near mask threshold in meters")
    box_filter_voxel_size_m: float = Field(default=0.1, description="Box filter voxel size in meters")
    box_filter_max_count: int = Field(default=3, description="Box filter max count")
    gaussians_scale: float = Field(default=0.01, description="Scale for the gaussians")


class CelsiusModelConfig(BaseModelConfig):
    """
    Configuration for the Celsius model.
    """

    name: Literal["celsius"]

    renderer: RendererConfigType

    track_padding_m: List[float] = Field(
        default=[0.1, 0.1, 0.1],
        description=(
            "Padding in meters for the track bounding box in the x, y, z dimensions (left/right, front/back, up/down). "
            "This is used when masking dynamic objects is needed in static models."
        ),
        min_length=3,
        max_length=3,
    )

    init_weights_path: Optional[str] = Field(
        default=None,
        description="Path to the initial weights for the Celsius model. If not provided, the model will be initialized randomly.",
    )

    init_token_scale: float = Field(default=0.02, description="Scale for the sky/motion tokens for initialization")

    patch_shape: Tuple[int, int] = Field(default=(8, 8))

    encoder: _CelsiusModelEncoderConfig
    sky_module: _CelsiusModelSkyModuleConfig
    motion_module: _CelsiusModelMotionModuleConfig
    affine_module: _CelsiusModelAffineModuleConfig
    activations: GaussiansActivationConfig = Field(
        default_factory=GaussiansActivationConfig, description="Activation functions configuration."
    )
    velocity_from_lidar: _CelsiusModelVelocityFromLidarConfig = Field(
        default_factory=_CelsiusModelVelocityFromLidarConfig, description="Velocity from lidar configuration."
    )

    use_deferred_bp: bool = Field(default=True)
    activation_checkpointing: bool = Field(default=False)

    use_patch_embed_norm: bool = Field(
        default=True, description="Whether to use layer normalization after the patch embedding"
    )
    use_encoder_norm: bool = Field(default=True, description="Whether to use layer normalization after the encoder")
    centroid_prediction: Literal["distance", "xyz"] = Field(
        default="distance", description="The type of centroid prediction"
    )

    difix: DifixModelConfig | None = Field(
        default=None,
        description="Difix model to be applied to each rendered frame as part of the model. None to disable.",
    )

    scene_rescale: float = Field(default=1.0, description="Rescale scenes for model input and output")

    legacy_mask_input: bool = Field(default=False, description="Whether to use the legacy mask input")

    export_preprocess: CelsiusExportPreprocessConfig = Field(
        default_factory=CelsiusExportPreprocessConfig,
        description="Per-chunk preprocess options for predict/export (filtering before merge or export).",
    )


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


class KelvinTokenGSDecoderConfig(BaseConfigSchema):
    name: Literal["token-gs-decoder"]
    depth: int
    use_qk_norm: bool = Field(default=True)
    layer_scale_init_values: Optional[float] = Field(
        default=1e-4, description="The initial values for the layer scale. If None, no layer scale is used."
    )
    num_gaussian_tokens: int = Field(default=1024, description="Number of Gaussian tokens")
    gaussian_token_init_std: float = Field(
        default=0.02, description="Standard deviation for the Gaussian token initialization"
    )
    use_decoder_norm: bool = Field(default=False, description="Whether to use layer normalization after the decoder")


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


class KelvinPointQueryCADecoderConfig(BaseConfigSchema):
    """Decoder that keeps DPT depth/context/motion heads but replaces the DPT gaussian head
    with a cross-attention head using depth-derived xyz positions (2D grid tokenization) as queries."""

    name: Literal["point-query-ca-decoder"]

    # Shared DPT fields (for depth, context, motion heads)
    dpt_dim: int
    dpt_reassemble_hidden_dims: List[int]
    checkpointing: bool = Field(default=False, description="Whether to use checkpointing for the DPT heads")
    dpt_chunk_size: int = Field(default=-1, description="Chunk size for the DPT heads. -1 to disable.")
    time_encoding_dim: int = Field(default=256, description="Dimension of the time sinusoidal encoding")
    motion_depth: int = Field(default=4, description="Depth of the motion head")

    # Cross-attention gaussian head
    ca_depth: int = Field(default=1, description="Number of cross-attention decoder blocks")
    use_qk_norm: bool = Field(default=True, description="Whether to use QK normalization in cross-attention")
    layer_scale_init_values: Optional[float] = Field(default=1e-4, description="Layer scale init values for CA blocks")
    xyz_downsample_stride: int = Field(default=4, description="Stride for downsampling depth map")
    grid_center_stride: int = Field(
        default=4,
        description="Grouping stride relative to the candidate grid. "
        "Each token covers a grid_center_stride x grid_center_stride block, so gs_per_token = grid_center_stride^2.",
    )
    use_gt_semantic_mask: bool = Field(
        default=False,
        description="If True, use ground-truth semantic labels instead of predicted semantics for the sky/ego opacity mask.",
    )

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
    decoder: KelvinTokenGSDecoderConfig | KelvinDPTDecoderConfig | KelvinPointQueryCADecoderConfig = Field(
        discriminator="name"
    )
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
