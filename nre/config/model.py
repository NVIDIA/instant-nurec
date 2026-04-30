# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

from contextlib import nullcontext
from enum import Enum
from typing import Any, Literal, Optional

from omegaconf import DictConfig, open_dict
from pydantic import model_validator

from nre.config.base_schema import BaseConfigSchema, Field
from nre.config.optim import OptimizerConfig, SchedulerConfig


class RendererBackend(Enum):
    """Renderer backend selection for serving and rendering.

    DEFAULT: Use the artifact's trained renderer as-is (PyTorch forward pass).
    GSPLAT:  Force GSplatRenderer override regardless of artifact configuration.
    NREND:   Use the fast NRendWrapper (direct C++/CUDA JIT, bypasses PyTorch forward).
    """

    DEFAULT = "default"
    GSPLAT = "gsplat"
    NREND = "nrend"


class SensorCalibConfig(BaseConfigSchema):
    """Sensor-specific calibration configuration."""

    enabled: bool


class CalibConfig(BaseConfigSchema):
    """Calibration component configuration."""

    name: str

    # Common fields
    enabled: bool = False
    start_global_step: int = 250
    skip_first_pose_delta: bool = True
    enable_torch_compile: bool = True

    # Sensor-specific configs
    lidar: SensorCalibConfig
    camera: SensorCalibConfig

    # Optimizer and scheduler configs
    optimizer: OptimizerConfig | None = None
    scheduler: SchedulerConfig | None = None


class BaseBackgroundConfig(BaseConfigSchema):
    """Base background configuration with common fields."""

    name: str

    # Common settings across all backgrounds
    composite_in_linear_space: bool = False


class SkyEnvMapBackgroundConfig(BaseBackgroundConfig):
    """Sky environment map background configuration."""

    name: Literal["sky-env-map"]

    # Required for env map
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    envmap_type: Literal["cubemap", "equirectangular"]
    should_inpaint: bool
    inpaint_threshold: float
    inpaint_kernel_size: int = Field(ge=1)
    min_grad_updates: int = Field(ge=0)
    saturate_radiance: bool = True


class SkyMlpBackgroundConfig(BaseBackgroundConfig):
    """Sky MLP background configuration."""

    name: Literal["sky-mlp"]

    # Required for MLP
    dir_encoding_config: Any
    mlp_network_config: Any
    enable_jit_if_supported: bool
    seed: int
    saturate_radiance: bool = True


class ExtraSignalSkyMlpBackgroundConfig(SkyMlpBackgroundConfig):
    """Sky MLP with extra signal support."""

    name: Literal["extra-signal-sky-mlp"]  # type: ignore[assignment]

    extra_signal: Any


class BackgroundColorConfig(BaseBackgroundConfig):
    """Solid color background configuration."""

    name: Literal["background-color"]

    color: Literal["white", "black", "random"]
    random_eval_color: list[float] | None = None


class SkipBackgroundConfig(BaseBackgroundConfig):
    """Skip/no-op background configuration."""

    name: Literal["skip-background"]


# Discriminated union of all background types
BackgroundConfigType = (
    SkyEnvMapBackgroundConfig
    | SkyMlpBackgroundConfig
    | ExtraSignalSkyMlpBackgroundConfig
    | BackgroundColorConfig
    | SkipBackgroundConfig
)


class ProjectionConfig(BaseConfigSchema):
    """Projection settings for NRend renderers."""

    n_rolling_shutter_iterations: int = 5
    ut_dim: int = 3
    ut_alpha: float = 1.0
    ut_beta: float = 2.0
    ut_kappa: float = 0.0
    ut_require_all_sigma_points: bool = False
    image_margin_factor: float = 0.1
    min_projected_ray_radius: float = 0.5477225575051661


class CullingConfig(BaseConfigSchema):
    """Culling settings for NRend renderers."""

    near_clip_distance: float = 0.01
    far_clip_distance: float | None = Field(
        default=None,
        deprecated="Use `far_clip_distance_camera` and `far_clip_distance_lidar` instead. "
        "If set, overwrites both sensor-specific fields.",
    )
    far_clip_distance_camera: float = 1e10
    far_clip_distance_lidar: float = 1e10
    near_far_z_culling: bool = False
    rect_bounding: bool = False
    tight_opacity_bounding: bool = False
    tile_based: bool = False
    enable_ray_based_culling: bool = False

    @model_validator(mode="before")
    @classmethod
    def _migrate_far_clip_distance(cls, data: dict[str, Any] | DictConfig) -> dict[str, Any] | DictConfig:
        """Backward compat: promote legacy ``far_clip_distance`` to the sensor-specific fields."""
        legacy = data.get("far_clip_distance")
        if legacy is not None:
            ctx = open_dict(data) if isinstance(data, DictConfig) else nullcontext()
            with ctx:
                if "far_clip_distance_camera" not in data:
                    data["far_clip_distance_camera"] = legacy
                if "far_clip_distance_lidar" not in data:
                    data["far_clip_distance_lidar"] = legacy
        return data


class CameraTilingConfig(BaseConfigSchema):
    """Camera tiling settings."""

    tile_width: int = 16
    tile_height: int = 16


class LidarTilingConfig(BaseConfigSchema):
    """LiDAR tiling settings."""

    tile_size_elevation: int = 16
    tile_size_azimuth: int = 16
    n_bins_elevation: int = 16
    resolution_elevation: int = 1600
    densification_factor_azimuth: int = 8


class TilingConfig(BaseConfigSchema):
    """Tiling settings with LiDAR support (base, used by NRend)."""

    lidar: LidarTilingConfig = Field(default_factory=LidarTilingConfig)


class CameraLidarTilingConfig(TilingConfig):
    """Tiling settings with both camera and LiDAR support (used by GSplat)."""

    camera: CameraTilingConfig = Field(default_factory=CameraTilingConfig)


class RenderModeConfig(BaseConfigSchema):
    """Render mode settings."""

    mode: str = ""
    k_buffer_size: int = 0
    enable_warp_atomic_optim: bool = True


class SensorOutputConfig(BaseConfigSchema):
    """Per-sensor rendering output channel configuration."""

    enable_features: bool = True
    enable_normals: bool = True
    enable_extended_features: bool = True
    enable_sensor_features: bool = True
    enable_ray_gradients: bool = False


class SceneOutputConfig(BaseConfigSchema):
    """Scene-level rendering output configuration."""

    enable_cumulated_weights: bool = False
    enable_visibility: bool = False


class OutputsConfig(BaseConfigSchema):
    """Rendering output channels configuration for all sensor types."""

    camera: SensorOutputConfig = Field(default_factory=SensorOutputConfig)
    lidar: SensorOutputConfig = Field(default_factory=SensorOutputConfig)
    scene: SceneOutputConfig = Field(default_factory=SceneOutputConfig)


class ProfilingRendererConfig(BaseConfigSchema):
    """Profiling configuration for renderers."""

    frequency: float = 0.0


class AntialiasingConfig(BaseConfigSchema):
    """Antialiasing settings for renderers."""

    lidar_divergence: float = 0.002


class NRendPipelineConfig(BaseConfigSchema):
    """Pipeline configuration for NRend renderers."""

    type: str = "reference"
    fwd_type: str | None = None
    bwd_type: str | None = None
    k_buffer_size: int = 16


class NRendPrimitivesConfig(BaseConfigSchema):
    """Primitives configuration for NRend renderers."""

    type: str = "transformed_aabb"
    density_scale: bool = True


class BaseRendererConfig(BaseConfigSchema):
    """Base renderer configuration with fields shared by GSplat and NRend."""

    name: str
    log_level: int = 3
    prepare_before_render: bool = False

    # Shared sub-configs
    background: Any = None
    projection: ProjectionConfig = Field(default_factory=ProjectionConfig)
    culling: CullingConfig = Field(default_factory=CullingConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)
    tiling: TilingConfig = Field(default_factory=TilingConfig)
    render: RenderModeConfig = Field(default_factory=RenderModeConfig)
    profiling: ProfilingRendererConfig = Field(default_factory=ProfilingRendererConfig)
    antialiasing: AntialiasingConfig = Field(default_factory=AntialiasingConfig)


class GSplatRendererConfig(BaseRendererConfig):
    """GSplat renderer configuration."""

    name: Literal["3dgs-gsplat", "3dgut-gsplat"]

    tiling: CameraLidarTilingConfig = Field(default_factory=CameraLidarTilingConfig)  # type: ignore[assignment]

    rasterize_mode: Literal["classic", "antialiased"] = "classic"

    log_level: int | str = 3  # type: ignore[assignment]
    sparse_grad: bool = False
    absgrad: bool = False
    packed: bool = False
    radius_clip: float = 0.0
    use_rays: bool = False

    @model_validator(mode="before")
    @classmethod
    def validate_use_rays(cls, config: dict[str, Any] | DictConfig):
        if config.get("name") == "3dgut-gsplat" and "use_rays" not in config:
            config["use_rays"] = True  # Should be True by default for 3dgut-gsplat
        return config

    @model_validator(mode="after")
    def validate_3dgut_rasterize_mode(self) -> "GSplatRendererConfig":
        if self.name == "3dgut-gsplat" and self.rasterize_mode == "antialiased":
            raise ValueError(
                "3dgut-gsplat does not support rasterize_mode='antialiased': "
                "3DGUT should use rasterize_mode='classic' to match nRend's opacity behavior "
                "when min_projected_ray_radius dilation is enabled."
            )
        return self


class NRendRendererConfig(BaseRendererConfig):
    """NRend renderer configuration (all NRend variants)."""

    name: Literal["3dgrt-optix-nrend", "3dgrt-rejection-optix-nrend", "3dgut-nrend", "3dgs-nrend", "3dgrut-nrend"]

    global_z_order: bool = False
    per_ray_features: bool = False
    use_gsplat_for_camera_rendering: bool = False
    checkpoint_friendly_backward: bool = Field(
        default=False,
        description="If true, use a non-reentrant checkpointing-friendly backward pass implementation (but slower).",
    )

    # NRend-specific nested configs
    # pipeline/primitives default to None because they are OptiX-specific — only 3dgrt renderers
    # use them. Sending default values to non-OptiX renderers (3dgut, 3dgs) would inject
    # unexpected pipeline/primitive settings into the C++ renderer config dict.
    pipeline: NRendPipelineConfig | None = None
    primitives: NRendPrimitivesConfig | None = None
    backward: dict[str, Any] | None = None


# Discriminated union of all renderer types
RendererConfigType = GSplatRendererConfig | NRendRendererConfig


class BaseStrategyConfig(BaseConfigSchema):
    """Base strategy configuration with common fields."""

    name: str

    # Common settings across all strategies
    print_stats: bool = False
    exclude_layer_ids: list[str] = Field(default_factory=list)


class DensifyConfig(BaseConfigSchema):
    """Gsplat densification sub-config."""

    frequency: int = Field(
        gt=0,
        description="Frequency of densification, in global iterations. `densification_interval` in 3DGS.",
    )
    start_iteration: int = Field(
        ge=0,
        description="Start densifying Gaussians from this global iteration. `densify_from_iter` in 3DGS.",
    )
    end_iteration: int = Field(
        ge=0,
        description="Stop densifying Gaussians at this global iteration. `densify_until_iter` in 3DGS.",
    )
    clone_grad_threshold: float = Field(
        ge=0.0,
        default=0.0002,
        description="Positional gradient threshold above which Gaussians are cloned. `densify_grad_threshold` in 3DGS.",
    )
    split_grad_threshold: float = Field(
        ge=0.0,
        default=0.0002,
        description="Positional gradient threshold above which Gaussians are split. `densify_grad_threshold` in 3DGS.",
    )
    relative_size_threshold: float = Field(
        ge=0.0,
        default=0.01,
        description=(
            "Gaussians larger than `relative_size_threshold * scene_extent` are split rather than cloned. "
            "`percent_dense` in 3DGS."
        ),
    )
    split_n_gaussians: int = Field(
        gt=0,
        default=2,
        description="Number of Gaussians produced per split. Hardcoded to 2 in 3DGS.",
    )


class PruneConfig(BaseConfigSchema):
    """Gsplat pruning sub-config."""

    frequency: int = Field(
        gt=0,
        description="Frequency of pruning, in global iterations.",
    )
    start_iteration: int = Field(
        ge=0,
        description="Start pruning Gaussians from this global iteration.",
    )
    end_iteration: int = Field(
        ge=0,
        description="Stop pruning Gaussians at this global iteration.",
    )
    density_threshold: float = Field(
        ge=0.0,
        le=1.0,
        default=0.01,
        description="Gaussians with density below this threshold are pruned.",
    )


class ResetDensityConfig(BaseConfigSchema):
    """Gsplat density reset sub-config."""

    frequency: int = Field(
        gt=0,
        description="Frequency of density reset, in global iterations. `opacity_reset_interval` in 3DGS.",
    )
    start_iteration: int = Field(
        ge=0,
        description="Start resetting densities from this global iteration.",
    )
    end_iteration: int = Field(
        ge=0,
        description="Stop resetting densities at this global iteration.",
    )
    new_max_density: float = Field(
        ge=0.0,
        le=1.0,
        default=0.01,
        description="Cap Gaussian densities at this value on reset. Hardcoded to 0.01 in 3DGS.",
    )


class GsplatStrategyConfig(BaseStrategyConfig):
    """Gsplat densification/pruning strategy configuration."""

    name: Literal["gsplat"]

    densify: DensifyConfig | None = None
    prune: PruneConfig | None = None
    reset_density: ResetDensityConfig | None = None


class MCMCStrategyConfig(BaseStrategyConfig):
    """MCMC sampling strategy configuration."""

    class MCMCPerturbationConfig(BaseConfigSchema):
        """Configuration settings of MCMC perturbations"""

        class MCMCNoiseLearningRateConfig(BaseConfigSchema):
            """Configuration settings of MCMC noise learning rate"""

            default: float = Field(
                default=5000.0,
                description="Default noise learning rate for MCMC strategy.",
            )
            layers: dict[str, float] = Field(
                default={
                    "dynamic_rigids": 5000.0,
                    "dynamic_deformables": 5000.0,
                    "road": 5000.0,
                },
                description="Noise learning rate overrides for specific layers.",
            )

        start_iteration: int = Field(
            ge=0,
            default=1,
            description="Start perturbing Gaussians from this global iteration.",
        )

        end_iteration: int = Field(
            ge=0,
            default=10_000_000,  # Default to a high number to make sure it runs until the end of training if not set
            description="Stop perturbing Gaussians from this global iteration.",
        )

        frequency: int = Field(
            gt=0,
            default=1,
            description="Frequency of applying Gaussian perturbation, in global iterations.",
        )

        noise_lr: MCMCNoiseLearningRateConfig = Field(
            default_factory=MCMCNoiseLearningRateConfig,
            description="Noise learning rate settings for MCMC strategy.",
        )

        move_outside_of_cuboid: bool = Field(
            default=False,
            description="Move the gaussians outside of the cuboid tracks if they are inside.",
        )

    class MCMCRelocationConfig(BaseConfigSchema):
        """Configuration settings of MCMC relocation"""

        start_iteration: int = Field(
            ge=0,
            default=500,
            description="Start relocating Gaussians from this global iteration.",
        )
        end_iteration: int = Field(
            ge=0,
            default=25_000,
            description="Stop relocating Gaussians from this global iteration.",
        )
        frequency: int = Field(
            gt=0,
            default=100,
            description="Frequency of Gaussians relocation, in global iterations.",
        )
        max_invisible_steps: Optional[int] = Field(
            default=None,
            description="Maximum number of steps a Gaussian can be invisible before being marked for relocation.",
        )

    class MCMCAdditionConfig(BaseConfigSchema):
        """Configuration settings of MCMC relocation"""

        start_iteration: int = Field(
            ge=0,
            default=500,
            description="Start adding new Gaussians from this global iteration.",
        )
        end_iteration: int = Field(
            ge=0,
            default=25_000,
            description="Stop adding new Gaussians from this global iteration.",
        )
        frequency: int = Field(
            gt=0,
            default=100,
            description="Frequency of adding new Gaussians, in global iterations.",
        )
        max_n_gaussians: int = Field(
            gt=0,
            default=2_000_000,
            description="Maximum number of Gaussians. New Gaussians will only be added until this number is reached.",
        )

    name: Literal["mcmc"]

    binom_n_max: int = Field(
        default=51,
        description=(
            "Maximum number of binomial coefficients to precompute for the MCMC strategy. "
            "Default value from MCMC paper (https://github.com/ubc-vision/3dgs-mcmc/issues/8)"
        ),
    )

    opacity_threshold: float = Field(
        default=0.005,
        description="Minimum opacity for a Gaussian to be marked alive",
    )

    relocate: MCMCRelocationConfig = Field(
        default_factory=MCMCRelocationConfig,
        description="Gaussian relocation settings for MCMC strategy.",
    )

    perturb: MCMCPerturbationConfig = Field(
        default_factory=MCMCPerturbationConfig,
        description="Gaussian perturbation settings for MCMC strategy.",
    )

    add: MCMCAdditionConfig = Field(
        default_factory=MCMCAdditionConfig,
        description="Gaussian addition settings for MCMC strategy.",
    )


# Discriminated union of all strategy types
StrategyConfigType = GsplatStrategyConfig | MCMCStrategyConfig


class BasePostProcessingConfig(BaseConfigSchema):
    """Base post-processing module configuration."""

    name: str

    # Common optional fields
    optimizer: OptimizerConfig | None = None
    scheduler: SchedulerConfig | None = None
    start_global_step: int | None = None


class BilateralGridPostProcessingConfig(BasePostProcessingConfig):
    """Bilateral grid post-processing configuration."""

    name: Literal["bilateral-grid-per-camera", "bilateral-grid-per-frame"]

    # Required grid dimensions
    width: int
    height: int
    depth: int


class PPISPPostProcessingConfig(BasePostProcessingConfig):
    """PPISP (Physics-based Image Signal Processing) configuration."""

    name: Literal["ppisp-post-processing"]

    use_slang: bool
    per_frame_ppisp_enabled: bool = True


class DINOv2PEPostProcessingConfig(BasePostProcessingConfig):
    """DINOv2 positional encoding post-processing configuration."""

    name: Literal["dinov2-pe-post-processing"]

    test_no_pe_map: bool
    pe_height: int
    pe_width: int
    n_dims: int


# Discriminated union of all post-processing types
PostProcessingConfigType = BilateralGridPostProcessingConfig | PPISPPostProcessingConfig | DINOv2PEPostProcessingConfig


class BaseLayerConfig(BaseConfigSchema):
    """Base layer configuration."""

    name: str

    nearest_neighbor_track_for_lidar: bool = False

    # May appear in any layer but not required by all
    initialization: Any = None
    class_labels: list[str] | None = None
    ignore_classes_from_layers: list[str] | None = None
    use_slang: bool = False


class SHGaussiansLayerConfig(BaseLayerConfig):
    """SH Gaussians layer configuration (static/background)."""

    name: Literal["sh-gaussians"]

    density_activation: str
    scale_activation: str
    rotation_activation: str
    progressive_training: Any
    particle: Any

    # May not be in legacy configs
    fourier_features_dim: int | None = None
    time_embed: Any = None

    # SH-specific with defaults
    precision: int | None = None
    transmittance_threshold: float | None = None
    scale_pos_lr_by_scene_extent: bool = False
    debug_viz: bool = False

    # Optional complex configs
    optimizer: Any = None
    optimizers: Any = None
    extra_signal: Any = None
    tracks: Any = None


class RigidGaussiansLayerConfig(SHGaussiansLayerConfig):
    """Rigid Gaussians layer configuration (dynamic with tracks)."""

    name: Literal["rigid-gaussians"]  # type: ignore[assignment]

    # Track-related
    tracks: Any
    optimize_track_albedo: bool
    optimize_track_scale: bool
    track_albedo: Any
    track_scale: Any
    tracks_calib: Any = None
    track_timestamp_est_precision: Literal["int32", "int64"] | None = None
    is_static: bool = False

    # May not be in legacy configs
    time_embed: Any = None


class DeformableGaussiansLayerConfig(RigidGaussiansLayerConfig):
    """Deformable Gaussians layer configuration."""

    name: Literal["deformable-gaussians"]  # type: ignore[assignment]

    # Required for deformable
    deform_network: Any
    deformnet_start_iteration: int
    cuboid_tracks_expand: float
    smoothness_frame_steps: int


class ElasticLoadedAHLayerConfig(RigidGaussiansLayerConfig):
    """Asset Harvester elastic loaded layer configuration."""

    name: Literal["elastic-loaded-ah"]  # type: ignore[assignment]

    # Required for asset harvester
    fourier_features_dim: int
    time_embed: Any
    particle: Any
    optimizers: Any
    deformation_network: Any
    restriction_policy: str

    # Optional with default fallbacks in code
    deform_matrix_size: int | None = None
    log_every_n_steps: int | None = None
    masking_end_step: int | None = None


# Discriminated union of all layer types
LayerConfigType = (
    SHGaussiansLayerConfig | RigidGaussiansLayerConfig | DeformableGaussiansLayerConfig | ElasticLoadedAHLayerConfig
)


class ModelConfig(BaseConfigSchema):
    """
    Model configuration for Gaussians-based systems.
    """

    name: str
    debug_viz: bool = False
    log_to_prog_bar: bool = True
    saturate_radiance: bool = True
    train_num_rays: int

    layers: dict[str, LayerConfigType] = Field(description="Layer-specific configurations")

    calib: CalibConfig | None = Field(default=None, description="Calibration component configuration")

    background: BackgroundConfigType | None = Field(
        default=None, discriminator="name", description="Background component configuration"
    )

    renderer: RendererConfigType = Field(discriminator="name", description="Renderer configuration")

    strategy: StrategyConfigType = Field(
        discriminator="name", description="Gaussian densification/pruning strategy configuration"
    )

    post_processing: dict[str, PostProcessingConfigType] | None = Field(
        default=None, description="Post-processing modules (bilateral grids, PPISP, DINOv2 PE, etc.)"
    )
