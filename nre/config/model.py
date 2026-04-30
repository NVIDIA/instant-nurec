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


# Discriminated background subclasses (sky env map / sky MLP / solid color /
# skip) were training-only — predict's renderer config has background: null.
# Keeping the BaseBackgroundConfig stub above so the optional field still
# accepts None.
BackgroundConfigType = BaseBackgroundConfig


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


RendererConfigType = NRendRendererConfig


