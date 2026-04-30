# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

import pytest

from pydantic import ValidationError

from nre.config.base_schema import config_to_primitive
from nre.config.model import (
    AntialiasingConfig,
    BaseRendererConfig,
    CameraLidarTilingConfig,
    CameraTilingConfig,
    CullingConfig,
    GSplatRendererConfig,
    LidarTilingConfig,
    NRendPipelineConfig,
    NRendPrimitivesConfig,
    NRendRendererConfig,
    OutputsConfig,
    ProfilingRendererConfig,
    ProjectionConfig,
    RenderModeConfig,
    SceneOutputConfig,
    SensorOutputConfig,
    TilingConfig,
)


def test_projection_config_defaults() -> None:
    config = ProjectionConfig()
    assert config.n_rolling_shutter_iterations == 5
    assert config.ut_dim == 3
    assert config.ut_alpha == 1.0
    assert config.ut_beta == 2.0
    assert config.ut_kappa == 0.0
    assert config.ut_require_all_sigma_points is False
    assert config.image_margin_factor == 0.1
    assert config.min_projected_ray_radius == pytest.approx(0.5477225575051661)


def test_projection_config_custom() -> None:
    config = ProjectionConfig.model_validate({"ut_dim": 5, "ut_alpha": 0.5, "n_rolling_shutter_iterations": 10})
    assert config.ut_dim == 5
    assert config.ut_alpha == 0.5
    assert config.n_rolling_shutter_iterations == 10


def test_culling_config_defaults() -> None:
    config = CullingConfig()
    assert config.near_clip_distance == 0.01
    assert config.near_far_z_culling is False
    assert config.rect_bounding is False
    assert config.enable_ray_based_culling is False


def test_culling_config_custom() -> None:
    config = CullingConfig.model_validate(
        {"near_clip_distance": 0.5, "rect_bounding": True, "enable_ray_based_culling": True}
    )
    assert config.near_clip_distance == 0.5
    assert config.rect_bounding is True
    assert config.enable_ray_based_culling is True


def test_culling_config_legacy_far_clip_distance() -> None:
    """Legacy ``far_clip_distance`` should be promoted to both sensor-specific fields."""
    config = CullingConfig.model_validate({"far_clip_distance": 500.0})
    assert config.far_clip_distance_camera == 500.0
    assert config.far_clip_distance_lidar == 500.0


def test_culling_config_legacy_far_clip_distance_no_override() -> None:
    """Sensor-specific values take precedence over legacy ``far_clip_distance``."""
    config = CullingConfig.model_validate({"far_clip_distance": 500.0, "far_clip_distance_camera": 200.0})
    assert config.far_clip_distance_camera == 200.0
    assert config.far_clip_distance_lidar == 500.0


def test_tiling_config_lidar_defaults() -> None:
    """Base TilingConfig has lidar with defaults."""
    config = TilingConfig()
    assert config.lidar.tile_size_elevation == 16
    assert config.lidar.tile_size_azimuth == 16


def test_tiling_config_lidar_custom() -> None:
    config = TilingConfig.model_validate({"lidar": {"tile_size_elevation": 8, "tile_size_azimuth": 8}})
    assert config.lidar.tile_size_azimuth == 8


def test_camera_lidar_tiling_config() -> None:
    """CameraLidarTilingConfig has both camera and lidar."""
    config = CameraLidarTilingConfig.model_validate(
        {
            "camera": {"tile_width": 32, "tile_height": 32},
            "lidar": {"tile_size_elevation": 8, "tile_size_azimuth": 8},
        }
    )
    assert config.camera.tile_width == 32
    assert config.lidar.tile_size_azimuth == 8


def test_camera_lidar_tiling_config_defaults() -> None:
    """CameraLidarTilingConfig has sensible defaults for both sensors."""
    config = CameraLidarTilingConfig()
    assert config.camera.tile_width == 16
    assert config.lidar.tile_size_elevation == 16


def test_render_mode_config() -> None:
    config = RenderModeConfig.model_validate({"mode": "kbuffer", "k_buffer_size": 16})
    assert config.mode == "kbuffer"
    assert config.k_buffer_size == 16
    assert config.enable_warp_atomic_optim is True


def test_sensor_output_config_defaults() -> None:
    config = SensorOutputConfig()
    assert config.enable_features is True
    assert config.enable_normals is True
    assert config.enable_extended_features is True
    assert config.enable_sensor_features is True
    assert config.enable_ray_gradients is False


def test_outputs_config_defaults() -> None:
    config = OutputsConfig()
    assert config.camera.enable_features is True
    assert config.lidar.enable_normals is True
    assert config.scene.enable_cumulated_weights is False


def test_profiling_config_defaults() -> None:
    config = ProfilingRendererConfig()
    assert config.frequency == 0.0


def test_antialiasing_config_defaults() -> None:
    config = AntialiasingConfig()
    assert config.lidar_divergence == 0.002


def test_nrend_pipeline_config() -> None:
    config = NRendPipelineConfig.model_validate({"type": "reference", "k_buffer_size": 32})
    assert config.type == "reference"
    assert config.k_buffer_size == 32
    assert config.fwd_type is None


def test_nrend_primitives_config() -> None:
    config = NRendPrimitivesConfig.model_validate({"type": "transformed_aabb", "density_scale": False})
    assert config.type == "transformed_aabb"
    assert config.density_scale is False


def test_gsplat_renderer_config_minimal() -> None:
    config = GSplatRendererConfig.model_validate(
        {
            "name": "3dgut-gsplat",
            "tiling": {"camera": {"tile_width": 16, "tile_height": 16}},
            "background": {"color": 0.0},
        }
    )
    assert config.name == "3dgut-gsplat"
    assert config.profiling.frequency == 0.0
    assert config.antialiasing.lidar_divergence == 0.002
    assert config.tiling is not None
    assert config.tiling.camera is not None
    assert config.tiling.camera.tile_width == 16


def test_gsplat_renderer_config_with_sub_configs() -> None:
    config = GSplatRendererConfig.model_validate(
        {
            "name": "3dgut-gsplat",
            "projection": {"ut_dim": 5, "ut_alpha": 0.5},
            "culling": {"near_clip_distance": 0.3, "enable_ray_based_culling": True},
            "tiling": {
                "camera": {"tile_width": 16, "tile_height": 16},
                "lidar": {"tile_size_elevation": 32, "tile_size_azimuth": 32},
            },
            "render": {"mode": "kbuffer", "k_buffer_size": 0},
            "profiling": {"frequency": 100.0},
            "antialiasing": {"lidar_divergence": 0.005},
            "background": {"color": 0.0},
        }
    )
    assert config.projection is not None
    assert config.projection.ut_dim == 5
    assert config.culling is not None
    assert config.culling.enable_ray_based_culling is True
    assert config.render is not None
    assert config.render.mode == "kbuffer"
    assert config.profiling.frequency == 100.0
    assert config.antialiasing.lidar_divergence == 0.005


def test_nrend_renderer_config() -> None:
    config = NRendRendererConfig.model_validate(
        {
            "name": "3dgut-nrend",
            "pipeline": {"type": "reference", "k_buffer_size": 16},
            "primitives": {"type": "transformed_aabb"},
            "tiling": {"camera": {"tile_width": 16, "tile_height": 16}},
            "background": {"color": 0.0},
        }
    )
    assert config.name == "3dgut-nrend"
    assert config.pipeline is not None
    assert config.pipeline.type == "reference"
    assert config.primitives is not None
    assert config.primitives.type == "transformed_aabb"


def test_profiling_defaults_on_renderer() -> None:
    """Profiling and antialiasing should have sensible defaults without explicit config."""
    config = GSplatRendererConfig.model_validate(
        {
            "name": "3dgs-gsplat",
            "background": {"color": 0.0},
        }
    )
    assert config.profiling.frequency == 0.0
    assert config.antialiasing.lidar_divergence == 0.002


def test_gsplat_renderer_name_literal() -> None:
    """GSplatRendererConfig.name must be one of the supported names."""
    with pytest.raises(ValidationError):
        GSplatRendererConfig.model_validate({"name": "unsupported-renderer"})


def test_nrend_renderer_name_literal() -> None:
    """NRendRendererConfig.name must be one of the supported names."""
    with pytest.raises(ValidationError):
        NRendRendererConfig.model_validate({"name": "unsupported-renderer"})


# --- config_to_primitive() round-trip tests ---


def test_nrend_config_to_primitive_excludes_none_optix_fields() -> None:
    """pipeline/primitives/backward must be absent from the dict when None.

    Sending defaults to non-OptiX renderers would inject unexpected settings
    into the C++ renderer config dict.
    """
    config = NRendRendererConfig.model_validate({"name": "3dgut-nrend", "background": {"color": 0.0}})
    d = config_to_primitive(config)
    assert "pipeline" not in d
    assert "primitives" not in d
    assert "backward" not in d


def test_nrend_config_to_primitive_includes_optix_fields_when_set() -> None:
    """pipeline/primitives should appear when explicitly configured."""
    config = NRendRendererConfig.model_validate(
        {
            "name": "3dgrt-optix-nrend",
            "pipeline": {"type": "reference", "k_buffer_size": 32},
            "primitives": {"type": "transformed_aabb"},
            "background": {"color": 0.0},
        }
    )
    d = config_to_primitive(config)
    assert "pipeline" in d
    assert d["pipeline"]["type"] == "reference"
    assert d["pipeline"]["k_buffer_size"] == 32
    assert "primitives" in d
    assert d["primitives"]["type"] == "transformed_aabb"


def test_tiling_config_to_primitive_lidar_only() -> None:
    """Base TilingConfig serializes with lidar defaults only (no camera key)."""
    config = TilingConfig()
    d = config_to_primitive(config)
    assert "lidar" in d
    assert d["lidar"]["tile_size_elevation"] == 16
    assert "camera" not in d


def test_camera_lidar_tiling_config_to_primitive() -> None:
    """CameraLidarTilingConfig serializes with both camera and lidar."""
    config = CameraLidarTilingConfig()
    d = config_to_primitive(config)
    assert "camera" in d
    assert "lidar" in d
    assert d["camera"]["tile_width"] == 16
    assert d["lidar"]["tile_size_elevation"] == 16


def test_culling_config_to_primitive_far_clip_migration() -> None:
    """Legacy far_clip_distance produces sensor-specific keys after serialization."""
    config = CullingConfig.model_validate({"far_clip_distance": 500.0})
    d = config_to_primitive(config)
    assert d["far_clip_distance_camera"] == 500.0
    assert d["far_clip_distance_lidar"] == 500.0


def test_outputs_config_to_primitive_nested_shape() -> None:
    """Outputs sub-structure must have the nested shape expected by the renderer."""
    config = OutputsConfig()
    d = config_to_primitive(config)
    assert "camera" in d
    assert "lidar" in d
    assert "scene" in d
    assert d["camera"]["enable_features"] is True
    assert d["camera"]["enable_normals"] is True
    assert d["lidar"]["enable_sensor_features"] is True
    assert d["scene"]["enable_cumulated_weights"] is False
