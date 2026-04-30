# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Any, Literal, Optional, Tuple

from nre.config.base_schema import BaseConfigSchema, Field
from nre.config.trainer import TrainerConfig


class PrecomputeExtraSignalsConfig(BaseConfigSchema):
    enabled: bool = False
    sample_every_n_frames: int = 0
    n_rays_per_frame: int = 0


class NrendTestConfig(BaseConfigSchema):
    enabled: bool = False
    rendered_model_json_path: Optional[str] = None
    renderer_hint: int = Field(
        description="Hint to select renderer implementation : [0=DEFAULT,1=FASTEST,2=FAST,3=QUALITY,4=HIGHEST_QUALITY]",
        default=0,
    )
    renderer_settings_json_path: Optional[str] = None
    log_level: int = Field(
        description="The logging level of nrend : [0=FATAL,1=ERROR,2=WARNING,3=INFO,4=DEBUG]",
        default=1,
    )
    profiling_frequency: float = 0.0
    lidar_only: bool = False
    create_test_case: bool = False
    create_test_case_update: bool = False
    overlay_render_time: bool = False


class SystemTestConfig(BaseConfigSchema):
    save_results: bool = False
    save_inputs: bool = False
    save_videos: bool = False
    save_stats: bool = False
    save_extra_signals: bool = False
    separate_val_dir_per_step: bool = Field(
        default=False,
        description="When true, validation outputs are saved to a separate subdirectory per global step "
        "(e.g. val/000200/), preventing overwrites across epochs during trainval.",
    )
    use_camera_name_dirs: bool = False
    frame_naming: Literal["batch-index", "frame-end-timestamp"] = "batch-index"
    precompute_extra_signals: PrecomputeExtraSignalsConfig = Field(default_factory=PrecomputeExtraSignalsConfig)
    video_fps: int
    nrend: NrendTestConfig = Field(default_factory=NrendTestConfig)
    track_ply: Any
    track_orbit: Any
    track_rotation: Any
    background_removal: Any
    val_render_selected_nodes: Any
    metrics: Any


class SaveRendersConfig(BaseConfigSchema):
    enabled: bool = False
    root: Tuple[float, float, float] = Field(default=(0, 0, 0))
    look_dir: Tuple[float, float, float] = Field(default=(1, 0, 0))
    up_dir: Tuple[float, float, float] = Field(default=(0, 0, 1))
    radius: float = Field(default=0.04)


class SaveFilteredPCConfig(BaseConfigSchema):
    enabled: bool = False
    filter_threshold: float = Field(default=0.02)


class LidarEvaluatorTestConfig(BaseConfigSchema):
    raydrop_threshold: float
    ROI: Any
    save_renders: SaveRendersConfig = Field(default_factory=SaveRendersConfig)
    save_filtered_pc: SaveFilteredPCConfig = Field(default_factory=SaveFilteredPCConfig)


class LidarSystemTestConfig(SystemTestConfig):
    lidar: LidarEvaluatorTestConfig


class RecordTimingsConfig(BaseConfigSchema):
    train_interval: int
    val_interval: int


class RecordQualityMetricsConfig(BaseConfigSchema):
    psnr_interval: int


class BaseSystemConfig(BaseConfigSchema):
    """
    Collect config options common to all systems.

    NOTE: this may interfere with how docs for subclasses are rendered (we'd like them to list
    all properties each time) - in that case we will want to get rid of this class.
    """

    max_split_size_mb: Optional[int] = Field(
        default=None,
        description="Read docs before using: https://pytorch.org/docs/stable/notes/cuda.html#environment-variables",
    )

    save_logger: bool

    warmup_steps: int

    test: SystemTestConfig

    optimizer: Optional[Any] = None
    optimizers: Optional[list[Any]] = None


class BaseGaussiansSystemConfig(BaseSystemConfig):
    scheduler: Optional[Any] = Field(
        default=None,
        description="Examples: SequentialLR, ChainedScheduler",
    )
    record_timings: RecordTimingsConfig = Field(description="Controls the frequency of log timing.")
    record_quality_metrics: RecordQualityMetricsConfig = Field(
        description="Controls the frequency of quality metrics recording."
    )

    collect_garbage_mem_usage: Optional[float] = Field(
        default=None,
        description="Collect garbage and empty cache when GPU memory usage exceeds this threshold (0.0-1.0, e.g., 0.8 for 80%)",
    )
    collect_garbage_check_interval: int = Field(
        default=10000,
        description="Check GPU memory usage every N global steps (to reduce overhead)",
    )


class GaussiansSystemConfig(BaseGaussiansSystemConfig):
    name: Literal["gaussians-system"]
    test: LidarSystemTestConfig


class NRendTestGaussiansSystemConfig(BaseGaussiansSystemConfig):
    name: Literal["nrend-test-gaussians-system"]
    test: LidarSystemTestConfig
