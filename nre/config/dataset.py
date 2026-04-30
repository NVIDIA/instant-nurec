# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from ncore.data import LabelSource as NCoreLabelSource
from nre.config.base_schema import BaseConfigSchema, Field


class _BaseDatasetSOConfig(BaseConfigSchema):
    n_samples_per_epoch: int
    n_train_sample_camera_rays: int
    n_train_sample_lidar_rays: int = Field(default=2048)
    n_val_image_subsample: int
    n_train_sequential_image_subsample: int
    aux_data: bool


class CuboidTracksConfig(BaseConfigSchema):
    def model_post_init(self, __context) -> None:
        assert self.track_label_sources is None or (
            all(
                (
                    # Label sources are optionally suffixed by a version string after the first '@'
                    # e.g. AUTOLABEL@v1.0 - check _only_ the source type prefix here as the version
                    # suffix is an arbitrary non-empty string
                    s.split("@", 1)[0] in NCoreLabelSource.__members__
                    for s in self.track_label_sources
                )
            )
            and
            # Check that the version suffixes (if specified) are non-empty
            all(len(s.split("@", 1)[1]) > 0 for s in self.track_label_sources if len(s.split("@")) > 1)
        ), (
            f"[NCOREDataSource]: Label sources {self.track_label_sources} need to be "
            f"prefixed strings from {list(NCoreLabelSource.__members__.keys())} with non-empty version suffixes (if specified)"
        )

    track_min_distance_m: float = Field(
        description="Distance (length of path taken by the object) threshold to classify objects to be dynamic (m)",
        default=0.1,
        ge=0.0,
    )
    track_min_displacement_m: float = Field(
        description="Displacement (difference between start and end positions) threshold to classify objects to be dynamic (m)",
        default=0.1,
        ge=0.0,
    )
    track_speed_reduction_op: Literal["median", "max"] = Field(
        description="Reduction operation [median, max] on all pose-inferred speeds to classify for dynamic/static objects",
        default="median",
    )
    track_min_speed_ms: float = Field(
        description="Speed threshold to classify objects to be dynamic (m/s)",
        default=1.5,
        ge=0.0,
    )
    track_min_centroid_rig_dist_m: float = Field(
        description="Distance threshold for cubic tracks to be considered self-classifications to skip (m)",
        default=3.0,
        ge=0.0,
    )
    track_extrapolate: bool = Field(
        description="Extrapolate label pose of dynamic tracks once at the start / end (to improve interpolation coverage)",
        default=True,
    )
    track_label_sources: Optional[list[str]] = Field(
        description="If specified, a list of allowed label sources from [AUTOLABEL, EXTERNAL, GT_SYNTHETIC, GT_ANNOTATION] with optional '@<source-version>' suffixes"
        "Required if dataset contains labels from more than a single source",
        default=None,
    )
    use_displacement_and_distance: bool = Field(
        description="Use displacement and distance thresholds to classify objects to be dynamic",
        default=True,
    )
    camera_visibility: bool = Field(
        description="If enabled, will only use camera-visible time-intervals for track dynamic classification",
        default=False,
    )


class ValidLidarPointsCuboidTrackConfig(BaseConfigSchema):
    track_padding_m: list[float] = Field(
        min_length=3,
        max_length=3,
    )


class ValidPixelsCuboidTrackConfig(BaseConfigSchema):
    track_padding_m: list[float] = Field(
        description="Cuboid bbox padding in local box x/y/z directions (m)",
        min_length=3,
        max_length=3,
    )
    frame_track_idxs: bool = Field(
        description="If enabled, store per-frame track image buffers with per-pixel track indices",
    )


class ValidPixelsSceneFlowConfig(BaseConfigSchema):
    flow_min_speed_ms: float = Field(
        description="Speed threshold to classify objects to be dynamic  [m/s]",
        ge=0.0,
    )
    flow_dilate_radius: int = Field(
        description="Dilation radius for the dynamic mask from scene flow (dilation is performed after downsampling the mask to a low-res)",
        ge=0,
    )
    flow_downsample_scale: int = Field(
        description="Downsample scale for dynamic mask from scene flow (only for accelerating the dilation operation)",
        ge=1,
    )


class ValidPixelsTrafficLightConfig(BaseConfigSchema):
    seg_dilate_radius: int = Field(
        default=21,
        ge=0,
    )


class ValidPixelsFrameMaskConfig(BaseConfigSchema):
    n_dilation_iterations: int = Field(
        description="Number of dilation iterations to apply to per-frame masks",
        default=10,
    )
    classes: Optional[List[str]] = Field(
        description="If specified, keys to query for per-frame masks. We construct per-frame masks by loading each key via `get_frame_generic_data` and include pixels marked as true as true in any of the per-key masks",
        default=None,
    )


class LidarDynamicPointsConfig(BaseConfigSchema):
    method: Literal["dynamic_tracks", "dynamic_flag"] = Field(
        default="dynamic_tracks",
        description="'dynamic_tracks': Dynamic point classification computed at data loading time from tracks classified to be dynamic\n"
        "'dynamic_flag': NCore-internal per-point 'dynamic_flag' is used [precomputed on potentially different dynamic object classification!]",
    )


class GenerateStaticRigidCuboidTracksConfig(BaseConfigSchema):
    enabled: bool = Field(
        default=False,
        description="If enabled, will generate cuboid tracks for traffic lights",
    )
    max_distance_m: float = Field(
        default=100.0,
        description="Maximum distance to generate static rigid cuboid tracks",
    )
    visibility_check_camera_ids: list[str] = Field(
        default=[],
        description="Camera ids to check for visibility of static rigid objects",
    )
    rigid_classes: list[str] = Field(
        default=["traffic light"],
        description="Classes to generate static rigid cuboid tracks for",
    )
    point_cloud_path: Optional[str] = Field(
        default=None,
        description="Path to accumulated point cloud PLY with segment_id. "
        "Required for lidar-free pipelines where lidar semantic segmentation is unavailable.",
    )


class NCoreDatasetConfig(_BaseDatasetSOConfig):
    name: Literal["ncore"]

    samplers: dict[str, Any] = Field(
        default={},
    )

    path: str = Field(
        description="Data-source specification by a 'path' to an NCore meta-file.\n"
        "The sequence time range can be restricted with additional 'seek_offset_sec' / 'duration_sec' parameters.\n",
    )
    seek_offset_sec: float | None
    duration_sec: float | None

    # Sensors and sensor models
    camera_ids: list[str]
    lidar_ids: list[str]

    # V3 sequence loader parameters
    cuboid_loading_max_workers: Optional[int] = Field(
        description="If specified, set the max_workers for thread parallel cuboid loading in the V3 sequence loader"
        " (automatically infered number for 'null' - serial execution with '0')",
        default=0,
    )

    # V4 sequence loader parameters
    poses_component_group: str = Field(
        description="Name of the V4 component group for poses",
        default="default",
    )
    intrinsics_component_group: str = Field(
        description="Name of the V4 component group for intrinsics",
        default="default",
    )
    masks_component_group: str = Field(
        description="Name of the V4 component group for masks",
        default="default",
    )
    cuboids_component_group: str = Field(
        description="Name of the V4 component group for cuboids",
        default="default",
    )

    train_camera_ids: list[str]
    train_lidar_ids: list[str]

    val_camera_ids: list[str]
    val_lidar_ids: list[str]

    lidar_ignore_rows: list[int] = Field(
        description="If specified, will ignore the specified rows of the lidar points",
        default=[],
    )

    show_progress_bars: bool = Field(
        description="If enabled (default), will show progress bars during dataset initialization.",
        default=True,
    )

    # Scale
    max_dist_m: float
    val_camera: bool = Field(
        description="If enabled, will return camera rays in validation/test time.",
        default=True,
    )
    val_lidar: bool = Field(
        description="If enabled, will return lidar rays in validation/test time.",
        default=False,
    )
    # Translation (meters, relative to rig frame) / rotation (deg, XYZ / roll-pitch-yaw Euler angles relative
    # to rig frame) extrinsic deltas to apply to all sensor extrinsics for validation / test data generation (e.g., for novel-view rendering)
    # If extrinsic deltas are applied, rays will be inconsistent with label data / most metrics can't be evaluated
    val_sensor_transl_delta_m: Optional[list[float]]
    val_sensor_rot_delta_deg: Optional[list[float]]
    val_camera_frame_start: Optional[int] = Field(
        description="Camera frame start index for validation / test mode. Used together with val_camera_frame_step.",
        default=0,
    )
    val_camera_frame_step: Optional[int] = Field(
        description="Camera frame step count for validation / test mode. Mutually exclusive with val_camera_exclude_frame_step, but one must be specified.",
        default=1,
    )
    val_camera_exclude_frame_start: Optional[int] = Field(
        description="Start camera frame index to exclude in validation / test mode. "
        "Used together with val_camera_exclude_frame_step.",
        default=None,
    )
    val_camera_exclude_frame_step: Optional[int] = Field(
        description="Excludes camera frames every nth step for validation / test mode. Mutually exclusive with val_camera_frame_step, but one must be specified.",
        default=None,
    )
    val_lidar_frame_start: Optional[int] = Field(
        description="Lidar frame start index for validation / test mode. Used together with val_lidar_frame_step.",
        default=0,
    )
    val_lidar_frame_step: Optional[int] = Field(
        description="Lidar frame step count for validation / test mode. Mutually exclusive with val_lidar_exclude_frame_step, but one must be specified.",
        default=1,
    )
    val_lidar_exclude_frame_start: Optional[int] = Field(
        description="Start lidar frame index to exclude in validation / test mode. "
        "Used together with val_lidar_exclude_frame_step.",
        default=None,
    )
    val_lidar_exclude_frame_step: Optional[int] = Field(
        description="Excludes lidar frames every nth step for validation / test mode. Mutually exclusive with val_lidar_frame_step, but one must be specified.",
        default=None,
    )

    camera_mask_sources: List[Literal["dataset", "aux"]] = Field(
        description="Priority order of data sources to load camera masks from (possible values: [dataset, aux])",
        default=["dataset", "aux"],
    )
    n_camera_mask_dilation_iterations: int = Field(
        description="Number of dilation iterations to apply to camera masks",
        default=30,
    )
    camera_mask_overrides: Dict[str, str] = Field(
        description="Override camera masks for specified sensors",
        default={},
    )
    camera_mask_border: Optional[Dict[str, list[int]]] = Field(
        description="If specified, excludes pixels within the specified border (top, right, bottom, left) for each listed camera",
        default=None,
    )

    open_consolidated: bool
    jpeg_backend_cpu: Literal["PIL", "simplejpeg"] = Field(
        description="Backend to use for CPU-based JPEG decoding [PIL, simplejpeg]",
        default="simplejpeg",
    )
    simplejpeg_fastdct: bool = Field(
        description="If enabled, use fastest DCT method for JPEG decoding (via simplejpeg) - speeds up decoding by 4-5% for a minor loss in quality",
        default=False,
    )
    simplejpeg_fastupsample: bool = Field(
        description="If enabled, use fastest color upsampling method for JPEG decoding (via simplejpeg) - speeds up decoding by 4-5% for a minor loss in quality",
        default=False,
    )
    frame_generic_data_pose_overwrite: bool = Field(
        description="If enabled, will overwrite per-frame sensor-world poses from generic data fields (used in some, e.g., separately optimized datasets) - it's an error if these are not provided by the data if enabled",
        default=False,
    )
    camera_max_fov_deg: float = Field(
        description="Limit effective FOV of omnidirectional camera models (like FTheta / OpenCVFisheye) to prevent erroneous flipped re-projections",
        default=190.0,
    )
    lidar_model_parameter_cwccw_fallback: bool = Field(
        description="If enabled, try to apply a backwards-compatiblity fallback for NCore lidar model parameters, which were stored with wrong spin direction conventions",
        default=False,
    )
    lidar_model_parameter_nominal_values: dict[str, Literal["HESAI-Pandar128", "HESAI-AT128"]] = Field(
        description="If specified, use the specified nominal lidar model parameters for each lidar_id",
        default={},
    )

    # The following paramaters are used for 'train_sequential' dataset splits, which exhaustively provides datasource samples in linear order (e.g., used for modules such as the uncertainty estimators that need to be fitted against training rays at the end of training)
    train_sequential_camera: bool = Field(
        description="Used for 'train_sequential' dataset splits, which exhaustively provide datasource samples in linear order "
        "(e.g., used for modules such as the uncertainty estimators that need to be fitted against training rays at the end of training)",
        default=True,
    )
    train_sequential_lidar: bool = Field(
        description="Used for 'train_sequential' dataset splits, which exhaustively provide datasource samples in linear order "
        "(e.g., used for modules such as the uncertainty estimators that need to be fitted against training rays at the end of training)",
        default=False,
    )
    train_sequential_camera_frame_start: Optional[int] = Field(
        description="Camera frame start index for 'train_sequential' mode. "
        "Used together with train_sequential_camera_frame_step.",
        default=0,
    )
    train_sequential_camera_frame_step: Optional[int] = Field(
        description="Used for 'train_sequential' dataset splits. "
        "Mutually exclusive with `train_sequential_camera_exclude_frame_step`, but one must be specified.",
        default=1,
    )
    train_sequential_camera_exclude_frame_start: Optional[int] = Field(
        description="Start camera frame index to exclude in 'train_sequential' mode. "
        "Used together with train_sequential_camera_exclude_frame_step.",
        default=None,
    )
    train_sequential_camera_exclude_frame_step: Optional[int] = Field(
        description="Used for 'train_sequential' dataset splits. "
        "Mutually exclusive with `train_sequential_camera_frame_step`, but one must be specified.",
        default=None,
    )
    train_sequential_lidar_frame_start: Optional[int] = Field(
        description="Lidar frame start index for 'train_sequential' mode. "
        "Used together with train_sequential_lidar_frame_step.",
        default=0,
    )
    train_sequential_lidar_frame_step: Optional[int] = Field(
        description="Used for 'train_sequential' dataset splits. "
        "Mutually exclusive with `train_sequential_lidar_exclude_frame_step`, but one must be specified.",
        default=1,
    )
    train_sequential_lidar_exclude_frame_start: Optional[int] = Field(
        description="Start lidar frame index to exclude in 'train_sequential' mode. "
        "Used together with train_sequential_lidar_exclude_frame_step.",
        default=None,
    )
    train_sequential_lidar_exclude_frame_step: Optional[int] = Field(
        description="Used for 'train_sequential' dataset splits. "
        "Mutually exclusive with `train_sequential_lidar_frame_step`, but one must be specified.",
        default=None,
    )

    cuboid_tracks_params: CuboidTracksConfig
    valid_measurements_method: Literal[
        "EGO",
        "EGO_CUBOIDTRACKS",
        "EGO_SCENEFLOW",
        "EGO_CUBOIDTRACKS_SCENEFLOW",
        "EGO_CUBOIDTRACKS_TRAFFICLIGHT",
        "EGO_FRAMEMASKS",
    ] = Field(
        description="Valid pixels/lidarpoints method to invoke. Options:\n"
        " - EGO: ego camera masks only\n"
        " - EGO_CUBOIDTRACKS: ego camera masks and frame-projected dynamic objects from cuboid tracks; dynamic_flag lidar points\n"
        " - EGO_SCENEFLOW: ego camera masks and scene flow masks\n"
        " - EGO_CUBOIDTRACKS_SCENEFLOW: ego camera masks, frame-projected dynamic objects from cuboid tracks, and scene flow masks;"
        " dynamic_flag lidar points\n"
        " - EGO_CUBOIDTRACKS_TRAFFICLIGHT: ego camera masks, frame-projected dynamic objects from cuboid tracks, and traffic light"
        " masks; dynamic_flag, non-traffic light, camera-visible lidar points\n"
        " - EGO_FRAMEMASKS: ego camera masks and per-frame masks\n",
    )

    valid_lidarpoints_cuboid_track_params: ValidLidarPointsCuboidTrackConfig
    valid_pixels_cuboid_track_params: ValidPixelsCuboidTrackConfig
    valid_pixels_scene_flow_params: ValidPixelsSceneFlowConfig
    valid_pixels_traffic_light_params: ValidPixelsTrafficLightConfig
    valid_pixels_frame_mask_params: ValidPixelsFrameMaskConfig
    lidar_dynamic_points: LidarDynamicPointsConfig
    camera_point_cloud_ignore_classes: list[str]
    camera_point_cloud_dynamic_classes: list[str]
    generate_static_rigid_cuboid_tracks: GenerateStaticRigidCuboidTracksConfig = Field(
        default_factory=GenerateStaticRigidCuboidTracksConfig
    )

    interp_with_rig: bool = Field(
        description="If enabled, will interpolate the camera poses with the rig trajectory.",
        default=True,
    )

    use_real_lidar_rays: bool = Field(
        description="If enabled, will use the lidar points to compute the lidar rays from the dataset.",
        default=False,
    )
