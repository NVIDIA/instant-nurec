# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Dict, List, Optional

from nre.config.base_schema import BaseConfigSchema, Field
from nre.config.version import Version, get_version


current_version = get_version()


class RotationSpec(BaseConfigSchema):
    """Single rigid rotation: axis (x/y/z) and degrees."""

    axis: str = "y"
    degrees: float = 0.0


class OutputRotationConfig(BaseConfigSchema):
    """Per-class rigid rotation applied to each PLY after lifting save.

    TokenGS outputs in OpenGL Y-up; some classes (vehicles, riders) end up
    180° from desired heading in NRE's downstream frame while others (e.g.
    persons) are correct as-is. Map keys are normalized label classes
    (mvdata.npct, e.g. "automobile", "rider", "person"). Classes not listed
    receive no rotation.
    """

    by_class: Dict[str, RotationSpec] = Field(default_factory=dict)


class OutputConfig(BaseConfigSchema):
    """Output-side options applied after Gaussian lifting."""

    rotation: OutputRotationConfig


class AssetHarvestingConfig(BaseConfigSchema):
    """Main configuration for asset harvesting pipeline"""

    class NCoreParserConfig(BaseConfigSchema):
        """Configuration for ncore parser stage of asset harvesting pipeline"""

        target_resolution: int
        num_lidar_ref_frames: int
        cam_pose_flip: List[int]
        max_threads: int
        occ_rate_threshold: float
        crop_min_area_ratio: float
        mask_exceed_threshold: float
        min_instance_pixels: int
        mask_overlap_threshold: float
        camera_ids: List[str]
        segmentation_ckpt: str = ""

    ncore_parser: NCoreParserConfig

    class TokengsLiftingConfig(BaseConfigSchema):
        """Configuration for TokenGS Gaussian lifting stage"""

        use_ttt: bool = False
        bbox_size: float = 1.0

    tokengs_lifting: TokengsLiftingConfig

    output: OutputConfig

    class CheckpointURLs(BaseConfigSchema):
        """NGC URLs for model checkpoints — typed for typo protection at parse time."""

        mv_diffusion_ckpt: str
        tokengs_ckpt: str
        segmentation_ckpt: str

    urls: CheckpointURLs


class MVDataView(BaseConfigSchema):
    """Data for a single view matching MVData structure"""

    frame: str  # file path to frame image (relative to metadata file)
    instance_mask: Optional[str] = None  # file path to instance mask (relative to metadata file)

    # Per-view metadata
    cam_pose: List[float]  # 3D camera position (from cam_poses array)
    dist: float  # distance (from dists array)
    fov: float  # field of view (from fov array)
    sensor_id: str  # camera sensor ID


class MultiViewData(BaseConfigSchema):
    """Container for multi-view data including bbox position and views"""

    bbox_pos: List[float]  # bbox pose relative to cam (moved from Asset)
    views: List[MVDataView]  # list of all views for this asset


class AssetMetrics(BaseConfigSchema):
    """Metrics for asset reconstruction quality"""

    psnr_mean: float
    psnr_std: float
    ssim_mean: float
    ssim_std: float


class Asset(BaseConfigSchema):
    """Matches the MVData (MultiView data) structure from new_mvdata"""

    # Identifiers
    clip_id: str
    track_id: str

    # Object properties
    label_class: str  # vehicle category
    cuboids_dims: List[float]  # vehicle dimensions [length, width, height]

    ply_file: str  # path to ply file for this asset

    # Reconstruction quality metrics
    metrics: Optional[AssetMetrics] = None

    # Multi-view input data
    multiview_data: MultiViewData


class AssetHarvestingMetadata(BaseConfigSchema):
    clip_id: str
    config: AssetHarvestingConfig
    assets: Optional[Dict[str, Asset]] = None
    version: Version | None = Field(
        default=current_version,
        description="Not to be set by the user. Used to detect NRE version mismatch when loading old configs. Not available in sandboxed test executions",
    )
