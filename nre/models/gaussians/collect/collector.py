# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Gaussian Parameter Collector Interface.

This module defines the interface for collecting and transforming Gaussian parameters
from multiple layers into a unified representation suitable for rendering.

Overview:
---------
The Gaussian parameter collector is responsible for:
1. Gathering Gaussian parameters (positions, rotations, scales, densities, features) from multiple layers
2. Applying activations (e.g., exp to scales, sigmoid to densities, normalization to rotations)
3. Transforming rigid and deformable layers using track poses
4. Concatenating all layers into unified output tensors
5. Computing time-based embeddings for temporal features (Fourier features)
6. Preparing input embeddings for hash grids (xyz + instance + time)

Architecture:
-------------
The collector supports multiple layer types:
- Base layers: Static Gaussians with basic properties
- Spherical Harmonics (SH) layers: Add view-dependent features (albedo/specular)
- Rigid layers: Gaussians attached to moving tracks (rigid transformations)
- Deformable layers: Gaussians with learned deformations on top of rigid motion

Workflow:
---------
1. Setup (once at collector creation):
   - Configuration: Define layer types and their properties via LayersConfig
2. Per-collection (every training step or rendered frame):
   - Data preparation: Pack per-layer data into LayerData structures
   - Collection: Call collect() to transform and concatenate all layers
   - Output: Receive unified tensors ready for rendering

The implementation uses Slang GPU kernels for high-performance parallel processing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import torch


# Type of activation supported by this library.
class RotationActivation(Enum):
    """Activation functions applied to rotation quaternions.

    NORMALIZE: Normalize quaternions to unit length to ensure valid rotations.
    """

    NORMALIZE = 0


class ScaleActivation(Enum):
    """Activation functions applied to scale parameters.

    EXP: Apply exponential to ensure positive scales.
    """

    EXP = 0


class DensityActivation(Enum):
    """Activation functions applied to density/opacity values.

    SIGMOID: Apply sigmoid to constrain densities to [0, 1] range.
    """

    SIGMOID = 0


#
# Config classes, describes what is in the layer.
#
@dataclass(slots=True)
class LayerConfigBase:
    """Base configuration for a Gaussian layer.

    Defines the activation functions to apply during collection.

    Attributes:
        rotation_activation: Activation function for rotations (e.g., normalization)
        scale_activation: Activation function for scales (e.g., exp)
        density_activation: Activation function for densities (e.g., sigmoid)
    """

    rotation_activation: RotationActivation
    scale_activation: ScaleActivation
    density_activation: DensityActivation


@dataclass(slots=True)
class EmbeddingConfig:
    """Base class for time embedding configurations."""

    pass


@dataclass(slots=True)
class IndividualRemapTimeInputEmbeddingConfig(EmbeddingConfig):
    """Time embedding with per-instance timestamp ranges.

    Each instance (e.g., dynamic object) has its own time range that gets
    remapped to a normalized interval for embedding.

    Attributes:
        timestamps_us_ranges: Tensor of shape [num_instances, 2] containing
            [min_timestamp, max_timestamp] in microseconds for each instance
        remap_min: Minimum value of the remapped time range
        remap_max: Maximum value of the remapped time range
    """

    timestamps_us_ranges: torch.Tensor
    remap_min: float
    remap_max: float


@dataclass(slots=True)
class HolisticRemapTimeInputEmbeddingConfig(EmbeddingConfig):
    """Time embedding with a single global timestamp range.

    All instances share the same time range for normalization.

    Attributes:
        timestamps_us_min: Minimum timestamp in microseconds
        timestamps_us_max: Maximum timestamp in microseconds
        remap_min: Minimum value of the remapped time range
        remap_max: Maximum value of the remapped time range
    """

    timestamps_us_min: int
    timestamps_us_max: int
    remap_min: float
    remap_max: float


@dataclass(slots=True)
class IndividualStepTimeInputEmbeddingConfig(EmbeddingConfig):
    """Step-function time embedding with per-instance timestamp ranges.

    Uses a differentiable step function from NSC paper (https://arxiv.org/abs/2306.07970) for temporal embedding,
    with per-instance time ranges.

    Attributes:
        timestamps_us_ranges: Tensor of shape [num_instances, 2] (int64) containing
            [min_timestamp, max_timestamp] in microseconds for each instance
        n_steps: Number of steps in the step function
        n_dims: Number of output dimensions
    """

    timestamps_us_ranges: torch.Tensor  # [num_instances, 2], int64
    n_steps: int
    n_dims: int


@dataclass(slots=True)
class LayerConfigSH(LayerConfigBase):
    """Configuration for a Spherical Harmonics layer with view-dependent features.

    Extends base layer with albedo and specular features, optionally with
    Fourier-based temporal modulation.

    Attributes:
        fourier_features_dim: Number of Fourier frequency bands for temporal features.
            If > 1, features are modulated over time using Fourier basis.
            If == 1, features are static (no temporal modulation).
        embed_config: Time embedding configuration (required if fourier_features_dim > 1,
            None otherwise)
    """

    fourier_features_dim: int
    embed_config: EmbeddingConfig | None


@dataclass(slots=True)
class LayerConfigRigid(LayerConfigSH):
    """Configuration for a rigid layer.

    Gaussians are attached to tracks (moving objects) and undergo rigid
    transformations based on track poses.
    """

    pass


@dataclass(slots=True)
class LayerConfigDeformable(LayerConfigRigid):
    """Configuration for a deformable layer.

    Extends rigid layer with learned per-Gaussian deformations applied
    on top of the rigid track motion.
    """

    pass


@dataclass(slots=True)
class LayersConfig:
    """Complete configuration for all layers in the collector.

    Attributes:
        layers: List of per-layer configurations
        extra_signal_dim: Dimension of extra signal features (shared across all layers)
        camera_extra_signal_dim: Dimension of camera-specific features (shared across all layers)
        lidar_extra_signal_dim: Dimension of lidar-specific features (shared across all layers)
        albedo_dim: Dimension of albedo (diffuse color) features
        specular_dim: Dimension of specular (view-dependent) features
    """

    layers: list[LayerConfigBase]
    extra_signal_dim: int
    camera_extra_signal_dim: int
    lidar_extra_signal_dim: int
    albedo_dim: int
    specular_dim: int


#
# Each layer's interface to the per-layer data needed by the collector.
#
@dataclass(slots=True)
class LayerDataBase:
    """Base data for a Gaussian layer.

    Contains the fundamental parameters for all Gaussians in a layer.

    Attributes:
        positions: Tensor of shape [num_gaussians, 3] - 3D positions in local space
        rotations: Tensor of shape [num_gaussians, 4] - Quaternions in wxyz format
        scales: Tensor of shape [num_gaussians, 3] - Log-space scales (before exp activation)
        densities: Tensor of shape [num_gaussians, 1] - Logit-space densities (before sigmoid)
        extra_signal: Tensor of shape [num_gaussians, extra_signal_dim] - Generic extra features
        camera_extra_signal: Tensor of shape [num_gaussians, camera_extra_signal_dim] - Camera features
        lidar_extra_signal: Tensor of shape [num_gaussians, lidar_extra_signal_dim] - Lidar features
    """

    positions: torch.Tensor
    rotations: torch.Tensor
    scales: torch.Tensor
    densities: torch.Tensor
    extra_signal: torch.Tensor
    camera_extra_signal: torch.Tensor
    lidar_extra_signal: torch.Tensor


@dataclass(slots=True)
class EmbeddingData:
    """Base class for per-layer embedding data."""

    pass


@dataclass(slots=True)
class IndividualRemapTimeInputEmbeddingData(EmbeddingData):
    """Per-Gaussian instance indices for individual time embedding.

    Attributes:
        instance_idx: Tensor of shape [num_gaussians] (int32) - Instance ID for each Gaussian,
            used to look up per-instance timestamp ranges
    """

    instance_idx: torch.Tensor


@dataclass(slots=True)
class IndividualStepTimeInputEmbeddingData(EmbeddingData):
    """Per-Gaussian data for step time embedding.

    Attributes:
        instance_idx: Tensor of shape [num_gaussians] (int32)
        u: Tensor of shape [num_instances, n_dims, n_steps] - current step positions (trainable)
        beta: Current beta value for stepness
    """

    instance_idx: torch.Tensor
    u: torch.Tensor  # Trainable!
    beta: torch.Tensor


@dataclass(slots=True)
class LayerDataSH(LayerDataBase):
    """Data for a Spherical Harmonics layer with view-dependent features.

    Extends base layer with appearance features.

    Attributes:
        features_albedo: Tensor of shape [num_gaussians, fourier_dim, albedo_dim] if using
            Fourier features, or [num_gaussians, albedo_dim] otherwise - Diffuse color features
        features_specular: Tensor of shape [num_gaussians, specular_dim] - View-dependent features
        embed_data: Optional embedding data for time-varying features
    """

    features_albedo: torch.Tensor
    features_specular: torch.Tensor
    embed_data: EmbeddingData | None


@dataclass(slots=True)
class TracksCalibData:
    """Base class for track calibration data.

    Used to apply extrinsic calibration corrections to track poses.

    Attributes:
        tracks_poses: Tensor of shape [num_poses, 7] - Poses as [translation(3), quaternion_xyzw(4)]
    """

    tracks_poses: torch.Tensor


@dataclass(slots=True)
class DirectTracksCalibData(TracksCalibData):
    """Direct parameterization of track calibration.

    Applies learned delta transformations to track poses.

    Attributes:
        gradient_mask: Tensor of shape [num_poses] (bool) - Whether to allow gradients per pose.
            Typically used to fix the first and last pose of a track, preventing them from
            being modified by optimization.
        tracks_delta_q: Tensor of shape [num_poses, 4] - Quaternion deltas (xyzw format)
        tracks_delta_t: Tensor of shape [num_poses, 3] - Translation deltas
    """

    gradient_mask: torch.Tensor
    tracks_delta_q: torch.Tensor
    tracks_delta_t: torch.Tensor


@dataclass(slots=True)
class TracksTimestampsData:
    """Base class for timestamp selection strategies during track interpolation."""

    pass


@dataclass(slots=True)
class TracksTimestampsGlobalData(TracksTimestampsData):
    """Use a single timestamp for all tracks.

    Attributes:
        timestamp: Timestamp in microseconds (int)
    """

    timestamp: int


@dataclass(slots=True)
class TracksTimestampsPerTrackData(TracksTimestampsData):
    """Use pre-specified timestamps per track.

    Attributes:
        timestamps: Tensor of shape [num_tracks] (int64) - Timestamp in microseconds per track
    """

    timestamps: torch.Tensor


@dataclass(slots=True)
class TracksTimestampsEstimationData(TracksTimestampsData):
    """Estimate timestamps per track via ray-cuboid intersection.

    Determines the best timestamp for each track by intersecting rays with
    track-aligned bounding boxes.

    Attributes:
        rays: Tensor of shape [num_rays, 6] - Ray origins (3) and directions (3)
        rays_timestamps: Tensor of shape [num_rays] (int64) - Timestamp per ray in microseconds
        cuboids_dimensions: Tensor of shape [num_tracks, 3] - Bounding box dimensions per track
        default_timestamp: Default timestamp in microseconds for tracks with no intersections
    """

    rays: torch.Tensor
    rays_timestamps: torch.Tensor
    cuboids_dimensions: torch.Tensor
    default_timestamp: int


@dataclass(slots=True)
class TracksInterpolationData:
    """Data for interpolating track poses at query timestamps.

    Tracks are represented as time-series of poses, packed into flat arrays.

    Attributes:
        tracks_poses: Tensor of shape [total_num_poses, 7] - All poses concatenated,
            format: [translation(3), quaternion_xyzw(4)]
        tracks_timestamps: Tensor of shape [total_num_poses] (int64) - Timestamp per pose
        tracks_packinfo: Tensor of shape [num_tracks, 2] (int32) - [start_index, length] per track
        timestamps_data: Strategy for selecting query timestamps
        nearest_neighbor: If True, use nearest neighbor instead of interpolation
    """

    tracks_poses: torch.Tensor
    tracks_timestamps: torch.Tensor
    tracks_packinfo: torch.Tensor
    timestamps_data: TracksTimestampsData
    nearest_neighbor: bool


@dataclass(slots=True)
class LayerDataRigid(LayerDataSH):
    """Data for a rigid layer attached to moving tracks.

    Each Gaussian is associated with a track and transforms according to
    the track's pose.

    Attributes:
        poses: Tensor of shape [num_tracks, 7] - Interpolated/calibrated poses for tracks,
            format: [translation(3), quaternion_xyzw(4)]
        keep_mask: Tensor of shape [num_tracks] (bool) - Whether each track is visible/active
        tracks_ids: Tensor of shape [num_gaussians] (int32) - Track ID for each Gaussian
    """

    poses: torch.Tensor
    keep_mask: torch.Tensor
    tracks_ids: torch.Tensor


@dataclass(slots=True)
class SceneContractorData:
    """Configuration for scene space contraction.

    Contracts world space into a bounded domain for hash grid encoding.

    Attributes:
        aabb_blb: Tensor of shape [num_instances, 3] - Axis-aligned bounding box
            bottom-left-back corner per instance
        aabb_trf: Tensor of shape [num_instances, 3] - Axis-aligned bounding box
            top-right-front corner per instance
        degree: Lp-norm degree for contraction (e.g., 2.0 for L2, inf for L-infinity),
            or None for no contraction
        is_merf: If True, use MERF-style contraction; otherwise use standard contraction
    """

    aabb_blb: torch.Tensor
    aabb_trf: torch.Tensor
    degree: float | None
    is_merf: bool


@dataclass(slots=True)
class InstanceEmbeddingData:
    """Base class for instance embedding data."""

    pass


@dataclass(slots=True)
class WeightedInstanceInputEmbeddingData(InstanceEmbeddingData):
    """Instance embeddings via learned weight vectors.

    Attributes:
        instance_embedding_weights: Tensor of shape [num_instances, embedding_dim] -
            Learned embedding vectors per instance
    """

    instance_embedding_weights: torch.Tensor


@dataclass(slots=True)
class InputEmbeddingData:
    """Data for computing input embeddings for hash grid (xyz + instance_emb + time_emb).

    Prepares concatenated feature vectors suitable for hash grid input, combining
    spatial coordinates, instance identity, and temporal information.

    Attributes:
        xyzs: Tensor of shape [num_gaussians, 3] - 3D world coordinates
        instance_idx: Tensor of shape [num_gaussians] (int32) - Instance ID per Gaussian
        timestamps_us: Timestamp in microseconds (single value for all Gaussians)
        scene_contractor: Configuration for spatial contraction
        instance_embedding_weights: Instance embedding configuration
        time_embedding_config: Time embedding configuration
        timestamps_delta: Optional tensor of shape [num_gaussians] (int64) - Per-Gaussian timestamp deltas
            in microseconds. If provided, output shape becomes [num_gaussians*3, base_dim] with embeddings
            stacked as [t, t-delta, t+delta] along the batch dimension
    """

    xyzs: torch.Tensor
    instance_idx: torch.Tensor
    timestamps_us: int

    scene_contractor: SceneContractorData
    instance_embedding_weights: InstanceEmbeddingData
    time_embedding_config: EmbeddingConfig

    timestamps_delta: torch.Tensor | None = None


@dataclass(slots=True)
class LayerDataDeformable(LayerDataRigid):
    """Data for a deformable layer with learned per-Gaussian deformations.

    Extends rigid layer with optional deformation offsets applied in local space
    before the rigid transformation.

    Attributes:
        deform_positions: Optional tensor of shape [num_gaussians, 3] - Position offsets
        deform_rotations: Optional tensor of shape [num_gaussians, 4] - Rotation offsets
            (quaternion xyzw format, as offset from identity)
    """

    deform_positions: torch.Tensor | None
    deform_rotations: torch.Tensor | None


@dataclass(slots=True)
class LayersData:
    """Complete input data for all layers.

    Attributes:
        layers: List of per-layer data structures
        frame_timestamp_us: Global frame timestamp in microseconds (used for Fourier features)
    """

    layers: list[LayerDataBase]
    frame_timestamp_us: int


# The returned concatenated tensors.
@dataclass(slots=True)
class CollectorResult:
    """Unified output from the collector.

    All tensors are concatenations across all layers, with activations applied.

    Attributes:
        positions: Tensor of shape [total_gaussians, 3] - World-space positions
        rotations: Tensor of shape [total_gaussians, 4] - Normalized quaternions (wxyz format)
        scales: Tensor of shape [total_gaussians, 3] - Positive scales (after exp)
        densities: Tensor of shape [total_gaussians, 1] - Densities in [0,1] (after sigmoid)
        extra_signal: Tensor of shape [total_gaussians, extra_signal_dim]
        camera_extra_signal: Tensor of shape [total_gaussians, camera_extra_signal_dim]
        lidar_extra_signal: Tensor of shape [total_gaussians, lidar_extra_signal_dim]
        features: Tensor of shape [total_gaussians, albedo_dim + specular_dim] -
            Concatenated appearance features (with Fourier applied if configured)
    """

    positions: torch.Tensor
    rotations: torch.Tensor
    scales: torch.Tensor
    densities: torch.Tensor
    extra_signal: torch.Tensor
    camera_extra_signal: torch.Tensor
    lidar_extra_signal: torch.Tensor
    features: torch.Tensor


# The actual collector interface.
class GaussianParameterCollector(ABC):
    """Abstract interface for Gaussian parameter collection and transformation.

    This is the main interface for collecting, transforming, and preparing Gaussian
    parameters for rendering. Implementations use GPU kernels for high performance.
    """

    @abstractmethod
    def __init__(self, layers_config: LayersConfig):
        """Initialize the collector with layer configuration.

        Args:
            layers_config: Configuration specifying layer types and dimensions
        """
        ...

    @abstractmethod
    def collect(self, layers_data: LayersData, layer_indices: Optional[List[int]] = None) -> CollectorResult:
        """Collect and transform Gaussians from all layers.

        This is the main entry point that:
        1. Applies activations to base parameters
        2. Transforms rigid/deformable layers using track poses
        3. Computes time-varying features (Fourier)
        4. Concatenates all layers into unified output tensors

        Args:
            layers_data: Input data for all layers
            layer_indices: Optional mapping from each entry in layers_data.layers
                to the corresponding layer_handler index. When None, layers_data.layers
                must contain all layers in order. When provided, enables collecting
                a subset of layers using the correct handler and layers_data.layers should
                only contain the subset of the collected layers, in the order matching the indices

        Returns:
            CollectorResult containing transformed and concatenated parameters
        """
        ...

    @abstractmethod
    def calibrate_tracks_poses(self, tracks_calib_data: TracksCalibData) -> torch.Tensor:
        """Apply calibration corrections to track poses.

        Adjusts track poses using learned extrinsic calibration parameters.

        Args:
            tracks_calib_data: Calibration parameters and base poses

        Returns:
            Calibrated poses tensor of shape [num_poses, 7] in format
            [translation(3), quaternion_xyzw(4)]
        """
        ...

    @abstractmethod
    def interpolate_tracks_poses(
        self, tracks_interpolation_data: TracksInterpolationData
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Interpolate track poses at query timestamps.

        Given time-series of poses per track, interpolates to find poses at
        specified timestamps using either linear interpolation or nearest neighbor.

        Args:
            tracks_interpolation_data: Packed poses, timestamps, and query strategy

        Returns:
            Tuple of:
            - poses: Tensor of shape [num_tracks, 7] - Interpolated poses
            - inside: Tensor of shape [num_tracks] (bool) - Whether timestamp was
              within the track's time range (True) or clamped to the first/last
              pose (False)
        """
        ...

    @abstractmethod
    def prepare_input_embeddings(self, input_embedding_data: InputEmbeddingData) -> torch.Tensor:
        """Prepare input embeddings for hash grid encoding.

        Computes concatenated feature vectors combining spatial coordinates,
        instance embeddings, and time embeddings.

        Args:
            input_embedding_data: Input coordinates, instance IDs, and embedding configs

        Returns:
            Tensor of shape:
            - [num_gaussians, base_dim] if timestamps_delta is None
            - [num_gaussians*3, base_dim] if timestamps_delta is provided (stacked as [t, t-delta, t+delta])
            where base_dim = 3 (xyz) + instance_embedding_dim + 1 (time)
        """
        ...
