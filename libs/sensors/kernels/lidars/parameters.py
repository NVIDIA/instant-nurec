# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""LiDAR parameter dataclasses for Layer 0 kernel operations.

These dataclasses mirror the Slang structs and contain working parameters for GPU execution.
They are used by Layer 0 kernels and created by Layer 2 models.
"""

from dataclasses import dataclass
from typing import ClassVar, Literal, Optional

import numpy as np
import torch

from scipy import spatial as scipy_spatial
from torch import Tensor


def _sensor_angles_to_rays_np(sensor_angles: np.ndarray) -> np.ndarray:
    """Convert sensor angles (elevation, azimuth) to unit rays (numpy version).

    Args:
        sensor_angles: (N, 2) array of (elevation, azimuth) in radians

    Returns:
        rays: (N, 3) array of unit direction vectors
    """
    elevation = sensor_angles[:, 0]
    azimuth = sensor_angles[:, 1]
    cos_elev = np.cos(elevation)
    x = cos_elev * np.cos(azimuth)
    y = cos_elev * np.sin(azimuth)
    z = np.sin(elevation)
    return np.stack([x, y, z], axis=-1)


def _to_numpy(t: Tensor) -> np.ndarray:
    """Convert tensor to numpy, handling device transfer."""
    return t.detach().cpu().numpy()


def _normalize_angle_np(angle: np.ndarray) -> np.ndarray:
    """Normalize angle to (-pi, pi] (numpy version)."""
    angle = angle.copy()
    angle[angle > np.pi] -= 2.0 * np.pi
    angle[angle <= -np.pi] += 2.0 * np.pi
    return angle


def build_angles_to_columns_map(
    n_rows: int,
    n_columns: int,
    row_elevations_rad: Tensor,
    column_azimuths_rad: Tensor,
    row_azimuth_offsets_rad: Optional[Tensor],
    fov_vert_start_rad: float,
    fov_vert_span_rad: float,
    fov_horiz_start_rad: float,
    fov_horiz_span_rad: float,
    spinning_direction: Literal["cw", "ccw"] = "cw",
    resolution_factor: int = 4,
) -> Tensor:
    """Build a 2D precomputed lookup map from (elevation, azimuth) to column indices.

    Creates a 2D map. The map is indexed by
    quantized (elevation_idx, azimuth_idx) and returns the nearest column index.
    This properly accounts for row azimuth offsets.

    Uses scipy's cKDTree for efficient O(n log n) nearest-neighbor search,

    The map is built by:
    1. Creating a regular grid of (elevation, azimuth) angles covering the FOV
    2. Converting all grid points and all sensor elements to unit rays
    3. Using KD-tree nearest-neighbor search to map grid points to sensor elements
    4. Extracting column indices

    Args:
        n_rows: Number of rows
        n_columns: Number of columns
        row_elevations_rad: (n_rows,) elevation angles in radians
        column_azimuths_rad: (n_columns,) azimuth angles in radians
        row_azimuth_offsets_rad: (n_rows,) azimuth offsets per row, or None
        fov_vert_start_rad: Start of vertical FOV in radians
        fov_vert_span_rad: Span of vertical FOV in radians
        fov_horiz_start_rad: Start of horizontal FOV in radians
        fov_horiz_span_rad: Span of horizontal FOV in radians
        spinning_direction: 'cw' or 'ccw'
        resolution_factor: Multiplier for map resolution (default 4).
            Map shape = (resolution_factor * n_rows, resolution_factor * n_columns).

    Returns:
        angles_to_columns_map: (height, width) int32 tensor where
            height = resolution_factor * n_rows
            width = resolution_factor * n_columns
    """
    device = column_azimuths_rad.device

    map_height = resolution_factor * n_rows
    map_width = resolution_factor * n_columns

    # Convert tensors to numpy for scipy operations
    row_elevations_np = _to_numpy(row_elevations_rad)
    column_azimuths_np = _to_numpy(column_azimuths_rad)
    row_offsets_np = _to_numpy(row_azimuth_offsets_rad) if row_azimuth_offsets_rad is not None else None

    # Create regular angle grid in the FOV of the sensor
    # Vertical: start at top (fov_vert_start), go down (cw convention for elevation)
    grid_elevations = np.linspace(
        fov_vert_start_rad,
        fov_vert_start_rad - fov_vert_span_rad,  # cw convention
        map_height,
    )

    # Horizontal: direction depends on spinning_direction
    if spinning_direction == "ccw":
        grid_azimuths = np.linspace(
            fov_horiz_start_rad,
            fov_horiz_start_rad + fov_horiz_span_rad,
            map_width,
        )
    else:  # cw
        grid_azimuths = np.linspace(
            fov_horiz_start_rad,
            fov_horiz_start_rad - fov_horiz_span_rad,
            map_width,
        )

    # Create meshgrid
    grid_elev, grid_az = np.meshgrid(grid_elevations, grid_azimuths, indexing="ij")

    # Convert grid to rays
    grid_angles = np.stack([grid_elev.flatten(), grid_az.flatten()], axis=-1)
    grid_rays = _sensor_angles_to_rays_np(grid_angles)

    # Compute actual sensor angles for all elements (accounting for row offsets)
    # element_azimuths[row, col] = column_azimuths[col] + row_offsets[row]
    if row_offsets_np is not None:
        element_azimuths = column_azimuths_np[np.newaxis, :] + row_offsets_np[:, np.newaxis]
        element_azimuths = _normalize_angle_np(element_azimuths)
    else:
        element_azimuths = np.broadcast_to(column_azimuths_np[np.newaxis, :], (n_rows, n_columns))

    # element_elevations[row, col] = row_elevations[row] (same for all columns)
    element_elevations = np.broadcast_to(row_elevations_np[:, np.newaxis], (n_rows, n_columns))

    # Stack and flatten: shape (n_rows * n_columns, 2)
    # Order: row-major (iterate over columns first for each row)
    sensor_angles = np.stack([element_elevations.flatten(), element_azimuths.flatten()], axis=-1)
    sensor_rays = _sensor_angles_to_rays_np(sensor_angles)

    # Use scipy's cKDTree for efficient O(n log n) nearest-neighbor search
    # Avoids O(n²) memory for large sensors
    kdtree = scipy_spatial.cKDTree(sensor_rays)
    _, nearest_indices = kdtree.query(grid_rays)

    # Convert flat index to column: index = row * n_columns + col => col = index % n_columns
    column_indices = (nearest_indices % n_columns).astype(np.int32)

    # Reshape to 2D map and convert back to torch tensor on original device
    angles_to_columns_map = torch.from_numpy(column_indices.reshape(map_height, map_width)).to(device)

    return angles_to_columns_map


@dataclass
class RowOffsetStructuredSpinningLidarProjection:
    """Structured spinning LiDAR projection (Layer 0 exposed type).

    Mirrors Slang RowOffsetStructuredSpinningLidarProjection struct.
    Contains working parameters for GPU execution.
    Compatible with row-offset models (Hesai P128, Waymo, Pandar) and basic structured LiDARs.

    For basic structured LiDAR (no row offsets):
    - Set row_azimuth_offsets_rad to None
    - Set spinning_frequency_hz to 0.0

    The angles_to_columns_map can be built via ensure_angles_map() or build_angles_to_columns_map().
    This 2D map enables O(1) column lookup for inverse projection with rolling shutter.
    If the map is not set, the kernel falls back to O(n) linear search per ray (inefficient).

    Attributes:
        n_rows: Number of rows (vertical channels)
        n_columns: Number of columns (horizontal samples)
        row_elevations_rad: (n_rows,) elevation angles in radians
        column_azimuths_rad: (n_columns,) azimuth angles in radians
        fov_horiz_start_rad: Start of horizontal FOV in radians
        fov_horiz_span_rad: Span of horizontal FOV in radians
        fov_vert_start_rad: Start of vertical FOV in radians
        fov_vert_span_rad: Span of vertical FOV in radians
        row_azimuth_offsets_rad: (n_rows,) azimuth offsets per row (optional, for row-offset models)
        spinning_frequency_hz: Rotation frequency in Hz (0.0 for non-spinning)
        spinning_direction: 'cw' (clockwise) or 'ccw' (counterclockwise)
        angles_to_columns_map: Optional precomputed inverse map (height, width) for fast lookup
        angles_to_columns_map_resolution_factor: Resolution multiplier for the map
    """

    # Maximum columns supported for linear search fallback
    _MAX_COLUMNS_WITHOUT_MAP: ClassVar[int] = 4096
    # Default resolution factor for map building
    _DEFAULT_MAP_RESOLUTION_FACTOR: ClassVar[int] = 4

    n_rows: int
    n_columns: int
    row_elevations_rad: Tensor  # (n_rows,) float
    column_azimuths_rad: Tensor  # (n_columns,) float
    fov_horiz_start_rad: float
    fov_horiz_span_rad: float
    fov_vert_start_rad: float
    fov_vert_span_rad: float
    row_azimuth_offsets_rad: Optional[Tensor] = None  # (n_rows,) float - optional for basic structured LiDAR
    spinning_frequency_hz: float = 0.0  # 0.0 for non-spinning LiDAR
    spinning_direction: Literal["cw", "ccw"] = "cw"  # Default to clockwise
    angles_to_columns_map: Optional[Tensor] = None  # Optional: (height, width) int - precomputed inverse map
    angles_to_columns_map_resolution_factor: int = 1

    def __post_init__(self):
        """Validate projection parameters."""
        # Basic shape checks for core angular tables
        if self.row_elevations_rad.ndim != 1 or self.row_elevations_rad.shape[0] != self.n_rows:
            raise ValueError(
                f"row_elevations_rad must be 1D with length n_rows={self.n_rows}, "
                f"got shape={tuple(self.row_elevations_rad.shape)}"
            )

        if self.column_azimuths_rad.ndim != 1 or self.column_azimuths_rad.shape[0] != self.n_columns:
            raise ValueError(
                f"column_azimuths_rad must be 1D with length n_columns={self.n_columns}, "
                f"got shape={tuple(self.column_azimuths_rad.shape)}"
            )

        if self.row_azimuth_offsets_rad is not None:
            if self.row_azimuth_offsets_rad.ndim != 1 or self.row_azimuth_offsets_rad.shape[0] != self.n_rows:
                raise ValueError(
                    f"row_azimuth_offsets_rad must be 1D with length n_rows={self.n_rows}, "
                    f"got shape={tuple(self.row_azimuth_offsets_rad.shape)}"
                )

        if self.angles_to_columns_map is not None:
            if self.angles_to_columns_map.dtype != torch.int32:
                raise ValueError(f"angles_to_columns_map must have dtype int32, got {self.angles_to_columns_map.dtype}")

        # Validate n_columns limit when no map is provided
        # The Slang kernel uses fallback linear search
        if self.angles_to_columns_map is None and self.n_columns > self._MAX_COLUMNS_WITHOUT_MAP:
            raise ValueError(
                f"n_columns={self.n_columns} exceeds {self._MAX_COLUMNS_WITHOUT_MAP} limit for linear search fallback. "
                f"Provide angles_to_columns_map or build it using build_angles_to_columns_map() for O(1) lookup, "
                f"or reduce n_columns to <= {self._MAX_COLUMNS_WITHOUT_MAP}."
            )

    def ensure_angles_map(self, resolution_factor: Optional[int] = None) -> None:
        """Build the angles-to-columns map if not already present.

        The map is built once and cached for subsequent uses. If map not provided,
        Call this method before inverse_project_spinning_lidar() to enable
        O(1) column lookup. Without the map, the kernel falls back to O(n)
        linear search per ray which is inefficient.

        Args:
            resolution_factor: Optional resolution multiplier. If not provided,
                uses angles_to_columns_map_resolution_factor if set > 1,
                otherwise uses _DEFAULT_MAP_RESOLUTION_FACTOR (4).
        """
        if self.angles_to_columns_map is not None:
            return  # Already built

        # Determine resolution factor
        if resolution_factor is None:
            resolution_factor = (
                self.angles_to_columns_map_resolution_factor
                if self.angles_to_columns_map_resolution_factor > 1
                else self._DEFAULT_MAP_RESOLUTION_FACTOR
            )

        # Build the map
        self.angles_to_columns_map = build_angles_to_columns_map(
            n_rows=self.n_rows,
            n_columns=self.n_columns,
            row_elevations_rad=self.row_elevations_rad,
            column_azimuths_rad=self.column_azimuths_rad,
            row_azimuth_offsets_rad=self.row_azimuth_offsets_rad,
            fov_vert_start_rad=self.fov_vert_start_rad,
            fov_vert_span_rad=self.fov_vert_span_rad,
            fov_horiz_start_rad=self.fov_horiz_start_rad,
            fov_horiz_span_rad=self.fov_horiz_span_rad,
            spinning_direction=self.spinning_direction,
            resolution_factor=resolution_factor,
        )
        self.angles_to_columns_map_resolution_factor = resolution_factor


# Type alias for any LiDAR projection
LidarProjection = RowOffsetStructuredSpinningLidarProjection


__all__ = [
    "RowOffsetStructuredSpinningLidarProjection",
    "LidarProjection",
    "build_angles_to_columns_map",
]
