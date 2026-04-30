# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Python bindings for LiDAR kernel operations.

This module provides thin wrappers around Slang LiDAR kernels via slangtorch.
These bindings accept exposed Slang data structures directly.

All functions are differentiable and support PyTorch autograd via torch.autograd.Function
wrappers that call the Slang-generated backward kernels (*_bwd_diff).
"""

import logging

from enum import IntEnum

import torch

from torch import Tensor

import libs.sensors.liblidar_slang_cc as lidar_slang  # type: ignore # pycena: skip

from libs.sensors.kernels.common import DynamicPose
from libs.sensors.kernels.lidars.parameters import (
    LidarProjection,
    RowOffsetStructuredSpinningLidarProjection,
)
from libs.slang_utils.utils import div_up


logger = logging.getLogger(__name__)


class SpinningDirection(IntEnum):
    """LiDAR spinning direction.

    Matches the SpinningDirection enum in interface.slang.
    Using IntEnum allows direct use as kernel parameters.
    """

    CLOCKWISE = 0
    COUNTERCLOCKWISE = 1

    @classmethod
    def from_string(cls, direction: str) -> "SpinningDirection":
        """Convert string representation to SpinningDirection.

        Args:
            direction: Either "cw" (clockwise) or "ccw" (counterclockwise)

        Returns:
            SpinningDirection enum value

        Raises:
            ValueError: If an unsupported direction string is provided.
        """
        value = direction.lower()
        if value == "ccw":
            return cls.COUNTERCLOCKWISE
        if value == "cw":
            return cls.CLOCKWISE
        raise ValueError(f"Invalid spinning direction {direction!r}, expected 'cw' or 'ccw'.")


# Slang module constants
_THREADS_PER_BLOCK = 256


# ============================================================================
# Helper Functions
# ============================================================================


def _to_dev(t: Tensor, device: torch.device, dtype: torch.dtype, allow_device_transfer: bool = False) -> Tensor:
    """Move tensor to device/dtype and make contiguous.

    Args:
        t: Input tensor
        device: Target device
        dtype: Target dtype
        allow_device_transfer: If False, raises error when device/dtype transfer is needed.
            If True, allows implicit transfer.

    Raises:
        RuntimeError: If tensor requires device/dtype transfer and allow_device_transfer is False.
    """
    needs_transfer = t.device != device or t.dtype != dtype
    if needs_transfer:
        if not allow_device_transfer:
            raise RuntimeError(
                f"Tensor on {t.device} (dtype={t.dtype}) requires transfer to {device} (dtype={dtype}). "
                f"To allow implicit device transfer, set allow_device_transfer=True. "
                f"For best performance, ensure all tensors are on the target device before calling this function."
            )
        return t.to(device=device, dtype=dtype).contiguous()
    return t.contiguous()


# ============================================================================
# Kernel Wrapper Functions
# ============================================================================


def _get_spinning_direction(projection: RowOffsetStructuredSpinningLidarProjection) -> SpinningDirection:
    """Convert spinning direction string to SpinningDirection enum."""
    return SpinningDirection.from_string(projection.spinning_direction)


def _get_lidar_projection_params(
    projection: RowOffsetStructuredSpinningLidarProjection,
    device,
    dtype,
    allow_device_transfer: bool = False,
):
    """Extract LiDAR projection parameters as contiguous tensors.

    Args:
        projection: LiDAR projection parameters
        device: Target device
        dtype: Target dtype
        allow_device_transfer: If False, raises error when device/dtype transfer is needed.
            If True, allows implicit transfer.

    Returns:
        Dictionary with all projection parameters for kernel calls.
    """
    row_elevations = _to_dev(projection.row_elevations_rad, device, dtype, allow_device_transfer)
    column_azimuths = _to_dev(projection.column_azimuths_rad, device, dtype, allow_device_transfer)

    # Row offsets - use zeros if not present
    if projection.row_azimuth_offsets_rad is not None:
        row_offsets = _to_dev(projection.row_azimuth_offsets_rad, device, dtype, allow_device_transfer)
        has_row_offsets = 1
    else:
        row_offsets = torch.zeros(projection.n_rows, device=device, dtype=dtype)
        has_row_offsets = 0

    # Angles to columns map - 2D map if present, otherwise empty
    # Map shape: (resolution_factor * n_rows, resolution_factor * n_columns)
    if projection.angles_to_columns_map is not None:
        # Keep dtype explicit (int32) and honor allow_device_transfer contract
        angles_to_columns_map = _to_dev(
            projection.angles_to_columns_map,
            device,
            torch.int32,
            allow_device_transfer,
        )
        has_angles_map = 1
        # 2D map dimensions
        map_height = angles_to_columns_map.shape[0]
        map_width = angles_to_columns_map.shape[1] if angles_to_columns_map.ndim > 1 else 1
        # Flatten for Slang (2D access via flat index)
        angles_to_columns_map = angles_to_columns_map.flatten()
    else:
        angles_to_columns_map = torch.zeros(1, device=device, dtype=torch.int32)
        has_angles_map = 0
        map_height = 0
        map_width = 0

    return {
        "n_rows": projection.n_rows,
        "n_columns": projection.n_columns,
        "row_elevations": row_elevations,
        "column_azimuths": column_azimuths,
        "row_offsets": row_offsets,
        "has_row_offsets": has_row_offsets,
        "spinning_frequency_hz": projection.spinning_frequency_hz,
        "spinning_direction": _get_spinning_direction(projection),
        "fov_horiz_start_rad": projection.fov_horiz_start_rad,
        "fov_horiz_span_rad": projection.fov_horiz_span_rad,
        "fov_vert_start_rad": projection.fov_vert_start_rad,
        "fov_vert_span_rad": projection.fov_vert_span_rad,
        "angles_to_columns_map": angles_to_columns_map,
        "has_angles_map": has_angles_map,
        "angles_to_columns_map_resolution_factor": projection.angles_to_columns_map_resolution_factor,
        "angles_to_columns_map_height": map_height,
        "angles_to_columns_map_width": map_width,
    }


# ============================================================================
# Autograd Function Wrappers for Differentiable LiDAR Operations
# ============================================================================


class GenerateSpinningLidarRaysFunction(torch.autograd.Function):
    """Differentiable spinning LiDAR ray generation with intrinsic gradients.

    This autograd function enables gradient flow through LiDAR intrinsic parameters
    (row elevations, column azimuths, row azimuth offsets) and pose parameters.
    """

    @staticmethod
    def forward(
        ctx,
        elements: Tensor | None,
        # LiDAR projection params (differentiable)
        row_elevations: Tensor,
        column_azimuths: Tensor,
        row_offsets: Tensor,
        # Pose params (differentiable)
        control_translations: Tensor,
        control_rotations: Tensor,
        control_times: Tensor,
        # Non-differentiable config
        n_rows: int,
        n_columns: int,
        has_row_offsets: int,
        spinning_frequency_hz: float,
        spinning_direction: int,
        fov_horiz_start_rad: float,
        fov_horiz_span_rad: float,
        fov_vert_start_rad: float,
        fov_vert_span_rad: float,
        angles_to_columns_map: Tensor,
        has_angles_map: int,
        angles_to_columns_map_resolution_factor: int,
        angles_to_columns_map_height: int,
        angles_to_columns_map_width: int,
        control_count: int,
        start_timestamp_us: int,
        end_timestamp_us: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        N = elements.shape[0] if elements is not None else n_rows * n_columns
        device = row_elevations.device
        dtype = torch.float32

        does_generate_elements = elements is None
        world_rays = torch.empty((N, 6), device=device, dtype=dtype)
        if elements is None:
            elements_float = torch.empty((1, 2), device=device, dtype=torch.float32)
        else:
            elements_float = elements.contiguous().to(dtype=dtype)
        timestamps_us = torch.empty(N, device=device, dtype=torch.int64)
        poses_translation = torch.empty((N, 3), device=device, dtype=dtype)
        poses_rotation = torch.empty((N, 4), device=device, dtype=dtype)

        blocks = div_up(N, _THREADS_PER_BLOCK)
        if blocks > 0:
            lidar_slang.generate_spinning_lidar_rays_row_offset(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                n_rows,
                n_columns,
                (row_elevations, (row_elevations,)),
                (column_azimuths, (column_azimuths,)),
                (row_offsets, (row_offsets,)),
                has_row_offsets,
                spinning_frequency_hz,
                spinning_direction,
                fov_horiz_start_rad,
                fov_horiz_span_rad,
                fov_vert_start_rad,
                fov_vert_span_rad,
                angles_to_columns_map,
                has_angles_map,
                angles_to_columns_map_resolution_factor,
                angles_to_columns_map_height,
                angles_to_columns_map_width,
                (control_translations, (control_translations,)),
                (control_rotations, (control_rotations,)),
                control_times,
                control_count,
                start_timestamp_us,
                end_timestamp_us,
                (elements_float, (elements_float,)),
                does_generate_elements,
                (world_rays, (world_rays,)),
                timestamps_us,
                poses_translation,
                poses_rotation,
            )

        # Save tensors for backward
        ctx.save_for_backward(
            elements_float,
            world_rays,
            timestamps_us,
            poses_translation,
            poses_rotation,
            row_elevations,
            column_azimuths,
            row_offsets,
            control_translations,
            control_rotations,
            control_times,
            angles_to_columns_map,
        )
        ctx.n_rows = n_rows
        ctx.n_columns = n_columns
        ctx.has_row_offsets = has_row_offsets
        ctx.spinning_frequency_hz = spinning_frequency_hz
        ctx.spinning_direction = spinning_direction
        ctx.fov_horiz_start_rad = fov_horiz_start_rad
        ctx.fov_horiz_span_rad = fov_horiz_span_rad
        ctx.fov_vert_start_rad = fov_vert_start_rad
        ctx.fov_vert_span_rad = fov_vert_span_rad
        ctx.has_angles_map = has_angles_map
        ctx.angles_to_columns_map_resolution_factor = angles_to_columns_map_resolution_factor
        ctx.angles_to_columns_map_height = angles_to_columns_map_height
        ctx.angles_to_columns_map_width = angles_to_columns_map_width
        ctx.control_count = control_count
        ctx.start_timestamp_us = int(start_timestamp_us)
        ctx.end_timestamp_us = int(end_timestamp_us)
        ctx.N = N
        ctx.does_generate_elements = does_generate_elements

        return world_rays, timestamps_us, poses_translation, poses_rotation

    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore[override]
        # timestamps_us, poses_translation, poses_rotation are non-differentiable outputs
        grad_world_rays, _grad_timestamps_us, _grad_poses_trans, _grad_poses_rot = grad_outputs
        (
            elements_float,
            world_rays,
            timestamps_us,
            poses_translation,
            poses_rotation,
            row_elevations,
            column_azimuths,
            row_offsets,
            control_translations,
            control_rotations,
            control_times,
            angles_to_columns_map,
        ) = ctx.saved_tensors

        grad_world_rays = grad_world_rays.contiguous()
        N = ctx.N

        # Initialize gradient tensors
        if ctx.does_generate_elements:
            grad_elements = torch.empty((1, 2), device=elements_float.device, dtype=elements_float.dtype)
        else:
            grad_elements = torch.zeros_like(elements_float)
        grad_row_elevations = torch.zeros_like(row_elevations)
        grad_column_azimuths = torch.zeros_like(column_azimuths)
        grad_row_offsets = torch.zeros_like(row_offsets)
        grad_control_translations = torch.zeros_like(control_translations)
        grad_control_rotations = torch.zeros_like(control_rotations)

        blocks = div_up(N, _THREADS_PER_BLOCK)
        if blocks > 0:
            lidar_slang.generate_spinning_lidar_rays_row_offset_bwd_diff(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                ctx.n_rows,
                ctx.n_columns,
                (row_elevations, (grad_row_elevations,)),
                (column_azimuths, (grad_column_azimuths,)),
                (row_offsets, (grad_row_offsets,)),
                ctx.has_row_offsets,
                ctx.spinning_frequency_hz,
                ctx.spinning_direction,
                ctx.fov_horiz_start_rad,
                ctx.fov_horiz_span_rad,
                ctx.fov_vert_start_rad,
                ctx.fov_vert_span_rad,
                angles_to_columns_map,
                ctx.has_angles_map,
                ctx.angles_to_columns_map_resolution_factor,
                ctx.angles_to_columns_map_height,
                ctx.angles_to_columns_map_width,
                (control_translations, (grad_control_translations,)),
                (control_rotations, (grad_control_rotations,)),
                control_times,
                ctx.control_count,
                ctx.start_timestamp_us,
                ctx.end_timestamp_us,
                (elements_float, (grad_elements,)),
                ctx.does_generate_elements,
                (world_rays, (grad_world_rays,)),
                timestamps_us,
                poses_translation,
                poses_rotation,
            )

        return (
            grad_elements if not ctx.does_generate_elements else None,
            grad_row_elevations,
            grad_column_azimuths,
            grad_row_offsets,
            grad_control_translations,
            grad_control_rotations,
            None,  # control_times
            None,  # n_rows
            None,  # n_columns
            None,  # has_row_offsets
            None,  # spinning_frequency_hz
            None,  # spinning_direction
            None,  # fov_horiz_start_rad
            None,  # fov_horiz_span_rad
            None,  # fov_vert_start_rad
            None,  # fov_vert_span_rad
            None,  # angles_to_columns_map
            None,  # has_angles_map
            None,  # angles_to_columns_map_resolution_factor
            None,  # angles_to_columns_map_height
            None,  # angles_to_columns_map_width
            None,  # control_count
            None,  # start_timestamp_us
            None,  # end_timestamp_us
        )


class ElementsToSensorAnglesFunction(torch.autograd.Function):
    """Differentiable element to sensor angle conversion with intrinsic gradients."""

    @staticmethod
    def forward(
        ctx,
        elements: Tensor,
        # LiDAR projection params (differentiable)
        row_elevations: Tensor,
        column_azimuths: Tensor,
        row_offsets: Tensor,
        # Non-differentiable config
        n_rows: int,
        n_columns: int,
        has_row_offsets: int,
        spinning_frequency_hz: float,
        spinning_direction: int,
        fov_horiz_start_rad: float,
        fov_horiz_span_rad: float,
        fov_vert_start_rad: float,
        fov_vert_span_rad: float,
        angles_to_columns_map: Tensor,
        has_angles_map: int,
        angles_to_columns_map_resolution_factor: int,
        angles_to_columns_map_height: int,
        angles_to_columns_map_width: int,
    ) -> tuple[Tensor, Tensor]:
        N = elements.shape[0]
        device = elements.device
        dtype = torch.float32

        elements_float = elements.contiguous().to(dtype=dtype)
        sensor_angles = torch.empty((N, 2), device=device, dtype=dtype)
        valid_flags = torch.empty((N,), device=device, dtype=torch.bool)

        blocks = div_up(N, _THREADS_PER_BLOCK)
        if blocks > 0:
            lidar_slang.elements_to_sensor_angles_row_offset(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                n_rows,
                n_columns,
                (row_elevations, (row_elevations,)),
                (column_azimuths, (column_azimuths,)),
                (row_offsets, (row_offsets,)),
                has_row_offsets,
                spinning_frequency_hz,
                spinning_direction,
                fov_horiz_start_rad,
                fov_horiz_span_rad,
                fov_vert_start_rad,
                fov_vert_span_rad,
                angles_to_columns_map,
                has_angles_map,
                angles_to_columns_map_resolution_factor,
                angles_to_columns_map_height,
                angles_to_columns_map_width,
                (elements_float, (elements_float,)),
                (sensor_angles, (sensor_angles,)),
                valid_flags,
            )

        # Save tensors for backward
        ctx.save_for_backward(
            elements_float,
            sensor_angles,
            row_elevations,
            column_azimuths,
            row_offsets,
            angles_to_columns_map,
        )
        ctx.n_rows = n_rows
        ctx.n_columns = n_columns
        ctx.has_row_offsets = has_row_offsets
        ctx.spinning_frequency_hz = spinning_frequency_hz
        ctx.spinning_direction = spinning_direction
        ctx.fov_horiz_start_rad = fov_horiz_start_rad
        ctx.fov_horiz_span_rad = fov_horiz_span_rad
        ctx.fov_vert_start_rad = fov_vert_start_rad
        ctx.fov_vert_span_rad = fov_vert_span_rad
        ctx.has_angles_map = has_angles_map
        ctx.angles_to_columns_map_resolution_factor = angles_to_columns_map_resolution_factor
        ctx.angles_to_columns_map_height = angles_to_columns_map_height
        ctx.angles_to_columns_map_width = angles_to_columns_map_width
        ctx.N = N

        return sensor_angles, valid_flags

    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore[override]
        # valid_flags is not differentiable
        grad_sensor_angles, _grad_valid_flags = grad_outputs
        (
            elements_float,
            sensor_angles,
            row_elevations,
            column_azimuths,
            row_offsets,
            angles_to_columns_map,
        ) = ctx.saved_tensors

        grad_sensor_angles = grad_sensor_angles.contiguous()
        N = ctx.N

        # Initialize gradient tensors
        grad_elements = torch.zeros_like(elements_float)
        grad_row_elevations = torch.zeros_like(row_elevations)
        grad_column_azimuths = torch.zeros_like(column_azimuths)
        grad_row_offsets = torch.zeros_like(row_offsets)

        # Create dummy valid_flags for backward kernel (not used but required by kernel signature)
        valid_flags = torch.empty((N,), device=elements_float.device, dtype=torch.bool)

        blocks = div_up(N, _THREADS_PER_BLOCK)
        if blocks > 0:
            lidar_slang.elements_to_sensor_angles_row_offset_bwd_diff(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                ctx.n_rows,
                ctx.n_columns,
                (row_elevations, (grad_row_elevations,)),
                (column_azimuths, (grad_column_azimuths,)),
                (row_offsets, (grad_row_offsets,)),
                ctx.has_row_offsets,
                ctx.spinning_frequency_hz,
                ctx.spinning_direction,
                ctx.fov_horiz_start_rad,
                ctx.fov_horiz_span_rad,
                ctx.fov_vert_start_rad,
                ctx.fov_vert_span_rad,
                angles_to_columns_map,
                ctx.has_angles_map,
                ctx.angles_to_columns_map_resolution_factor,
                ctx.angles_to_columns_map_height,
                ctx.angles_to_columns_map_width,
                (elements_float, (grad_elements,)),
                (sensor_angles, (grad_sensor_angles,)),
                valid_flags,
            )

        return (
            None,  # elements (int indices, not differentiable)
            grad_row_elevations,
            grad_column_azimuths,
            grad_row_offsets,
            None,  # n_rows
            None,  # n_columns
            None,  # has_row_offsets
            None,  # spinning_frequency_hz
            None,  # spinning_direction
            None,  # fov_horiz_start_rad
            None,  # fov_horiz_span_rad
            None,  # fov_vert_start_rad
            None,  # fov_vert_span_rad
            None,  # angles_to_columns_map
            None,  # has_angles_map
            None,  # angles_to_columns_map_resolution_factor
            None,  # angles_to_columns_map_height
            None,  # angles_to_columns_map_width
        )


class InverseProjectSpinningLidarFunction(torch.autograd.Function):
    """Differentiable inverse projection with intrinsic gradients."""

    @staticmethod
    def forward(
        ctx,
        world_points: Tensor,
        # LiDAR projection params (differentiable)
        row_elevations: Tensor,
        column_azimuths: Tensor,
        row_offsets: Tensor,
        # Pose params (differentiable)
        control_translations: Tensor,
        control_rotations: Tensor,
        control_times: Tensor,
        # Non-differentiable config
        n_rows: int,
        n_columns: int,
        has_row_offsets: int,
        spinning_frequency_hz: float,
        spinning_direction: int,
        fov_horiz_start_rad: float,
        fov_horiz_span_rad: float,
        fov_vert_start_rad: float,
        fov_vert_span_rad: float,
        angles_to_columns_map: Tensor,
        has_angles_map: int,
        angles_to_columns_map_resolution_factor: int,
        angles_to_columns_map_height: int,
        angles_to_columns_map_width: int,
        control_count: int,
        max_iterations: int,
        stop_mean_relative_time_error: float,
        stop_delta_mean_relative_time_error: float,
        start_timestamp_us: int,
        end_timestamp_us: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        N = world_points.shape[0]
        device = world_points.device
        dtype = world_points.dtype

        world_points = world_points.contiguous()
        sensor_angles = torch.empty((N, 2), device=device, dtype=dtype)
        valid_flags = torch.empty(N, device=device, dtype=torch.bool)
        timestamps_us = torch.empty(N, device=device, dtype=torch.int64)
        poses_translation = torch.empty((N, 3), device=device, dtype=dtype)
        poses_rotation = torch.empty((N, 4), device=device, dtype=dtype)

        blocks = div_up(N, _THREADS_PER_BLOCK)
        if blocks > 0:
            lidar_slang.inverse_project_spinning_lidar_row_offset(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                n_rows,
                n_columns,
                (row_elevations, (row_elevations,)),
                (column_azimuths, (column_azimuths,)),
                (row_offsets, (row_offsets,)),
                has_row_offsets,
                spinning_frequency_hz,
                spinning_direction,
                fov_horiz_start_rad,
                fov_horiz_span_rad,
                fov_vert_start_rad,
                fov_vert_span_rad,
                angles_to_columns_map,
                has_angles_map,
                angles_to_columns_map_resolution_factor,
                angles_to_columns_map_height,
                angles_to_columns_map_width,
                (control_translations, (control_translations,)),
                (control_rotations, (control_rotations,)),
                control_times,
                control_count,
                max_iterations,
                stop_mean_relative_time_error,
                stop_delta_mean_relative_time_error,
                start_timestamp_us,
                end_timestamp_us,
                (world_points, (world_points,)),
                (sensor_angles, (sensor_angles,)),
                valid_flags,
                timestamps_us,
                poses_translation,
                poses_rotation,
            )

        # Save tensors for backward
        ctx.save_for_backward(
            world_points,
            sensor_angles,
            valid_flags,
            timestamps_us,
            poses_translation,
            poses_rotation,
            row_elevations,
            column_azimuths,
            row_offsets,
            control_translations,
            control_rotations,
            control_times,
            angles_to_columns_map,
        )
        ctx.n_rows = n_rows
        ctx.n_columns = n_columns
        ctx.has_row_offsets = has_row_offsets
        ctx.spinning_frequency_hz = spinning_frequency_hz
        ctx.spinning_direction = spinning_direction
        ctx.fov_horiz_start_rad = fov_horiz_start_rad
        ctx.fov_horiz_span_rad = fov_horiz_span_rad
        ctx.fov_vert_start_rad = fov_vert_start_rad
        ctx.fov_vert_span_rad = fov_vert_span_rad
        ctx.has_angles_map = has_angles_map
        ctx.angles_to_columns_map_resolution_factor = angles_to_columns_map_resolution_factor
        ctx.angles_to_columns_map_height = angles_to_columns_map_height
        ctx.angles_to_columns_map_width = angles_to_columns_map_width
        ctx.control_count = control_count
        ctx.max_iterations = max_iterations
        ctx.stop_mean_relative_time_error = stop_mean_relative_time_error
        ctx.stop_delta_mean_relative_time_error = stop_delta_mean_relative_time_error
        ctx.start_timestamp_us = int(start_timestamp_us)
        ctx.end_timestamp_us = int(end_timestamp_us)
        ctx.N = N

        return sensor_angles, valid_flags, timestamps_us, poses_translation, poses_rotation

    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore[override]
        # valid_flags, timestamps_us, poses_translation, poses_rotation are non-differentiable outputs
        grad_sensor_angles, _grad_valid_flags, _grad_timestamps_us, _grad_poses_trans, _grad_poses_rot = grad_outputs
        (
            world_points,
            sensor_angles,
            valid_flags,
            timestamps_us,
            poses_translation,
            poses_rotation,
            row_elevations,
            column_azimuths,
            row_offsets,
            control_translations,
            control_rotations,
            control_times,
            angles_to_columns_map,
        ) = ctx.saved_tensors

        grad_sensor_angles = grad_sensor_angles.contiguous()
        N = ctx.N

        # Initialize gradient tensors
        grad_world_points = torch.zeros_like(world_points)
        grad_row_elevations = torch.zeros_like(row_elevations)
        grad_column_azimuths = torch.zeros_like(column_azimuths)
        grad_row_offsets = torch.zeros_like(row_offsets)
        grad_control_translations = torch.zeros_like(control_translations)
        grad_control_rotations = torch.zeros_like(control_rotations)

        blocks = div_up(N, _THREADS_PER_BLOCK)
        if blocks > 0:
            lidar_slang.inverse_project_spinning_lidar_row_offset_bwd_diff(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                ctx.n_rows,
                ctx.n_columns,
                (row_elevations, (grad_row_elevations,)),
                (column_azimuths, (grad_column_azimuths,)),
                (row_offsets, (grad_row_offsets,)),
                ctx.has_row_offsets,
                ctx.spinning_frequency_hz,
                ctx.spinning_direction,
                ctx.fov_horiz_start_rad,
                ctx.fov_horiz_span_rad,
                ctx.fov_vert_start_rad,
                ctx.fov_vert_span_rad,
                angles_to_columns_map,
                ctx.has_angles_map,
                ctx.angles_to_columns_map_resolution_factor,
                ctx.angles_to_columns_map_height,
                ctx.angles_to_columns_map_width,
                (control_translations, (grad_control_translations,)),
                (control_rotations, (grad_control_rotations,)),
                control_times,
                ctx.control_count,
                ctx.max_iterations,
                ctx.stop_mean_relative_time_error,
                ctx.stop_delta_mean_relative_time_error,
                ctx.start_timestamp_us,
                ctx.end_timestamp_us,
                (world_points, (grad_world_points,)),
                (sensor_angles, (grad_sensor_angles,)),
                valid_flags,
                timestamps_us,
                poses_translation,
                poses_rotation,
            )

        return (
            grad_world_points,
            grad_row_elevations,
            grad_column_azimuths,
            grad_row_offsets,
            grad_control_translations,
            grad_control_rotations,
            None,  # control_times
            None,  # n_rows
            None,  # n_columns
            None,  # has_row_offsets
            None,  # spinning_frequency_hz
            None,  # spinning_direction
            None,  # fov_horiz_start_rad
            None,  # fov_horiz_span_rad
            None,  # fov_vert_start_rad
            None,  # fov_vert_span_rad
            None,  # angles_to_columns_map
            None,  # has_angles_map
            None,  # angles_to_columns_map_resolution_factor
            None,  # angles_to_columns_map_height
            None,  # angles_to_columns_map_width
            None,  # control_count
            None,  # max_iterations
            None,  # stop_mean_relative_time_error
            None,  # stop_delta_mean_relative_time_error
            None,  # start_timestamp_us
            None,  # end_timestamp_us
        )


class SensorRaysToSensorAnglesFunction(torch.autograd.Function):
    """Differentiable sensor ray to angle conversion."""

    @staticmethod
    def forward(ctx, sensor_rays: Tensor) -> Tensor:
        N = sensor_rays.shape[0]
        device = sensor_rays.device
        dtype = sensor_rays.dtype

        sensor_rays = sensor_rays.contiguous()
        sensor_angles = torch.empty((N, 2), device=device, dtype=dtype)

        blocks = div_up(N, _THREADS_PER_BLOCK)
        if blocks > 0:
            lidar_slang.sensor_rays_to_sensor_angles_kernel(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (sensor_rays, (sensor_rays,)),
                (sensor_angles, (sensor_angles,)),
            )

        ctx.save_for_backward(sensor_rays, sensor_angles)
        ctx.N = N

        return sensor_angles

    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore[override]
        (grad_sensor_angles,) = grad_outputs
        sensor_rays, sensor_angles = ctx.saved_tensors
        grad_sensor_angles = grad_sensor_angles.contiguous()
        N = ctx.N

        grad_sensor_rays = torch.zeros_like(sensor_rays)

        blocks = div_up(N, _THREADS_PER_BLOCK)
        if blocks > 0:
            lidar_slang.sensor_rays_to_sensor_angles_kernel_bwd_diff(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (sensor_rays, (grad_sensor_rays,)),
                (sensor_angles, (grad_sensor_angles,)),
            )

        return grad_sensor_rays


class SensorAnglesToSensorRaysFunction(torch.autograd.Function):
    """Differentiable sensor angle to ray conversion."""

    @staticmethod
    def forward(ctx, sensor_angles: Tensor) -> Tensor:
        N = sensor_angles.shape[0]
        device = sensor_angles.device
        dtype = sensor_angles.dtype

        sensor_angles = sensor_angles.contiguous()
        sensor_rays = torch.empty((N, 3), device=device, dtype=dtype)

        blocks = div_up(N, _THREADS_PER_BLOCK)
        if blocks > 0:
            lidar_slang.sensor_angles_to_sensor_rays_kernel(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (sensor_angles, (sensor_angles,)),
                (sensor_rays, (sensor_rays,)),
            )

        ctx.save_for_backward(sensor_angles, sensor_rays)
        ctx.N = N

        return sensor_rays

    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore[override]
        (grad_sensor_rays,) = grad_outputs
        sensor_angles, sensor_rays = ctx.saved_tensors
        grad_sensor_rays = grad_sensor_rays.contiguous()
        N = ctx.N

        grad_sensor_angles = torch.zeros_like(sensor_angles)

        blocks = div_up(N, _THREADS_PER_BLOCK)
        if blocks > 0:
            lidar_slang.sensor_angles_to_sensor_rays_kernel_bwd_diff(
                (_THREADS_PER_BLOCK, 1, 1),
                (blocks, 1, 1),
                (sensor_angles, (grad_sensor_angles,)),
                (sensor_rays, (grad_sensor_rays,)),
            )

        return grad_sensor_angles


def generate_spinning_lidar_rays(
    projection: LidarProjection,  # Layer 0 projection parameters
    elements: Tensor | None,  # (N, 2) [row, col] int or None if elements are generated
    dynamic_pose: DynamicPose,  # Time-varying dynamic pose (sensor motion)
    start_timestamp_us: int | None = None,
    end_timestamp_us: int | None = None,
    allow_device_transfer: bool = False,
    return_timestamps: bool = False,
    return_poses: bool = False,
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
    """
    Generate world rays for spinning LiDAR with rolling shutter.

    Uses ILidarProjection interface for projection operations - works with any projection type.
    Interpolates pose based on azimuth angle and spinning direction.
    This function is fully differentiable with respect to LiDAR intrinsic parameters
    (row elevations, column azimuths, row offsets) and pose parameters.

    Note: This function does not have a return_valid_flags parameter because element-based
    ray generation always produces valid rays (elements are assumed to be valid indices).

    Args:
        projection: Layer 0 exposed projection parameters (supports gradient flow)
        elements: (N, 2) element indices [row_idx, col_idx] int32 or None if elements are generated between (0, 0) and (n_rows-1, n_columns-1)
        dynamic_pose: Time-varying dynamic pose from geometry module (supports gradient flow)
        start_timestamp_us: Start timestamp in microseconds for absolute timestamp computation
        end_timestamp_us: End timestamp in microseconds for absolute timestamp computation
        allow_device_transfer: If False (default), raises error when projection/pose
            tensors require device or dtype transfer. If True, allows implicit transfer.
        return_timestamps: If True, compute and return absolute timestamps
        return_poses: If True, compute and return per-ray poses

    Returns:
        world_rays: (N, 6) [origin.xyz, direction.xyz] in world frame
        timestamps_us: (N,) absolute timestamps in microseconds (int64), or None if return_timestamps=False
        poses_translation: (N, 3) per-ray interpolated pose translations, or None if return_poses=False
        poses_rotation: (N, 4) per-ray interpolated pose rotations (quaternions), or None if return_poses=False
    """
    if return_timestamps:
        assert start_timestamp_us is not None, "start_timestamp_us must be provided when return_timestamps=True"
        assert end_timestamp_us is not None, "end_timestamp_us must be provided when return_timestamps=True"

    N = elements.shape[0] if elements is not None else projection.n_rows * projection.n_columns
    device = dynamic_pose.start_pose.translation.device
    dtype = torch.float32

    if N == 0:
        return (
            torch.empty((0, 6), device=device, dtype=dtype),
            torch.empty(0, device=device, dtype=torch.int64) if return_timestamps else None,
            torch.empty((0, 3), device=device, dtype=dtype) if return_poses else None,
            torch.empty((0, 4), device=device, dtype=dtype) if return_poses else None,
        )

    # Extract trajectory data on the same device/dtype as the outputs
    trajectory = dynamic_pose.to_trajectory()
    control_count = trajectory.control_count

    control_translations = _to_dev(
        torch.stack([pose.translation for pose in trajectory.control_poses]).contiguous(),
        device,
        dtype,
        allow_device_transfer,
    )
    control_rotations = _to_dev(
        torch.stack([pose.rotation for pose in trajectory.control_poses]).contiguous(),
        device,
        dtype,
        allow_device_transfer,
    )
    control_times = _to_dev(trajectory.control_times.contiguous(), device, dtype, allow_device_transfer)

    # Extract projection parameters
    proj_params = _get_lidar_projection_params(projection, device, dtype, allow_device_transfer)

    # Convert None timestamps to 0
    start_ts = 0 if start_timestamp_us is None else start_timestamp_us
    end_ts = 0 if end_timestamp_us is None else end_timestamp_us

    # Call autograd function
    world_rays, timestamps_us, poses_translation, poses_rotation = GenerateSpinningLidarRaysFunction.apply(
        elements,
        proj_params["row_elevations"],
        proj_params["column_azimuths"],
        proj_params["row_offsets"],
        control_translations,
        control_rotations,
        control_times,
        proj_params["n_rows"],
        proj_params["n_columns"],
        proj_params["has_row_offsets"],
        proj_params["spinning_frequency_hz"],
        int(proj_params["spinning_direction"]),
        proj_params["fov_horiz_start_rad"],
        proj_params["fov_horiz_span_rad"],
        proj_params["fov_vert_start_rad"],
        proj_params["fov_vert_span_rad"],
        proj_params["angles_to_columns_map"],
        proj_params["has_angles_map"],
        proj_params["angles_to_columns_map_resolution_factor"],
        proj_params["angles_to_columns_map_height"],
        proj_params["angles_to_columns_map_width"],
        control_count,
        start_ts,
        end_ts,
    )

    return (
        world_rays,
        timestamps_us if return_timestamps else None,
        poses_translation if return_poses else None,
        poses_rotation if return_poses else None,
    )


def elements_to_sensor_angles(
    projection: LidarProjection,  # Layer 0 projection parameters
    elements: Tensor,  # (N, 2) [row, col] int
    allow_device_transfer: bool = False,
    return_valid_flags: bool = False,
) -> tuple[Tensor, Tensor | None]:
    """
    Convert element indices to sensor angles (no pose transformation).

    Uses ILidarProjection interface - works with any projection type.
    Simple lookup operation from element grid to angle space.
    This function is fully differentiable with respect to LiDAR intrinsic parameters
    (row elevations, column azimuths, row offsets).

    Args:
        projection: Layer 0 exposed projection parameters (supports gradient flow)
        elements: (N, 2) element indices [row_idx, col_idx] int32
        allow_device_transfer: If False (default), raises error when projection
            tensors require device or dtype transfer. If True, allows implicit transfer.
        return_valid_flags: If True, also return validity flags indicating whether
            each element index was within bounds.

    Returns:
        sensor_angles: (N, 2) [elevation_rad, azimuth_rad]
        valid_flags: (N,) bool or None - None if return_valid_flags=False.
            False if element indices are out of bounds.
    """
    N = elements.shape[0]
    device = elements.device
    dtype = torch.float32

    if N == 0:
        return (
            torch.empty((0, 2), device=device, dtype=dtype),
            torch.empty((0,), device=device, dtype=torch.bool) if return_valid_flags else None,
        )

    # Extract projection parameters
    proj_params = _get_lidar_projection_params(projection, device, dtype, allow_device_transfer)

    # Call autograd function
    sensor_angles, valid_flags = ElementsToSensorAnglesFunction.apply(
        elements,
        proj_params["row_elevations"],
        proj_params["column_azimuths"],
        proj_params["row_offsets"],
        proj_params["n_rows"],
        proj_params["n_columns"],
        proj_params["has_row_offsets"],
        proj_params["spinning_frequency_hz"],
        int(proj_params["spinning_direction"]),
        proj_params["fov_horiz_start_rad"],
        proj_params["fov_horiz_span_rad"],
        proj_params["fov_vert_start_rad"],
        proj_params["fov_vert_span_rad"],
        proj_params["angles_to_columns_map"],
        proj_params["has_angles_map"],
        proj_params["angles_to_columns_map_resolution_factor"],
        proj_params["angles_to_columns_map_height"],
        proj_params["angles_to_columns_map_width"],
    )

    return sensor_angles, valid_flags if return_valid_flags else None


def inverse_project_spinning_lidar(
    projection: LidarProjection,  # Layer 0 projection parameters
    world_points: Tensor,  # (N, 3)
    dynamic_pose: DynamicPose,  # Time-varying dynamic pose (sensor motion)
    start_timestamp_us: int | None = None,
    end_timestamp_us: int | None = None,
    max_iterations: int = 10,
    stop_mean_relative_time_error: float = 0.0001,
    stop_delta_mean_relative_time_error: float = 0.000001,
    allow_device_transfer: bool = False,
    lazy_init_angles_map: bool = False,
    return_valid_flags: bool = False,
    return_timestamps: bool = False,
    return_poses: bool = False,
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    """
    Inverse project world points to sensor angles with rolling shutter.

    Uses ILidarProjection interface for projection operations - works with any projection type.
    Uses iterative refinement to handle rolling shutter projection.
    This function is fully differentiable with respect to world points and pose parameters.

    Args:
        projection: Layer 0 exposed projection parameters
        world_points: (N, 3) world coordinates (supports gradient flow)
        dynamic_pose: Time-varying dynamic pose from geometry module (supports gradient flow)
        start_timestamp_us: Start timestamp in microseconds for absolute timestamp computation
        end_timestamp_us: End timestamp in microseconds for absolute timestamp computation
        max_iterations: Maximum iterations for convergence
        stop_mean_relative_time_error: Stopping criterion for mean error
        stop_delta_mean_relative_time_error: Stopping criterion for error change
        allow_device_transfer: If False (default), raises error when pose
            tensors require device or dtype transfer. If True, allows implicit transfer.
        lazy_init_angles_map: If True, lazily builds the angles-to-columns map if not
            already present. If False (default), logs a warning if the map is not set
            and the kernel will fall back to O(n) linear search per ray (inefficient).
            It is recommended to build the map ahead of time and set it or set lazy_init_angles_map to True.
        return_valid_flags: If True, compute and return validity mask
        return_timestamps: If True, compute and return absolute timestamps
        return_poses: If True, compute and return per-point poses

    Returns:
        sensor_angles: (N, 2) [elevation_rad, azimuth_rad]
        valid_flags: (N,) bool - whether points project into sensor FOV, or None if return_valid_flags=False
        timestamps_us: (N,) absolute timestamps in microseconds (int64), or None if return_timestamps=False
        poses_translation: (N, 3) per-point interpolated pose translations, or None if return_poses=False
        poses_rotation: (N, 4) per-point interpolated pose rotations (quaternions), or None if return_poses=False
    """
    if return_timestamps:
        assert start_timestamp_us is not None, "start_timestamp_us must be provided when return_timestamps=True"
        assert end_timestamp_us is not None, "end_timestamp_us must be provided when return_timestamps=True"

    # Handle angles-to-columns map for efficient column lookup
    if lazy_init_angles_map:
        projection.ensure_angles_map()
    elif projection.angles_to_columns_map is None:
        logger.warning(
            "angles_to_columns_map not set for inverse_project_spinning_lidar. "
            "Kernel will fall back to O(n) linear search per ray, which is inefficient. "
            "It is recommended to provide the map or build it using "
            "projection.ensure_angles_map() or before calling this function."
        )

    N = world_points.shape[0]
    device = world_points.device
    dtype = world_points.dtype

    if N == 0:
        return (
            torch.empty((0, 2), device=device, dtype=dtype),
            torch.empty(0, device=device, dtype=torch.bool) if return_valid_flags else None,
            torch.empty(0, device=device, dtype=torch.int64) if return_timestamps else None,
            torch.empty((0, 3), device=device, dtype=dtype) if return_poses else None,
            torch.empty((0, 4), device=device, dtype=dtype) if return_poses else None,
        )

    # Extract trajectory data on the same device/dtype as the inputs/outputs
    trajectory = dynamic_pose.to_trajectory()
    control_count = trajectory.control_count

    control_translations = _to_dev(
        torch.stack([pose.translation for pose in trajectory.control_poses]).contiguous(),
        device,
        dtype,
        allow_device_transfer,
    )
    control_rotations = _to_dev(
        torch.stack([pose.rotation for pose in trajectory.control_poses]).contiguous(),
        device,
        dtype,
        allow_device_transfer,
    )
    control_times = _to_dev(trajectory.control_times.contiguous(), device, dtype, allow_device_transfer)

    # Extract projection parameters
    proj_params = _get_lidar_projection_params(projection, device, dtype, allow_device_transfer)

    # Convert None timestamps to 0
    start_ts = 0 if start_timestamp_us is None else start_timestamp_us
    end_ts = 0 if end_timestamp_us is None else end_timestamp_us

    # Call autograd function
    sensor_angles, valid_flags, timestamps_us, poses_translation, poses_rotation = (
        InverseProjectSpinningLidarFunction.apply(
            world_points.contiguous(),
            proj_params["row_elevations"],
            proj_params["column_azimuths"],
            proj_params["row_offsets"],
            control_translations,
            control_rotations,
            control_times,
            proj_params["n_rows"],
            proj_params["n_columns"],
            proj_params["has_row_offsets"],
            proj_params["spinning_frequency_hz"],
            int(proj_params["spinning_direction"]),
            proj_params["fov_horiz_start_rad"],
            proj_params["fov_horiz_span_rad"],
            proj_params["fov_vert_start_rad"],
            proj_params["fov_vert_span_rad"],
            proj_params["angles_to_columns_map"],
            proj_params["has_angles_map"],
            proj_params["angles_to_columns_map_resolution_factor"],
            proj_params["angles_to_columns_map_height"],
            proj_params["angles_to_columns_map_width"],
            control_count,
            max_iterations,
            stop_mean_relative_time_error,
            stop_delta_mean_relative_time_error,
            start_ts,
            end_ts,
        )
    )

    return (
        sensor_angles,
        valid_flags if return_valid_flags else None,
        timestamps_us if return_timestamps else None,
        poses_translation if return_poses else None,
        poses_rotation if return_poses else None,
    )


def sensor_rays_to_sensor_angles(
    projection: LidarProjection,  # Layer 0 projection parameters
    sensor_rays: Tensor,  # (N, 3) normalized direction
) -> Tensor:
    """
    Convert sensor rays to sensor angles (elevation, azimuth).

    Simple coordinate conversion from Cartesian to spherical.
    No pose transformation.
    This function is fully differentiable with respect to input rays.

    Args:
        projection: Layer 0 exposed projection parameters
        sensor_rays: (N, 3) normalized direction vectors in sensor frame (supports gradient flow)

    Returns:
        sensor_angles: (N, 2) [elevation_rad, azimuth_rad]
    """
    N = sensor_rays.shape[0]
    device = sensor_rays.device
    dtype = sensor_rays.dtype

    if N == 0:
        return torch.empty((0, 2), device=device, dtype=dtype)

    # Call autograd function
    sensor_angles = SensorRaysToSensorAnglesFunction.apply(sensor_rays.contiguous())

    return sensor_angles


def sensor_angles_to_sensor_rays(
    projection: LidarProjection,  # Layer 0 projection parameters
    sensor_angles: Tensor,  # (N, 2) [elevation, azimuth]
) -> Tensor:
    """
    Convert sensor angles to sensor rays.

    Simple coordinate conversion from spherical to Cartesian.
    No pose transformation.
    This function is fully differentiable with respect to input angles.

    Args:
        projection: Layer 0 exposed projection parameters
        sensor_angles: (N, 2) [elevation_rad, azimuth_rad] (supports gradient flow)

    Returns:
        sensor_rays: (N, 3) normalized direction vectors in sensor frame
    """
    N = sensor_angles.shape[0]
    device = sensor_angles.device
    dtype = sensor_angles.dtype

    if N == 0:
        return torch.empty((0, 3), device=device, dtype=dtype)

    # Call autograd function
    sensor_rays = SensorAnglesToSensorRaysFunction.apply(sensor_angles.contiguous())

    return sensor_rays


__all__ = [
    "generate_spinning_lidar_rays",
    "elements_to_sensor_angles",
    "inverse_project_spinning_lidar",
    "sensor_rays_to_sensor_angles",
    "sensor_angles_to_sensor_rays",
]
