# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging

from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np

# Pre-load dynamic torch dependencies, otherwise runtime-lookup will fail for torch-specific .so's
import torch

from libs.vren.interface import vren  # type: ignore
from ncore.data import ConcreteLidarModelParametersUnion, RowOffsetStructuredSpinningLidarModelParameters
from ncore.impl.data.util import relative_angle
from ncore.sensors import LidarModel, RowOffsetStructuredSpinningLidarModel


# LiDAR model parameter pack type definitions
VrenLidarModelParametersUnion: TypeAlias = vren.RowOffsetStructuredSpinningLidarModelParameters


@dataclass
class TilingReturns:
    n_bins_azimuth: int
    n_bins_elevation: int
    cdf_elevation: torch.Tensor
    cdf_dense_ray_mask: torch.Tensor
    tiles_to_elements_map: torch.Tensor
    tiles_pack_info: torch.Tensor

    @dataclass
    class ExtraDetails:
        edges_azimuth: torch.Tensor
        edges_elevation: torch.Tensor
        hist2d: torch.Tensor

    details: ExtraDetails

    def __post_init__(self):
        n = cast(float, self.cdf_elevation[-1].item())
        assert n == self.n_bins_elevation, "n_bins_elevation must be equal to cdf_elevation[-1]"


@dataclass
class PreprocessedLidarModelRaygenOnly:
    parameters: VrenLidarModelParametersUnion
    # device memory handlers
    _row_elevations_rad: torch.Tensor
    _column_azimuths_rad: torch.Tensor
    _row_azimuth_offsets_rad: torch.Tensor


@dataclass
class PreprocessedLidarModel(PreprocessedLidarModelRaygenOnly):
    parameters: VrenLidarModelParametersUnion
    # device memory handlers
    _tiling: TilingReturns
    _angles_to_columns_map: torch.Tensor

    def __post_init__(self):
        assert self._angles_to_columns_map.dtype == torch.int32, "angles_to_columns_map must be of dtype int32"
        assert self._angles_to_columns_map.is_cuda, "angles_to_columns_map must be on CUDA device"
        assert self._angles_to_columns_map.is_contiguous(), "angles_to_columns_map must be contiguous"


def parse_lidar_spinning_direction(spinning_direction: str) -> vren.SpinningDirection:
    if spinning_direction == "cw":
        return vren.SpinningDirection.CLOCK_WISE
    elif spinning_direction == "ccw":
        return vren.SpinningDirection.COUNTER_CLOCK_WISE
    else:
        raise ValueError(f"Invalid spinning direction: {spinning_direction}")


def preprocess_lidar_raygen_only(
    ncore_lidar_model_parameters: ConcreteLidarModelParametersUnion,
    device: torch.device,
) -> PreprocessedLidarModelRaygenOnly:
    """Provides lidar model parameter (generic over lidar model type) by either

        - initializing from corresponding NCore lidar model parameter type
        - forwarding argument directly if of 'vren.<LidarModelType>Parameters' type

    Errors out if non-supported lidar model parameter structure is provided"""

    match ncore_lidar_model_parameters:
        # row offset structured spinning lidar model
        case RowOffsetStructuredSpinningLidarModelParameters(
            spinning_frequency_hz=spinning_frequency_hz,
            spinning_direction=spinning_direction,
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=row_elevations_rad,
            column_azimuths_rad=column_azimuths_rad,
            row_azimuth_offsets_rad=row_azimuth_offsets_rad,
        ):
            vren_params = vren.RowOffsetStructuredSpinningLidarModelParameters()
            vren_params.spinning_frequency_hz = spinning_frequency_hz
            vren_params.spinning_direction = parse_lidar_spinning_direction(spinning_direction)
            vren_params.n_rows = n_rows
            vren_params.n_columns = n_columns

            fov_horiz = ncore_lidar_model_parameters.get_horizontal_fov()
            vren_params.fov_horiz_start_rad = fov_horiz.start_rad
            vren_params.fov_horiz_span_rad = fov_horiz.span_rad

            fov_vert = ncore_lidar_model_parameters.get_vertical_fov()
            vren_params.fov_vert_start_rad = fov_vert.start_rad
            vren_params.fov_vert_span_rad = fov_vert.span_rad

            th_row_elevations_rad = torch.tensor(row_elevations_rad, device=device, dtype=torch.float32)
            th_column_azimuths_rad = torch.tensor(column_azimuths_rad, device=device, dtype=torch.float32)
            th_row_azimuth_offsets_rad = torch.tensor(row_azimuth_offsets_rad, device=device, dtype=torch.float32)
            vren_params.set_row_column_angles_and_offsets(
                th_row_elevations_rad,
                th_column_azimuths_rad,
                th_row_azimuth_offsets_rad,
            )

            return PreprocessedLidarModelRaygenOnly(
                parameters=vren_params,
                _row_elevations_rad=th_row_elevations_rad,
                _column_azimuths_rad=th_column_azimuths_rad,
                _row_azimuth_offsets_rad=th_row_azimuth_offsets_rad,
            )

        # Error out if unsupported lidar model parameter structure is provided
        case _:
            raise ValueError(f"Unsupported lidar model parameters type {type(ncore_lidar_model_parameters)}")


def preprocess_lidar(
    ncore_lidar_model_parameters: ConcreteLidarModelParametersUnion,
    *,
    n_bins_elevation: int,
    max_pts_per_tile: int,  # Must match the renderer's configuration
    resolution_elevation: int,
    densification_factor_azimuth: int,
    device: torch.device,
) -> PreprocessedLidarModel:
    """Provides lidar model parameter (generic over lidar model type) by either

        - initializing from corresponding NCore lidar model parameter type
        - forwarding argument directly if of 'vren.<LidarModelType>Parameters' type

    Errors out if non-supported lidar model parameter structure is provided"""

    logging.getLogger(__name__).info(f"LiDAR number of bins in elevation = {n_bins_elevation}")
    logging.getLogger(__name__).info(f"LiDAR maximum points per tile = {max_pts_per_tile}")
    if max_pts_per_tile < 32:
        raise ValueError(f"max_pts_per_tile must be greater than 32 (the size of one GPU warp), got {max_pts_per_tile}")

    base_model = preprocess_lidar_raygen_only(ncore_lidar_model_parameters, device=device)
    vren_params = base_model.parameters

    match ncore_lidar_model_parameters:
        # row offset structured spinning lidar model
        case RowOffsetStructuredSpinningLidarModelParameters(
            spinning_frequency_hz=spinning_frequency_hz,
            spinning_direction=spinning_direction,
            n_rows=n_rows,
            n_columns=n_columns,
            row_elevations_rad=row_elevations_rad,
            column_azimuths_rad=column_azimuths_rad,
            row_azimuth_offsets_rad=row_azimuth_offsets_rad,
        ):
            ncore_model = RowOffsetStructuredSpinningLidarModel(
                ncore_lidar_model_parameters,
                angles_to_columns_map_init=True,
                device=device,
            )
            assert ncore_model.angles_to_columns_map is not None, (
                "Angles to columns map not initialized by the ncore model"
            )

            vren_params.set_angles_to_columns_map(
                angles_to_columns_map := ncore_model.angles_to_columns_map.int().contiguous().to(device),
                ncore_model.angles_to_columns_map_resolution_factor,
            )

            tiling = compute_tiling(
                vren_params,
                n_bins_elevation=n_bins_elevation,
                max_pts_per_tile=max_pts_per_tile,
                resolution_elevation=resolution_elevation,
                densification_factor_azimuth=densification_factor_azimuth,
                device=device,
            )
            vren_params.set_tiling_info(
                tiling.n_bins_azimuth,
                tiling.n_bins_elevation,
                densification_factor_azimuth,
                max_pts_per_tile,
                tiling.cdf_elevation,
                tiling.cdf_dense_ray_mask,
                tiling.tiles_to_elements_map,
                tiling.tiles_pack_info,
            )

            return PreprocessedLidarModel(
                parameters=vren_params,
                _tiling=tiling,
                _angles_to_columns_map=angles_to_columns_map,
                _row_elevations_rad=base_model._row_elevations_rad,
                _column_azimuths_rad=base_model._column_azimuths_rad,
                _row_azimuth_offsets_rad=base_model._row_azimuth_offsets_rad,
            )

        # Error out if unsupported lidar model parameter structure is provided
        case _:
            raise ValueError(f"Unsupported lidar model parameters type {type(ncore_lidar_model_parameters)}")


def angles_to_dense_ray_mask_cdf(
    parameters: vren.RowOffsetStructuredSpinningLidarModelParameters,
    angles: torch.Tensor,
    *,
    resolution_elevation: int,
    resolution_azimuth: int,
):
    # dense tile indices
    def uniform_quantization(x: torch.Tensor, n_bins: int):
        return (x * n_bins).int() % n_bins

    elevations_normalized = (
        relative_angle(
            parameters.fov_vert_start_rad,
            angles[:, 0],
            "cw",
        ).relative_angle_rad
        / parameters.fov_vert_span_rad
    )

    azimuths_normalized = (
        relative_angle(
            parameters.fov_horiz_start_rad,
            angles[:, 1],
            "cw" if parameters.spinning_direction == vren.SpinningDirection.CLOCK_WISE else "ccw",
        ).relative_angle_rad
        / parameters.fov_horiz_span_rad
    )

    elevations_indices = uniform_quantization(elevations_normalized, resolution_elevation)
    azimuths_indices = uniform_quantization(azimuths_normalized, resolution_azimuth)
    indices = elevations_indices + azimuths_indices * resolution_elevation

    masks = torch.zeros(resolution_azimuth * resolution_elevation, device=angles.device, dtype=torch.int32)
    masks[indices] = 1

    masks2d = masks.reshape(resolution_azimuth, resolution_elevation)
    masks2d_padded = torch.zeros(
        resolution_azimuth + 1, resolution_elevation + 1, device=angles.device, dtype=torch.int32
    )
    masks2d_padded[1:, 1:] = masks2d
    masks2d_integral = masks2d_padded.cumsum(dim=0).cumsum(dim=1)
    return masks2d_integral.int()


def angles_to_tile_indices(
    parameters: vren.RowOffsetStructuredSpinningLidarModelParameters,
    angles: torch.Tensor,
    *,
    n_bins_azimuth: int,
    n_bins_elevation: int,
    cdf_elevation: torch.Tensor,
):
    # the length of cdf_elevation is one plus the number of bins, so we need to subtract one here
    resolution = len(cdf_elevation) - 1

    elevations_normalized = (
        relative_angle(
            parameters.fov_vert_start_rad,
            angles[:, 0],
            "cw",
        ).relative_angle_rad
        / parameters.fov_vert_span_rad
        * resolution
    )

    azimuths_normalized = (
        relative_angle(
            parameters.fov_horiz_start_rad,
            angles[:, 1],
            "cw" if parameters.spinning_direction == vren.SpinningDirection.CLOCK_WISE else "ccw",
        ).relative_angle_rad
        / parameters.fov_horiz_span_rad
        * n_bins_azimuth
    )

    # compute the azimuth tile indices directly
    azimuths_indices = azimuths_normalized.int() % n_bins_azimuth

    # remap the elevations
    elevations_indices_cdf = torch.clamp(elevations_normalized, 0, resolution - 1).int()
    elevations_indices = cdf_elevation[elevations_indices_cdf].int()

    # NOTE: tile indices are row-major
    return elevations_indices + azimuths_indices * n_bins_elevation


def compute_histogram_equalization(
    parameters: vren.RowOffsetStructuredSpinningLidarModelParameters,
    *,
    n_bins_elevation: int,
    resolution_elevation: int,
    max_pts_per_tile: int,
    device: torch.device,
) -> tuple[int, torch.Tensor, TilingReturns.ExtraDetails]:
    # --------------------------------
    # generate a grid of elements of size (128 * 3600)
    # --------------------------------
    elements = (
        torch.stack(
            torch.meshgrid(
                torch.arange(parameters.n_rows, device=device, dtype=torch.int32),
                torch.arange(parameters.n_columns, device=device, dtype=torch.int32),
                indexing="ij",
            ),
            dim=-1,
        )
        .permute(1, 0, 2)
        .reshape(-1, 2)
    )

    angles = vren.elements_to_sensor_angles(parameters, elements).reshape(parameters.n_rows, parameters.n_columns, 2)

    angles_elevation = relative_angle(
        parameters.fov_vert_start_rad, angles[..., 0].contiguous(), "cw"
    ).relative_angle_rad
    angles_azimuth = relative_angle(
        parameters.fov_horiz_start_rad,
        angles[..., 1].contiguous(),
        "cw" if parameters.spinning_direction == vren.SpinningDirection.CLOCK_WISE else "ccw",
    ).relative_angle_rad

    ranges_azimuth = (0.0, parameters.fov_horiz_span_rad)
    ranges_elevation = (0.0, parameters.fov_vert_span_rad)
    assert torch.all(
        torch.logical_and(
            angles_elevation <= parameters.fov_vert_span_rad + 4 * torch.finfo(torch.float32).eps,
            angles_azimuth <= parameters.fov_horiz_span_rad + 4 * torch.finfo(torch.float32).eps,
        )
    ), (
        f"angles are out of bounds, angles_elevation {angles_elevation.max()} > {parameters.fov_vert_span_rad}, angles_azimuth {angles_azimuth.max()} > {parameters.fov_horiz_span_rad}"
    )

    # --------------------------------
    # Histogram Equalization
    # --------------------------------
    def compute_hist1d(data: torch.Tensor, bins: int | np.ndarray | torch.Tensor, range: tuple[float, float]):
        if isinstance(bins, torch.Tensor):
            bins = bins.cpu().numpy()
        hist, *others = np.histogram(data.cpu().numpy(), bins=bins)
        return torch.tensor(hist, device=data.device, dtype=torch.int32), *others

    # 1. compute the prefix sum of data
    hist, _ = compute_hist1d(angles_elevation, bins=resolution_elevation, range=ranges_elevation)

    tot = torch.sum(hist)
    cdf = torch.zeros((len(hist) + 1), device=device)
    cdf[1:] = torch.cumsum(hist, dim=0)
    cdf = cdf / tot * (n_bins_elevation)

    # 2. interpolate the new values
    edges_list = [0]
    curr = 1
    for i in range(len(cdf)):
        if cdf[i] >= curr:
            edges_list.append(i)
            curr += 1
    edges_list[-1] = len(cdf) - 1
    edges = torch.tensor(edges_list, device=device, dtype=torch.float32)

    # recompute the histograms
    edges_elevation = edges / resolution_elevation * parameters.fov_vert_span_rad
    hist_elevations, _ = compute_hist1d(angles_elevation, bins=edges_elevation, range=ranges_elevation)

    n_bins_azimuth = int(
        torch.ceil(hist_elevations.float().mean() / max_pts_per_tile)
    )  # estimate the number of bins for azimuths
    assert n_bins_azimuth > 0

    # now we need to find the smallest number of bins that satisfies the max_pts_per_tile
    angles_azimuth_np = angles_azimuth.flatten().cpu().numpy()
    angles_elevation_np = angles_elevation.flatten().cpu().numpy()
    edges_elevation_np = edges_elevation.cpu().numpy()

    def compute_hist2d(n_bins_azimuth):
        return np.histogram2d(
            angles_azimuth_np,
            angles_elevation_np,
            bins=[n_bins_azimuth, edges_elevation_np],
            range=[ranges_azimuth, ranges_elevation],
        )

    hist2d, edges_azimuth, _ = compute_hist2d(n_bins_azimuth)
    while hist2d.max() > max_pts_per_tile:
        n_bins_azimuth += 1
        hist2d, edges_azimuth, _ = compute_hist2d(n_bins_azimuth)
    hist2d = torch.tensor(hist2d, device=device, dtype=torch.int32)
    edges_azimuth = torch.tensor(edges_azimuth, device=device, dtype=torch.float32)

    assert isinstance(edges_azimuth, torch.Tensor)
    assert isinstance(edges_elevation, torch.Tensor)
    assert isinstance(hist2d, torch.Tensor)
    assert isinstance(cdf, torch.Tensor)

    return (
        n_bins_azimuth,
        cdf,
        TilingReturns.ExtraDetails(
            edges_azimuth=edges_azimuth,
            edges_elevation=edges_elevation,
            hist2d=hist2d,
        ),
    )


def compute_tiles_to_elements_map(
    parameters: vren.RowOffsetStructuredSpinningLidarModelParameters,
    *,
    n_bins_azimuth: int,
    densification_factor_azimuth: int,
    cdf_elevation: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the mapping from tiles to elements for the given lidar model and parameters"""

    # compute the number of bins for elevation
    n_bins_elevation = cast(float, cdf_elevation[-1].item())
    assert n_bins_elevation.is_integer(), "CDF elevation must be an integer"
    n_bins_elevation = int(n_bins_elevation)

    # create all element indices [relative to the static model]
    elements = torch.stack(
        torch.meshgrid(
            torch.arange(parameters.n_rows, dtype=torch.int32, device=device),
            torch.arange(parameters.n_columns, dtype=torch.int32, device=device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)
    angles = vren.elements_to_sensor_angles(parameters, elements)

    tile_indices = angles_to_tile_indices(
        parameters,
        angles,
        n_bins_azimuth=n_bins_azimuth,
        n_bins_elevation=n_bins_elevation,
        cdf_elevation=cdf_elevation,
    )
    tile_counts = torch.bincount(tile_indices, minlength=n_bins_azimuth * n_bins_elevation)
    tile_starts = torch.cumsum(tile_counts, dim=0) - tile_counts
    tiles_pack_info = torch.stack([tile_starts, tile_counts], dim=-1).int().cuda().contiguous()

    # sort the elements by tile indices
    sorted_element_indices = torch.argsort(tile_indices)
    sorted_elements = elements[sorted_element_indices]

    # compute the dense mask
    cdf_dense_ray_mask = angles_to_dense_ray_mask_cdf(
        parameters,
        angles,
        resolution_elevation=len(cdf_elevation) - 1,
        resolution_azimuth=n_bins_azimuth * densification_factor_azimuth,
    )

    return sorted_elements, tiles_pack_info, cdf_dense_ray_mask


def compute_tiling(
    parameters: vren.RowOffsetStructuredSpinningLidarModelParameters,
    *,
    n_bins_elevation: int,
    max_pts_per_tile: int,
    resolution_elevation: int,
    densification_factor_azimuth: int,
    device: torch.device,
) -> TilingReturns:
    n_bins_azimuth, cdf_elevation, details = compute_histogram_equalization(
        parameters,
        n_bins_elevation=n_bins_elevation,
        resolution_elevation=resolution_elevation,
        max_pts_per_tile=max_pts_per_tile,
        device=device,
    )
    tiles_to_elements_map, tiles_pack_info, cdf_dense_ray_mask = compute_tiles_to_elements_map(
        parameters,
        n_bins_azimuth=n_bins_azimuth,
        densification_factor_azimuth=densification_factor_azimuth,
        cdf_elevation=cdf_elevation,
        device=device,
    )

    return TilingReturns(
        n_bins_azimuth=n_bins_azimuth,
        n_bins_elevation=n_bins_elevation,
        cdf_elevation=cdf_elevation.int(),
        cdf_dense_ray_mask=cdf_dense_ray_mask,
        tiles_to_elements_map=tiles_to_elements_map,
        tiles_pack_info=tiles_pack_info,
        details=details,
    )


def valid_sensor_angles(
    parameters: vren.RowOffsetStructuredSpinningLidarModelParameters, sensor_angles: torch.Tensor
) -> torch.Tensor:
    """Checks if a sensor angles are valid / within the sensor's field of view"""

    relative_elevations_rad = relative_angle(
        parameters.fov_vert_start_rad, sensor_angles[:, 0], "cw"
    ).relative_angle_rad
    relative_azimuths_rad = relative_angle(
        parameters.fov_horiz_start_rad,
        sensor_angles[:, 1],
        "cw" if parameters.spinning_direction == vren.SpinningDirection.CLOCK_WISE else "ccw",
    ).relative_angle_rad

    return torch.logical_and(
        relative_elevations_rad <= parameters.fov_vert_span_rad + 4 * torch.finfo(torch.float32).eps,
        relative_azimuths_rad <= parameters.fov_horiz_span_rad + 4 * torch.finfo(torch.float32).eps,
    )


def elements_to_world_rays_shutter_pose(
    vren_lidar_model_parameters: VrenLidarModelParametersUnion,
    element: torch.Tensor,
    T_sensor_worlds: torch.Tensor,
    timestamps_us: torch.Tensor,
) -> LidarModel.WorldRaysReturn:
    """
    Computes the world rays for the given 2D indices of rays, sensor-to-world transforms, and timestamps considering the rolling shutter effect.

    Args:
        vren_lidar_model_parameters: the vren lidar model parameters
        element: (n, 2) - 2D indices of grid of rays in lidar model
        T_sensor_worlds: (2, 7) - the sensor-to-world transform
        timestamps_us: (2,) - the timestamps in microseconds

    Returns:
        LidarModel.WorldRaysReturn: the world rays and corresponding timestamps
    """
    assert element.dim() == 2 and element.shape[1] == 2 and element.dtype == torch.int32
    assert T_sensor_worlds.shape == (2, 7) and T_sensor_worlds.dtype == torch.float32
    assert timestamps_us.shape == (2,) and timestamps_us.dtype == torch.int64

    # Use tensor-based version that avoids GPU->CPU transfers
    # Ensure tensors are on GPU and contiguous (only transfer if needed)
    element_gpu = element.cuda().contiguous()
    device = element_gpu.device
    T_sensor_worlds_gpu = T_sensor_worlds.to(device=device, non_blocking=True).contiguous()
    timestamps_us_gpu = timestamps_us.to(device=device, non_blocking=True).contiguous()

    (
        world_rays,
        timestamps_us_out,
        T_sensor_worlds_out,
    ) = vren.elements_to_world_rays_shutter_pose(
        vren_lidar_model_parameters, T_sensor_worlds_gpu, timestamps_us_gpu, element_gpu
    )

    return LidarModel.WorldRaysReturn(
        world_rays=world_rays,
        timestamps_us=timestamps_us_out,
        T_sensor_worlds=T_sensor_worlds_out,
    )
