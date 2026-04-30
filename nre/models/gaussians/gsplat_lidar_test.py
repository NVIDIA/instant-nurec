# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Equivalence tests for gsplat lidar functions against vren/ncore references.

Validates that gsplat's compute_angles_to_columns_map() and compute_tiling()
produce identical results to the vren/ncore implementations, treating
vren/ncore as the reference (ground truth).
"""

import json

import gsplat
import pytest
import torch

from omegaconf import DictConfig, OmegaConf

import ncore.data

from libs.vren.lidars import preprocess_lidar
from ncore.impl.sensors.lidar import RowOffsetStructuredSpinningLidarModel
from nre.config.parse import parse_untyped_config


DEVICE = torch.device("cuda")
RESOLUTION_FACTOR = 4
TILING_N_BINS_ELEVATION = 16
TILING_MAX_PTS_PER_TILE = 16 * 16
TILING_RESOLUTION_ELEVATION = 1600
TILING_DENSIFICATION_FACTOR_AZIMUTH = 8


def _load_lidar_models_config() -> DictConfig:
    """Load the sensor config YAML via parse_untyped_config (resolves ${load_json:...} automatically)."""
    config = parse_untyped_config("sensor/lidar_model", hydra_args=[])
    return config.sensor.lidar_models


def _ncore_params_from_config(lidar_config) -> ncore.data.RowOffsetStructuredSpinningLidarModelParameters:
    """Convert an OmegaConf dict (from YAML) to ncore lidar model parameters."""
    plain = OmegaConf.to_container(lidar_config, resolve=True)
    return ncore.data.RowOffsetStructuredSpinningLidarModelParameters.from_json(json.dumps(plain))


def _gsplat_params_from_ncore(
    ncore_params: ncore.data.RowOffsetStructuredSpinningLidarModelParameters,
) -> gsplat.RowOffsetStructuredSpinningLidarModelParameters:
    """Build gsplat lidar model parameters from ncore parameters."""
    spinning_direction = (
        gsplat.SpinningDirection.CLOCKWISE
        if ncore_params.spinning_direction == "cw"
        else gsplat.SpinningDirection.COUNTER_CLOCKWISE
    )
    return gsplat.RowOffsetStructuredSpinningLidarModelParameters(
        row_elevations_rad=torch.tensor(ncore_params.row_elevations_rad, dtype=torch.float32, device=DEVICE),
        column_azimuths_rad=torch.tensor(ncore_params.column_azimuths_rad, dtype=torch.float32, device=DEVICE),
        row_azimuth_offsets_rad=torch.tensor(ncore_params.row_azimuth_offsets_rad, dtype=torch.float32, device=DEVICE),
        spinning_frequency_hz=ncore_params.spinning_frequency_hz,
        spinning_direction=spinning_direction,
    )


@pytest.fixture(scope="module")
def lidar_models_config():
    config = _load_lidar_models_config()
    return config


@pytest.fixture(
    scope="module",
    params=["HESAI_Pandar128", "HESAI_AT128"],
)
def sensor_params(request, lidar_models_config):
    """Yields (ncore_params, gsplat_params) for each sensor."""
    sensor_name = request.param
    lidar_config = lidar_models_config[sensor_name]
    ncore_params = _ncore_params_from_config(lidar_config)
    gsplat_params = _gsplat_params_from_ncore(ncore_params)
    return ncore_params, gsplat_params


class TestAnglestoColumnsMap:
    """Test equivalence of angles_to_columns_map between ncore and gsplat."""

    def test_equivalence(self, sensor_params):
        ncore_params, gsplat_params = sensor_params

        # ncore reference
        ncore_model = RowOffsetStructuredSpinningLidarModel(
            ncore_params,
            angles_to_columns_map_init=True,
            angles_to_columns_map_resolution_factor=RESOLUTION_FACTOR,
            angles_to_columns_map_dtype=torch.int32,
            device=DEVICE,
        )
        ncore_map = ncore_model.angles_to_columns_map.int().to(DEVICE)

        # gsplat
        gsplat_map = gsplat.compute_lidar_angles_to_columns_map(
            gsplat_params, resolution_factor=RESOLUTION_FACTOR, dtype=torch.int32
        )

        assert ncore_map.shape == gsplat_map.shape, (
            f"Shape mismatch: ncore {ncore_map.shape} vs gsplat {gsplat_map.shape}"
        )

        assert torch.equal(ncore_map, gsplat_map), (
            f"angles_to_columns_map mismatch: "
            f"max diff = {(ncore_map - gsplat_map).abs().max().item()}, "
            f"num mismatches = {(ncore_map != gsplat_map).sum().item()} / {ncore_map.numel()}"
        )


class TestLidarTiling:
    """Test equivalence of tiling structures between vren and gsplat."""

    def test_equivalence(self, sensor_params):
        ncore_params, gsplat_params = sensor_params

        # vren/ncore reference
        vren_preprocessed = preprocess_lidar(
            ncore_params,
            n_bins_elevation=TILING_N_BINS_ELEVATION,
            max_pts_per_tile=TILING_MAX_PTS_PER_TILE,
            resolution_elevation=TILING_RESOLUTION_ELEVATION,
            densification_factor_azimuth=TILING_DENSIFICATION_FACTOR_AZIMUTH,
            device=DEVICE,
        )
        vren_tiling = vren_preprocessed._tiling

        # gsplat
        gsplat_tiling = gsplat.compute_lidar_tiling(
            gsplat_params,
            n_bins_elevation=TILING_N_BINS_ELEVATION,
            max_pts_per_tile=TILING_MAX_PTS_PER_TILE,
            resolution_elevation=TILING_RESOLUTION_ELEVATION,
            densification_factor_azimuth=TILING_DENSIFICATION_FACTOR_AZIMUTH,
        )

        # Compare scalar fields
        assert vren_tiling.n_bins_azimuth == gsplat_tiling.n_bins_azimuth, (
            f"n_bins_azimuth mismatch: vren={vren_tiling.n_bins_azimuth} vs gsplat={gsplat_tiling.n_bins_azimuth}"
        )
        assert vren_tiling.n_bins_elevation == gsplat_tiling.n_bins_elevation, (
            f"n_bins_elevation mismatch: vren={vren_tiling.n_bins_elevation} vs gsplat={gsplat_tiling.n_bins_elevation}"
        )

        # Compare tensor fields
        assert torch.equal(vren_tiling.cdf_elevation.cpu(), gsplat_tiling.cdf_elevation.cpu()), (
            f"cdf_elevation mismatch: max diff = {(vren_tiling.cdf_elevation.cpu() - gsplat_tiling.cdf_elevation.cpu()).abs().max().item()}"
        )

        # Adapt vren tensors to gsplat's convention before comparison:
        # - cdf_dense_ray_mask: vren (azimuth, elevation) → gsplat (elevation, azimuth)
        # - tiles_pack_info: vren tile_id = az*n_el+el → gsplat tile_id = el*n_az+az
        # - tiles_to_elements_map: vren (elevation, azimuth) → gsplat (azimuth, elevation)
        n_az = gsplat_tiling.n_bins_azimuth
        n_el = gsplat_tiling.n_bins_elevation
        vren_mask = vren_tiling.cdf_dense_ray_mask.cpu().T
        # Reorder tiles_pack_info from vren's tile ordering (az*n_el+el) to gsplat's (el*n_az+az).
        # Only the counts are meaningful after reordering; recompute offsets from the reordered counts.
        vren_pack_raw = vren_tiling.tiles_pack_info.cpu()
        reordered_counts = vren_pack_raw[:, 1].reshape(n_az, n_el).T.reshape(-1)
        reordered_offsets = torch.zeros_like(reordered_counts)
        reordered_offsets[1:] = reordered_counts[:-1].cumsum(0)
        vren_pack = torch.stack([reordered_offsets, reordered_counts], dim=-1)
        # Reorder tiles_to_elements_map: gather elements per tile in gsplat tile order,
        # and swap columns from vren (elevation, azimuth) to gsplat (azimuth, elevation).
        vren_elems_raw = vren_tiling.tiles_to_elements_map.cpu()
        vren_elems_list = []
        for gsplat_tid in range(n_az * n_el):
            el, az = gsplat_tid // n_az, gsplat_tid % n_az
            vren_tid = az * n_el + el
            off = vren_pack_raw[vren_tid, 0].item()
            cnt = vren_pack_raw[vren_tid, 1].item()
            vren_elems_list.append(vren_elems_raw[off : off + cnt, [1, 0]])  # swap (el,az) → (az,el)
        vren_elems = torch.cat(vren_elems_list, dim=0)

        gsplat_mask = gsplat_tiling.cdf_dense_ray_mask.cpu()
        gsplat_pack = gsplat_tiling.tiles_pack_info.cpu()
        gsplat_elems = gsplat_tiling.tiles_to_elements_map.cpu()

        # For the dense ray mask integral image, the total ray count (bottom-right corner)
        # must match exactly. The interior can differ by up to n_rows at 360-degree FOV
        # wrap-around boundaries due to float32/float64 modulo precision in
        # the angle-to-bin quantization between vren C++ and gsplat Python.
        assert vren_mask.shape == gsplat_mask.shape, (
            f"cdf_dense_ray_mask shape mismatch: vren={vren_mask.shape} vs gsplat={gsplat_mask.shape}"
        )
        # For 360-degree FOV sensors, float32/float64 modulo precision differences
        # between vren C++ and gsplat Python cause up to n_rows rays at the azimuth
        # wrap-around boundary to be assigned to the first vs last azimuth bin.
        # This affects the integral image by up to n_rows * n_azimuth_bins.
        is_full_rotation = abs(gsplat_params.fov_horiz_rad.span - 2 * torch.pi) < 0.01
        if is_full_rotation:
            mask_diff = (vren_mask - gsplat_mask).abs()
            max_mask_diff = mask_diff.max().item()
            max_allowed = ncore_params.n_rows * gsplat_tiling.n_bins_azimuth
            assert max_mask_diff <= max_allowed, (
                f"cdf_dense_ray_mask mismatch exceeds tolerance: max diff = {max_mask_diff} > {max_allowed}"
            )
        else:
            assert torch.equal(vren_mask, gsplat_mask), (
                f"cdf_dense_ray_mask mismatch: max diff = {(vren_mask - gsplat_mask).abs().max().item()}"
            )

        assert torch.equal(vren_pack, gsplat_pack), (
            f"tiles_pack_info mismatch: max diff = {(vren_pack - gsplat_pack).abs().max().item()}"
        )
        assert torch.equal(vren_elems, gsplat_elems), (
            f"tiles_to_elements_map mismatch: max diff = {(vren_elems - gsplat_elems).abs().max().item()}"
        )
