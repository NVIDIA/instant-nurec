# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import logging

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import torch

import nre.models.gaussians.collect as collect

from nre.models.gaussians.gaussians_model import (
    BaseGaussianModel,
    distributed_all_gather_gaussian_parameters,
)
from nre.utils.batch import RenderingData


log = logging.getLogger(__name__)


def _check_tensor_non_finite(tensor: torch.Tensor, name: str, source: str, timestamp_us: Optional[int] = None) -> None:
    """Check for NaN/Inf values in a tensor and log warnings with details."""
    if tensor is None:
        return

    non_finite_mask = ~torch.isfinite(tensor)
    non_finite_count = non_finite_mask.sum().item()

    if non_finite_count > 0:
        total_elements = tensor.numel()
        pct = (non_finite_count / total_elements) * 100
        indices = torch.nonzero(non_finite_mask, as_tuple=False)[:10].tolist()
        ts_str = f" at timestamp_us={timestamp_us}" if timestamp_us is not None else ""
        log.warning(
            "NaN/Inf detected in '%s' from %s%s: %s/%s values (%.2f%%) are non-finite. Example indices: %s",
            name,
            source,
            ts_str,
            non_finite_count,
            total_elements,
            pct,
            indices,
        )


def _check_params_non_finite(params: Dict[str, torch.Tensor], source: str, timestamp_us: Optional[int] = None) -> None:
    """Check model parameters dict for NaN/Inf values."""
    for key in ["positions", "rotations", "scales", "densities", "features_albedo", "features_specular"]:
        if key in params and params[key] is not None:
            _check_tensor_non_finite(params[key], key, source, timestamp_us)


@dataclass
class GaussianAttributes:
    """
    Export-friendly gaussian attributes. Rotations are wxyz and normalized. Arrays are numpy (CPU).

    Attributes:
        positions: [N, 3]
        rotations: [N, 4] (wxyz)
        scales: [N, 3]
        densities: [N, 1] or [N]
        extra_signal: Optional packed common signals
        camera_extra_signal: Optional packed camera signals
        lidar_extra_signal: Optional packed lidar signals
    """

    positions: np.ndarray
    rotations: np.ndarray
    scales: np.ndarray
    densities: np.ndarray
    albedo_coefficients: np.ndarray
    specular_coefficients: Optional[np.ndarray] = None
    extra_signal: Optional[np.ndarray] = None
    camera_extra_signal: Optional[np.ndarray] = None
    lidar_extra_signal: Optional[np.ndarray] = None

    def filter_non_finite_gaussians(self) -> Tuple["GaussianAttributes", np.ndarray]:
        """Filter out Gaussians with NaN/Inf (non-finite) values in any critical attribute.

        Creates a valid mask where a Gaussian is valid only if ALL of its critical
        attributes (positions, rotations, scales, densities, albedo) are finite.
        Applies this mask consistently to ALL attributes to prevent misalignment.

        Returns:
            Tuple of:
                - New GaussianAttributes with only valid Gaussians
                - Boolean mask array [N] indicating which original Gaussians were kept
        """
        num_gaussians = len(self.positions)

        # Build valid mask: Gaussian is valid if ALL critical attributes are finite (exclude NaN and Inf)
        valid_positions = np.all(np.isfinite(self.positions), axis=1)
        valid_rotations = np.all(np.isfinite(self.rotations), axis=1)
        valid_scales = np.all(np.isfinite(self.scales), axis=1)

        # Handle densities which can be [N, 1] or [N]
        densities_2d = self.densities.reshape(-1, 1) if self.densities.ndim == 1 else self.densities
        valid_densities = np.all(np.isfinite(densities_2d), axis=1)

        valid_albedo = np.all(np.isfinite(self.albedo_coefficients), axis=1)

        # Combine all masks - Gaussian must be valid in ALL attributes
        valid_mask = valid_positions & valid_rotations & valid_scales & valid_densities & valid_albedo

        # Also check optional specular coefficients if present
        if self.specular_coefficients is not None:
            valid_specular = np.all(np.isfinite(self.specular_coefficients), axis=1)
            valid_mask = valid_mask & valid_specular

        num_invalid = num_gaussians - np.sum(valid_mask)
        if num_invalid > 0:
            log.warning(
                f"Filtering {num_invalid}/{num_gaussians} Gaussians ({100 * num_invalid / num_gaussians:.4f}%) "
                f"with NaN/Inf values in critical attributes"
            )

            # Log breakdown by attribute for debugging
            nan_breakdown = {
                "positions": num_gaussians - np.sum(valid_positions),
                "rotations": num_gaussians - np.sum(valid_rotations),
                "scales": num_gaussians - np.sum(valid_scales),
                "densities": num_gaussians - np.sum(valid_densities),
                "albedo": num_gaussians - np.sum(valid_albedo),
            }
            if self.specular_coefficients is not None:
                nan_breakdown["specular"] = num_gaussians - np.sum(
                    np.all(np.isfinite(self.specular_coefficients), axis=1)
                )
            log.debug(f"NaN/Inf breakdown by attribute: {nan_breakdown}")

        # Apply mask to all attributes
        filtered = GaussianAttributes(
            positions=self.positions[valid_mask],
            rotations=self.rotations[valid_mask],
            scales=self.scales[valid_mask],
            densities=self.densities[valid_mask],
            albedo_coefficients=self.albedo_coefficients[valid_mask],
            specular_coefficients=self.specular_coefficients[valid_mask]
            if self.specular_coefficients is not None
            else None,
            extra_signal=self.extra_signal[valid_mask] if self.extra_signal is not None else None,
            camera_extra_signal=self.camera_extra_signal[valid_mask] if self.camera_extra_signal is not None else None,
            lidar_extra_signal=self.lidar_extra_signal[valid_mask] if self.lidar_extra_signal is not None else None,
        )

        return filtered, valid_mask


@dataclass
class ModelCapabilities:
    """
    Summarizes model features required by exporters.
    """

    has_spherical_harmonics: bool
    has_rigid_tracks: bool
    has_deformation: bool
    can_deform_positions: bool
    can_deform_rotations: bool
    can_deform_scales: bool
    has_temporal_appearance: bool
    sh_degree: Optional[int] = None
    is_planar_gaussian: bool = False  # True for 2D Gaussian surflets (flat disks)
    radiance_sph_O0: bool = False  # SH degree 0 coefficients directly encode radiance (no SH basis scaling)


class GaussianExportAccessor:
    """
    Collector-backed accessor to export gaussian data from a BaseGaussianModel.
    """

    def __init__(self, model: BaseGaussianModel):
        self.model = model
        self.layer_config = model.get_layer_config()

        # Get extra signal dimensions from model config
        extra_signal_dim = model.config.particle.extra_signal_dim
        camera_extra_signal_dim = model.config.particle.camera_extra_signal_dim
        lidar_extra_signal_dim = model.config.particle.lidar_extra_signal_dim

        # Get SH dimensions - all supported layer configs
        # Note: LayerConfigRigid and LayerConfigDeformable inherit from LayerConfigSH,
        # so this isinstance check handles all three config types
        if isinstance(self.layer_config, collect.LayerConfigSH):
            albedo_dim = int(model.get_albedo_sh_dim())  # type: ignore[operator]
            specular_dim = int(model.get_specular_sh_dim())  # type: ignore[operator]
        else:
            raise ValueError(f"Unsupported layer config type: {type(self.layer_config)}")

        layers_config = collect.LayersConfig(
            layers=[self.layer_config],
            extra_signal_dim=extra_signal_dim,
            camera_extra_signal_dim=camera_extra_signal_dim,
            lidar_extra_signal_dim=lidar_extra_signal_dim,
            albedo_dim=albedo_dim,
            specular_dim=specular_dim,
        )
        self.collector = collect.CreateGaussianParameterCollector(layers_config)

    #
    # Basic info
    #
    def get_num_gaussians(self) -> int:
        return self.model.get_num_gaussians()

    def get_capabilities(self) -> ModelCapabilities:
        has_sh = isinstance(self.layer_config, collect.LayerConfigSH)
        has_rigid = isinstance(self.layer_config, collect.LayerConfigRigid)
        has_deform = isinstance(self.layer_config, collect.LayerConfigDeformable)
        has_temporal_appearance = (
            isinstance(self.layer_config, collect.LayerConfigSH) and self.layer_config.embed_config is not None
        )

        # Get SH degree from config.particle.radiance_sph_degree
        sh_degree: Optional[int] = None
        if has_sh:
            sh_degree = self.model.config.particle.radiance_sph_degree

        # Check if model uses planar (2D) gaussians (surflets)
        is_planar = getattr(self.model.config.particle, "density_kernel_planar", False)

        # Check if SH degree 0 coefficients directly encode radiance (no SH basis scaling)
        radiance_sph_O0 = getattr(self.model.config.particle, "radiance_sph_O0", False)

        return ModelCapabilities(
            has_spherical_harmonics=has_sh,
            has_rigid_tracks=has_rigid,
            has_deformation=has_deform,
            can_deform_positions=has_deform,
            can_deform_rotations=has_deform,
            can_deform_scales=False,
            has_temporal_appearance=has_temporal_appearance,
            sh_degree=sh_degree,
            is_planar_gaussian=is_planar,
            radiance_sph_O0=radiance_sph_O0,
        )

    #
    # Attributes at a timestamp
    #
    @torch.no_grad()
    def get_attributes_at_timestamp(
        self, timestamp_us: int, preactivation: bool = False
    ) -> Tuple[GaussianAttributes, Optional[np.ndarray], Optional[np.ndarray]]:
        # gather model parameters
        params = self.model.get_parameters()
        params = distributed_all_gather_gaussian_parameters(params)

        # Check raw params for NaN before any processing
        _check_params_non_finite(params, "raw_params", timestamp_us)

        # create context
        rendering_data = RenderingData(
            rays=torch.empty((1, 1, 1, 6), dtype=torch.float32),
            sensor_model_parameters=[None],  # type: ignore[list-item]
            poses_tquat_startend=torch.empty((1, 2, 7), dtype=torch.float32),
            timestamps_startend_us=torch.empty((1, 2), dtype=torch.int64),
            rays_timestamps_us=None,
            _rays_footprints=None,
            timestamps_startend_us_cpu=torch.tensor([[timestamp_us, timestamp_us]], dtype=torch.int64, device="cpu"),
        )
        context = BaseGaussianModel.CollectionContext(
            rendering_data=rendering_data, is_training_batch=False, tracks_edit=None
        )

        # collect layer data
        layer_data = self.model.get_layer_data(context, params)
        has_rigid_tracks = isinstance(self.layer_config, collect.LayerConfigRigid)
        interpolated_track_poses: Optional[torch.Tensor] = None
        if has_rigid_tracks:
            # Cast to LayerDataRigid to access poses attribute
            rigid_layer_data = cast(collect.LayerDataRigid, layer_data)
            interpolated_track_poses = rigid_layer_data.poses.clone()
            rigid_layer_data.poses[:, :3] = torch.zeros_like(
                rigid_layer_data.poses[:, :3], device=rigid_layer_data.poses.device, dtype=torch.float32
            )
            rigid_layer_data.poses[:, 3:] = torch.tensor(
                [0, 0, 0, 1], device=rigid_layer_data.poses.device, dtype=torch.float32
            )

        layers_data = collect.LayersData(
            layers=[layer_data], frame_timestamp_us=BaseGaussianModel.get_frame_timestamp(rendering_data)
        )

        # process data
        collected = self.collector.collect(layers_data)

        # Check collected data for NaN after processing (includes track transforms and deformations)
        _check_tensor_non_finite(collected.positions, "positions", "collector", timestamp_us)
        _check_tensor_non_finite(collected.rotations, "rotations", "collector", timestamp_us)
        _check_tensor_non_finite(collected.scales, "scales", "collector", timestamp_us)
        _check_tensor_non_finite(collected.densities, "densities", "collector", timestamp_us)
        _check_tensor_non_finite(collected.features, "features", "collector", timestamp_us)

        # Per-track visibility: True when timestamp is within the track's pose range (no extrapolation)
        track_visibility_mask: Optional[np.ndarray] = None
        if has_rigid_tracks:
            track_visibility_mask = self._to_numpy(cast(collect.LayerDataRigid, layer_data).keep_mask)

        # Use collected positions/scales/densities as they include track transforms and deformations
        return (
            GaussianAttributes(
                positions=self._to_numpy(collected.positions),
                rotations=self._to_numpy(collected.rotations),
                scales=self._to_numpy(params["scales"] if preactivation else collected.scales),
                densities=self._to_numpy(params["densities"] if preactivation else collected.densities),
                extra_signal=self._to_numpy_optional(params.get("extra_signal")),
                camera_extra_signal=self._to_numpy_optional(params.get("camera_extra_signal")),
                lidar_extra_signal=self._to_numpy_optional(params.get("lidar_extra_signal")),
                albedo_coefficients=self._to_numpy(collected.features[:, :3]),
                specular_coefficients=self._to_numpy(collected.features[:, 3:]),
            ),
            self._to_numpy_optional(interpolated_track_poses),
            track_visibility_mask,
        )

    #
    # NaN filtering
    #
    @torch.no_grad()
    def get_valid_gaussian_mask(self, timestamp_us: int, preactivation: bool = False) -> np.ndarray:
        """Get a boolean mask indicating which Gaussians have valid (finite, i.e. non-NaN and non-Inf) values.

        This checks positions, rotations, scales, densities, and albedo coefficients.
        A Gaussian is valid only if ALL its critical attributes are finite.

        Args:
            timestamp_us: Timestamp to evaluate attributes at
            preactivation: Whether to use pre-activation values

        Returns:
            Boolean mask of shape [N] where True means the Gaussian is valid
        """
        attributes, _, _ = self.get_attributes_at_timestamp(timestamp_us, preactivation)
        _, valid_mask = attributes.filter_non_finite_gaussians()
        return valid_mask

    #
    # Tracks helpers
    #
    @torch.no_grad()
    def get_track_gaussian_mapping(self) -> Dict[str, torch.Tensor]:
        """
        Map track_id (string) to gaussian indices.
        """
        if not isinstance(self.layer_config, collect.LayerConfigRigid):
            return {}
        mapping: Dict[str, torch.Tensor] = {}
        cuboid_tracks: Any = self.model.cuboid_tracks  # type: ignore[attr-defined]
        gids: torch.Tensor = self.model.gaussian_cuboid_ids  # type: ignore[attr-defined, assignment]
        for track_idx, track_id in enumerate(cuboid_tracks.tracks_id):
            mask = gids == int(track_idx)
            if torch.count_nonzero(mask).item() == 0:
                continue
            mapping[track_id] = torch.where(mask)[0]
        return mapping

    @torch.no_grad()
    def get_track_time_range(self, track_index: int) -> Tuple[int, int]:
        tracks = self.model.cuboid_tracks  # type: ignore[attr-defined]
        tracks_data = tracks.tracks_data  # type: ignore[union-attr]
        start_idx, n_poses = tracks_data.tracks_packinfo[track_index].tolist()  # type: ignore[union-attr, index]
        ts = tracks_data.tracks_timestamps_us[start_idx : start_idx + n_poses]  # type: ignore[union-attr, index]
        return int(ts.min().item()), int(ts.max().item())

    @torch.no_grad()
    def get_track_time_range_safe(self, track_index: int) -> Optional[Tuple[int, int]]:
        """Return (min_us, max_us) for the track, or None if the track has no poses."""
        if not isinstance(self.layer_config, collect.LayerConfigRigid):
            return None
        tracks = self.model.cuboid_tracks  # type: ignore[attr-defined]
        tracks_data = tracks.tracks_data  # type: ignore[union-attr]
        start_idx, n_poses = tracks_data.tracks_packinfo[track_index].tolist()  # type: ignore[union-attr, index]
        if n_poses == 0:
            return None
        ts = tracks_data.tracks_timestamps_us[start_idx : start_idx + n_poses]  # type: ignore[union-attr, index]
        return int(ts.min().item()), int(ts.max().item())

    @torch.no_grad()
    def get_track_timestamps(self, track_index: int) -> List[int]:
        tracks = self.model.cuboid_tracks  # type: ignore[attr-defined]
        tracks_data = tracks.tracks_data  # type: ignore[union-attr]
        start_idx, n_poses = tracks_data.tracks_packinfo[track_index].tolist()  # type: ignore[union-attr, index]
        ts = tracks_data.tracks_timestamps_us[start_idx : start_idx + n_poses]  # type: ignore[union-attr, index]
        return ts.cpu().numpy().astype(int).tolist()

    @torch.no_grad()
    def get_cuboid_tracks(self):
        if not isinstance(self.layer_config, collect.LayerConfigRigid):
            raise RuntimeError("Model does not provide cuboid tracks")
        return self.model.cuboid_tracks  # type: ignore[attr-defined]

    @staticmethod
    def _to_numpy(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    @staticmethod
    def _to_numpy_optional(t: Optional[torch.Tensor]) -> Optional[np.ndarray]:
        return None if t is None else t.detach().cpu().numpy()
