# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Comprehensive tests for GaussianExportAccessor.

Covers:
- Basic info (num_gaussians, device, capabilities)
- Attributes at timestamp
- Spherical harmonics coefficients
- Temporal/animated albedo
- Rigid track helpers
- Deformable model detection
- Extra signal access
"""

import unittest

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch

import nre.models.gaussians.collect as collect

from nre.datasets.tracks import CuboidTracks, TracksData
from nre.utils.io.export.gaussian_export_accessor import (
    GaussianAttributes,
    GaussianExportAccessor,
    ModelCapabilities,
)


# =============================================================================
# Mock Classes
# =============================================================================


class MockCollectorResult:
    """Mock for collector.collect() result."""

    def __init__(
        self,
        n_gaussians: int,
        device: torch.device,
        features_dim: int = 48,
        extra_signal_dim: int = 0,
        camera_extra_signal_dim: int = 0,
        lidar_extra_signal_dim: int = 0,
    ):
        self.positions = torch.randn(n_gaussians, 3, device=device)
        self.rotations = torch.randn(n_gaussians, 4, device=device)
        self.scales = torch.randn(n_gaussians, 3, device=device)
        self.densities = torch.randn(n_gaussians, 1, device=device)
        self.features = torch.randn(n_gaussians, features_dim, device=device)
        self.extra_signal = torch.randn(n_gaussians, extra_signal_dim, device=device) if extra_signal_dim > 0 else None
        self.camera_extra_signal = (
            torch.randn(n_gaussians, camera_extra_signal_dim, device=device) if camera_extra_signal_dim > 0 else None
        )
        self.lidar_extra_signal = (
            torch.randn(n_gaussians, lidar_extra_signal_dim, device=device) if lidar_extra_signal_dim > 0 else None
        )


class MockCollector:
    """Mock for gaussian parameter collector."""

    def __init__(self, result: MockCollectorResult):
        self._result = result

    def collect(self, layers_data: Any) -> MockCollectorResult:
        return self._result


class MockBaseGaussianModel:
    """
    Base mock for gaussian models.
    Provides minimal interface for GaussianExportAccessor.
    """

    def __init__(
        self,
        n_gaussians: int = 100,
        device: str = "cpu",
        extra_signal_dim: int = 0,
        camera_extra_signal_dim: int = 0,
        lidar_extra_signal_dim: int = 0,
        extra_signal_names: Optional[Dict[str, Tuple[int, int]]] = None,
        camera_extra_signal_names: Optional[Dict[str, Tuple[int, int]]] = None,
        lidar_extra_signal_names: Optional[Dict[str, Tuple[int, int]]] = None,
    ):
        self.n_gaussians = n_gaussians
        self._device = torch.device(device)

        # Core parameters
        self.positions = torch.randn(n_gaussians, 3, device=self._device)
        self.rotations = torch.randn(n_gaussians, 4, device=self._device)
        self.scales = torch.randn(n_gaussians, 3, device=self._device)
        self.densities = torch.randn(n_gaussians, 1, device=self._device)

        # Extra signals
        self.extra_signal = torch.randn(n_gaussians, extra_signal_dim, device=self._device)
        self.camera_extra_signal = torch.randn(n_gaussians, camera_extra_signal_dim, device=self._device)
        self.lidar_extra_signal = torch.randn(n_gaussians, lidar_extra_signal_dim, device=self._device)

        # Signal param infos: ({name: index}, [dims])
        self.extra_signal_param_infos = self._build_signal_infos(extra_signal_names)
        self.camera_extra_signal_param_infos = self._build_signal_infos(camera_extra_signal_names)
        self.lidar_extra_signal_param_infos = self._build_signal_infos(lidar_extra_signal_names)

        # Config mock
        self.config = Mock()
        self.config.particle = Mock()
        self.config.particle.extra_signal_dim = extra_signal_dim
        self.config.particle.camera_extra_signal_dim = camera_extra_signal_dim
        self.config.particle.lidar_extra_signal_dim = lidar_extra_signal_dim
        self.config.rotation_activation = "normalize"
        self.config.scale_activation = "exp"
        self.config.density_activation = "sigmoid"

        # Collector result
        self._collector_result = MockCollectorResult(
            n_gaussians=n_gaussians,
            device=self._device,
            extra_signal_dim=extra_signal_dim,
            camera_extra_signal_dim=camera_extra_signal_dim,
            lidar_extra_signal_dim=lidar_extra_signal_dim,
        )
        self.collector = MockCollector(self._collector_result)

    def _build_signal_infos(self, names: Optional[Dict[str, Tuple[int, int]]]) -> Tuple[Dict[str, int], List[int]]:
        if names is None:
            return ({}, [])
        name_to_index = {}
        dims = []
        for i, (name, (idx, dim)) in enumerate(names.items()):
            name_to_index[name] = idx
            dims.append(dim)
        return (name_to_index, dims)

    def get_num_gaussians(self) -> int:
        return self.n_gaussians

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        return {
            "positions": self.positions,
            "rotations": self.rotations,
            "scales": self.scales,
            "densities": self.densities,
            "extra_signal": self.extra_signal,
            "camera_extra_signal": self.camera_extra_signal,
            "lidar_extra_signal": self.lidar_extra_signal,
        }

    def get_layer_config(self) -> collect.LayerConfigSH:
        # Use LayerConfigSH as base since collector only supports SH, Rigid, Deformable
        return collect.LayerConfigSH(
            rotation_activation=collect.RotationActivation.NORMALIZE,
            scale_activation=collect.ScaleActivation.EXP,
            density_activation=collect.DensityActivation.SIGMOID,
            fourier_features_dim=1,
            embed_config=None,
        )

    def get_albedo_sh_dim(self) -> int:
        return 3

    def get_specular_sh_dim(self) -> int:
        return 24  # Default for sh_degree=2

    def get_layer_data(self, context: Any, gathered_params: Dict[str, torch.Tensor]) -> collect.LayerDataBase:
        return collect.LayerDataBase(
            positions=gathered_params["positions"],
            rotations=gathered_params["rotations"],
            scales=gathered_params["scales"],
            densities=gathered_params["densities"],
            extra_signal=gathered_params["extra_signal"],
            camera_extra_signal=gathered_params["camera_extra_signal"],
            lidar_extra_signal=gathered_params["lidar_extra_signal"],
        )


class MockSHGaussianModel(MockBaseGaussianModel):
    """Mock for SH gaussian model with spherical harmonics."""

    def __init__(
        self,
        n_gaussians: int = 100,
        sh_degree: int = 2,
        device: str = "cpu",
        fourier_features_dim: int = 1,
        **kwargs,
    ):
        super().__init__(n_gaussians=n_gaussians, device=device, **kwargs)

        self.sh_degree = sh_degree
        self.fourier_features_dim = fourier_features_dim
        self.specular_sh_dim = (sh_degree + 1) ** 2 - 1
        self.features_albedo = torch.randn(n_gaussians, 1, 3, device=self._device)
        self.features_specular = torch.randn(n_gaussians, self.specular_sh_dim * 3, device=self._device)

        self.config.sh_degree = sh_degree
        self.config.fourier_features_dim = fourier_features_dim

        # Update collector result with features
        features_dim = 3 + self.specular_sh_dim * 3
        self._collector_result.features = torch.randn(n_gaussians, features_dim, device=self._device)

    def get_albedo_sh_dim(self) -> int:
        return 3

    def get_specular_sh_dim(self) -> int:
        return self.specular_sh_dim * 3

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        params = super().get_parameters()
        params["features_albedo"] = self.features_albedo
        params["features_specular"] = self.features_specular
        return params

    def get_layer_config(self) -> collect.LayerConfigSH:
        embed_config = None
        if self.fourier_features_dim > 1:
            # Use a real EmbeddingConfig for temporal appearance
            embed_config = collect.HolisticRemapTimeInputEmbeddingConfig(
                timestamps_us_min=0,
                timestamps_us_max=1000000,
                remap_min=0.0,
                remap_max=1.0,
            )
        return collect.LayerConfigSH(
            rotation_activation=collect.RotationActivation.NORMALIZE,
            scale_activation=collect.ScaleActivation.EXP,
            density_activation=collect.DensityActivation.SIGMOID,
            fourier_features_dim=self.fourier_features_dim,
            embed_config=embed_config,
        )


class MockRigidGaussianModel(MockBaseGaussianModel):
    """Mock for rigid gaussian model with cuboid tracks."""

    def __init__(
        self,
        n_gaussians: int = 100,
        n_tracks: int = 5,
        timestamps_per_track: int = 10,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(n_gaussians=n_gaussians, device=device, **kwargs)

        # Assign gaussians to tracks
        self.gaussian_cuboid_ids = torch.randint(0, n_tracks, (n_gaussians,), device=self._device)

        # Build track data
        track_ids = [f"track_{i:03d}" for i in range(n_tracks)]
        all_timestamps = []
        all_poses_tquat = []
        tracks_packinfo = []

        for track_idx in range(n_tracks):
            start_idx = track_idx * timestamps_per_track
            # Timestamps in microseconds
            for t_idx in range(timestamps_per_track):
                ts = (start_idx + t_idx) * 100000
                all_timestamps.append(ts)
                # Random pose: translation + quaternion
                t = torch.randn(3, device=self._device)
                q = torch.nn.functional.normalize(torch.randn(4, device=self._device), dim=-1)
                all_poses_tquat.append(torch.cat([t, q]))
            tracks_packinfo.append([track_idx * timestamps_per_track, timestamps_per_track])

        tracks_data = Mock(spec=TracksData)
        tracks_data.tracks_timestamps_us = torch.tensor(all_timestamps, dtype=torch.long, device=self._device)
        tracks_data.tracks_poses_tquat = torch.stack(all_poses_tquat)
        tracks_data.tracks_packinfo = torch.tensor(tracks_packinfo, dtype=torch.int32, device=self._device)

        self.cuboid_tracks = Mock(spec=CuboidTracks)
        self.cuboid_tracks.tracks_id = track_ids
        self.cuboid_tracks.n_tracks = n_tracks
        self.cuboid_tracks.tracks_data = tracks_data
        self.cuboid_tracks.device = self._device

        self._calibrated_tracks = Mock()

    def tracks_calib(self, tracks: Any) -> Any:
        """Return calibrated tracks."""
        return self._calibrated_tracks

    def get_albedo_sh_dim(self) -> int:
        return 3

    def get_specular_sh_dim(self) -> int:
        return 24  # Default for sh_degree=2

    def get_layer_config(self) -> collect.LayerConfigRigid:
        # LayerConfigRigid extends LayerConfigSH, so has SH capabilities
        return collect.LayerConfigRigid(
            rotation_activation=collect.RotationActivation.NORMALIZE,
            scale_activation=collect.ScaleActivation.EXP,
            density_activation=collect.DensityActivation.SIGMOID,
            fourier_features_dim=1,
            embed_config=None,
        )


class MockDeformableGaussianModel(MockBaseGaussianModel):
    """Mock for deformable gaussian model."""

    def __init__(self, n_gaussians: int = 100, device: str = "cpu", **kwargs):
        super().__init__(n_gaussians=n_gaussians, device=device, **kwargs)
        self.deform_network = Mock()
        self.deform_network.forward = Mock(return_value=torch.randn(n_gaussians, 3, device=self._device))

    def get_albedo_sh_dim(self) -> int:
        return 3

    def get_specular_sh_dim(self) -> int:
        return 24  # Default for sh_degree=2

    def get_layer_config(self) -> collect.LayerConfigDeformable:
        # LayerConfigDeformable extends LayerConfigRigid (which extends LayerConfigSH)
        return collect.LayerConfigDeformable(
            rotation_activation=collect.RotationActivation.NORMALIZE,
            scale_activation=collect.ScaleActivation.EXP,
            density_activation=collect.DensityActivation.SIGMOID,
            fourier_features_dim=1,
            embed_config=None,
        )


class MockSHRigidGaussianModel(MockSHGaussianModel, MockRigidGaussianModel):
    """Mock for SH model with rigid tracks (common combo)."""

    def __init__(
        self,
        n_gaussians: int = 100,
        sh_degree: int = 2,
        n_tracks: int = 5,
        timestamps_per_track: int = 10,
        device: str = "cpu",
        fourier_features_dim: int = 1,
        **kwargs,
    ):
        MockSHGaussianModel.__init__(
            self,
            n_gaussians=n_gaussians,
            sh_degree=sh_degree,
            device=device,
            fourier_features_dim=fourier_features_dim,
            **kwargs,
        )
        # Add rigid track attributes from MockRigidGaussianModel
        MockRigidGaussianModel.__init__(
            self,
            n_gaussians=n_gaussians,
            n_tracks=n_tracks,
            timestamps_per_track=timestamps_per_track,
            device=device,
            **kwargs,
        )

    def get_layer_config(self) -> collect.LayerConfigRigid:
        # Override to return LayerConfigRigid for combined SH + Rigid model
        return collect.LayerConfigRigid(
            rotation_activation=collect.RotationActivation.NORMALIZE,
            scale_activation=collect.ScaleActivation.EXP,
            density_activation=collect.DensityActivation.SIGMOID,
            fourier_features_dim=self.fourier_features_dim,
            embed_config=None,
        )


# =============================================================================
# Tests
# =============================================================================


class TestAccessorBasicInfo(unittest.TestCase):
    """Tests for basic accessor info methods."""

    def test_get_num_gaussians(self):
        model = MockBaseGaussianModel(n_gaussians=150)
        accessor = GaussianExportAccessor(model)
        self.assertEqual(accessor.get_num_gaussians(), 150)


class TestAccessorCapabilities(unittest.TestCase):
    """Tests for get_capabilities() across model types."""

    def test_capabilities_base_model(self):
        model = MockBaseGaussianModel()
        accessor = GaussianExportAccessor(model)
        caps = accessor.get_capabilities()

        # MockBaseGaussianModel now uses LayerConfigSH for collector compatibility
        self.assertTrue(caps.has_spherical_harmonics)
        self.assertFalse(caps.has_rigid_tracks)
        self.assertFalse(caps.has_deformation)
        self.assertFalse(caps.has_temporal_appearance)

    def test_capabilities_sh_model(self):
        model = MockSHGaussianModel(sh_degree=3)
        accessor = GaussianExportAccessor(model)
        caps = accessor.get_capabilities()

        self.assertTrue(caps.has_spherical_harmonics)
        self.assertFalse(caps.has_rigid_tracks)
        self.assertFalse(caps.has_deformation)
        self.assertFalse(caps.has_temporal_appearance)

    def test_capabilities_sh_model_temporal(self):
        """Test SH model with temporal appearance (animated albedo)."""
        model = MockSHGaussianModel(sh_degree=2, fourier_features_dim=8)
        accessor = GaussianExportAccessor(model)
        caps = accessor.get_capabilities()

        self.assertTrue(caps.has_spherical_harmonics)
        self.assertTrue(caps.has_temporal_appearance)

    def test_capabilities_rigid_model(self):
        model = MockRigidGaussianModel(n_tracks=5)
        accessor = GaussianExportAccessor(model)
        caps = accessor.get_capabilities()

        # LayerConfigRigid extends LayerConfigSH, so has SH capabilities
        self.assertTrue(caps.has_spherical_harmonics)
        self.assertTrue(caps.has_rigid_tracks)
        self.assertFalse(caps.has_deformation)

    def test_capabilities_deformable_model(self):
        model = MockDeformableGaussianModel()
        accessor = GaussianExportAccessor(model)
        caps = accessor.get_capabilities()

        # LayerConfigDeformable extends LayerConfigRigid (which extends LayerConfigSH)
        self.assertTrue(caps.has_spherical_harmonics)
        self.assertTrue(caps.has_rigid_tracks)
        self.assertTrue(caps.has_deformation)
        self.assertTrue(caps.can_deform_positions)
        self.assertTrue(caps.can_deform_rotations)
        self.assertFalse(caps.can_deform_scales)

    def test_capabilities_sh_rigid_model(self):
        """Test combined SH + Rigid model."""
        model = MockSHRigidGaussianModel(sh_degree=2, n_tracks=3)
        accessor = GaussianExportAccessor(model)
        caps = accessor.get_capabilities()

        self.assertTrue(caps.has_spherical_harmonics)
        self.assertTrue(caps.has_rigid_tracks)
        self.assertFalse(caps.has_deformation)


class TestAccessorTracks(unittest.TestCase):
    """Tests for rigid track helpers."""

    def test_get_track_gaussian_mapping(self):
        n_gaussians = 100
        n_tracks = 5
        model = MockRigidGaussianModel(n_gaussians=n_gaussians, n_tracks=n_tracks)
        accessor = GaussianExportAccessor(model)

        mapping = accessor.get_track_gaussian_mapping()

        self.assertEqual(len(mapping), n_tracks)
        for track_id in model.cuboid_tracks.tracks_id:
            self.assertIn(track_id, mapping)
            self.assertIsInstance(mapping[track_id], torch.Tensor)

    def test_get_track_gaussian_mapping_empty_for_non_rigid(self):
        model = MockBaseGaussianModel()
        accessor = GaussianExportAccessor(model)

        mapping = accessor.get_track_gaussian_mapping()
        self.assertEqual(mapping, {})

    def test_get_track_time_range(self):
        model = MockRigidGaussianModel(n_tracks=3, timestamps_per_track=10)
        accessor = GaussianExportAccessor(model)

        start_us, end_us = accessor.get_track_time_range(track_index=0)

        self.assertIsInstance(start_us, int)
        self.assertIsInstance(end_us, int)
        self.assertLess(start_us, end_us)

    def test_get_track_timestamps(self):
        timestamps_per_track = 15
        model = MockRigidGaussianModel(n_tracks=2, timestamps_per_track=timestamps_per_track)
        accessor = GaussianExportAccessor(model)

        timestamps = accessor.get_track_timestamps(track_index=0)

        self.assertIsInstance(timestamps, list)
        self.assertEqual(len(timestamps), timestamps_per_track)
        self.assertTrue(all(isinstance(t, int) for t in timestamps))
        # Should be sorted
        self.assertEqual(timestamps, sorted(timestamps))

    def test_get_cuboid_tracks(self):
        model = MockRigidGaussianModel(n_tracks=4)
        accessor = GaussianExportAccessor(model)

        tracks = accessor.get_cuboid_tracks()

        self.assertEqual(tracks.n_tracks, 4)

    def test_get_cuboid_tracks_raises_for_non_rigid(self):
        model = MockBaseGaussianModel()
        accessor = GaussianExportAccessor(model)

        with self.assertRaises(RuntimeError) as ctx:
            accessor.get_cuboid_tracks()
        self.assertIn("cuboid tracks", str(ctx.exception))


class TestAnimatedTracks(unittest.TestCase):
    """Tests for animated track scenarios."""

    def test_track_poses_over_time(self):
        """Verify track timestamps represent animation frames."""
        timestamps_per_track = 20
        model = MockRigidGaussianModel(n_tracks=3, timestamps_per_track=timestamps_per_track)
        accessor = GaussianExportAccessor(model)

        for track_idx in range(3):
            timestamps = accessor.get_track_timestamps(track_idx)
            self.assertEqual(len(timestamps), timestamps_per_track)

            start, end = accessor.get_track_time_range(track_idx)
            self.assertEqual(start, timestamps[0])
            self.assertEqual(end, timestamps[-1])

    def test_gaussian_to_track_assignment(self):
        """Verify all gaussians are assigned to tracks."""
        n_gaussians = 100
        n_tracks = 5
        model = MockRigidGaussianModel(n_gaussians=n_gaussians, n_tracks=n_tracks)
        accessor = GaussianExportAccessor(model)

        mapping = accessor.get_track_gaussian_mapping()

        # Count total gaussians across all tracks
        total = sum(len(indices) for indices in mapping.values())
        self.assertEqual(total, n_gaussians)


if __name__ == "__main__":
    unittest.main()
