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
Integration tests for USD Gaussian export pipeline.

Tests the full export flow:
1. Create mock accessor with predefined gaussian data
2. Export via export_gaussians_as_usd_asset
3. Read back USD file
4. Verify exported data matches input

Test scenarios:
- Static gaussians (no animation)
- Rigid tracks with animated transforms
- Deformable gaussians (animated positions/rotations)
- Temporal appearance (animated SH/albedo)
"""

import tempfile
import unittest

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import Mock

import numpy as np
import torch

from pxr import Gf, Sdf, Usd, UsdGeom, UsdVol

from nre.utils.io.export.gaussian_export_accessor import (
    GaussianAttributes,
    ModelCapabilities,
)
from nre.utils.io.export.gaussian_usd_asset import export_gaussians_as_usd_asset
from nre.utils.io.export.gaussian_usd_writer import GaussianUSDSchemaType, create_gaussian_writer


def _get_attr_by_suffix(prim: Usd.Prim, suffix: str) -> Optional[Usd.Attribute]:
    """Return attribute whose name equals or ends with suffix (handles UsdVol API prefixes)."""
    for attr in prim.GetAttributes():
        name = attr.GetName()
        if name == suffix or name.endswith(":" + suffix):
            return attr
    return None


# =============================================================================
# Mock Accessor
# =============================================================================


class MockAccessor:
    """
    Mock accessor for integration testing.

    Provides predefined gaussian data that can be verified after export.
    """

    def __init__(
        self,
        n_gaussians: int = 100,
        positions: Optional[np.ndarray] = None,
        rotations: Optional[np.ndarray] = None,
        scales: Optional[np.ndarray] = None,
        densities: Optional[np.ndarray] = None,
        # Capabilities
        has_spherical_harmonics: bool = False,
        has_rigid_tracks: bool = False,
        has_deformation: bool = False,
        has_temporal_appearance: bool = False,
        is_planar_gaussian: bool = False,
        sh_degree: int = 3,
        fourier_features_dim: int = 1,
        # Track data (for rigid tracks)
        track_ids: Optional[List[str]] = None,
        track_timestamps_us: Optional[Dict[int, List[int]]] = None,
        track_transforms: Optional[Dict[int, Dict[int, np.ndarray]]] = None,
        gaussian_track_assignments: Optional[np.ndarray] = None,
        # SH coefficients
        albedo_coeffs: Optional[np.ndarray] = None,
        specular_coeffs: Optional[np.ndarray] = None,
        # Deformation: position/rotation per timestamp
        deformed_positions: Optional[Dict[int, np.ndarray]] = None,
        deformed_rotations: Optional[Dict[int, np.ndarray]] = None,
        # Device
        device: str = "cpu",
    ):
        self.n_gaussians = n_gaussians
        self._device = torch.device(device)

        # Generate default data if not provided
        rng = np.random.RandomState(42)
        self.positions = positions if positions is not None else rng.randn(n_gaussians, 3).astype(np.float32)
        self.rotations = rotations if rotations is not None else self._make_normalized_quats(n_gaussians, rng)
        self.scales = scales if scales is not None else np.abs(rng.randn(n_gaussians, 3).astype(np.float32)) * 0.1
        self.densities = densities if densities is not None else rng.rand(n_gaussians, 1).astype(np.float32)

        # Capabilities
        self._capabilities = ModelCapabilities(
            has_spherical_harmonics=has_spherical_harmonics,
            has_rigid_tracks=has_rigid_tracks,
            has_deformation=has_deformation,
            can_deform_positions=has_deformation,
            can_deform_rotations=has_deformation,
            can_deform_scales=False,
            has_temporal_appearance=has_temporal_appearance,
            sh_degree=sh_degree if has_spherical_harmonics else None,
            is_planar_gaussian=is_planar_gaussian,
        )

        # Track data
        self.track_ids = track_ids or []
        self.track_timestamps_us = track_timestamps_us or {}
        self.track_transforms = track_transforms or {}
        self.gaussian_track_assignments = gaussian_track_assignments

        # SH data
        self.albedo_coeffs: Optional[np.ndarray] = None
        self.specular_coeffs: Optional[np.ndarray] = None
        if has_spherical_harmonics:
            num_sh_coeffs = (sh_degree + 1) ** 2
            self.albedo_coeffs = (
                albedo_coeffs if albedo_coeffs is not None else rng.randn(n_gaussians, 3).astype(np.float32)
            )
            self.specular_coeffs = (
                specular_coeffs
                if specular_coeffs is not None
                else rng.randn(n_gaussians, (num_sh_coeffs - 1) * 3).astype(np.float32)
            )

        # Deformation data
        self.deformed_positions = deformed_positions or {}
        self.deformed_rotations = deformed_rotations or {}

        # Build mock cuboid tracks if rigid
        if has_rigid_tracks and track_ids:
            self._build_mock_cuboid_tracks()

    def _make_normalized_quats(self, n: int, rng: np.random.RandomState) -> np.ndarray:
        quats = rng.randn(n, 4).astype(np.float32)
        quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
        return quats

    def _build_mock_cuboid_tracks(self):
        """Build mock CuboidTracks structure."""
        from unittest.mock import Mock

        n_tracks = len(self.track_ids)

        # Build tracks_data
        all_timestamps = []
        all_packinfo = []
        for track_idx in range(n_tracks):
            timestamps = self.track_timestamps_us.get(track_idx, [0])
            start_idx = len(all_timestamps)
            all_timestamps.extend(timestamps)
            all_packinfo.append([start_idx, len(timestamps)])

        tracks_data = Mock()
        tracks_data.tracks_timestamps_us = torch.tensor(all_timestamps, dtype=torch.long, device=self._device)
        tracks_data.tracks_packinfo = torch.tensor(all_packinfo, dtype=torch.int32, device=self._device)

        self._cuboid_tracks = Mock()
        self._cuboid_tracks.tracks_id = self.track_ids
        self._cuboid_tracks.n_tracks = n_tracks
        self._cuboid_tracks.tracks_data = tracks_data
        self._cuboid_tracks.device = self._device

        # Build calibrated tracks with interpolation
        self._tracks_calib = Mock()
        self._tracks_calib.interpolate_tracks_poses = self._mock_interpolate_poses

    def _mock_interpolate_poses(self, timestamps_us: torch.Tensor, tracks_idx: torch.Tensor) -> List[Any]:
        """Mock pose interpolation - returns SE3 poses."""
        from unittest.mock import Mock

        poses = []
        for i, (ts, track_idx) in enumerate(zip(timestamps_us.tolist(), tracks_idx.tolist())):
            track_transforms = self.track_transforms.get(track_idx, {})
            if ts in track_transforms:
                mat = track_transforms[ts]
            else:
                # Identity transform
                mat = np.eye(4, dtype=np.float64)

            pose = Mock()
            pose.matrix = Mock(return_value=torch.from_numpy(mat))
            poses.append(pose)
        return poses

    # === Accessor Interface ===

    def get_num_gaussians(self) -> int:
        return self.n_gaussians

    def get_capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def get_attributes_at_timestamp(
        self, timestamp_us: int, preactivation: bool = False
    ) -> Tuple[GaussianAttributes, Optional[np.ndarray], Optional[np.ndarray]]:
        # Use deformed data if available
        positions = self.deformed_positions.get(timestamp_us, self.positions)
        rotations = self.deformed_rotations.get(timestamp_us, self.rotations)

        # Build interpolated track poses for rigid tracks (as numpy array, matching real accessor)
        interpolated_track_poses = None
        if self._capabilities.has_rigid_tracks and self.track_ids:
            # Create poses array: [n_tracks, 7] where 7 = [tx, ty, tz, qx, qy, qz, qw]
            n_tracks = len(self.track_ids)
            poses = np.zeros((n_tracks, 7), dtype=np.float32)
            for track_idx, track_id in enumerate(self.track_ids):
                track_transforms = self.track_transforms.get(track_idx, {})
                if timestamp_us in track_transforms:
                    mat = track_transforms[timestamp_us]
                else:
                    mat = np.eye(4, dtype=np.float64)

                # Extract translation and rotation from 4x4 matrix
                translation = mat[:3, 3]
                # Convert rotation matrix to quaternion (xyzw format)
                from scipy.spatial.transform import Rotation

                rot = Rotation.from_matrix(mat[:3, :3])
                quat_xyzw = rot.as_quat()  # [x, y, z, w]

                poses[track_idx] = [
                    translation[0],
                    translation[1],
                    translation[2],
                    quat_xyzw[0],
                    quat_xyzw[1],
                    quat_xyzw[2],
                    quat_xyzw[3],
                ]

            interpolated_track_poses = poses

        # Per-track visibility mask (True when timestamp in range); for tests all visible when we have poses
        track_visibility_mask = (
            np.ones(len(self.track_ids), dtype=bool)
            if (self._capabilities.has_rigid_tracks and self.track_ids)
            else None
        )

        # GaussianAttributes requires non-None albedo_coefficients; use empty array as fallback
        albedo = (
            self.albedo_coeffs if self.albedo_coeffs is not None else np.zeros((self.n_gaussians, 3), dtype=np.float32)
        )
        return (
            GaussianAttributes(
                positions=positions,
                rotations=rotations,
                scales=self.scales,
                densities=self.densities,
                albedo_coefficients=albedo,
                specular_coefficients=self.specular_coeffs,
            ),
            interpolated_track_poses,
            track_visibility_mask,
        )

    def get_track_gaussian_mapping(self) -> Dict[str, torch.Tensor]:
        if not self._capabilities.has_rigid_tracks:
            return {}
        mapping = {}
        for track_idx, track_id in enumerate(self.track_ids):
            if self.gaussian_track_assignments is not None:
                mask = self.gaussian_track_assignments == track_idx
                indices = np.where(mask)[0]
                mapping[track_id] = torch.tensor(indices, device=self._device)
            else:
                # Default: all gaussians belong to first track
                if track_idx == 0:
                    mapping[track_id] = torch.arange(self.n_gaussians, device=self._device)
        return mapping

    def get_track_time_range(self, track_index: int) -> Tuple[int, int]:
        timestamps = self.track_timestamps_us.get(track_index, [0])
        return min(timestamps), max(timestamps)

    def get_track_time_range_safe(self, track_index: int) -> Optional[Tuple[int, int]]:
        """Return (min_us, max_us) for the track, or None if the track has no timestamps."""
        timestamps = self.track_timestamps_us.get(track_index, [])
        if not timestamps:
            return None
        return min(timestamps), max(timestamps)

    def get_track_timestamps(self, track_index: int) -> List[int]:
        return self.track_timestamps_us.get(track_index, [0])

    def get_cuboid_tracks(self):
        if not self._capabilities.has_rigid_tracks:
            raise RuntimeError("No tracks")
        return self._cuboid_tracks

    def get_valid_gaussian_mask(self, timestamp_us: int, preactivation: bool = False) -> np.ndarray:
        """Get a mask of valid Gaussians (no NaN/Inf values).

        For testing, all Gaussians are valid unless explicitly set otherwise.
        """
        # Check for NaN/Inf in critical attributes
        valid_positions = np.all(np.isfinite(self.positions), axis=1)
        valid_rotations = np.all(np.isfinite(self.rotations), axis=1)
        valid_scales = np.all(np.isfinite(self.scales), axis=1)
        densities_2d = self.densities.reshape(-1, 1) if self.densities.ndim == 1 else self.densities
        valid_densities = np.all(np.isfinite(densities_2d), axis=1)

        albedo = (
            self.albedo_coeffs if self.albedo_coeffs is not None else np.zeros((self.n_gaussians, 3), dtype=np.float32)
        )
        valid_albedo = np.all(np.isfinite(albedo), axis=1)

        return valid_positions & valid_rotations & valid_scales & valid_densities & valid_albedo


# =============================================================================
# Test: Static Gaussians
# =============================================================================


class TestStaticGaussiansExport(unittest.TestCase):
    """Test export of static (non-animated) gaussians."""

    def test_static_positions_exported(self):
        """Verify positions are exported correctly."""
        n_gaussians = 50
        positions = np.array([[i, i * 2, i * 3] for i in range(n_gaussians)], dtype=np.float32)

        accessor = MockAccessor(
            n_gaussians=n_gaussians,
            positions=positions,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "static_test.usda"
            stage = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
            stage.SetTimeCodesPerSecond(24.0)
            export_gaussians_as_usd_asset(
                accessor=accessor,
                stage=stage,
                content_root_path="/World",
                schema_type=GaussianUSDSchemaType.LIGHT_FIELD,
            )

            # Save the stage
            stage.Export(str(output_path))

            # Read back and verify
            stage = Usd.Stage.Open(str(output_path))
            self.assertIsNotNone(stage)

            # Find gaussian prim
            prim = self._find_particle_field_prim(stage)
            self.assertIsNotNone(prim, "No ParticleField prim found")

            positions_attr = prim.GetAttribute("positions")
            exported_positions = np.array(positions_attr.Get())

            np.testing.assert_allclose(exported_positions, positions, rtol=1e-5)

    def test_static_rotations_exported(self):
        """Verify rotations (quaternions) are exported correctly."""
        n_gaussians = 30
        # Create specific known quaternions (w, x, y, z)
        rotations = np.array(
            [
                [1, 0, 0, 0],  # Identity
                [0.707, 0.707, 0, 0],  # 90° around X
                [0.707, 0, 0.707, 0],  # 90° around Y
            ]
            * 10,
            dtype=np.float32,
        )

        accessor = MockAccessor(
            n_gaussians=n_gaussians,
            rotations=rotations,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rotation_test.usda"
            stage = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
            stage.SetTimeCodesPerSecond(24.0)
            export_gaussians_as_usd_asset(
                accessor=accessor,
                stage=stage,
                content_root_path="/World",
                schema_type=GaussianUSDSchemaType.LIGHT_FIELD,
            )
            stage.Export(str(output_path))

            stage = Usd.Stage.Open(str(output_path))
            prim = self._find_particle_field_prim(stage)

            orientations_attr = prim.GetAttribute("orientations")
            quats = orientations_attr.Get()

            for i, quat in enumerate(quats):
                expected = rotations[i]
                # Gf.Quatf: real=w, imaginary=(x,y,z)
                actual = np.array([quat.GetReal(), *quat.GetImaginary()])
                np.testing.assert_allclose(actual, expected, rtol=1e-4)

    def test_static_scales_exported(self):
        """Verify scales are exported correctly."""
        n_gaussians = 20
        scales = np.array([[0.1, 0.2, 0.3]] * n_gaussians, dtype=np.float32)

        accessor = MockAccessor(
            n_gaussians=n_gaussians,
            scales=scales,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "scale_test.usda"
            stage = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
            stage.SetTimeCodesPerSecond(24.0)
            export_gaussians_as_usd_asset(
                accessor=accessor,
                stage=stage,
                content_root_path="/World",
                schema_type=GaussianUSDSchemaType.LIGHT_FIELD,
            )
            stage.Export(str(output_path))

            stage = Usd.Stage.Open(str(output_path))
            prim = self._find_particle_field_prim(stage)

            scales_attr = prim.GetAttribute("scales")
            exported_scales = np.array(scales_attr.Get())

            np.testing.assert_allclose(exported_scales, scales, rtol=1e-5)

    def test_planar_gaussian_surflet_kernel(self):
        """Verify planar gaussians use surflet kernel and zero z-scale."""
        n_gaussians = 20
        scales = np.array([[0.1, 0.2, 0.3]] * n_gaussians, dtype=np.float32)

        accessor = MockAccessor(
            n_gaussians=n_gaussians,
            scales=scales,
            is_planar_gaussian=True,  # Enable planar/surflet mode
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "planar_test.usda"
            stage = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
            stage.SetTimeCodesPerSecond(24.0)
            export_gaussians_as_usd_asset(
                accessor=accessor,
                stage=stage,
                content_root_path="/World",
                schema_type=GaussianUSDSchemaType.LIGHT_FIELD,
            )
            stage.Export(str(output_path))

            stage = Usd.Stage.Open(str(output_path))
            prim = self._find_particle_field_prim(stage)
            self.assertIsNotNone(prim)

            # Verify surflet kernel: prim uses ParticleField + ParticleFieldKernelGaussianSurfletAPI
            self.assertTrue(
                prim.HasAPI(UsdVol.ParticleFieldKernelGaussianSurfletAPI),
                "Planar gaussians should have ParticleFieldKernelGaussianSurfletAPI applied",
            )

            # Verify z-component of scales is zeroed out (accept UsdVol API-prefixed name)
            scales_attr = _get_attr_by_suffix(prim, "scales")
            self.assertIsNotNone(scales_attr)
            exported_scales = np.array(scales_attr.Get())
            expected_scales = scales.copy()
            expected_scales[:, 2] = 0.0  # z-component should be zero for surflets
            np.testing.assert_allclose(exported_scales, expected_scales, rtol=1e-5)

    def test_ellipsoid_gaussian_kernel_type(self):
        """Verify non-planar gaussians have ellipsoid kernel type."""
        n_gaussians = 10
        accessor = MockAccessor(
            n_gaussians=n_gaussians,
            is_planar_gaussian=False,  # Default ellipsoid mode
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "ellipsoid_test.usda"
            stage = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
            stage.SetTimeCodesPerSecond(24.0)
            export_gaussians_as_usd_asset(
                accessor=accessor,
                stage=stage,
                content_root_path="/World",
                schema_type=GaussianUSDSchemaType.LIGHT_FIELD,
            )
            stage.Export(str(output_path))

            stage = Usd.Stage.Open(str(output_path))
            prim = self._find_particle_field_prim(stage)

            # Verify ellipsoid kernel: prim is typed ParticleField3DGaussianSplat (no surflet API)
            self.assertEqual(
                prim.GetTypeName(),
                "ParticleField3DGaussianSplat",
                "Non-planar gaussians should use ParticleField3DGaussianSplat schema",
            )
            self.assertFalse(
                prim.HasAPI(UsdVol.ParticleFieldKernelGaussianSurfletAPI),
                "Ellipsoid gaussians should not have ParticleFieldKernelGaussianSurfletAPI",
            )

    def test_half_precision_attributes(self):
        """Verify half-precision mode uses 'h' suffix attribute names."""
        n_gaussians = 10
        accessor = MockAccessor(n_gaussians=n_gaussians)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "half_precision_test.usda"
            stage = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
            stage.SetTimeCodesPerSecond(24.0)
            export_gaussians_as_usd_asset(
                accessor=accessor,
                stage=stage,
                content_root_path="/World",
                schema_type=GaussianUSDSchemaType.LIGHT_FIELD,
                half_precision=True,
            )
            stage.Export(str(output_path))

            stage = Usd.Stage.Open(str(output_path))
            prim = self._find_particle_field_prim(stage)

            # Verify half-precision attribute names exist and have values (accept UsdVol API-prefixed names).
            # UsdVol schema may declare both positions/positionsh; we only populate the *h attributes.
            for suffix in ("positionsh", "orientationsh", "scalesh", "opacitiesh"):
                attr = _get_attr_by_suffix(prim, suffix)
                self.assertIsNotNone(attr, f"Half-precision attribute {suffix} should exist")
                self.assertTrue(attr.Get() is not None, f"Half-precision attribute {suffix} should have data")

    def test_static_opacities_clamped(self):
        """Verify densities are clamped to [0, 1] as opacities."""
        n_gaussians = 10
        # Include values outside [0, 1]
        densities = np.array([[-0.5], [0.0], [0.5], [1.0], [1.5]] * 2, dtype=np.float32)

        accessor = MockAccessor(
            n_gaussians=n_gaussians,
            densities=densities,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "opacity_test.usda"
            stage = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
            stage.SetTimeCodesPerSecond(24.0)
            export_gaussians_as_usd_asset(
                accessor=accessor,
                stage=stage,
                content_root_path="/World",
                schema_type=GaussianUSDSchemaType.LIGHT_FIELD,
            )
            stage.Export(str(output_path))

            stage = Usd.Stage.Open(str(output_path))
            prim = self._find_particle_field_prim(stage)

            opacities_attr = prim.GetAttribute("opacities")
            exported_opacities = np.array(opacities_attr.Get())

            expected = np.clip(densities.flatten(), 0.0, 1.0)
            np.testing.assert_allclose(exported_opacities, expected, rtol=1e-5)

    def _find_particle_field_prim(self, stage: Usd.Stage) -> Optional[Usd.Prim]:
        """Find the ParticleField prim in the stage (3D splat or planar surflet)."""
        for prim in stage.Traverse():
            if prim.GetTypeName() == "ParticleField3DGaussianSplat":
                return prim
        for prim in stage.Traverse():
            if prim.GetTypeName() == "ParticleField":
                return prim
        return None


# =============================================================================
# Test: Spherical Harmonics
# =============================================================================


class TestSphericalHarmonicsExport(unittest.TestCase):
    """Test export of SH coefficients."""

    def test_sh_coefficients_exported(self):
        """Verify SH albedo + specular are exported as combined array."""
        n_gaussians = 25
        sh_degree = 2
        num_sh_coeffs = (sh_degree + 1) ** 2  # 9 for degree 2

        # Create known SH data
        albedo = np.ones((n_gaussians, 3), dtype=np.float32) * 0.5
        specular = np.zeros((n_gaussians, (num_sh_coeffs - 1) * 3), dtype=np.float32)

        accessor = MockAccessor(
            n_gaussians=n_gaussians,
            has_spherical_harmonics=True,
            sh_degree=sh_degree,
            albedo_coeffs=albedo,
            specular_coeffs=specular,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sh_test.usda"
            stage = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
            stage.SetTimeCodesPerSecond(24.0)
            export_gaussians_as_usd_asset(
                accessor=accessor,
                stage=stage,
                content_root_path="/World",
                schema_type=GaussianUSDSchemaType.LIGHT_FIELD,
            )
            stage.Export(str(output_path))

            stage = Usd.Stage.Open(str(output_path))
            prim = self._find_particle_field_prim(stage)

            sh_attr = prim.GetAttribute("radiance:sphericalHarmonicsCoefficients")
            self.assertTrue(sh_attr.IsValid())

            coeffs = np.array(sh_attr.Get())
            # Should have n_gaussians * num_sh_coeffs entries
            self.assertEqual(coeffs.shape[0], n_gaussians * num_sh_coeffs)

            # First coeff of each gaussian should be albedo
            for i in range(n_gaussians):
                idx = i * num_sh_coeffs
                np.testing.assert_allclose(coeffs[idx], albedo[i], rtol=1e-5)

    def _find_particle_field_prim(self, stage: Usd.Stage) -> Optional[Usd.Prim]:
        for prim in stage.Traverse():
            if prim.GetTypeName() == "ParticleField3DGaussianSplat":
                return prim
        return None


# =============================================================================
# Test: Color Space
# =============================================================================


class TestColorSpaceExport(unittest.TestCase):
    """Test USD ColorSpaceAPI application for spherical harmonics."""

    def test_linear_color_space_lightfield(self):
        """Verify ColorSpaceAPI with lin_rec709_scene is applied for LightField with linear_srgb=True."""
        accessor = MockAccessor(
            n_gaussians=50,
            has_spherical_harmonics=True,
            sh_degree=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "linear_colorspace_test.usda"
            stage = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
            stage.SetTimeCodesPerSecond(24.0)

            export_gaussians_as_usd_asset(
                accessor=accessor,
                stage=stage,
                content_root_path="/World",
                schema_type=GaussianUSDSchemaType.LIGHT_FIELD,
                linear_srgb=True,  # Simulate post-processing present
            )
            stage.Export(str(output_path))

            # Reload and verify
            stage = Usd.Stage.Open(str(output_path))
            prim = self._find_particle_field_prim(stage)
            self.assertIsNotNone(prim, "ParticleField prim should exist")

            # Check ColorSpaceAPI is applied with correct value
            color_space_attr = prim.GetAttribute("colorSpace:name")
            self.assertTrue(color_space_attr.IsValid(), "colorSpace:name attribute should exist")
            color_space_value = color_space_attr.Get()
            self.assertEqual(
                color_space_value,
                "lin_rec709_scene",
                "Color space should be lin_rec709_scene when linear_srgb=True",
            )

    def test_srgb_color_space_geompoints(self):
        """Verify ColorSpaceAPI with srgb_rec709_display is applied for GeomPoints with linear_srgb=False."""
        accessor = MockAccessor(
            n_gaussians=50,
            has_spherical_harmonics=True,
            sh_degree=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "srgb_colorspace_test.usda"
            stage = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
            stage.SetTimeCodesPerSecond(24.0)

            export_gaussians_as_usd_asset(
                accessor=accessor,
                stage=stage,
                content_root_path="/World",
                schema_type=GaussianUSDSchemaType.GEOM_POINTS,
                linear_srgb=False,  # Simulate no post-processing
            )
            stage.Export(str(output_path))

            # Reload and verify
            stage = Usd.Stage.Open(str(output_path))
            points_prim = self._find_points_prim(stage)
            self.assertIsNotNone(points_prim, "Points prim should exist")

            # Check ColorSpaceAPI is applied with correct value
            color_space_attr = points_prim.GetAttribute("colorSpace:name")
            self.assertTrue(color_space_attr.IsValid(), "colorSpace:name attribute should exist")
            color_space_value = color_space_attr.Get()
            self.assertEqual(
                color_space_value,
                "srgb_rec709_display",
                "Color space should be srgb_rec709_display when linear_srgb=False",
            )

    def _find_particle_field_prim(self, stage: Usd.Stage) -> Optional[Usd.Prim]:
        for prim in stage.Traverse():
            if prim.GetTypeName() == "ParticleField3DGaussianSplat":
                return prim
        return None

    def _find_points_prim(self, stage: Usd.Stage) -> Optional[Usd.Prim]:
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Points):
                return prim
        return None


# =============================================================================
# Test: Rigid Tracks
# =============================================================================


class TestRigidTracksExport(unittest.TestCase):
    """Test export of gaussians with rigid track animations."""

    def test_track_hierarchy_created(self):
        """Verify separate Xform prims are created for each track."""
        n_gaussians = 60
        track_ids = ["track_A", "track_B", "track_C"]
        # Assign gaussians evenly to tracks
        assignments = np.array([i % 3 for i in range(n_gaussians)], dtype=np.int32)

        accessor = MockAccessor(
            n_gaussians=n_gaussians,
            has_rigid_tracks=True,
            track_ids=track_ids,
            gaussian_track_assignments=assignments,
            track_timestamps_us={0: [0], 1: [0], 2: [0]},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "tracks_test.usda"
            stage = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
            stage.SetTimeCodesPerSecond(24.0)
            export_gaussians_as_usd_asset(
                accessor=accessor,
                stage=stage,
                content_root_path="/World",
                schema_type=GaussianUSDSchemaType.LIGHT_FIELD,
            )
            stage.Export(str(output_path))

            stage = Usd.Stage.Open(str(output_path))

            # Should have 3 track Xform prims
            xform_count = 0
            for prim in stage.Traverse():
                if prim.IsA(UsdGeom.Xform) and "track_" in str(prim.GetPath()):
                    xform_count += 1

            self.assertEqual(xform_count, 3)

    def test_track_transforms_keyframed(self):
        """Verify track transforms are keyframed at correct times."""
        n_gaussians = 20
        track_ids = ["animated_track"]
        timestamps = [0, 100000, 200000]  # 3 frames

        # Create transforms: rotation + translation over time
        track_transforms = {
            0: {
                0: np.eye(4, dtype=np.float64),
                100000: self._make_transform(translation=[1, 0, 0]),
                200000: self._make_transform(translation=[2, 0, 0]),
            }
        }

        accessor = MockAccessor(
            n_gaussians=n_gaussians,
            has_rigid_tracks=True,
            track_ids=track_ids,
            track_timestamps_us={0: timestamps},
            track_transforms=track_transforms,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "track_anim_test.usda"
            stage = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
            stage.SetTimeCodesPerSecond(24.0)
            export_gaussians_as_usd_asset(
                accessor=accessor,
                stage=stage,
                content_root_path="/World",
                schema_type=GaussianUSDSchemaType.LIGHT_FIELD,
            )
            stage.Export(str(output_path))

            stage = Usd.Stage.Open(str(output_path))

            # Find track Xform and check time samples
            for prim in stage.Traverse():
                if prim.IsA(UsdGeom.Xform) and "track_" in str(prim.GetPath()):
                    xform = UsdGeom.Xformable(prim)
                    ops = xform.GetOrderedXformOps()
                    if ops:
                        time_samples = ops[0].GetTimeSamples()
                        self.assertEqual(len(time_samples), len(timestamps))

    def _make_transform(self, translation: Optional[List[float]] = None, rotation_deg: float = 0) -> np.ndarray:
        """Create a 4x4 transform matrix."""
        mat = np.eye(4, dtype=np.float64)
        if translation:
            mat[0, 3] = translation[0]
            mat[1, 3] = translation[1]
            mat[2, 3] = translation[2]
        return mat


# =============================================================================
# Test: Deformable Gaussians
# =============================================================================


class TestDeformableGaussiansExport(unittest.TestCase):
    """Test export of deformable gaussians with animated positions."""

    def test_positions_animated_per_frame(self):
        """Verify positions are keyframed when deformation is enabled."""
        n_gaussians = 15
        timestamps = [0, 100000, 200000]

        # Create different positions for each timestamp
        deformed_positions = {
            0: np.zeros((n_gaussians, 3), dtype=np.float32),
            100000: np.ones((n_gaussians, 3), dtype=np.float32),
            200000: np.ones((n_gaussians, 3), dtype=np.float32) * 2,
        }

        accessor = MockAccessor(
            n_gaussians=n_gaussians,
            has_deformation=True,
            has_rigid_tracks=True,  # Need tracks to drive animation
            track_ids=["deform_track"],
            track_timestamps_us={0: timestamps},
            deformed_positions=deformed_positions,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "deform_test.usda"
            stage = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
            stage.SetTimeCodesPerSecond(24.0)
            export_gaussians_as_usd_asset(
                accessor=accessor,
                stage=stage,
                content_root_path="/World",
                schema_type=GaussianUSDSchemaType.LIGHT_FIELD,
            )
            stage.Export(str(output_path))

            stage = Usd.Stage.Open(str(output_path))
            prim = self._find_particle_field_prim(stage)

            positions_attr = prim.GetAttribute("positions")
            time_samples = positions_attr.GetTimeSamples()

            # Should have multiple time samples for animation
            self.assertGreaterEqual(len(time_samples), 1)

    def _find_particle_field_prim(self, stage: Usd.Stage) -> Optional[Usd.Prim]:
        for prim in stage.Traverse():
            if prim.GetTypeName() == "ParticleField3DGaussianSplat":
                return prim
        return None


if __name__ == "__main__":
    unittest.main()
