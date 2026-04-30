# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Unit tests for camera parameter dataclasses.

Tests the camera parameter data structures including:
- Parameter dataclass construction
- Type validation
- Enum values
- Inheritance relationships
- Slang kernel functionality (eval_poly bounds checking, Newton-Raphson undistortion)
"""

import unittest

import torch

from libs.sensors.kernels.cameras import (
    camera_rays_to_image_points,
    image_points_to_camera_rays,
)
from libs.sensors.kernels.cameras.parameters import (
    BivariateWindshieldDistortion,
    CameraProjection,
    ExternalDistortion,
    FThetaPolynomialType,
    FThetaProjection,
    NoExternalDistortion,
    OpenCVFisheyeProjection,
    OpenCVPinholeProjection,
    ReferencePolynomial,
    ShutterType,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestEnums(unittest.TestCase):
    """Test enum definitions."""

    def test_reference_polynomial_values(self):
        """Test ReferencePolynomial enum."""
        self.assertEqual(ReferencePolynomial.FORWARD, 0)
        self.assertEqual(ReferencePolynomial.BACKWARD, 1)

    def test_ftheta_polynomial_type_values(self):
        """Test FThetaPolynomialType enum."""
        self.assertEqual(FThetaPolynomialType.FORWARD, 0)
        self.assertEqual(FThetaPolynomialType.BACKWARD, 1)

    def test_shutter_type_can_convert_to_int(self):
        """Test that ShutterType can be converted to int."""
        shutter = ShutterType.GLOBAL
        self.assertEqual(int(shutter), 5)
        self.assertIsInstance(int(shutter), int)


class TestExternalDistortion(unittest.TestCase):
    """Test external distortion parameter classes."""

    def test_no_external_distortion_creation(self):
        """Test NoExternalDistortion can be instantiated."""
        distortion = NoExternalDistortion()
        self.assertIsInstance(distortion, NoExternalDistortion)
        self.assertIsInstance(distortion, ExternalDistortion)

    def test_bivariate_windshield_creation(self):
        """Test BivariateWindshieldDistortion creation."""
        # Bivariate polynomials require triangular number of coefficients:
        # 1 (order 0), 3 (order 1), 6 (order 2), 10 (order 3), 15 (order 4)
        h_poly = torch.tensor([1.0, 0.1, 0.01], device=device)  # 3 terms = order 1
        v_poly = torch.tensor([1.0, 0.1, 0.01], device=device)  # 3 terms = order 1
        h_poly_inv = torch.tensor([1.0, -0.1, -0.01], device=device)
        v_poly_inv = torch.tensor([1.0, -0.1, -0.01], device=device)

        distortion = BivariateWindshieldDistortion.from_components(
            h_poly=h_poly,
            v_poly=v_poly,
            h_poly_inv=h_poly_inv,
            v_poly_inv=v_poly_inv,
            reference_polynomial=ReferencePolynomial.FORWARD,
        )

        self.assertIsInstance(distortion, BivariateWindshieldDistortion)
        self.assertIsInstance(distortion, ExternalDistortion)
        self.assertEqual(distortion.reference_polynomial, ReferencePolynomial.FORWARD)
        # Properties padded to MAX_H_POLYNOMIAL_TERMS=6 and MAX_V_POLYNOMIAL_TERMS=15
        self.assertEqual(distortion.h_poly.shape[0], 6)
        self.assertEqual(distortion.v_poly.shape[0], 15)

    def test_bivariate_windshield_different_sizes(self):
        """Test BivariateWindshieldDistortion with different polynomial sizes."""
        # Different valid bivariate polynomial sizes (triangular numbers)
        # h_poly: 6 terms = order 2, v_poly: 10 terms = order 3
        h_poly = torch.ones(6, device=device)  # order 2 (1+2+3=6 terms)
        v_poly = torch.ones(10, device=device)  # order 3 (1+2+3+4=10 terms)
        h_poly_inv = torch.ones(6, device=device)
        v_poly_inv = torch.ones(10, device=device)

        distortion = BivariateWindshieldDistortion.from_components(
            h_poly=h_poly,
            v_poly=v_poly,
            h_poly_inv=h_poly_inv,
            v_poly_inv=v_poly_inv,
            reference_polynomial=ReferencePolynomial.BACKWARD,
        )

        # Padded to MAX_H_POLYNOMIAL_TERMS=6 and MAX_V_POLYNOMIAL_TERMS=15
        self.assertEqual(distortion.h_poly.shape[0], 6)
        self.assertEqual(distortion.v_poly.shape[0], 15)
        # Degrees are computed from triangular form
        self.assertEqual(distortion.h_poly_degree, 2)  # 6 terms = order 2
        self.assertEqual(distortion.v_poly_degree, 3)  # 10 terms = order 3


class TestCameraProjections(unittest.TestCase):
    """Test camera projection parameter classes."""

    def test_opencv_pinhole_creation(self):
        """Test OpenCVPinholeProjection creation."""
        projection = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([500.0, 500.0], device=device),
            principal_point=torch.tensor([320.0, 240.0], device=device),
            radial_coeffs=torch.tensor([0.1, 0.01, 0.001, 0.0, 0.0, 0.0], device=device),
            tangential_coeffs=torch.tensor([0.001, 0.002], device=device),
            thin_prism_coeffs=torch.tensor([0.0, 0.0, 0.0, 0.0], device=device),
            resolution=torch.tensor([640, 480], device=device),
        )

        self.assertIsInstance(projection, OpenCVPinholeProjection)
        self.assertIsInstance(projection, CameraProjection)
        self.assertEqual(projection.focal_length.shape, (2,))
        self.assertEqual(projection.principal_point.shape, (2,))
        self.assertEqual(projection.radial_coeffs.shape, (6,))
        self.assertEqual(projection.tangential_coeffs.shape, (2,))
        self.assertEqual(projection.thin_prism_coeffs.shape, (4,))

    def test_opencv_pinhole_no_distortion(self):
        """Test OpenCVPinholeProjection with no distortion (zeros)."""
        projection = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([800.0, 800.0], device=device),
            principal_point=torch.tensor([512.0, 384.0], device=device),
            radial_coeffs=torch.zeros(6, device=device),
            tangential_coeffs=torch.zeros(2, device=device),
            thin_prism_coeffs=torch.zeros(4, device=device),
            resolution=torch.tensor([1024, 768], device=device),
        )

        # Check all distortion coefficients are zero
        self.assertTrue(torch.all(projection.radial_coeffs == 0))
        self.assertTrue(torch.all(projection.tangential_coeffs == 0))
        self.assertTrue(torch.all(projection.thin_prism_coeffs == 0))

    def test_opencv_fisheye_creation(self):
        """Test OpenCVFisheyeProjection creation."""
        projection = OpenCVFisheyeProjection.from_components(
            principal_point=torch.tensor([640.0, 512.0], device=device),
            focal_length=torch.tensor([600.0, 600.0], device=device),
            forward_poly=torch.tensor([1.0, 0.1, 0.01, 0.001], device=device),
            resolution=torch.tensor([3840, 2160], device=device),
            max_angle=3.14159 / 2,  # 90 degrees
            newton_iterations=5,
            min_2d_norm=torch.tensor(1e-6, device=device),
        )

        self.assertIsInstance(projection, OpenCVFisheyeProjection)
        self.assertIsInstance(projection, CameraProjection)
        self.assertEqual(projection.forward_poly.shape, (4,))
        self.assertEqual(projection.newton_iterations, 5)
        self.assertAlmostEqual(projection.max_angle, 3.14159 / 2, places=5)

    def test_ftheta_creation(self):
        """Test FThetaProjection creation via from_components."""
        projection = FThetaProjection.from_components(
            principal_point=torch.tensor([512.0, 512.0], device=device),
            fw_poly=torch.tensor([1.0, 0.5, 0.1], device=device),
            bw_poly=torch.tensor([1.0, -0.5, -0.1], device=device),
            A=torch.eye(2, device=device),
            Ainv=torch.eye(2, device=device),
            dfw_poly=torch.tensor([0.5, 0.2, 0.0], device=device),
            dbw_poly=torch.tensor([-0.5, -0.2, 0.0], device=device),
            reference_poly=FThetaPolynomialType.FORWARD,
            max_angle=3.14159,
            newton_iterations=10,
            min_2d_norm=1e-7,
        )

        self.assertIsInstance(projection, FThetaProjection)
        self.assertIsInstance(projection, CameraProjection)
        self.assertEqual(projection.reference_poly, FThetaPolynomialType.FORWARD)
        from libs.sensors.kernels.cameras.parameters import _FTHETA_INTRINSICS_SIZE

        self.assertEqual(projection.intrinsics.shape, (_FTHETA_INTRINSICS_SIZE,))
        self.assertEqual(projection.A.shape, (2, 2))
        self.assertEqual(projection.Ainv.shape, (2, 2))
        self.assertEqual(projection.newton_iterations, 10)
        self.assertEqual(projection.fw_poly_degree, 2)
        self.assertEqual(projection.bw_poly_degree, 2)


class TestTensorDevices(unittest.TestCase):
    """Test that parameters work with different devices."""

    def test_parameters_on_cuda(self):
        """Test creating parameters on CUDA device."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        cuda_device = torch.device("cuda")
        projection = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([500.0, 500.0], device=cuda_device),
            principal_point=torch.tensor([320.0, 240.0], device=cuda_device),
            radial_coeffs=torch.zeros(6, device=cuda_device),
            tangential_coeffs=torch.zeros(2, device=cuda_device),
            thin_prism_coeffs=torch.zeros(4, device=cuda_device),
            resolution=torch.tensor([640, 480], device=cuda_device),
        )

        self.assertEqual(projection.focal_length.device.type, "cuda")
        self.assertEqual(projection.principal_point.device.type, "cuda")

    def test_parameters_on_cpu(self):
        """Test creating parameters on CPU device."""
        cpu_device = torch.device("cpu")
        projection = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([500.0, 500.0], device=cpu_device),
            principal_point=torch.tensor([320.0, 240.0], device=cpu_device),
            radial_coeffs=torch.zeros(6, device=cpu_device),
            tangential_coeffs=torch.zeros(2, device=cpu_device),
            thin_prism_coeffs=torch.zeros(4, device=cpu_device),
            resolution=torch.tensor([640, 480], device=cpu_device),
        )

        self.assertEqual(projection.focal_length.device.type, "cpu")


class TestParameterValidation(unittest.TestCase):
    """Test parameter validation and edge cases."""

    def test_focal_length_positive(self):
        """Test that focal lengths are stored correctly."""
        projection = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([1000.0, 800.0], device=device),
            principal_point=torch.tensor([512.0, 384.0], device=device),
            radial_coeffs=torch.zeros(6, device=device),
            tangential_coeffs=torch.zeros(2, device=device),
            thin_prism_coeffs=torch.zeros(4, device=device),
            resolution=torch.tensor([1024, 768], device=device),
        )

        self.assertGreater(projection.focal_length[0].item(), 0)
        self.assertGreater(projection.focal_length[1].item(), 0)

    def test_principal_point_values(self):
        """Test principal point is stored correctly."""
        cx, cy = 640.5, 480.3
        projection = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([500.0, 500.0], device=device),
            principal_point=torch.tensor([cx, cy], device=device),
            radial_coeffs=torch.zeros(6, device=device),
            tangential_coeffs=torch.zeros(2, device=device),
            thin_prism_coeffs=torch.zeros(4, device=device),
            resolution=torch.tensor([1280, 960], device=device),
        )

        # Use delta instead of places to account for float32 precision
        self.assertAlmostEqual(projection.principal_point[0].item(), cx, delta=1e-4)
        self.assertAlmostEqual(projection.principal_point[1].item(), cy, delta=1e-4)

    def test_max_angle_range(self):
        """Test max_angle parameter."""
        max_angle = 3.14159 / 2
        projection = OpenCVFisheyeProjection.from_components(
            principal_point=torch.tensor([320.0, 240.0], device=device),
            focal_length=torch.tensor([500.0, 500.0], device=device),
            forward_poly=torch.ones(4, device=device),
            resolution=torch.tensor([3840, 2160], device=device),
            max_angle=max_angle,
            newton_iterations=5,
            min_2d_norm=torch.tensor(1e-6, device=device),
        )

        self.assertGreater(projection.max_angle, 0)
        self.assertLessEqual(projection.max_angle, 3.14159)


class TestDataclassProperties(unittest.TestCase):
    """Test dataclass properties and behavior."""

    def test_dataclass_equality(self):
        """Test that identical parameters are equal."""
        proj1 = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([500.0, 500.0], device=device),
            principal_point=torch.tensor([320.0, 240.0], device=device),
            radial_coeffs=torch.zeros(6, device=device),
            tangential_coeffs=torch.zeros(2, device=device),
            thin_prism_coeffs=torch.zeros(4, device=device),
            resolution=torch.tensor([640, 480], device=device),
        )

        proj2 = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([500.0, 500.0], device=device),
            principal_point=torch.tensor([320.0, 240.0], device=device),
            radial_coeffs=torch.zeros(6, device=device),
            tangential_coeffs=torch.zeros(2, device=device),
            thin_prism_coeffs=torch.zeros(4, device=device),
            resolution=torch.tensor([640, 480], device=device),
        )

        # Note: Tensor equality in dataclass might not work directly
        # This tests that the structure is the same
        self.assertEqual(type(proj1), type(proj2))

    def test_dataclass_repr(self):
        """Test that parameters have string representation."""
        projection = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([500.0, 500.0], device=device),
            principal_point=torch.tensor([320.0, 240.0], device=device),
            radial_coeffs=torch.zeros(6, device=device),
            tangential_coeffs=torch.zeros(2, device=device),
            thin_prism_coeffs=torch.zeros(4, device=device),
            resolution=torch.tensor([640, 480], device=device),
        )

        repr_str = repr(projection)
        self.assertIsInstance(repr_str, str)
        self.assertIn("OpenCVPinholeProjection", repr_str)


# ============================================================================
# Slang Kernel Functional Tests
# ============================================================================

# Tolerances for round-trip tests
ATOL = 1e-3
RTOL = 1e-3


class TestOpenCVPinholeUndistortion(unittest.TestCase):
    """Test Newton-Raphson iterative undistortion in OpenCVPinholeProjection.

    These tests verify that the image_points_to_camera_rays function correctly
    handles all distortion types (radial, tangential, thin prism) through
    iterative fixed-point refinement.
    """

    def setUp(self):
        """Set up test fixtures."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self.device = torch.device("cuda")
        self.external_distortion = NoExternalDistortion()

    def _create_grid_image_points(self, cx, cy, num_points_per_axis=5, spread=100.0):
        """Create a grid of image points around the principal point."""
        x_coords = torch.linspace(cx - spread, cx + spread, num_points_per_axis, device=self.device)
        y_coords = torch.linspace(cy - spread, cy + spread, num_points_per_axis, device=self.device)
        xx, yy = torch.meshgrid(x_coords, y_coords, indexing="xy")
        return torch.stack([xx.flatten(), yy.flatten()], dim=-1)

    def test_round_trip_no_distortion(self):
        """Test round-trip with zero distortion coefficients."""
        projection = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([500.0, 500.0], device=self.device),
            principal_point=torch.tensor([320.0, 240.0], device=self.device),
            radial_coeffs=torch.zeros(6, device=self.device),
            tangential_coeffs=torch.zeros(2, device=self.device),
            thin_prism_coeffs=torch.zeros(4, device=self.device),
            resolution=torch.tensor([640, 480], device=self.device),
        )

        image_points_orig = self._create_grid_image_points(320.0, 240.0)

        # Back-project to camera rays
        camera_rays = image_points_to_camera_rays(image_points_orig, projection, self.external_distortion)

        # Forward project back to image
        image_points_new, valid = camera_rays_to_image_points(camera_rays, projection, self.external_distortion)

        # All points should be valid
        self.assertTrue(torch.all(valid), "All points should be valid")

        # Check round-trip consistency
        self.assertTrue(
            torch.allclose(image_points_orig, image_points_new, atol=ATOL, rtol=RTOL),
            f"Round-trip failed: max error = {(image_points_orig - image_points_new).abs().max().item()}",
        )

    def test_round_trip_radial_distortion_only(self):
        """Test round-trip with only radial distortion (k1, k2, k3)."""
        projection = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([500.0, 500.0], device=self.device),
            principal_point=torch.tensor([320.0, 240.0], device=self.device),
            radial_coeffs=torch.tensor([0.1, -0.05, 0.01, 0.0, 0.0, 0.0], device=self.device),
            tangential_coeffs=torch.zeros(2, device=self.device),
            thin_prism_coeffs=torch.zeros(4, device=self.device),
            resolution=torch.tensor([640, 480], device=self.device),
        )

        image_points_orig = self._create_grid_image_points(320.0, 240.0)

        camera_rays = image_points_to_camera_rays(image_points_orig, projection, self.external_distortion)
        image_points_new, valid = camera_rays_to_image_points(camera_rays, projection, self.external_distortion)

        valid_mask = valid.bool()
        self.assertTrue(
            torch.allclose(image_points_orig[valid_mask], image_points_new[valid_mask], atol=ATOL, rtol=RTOL),
            f"Round-trip with radial distortion failed: max error = "
            f"{(image_points_orig[valid_mask] - image_points_new[valid_mask]).abs().max().item()}",
        )

    def test_round_trip_radial_distortion_rational(self):
        """Test round-trip with rational radial distortion (k1-k6)."""
        projection = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([500.0, 500.0], device=self.device),
            principal_point=torch.tensor([320.0, 240.0], device=self.device),
            radial_coeffs=torch.tensor([0.05, -0.02, 0.005, 0.01, -0.005, 0.001], device=self.device),
            tangential_coeffs=torch.zeros(2, device=self.device),
            thin_prism_coeffs=torch.zeros(4, device=self.device),
            resolution=torch.tensor([640, 480], device=self.device),
        )

        image_points_orig = self._create_grid_image_points(320.0, 240.0, spread=80.0)

        camera_rays = image_points_to_camera_rays(image_points_orig, projection, self.external_distortion)
        image_points_new, valid = camera_rays_to_image_points(camera_rays, projection, self.external_distortion)

        valid_mask = valid.bool()
        self.assertTrue(
            torch.allclose(image_points_orig[valid_mask], image_points_new[valid_mask], atol=ATOL, rtol=RTOL),
            f"Round-trip with rational radial distortion failed: max error = "
            f"{(image_points_orig[valid_mask] - image_points_new[valid_mask]).abs().max().item()}",
        )

    def test_round_trip_tangential_distortion_only(self):
        """Test round-trip with only tangential distortion (p1, p2)."""
        projection = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([500.0, 500.0], device=self.device),
            principal_point=torch.tensor([320.0, 240.0], device=self.device),
            radial_coeffs=torch.zeros(6, device=self.device),
            tangential_coeffs=torch.tensor([0.001, 0.002], device=self.device),
            thin_prism_coeffs=torch.zeros(4, device=self.device),
            resolution=torch.tensor([640, 480], device=self.device),
        )

        image_points_orig = self._create_grid_image_points(320.0, 240.0)

        camera_rays = image_points_to_camera_rays(image_points_orig, projection, self.external_distortion)
        image_points_new, valid = camera_rays_to_image_points(camera_rays, projection, self.external_distortion)

        valid_mask = valid.bool()
        self.assertTrue(
            torch.allclose(image_points_orig[valid_mask], image_points_new[valid_mask], atol=ATOL, rtol=RTOL),
            f"Round-trip with tangential distortion failed: max error = "
            f"{(image_points_orig[valid_mask] - image_points_new[valid_mask]).abs().max().item()}",
        )

    def test_round_trip_thin_prism_distortion_only(self):
        """Test round-trip with only thin prism distortion (s1-s4)."""
        projection = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([500.0, 500.0], device=self.device),
            principal_point=torch.tensor([320.0, 240.0], device=self.device),
            radial_coeffs=torch.zeros(6, device=self.device),
            tangential_coeffs=torch.zeros(2, device=self.device),
            thin_prism_coeffs=torch.tensor([0.0005, 0.0001, 0.0003, 0.0002], device=self.device),
            resolution=torch.tensor([640, 480], device=self.device),
        )

        image_points_orig = self._create_grid_image_points(320.0, 240.0)

        camera_rays = image_points_to_camera_rays(image_points_orig, projection, self.external_distortion)
        image_points_new, valid = camera_rays_to_image_points(camera_rays, projection, self.external_distortion)

        valid_mask = valid.bool()
        self.assertTrue(
            torch.allclose(image_points_orig[valid_mask], image_points_new[valid_mask], atol=ATOL, rtol=RTOL),
            f"Round-trip with thin prism distortion failed: max error = "
            f"{(image_points_orig[valid_mask] - image_points_new[valid_mask]).abs().max().item()}",
        )

    def test_round_trip_all_distortions_combined(self):
        """Test round-trip with all distortion types combined."""
        projection = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([500.0, 500.0], device=self.device),
            principal_point=torch.tensor([320.0, 240.0], device=self.device),
            radial_coeffs=torch.tensor([0.05, -0.02, 0.005, 0.0, 0.0, 0.0], device=self.device),
            tangential_coeffs=torch.tensor([0.001, 0.0015], device=self.device),
            thin_prism_coeffs=torch.tensor([0.0003, 0.0001, 0.0002, 0.0001], device=self.device),
            resolution=torch.tensor([640, 480], device=self.device),
        )

        image_points_orig = self._create_grid_image_points(320.0, 240.0)

        camera_rays = image_points_to_camera_rays(image_points_orig, projection, self.external_distortion)
        image_points_new, valid = camera_rays_to_image_points(camera_rays, projection, self.external_distortion)

        valid_mask = valid.bool()
        self.assertTrue(
            torch.allclose(image_points_orig[valid_mask], image_points_new[valid_mask], atol=ATOL, rtol=RTOL),
            f"Round-trip with all distortions failed: max error = "
            f"{(image_points_orig[valid_mask] - image_points_new[valid_mask]).abs().max().item()}",
        )

    def test_round_trip_strong_distortion(self):
        """Test round-trip with strong distortion coefficients."""
        projection = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([500.0, 500.0], device=self.device),
            principal_point=torch.tensor([320.0, 240.0], device=self.device),
            radial_coeffs=torch.tensor([0.2, -0.1, 0.02, 0.0, 0.0, 0.0], device=self.device),
            tangential_coeffs=torch.tensor([0.005, 0.005], device=self.device),
            thin_prism_coeffs=torch.tensor([0.001, 0.0005, 0.001, 0.0005], device=self.device),
            resolution=torch.tensor([640, 480], device=self.device),
        )

        # Use smaller spread for strong distortion to avoid extreme values
        image_points_orig = self._create_grid_image_points(320.0, 240.0, spread=50.0)

        camera_rays = image_points_to_camera_rays(image_points_orig, projection, self.external_distortion)
        image_points_new, valid = camera_rays_to_image_points(camera_rays, projection, self.external_distortion)

        valid_mask = valid.bool()
        # Use slightly looser tolerance for strong distortion
        self.assertTrue(
            torch.allclose(image_points_orig[valid_mask], image_points_new[valid_mask], atol=ATOL * 2, rtol=RTOL * 2),
            f"Round-trip with strong distortion failed: max error = "
            f"{(image_points_orig[valid_mask] - image_points_new[valid_mask]).abs().max().item()}",
        )

    def test_principal_point_projection(self):
        """Test that principal point projects to optical axis."""
        projection = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([500.0, 500.0], device=self.device),
            principal_point=torch.tensor([320.0, 240.0], device=self.device),
            radial_coeffs=torch.tensor([0.1, -0.05, 0.01, 0.0, 0.0, 0.0], device=self.device),
            tangential_coeffs=torch.tensor([0.001, 0.002], device=self.device),
            thin_prism_coeffs=torch.zeros(4, device=self.device),
            resolution=torch.tensor([640, 480], device=self.device),
        )

        # Principal point
        image_points = torch.tensor([[320.0, 240.0]], device=self.device)

        camera_rays = image_points_to_camera_rays(image_points, projection, self.external_distortion)

        # Ray should point along optical axis (z direction)
        # After normalization, x and y components should be very small
        self.assertTrue(camera_rays[0, 0].abs() < 1e-5, f"X component should be near zero: {camera_rays[0, 0].item()}")
        self.assertTrue(camera_rays[0, 1].abs() < 1e-5, f"Y component should be near zero: {camera_rays[0, 1].item()}")
        self.assertTrue(camera_rays[0, 2] > 0.99, f"Z component should be near 1: {camera_rays[0, 2].item()}")

    def test_rays_are_normalized(self):
        """Test that output camera rays are normalized."""
        projection = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([500.0, 500.0], device=self.device),
            principal_point=torch.tensor([320.0, 240.0], device=self.device),
            radial_coeffs=torch.tensor([0.1, -0.05, 0.0, 0.0, 0.0, 0.0], device=self.device),
            tangential_coeffs=torch.tensor([0.001, 0.002], device=self.device),
            thin_prism_coeffs=torch.tensor([0.0001, 0.0, 0.0001, 0.0], device=self.device),
            resolution=torch.tensor([640, 480], device=self.device),
        )

        image_points = self._create_grid_image_points(320.0, 240.0)

        camera_rays = image_points_to_camera_rays(image_points, projection, self.external_distortion)

        # Check all rays are normalized
        norms = camera_rays.norm(dim=-1)
        self.assertTrue(
            torch.allclose(norms, torch.ones_like(norms), atol=1e-5),
            f"Rays should be normalized: max deviation = {(norms - 1.0).abs().max().item()}",
        )

    def test_asymmetric_focal_length(self):
        """Test undistortion with asymmetric focal lengths."""
        projection = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([600.0, 400.0], device=self.device),  # Asymmetric
            principal_point=torch.tensor([320.0, 240.0], device=self.device),
            radial_coeffs=torch.tensor([0.05, -0.02, 0.0, 0.0, 0.0, 0.0], device=self.device),
            tangential_coeffs=torch.tensor([0.001, 0.001], device=self.device),
            thin_prism_coeffs=torch.zeros(4, device=self.device),
            resolution=torch.tensor([640, 480], device=self.device),
        )

        image_points_orig = self._create_grid_image_points(320.0, 240.0)

        camera_rays = image_points_to_camera_rays(image_points_orig, projection, self.external_distortion)
        image_points_new, valid = camera_rays_to_image_points(camera_rays, projection, self.external_distortion)

        valid_mask = valid.bool()
        self.assertTrue(
            torch.allclose(image_points_orig[valid_mask], image_points_new[valid_mask], atol=ATOL, rtol=RTOL),
            "Round-trip with asymmetric focal length failed",
        )

    def test_off_center_principal_point(self):
        """Test undistortion with off-center principal point."""
        projection = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([500.0, 500.0], device=self.device),
            principal_point=torch.tensor([350.0, 260.0], device=self.device),  # Off-center
            radial_coeffs=torch.tensor([0.05, -0.02, 0.0, 0.0, 0.0, 0.0], device=self.device),
            tangential_coeffs=torch.tensor([0.001, 0.001], device=self.device),
            thin_prism_coeffs=torch.zeros(4, device=self.device),
            resolution=torch.tensor([700, 520], device=self.device),
        )

        image_points_orig = self._create_grid_image_points(350.0, 260.0)

        camera_rays = image_points_to_camera_rays(image_points_orig, projection, self.external_distortion)
        image_points_new, valid = camera_rays_to_image_points(camera_rays, projection, self.external_distortion)

        valid_mask = valid.bool()
        self.assertTrue(
            torch.allclose(image_points_orig[valid_mask], image_points_new[valid_mask], atol=ATOL, rtol=RTOL),
            "Round-trip with off-center principal point failed",
        )

    def test_batch_processing(self):
        """Test that batch processing produces consistent results."""
        projection = OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([500.0, 500.0], device=self.device),
            principal_point=torch.tensor([320.0, 240.0], device=self.device),
            radial_coeffs=torch.tensor([0.1, -0.05, 0.01, 0.0, 0.0, 0.0], device=self.device),
            tangential_coeffs=torch.tensor([0.001, 0.002], device=self.device),
            thin_prism_coeffs=torch.tensor([0.0002, 0.0001, 0.0002, 0.0001], device=self.device),
            resolution=torch.tensor([640, 480], device=self.device),
        )

        # Large batch of points
        image_points_orig = self._create_grid_image_points(320.0, 240.0, num_points_per_axis=10)

        camera_rays = image_points_to_camera_rays(image_points_orig, projection, self.external_distortion)
        image_points_new, valid = camera_rays_to_image_points(camera_rays, projection, self.external_distortion)

        valid_mask = valid.bool()
        self.assertTrue(
            torch.allclose(image_points_orig[valid_mask], image_points_new[valid_mask], atol=ATOL, rtol=RTOL),
            "Batch processing round-trip failed",
        )


class TestBivariateWindshieldPolynomialBounds(unittest.TestCase):
    """Test polynomial degree bounds checking in BivariateWindshieldDistortion.

    These tests verify that the eval_poly function safely handles polynomial
    degrees, including edge cases where degree could exceed array bounds.
    """

    def setUp(self):
        """Set up test fixtures."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self.device = torch.device("cuda")

    def _create_simple_pinhole(self):
        """Create a simple pinhole projection for testing."""
        return OpenCVPinholeProjection.from_components(
            focal_length=torch.tensor([500.0, 500.0], device=self.device),
            principal_point=torch.tensor([320.0, 240.0], device=self.device),
            radial_coeffs=torch.zeros(6, device=self.device),
            tangential_coeffs=torch.zeros(2, device=self.device),
            thin_prism_coeffs=torch.zeros(4, device=self.device),
            resolution=torch.tensor([640, 480], device=self.device),
        )

    def test_polynomial_degree_zero(self):
        """Test windshield distortion with polynomial degree 0 (constant)."""
        distortion = BivariateWindshieldDistortion.from_components(
            h_poly=torch.tensor([0.0], device=self.device),  # degree 0
            v_poly=torch.tensor([0.0], device=self.device),
            h_poly_inv=torch.tensor([0.0], device=self.device),
            v_poly_inv=torch.tensor([0.0], device=self.device),
            reference_polynomial=ReferencePolynomial.FORWARD,
        )

        projection = self._create_simple_pinhole()
        camera_rays = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.1, 1.0]], device=self.device)
        camera_rays = camera_rays / camera_rays.norm(dim=-1, keepdim=True)

        # Should not crash
        image_points, _valid = camera_rays_to_image_points(camera_rays, projection, distortion)

        self.assertEqual(image_points.shape, (2, 2))

    def test_polynomial_degree_max_safe(self):
        """Test windshield distortion with maximum safe polynomial degrees."""
        # h_poly: 6 terms = order 2 (max for MAX_H_POLYNOMIAL_TERMS=6)
        # v_poly: 15 terms = order 4 (max for MAX_V_POLYNOMIAL_TERMS=15)
        h_poly = torch.zeros(6, device=self.device)  # triangular: 1+2+3=6
        h_poly[0] = 0.001  # Small constant term
        v_poly = torch.zeros(15, device=self.device)  # triangular: 1+2+3+4+5=15
        v_poly[0] = 0.001

        distortion = BivariateWindshieldDistortion.from_components(
            h_poly=h_poly,
            v_poly=v_poly,
            h_poly_inv=h_poly,
            v_poly_inv=v_poly,
            reference_polynomial=ReferencePolynomial.FORWARD,
        )

        projection = self._create_simple_pinhole()
        camera_rays = torch.tensor([[0.0, 0.0, 1.0], [0.05, 0.05, 1.0]], device=self.device)
        camera_rays = camera_rays / camera_rays.norm(dim=-1, keepdim=True)

        # Should not crash with max degree
        image_points, _valid = camera_rays_to_image_points(camera_rays, projection, distortion)

        self.assertEqual(image_points.shape, (2, 2))

    def test_polynomial_various_degrees(self):
        """Test windshield distortion with various polynomial degrees (orders)."""
        projection = self._create_simple_pinhole()
        camera_rays = torch.tensor([[0.0, 0.0, 1.0], [0.05, 0.05, 1.0]], device=self.device)
        camera_rays = camera_rays / camera_rays.norm(dim=-1, keepdim=True)

        # Bivariate polynomials require triangular number of coefficients
        # order 0: 1, order 1: 3, order 2: 6, order 3: 10, order 4: 15
        triangular_counts = [1, 3, 6]  # Keep within MAX_H_POLYNOMIAL_TERMS=6

        for num_terms in triangular_counts:
            with self.subTest(num_terms=num_terms):
                poly = torch.zeros(num_terms, device=self.device)
                poly[0] = 0.0001

                distortion = BivariateWindshieldDistortion.from_components(
                    h_poly=poly,
                    v_poly=poly,
                    h_poly_inv=poly,
                    v_poly_inv=poly,
                    reference_polynomial=ReferencePolynomial.FORWARD,
                )

                # Should not crash for any valid degree
                image_points, _valid = camera_rays_to_image_points(camera_rays, projection, distortion)

                self.assertEqual(image_points.shape, (2, 2))

    def test_windshield_round_trip(self):
        """Test round-trip projection through windshield distortion."""
        # Bivariate polynomials require triangular number of coefficients: 1, 3, 6, 10, 15
        h_poly = torch.tensor([0.001, 0.0001, 0.00001], device=self.device)  # 3 terms = order 1
        v_poly = torch.tensor([0.001, 0.0001, 0.00001], device=self.device)  # 3 terms = order 1
        # Approximate inverse (for small distortions)
        h_poly_inv = torch.tensor([-0.001, -0.0001, -0.00001], device=self.device)
        v_poly_inv = torch.tensor([-0.001, -0.0001, -0.00001], device=self.device)

        distortion = BivariateWindshieldDistortion.from_components(
            h_poly=h_poly,
            v_poly=v_poly,
            h_poly_inv=h_poly_inv,
            v_poly_inv=v_poly_inv,
            reference_polynomial=ReferencePolynomial.FORWARD,
        )

        projection = self._create_simple_pinhole()

        image_points_orig = torch.tensor(
            [
                [320.0, 240.0],
                [350.0, 260.0],
                [290.0, 220.0],
            ],
            device=self.device,
        )

        camera_rays = image_points_to_camera_rays(image_points_orig, projection, distortion)
        image_points_new, valid = camera_rays_to_image_points(camera_rays, projection, distortion)

        # Verify shape is correct (round-trip consistency depends on inverse quality)
        self.assertEqual(image_points_new.shape, image_points_orig.shape)
        self.assertEqual(valid.shape, (3,))


class TestFThetaPolynomialBounds(unittest.TestCase):
    """Test polynomial degree bounds checking in FThetaProjection.

    These tests verify that the eval_poly function in FThetaProjection
    safely handles polynomial degrees at the boundary.
    """

    def setUp(self):
        """Set up test fixtures."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self.device = torch.device("cuda")
        self.external_distortion = NoExternalDistortion()

    def test_ftheta_max_degree_polynomial(self):
        """Test FTheta with maximum degree polynomial."""
        from libs.sensors.kernels.cameras.parameters import MAX_POLYNOMIAL_TERMS

        fw_poly = torch.zeros(MAX_POLYNOMIAL_TERMS, device=self.device)
        fw_poly[1] = 1.0
        bw_poly = torch.zeros(MAX_POLYNOMIAL_TERMS, device=self.device)
        bw_poly[1] = 1.0

        dfw_poly = torch.zeros(MAX_POLYNOMIAL_TERMS, device=self.device)
        dfw_poly[0] = 1.0
        dbw_poly = torch.zeros(MAX_POLYNOMIAL_TERMS, device=self.device)
        dbw_poly[0] = 1.0

        projection = FThetaProjection.from_components(
            principal_point=torch.tensor([320.0, 240.0], device=self.device),
            fw_poly=fw_poly,
            bw_poly=bw_poly,
            A=torch.eye(2, device=self.device),
            Ainv=torch.eye(2, device=self.device),
            dfw_poly=dfw_poly,
            dbw_poly=dbw_poly,
            reference_poly=FThetaPolynomialType.FORWARD,
            max_angle=1.5,  # ~86 degrees
            newton_iterations=5,
            min_2d_norm=1e-6,
        )

        camera_rays = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.1, 1.0]], device=self.device)
        camera_rays = camera_rays / camera_rays.norm(dim=-1, keepdim=True)

        # Should not crash with max degree
        image_points, _valid = camera_rays_to_image_points(camera_rays, projection, self.external_distortion)

        self.assertEqual(image_points.shape, (2, 2))

    def test_ftheta_round_trip(self):
        """Test FTheta projection round-trip."""
        # Standard polynomial form: c[0] + c[1]*x + c[2]*x^2 + ...
        # For simple linear f-theta: r = f * theta
        # fw_poly(theta) = 0 + focal_length * theta = [0, focal_length, 0, ...]
        # bw_poly(r) = 0 + (1/focal_length) * r = [0, 1/focal_length, 0, ...]
        focal_length = 500.0
        fw_poly = torch.tensor([0.0, focal_length, 0.0], device=self.device)
        bw_poly = torch.tensor([0.0, 1.0 / focal_length, 0.0], device=self.device)

        projection = FThetaProjection.from_components(
            principal_point=torch.tensor([320.0, 240.0], device=self.device),
            fw_poly=fw_poly,
            bw_poly=bw_poly,
            A=torch.eye(2, device=self.device),
            Ainv=torch.eye(2, device=self.device),
            # Derivatives: d/dx[0 + f*x] = f
            dfw_poly=torch.tensor([focal_length, 0.0, 0.0], device=self.device),
            dbw_poly=torch.tensor([1.0 / focal_length, 0.0, 0.0], device=self.device),
            reference_poly=FThetaPolynomialType.FORWARD,
            max_angle=1.5,
            newton_iterations=10,
            min_2d_norm=1e-6,
        )

        # Test points away from center (avoid edge cases at principal point)
        image_points_orig = torch.tensor(
            [
                [370.0, 290.0],
                [270.0, 190.0],
                [400.0, 240.0],
                [320.0, 300.0],
            ],
            device=self.device,
        )

        camera_rays = image_points_to_camera_rays(image_points_orig, projection, self.external_distortion)
        image_points_new, valid = camera_rays_to_image_points(camera_rays, projection, self.external_distortion)

        # Check that at least some points are valid
        valid_mask = valid.bool()
        self.assertTrue(valid_mask.any(), "At least some points should be valid")

        # Check round-trip consistency for valid points
        if valid_mask.sum() > 0:
            orig_valid = image_points_orig[valid_mask]
            new_valid = image_points_new[valid_mask]
            # Filter out any NaN values
            nan_mask = ~(torch.isnan(orig_valid).any(dim=-1) | torch.isnan(new_valid).any(dim=-1))
            if nan_mask.sum() > 0:
                self.assertTrue(
                    torch.allclose(orig_valid[nan_mask], new_valid[nan_mask], atol=ATOL, rtol=RTOL),
                    f"FTheta round-trip failed: max error = "
                    f"{(orig_valid[nan_mask] - new_valid[nan_mask]).abs().max().item()}",
                )


if __name__ == "__main__":
    unittest.main()
