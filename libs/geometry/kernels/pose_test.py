# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import unittest

import torch

from libs.geometry.kernels.pose import (
    frame_transform_poses_tquat,
    se3pose_from_matrix,
    se3pose_inverse_transform_direction,
    se3pose_inverse_transform_point,
    se3pose_to_inverse_matrix,
    se3pose_to_matrix,
    se3pose_transform_direction,
    se3pose_transform_point,
    trajectory_get_rotation_2poses,
    trajectory_transform_point_1pose,
    trajectory_transform_point_2poses,
)
from libs.geometry.kernels.quaternion import (
    quat_conjugate,
    quat_multiply,
    quat_rotate_vector,
    quat_slerp,
    quat_to_matrix,
)


device = torch.device("cuda")
# Deterministic random for reproducibility
torch.manual_seed(42)


class TestSE3Pose(unittest.TestCase):
    """Tests for SE3 pose operations using compiled Slang kernels."""

    def test_se3_transform_point(self):
        """Test SE3 point transformation - should apply rotation then translation"""
        N = 10000

        # Random poses and points (use CUDA tensors)
        translations = torch.randn(N, 3, device=device)
        rotations = torch.nn.functional.normalize(torch.randn(N, 4, device=device))
        points = torch.randn(N, 3, device=device)

        # Compute using Slang kernel
        output = se3pose_transform_point(translations, rotations, points)

        # Compute ground truth: T * p = R(p) + t
        rotated = quat_rotate_vector(rotations, points)
        expected = rotated + translations

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_se3_transform_direction(self):
        """Test SE3 direction transformation - should only apply rotation, not translation"""
        N = 10000

        # Random poses and directions
        translations = torch.randn(N, 3, device=device)
        rotations = torch.nn.functional.normalize(torch.randn(N, 4, device=device))
        directions = torch.randn(N, 3, device=device)

        # Compute using Slang kernel
        output = se3pose_transform_direction(translations, rotations, directions)

        # Compute ground truth: should only rotate, ignore translation
        expected = quat_rotate_vector(rotations, directions)

        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_se3_inverse_transform_point(self):
        """Test SE3 inverse point transformation - should invert the transformation"""
        N = 10000

        # Random poses and points
        translations = torch.randn(N, 3, device=device)
        rotations = torch.nn.functional.normalize(torch.randn(N, 4, device=device))
        points = torch.randn(N, 3, device=device)

        # Forward then inverse transform
        forward = se3pose_transform_point(translations, rotations, points)
        inverse = se3pose_inverse_transform_point(translations, rotations, forward)

        # Should recover original points
        self.assertTrue(torch.allclose(inverse, points, atol=1e-5))

    def test_se3_inverse_transform_direction(self):
        """Test SE3 inverse direction transformation - should invert rotation only"""
        N = 10000

        # Random poses and directions
        translations = torch.randn(N, 3, device=device)
        rotations = torch.nn.functional.normalize(torch.randn(N, 4, device=device))
        directions = torch.randn(N, 3, device=device)

        # Forward then inverse transform
        forward = se3pose_transform_direction(translations, rotations, directions)
        inverse = se3pose_inverse_transform_direction(translations, rotations, forward)

        # Should recover original directions
        self.assertTrue(torch.allclose(inverse, directions, atol=1e-5))

    def test_se3_to_matrix(self):
        """Test SE3 pose to 4x4 matrix conversion - should produce correct transformation matrix"""
        N = 10000

        # Random poses
        translations = torch.randn(N, 3, device=device)
        rotations = torch.nn.functional.normalize(torch.randn(N, 4, device=device))

        # Compute using Slang kernel
        matrices = se3pose_to_matrix(translations, rotations)

        # Verify structure
        # Upper-left 3x3 should be rotation matrix
        rot_matrices = quat_to_matrix(rotations)  # (N, 9)
        rot_matrices = rot_matrices.view(N, 3, 3)

        self.assertTrue(torch.allclose(matrices[:, :3, :3], rot_matrices, atol=1e-6))

        # Fourth column should be [t; 1]
        self.assertTrue(torch.allclose(matrices[:, :3, 3], translations, atol=1e-6))
        self.assertTrue(torch.allclose(matrices[:, 3, 3], torch.ones(N, device=device), atol=1e-6))

        # Bottom row (except last element) should be zeros
        self.assertTrue(torch.allclose(matrices[:, 3, :3], torch.zeros(N, 3, device=device), atol=1e-6))

    def test_se3_from_matrix(self):
        """Test SE3 matrix to pose conversion - should extract translation and rotation correctly"""
        N = 10000

        # Random poses
        translations = torch.randn(N, 3, device=device)
        rotations = torch.nn.functional.normalize(torch.randn(N, 4, device=device))

        # Convert to matrices
        matrices = se3pose_to_matrix(translations, rotations)

        # Convert back to translation and rotation
        extracted_trans, extracted_rot = se3pose_from_matrix(matrices)

        # Check translation matches exactly
        self.assertTrue(torch.allclose(extracted_trans, translations, atol=1e-6))

        # Check rotation correctness: verify rotations produce same transformation
        # by rotating test vectors and comparing results
        # Note: Quaternions q and -q represent the same rotation, so we can't check
        # quaternion values directly. We must check the actual rotation effect.
        test_vectors = torch.randn(N, 3, device=device)
        rotated_original = quat_rotate_vector(rotations, test_vectors)
        rotated_extracted = quat_rotate_vector(extracted_rot, test_vectors)

        self.assertTrue(torch.allclose(rotated_original, rotated_extracted, atol=1e-5))

    def test_se3_from_matrix_round_trip(self):
        """Test round-trip conversion: pose → matrix → pose should recover original pose"""
        N = 10000

        # Random poses
        translations = torch.randn(N, 3, device=device)
        rotations = torch.nn.functional.normalize(torch.randn(N, 4, device=device))

        # Round-trip: pose → matrix → pose
        matrices = se3pose_to_matrix(translations, rotations)
        extracted_trans, extracted_rot = se3pose_from_matrix(matrices)

        # Test that extracted pose produces same transformations as original
        test_points = torch.randn(N, 3, device=device)

        transformed_original = se3pose_transform_point(translations, rotations, test_points)
        transformed_extracted = se3pose_transform_point(extracted_trans, extracted_rot, test_points)

        self.assertTrue(torch.allclose(transformed_original, transformed_extracted, atol=1e-5))

    def test_se3_from_matrix_backward(self):
        """Test that gradients flow correctly through matrix to pose conversion"""
        N = 1000

        # Random matrices with gradients enabled
        translations = torch.randn(N, 3, device=device, requires_grad=True)
        rotations = torch.nn.functional.normalize(torch.randn(N, 4, device=device, requires_grad=True))
        rotations.retain_grad()

        # Create matrices
        matrices = se3pose_to_matrix(translations, rotations)

        # Extract pose (this is what we're testing gradients for)
        extracted_trans, extracted_rot = se3pose_from_matrix(matrices)

        # Create a loss from extracted values
        loss = extracted_trans.sum() + extracted_rot.sum()

        # Backward pass
        loss.backward()

        # Check that gradients exist and are non-zero
        self.assertIsNotNone(translations.grad)
        self.assertIsNotNone(rotations.grad)
        self.assertTrue(torch.any(translations.grad != 0))
        self.assertTrue(torch.any(rotations.grad != 0))

    def test_se3_to_inverse_matrix(self):
        """Test SE3 pose to inverse 4x4 matrix conversion - should produce correct inverse transformation matrix"""
        N = 10000

        # Random poses
        translations = torch.randn(N, 3, device=device)
        rotations = torch.nn.functional.normalize(torch.randn(N, 4, device=device))

        # Compute using Slang kernel
        inverse_matrices = se3pose_to_inverse_matrix(translations, rotations)

        # Get forward matrices
        forward_matrices = se3pose_to_matrix(translations, rotations)

        # Multiply forward and inverse - should get identity
        identity = torch.bmm(forward_matrices, inverse_matrices)
        expected_identity = torch.eye(4, device=device).unsqueeze(0).repeat(N, 1, 1)

        self.assertTrue(torch.allclose(identity, expected_identity, atol=1e-5))

        # Also check that inverse * forward = identity
        identity2 = torch.bmm(inverse_matrices, forward_matrices)
        self.assertTrue(torch.allclose(identity2, expected_identity, atol=1e-5))

    def test_se3_to_inverse_matrix_structure(self):
        """Test SE3 inverse matrix has correct structure - R^T in upper-left, -R^T*t in translation"""
        N = 10000

        # Random poses
        translations = torch.randn(N, 3, device=device)
        rotations = torch.nn.functional.normalize(torch.randn(N, 4, device=device))

        # Compute inverse matrix using Slang kernel
        inverse_matrices = se3pose_to_inverse_matrix(translations, rotations)

        # Get rotation matrix and its transpose
        rot_matrices = quat_to_matrix(rotations).view(N, 3, 3)
        rot_matrices_T = rot_matrices.transpose(1, 2)

        # Upper-left 3x3 should be R^T
        self.assertTrue(torch.allclose(inverse_matrices[:, :3, :3], rot_matrices_T, atol=1e-6))

        # Translation part should be -R^T * t
        expected_trans = -torch.bmm(rot_matrices_T, translations.unsqueeze(-1)).squeeze(-1)
        self.assertTrue(torch.allclose(inverse_matrices[:, :3, 3], expected_trans, atol=1e-6))

        # Bottom row should be [0, 0, 0, 1]
        self.assertTrue(torch.allclose(inverse_matrices[:, 3, :3], torch.zeros(N, 3, device=device), atol=1e-6))
        self.assertTrue(torch.allclose(inverse_matrices[:, 3, 3], torch.ones(N, device=device), atol=1e-6))

    def test_se3_to_inverse_matrix_wxyz_format(self):
        """Test SE3 inverse matrix with wxyz quaternion format"""
        N = 10000

        # Random poses in xyzw format
        translations = torch.randn(N, 3, device=device)
        rotations_xyzw = torch.nn.functional.normalize(torch.randn(N, 4, device=device))

        # Convert to wxyz format (swap first and last components)
        rotations_wxyz = torch.cat([rotations_xyzw[:, 3:4], rotations_xyzw[:, :3]], dim=1)

        # Compute inverse matrix using wxyz format flag
        inverse_matrices_wxyz = se3pose_to_inverse_matrix(translations, rotations_wxyz, wxyz_format=True)

        # Compute inverse matrix using xyzw format (default)
        inverse_matrices_xyzw = se3pose_to_inverse_matrix(translations, rotations_xyzw, wxyz_format=False)

        # Both should produce the same result
        self.assertTrue(torch.allclose(inverse_matrices_wxyz, inverse_matrices_xyzw, atol=1e-6))

    def test_se3_to_inverse_matrix_transform_point(self):
        """Test that inverse matrix correctly inverse-transforms points"""
        N = 10000

        # Random poses and points
        translations = torch.randn(N, 3, device=device)
        rotations = torch.nn.functional.normalize(torch.randn(N, 4, device=device))
        points = torch.randn(N, 3, device=device)

        # Forward transform using SE3 function
        transformed_points = se3pose_transform_point(translations, rotations, points)

        # Get inverse matrix
        inverse_matrices = se3pose_to_inverse_matrix(translations, rotations)

        # Apply inverse matrix to transformed points
        points_homogeneous = torch.cat([transformed_points, torch.ones(N, 1, device=device)], dim=1)
        recovered_points = torch.bmm(inverse_matrices, points_homogeneous.unsqueeze(-1)).squeeze(-1)[:, :3]

        # Should recover original points
        self.assertTrue(torch.allclose(recovered_points, points, atol=1e-5))

    def test_se3_to_inverse_matrix_backward(self):
        """Test that gradients flow correctly through SE3 to inverse matrix conversion"""
        N = 1000

        # Random poses with gradients enabled
        translations = torch.randn(N, 3, device=device, requires_grad=True)
        rotations = torch.nn.functional.normalize(torch.randn(N, 4, device=device, requires_grad=True))
        rotations.retain_grad()  # Need this for non-leaf tensors

        # Forward pass
        inverse_matrices = se3pose_to_inverse_matrix(translations, rotations)

        # Backward pass
        grad_output = torch.randn_like(inverse_matrices)
        inverse_matrices.backward(grad_output)

        # Check that gradients exist and are non-zero
        self.assertIsNotNone(translations.grad)
        self.assertIsNotNone(rotations.grad)
        self.assertTrue(torch.any(translations.grad != 0))
        self.assertTrue(torch.any(rotations.grad != 0))

    def test_se3_to_inverse_matrix_backward_wxyz(self):
        """Test that gradients flow correctly with wxyz format"""
        N = 1000

        # Random poses with gradients enabled (in wxyz format)
        translations = torch.randn(N, 3, device=device, requires_grad=True)
        rotations = torch.nn.functional.normalize(torch.randn(N, 4, device=device, requires_grad=True))
        rotations.retain_grad()  # Need this for non-leaf tensors

        # Forward pass with wxyz format
        inverse_matrices = se3pose_to_inverse_matrix(translations, rotations, wxyz_format=True)

        # Backward pass
        grad_output = torch.randn_like(inverse_matrices)
        inverse_matrices.backward(grad_output)

        # Check that gradients exist and are non-zero
        self.assertIsNotNone(translations.grad)
        self.assertIsNotNone(rotations.grad)
        self.assertTrue(torch.any(translations.grad != 0))
        self.assertTrue(torch.any(rotations.grad != 0))

    def test_se3_identity(self):
        """Test that identity pose doesn't change points"""
        N = 10000

        # Identity pose
        translations = torch.zeros(N, 3, device=device)
        rotations = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).repeat(N, 1)
        points = torch.randn(N, 3, device=device)

        # Transform should be identity
        output = se3pose_transform_point(translations, rotations, points)

        self.assertTrue(torch.allclose(output, points, atol=1e-6))

    def test_se3_composition_via_matrix(self):
        """Test that composing poses works correctly via matrix multiplication"""
        N = 5000

        # Two random poses
        t1 = torch.randn(N, 3, device=device)
        r1 = torch.nn.functional.normalize(torch.randn(N, 4, device=device))
        t2 = torch.randn(N, 3, device=device)
        r2 = torch.nn.functional.normalize(torch.randn(N, 4, device=device))

        # Get matrices and compose
        m1 = se3pose_to_matrix(t1, r1)
        m2 = se3pose_to_matrix(t2, r2)

        # Compose via matrix multiplication
        m_composed = torch.bmm(m1, m2)

        # Test on random points
        points = torch.randn(N, 3, device=device)
        points_homogeneous = torch.cat([points, torch.ones(N, 1, device=device)], dim=1)

        # Apply composed transform
        result_matrix = torch.bmm(m_composed, points_homogeneous.unsqueeze(-1)).squeeze(-1)[:, :3]

        # Apply transforms sequentially using Slang kernels
        intermediate = se3pose_transform_point(t2, r2, points)
        result_sequential = se3pose_transform_point(t1, r1, intermediate)

        self.assertTrue(torch.allclose(result_matrix, result_sequential, atol=1e-5))

    def test_se3_transform_point_backward(self):
        """Test that gradients flow correctly through SE3 point transformation"""
        N = 1000

        # Random poses and points with gradients enabled
        translations = torch.randn(N, 3, device=device, requires_grad=True)
        rotations = torch.nn.functional.normalize(torch.randn(N, 4, device=device, requires_grad=True))
        rotations.retain_grad()  # Need this for non-leaf tensors
        points = torch.randn(N, 3, device=device, requires_grad=True)

        # Forward pass
        output = se3pose_transform_point(translations, rotations, points)

        # Backward pass
        grad_output = torch.randn_like(output)
        output.backward(grad_output)

        # Check that gradients exist and are non-zero
        self.assertIsNotNone(translations.grad)
        self.assertIsNotNone(rotations.grad)
        self.assertIsNotNone(points.grad)
        self.assertTrue(torch.any(translations.grad != 0))
        self.assertTrue(torch.any(rotations.grad != 0))
        self.assertTrue(torch.any(points.grad != 0))

    def test_se3_to_matrix_backward(self):
        """Test that gradients flow correctly through SE3 to matrix conversion"""
        N = 1000

        # Random poses with gradients enabled
        translations = torch.randn(N, 3, device=device, requires_grad=True)
        rotations = torch.nn.functional.normalize(torch.randn(N, 4, device=device, requires_grad=True))
        rotations.retain_grad()  # Need this for non-leaf tensors

        # Forward pass
        matrices = se3pose_to_matrix(translations, rotations)

        # Backward pass
        grad_output = torch.randn_like(matrices)
        matrices.backward(grad_output)

        # Check that gradients exist and are non-zero
        self.assertIsNotNone(translations.grad)
        self.assertIsNotNone(rotations.grad)
        self.assertTrue(torch.any(translations.grad != 0))
        self.assertTrue(torch.any(rotations.grad != 0))

    def test_se3_to_inverse_matrix_backward_grad_shape(self):
        """Verify backward receives grad_result as (N,16) matching the Slang kernel layout.

        The forward creates result as (N,16) internally and reshapes to (N,4,4)
        on return (c0d61f49d). The backward must reshape grad_result back to
        (N,16) before passing to the Slang kernel. This test verifies the
        gradient values are correct by comparing against a reference computation:
        the forward matrix composed with torch.inverse should give the same
        inverse, and gradients through both paths should agree.
        """
        N = 100
        torch.manual_seed(42)

        translations = torch.randn(N, 3, device=device, requires_grad=True)
        rotations = torch.nn.functional.normalize(torch.randn(N, 4, device=device), dim=-1).requires_grad_(True)
        rotations.retain_grad()

        inverse_matrices = se3pose_to_inverse_matrix(translations, rotations)
        grad_output = torch.randn_like(inverse_matrices)
        inverse_matrices.backward(grad_output)

        grad_t_slang = translations.grad.clone()
        grad_r_slang = rotations.grad.clone()

        translations2 = translations.detach().clone().requires_grad_(True)
        rotations2 = rotations.detach().clone().requires_grad_(True)
        rotations2.retain_grad()

        forward_matrices = se3pose_to_matrix(translations2, rotations2)
        inverse_ref = torch.inverse(forward_matrices)
        inverse_ref.backward(grad_output)

        self.assertTrue(
            torch.all(torch.isfinite(grad_t_slang)),
            f"translations.grad has non-finite values: "
            f"NaN={grad_t_slang.isnan().sum()}, Inf={grad_t_slang.isinf().sum()}",
        )
        self.assertTrue(
            torch.all(torch.isfinite(grad_r_slang)),
            f"rotations.grad has non-finite values: NaN={grad_r_slang.isnan().sum()}, Inf={grad_r_slang.isinf().sum()}",
        )
        self.assertTrue(
            torch.allclose(grad_t_slang, translations2.grad, atol=1e-4),
            f"translation grads diverge from reference: "
            f"max diff={torch.max(torch.abs(grad_t_slang - translations2.grad)):.6f}",
        )
        self.assertTrue(
            torch.allclose(grad_r_slang, rotations2.grad, atol=1e-4),
            f"rotation grads diverge from reference: "
            f"max diff={torch.max(torch.abs(grad_r_slang - rotations2.grad)):.6f}",
        )

    def test_se3_to_matrix_backward_grad_shape(self):
        """Same gradient-correctness test for the forward matrix function."""
        N = 100
        torch.manual_seed(42)

        translations = torch.randn(N, 3, device=device, requires_grad=True)
        rotations = torch.nn.functional.normalize(torch.randn(N, 4, device=device), dim=-1).requires_grad_(True)
        rotations.retain_grad()

        matrices = se3pose_to_matrix(translations, rotations)
        grad_output = torch.randn_like(matrices)
        matrices.backward(grad_output)

        self.assertTrue(
            torch.all(torch.isfinite(translations.grad)),
            f"translations.grad has non-finite values: "
            f"NaN={translations.grad.isnan().sum()}, Inf={translations.grad.isinf().sum()}",
        )
        self.assertTrue(
            torch.all(torch.isfinite(rotations.grad)),
            f"rotations.grad has non-finite values: "
            f"NaN={rotations.grad.isnan().sum()}, Inf={rotations.grad.isinf().sum()}",
        )


class TestTrajectory2Poses(unittest.TestCase):
    """Tests for 2-pose trajectory operations."""

    def test_trajectory_transform_point_interpolation(self):
        """Test that interpolation at t=0.5 produces correct midpoint"""
        N = 10000

        # Create two poses with identity rotation and different translations
        t0 = torch.zeros(N, 3, device=device)
        t1 = torch.full((N, 3), 2.0, device=device)
        r = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).repeat(N, 1)  # Identity rotation

        time0 = torch.zeros(N, device=device)
        time1 = torch.ones(N, device=device)
        query_time = torch.full((N,), 0.5, device=device)

        # Point at origin
        point = torch.zeros(N, 3, device=device)

        # Expected: at t=0.5, translation should be [1.0, 1.0, 1.0]
        expected_trans = torch.ones(N, 3, device=device)

        # Call kernel
        result = trajectory_transform_point_2poses(t0, r, time0, t1, r, time1, point, query_time)

        self.assertTrue(torch.allclose(result["point"], expected_trans, atol=1e-6))
        self.assertFalse(torch.any(result["out_of_bounds"]))

    def test_trajectory_transform_point_extrapolation(self):
        """Test that extrapolation beyond bounds works and sets out_of_bounds flag"""
        N = 10000

        # Create two poses with linear motion
        t0 = torch.zeros(N, 3, device=device)
        t1 = torch.ones(N, 3, device=device)
        r = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).repeat(N, 1)  # Identity rotation

        time0 = torch.zeros(N, device=device)
        time1 = torch.ones(N, device=device)
        query_time = torch.full((N,), 2.0, device=device)  # Beyond bounds

        # Point at origin
        point = torch.zeros(N, 3, device=device)

        # Expected: at t=2.0, translation should be [2.0, 2.0, 2.0] (extrapolated)
        expected_trans = torch.full((N, 3), 2.0, device=device)

        # Call kernel
        result = trajectory_transform_point_2poses(t0, r, time0, t1, r, time1, point, query_time)

        self.assertTrue(torch.allclose(result["point"], expected_trans, atol=1e-6))
        self.assertTrue(torch.all(result["out_of_bounds"]))

    def test_trajectory_get_rotation_interpolation(self):
        """Test rotation interpolation using SLERP"""
        N = 10000

        # Two rotations - identity and 90-degree around Z
        r0 = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).repeat(N, 1)
        # 90 degrees around Z: quat = [0, 0, sin(45°), cos(45°)] = [0, 0, 0.707, 0.707]
        r1 = torch.tensor([0.0, 0.0, 0.7071068, 0.7071068], device=device).repeat(N, 1)

        t0 = torch.zeros(N, 3, device=device)
        t1 = torch.zeros(N, 3, device=device)

        time0 = torch.zeros(N, device=device)
        time1 = torch.ones(N, device=device)
        query_time = torch.full((N,), 0.5, device=device)

        # Call kernel
        result = trajectory_get_rotation_2poses(t0, r0, time0, t1, r1, time1, query_time)

        # Expected: SLERP at t=0.5 between identity and 90° rotation
        expected = quat_slerp(r0, r1, 0.5)

        self.assertTrue(torch.allclose(result["quat"], expected, atol=1e-6))
        self.assertFalse(torch.any(result["out_of_bounds"]))

    def test_trajectory_transform_point_backward(self):
        """Test that gradients flow correctly through trajectory transformation"""
        N = 1000

        # Random trajectory with gradients enabled
        # Use simple, independent time values to avoid gradient cancellation
        t0 = torch.randn(N, 3, device=device, requires_grad=True)
        r0 = torch.nn.functional.normalize(torch.randn(N, 4, device=device, requires_grad=True))
        r0.retain_grad()  # Need this for non-leaf tensors
        time0 = torch.zeros(N, device=device)
        t1 = torch.randn(N, 3, device=device, requires_grad=True)
        r1 = torch.nn.functional.normalize(torch.randn(N, 4, device=device, requires_grad=True))
        r1.retain_grad()  # Need this for non-leaf tensors
        time1 = torch.ones(N, device=device)
        points = torch.randn(N, 3, device=device, requires_grad=True)
        query_time = torch.full((N,), 0.5, device=device)  # Query at midpoint

        # Forward pass
        result = trajectory_transform_point_2poses(t0, r0, time0, t1, r1, time1, points, query_time)

        # Backward pass
        grad_output = torch.randn_like(result["point"])
        result["point"].backward(grad_output)

        # Check that gradients exist and are non-zero for the main parameters
        # (translations, rotations, points - these should always have meaningful gradients)
        self.assertIsNotNone(t0.grad)
        self.assertIsNotNone(r0.grad)
        self.assertIsNotNone(t1.grad)
        self.assertIsNotNone(r1.grad)
        self.assertIsNotNone(points.grad)
        self.assertTrue(torch.any(t0.grad != 0))
        self.assertTrue(torch.any(r0.grad != 0))
        self.assertTrue(torch.any(t1.grad != 0))
        self.assertTrue(torch.any(r1.grad != 0))
        self.assertTrue(torch.any(points.grad != 0))


class TestTrajectory1Pose(unittest.TestCase):
    """Tests for 1-pose trajectory operations."""

    def test_trajectory_transform_point_1pose(self):
        """Test that 1-pose trajectory returns the pose transformation"""
        N = 10000

        # Single pose
        trans = torch.randn(N, 3, device=device)
        rot = torch.nn.functional.normalize(torch.randn(N, 4, device=device))
        time = torch.zeros(N, device=device)
        point = torch.randn(N, 3, device=device)
        query_time = torch.zeros(N, device=device)  # Query at same time

        # Call kernel
        result = trajectory_transform_point_1pose(trans, rot, time, point, query_time)

        # Expected: same as SE3 transformation
        expected = se3pose_transform_point(trans, rot, point)

        self.assertTrue(torch.allclose(result["point"], expected, atol=1e-6))
        self.assertFalse(torch.any(result["out_of_bounds"]))

    def test_trajectory_transform_point_1pose_out_of_bounds(self):
        """Test that 1-pose trajectory sets out_of_bounds flag when query_time != time"""
        N = 10000

        # Single pose
        trans = torch.randn(N, 3, device=device)
        rot = torch.nn.functional.normalize(torch.randn(N, 4, device=device))
        time = torch.zeros(N, device=device)
        point = torch.randn(N, 3, device=device)
        query_time = torch.ones(N, device=device)  # Query at different time

        # Call kernel
        result = trajectory_transform_point_1pose(trans, rot, time, point, query_time)

        # Expected: same transformation but out_of_bounds should be true
        expected = se3pose_transform_point(trans, rot, point)

        self.assertTrue(torch.allclose(result["point"], expected, atol=1e-6))
        self.assertTrue(torch.all(result["out_of_bounds"]))

    def test_trajectory_transform_point_1pose_backward(self):
        """Test that gradients flow correctly through 1-pose trajectory"""
        N = 1000

        # Random trajectory with gradients enabled
        trans = torch.randn(N, 3, device=device, requires_grad=True)
        rot = torch.nn.functional.normalize(torch.randn(N, 4, device=device, requires_grad=True))
        rot.retain_grad()  # Need this for non-leaf tensors
        time = torch.zeros(N, device=device)
        points = torch.randn(N, 3, device=device, requires_grad=True)
        query_time = torch.zeros(N, device=device)  # Query at same time as the pose

        # Forward pass
        result = trajectory_transform_point_1pose(trans, rot, time, points, query_time)

        # Backward pass
        grad_output = torch.randn_like(result["point"])
        result["point"].backward(grad_output)

        # Check that gradients exist and are non-zero for the main parameters
        # (translation, rotation, points - these should always have meaningful gradients)
        self.assertIsNotNone(trans.grad)
        self.assertIsNotNone(rot.grad)
        self.assertIsNotNone(points.grad)
        self.assertTrue(torch.any(trans.grad != 0))
        self.assertTrue(torch.any(rot.grad != 0))
        self.assertTrue(torch.any(points.grad != 0))


class TestFrameTransformPosesTQuat(unittest.TestCase):
    """Tests for frame transform poses tquat."""

    def test_frame_transform_poses_tquat(self):
        """Test frame transform poses tquat with random transforms."""
        N = 1000

        # Random input poses (normalize quaternion part)
        tquat_poses = torch.randn(N, 7, device=device)
        tquat_poses[..., 3:] = torch.nn.functional.normalize(tquat_poses[..., 3:], dim=-1)

        # Random frame transform (normalize quaternion)
        frame_quat = torch.randn(4)
        frame_quat = torch.nn.functional.normalize(frame_quat, dim=-1)
        rotation = tuple(frame_quat.tolist())  # (qx, qy, qz, qw)

        frame_trans = torch.randn(3)
        translation = tuple(frame_trans.tolist())

        scale = torch.rand(1).item() * 2.0 + 0.5  # random scale in [0.5, 2.5]

        # GPU kernel result
        result = frame_transform_poses_tquat(tquat_poses, rotation, translation, scale)

        # Reference implementation (CPU, using existing quaternion ops)
        from libs.geometry.kernels.quaternion import quat_multiply, quat_rotate_vector

        frame_quat_tensor = torch.tensor(rotation, device=device).unsqueeze(0)  # (1, 4)
        frame_trans_tensor = torch.tensor(translation, device=device).unsqueeze(0)  # (1, 3)

        pose_t = tquat_poses[..., :3]  # (N, 3)
        pose_R = tquat_poses[..., 3:]  # (N, 4)

        expected_R = quat_multiply(frame_quat_tensor.expand_as(pose_R), pose_R)
        rotated_t = quat_rotate_vector(frame_quat_tensor.expand_as(pose_R), pose_t)
        expected_t = scale * (rotated_t + frame_trans_tensor)
        expected = torch.cat([expected_t, expected_R], dim=-1)

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
