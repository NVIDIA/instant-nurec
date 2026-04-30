# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Comprehensive unit tests for quaternion operations.

Tests the Slang-backed Python quaternion functions by verifying mathematical
properties and comparing against PyTorch ground truth implementations.
"""

import unittest

import torch

from libs.geometry.kernels import quaternion as quat


# Test configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test parameters
NUM_QUATERNIONS = 128
ATOL = 1e-5
ATOL_STRICT = 1e-6


def random_quaternions(n: int, normalized: bool = True) -> torch.Tensor:
    """Generate random quaternions.

    Args:
        n: Number of quaternions
        normalized: Whether to normalize the quaternions

    Returns:
        (n, 4) tensor of quaternions in xyzw format
    """
    quats = torch.randn(n, 4, device=device)
    if normalized:
        quats = quat.quat_normalize_safe(quats)
    return quats


def random_vectors(n: int) -> torch.Tensor:
    """Generate random 3D vectors.

    Args:
        n: Number of vectors

    Returns:
        (n, 3) tensor of vectors
    """
    return torch.randn(n, 3, device=device)


def quat_to_matrix_pytorch(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to rotation matrix using pure PyTorch (ground truth).

    Args:
        q: (..., 4) quaternion(s) in xyzw format

    Returns:
        (..., 3, 3) rotation matrix/matrices
    """
    # Normalize first
    q = q / torch.norm(q, dim=-1, keepdim=True)

    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Precompute repeated terms
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, xw = x * y, x * z, x * w
    yz, yw, zw = y * z, y * w, z * w

    # Build rotation matrix
    shape = q.shape[:-1] + (3, 3)
    matrix = torch.zeros(shape, dtype=q.dtype, device=q.device)

    matrix[..., 0, 0] = 1.0 - 2.0 * (y2 + z2)
    matrix[..., 0, 1] = 2.0 * (xy - zw)
    matrix[..., 0, 2] = 2.0 * (xz + yw)

    matrix[..., 1, 0] = 2.0 * (xy + zw)
    matrix[..., 1, 1] = 1.0 - 2.0 * (x2 + z2)
    matrix[..., 1, 2] = 2.0 * (yz - xw)

    matrix[..., 2, 0] = 2.0 * (xz - yw)
    matrix[..., 2, 1] = 2.0 * (yz + xw)
    matrix[..., 2, 2] = 1.0 - 2.0 * (x2 + y2)

    return matrix


class TestQuaternionOperations(unittest.TestCase):
    """Test quaternion operations using Slang-backed Python functions."""

    def setUp(self):
        """Set up test fixtures - ensure deterministic seeding."""
        SEED = 42
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)

    def test_normalize_safe(self):
        """Test quaternion normalization."""
        quats = torch.randn(NUM_QUATERNIONS, 4, device=device)
        normalized = quat.quat_normalize_safe(quats)

        # Check that result is unit quaternions
        norms = torch.norm(normalized, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=ATOL))

    def test_normalize_safe_degenerate(self):
        """Test normalization with near-zero quaternions."""
        quats = torch.zeros(10, 4, device=device)
        quats[5:] = 1e-10  # Very small values

        normalized = quat.quat_normalize_safe(quats)

        # Should return identity quaternion
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
        self.assertTrue(torch.allclose(normalized, expected.expand_as(normalized), atol=ATOL))

    def test_conjugate(self):
        """Test quaternion conjugate."""
        quats = random_quaternions(NUM_QUATERNIONS)
        conj = quat.quat_conjugate(quats)

        # xyz should be negated, w should stay the same
        self.assertTrue(torch.allclose(conj[..., :3], -quats[..., :3], atol=ATOL))
        self.assertTrue(torch.allclose(conj[..., 3], quats[..., 3], atol=ATOL))

    def test_inverse(self):
        """Test quaternion inverse."""
        quats = random_quaternions(NUM_QUATERNIONS)
        inv = quat.quat_inverse(quats)

        # For unit quaternions, q * q^(-1) = identity
        product = quat.quat_multiply(quats, inv)
        identity = quat.quat_identity((NUM_QUATERNIONS,), device=device)

        self.assertTrue(torch.allclose(product, identity, atol=ATOL))

    def test_multiply(self):
        """Test quaternion multiplication."""
        q1 = random_quaternions(NUM_QUATERNIONS)
        q2 = random_quaternions(NUM_QUATERNIONS)
        q3 = random_quaternions(NUM_QUATERNIONS)

        # Test associativity: (q1 * q2) * q3 = q1 * (q2 * q3)
        result1 = quat.quat_multiply(quat.quat_multiply(q1, q2), q3)
        result2 = quat.quat_multiply(q1, quat.quat_multiply(q2, q3))

        self.assertTrue(torch.allclose(result1, result2, atol=ATOL))

    def test_multiply_identity(self):
        """Test multiplication with identity."""
        quats = random_quaternions(NUM_QUATERNIONS)
        identity = quat.quat_identity((NUM_QUATERNIONS,), device=device)

        result = quat.quat_multiply(quats, identity)
        self.assertTrue(torch.allclose(result, quats, atol=ATOL))

    def test_to_matrix(self):
        """Test quaternion to matrix conversion."""
        quats = random_quaternions(NUM_QUATERNIONS)
        matrices = quat.quat_to_matrix(quats)

        # Compare with PyTorch ground truth
        matrices_expected = quat_to_matrix_pytorch(quats)
        self.assertTrue(torch.allclose(matrices, matrices_expected, atol=ATOL_STRICT))

    def test_matrix_is_rotation(self):
        """Test that quaternion to matrix produces valid rotation matrices."""
        quats = random_quaternions(NUM_QUATERNIONS)
        matrices = quat.quat_to_matrix(quats)

        # Check orthogonality: R^T * R = I
        identity = torch.eye(3, device=device).unsqueeze(0).expand(NUM_QUATERNIONS, 3, 3)
        product = torch.bmm(matrices.transpose(-2, -1), matrices)
        self.assertTrue(torch.allclose(product, identity, atol=ATOL))

        # Check determinant = 1
        det = torch.linalg.det(matrices)
        self.assertTrue(torch.allclose(det, torch.ones_like(det), atol=ATOL))

    def test_rotate_vector_vs_matrix(self):
        """Test that vector rotation matches matrix multiplication."""
        quats = random_quaternions(NUM_QUATERNIONS)
        vectors = random_vectors(NUM_QUATERNIONS)

        # Rotate using quaternion
        rotated_quat = quat.quat_rotate_vector(quats, vectors)

        # Rotate using matrix
        matrices = quat.quat_to_matrix(quats)
        rotated_matrix = torch.bmm(matrices, vectors.unsqueeze(-1)).squeeze(-1)

        # Should be the same
        self.assertTrue(torch.allclose(rotated_quat, rotated_matrix, atol=ATOL))

    def test_rotate_preserves_length(self):
        """Test that rotation preserves vector length."""
        quats = random_quaternions(NUM_QUATERNIONS)
        vectors = random_vectors(NUM_QUATERNIONS)

        original_norms = torch.norm(vectors, dim=-1)
        rotated = quat.quat_rotate_vector(quats, vectors)
        rotated_norms = torch.norm(rotated, dim=-1)

        self.assertTrue(torch.allclose(original_norms, rotated_norms, atol=ATOL))

    def test_from_axis_angle(self):
        """Test axis-angle to quaternion conversion."""
        # Generate random axes and angles
        axes = random_vectors(NUM_QUATERNIONS)
        axes = axes / torch.norm(axes, dim=-1, keepdim=True)  # Normalize
        angles = torch.rand(NUM_QUATERNIONS, device=device) * 2 * torch.pi

        # Convert to quaternion
        quats = quat.quat_from_axis_angle(axes, angles)

        # Check that result is unit quaternions
        norms = torch.norm(quats, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=ATOL))

    def test_lerp_endpoints(self):
        """Test linear interpolation endpoints."""
        q1 = random_quaternions(NUM_QUATERNIONS)
        q2 = random_quaternions(NUM_QUATERNIONS)

        # Test endpoints
        result_0 = quat.quat_lerp(q1, q2, 0.0)
        result_1 = quat.quat_lerp(q1, q2, 1.0)

        # At t=0, should be close to q1 (or -q1) - element-wise check
        matches_q1 = (torch.abs(result_0 - q1) < ATOL).all(dim=1)
        matches_neg_q1 = (torch.abs(result_0 + q1) < ATOL).all(dim=1)
        all_match_q1 = (matches_q1 | matches_neg_q1).all()
        self.assertTrue(all_match_q1)

        # At t=1, should be close to q2 (or -q2) - element-wise check
        matches_q2 = (torch.abs(result_1 - q2) < ATOL).all(dim=1)
        matches_neg_q2 = (torch.abs(result_1 + q2) < ATOL).all(dim=1)
        all_match_q2 = (matches_q2 | matches_neg_q2).all()
        self.assertTrue(all_match_q2)

    def test_slerp_endpoints(self):
        """Test spherical linear interpolation endpoints."""
        q1 = random_quaternions(NUM_QUATERNIONS)
        q2 = random_quaternions(NUM_QUATERNIONS)

        # Test endpoints
        result_0 = quat.quat_slerp(q1, q2, 0.0)
        result_1 = quat.quat_slerp(q1, q2, 1.0)

        # At t=0, should be close to q1 (or -q1) - element-wise check
        matches_q1 = (torch.abs(result_0 - q1) < ATOL).all(dim=1)
        matches_neg_q1 = (torch.abs(result_0 + q1) < ATOL).all(dim=1)
        all_match_q1 = (matches_q1 | matches_neg_q1).all()
        self.assertTrue(all_match_q1)

        # At t=1, should be close to q2 (or -q2) - element-wise check
        matches_q2 = (torch.abs(result_1 - q2) < ATOL).all(dim=1)
        matches_neg_q2 = (torch.abs(result_1 + q2) < ATOL).all(dim=1)
        all_match_q2 = (matches_q2 | matches_neg_q2).all()
        self.assertTrue(all_match_q2)

    def test_slerp_unit_quaternions(self):
        """Test that SLERP produces unit quaternions."""
        q1 = random_quaternions(NUM_QUATERNIONS)
        q2 = random_quaternions(NUM_QUATERNIONS)

        t_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        for t in t_values:
            result = quat.quat_slerp(q1, q2, t)
            norms = torch.norm(result, dim=-1)
            self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=ATOL), f"Failed at t={t}")

    def test_slerp_batched_endpoints(self):
        """Test batched SLERP with per-element t at endpoints."""
        q1 = random_quaternions(NUM_QUATERNIONS)
        q2 = random_quaternions(NUM_QUATERNIONS)

        # Test t=0 for all
        t_zeros = torch.zeros(NUM_QUATERNIONS, device=device)
        result_0 = quat.quat_slerp_batched(q1, q2, t_zeros)

        matches_q1 = (torch.abs(result_0 - q1) < ATOL).all(dim=1)
        matches_neg_q1 = (torch.abs(result_0 + q1) < ATOL).all(dim=1)
        self.assertTrue((matches_q1 | matches_neg_q1).all())

        # Test t=1 for all
        t_ones = torch.ones(NUM_QUATERNIONS, device=device)
        result_1 = quat.quat_slerp_batched(q1, q2, t_ones)

        matches_q2 = (torch.abs(result_1 - q2) < ATOL).all(dim=1)
        matches_neg_q2 = (torch.abs(result_1 + q2) < ATOL).all(dim=1)
        self.assertTrue((matches_q2 | matches_neg_q2).all())

    def test_slerp_batched_per_element_t(self):
        """Test batched SLERP with varying per-element t values."""
        q1 = random_quaternions(NUM_QUATERNIONS)
        q2 = random_quaternions(NUM_QUATERNIONS)

        # Random t values per element
        t = torch.rand(NUM_QUATERNIONS, device=device)
        result = quat.quat_slerp_batched(q1, q2, t)

        # Result should be unit quaternions
        norms = torch.norm(result, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=ATOL))

    def test_slerp_batched_matches_scalar_slerp(self):
        """Test that batched SLERP with uniform t matches scalar SLERP."""
        q1 = random_quaternions(NUM_QUATERNIONS)
        q2 = random_quaternions(NUM_QUATERNIONS)

        for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            # Scalar slerp
            result_scalar = quat.quat_slerp(q1, q2, t_val)

            # Batched slerp with uniform t
            t_batched = torch.full((NUM_QUATERNIONS,), t_val, device=device)
            result_batched = quat.quat_slerp_batched(q1, q2, t_batched)

            # Should match (accounting for sign ambiguity)
            matches = (torch.abs(result_scalar - result_batched) < ATOL).all(dim=1)
            matches_neg = (torch.abs(result_scalar + result_batched) < ATOL).all(dim=1)
            self.assertTrue((matches | matches_neg).all(), f"Mismatch at t={t_val}")

    def test_slerp_batched_unit_quaternions(self):
        """Test that batched SLERP produces unit quaternions for all t values."""
        q1 = random_quaternions(NUM_QUATERNIONS)
        q2 = random_quaternions(NUM_QUATERNIONS)

        # Test with random t values
        t = torch.rand(NUM_QUATERNIONS, device=device)
        result = quat.quat_slerp_batched(q1, q2, t)
        norms = torch.norm(result, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=ATOL))

    def test_manifold_interp_endpoints(self):
        """Test manifold interpolation endpoints."""
        q1 = random_quaternions(NUM_QUATERNIONS)
        q2 = random_quaternions(NUM_QUATERNIONS)

        # Test endpoints
        result_0 = quat.quat_manifold_interp(q1, q2, 0.0)
        result_1 = quat.quat_manifold_interp(q1, q2, 1.0)

        # At t=0, should be close to q1 (or -q1) - element-wise check
        matches_q1 = (torch.abs(result_0 - q1) < ATOL).all(dim=1)
        matches_neg_q1 = (torch.abs(result_0 + q1) < ATOL).all(dim=1)
        all_match_q1 = (matches_q1 | matches_neg_q1).all()
        self.assertTrue(all_match_q1)

        # At t=1, should be close to q2 (or -q2) - element-wise check
        matches_q2 = (torch.abs(result_1 - q2) < ATOL).all(dim=1)
        matches_neg_q2 = (torch.abs(result_1 + q2) < ATOL).all(dim=1)
        all_match_q2 = (matches_q2 | matches_neg_q2).all()
        self.assertTrue(all_match_q2)

    def test_angular_distance(self):
        """Test angular distance computation."""
        quats = random_quaternions(NUM_QUATERNIONS)

        # Distance to itself should be 0 (use relaxed tolerance due to acos numerical precision)
        dist_self = quat.quat_angular_distance(quats, quats)
        self.assertTrue(torch.allclose(dist_self, torch.zeros_like(dist_self), atol=1e-3))

        # Distance is symmetric
        q2 = random_quaternions(NUM_QUATERNIONS)
        dist_12 = quat.quat_angular_distance(quats, q2)
        dist_21 = quat.quat_angular_distance(q2, quats)
        self.assertTrue(torch.allclose(dist_12, dist_21, atol=ATOL))

        # Distance should be in [0, pi]
        self.assertTrue((dist_12 >= 0).all())
        self.assertTrue((dist_12 <= torch.pi).all())

    def test_so3_exp_small_angle_taylor_series(self):
        """Test that so3Exp Taylor series is correct for small angles.

        Regression test: the Taylor series coefficients for cos(theta/2) and
        sin(theta/2)/theta were previously incorrect in the small-angle branch.
        The cos(theta/2) term was 1 - theta^2/4 instead of 1 - theta^2/8,
        and the sin(theta/2)/theta 4th-order term was 1/2880 instead of 1/3840.
        """
        N = 128
        # Generate small omega vectors with |omega|^2 in [5e-7, 9.99e-7],
        # just below the SMALL_ANGLE_THRESHOLD (1e-6) to exercise the Taylor
        # series branch. These are the largest angles that use the Taylor path,
        # maximizing the detectable error from wrong coefficients.
        directions = torch.randn(N, 3, device=device)
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)
        theta_sq = torch.linspace(5e-7, 9.99e-7, N, device=device)
        theta = torch.sqrt(theta_sq)
        omega = directions * theta.unsqueeze(-1)

        # Compute via Slang so3Exp
        result = quat.so3_exp(omega)

        # Compute ground truth with double precision, cast to float
        # (mirrors the kernel's internal double->float path)
        theta_d = theta.double()
        omega_d = omega.double()
        half_theta_d = theta_d / 2
        expected_w = torch.cos(half_theta_d).float()
        sin_half_over_theta = torch.sin(half_theta_d) / theta_d
        expected_xyz = (omega_d * sin_half_over_theta.unsqueeze(-1)).float()

        # The w component (cos(theta/2)) is where the old bug is most visible.
        # With wrong coefficients (1 - theta^2/4 instead of 1 - theta^2/8),
        # the w error is ~1.19e-7 at these angles — exceeding 1e-7 tolerance.
        # The correct Taylor series matches double-precision sin/cos exactly.
        w_error = (result[..., 3] - expected_w).abs().max().item()
        self.assertLessEqual(
            w_error,
            1e-7,
            f"so3Exp small-angle w component error too large: {w_error:.2e} > 1e-7",
        )

        # xyz components should also match ground truth
        xyz_error = (result[..., :3] - expected_xyz).abs().max().item()
        self.assertLessEqual(
            xyz_error,
            1e-7,
            f"so3Exp small-angle xyz component error too large: {xyz_error:.2e} > 1e-7",
        )

        # Verify unit norm property
        norms = torch.norm(result, dim=-1)
        self.assertTrue(
            torch.allclose(norms, torch.ones_like(norms), atol=1e-6),
            f"so3Exp result not unit quaternion: max norm error = {(norms - 1).abs().max().item():.2e}",
        )

    def test_batch_shapes(self):
        """Test that functions work with various batch shapes."""
        # Test with different batch shapes
        shapes = [(5,), (3, 4), (2, 3, 4)]

        for shape in shapes:
            q1 = torch.randn(*shape, 4, device=device)
            q1 = quat.quat_normalize_safe(q1)

            q2 = torch.randn(*shape, 4, device=device)
            q2 = quat.quat_normalize_safe(q2)

            # Test various operations preserve shape
            result = quat.quat_conjugate(q1)
            self.assertEqual(result.shape, (*shape, 4))

            result = quat.quat_multiply(q1, q2)
            self.assertEqual(result.shape, (*shape, 4))

            result = quat.quat_to_matrix(q1)
            self.assertEqual(result.shape, (*shape, 3, 3))

            vectors = torch.randn(*shape, 3, device=device)
            result = quat.quat_rotate_vector(q1, vectors)
            self.assertEqual(result.shape, (*shape, 3))


if __name__ == "__main__":
    unittest.main()
