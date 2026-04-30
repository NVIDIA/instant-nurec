# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import unittest

from typing import Tuple

import numpy as np

from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from nre.utils.geometry import (
    Plane,
    PlaneDetectorRansac,
    hann,
    quat_to_so3_matrix,
    se3_matrix_inverse,
    se3_matrix_to_tquat,
    so3_matrix_to_quat,
    tquat_to_se3_matrix,
    vector_align_rotation,
)


class TestGeometryConversions(unittest.TestCase):
    def setUp(self):
        self.random_rotations = Rotation.random(1000)
        self.random_translations = np.random.rand(1000, 3)

    def test_quat_to_so3(self):
        ref_scipy = Rotation.from_quat(self.random_rotations.as_quat(canonical=False)).as_matrix()
        ours = quat_to_so3_matrix(self.random_rotations.as_quat(canonical=False))

        np.testing.assert_array_almost_equal(ref_scipy, ours, decimal=6)

    def test_so3_to_quat(self):
        ref_scipy = Rotation.from_matrix(self.random_rotations.as_matrix()).as_quat(canonical=False)
        ours = so3_matrix_to_quat(self.random_rotations.as_matrix())

        np.testing.assert_array_almost_equal(ref_scipy, ours, decimal=6)

    def test_so3_round_trip(self):
        ref_so3 = self.random_rotations.as_matrix()
        quat_ours = so3_matrix_to_quat(ref_so3)
        so3_ours = quat_to_so3_matrix(quat_ours)
        np.testing.assert_array_almost_equal(ref_so3, so3_ours, decimal=6)

    def test_se3_round_trip(self):
        random_se3 = np.eye(4).reshape(1, 4, 4).repeat(self.random_translations.shape[0], axis=0)
        random_se3[:, :3, :3] = self.random_rotations.as_matrix()
        random_se3[:, :3, 3] = self.random_translations

        tquat_ours = se3_matrix_to_tquat(random_se3)
        se3_ours = tquat_to_se3_matrix(tquat_ours)

        np.testing.assert_array_almost_equal(random_se3, se3_ours, decimal=6)

    def test_se3_matrix_inverse_round_trip(self):
        random_se3 = np.eye(4).reshape(1, 4, 4).repeat(self.random_translations.shape[0], axis=0)
        random_se3[:, :3, :3] = self.random_rotations.as_matrix()
        random_se3[:, :3, 3] = self.random_translations

        random_se3_inverse = se3_matrix_inverse(random_se3)
        random_se3_inverse_inverse = se3_matrix_inverse(random_se3_inverse)

        np.testing.assert_array_almost_equal(random_se3, random_se3_inverse_inverse, decimal=6)
        np.testing.assert_array_almost_equal(
            random_se3_inverse @ random_se3,
            np.eye(4).reshape(1, 4, 4).repeat(self.random_translations.shape[0], axis=0),
            decimal=6,
        )

    def test_se3_matrix_inverse_round_trip_single(self):
        """Roundtrip test (inverse of an inverse) for a single se3 matrix."""
        identity = np.eye(4)
        random_se3 = identity.copy()
        random_se3[:3, :3] = self.random_rotations[0].as_matrix()
        random_se3[:3, 3] = self.random_translations[0]

        random_se3_inverse = se3_matrix_inverse(random_se3)
        random_se3_inverse_inverse = se3_matrix_inverse(random_se3_inverse)

        np.testing.assert_array_almost_equal(random_se3, random_se3_inverse_inverse, decimal=6)
        np.testing.assert_array_almost_equal(random_se3_inverse @ random_se3, identity, decimal=6)

    def test_se3_matrix_inverse_identity(self):
        """Test with identity matrix"""
        result = se3_matrix_inverse(np.eye(4))
        np.testing.assert_array_almost_equal(result, np.eye(4), decimal=6)


# Useful for testing plane-related code
def get_plane_to_world_transf(plane: Plane) -> Tuple[np.ndarray, np.ndarray]:
    """Constructs a 3D reference frame aligned with the plane and returns the Euclidean transform from it to the world.

    The plane's 3D reference frame is constructed such that:
    - its Z-axis aligns with the plane normal.
    - the plane itself is defined by Z=0.
    - the origin is the projection of the world origin to the plane.
    - the distance of any point from the plane is given by its Z-coordinate in this frame.

    Any 3D point [X,Y,Z] given in the plane's designed 3D reference frame transforms to the world frame as follows:
      world_point = rotation_matrix @ point + plane_origin

    Returns (rotation_matrix, plane_origin)
    """

    normal_length = np.linalg.norm(plane.normal)
    unit_normal = plane.normal / normal_length

    # Projecton of the world origin to the plane.
    plane_origin = -plane.offset / normal_length * unit_normal
    # The z-axis of the plane reference frame is aligned with the plane normal.
    axis_z = unit_normal
    # Find the world axis whose projection to the plane is the longest, i.e. projection to the normal is the shortest.
    # This is a safe way to pick a direction deterministically in the plane, that is also safe to normalize.
    # The projection length of the world axes to the normal are the abs values of the normal's components, respectively.
    i = np.argmin(np.abs(plane.normal))
    world_axis = np.zeros((3,), dtype=np.float32)
    world_axis[i] = 1.0

    # Project the world axis to the plane to obtain x-axis in the plane.
    axis_x = world_axis - np.dot(unit_normal, world_axis) * unit_normal
    axis_x /= np.linalg.norm(axis_x)
    assert np.isclose(np.dot(axis_x, axis_z), 0.0)  # Checks that the x-axis is parallel to the plane, as required.
    axis_y = np.cross(axis_z, axis_x)
    axis_y /= np.linalg.norm(axis_y)  # Should be normalized already, so just for numerical accuracy

    # Place the constructed axes of the plane reference frame into columns of a 3x3 rotation matrix.
    rotation_matrix = np.vstack([axis_x, axis_y, axis_z]).T
    return rotation_matrix, plane_origin


def test_plane_init() -> None:
    normal = np.array([1.1, -2.2, 3.3])
    offset = 1.5
    plane = Plane(normal, offset)
    assert np.all(plane.normal == normal)
    assert plane.offset == offset


def test_plane_to_point_distance() -> None:
    plane = Plane(np.array([1.1, -2.2, 3.3]), 1.5)
    points = np.array([[0, 0, 0], [-3.1, -2.3, 1.1]], dtype=np.float32)
    assert points.shape == (2, 3)
    distances = plane.distance_from_points(points)
    assert distances.shape == (len(points),)
    norm = np.linalg.norm(plane.normal)
    assert np.isclose(distances[0], 1.5 / norm)
    assert np.isclose(distances[1], 6.78 / norm)

    get_plane_to_world_transf(plane)


# Ensures that get_plane_to_world_transf() is functional to be used in tests.
def test_get_plane_to_world_transf() -> None:
    plane = Plane(np.array([1.1, -2.2, 3.3]), 1.5)

    rotation_matrix, plane_origin = get_plane_to_world_transf(plane)

    # The plane origin must be on the plane
    assert np.isclose(plane.distance_from_points(plane_origin.reshape(1, 3)), 0.0)
    # The rotation matrix must be orthonormal and represent a right-handed system.
    assert np.allclose(rotation_matrix @ rotation_matrix.T, np.identity(3))
    assert np.isclose(np.linalg.det(rotation_matrix), 1.0)
    # The distance of any point from the plane is given by its third coordinate in the plane reference frame.
    points = np.array([[-3.4, 2.6, 1.1], [5.5, 4.4, -2.2]])
    assert np.allclose(plane.distance_from_points(points @ rotation_matrix.T + plane_origin), [1.1, 2.2])


def test_gen_planes_from_single_points() -> None:
    points = np.array([[1, -2, 3], [-3, 4, -5]], dtype=np.float32)
    normals = np.array([[0.5, -1.5, 2.5], [3.3, 1.1, 2.2]], dtype=np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    num_planes = 2
    np.random.seed(123)
    ransac = PlaneDetectorRansac()
    planes, sample_indices = ransac.generate_planes_from_single_points(points, normals, num_planes)

    assert len(planes) == num_planes
    assert sample_indices.shape == (num_planes,)
    for plane_idx in range(num_planes):
        plane = planes[plane_idx]
        point_idx = sample_indices[plane_idx]
        # Plane normal should be the normal of its seed point.
        assert np.allclose(plane.normal, normals[point_idx])
        # Seed point must lie on the plane.
        assert np.isclose(plane.distance_from_points(points[point_idx].reshape(1, 3)), 0.0)


def test_gen_planes_from_single_triplet() -> None:
    points = np.array([[1, 2, 0], [3, 4, 0], [5, -6, 0]], dtype=np.float32)
    normals = np.array([[0, 0, -1], [0, 0, -1], [0, 0, -1]], dtype=np.float32)

    np.random.seed(123)
    ransac = PlaneDetectorRansac()
    planes, triplets = ransac.generate_planes_from_point_triplets(points, normals, 1)

    assert len(planes) == 1
    assert triplets.shape == (1, 3)
    assert np.allclose(planes[0].normal, np.array([[0, 0, -1]]))
    assert np.allclose(planes[0].offset, 0.0)
    # Triplet should be the indices of the only 3 points fed. Makes sure sampling was without replacement.
    assert sorted(triplets.flatten().tolist()) == [0, 1, 2]


def generate_xy_test_points() -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    x_grid, y_grid = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
    points = np.stack([x_grid.flatten(), y_grid.flatten(), np.zeros((x_grid.size,), dtype=np.float32)], axis=1)
    normal = np.array([0, 0, 1], dtype=np.float32)  # Same for all points
    normals = np.repeat(normal.reshape(1, 3), len(points), axis=0)
    assert points.shape == (x_grid.size, 3)
    assert np.isclose(np.linalg.norm(normal), 1.0)
    return points, normals


def test_gen_planes_from_planar_triplets() -> None:
    # Plane in a general arbitrary orientation.
    actual_plane = Plane(np.array([1.1, -2.2, 3.3]), 1.5)

    # Generate points and normals on the plane and transform them into the world reference frame.
    points, normals = generate_xy_test_points()
    plane_rotation, plane_origin = get_plane_to_world_transf(actual_plane)
    points = points @ plane_rotation.T + plane_origin
    normals = normals @ plane_rotation.T

    # Plane hypothesis generation.
    num_planes = 10
    np.random.seed(123)
    ransac = PlaneDetectorRansac()
    planes, triplets = ransac.generate_planes_from_point_triplets(points, normals, num_planes)

    # Tests the output of plane hypothesis generation.
    assert len(planes) == num_planes
    assert triplets.shape == (num_planes, 3)
    for plane_idx in range(num_planes):
        plane = planes[plane_idx]
        # All points must lie on the plane hypothesis.
        assert np.allclose(plane.distance_from_points(points), 0.0)
        # Plane must be oriented to be consistent with the point normals.
        # Ideal case because all test point normals are oriented consistently and perpendicular to the plane.
        assert np.allclose(normals, plane.normal / np.linalg.norm(plane.normal))


def test_gen_planes_from_spatial_triplets() -> None:
    # This tests generates a points within a cube (so not on a plane).
    points = (np.random.rand(100, 3) - 0.5) * 100
    # Normals do not play a role in this test but still need to be well-formed.
    normals = np.repeat(np.array([0, 0, 1], dtype=np.float32).reshape(1, 3), len(points), axis=0)

    # Plane hypothesis generation.
    num_planes = 10
    np.random.seed(123)
    ransac = PlaneDetectorRansac()
    planes, triplets = ransac.generate_planes_from_point_triplets(points, normals, num_planes)

    # Tests the output of plane hypothesis generation.
    assert len(planes) == num_planes
    assert triplets.shape == (num_planes, 3)
    for plane_idx in range(num_planes):
        plane = planes[plane_idx]
        seed_indices = triplets[plane_idx]
        # All three seed points must lie on the plane hypothesis.
        assert np.allclose(plane.distance_from_points(points[seed_indices]), 0.0)


def test_hann_scoring_function() -> None:
    threshold = 10.0
    assert np.isscalar(hann(0.0, threshold))
    assert np.isclose(hann(0.0, threshold), 1.0)
    assert np.isclose(hann(10.0, threshold), 0.0)
    assert np.isclose(hann(-10.0, threshold), 0.0)
    assert np.isclose(hann(0.5 * threshold, threshold), 0.5)
    assert np.isclose(hann(-0.5 * threshold, threshold), 0.5)
    assert np.isclose(hann(threshold / 3, threshold), 0.75)
    assert np.isclose(hann(-threshold / 3, threshold), 0.75)
    assert np.isclose(hann(2 * threshold / 3, threshold), 0.25)
    assert np.isclose(hann(-2 * threshold / 3, threshold), 0.25)
    # Tests passing an array, as well as that the function is always positive
    num_values = 20
    values = hann(np.linspace(-2 * threshold, 2 * threshold, num_values, dtype=np.float32), threshold)
    # assert isinstance(values, np.ndarray) and values.shape = (1,)
    assert isinstance(values, np.ndarray)
    assert values.dtype == np.float32
    assert values.shape == (num_values,)
    assert np.all(values >= 0.0)
    assert np.all(values <= 1.0)


def test_vector_align_rotation() -> None:
    v1 = np.random.rand(10, 3) - 0.5
    v2 = np.random.rand(10, 3) - 0.5

    v1_length = np.linalg.norm(v1, axis=1, keepdims=True)
    v2_length = np.linalg.norm(v2, axis=1, keepdims=True)

    mat = vector_align_rotation(v1, v2)

    np.testing.assert_array_almost_equal(np.matmul(mat, v1[..., None]).squeeze() * v2_length, v2 * v1_length, decimal=6)

    # special case: v2 are zero vectors
    v1 = np.random.rand(10, 3) - 0.5
    v2 = np.zeros((10, 3))

    mat = vector_align_rotation(v1, v2)

    np.testing.assert_array_almost_equal(mat, np.tile(np.eye(3), (10, 1, 1)), decimal=6)

    # special case: v1 are zero vectors
    v1 = np.zeros((10, 3))
    v2 = np.random.rand(10, 3) - 0.5

    mat = vector_align_rotation(v1, v2)

    np.testing.assert_array_almost_equal(mat, np.tile(np.eye(3), (10, 1, 1)), decimal=6)
