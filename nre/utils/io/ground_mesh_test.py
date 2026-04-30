# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np

from nre.utils.io.ground_mesh import (
    DelaunayElevationMeshingAlgorithm,
    calculate_edge_lengths,
    downsample_3d_points_on_2d_grid,
    get_mesh_edges,
    get_vertex_neighbors,
    mesh_smoothing_along_axis,
)


def test_get_mesh_edges() -> None:
    triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    edges = get_mesh_edges(triangles)
    print(edges)
    assert np.all(edges == np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]], dtype=np.int32))


def test_calculate_edge_lengths() -> None:
    vertices = np.array([[0, 3, 0], [0, 0, 0], [4, 3, 0], [4, 0, 0]], dtype=np.float32)
    edges = np.array([[0, 1], [1, 2], [2, 0], [1, 3], [3, 2]], dtype=np.int32)
    edge_lengths = calculate_edge_lengths(vertices, edges)
    print(edge_lengths)
    assert np.allclose(edge_lengths, [3, 5, 4, 4, 3], rtol=0.0, atol=1e-5)


def test_get_vertex_neighbors() -> None:
    vertices = np.array([[0, 3, 0], [0, 0, 0], [4, 3, 0], [4, 0, 0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    neighbors, distances = get_vertex_neighbors(vertices, triangles)
    print("neighbors:\n", neighbors)
    print("distances:\n", distances)
    assert neighbors == [[1, 2], [0, 2, 3], [0, 1, 3], [1, 2]]
    expected_distances = [[3.0, 4.0], [3.0, 5.0, 4.0], [4.0, 5.0, 3.0], [4.0, 3.0]]
    for distance, expected in zip(distances, expected_distances):
        assert np.allclose(distance, expected, rtol=0.0, atol=1e-5)


def test_smooth_mesh_along_axis() -> None:
    vertices = np.array([[0, 3, 10], [0, 0, 60], [4, 3, 30], [4, 0, 50]], dtype=np.float32)
    triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    neighbors, distances = get_vertex_neighbors(vertices, triangles)
    axis = 2
    smooth_vertices = mesh_smoothing_along_axis(vertices, neighbors, distances, axis=axis, smoothness=1.0)
    print("vertices\n", vertices)
    print("neighbors\n", neighbors)
    print("smooth_vertices\n", smooth_vertices)
    # Make sure the coordinates are unchanged along the non-smoothing dimensions
    fixed_axes = [0, 1, 2]
    fixed_axes.remove(axis)
    assert np.all(smooth_vertices[:, fixed_axes] == vertices[:, fixed_axes])
    # The smoothed vertex heights should be somewhere between the min and max heights of the original neighbors
    # This is not a strict test, allowing some space to improve the smoothing algorithm itself
    assert smooth_vertices[0, axis] > np.min(vertices[[1, 2], axis])
    assert smooth_vertices[0, axis] < np.max(vertices[[1, 2], axis])
    assert smooth_vertices[1, axis] > np.min(vertices[[0, 2, 3], axis])
    assert smooth_vertices[1, axis] < np.max(vertices[[0, 2, 3], axis])
    assert smooth_vertices[2, axis] > np.min(vertices[[0, 1, 3], axis])
    assert smooth_vertices[2, axis] < np.max(vertices[[0, 1, 3], axis])
    assert smooth_vertices[3, axis] > np.min(vertices[[1, 2], axis])
    assert smooth_vertices[3, axis] < np.max(vertices[[1, 2], axis])


def test_downsample_3d_points_on_2d_grid() -> None:
    points = np.array(
        [
            # Cell (0,0) with center (2.0, 0.0)
            [1.0, -1.0, 11.1],  # Bottom left point of the grid (bbox)
            [2.5, 0.5, 13.3],
            # Cell (1,1) with center (3.0, 2.0)
            [4.0, 2.0, -3.3],
            # Cell (3,2) with center (8.0, 4.0)
            [7.0, 3.0, 4.4],  # Top right point of the grid (bbox)
            # Cell (2,0) with center (6, 0.0)
            [5.5, -0.5, 2.0],
            [5.5, 0.5, 3.0],
            [6.5, 0.5, 4.0],
        ]
    )
    expected_points = np.array([[2.0, 0.0, 12.2], [6.0, 0.0, 3.0], [4.0, 2.0, -3.3], [8.0, 4.0, 4.4]])
    downsampled_points, downsampled_colors = downsample_3d_points_on_2d_grid(points, cell_size=2.0)
    assert downsampled_points.shape == (4, 3)
    assert downsampled_colors is None
    assert np.allclose(expected_points, downsampled_points, rtol=0.0, atol=1e-4)


def test_downsample_3d_points_on_2d_grid_with_colors() -> None:
    """Test that per-point colors are averaged per cell during downsampling."""
    # Same 7-point layout as test_downsample_3d_points_on_2d_grid (4 cells, cell_size=2.0)
    points = np.array(
        [
            # Cell (0,0): 2 points
            [1.0, -1.0, 11.1],
            [2.5, 0.5, 13.3],
            # Cell (1,1): 1 point
            [4.0, 2.0, -3.3],
            # Cell (3,2): 1 point
            [7.0, 3.0, 4.4],
            # Cell (2,0): 3 points
            [5.5, -0.5, 2.0],
            [5.5, 0.5, 3.0],
            [6.5, 0.5, 4.0],
        ],
        dtype=np.float32,
    )
    colors = np.array(
        [
            [100, 0, 50],  # Cell (0,0)
            [200, 100, 50],  # Cell (0,0)
            [10, 20, 30],  # Cell (1,1)
            [255, 255, 0],  # Cell (3,2)
            [60, 90, 120],  # Cell (2,0)
            [90, 90, 180],  # Cell (2,0)
            [150, 120, 0],  # Cell (2,0)
        ],
        dtype=np.uint8,
    )
    downsampled_points, downsampled_colors = downsample_3d_points_on_2d_grid(points, cell_size=2.0, colors=colors)

    assert downsampled_colors is not None
    assert downsampled_colors.shape == (4, 3)
    assert downsampled_colors.dtype == np.uint8

    # Build expected colors in the same cell order as downsampled_points.
    # Cell (0,0): mean([100,200], [0,100], [50,50]) = [150.0, 50.0, 50.0] -> uint8 [150, 50, 50]
    # Cell (2,0): mean([60,90,150], [90,90,120], [120,180,0]) = [100.0, 100.0, 100.0] -> [100, 100, 100]
    # Cell (1,1): single point -> [10, 20, 30]
    # Cell (3,2): single point -> [255, 255, 0]
    expected_colors = np.array(
        [
            [150, 50, 50],  # Cell (0,0)
            [100, 100, 100],  # Cell (2,0)
            [10, 20, 30],  # Cell (1,1)
            [255, 255, 0],  # Cell (3,2)
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(downsampled_colors, expected_colors)


def test_build_mesh_colors_propagated() -> None:
    """Test that colors survive downsampling and Delaunay meshing."""
    rng = np.random.RandomState(42)
    n_points = 50
    # Random XY in [0, 10], small Z variation
    points = np.column_stack(
        [rng.uniform(0, 10, n_points), rng.uniform(0, 10, n_points), rng.uniform(-0.1, 0.1, n_points)]
    ).astype(np.float32)
    colors = rng.randint(0, 256, (n_points, 3)).astype(np.uint8)

    mesher = DelaunayElevationMeshingAlgorithm(
        enable_downsampling=True,
        voxel_size=5.0,  # Large voxel to force merging
        smoothing_passes=0,
    )
    smooth_vertices, triangles, vertices, vertex_colors = mesher.build_mesh_from_points(points, colors=colors)

    assert vertex_colors is not None
    assert vertex_colors.shape[0] == smooth_vertices.shape[0]
    assert vertex_colors.shape[1] == 3
    assert vertex_colors.dtype == np.uint8
    assert triangles.shape[1] == 3


def test_build_mesh_no_downsampling_colors_passthrough() -> None:
    """Without downsampling, colors pass through unchanged."""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128]], dtype=np.uint8)

    mesher = DelaunayElevationMeshingAlgorithm(enable_downsampling=False, smoothing_passes=0)
    _, _, _, vertex_colors = mesher.build_mesh_from_points(points, colors=colors)

    assert vertex_colors is not None
    assert np.array_equal(vertex_colors, colors)


def test_build_mesh_colors_none() -> None:
    """No colors passed -> vertex_colors is None."""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)

    mesher = DelaunayElevationMeshingAlgorithm(enable_downsampling=False, smoothing_passes=0)
    _, _, _, vertex_colors = mesher.build_mesh_from_points(points)

    assert vertex_colors is None
