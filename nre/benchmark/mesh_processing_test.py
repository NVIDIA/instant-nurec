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

import numpy as np

from nre.benchmark.mesh_processing import find_boundaries, find_boundary_edges, link_boundary_edges


class TwoFaceMeshTests(unittest.TestCase):
    def setUp(self):
        self.mesh_faces = np.array([[0, 1, 3], [3, 1, 2]], dtype=np.int32)
        self.expected_boundary_vertex_chains = [[0, 3, 2, 1]]
        self.expected_boundary_edge_chains = [[0, 1, 3, 2]]

    def test_find_boundary_edges(self):
        edges = find_boundary_edges(self.mesh_faces)
        np.testing.assert_array_equal(edges, np.array([[0, 1], [0, 3], [1, 2], [2, 3]], dtype=np.int32))

    def test_link_boundary_edges(self):
        edges = find_boundary_edges(self.mesh_faces)
        vertex_chains, edge_chains = link_boundary_edges(edges)
        assert vertex_chains == self.expected_boundary_vertex_chains
        assert edge_chains == self.expected_boundary_edge_chains

    def test_find_boundaries(self):
        boundaries = find_boundaries(self.mesh_faces)
        assert boundaries == self.expected_boundary_vertex_chains


class ManifoldMeshWithHole(unittest.TestCase):
    def setUp(self):
        # And rectangle-shaped mesh over a 4x4 grid of vertices, with a hole in the middle, unoriented faces
        # Boundary polygons:
        #   Outer: 0-1-2-3-7-11-15-14-13-12-8-4-0 (12 vertices clockwise)
        #   Hole: 5-6-10-9 (4 vertices clockwise)
        self.mesh_faces = np.array(
            [
                [0, 1, 4],
                [1, 5, 4],
                [1, 2, 5],
                [5, 6, 2],
                [2, 3, 6],
                [3, 7, 6],
                [4, 5, 8],
                [5, 9, 8],
                # [5, 6, 9],  # Excluded triangle to make a hole
                # [9, 10, 6], # Excluded triangle to make a hole
                [6, 7, 10],
                [7, 11, 10],
                [8, 9, 12],
                [9, 13, 12],
                [9, 10, 13],
                [13, 14, 10],
                [10, 11, 14],
                [11, 15, 14],
            ],
            dtype=np.int32,
        )
        self.expected_boundary_vertex_chains = [[0, 4, 8, 12, 13, 14, 15, 11, 7, 3, 2, 1], [5, 9, 10, 6]]

    def test_find_boundary_edges(self):
        edges = find_boundary_edges(self.mesh_faces)
        print(edges)
        np.testing.assert_array_equal(
            edges,
            np.array(
                [
                    [0, 1],
                    [0, 4],
                    [1, 2],
                    [2, 3],
                    [3, 7],
                    [4, 8],
                    [5, 6],  # Hole edge
                    [5, 9],  # Hole edge
                    [6, 10],  # Hole edge
                    [7, 11],
                    [8, 12],
                    [9, 10],  # Hole edge
                    [11, 15],
                    [12, 13],
                    [13, 14],
                    [14, 15],
                ],
                dtype=np.int32,
            ),
        )

    def test_link_boundary_edges(self):
        edges = find_boundary_edges(self.mesh_faces)
        vertex_chains, edge_chains = link_boundary_edges(edges)
        assert vertex_chains == self.expected_boundary_vertex_chains
        # TODO: work out expected value for edge chains

    def test_find_boundaries(self):
        boundaries = find_boundaries(self.mesh_faces)
        assert boundaries == self.expected_boundary_vertex_chains
