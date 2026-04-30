# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Dict, List, Tuple

import numpy as np

from numpy.typing import NDArray


def find_boundary_edges(mesh_faces: NDArray[np.int32]) -> NDArray[np.int32]:
    """Find all boundary edges of a mesh and return them in a (E, 2) array"""

    assert mesh_faces.ndim == 2, "Mesh faces must be a 2D array"
    assert mesh_faces.shape[1] == 3, "Mesh faces must be triangles"

    edges = np.vstack([mesh_faces[:, [0, 1]], mesh_faces[:, [1, 2]], mesh_faces[:, [2, 0]]])
    assert edges.ndim == 2
    assert edges.shape[1] == 2

    # Sort each edge so that lower vertex index comes first
    edges = np.sort(edges, axis=1)

    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    assert len(counts) == len(unique_edges)

    # Boundary edges are edges that have only one adjacent triangle, and therefore are listed only once
    boundary_edges = unique_edges[counts == 1, :]

    return boundary_edges


def link_boundary_edges(boundary_edges: NDArray[np.int32]) -> Tuple[List[List[int]], List[List[int]]]:
    """Given a list of boundary edges, link them into polygons and return the vertex and edge chains per boundary"""

    assert boundary_edges.ndim == 2
    assert boundary_edges.shape[1] == 2

    # Map from vertex indices to edge indices
    vertex_to_edges: Dict[int, List[int]] = {}
    for edge_idx, edge in enumerate(boundary_edges):
        v0, v1 = edge
        try:
            vertex_to_edges[v0].append(edge_idx)
        except KeyError:
            vertex_to_edges[v0] = [edge_idx]
        try:
            vertex_to_edges[v1].append(edge_idx)
        except KeyError:
            vertex_to_edges[v1] = [edge_idx]

    # Link edges into polygons

    edge_chains = []
    vertex_chains = []

    edge_mask = np.zeros(len(boundary_edges), dtype=bool)

    # Find first unvisited edge to start a new polygon from
    unvisited_edges = np.nonzero(~edge_mask)[0]

    while len(unvisited_edges) > 0:
        # Initialize new polygon to trace
        current_edge_idx = unvisited_edges[0]
        current_vertex_idx = boundary_edges[current_edge_idx][0]
        edge_chain = [current_edge_idx]
        vertex_chain = [current_vertex_idx]

        # Link the polygon until no more unvisited edges remain in the adjacency graph
        while 1:
            current_edge_idx = edge_chain[-1]
            current_vertex_idx = vertex_chain[-1]

            edge_mask[current_edge_idx] = True  # Mark edge as visited

            # All edges adjacent to the current vertex
            edge_candidates: List[int] = vertex_to_edges[current_vertex_idx]

            next_edge_idx = -1
            for i in edge_candidates:
                if not edge_mask[i]:
                    next_edge_idx = i
                    break

            if next_edge_idx == -1:
                # No unvisited edge found, we're done tracing this polygon
                break

            edge_chain.append(next_edge_idx)

            v0, v1 = boundary_edges[next_edge_idx, :]
            next_vertex_idx = v1 if v0 == current_vertex_idx else v0
            vertex_chain.append(next_vertex_idx)

        # Add the polygon to the list of polygons
        vertex_chains.append(vertex_chain)
        edge_chains.append(edge_chain)

        # Find unvisited edges to start a new polygon from
        unvisited_edges = np.nonzero(~edge_mask)[0]

    return vertex_chains, edge_chains


def find_boundaries(mesh_faces: NDArray[np.int32]) -> List[List[int]]:
    """Find all boundaries of a mesh incl. boundaries of holes, and return the list of vertex chains per boundary.

    Args:
        mesh_vertices: (M, 3) array of mesh vertices.
        mesh_faces: (F, 3) array of mesh faces.

    Returns:
        List[List[int]] - A list of closed boundaries, each boundary being a list of vertex indices.
    """
    boundary_edges = find_boundary_edges(mesh_faces)
    vertex_chains, _ = link_boundary_edges(boundary_edges)
    return vertex_chains
