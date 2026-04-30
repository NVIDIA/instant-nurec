# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import tempfile
import time

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, TypeAlias

import numpy as np
import point_cloud_utils as pcu
import scipy
import torch

from poisson_recon import reconstruct_surface
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
from scipy.sparse import lil_matrix

from nre.utils.io.ply import save_ply
from nre.utils.io.utils import USDReferences, initialize_usd_stage
from nre.utils.misc import unpack_optional
from nre.utils.types import (
    ArtifactContents,
    FrameConversion,
    NamedSerialized,
    NamedUSDStage,
    PointCloud,
    RayFlags,
    RigTrajectories,
)


logger = logging.getLogger(__name__)

MeshLabels: TypeAlias = Dict[str, List[int]]


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    labels: MeshLabels | None = None
    colors: np.ndarray | None = None  # Per-vertex RGB colors [V,3] uint8, optional


def clean_faces_with_duplicate_vertices(faces: np.ndarray) -> np.ndarray:
    # WAR for PCU bug: https://github.com/fwilliams/point-cloud-utils/issues/94
    faces_cleaned = []
    for face in faces:
        if len(face) == len(set(face)):
            faces_cleaned.append(face)
    return np.stack(faces_cleaned)


def poisson_reconstruction(
    points: np.ndarray,
    point_origins: np.ndarray,
    logger: logging.Logger,
    n_neighbors: int,
    trim_distance: float,
) -> Mesh:
    logger.info("Estimating normal vectors")
    start_time = time.perf_counter()
    dirs = point_origins - points
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    point_indices, normals_filtered = pcu.estimate_point_cloud_normals_knn(
        points, n_neighbors, view_directions=dirs, drop_angle_threshold=np.pi / 2
    )
    points_filtered = points[point_indices]
    elapsed_time = time.perf_counter() - start_time
    logger.info(f"{len(normals_filtered)} normals computed from k-NNs (k={n_neighbors}) in {elapsed_time:.3f} s")

    start_time = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".ply") as temp_fp, tempfile.NamedTemporaryFile(suffix=".ply") as temp_rp:
        pcu.save_mesh_vn(temp_fp.name, points_filtered, normals_filtered)
        logger.info("Running Poisson surface reconstruction")
        reconstruct_surface(
            input_file=temp_fp.name, output_file=temp_rp.name, width=0.1, density=True, samples_per_node=1.0
        )
        v, f = pcu.load_mesh_vf(temp_rp.name)
    elapsed_time = time.perf_counter() - start_time
    logger.info(f"Mesh of {len(v)} vertices and {len(f)} faces computed in {elapsed_time:.3f} s")

    start_time = time.perf_counter()
    logger.info("Trimming the reconstructed mesh")
    nn_dist, _ = pcu.k_nearest_neighbors(v.astype(np.float32), points_filtered.astype(np.float32), k=2)
    nn_dist = nn_dist[:, 1]
    f_mask = np.stack([nn_dist[f[:, i]] < trim_distance for i in range(f.shape[1])], axis=-1)
    f_mask = np.all(f_mask, axis=-1)
    f = f[f_mask]
    elapsed_time = time.perf_counter() - start_time
    logger.info(f"Mesh trimmed to {len(f)} faces in {elapsed_time:.3f} s")

    start_time = time.perf_counter()
    logger.info("Removing duplicate and unreferenced vertices")
    v_clean, f_clean, _, _ = pcu.deduplicate_mesh_vertices(v, f, 1e-7)
    f_clean = clean_faces_with_duplicate_vertices(f_clean)

    v_clean, f_clean, _, _ = pcu.remove_unreferenced_mesh_vertices(v_clean, f_clean)
    elapsed_time = time.perf_counter() - start_time
    logger.info(f"Mesh cleaned up ({len(v_clean)} vertices, {len(f_clean)} faces) in {elapsed_time:.3f} s")

    return Mesh(vertices=v_clean, faces=f_clean)


def smooth_mesh(
    mesh: Mesh,
    rig_trajectories: RigTrajectories,
    filter_radius_sq: float = 49.0,
    vert_angle_thresh_cos: float = 0.9,
    eigenvector_count: int = 80,
) -> Mesh:
    logger = logging.getLogger(__name__)

    camera_positions_l = []
    for rt in rig_trajectories.rig_trajectories:
        rig_poses = rt.T_rig_worlds
        for i in range(len(rig_poses)):
            camera_positions_l.append(list(rig_poses[i, :3, 3]))
    camera_positions = np.array(camera_positions_l)

    # Part one: Filter road mesh
    v_input = np.copy(mesh.vertices)
    f_input = np.copy(mesh.faces)

    logger.info("Mesh smoothing: Selecting close vertices")

    close_vertex_mask = np.full(v_input.shape[0], False, dtype=np.bool_)
    for i in range(camera_positions.shape[0]):
        distance_squared = np.sum(np.square(v_input - camera_positions[i, :]), axis=1)
        close_vertex_mask |= distance_squared < filter_radius_sq

    v, f = pcu.remove_mesh_vertices(v_input, f_input, close_vertex_mask)
    v_input_indices = np.flatnonzero(close_vertex_mask)  # tracks the source indices of the current set of vertices

    logger.info("Mesh smoothing: Filter faces that are not horizontal enough")

    # n is a NumPy array of shape [nf, 3] where n[i] is the normal of face f[i]
    n = pcu.estimate_mesh_face_normals(v, f)

    z_thresh = n[:, 2] >= vert_angle_thresh_cos
    v, f, vidxs, _ = pcu.remove_unreferenced_mesh_vertices(v, f[z_thresh])
    v_input_indices = v_input_indices[vidxs]

    logger.info("Mesh smoothing: Find the connected component which corresponds to the road")
    # cf is the index of the connected component of each face
    # nf is the number of faces per connected component
    _, _, cf, nf = pcu.connected_components(v, f)

    max_cluster_index = np.argmax(nf)

    v, f, vidxs, _ = pcu.remove_unreferenced_mesh_vertices(v, f[cf == max_cluster_index])
    v_input_indices = v_input_indices[vidxs]

    # Part two: filtering
    logger.info("Mesh smoothing: Estimating Laplacian")

    # We want to filter the road geometry by running a low pass filter on the mesh.
    # This is because the mesh needs to be locally smooth and other filtering
    # techniques dampen but don't remove high frequency noise.
    adj_list = pcu.adjacency_list(f)

    vertex_count = v.shape[0]
    edges = {}

    for i in range(len(adj_list)):
        for other in adj_list[i]:
            edges[(i, other)] = 1

    adjacency_matrix = scipy.sparse.dok_matrix((vertex_count, vertex_count), dtype=np.int32)
    adjacency_matrix._update(edges)

    diagonals = scipy.sparse.spdiags(np.sum(adjacency_matrix, axis=0), 0, vertex_count, vertex_count)
    laplacian = diagonals - adjacency_matrix

    # We select a certain number of lowest eigenvectors of the laplacian which correspond to the lowest frequencies
    logger.info("Mesh smoothing: Eigenvector estimation")
    eigen_values, eigen_vectors = scipy.sparse.linalg.eigs(laplacian.asfptype(), eigenvector_count, which="SM")
    eigen_value_order = np.argsort(eigen_values)
    eigen_values = scipy.sparse.diags(eigen_values)
    eigen_vectors = eigen_vectors[:, eigen_value_order].real

    # Projecting the vertices onto the eigenvectors gives us the smoothed mesh
    logger.info("Mesh smoothing: Projecting vertices onto eigenvectors")
    projected_verts = np.matmul(v.T, eigen_vectors)
    verts_smoothed = np.matmul(projected_verts, eigen_vectors.T).T

    # Part three: Combining
    logger.info("Mesh smoothing: Combining existing mesh with new smoothed data")

    # Map the vertices in the smoothed mesh to those in the old mesh
    v_input[v_input_indices] = verts_smoothed

    # Assumption: smoothing method did not add or change the order of any faces, so that labels stay the same.
    return Mesh(vertices=v_input, faces=f_input, labels=mesh.labels)


def segment_mesh_road_nonroad(
    mesh: Mesh,
    points: np.ndarray,  # lidar points
    point_flags: np.ndarray,
    logger: logging.Logger,
) -> Mesh:
    def compute_adjacency_list(faces):
        """Computes face adjacency of a mesh"""
        num_faces = len(faces)
        adjacency_list = lil_matrix((num_faces, num_faces), dtype=bool)
        edge_to_faces = {}

        for i, face in enumerate(faces):
            assert len(set(list(face))) == len(list(face))
            edges = [(face[j], face[(j + 1) % 3]) for j in range(3)]
            for edge in edges:
                edge = tuple(sorted(edge))
                if edge in edge_to_faces:
                    for adj_face in edge_to_faces[edge]:
                        assert i != adj_face
                        adjacency_list[i, adj_face] = True
                        adjacency_list[adj_face, i] = True
                    edge_to_faces[edge].append(i)
                else:
                    edge_to_faces[edge] = [i]

        return adjacency_list

    def propagate_labels(face_labels, adjacency_list):
        """Propagate labels into unlabelled faces via BFS"""
        # Seed queue with all labelled faces. [0] is necessary because np.where returns a tuple even for 1D arrays.
        queue = deque(np.where(face_labels != -1)[0])
        while queue:
            current_face = queue.popleft()
            current_label = face_labels[current_face]
            neighbors = adjacency_list.rows[current_face]
            for neighbor in neighbors:
                if face_labels[neighbor] == -1:
                    face_labels[neighbor] = current_label
                    queue.append(neighbor)
        return face_labels

    segmentation_mask_bckg = ~((point_flags & (RayFlags.INVALID | RayFlags.ROAD_SEMANTIC)) != 0)
    segmentation_mask_road = (point_flags & RayFlags.ROAD_SEMANTIC) != 0

    points_bckgs = points[segmentation_mask_bckg, :]  # idx1 contains indices of points in background
    points_road = points[segmentation_mask_road, :]  # idx2 contains indices of points on the road

    logger.info("Projecting lidar points on the mesh...")
    start = time.time()
    _, face_ids_bckg, _ = pcu.closest_points_on_mesh(points_bckgs.astype(np.float32), mesh.vertices, mesh.faces)
    _, face_ids_road, _ = pcu.closest_points_on_mesh(points_road.astype(np.float32), mesh.vertices, mesh.faces)
    end = time.time()
    logger.info(f"Projecting lidar points on the mesh: {end - start} sec.")

    # Compute the adjacency list
    logger.info("Computing face adjacency list...")
    start = time.time()
    adjacency_list = compute_adjacency_list(mesh.faces)
    end = time.time()
    logger.info(f"Computing adjacency list: {end - start} sec.")

    logger.info("Propagating labels...")
    # Initialize face labels (-1 for unlabeled, 1 for background, 2 for road)
    face_labels = np.full(len(mesh.faces), -1)
    face_labels[face_ids_bckg] = 1
    face_labels[face_ids_road] = 2
    start = time.time()
    # Propagate labels
    face_labels = propagate_labels(face_labels, adjacency_list)
    end = time.time()
    logger.info(f"Propagating labels: {end - start} sec.")

    logger.info("Find max connected component corresponding to road...")
    # Extract separate meshes for L1 and L2
    mesh_bckg_faces = mesh.faces[face_labels == 1]
    mesh_road_faces = mesh.faces[face_labels == 2]
    start = time.time()
    # cf is the index of the connected component of each face
    # nf is the number of faces per connected component
    _, _, cf, nf = pcu.connected_components(mesh.vertices, mesh_road_faces)
    max_cluster_index = np.argmax(nf)
    mesh_bckg_faces = np.concatenate((mesh_bckg_faces, mesh_road_faces[cf != max_cluster_index]), axis=0)
    mesh_road_faces = mesh_road_faces[cf == max_cluster_index]
    end = time.time()
    logger.info(f"Find max connected component corresponding to road: {end - start} sec.")

    faces = np.vstack((mesh_bckg_faces, mesh_road_faces))
    labels = {
        "nonroad": list(range(0, len(mesh_bckg_faces))),
        "road": list(range(len(mesh_bckg_faces), len(faces))),
    }

    return Mesh(vertices=mesh.vertices, faces=faces, labels=labels)


def mesh_from_point_cloud(
    point_cloud: PointCloud,
    n_neighbors: int,
    trim_distance: float,
    apply_road_segmentation: bool,
    source_to_target: FrameConversion | None = None,
) -> Mesh:
    logger = logging.getLogger(__name__)

    points_have_semantics = False
    if point_cloud.flags is not None:
        points_have_semantics = bool(
            torch.any(torch.bitwise_and(point_cloud.flags, RayFlags.VALID_SEMANTIC.value)).item()
        )
    if apply_road_segmentation and not points_have_semantics:
        raise ValueError(f"Mesh semantic segmentation requested, but point cloud does not have semantic data!")

    if source_to_target is not None:
        logger.info("Transforming points back to target frame")
        start_time = time.perf_counter()
        points, point_origins = (
            source_to_target.transform_points(np.array(point_cloud.xyz_end)),
            source_to_target.transform_points(np.array(point_cloud.xyz_start)),
        )
        elapsed_time = time.perf_counter() - start_time
        logger.info(f"{len(points)} points transformed in {elapsed_time:.3f} s")

        # "transform_points" changes the row / col major ordering, so fix again.
        # This is necessary for k_nearest_neighbors.
        points = np.copy(points, order="C")
        point_origins = np.copy(point_origins, order="C")
    else:
        points, point_origins = (np.array(point_cloud.xyz_end), np.array(point_cloud.xyz_start))

    logger.info("Constructing mesh")
    mesh = poisson_reconstruction(points, point_origins, logger, n_neighbors, trim_distance)

    if apply_road_segmentation:
        logger.info("Segmenting mesh")
        start_time = time.perf_counter()
        point_flags = np.array(unpack_optional(point_cloud.flags))
        mesh = segment_mesh_road_nonroad(mesh, points, point_flags, logger)
        elapsed_time = time.perf_counter() - start_time
        logger.info(f"Segmenting mesh took {elapsed_time:.3f} s")

    return mesh


def translate_nre_label_to_drive_sim_label(label: str):
    translation = {"nonroad": "non-drivable", "road": "drivable"}
    if not label in translation:
        raise ValueError(f"Unknown label: {label}")
    return translation[label]


def serialize_mesh_usd(
    mesh: Mesh,
) -> Usd.Stage:
    stage = initialize_usd_stage()

    # Create mesh
    mesh_path = "/World/mesh"
    usd_mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh_prim = usd_mesh.GetPrim()

    usd_mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(mesh.vertices))
    # Assume that all faces contain the same number of vertices!
    usd_mesh.GetFaceVertexCountsAttr().Set([mesh.faces.shape[1]] * mesh.faces.shape[0])
    usd_mesh.GetFaceVertexIndicesAttr().Set(mesh.faces)

    # Set mesh primvars important for using as matte object
    primvars_api = UsdGeom.PrimvarsAPI(mesh_prim)
    primvars_api.CreatePrimvar("hideForCamera", Sdf.ValueTypeNames.Bool).Set(False)
    primvars_api.CreatePrimvar("invisibleToSecondaryRays", Sdf.ValueTypeNames.Bool).Set(False)
    primvars_api.CreatePrimvar("isMatteObject", Sdf.ValueTypeNames.Bool).Set(True)

    # Write per-vertex display colors if available (uint8 RGB -> float32 [0,1])
    if mesh.colors is not None:
        colors_float = mesh.colors.astype(np.float32) / 255.0
        display_color = primvars_api.CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
        )
        display_color.Set(Vt.Vec3fArray.FromNumpy(colors_float))

    def apply_diffuse(usd_path: str, usd_prim):
        # Create and apply diffuse material
        material = UsdShade.Material.Define(stage, usd_path)
        pbr_shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild("pbrShader"))
        pbr_shader.CreateIdAttr("UsdPreviewSurface")
        pbr_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        pbr_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        pbr_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5, 0.5, 0.5))
        material.CreateSurfaceOutput().ConnectToSource(pbr_shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(usd_prim).Bind(material)

    def apply_semantics(prim, label):
        prim.AddAppliedSchema("SemanticsAPI:Semantics")
        prim.CreateAttribute("semantic:Semantics:params:semanticType", Sdf.ValueTypeNames.String, False).Set("class")
        prim.CreateAttribute("semantic:Semantics:params:semanticData", Sdf.ValueTypeNames.String, False).Set(
            translate_nre_label_to_drive_sim_label(label)
        )

    if mesh.labels is None or len(mesh.labels) == 1:
        apply_diffuse("/World/material", mesh_prim)
        if mesh.labels:
            apply_semantics(mesh_prim, next(iter(mesh.labels)))
    else:
        # Create subset for each label and apply semantics schema
        for label, idcs in mesh.labels.items():
            geom_subset = UsdGeom.Subset.Define(stage, mesh_prim.GetPath().AppendChild(label))
            geom_subset_prim = geom_subset.GetPrim()
            geom_subset_prim.CreateAttribute("elementType", Sdf.ValueTypeNames.Token).Set("face")
            geom_subset_prim.CreateAttribute("familyName", Sdf.ValueTypeNames.Token).Set("materialBind")

            # Define the indices for each subset based on semantic labels
            indices_subset = Vt.IntArray(idcs)
            geom_subset.GetIndicesAttr().Set(indices_subset)

            apply_diffuse(f"/World/material_{label}", geom_subset_prim)
            apply_semantics(geom_subset_prim, label)

    return stage


def get_usd_references(meshes: ArtifactContents) -> USDReferences:
    res: USDReferences = []
    for mesh in meshes:
        if isinstance(mesh, NamedUSDStage):
            # Inspect stage and extract all mesh prims
            for prim in mesh.stage.Traverse():
                if prim.IsA(UsdGeom.Mesh):
                    # Assume that all files are stored in the same directory, hence the path is equal to the filename.
                    res.append((mesh, str(prim.GetPath())))
    return res


def extract_sub_mesh_with_label(mesh: Mesh, label: str):
    if mesh.labels is None or label not in mesh.labels:
        raise ValueError(f"Requested label is not in mesh: {label}")
    idcs = mesh.labels[label]
    v, f, _, _ = pcu.remove_unreferenced_mesh_vertices(mesh.vertices, mesh.faces[idcs])
    return Mesh(vertices=v, faces=f, labels={label: list(range(0, len(f)))})


def serialize_mesh(
    mesh: Mesh,
    export_disjoint_meshes: bool,
    filename: str = "mesh",
    formats: List[Literal["ply", "usd"]] = ["ply", "usd"],
) -> ArtifactContents:
    res = []
    # If disjoint meshes are requested, split the mesh and call serialize_mesh recursively.
    if export_disjoint_meshes:
        if mesh.labels is None:
            raise ValueError("Disjoint mesh export was requested but mesh contains no labels!")
        for label in mesh.labels:
            res.extend(
                serialize_mesh(
                    mesh=extract_sub_mesh_with_label(mesh, label),
                    export_disjoint_meshes=False,
                    filename=filename + "_" + label,
                    formats=formats,
                )
            )
    else:
        for file_format in formats:
            filename_with_suffix = filename + "." + file_format
            match file_format:
                case "ply":
                    with tempfile.NamedTemporaryFile(suffix="." + file_format) as tmp_file:
                        save_ply(tmp_file.name, mesh.vertices, mesh.faces, colors=mesh.colors)
                        res.append(NamedSerialized(filename=filename_with_suffix, serialized=tmp_file.read()))
                case "usd" | "usda":
                    res.append(NamedUSDStage(filename=filename_with_suffix, stage=serialize_mesh_usd(mesh)))
                case _:
                    raise ValueError(f"The following mesh format is not supported: {file_format}")
    return res
