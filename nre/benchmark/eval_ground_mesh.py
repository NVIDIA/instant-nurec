# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import io
import json
import logging
import os
import time

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import numpy as np
import point_cloud_utils as pcu
import trimesh
import yaml

from numpy.typing import NDArray

from nre.artifact import Artifact
from nre.benchmark.mesh_processing import find_boundaries
from nre.utils.io.ply import save_ply  # type: ignore
from nre.utils.types import RigTrajectories


log = logging.getLogger(__name__)


class Polygon:
    def __init__(self, vertices: NDArray[np.float32]):
        assert vertices.ndim == 2
        assert vertices.shape[1] == 3 or vertices.shape[1] == 2
        self._vertices = vertices

    @property
    def vertices(self) -> NDArray[np.float32]:
        return self._vertices

    @property
    def num_vertices(self) -> int:
        return len(self._vertices)

    def total_length(self) -> float:
        return (
            np.sum(np.linalg.norm(self._vertices[1:] - self._vertices[:-1], axis=1)).item()
            if len(self._vertices) > 1
            else 0.0
        )

    def uniform_resampling(self, num_samples: int) -> NDArray[np.float32]:
        assert num_samples > 1, "Number of samples must be greater than 1"

        # Arc-length parameters per vertex, i.e. distance of each vertex along the polygon from the first vertex.
        vertex_arclen = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(self._vertices[1:] - self._vertices[:-1], axis=1))]
        )
        sample_arclen = np.linspace(0, vertex_arclen[-1], num_samples)

        # Linear interpolation of the vertices to the desired number of samples.
        # TODO: Test this under repeated vertices.
        xi = np.interp(sample_arclen, vertex_arclen, self._vertices[:, 0]).astype(np.float32)
        yi = np.interp(sample_arclen, vertex_arclen, self._vertices[:, 1]).astype(np.float32)
        zi = np.interp(sample_arclen, vertex_arclen, self._vertices[:, 2]).astype(np.float32)

        return np.stack([xi, yi, zi], axis=1).astype(np.float32)


def load_ground_mesh_from_artifact(artifact: Artifact) -> Tuple[NDArray[np.float32], NDArray[np.int32]]:
    mesh_bytes_io = io.BytesIO(artifact.ground_mesh_ply)
    mesh = trimesh.load(mesh_bytes_io, file_type="ply")
    return np.array(mesh.vertices).astype(np.float32), np.array(mesh.faces).astype(np.int32)


def load_ground_mesh_from_ply(ply_path: str) -> Tuple[NDArray[np.float32], NDArray[np.int32]]:
    mesh = trimesh.load(ply_path, file_type="ply")
    return np.array(mesh.vertices).astype(np.float32), np.array(mesh.faces).astype(np.int32)


def load_rig_trajectory_from_dict(rig_trajectories_dict: Dict[str, Any]) -> NDArray[np.float32]:
    """Import rig poses from a dictionary output by the NRE training and return them as a (N, 4, 4) numpy array"""
    rig_trajectories = RigTrajectories.from_dict(rig_trajectories_dict)
    if len(rig_trajectories.rig_trajectories) == 0:
        raise ValueError("No rig trajectories found in the artifact")
    rig_trajectory = rig_trajectories.rig_trajectories[0]
    T_rig_worlds = rig_trajectory.T_rig_worlds
    return T_rig_worlds.cpu().numpy().astype(np.float32)


def load_rig_trajectory_from_artifact(artifact: Artifact) -> NDArray[np.float32]:
    """Import rig poses from an artifact and return them as a (N, 4, 4) numpy array"""
    return load_rig_trajectory_from_dict(artifact.rig_trajectories)


def load_rig_trajectory_from_json(json_path: str) -> NDArray[np.float32]:
    """Import rig poses from a JSON file output by the NRE training and return them as a (N, 4, 4) numpy array"""
    with open(json_path, "r") as f:
        rig_trajectories_dict = json.load(f)
    return load_rig_trajectory_from_dict(rig_trajectories_dict)


def calculate_vehicle_trajectory_vs_ground_consistency_metrics(
    mesh_vertices: NDArray[np.float32],
    mesh_faces: NDArray[np.int32],
    T_rig_worlds: NDArray[np.float32],
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Compare the rig trajectory with the ground mesh and calculate coverage and distance metrics.

    Assumption: rig positions are on the nominal ground under the vehicle. A discrepancy between the rig positions
    and the ground mesh may only be due to vehicle suspension.

    Args:
        mesh_vertices: (M, 3) array of mesh vertices.
        mesh_faces: (F, 3) array of mesh faces.
        T_rig_worlds: (N, 4, 4) array of poses, as 4x4 transformations from the rig frame to the world frame.
        output_dir: Optional directory path for exporting diagnostic data.

    Returns:
        A dictionary of metrics that maps metric name to value.
    """

    # Defaulting to the worst possible values to make it possible to return from the function in case of an error.
    # TODO: Replace with dataclass for typed metrics (discriminate float vs. int values)
    metrics = {
        "trajectory_sample_count": len(T_rig_worlds),
        "trajectory_length_meters": 0.0,
        "trajectory_ground_coverage_percentage": 0.0,
        "trajectory_ground_distance_max_meters": np.inf,
        "trajectory_ground_distance_med_meters": np.inf,
    }

    if len(mesh_vertices) < 3 or len(mesh_faces) == 0:
        log.warning("Mesh does not have a single valid face, skipping metrics")
        return metrics

    if len(T_rig_worlds) == 0:
        log.warning("Rig trajectory is empty, skipping trajectory-based metrics")
        return metrics

    # Extract rig positions from the rig trajectory.
    # Rig positions are expected in the world frame, on the nominal ground under the vehicle (NCore convention).
    # Spacing of adjacent positions is based on vehicle speed and is non-uniform.
    rig_positions = np.ascontiguousarray(T_rig_worlds[:, :3, 3]).astype(np.float32)

    # Add trajectory length to the metrics.
    rig_positions_polygon = Polygon(rig_positions)
    trajectory_length_meters = rig_positions_polygon.total_length()
    metrics["trajectory_length_meters"] = trajectory_length_meters
    log.info(f"Trajectory length: {trajectory_length_meters:.2f} meters")

    if trajectory_length_meters == 0.0:
        log.warning("Trajectory length is zero, skipping trajectory-based metrics")
        return metrics

    # TODO: Resample rig trajectory uniformly along the trajectory to improve accuracy of the quality metrics.

    # Extract the per-sample vehicle up and lateral directions in the world frame.
    # These are the vehicle Z- and Y-axis directions in the world frame.
    vehicle_up_directions = np.ascontiguousarray(T_rig_worlds[:, :3, 2])

    # Note that any per-vertex shift of a curve may cause order reversion in space, so smoothness is not preserved,
    # i.e. do not calculate smoothness metrics on shifted curves in the future.
    rig_positions_up_shifted = rig_positions + vehicle_up_directions

    # Rig positions are in the nominal ground under the vehicle by NCore convention.
    # Project rig positions shifted in the vehicle up direction back onto the mesh in the vehicle down direction.
    # If there is no intersection in ray direction from the ray origin, the face index is -1 and the distance is inf.
    # The intersections are in the direction of the ray, not in the opposite direction, so distances are not signed.
    intersector = pcu.RayMeshIntersector(mesh_vertices, mesh_faces)
    ray_origins = rig_positions_up_shifted
    # ray_origins = rig_positions - vehicle_up_directions # For testing when there is no valid ground projection
    ray_directions = -vehicle_up_directions
    face_indices, bary_coords, ray_intersection_distances = intersector.intersect_rays(ray_origins, ray_directions)

    ground_projection_mask = face_indices >= 0 & np.isfinite(ray_intersection_distances)
    ground_projections = ray_origins + ray_directions * ray_intersection_distances.reshape(-1, 1)
    ground_projections = ground_projections[ground_projection_mask]

    metrics["trajectory_ground_coverage_percentage"] = 100.0 * ground_projection_mask.sum().item() / len(ray_origins)
    log.info(f"Trajectory ground coverage: {metrics['trajectory_ground_coverage_percentage']:.2f}%")

    # Calculate trajectory (on nominal ground) vs. ground mesh distance on rig positions covered by the ground mesh.
    if len(ground_projections) > 0:
        trajectory_ground_distances = np.linalg.norm(ground_projections - rig_positions[ground_projection_mask], axis=1)
        metrics["trajectory_ground_distance_max_meters"] = trajectory_ground_distances.max().item()
        metrics["trajectory_ground_distance_med_meters"] = trajectory_ground_distances.mean().item()
        log.info(f"Trajectory-ground distance max: {metrics['trajectory_ground_distance_max_meters']:.2f} meters")
        log.info(f"Trajectory-ground distance median: {metrics['trajectory_ground_distance_med_meters']:.2f} meters")
    else:
        log.warning("No ground projections found, skipping trajectory-ground distance metrics")

    # TODO: Uniform resampling of the trajectory before calculating the above metrics.
    # (Also needs to handle the edge-case when trajectory is shorter than the sampling distance.)

    # TODO: Calculate the above metrics not only along the rig trajectory samples but in a vehicle-wide stripe.

    # TODO: Export plot of trajectory ground distances vs the distance along the trajectory

    if output_dir is not None:
        # Save rig positions to a PLY for visualization
        save_ply(os.path.join(output_dir, "rig_positions.ply"), vertices=rig_positions, logger=log)
        save_ply(
            os.path.join(output_dir, "rig_positions_up_shifted.ply"), vertices=rig_positions_up_shifted, logger=log
        )
        save_ply(os.path.join(output_dir, "trajectory_ground_projections.ply"), vertices=ground_projections, logger=log)

    return metrics


def calculate_ground_mesh_metrics(
    mesh_vertices: NDArray[np.float32],
    mesh_faces: NDArray[np.int32],
    T_rig_worlds: NDArray[np.float32],
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Calculate quality metrics for the ground mesh and optionally export diagnostic data.

    Args:
        mesh_vertices: (M, 3) array of mesh vertices.
        mesh_faces: (F, 3) array of mesh faces.
        T_rig_worlds: (N, 4, 4) array of poses, as 4x4 transformations from the rig frame to the world frame.
        output_dir: Optional directory path for exporting diagnostic data.

    Returns:
        A list of metrics as a mapping from metric name to value.
    """

    # TODO: Replace with dataclass for typed metrics (discriminate float vs. int values)
    metrics = {
        "mesh_area_m2": 0.0,
        "mesh_vertex_count": len(mesh_vertices),
        "mesh_face_count": len(mesh_faces),
        "mesh_boundary_count": 0,
    }

    if len(mesh_vertices) < 3 or len(mesh_faces) == 0:
        log.warning("Mesh does not have a single valid face, skipping metrics")
        return metrics

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    if output_dir is not None:
        # Export loaded mesh to a PLY file for verification.
        save_ply(os.path.join(output_dir, "mesh.ply"), vertices=mesh_vertices, triangles=mesh_faces, logger=log)

    # Add mesh area to the metrics.
    trimesh_mesh = trimesh.Trimesh(mesh_vertices, mesh_faces)
    metrics["mesh_area_m2"] = float(trimesh_mesh.area)
    log.info(f"Mesh area: {metrics['mesh_area_m2']:.2f} m2")

    log.info("Tracing mesh boundaries incl. hole boundaries")
    # TODO: Remove the try block once we are confident enough that this does not break the evaluation.
    try:
        start_time = time.perf_counter()
        mesh_boundaries = find_boundaries(mesh_faces)
        elapsed_time_sec = time.perf_counter() - start_time
        metrics["mesh_boundary_count"] = len(mesh_boundaries)
        log.info(f"Found {len(mesh_boundaries)} boundaries in {elapsed_time_sec:.1f} seconds")
    except:
        log.warning(f"Mesh boundary tracing failed, skipping boundary metric")

    # Add mesh quality metrics based on a vehicle/rig trajectory on the nominal ground (treated as ground-truth).
    traj_ground_metrics = calculate_vehicle_trajectory_vs_ground_consistency_metrics(
        mesh_vertices, mesh_faces, T_rig_worlds, output_dir=output_dir
    )

    metrics.update(traj_ground_metrics)

    return metrics


def log_metrics(metrics: Dict[str, float], title: Optional[str] = None) -> None:
    if title is not None:
        log.info(f"{title}:")
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            log.info(f"  {metric_name}: {metric_value:.2f}")
        else:
            log.info(f"  {metric_name}: {metric_value}")


def save_metrics_yaml(metrics: Dict[str, float], file_path: str) -> None:
    if not file_path.endswith(".yaml"):
        file_path += ".yaml"
    with open(file_path, "w") as f:
        yaml.dump(metrics, f)
    log.info(f"Saved metrics to {file_path}")


def save_metrics_json(metrics: Dict[str, float], file_path: str) -> None:
    if not file_path.endswith(".json"):
        file_path += ".json"
    with open(file_path, "w") as f:
        json.dump(metrics, f)
    log.info(f"Saved metrics to {file_path}")


def save_metrics_csv(metrics: Dict[str, float], file_path: str) -> None:
    if not file_path.endswith(".csv"):
        file_path += ".csv"
    with open(file_path, "w") as f:
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                f.write(f"{metric_name},{metric_value:.6f}\n")
            else:
                f.write(f"{metric_name},{metric_value}\n")
    log.info(f"Saved metrics to {file_path}")


def save_metrics_transposed_csv(metrics: Dict[str, float], file_path: str) -> None:
    if not file_path.endswith(".csv"):
        file_path += ".csv"
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    with open(file_path, "w") as f:
        f.write(",".join(metric_names) + "\n")
        f.write(",".join(str(round(v, 6)) if isinstance(v, float) else str(v) for v in metric_values))
        f.write("\n")
    log.info(f"Saved metrics to {file_path}")


@click.command("eval-ground-mesh")
@click.option(
    "--output-dir",
    type=str,
    help="Path to the output rendered image",
    required=True,
    default=None,
)
@click.option(
    "--usdz-path",
    type=str,
    help=(
        "Path to an USDZ artifact file exported from training, e.g. last.usdz. "
        "If not provided, you must provide --ground-mesh-path and --rig-trajectory-path."
    ),
    required=False,
    default=None,
)
@click.option(
    "--ground-mesh-path",
    type=str,
    help="Path to a PLY file to load the ground mesh from, otherwise it will be loaded from the USDZ artifact.",
    required=False,
    default=None,
)
@click.option(
    "--rig-trajectory-path",
    type=str,
    help="Path to a JSON file to load the rig trajectory from, otherwise it will be loaded from the USDZ artifact.",
    required=False,
    default=None,
)
def eval_ground_mesh(
    output_dir: str,
    usdz_path: Optional[str],
    rig_trajectory_path: Optional[str],
    ground_mesh_path: Optional[str],
) -> None:
    """Evaluates a ground mesh stored in a USDZ artifact or in a PLY file."""

    artifact = None
    if usdz_path is not None:
        log.info(f"Loading USDZ artifact {usdz_path}")
        artifact = Artifact(Path(usdz_path))

    # Load the ground mesh from PLY if specified, or from the USDZ artifact otherwise.
    if ground_mesh_path is not None:
        log.info(f"Loading ground mesh from {ground_mesh_path}")
        mesh_vertices, mesh_faces = load_ground_mesh_from_ply(ground_mesh_path)
        if usdz_path is not None:
            log.warning("USDZ specified but ground mesh loaded from the provided PLY")
    elif artifact is not None:
        log.info(f"Loading ground mesh from the USDZ artifact")
        mesh_vertices, mesh_faces = load_ground_mesh_from_artifact(artifact)
    else:
        raise ValueError("Either --ground-mesh-path or --usdz-path must be provided")
    log.info(f"Loaded mesh with {len(mesh_vertices)} vertices, {len(mesh_faces)} faces")

    # Load the rig trajectories from a JSON if specified, or from the USDZ artifact otherwise.
    if rig_trajectory_path is not None:
        log.info(f"Loading rig trajectory from {rig_trajectory_path}")
        T_rig_worlds = load_rig_trajectory_from_json(rig_trajectory_path)
        if usdz_path is not None:
            log.warning("USDZ specified but rig trajectory loaded from the provided JSON")
    elif artifact is not None:
        log.info(f"Loading rig trajectory from the USDZ artifact")
        T_rig_worlds = load_rig_trajectory_from_artifact(artifact)
    else:
        raise ValueError("Either --rig-trajectory-path or --usdz-path must be provided")
    log.info(f"Loaded {len(T_rig_worlds)} rig poses")

    log.info(f"Calculating ground mesh metrics")
    metrics = calculate_ground_mesh_metrics(mesh_vertices, mesh_faces, T_rig_worlds, output_dir=output_dir)

    log_metrics(metrics, title="Ground mesh metrics")

    output_file_path = os.path.join(output_dir, "ground_mesh_quality_metrics")
    save_metrics_yaml(metrics, output_file_path)
    save_metrics_json(metrics, output_file_path)
    save_metrics_csv(metrics, output_file_path)
    save_metrics_transposed_csv(metrics, output_file_path + "_transposed.csv")
