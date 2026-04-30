# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Generator, List, Optional, Tuple

import numpy as np
import PIL.Image as PILImage
import point_cloud_utils as pcu
import torch

from numpy.typing import ArrayLike, NDArray
from scipy.spatial import Delaunay

from nre.datasets.ncore import NCOREDataSource
from nre.utils.geometry import Plane, PlaneDetectorRansac
from nre.utils.io.ply import save_ply
from nre.utils.misc import unpack_optional
from nre.utils.types import FrameConversion, PointCloud, RayFlags


logger = logging.getLogger(__name__)

GROUND_MESH_RANDOM_SEED = 123  # Seed used for deterministic results throughout the module


@dataclass
class DominantPlaneRoadSegmenter:
    """Algorithm to segment the road in Lidar spins around an ego vehicle, based on dominant plane detection"""

    # Point cloud filtering parameters
    min_ray_length: float

    # Parameters controlling geometric constraints for plane hypotheses.
    ground_control_point: NDArray[np.float32]
    ground_compat_max_distance: float
    ground_compat_max_angle_deg: float

    # Parameters for RANSAC plane fitting.
    num_plane_hypotheses: int
    plane_max_distance: float
    plane_max_angle_deg: float
    plane_min_eval_points: int  # Min points to evaluate plane hypotheses on, skipping frame if not reached
    plane_max_eval_points: int  # Max points to evaluate plane hypotheses on, subsampling if exceeded
    plane_min_inliers: int  # Min number of inliers for accepting the dominant plane, skipping frame if not reached
    # Whether to extend the found dominant plane to all inlier points inside a full Lidar spin (True) or
    # only use the subset initially marked road (False).
    enable_plane_extension: bool = False

    # Parameters to control the output.
    export_per_frame_diagnostics: bool = False
    output_path: Path = Path(".")
    verbose: bool = False

    def __post_init__(self):
        assert self.ground_control_point.size == 3
        assert self.ground_compat_max_distance > 0.0
        assert self.ground_compat_max_angle_deg > 0.0
        assert self.num_plane_hypotheses > 0
        assert self.plane_max_distance > 0.0
        assert self.plane_max_angle_deg > 0.0
        assert self.plane_min_eval_points > 0
        assert self.plane_max_eval_points > 0
        assert self.plane_min_eval_points < self.plane_max_eval_points
        assert self.plane_min_inliers > 0

    def segment_road(
        self, point_cloud_generator: Generator[PointCloud, None, None], transformation: Optional[FrameConversion] = None
    ) -> Tuple[
        NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_], NDArray[np.bool_], Optional[NDArray[np.uint8]]
    ]:
        # Assumption: point_cloud_generator provides points per Lidar spin
        # Note: the input already contains semantic road segmentation but not geometrically verified.
        collected_points: List[NDArray[np.float32]] = []
        collected_normals: List[NDArray[np.float32]] = []
        collected_road_masks: List[NDArray[np.bool_]] = []
        collected_road_prior_masks: List[NDArray[np.bool_]] = []
        collected_colors: List[NDArray[np.uint8]] = []
        has_colors: Optional[bool] = None  # Determined from first frame

        highlight_color = np.array([255, 255, 50], dtype=np.uint8)

        start_time_loop = perf_counter()

        logger.info("Road segmentation per lidar frame")
        for idx, point_cloud in enumerate(point_cloud_generator):
            # Output path to be used in frame export functions
            frame_output_path = self.output_path / f"{idx:06d}"

            ray_starts = point_cloud.xyz_start.cpu().numpy()
            points = point_cloud.xyz_end.cpu().numpy()
            # color_type is set uniformly for the entire generator, so in practice all frames either
            # have or lack colors. This guard is defensive against future changes to the generator contract.
            frame_colors = point_cloud.color.cpu().numpy() if point_cloud.color is not None else None
            if has_colors is None:
                has_colors = frame_colors is not None
            elif has_colors and frame_colors is None:
                logger.warning("Frame has no colors despite previous frames having colors; disabling color collection")
                has_colors = False

            if self.verbose:
                logger.info(f"Frame {idx} contains {len(points)} rays")
            if len(points) == 0:  # Just in case
                logger.warning(f"Frame {idx} contains no rays, skipping frame")
                continue

            if transformation is not None:
                ray_starts = transformation.transform_points(ray_starts)
                points = transformation.transform_points(points)

            # Find LiDAR returns closer to the sensor than the near treshold.
            # This is to account for LiDAR points randomly being on the far surface or very close to the sensor,
            # an issue we experienced with some data (H8.1), and that can tear apart the LiDAR spin mesh.
            ray_lengths = np.sqrt(np.sum((points - ray_starts) ** 2, axis=1))
            mask_invalid_points = ray_lengths < self.min_ray_length
            mask_valid_points = ~mask_invalid_points
            if self.verbose:
                logger.info(
                    f"{np.sum(mask_invalid_points)} near points (d<{self.min_ray_length}) detected "
                    f"({100.0 * np.sum(mask_invalid_points) / max(len(points), 1):.1f}%)"
                )

            if self.export_per_frame_diagnostics:
                colors = np.zeros((len(points), 3), dtype=np.uint8)
                colors[mask_invalid_points] = highlight_color  # No failure in case of empty index list, TODO: test
                save_ply(str(frame_output_path) + "_input_near_points_marked.ply", points, colors=colors, logger=logger)

            # IMPORTANT PRIOR: Road is already segmented in the input points based on a prior semantic segmentation
            # obtained by transferring semantic labels from camera images to the point cloud.
            # However, this segmentation contains outliers (points marked road far outside the roads plane)
            # to be filtered out prior to meshing.
            road_prior_mask = (
                torch.bitwise_and(unpack_optional(point_cloud.flags), RayFlags.ROAD_SEMANTIC.value)
                .cpu()
                .numpy()
                .astype(bool)
            )
            road_prior_indices = np.where(road_prior_mask)[0]
            if self.verbose:
                logger.info(f"{len(road_prior_indices)} input points initially marked road")

            if self.export_per_frame_diagnostics:
                colors = np.zeros((len(points), 3), dtype=np.uint8)
                colors[road_prior_indices] = highlight_color  # No failure in case of empty index list
                save_ply(str(frame_output_path) + "_input_points.ply", points, colors=colors, logger=logger)

            # Compute normals for valid points within the Lidar spin via a special "lidar spin mesh" reconstruction.
            # This approach constructs a mesh in the azimuth-elevation (scanning pattern) space, and
            # is robust to ray drops and to the sparse sampling in elevation typical for spinning mobile lidars.
            # Point centering translates the rays to start at the origin. This approximates the original rays in the
            # Lidar sensor's frame by discarding the sensor-to-world rotation and the per-ray motion compensation.
            # The relevant part of the mesh topology is practically invariant to the discarded transformations.
            # The centering assumes that all points were scanned from a sensor under at most a translational motion.
            # This assumption does not strictly hold when the host vehicle (sensor rig) turns during the lidar spin
            # but a vehicle turn seems to be small enough in practice during the split-second lidar spin
            # to preserve the scanning pattern and not cause noticable artifacts in the mesh and normals.
            # TODO: if artifacts discovered, replace centering with properly transforming rays back to sensor frame.
            # The standard methods of point cloud normal computation based on k-NNs is unstable within a Lidar spin
            # and degrades with distance, because close point neighbors are likely in the same scanline.
            centered_points = points - ray_starts
            subset_normals, subset_triangles = compute_mesh_normals_in_lidar_spin(
                centered_points[mask_valid_points], verbose=self.verbose
            )

            # LiDAR spin mesh was computed from only valid points.
            # Maps normals and triangles back to the original points.
            index_remap = np.arange(len(points), dtype=np.int32)[mask_valid_points]
            triangles = index_remap[subset_triangles]
            normals = np.tile(np.array([0, 0, 1], dtype=np.float32), (len(points), 1))
            normals[mask_valid_points] = subset_normals

            if self.export_per_frame_diagnostics:
                save_ply(
                    str(frame_output_path) + "_spin_mesh.ply",
                    points,
                    triangles,
                    normals,
                    logger=logger,
                )

            # Collect relevant data from each frame.
            collected_points.append(points)
            collected_normals.append(normals)
            collected_road_prior_masks.append(road_prior_mask)
            if has_colors and frame_colors is not None:
                collected_colors.append(frame_colors)
            # The rest of the cycle tries to identify road inliers and return the corresponding fixed-size mask.
            # Append an empty mask upfront to make it possible to skip the rest of the cycle at any stage
            # with no points detected as road, if results do not meet certain requirements.
            collected_road_masks.append(np.zeros((len(points),), dtype=bool))

            # Filter oriented points based on simple "ground-compatibility" geometric priors:
            # 1. Check if point and normal define a plane that passes near a nominal ground point under the sensor.
            # 2. Check if point's normal is within an angle w.r.t. the up direction.
            #    Account for road inclination, vehicle rolling/pitching (dynamics) and noise in the normals.
            #    Extreme San Francisco streets can go up to 40% grade ~ 22 deg inclination.
            #    Passenger car pitch/roll <2 deg at normal driving, can reach 5 deg with agressive driving [ChatGPT].
            #    SUVs and trucks can have even higher pitch/roll values.
            # Motivation: oriented points will generate plane hypotheses in a follow-up step, and we want to avoid
            # generating and verifying hypotheses that do not satisfy these simple conditions.
            if self.verbose:
                logger.info("Finding ground-compatible points")
            up_vector = np.array([0, 0, 1], dtype=np.float32)
            gc_mask = find_ground_compatible_points(
                centered_points[road_prior_indices],
                normals[road_prior_indices],
                self.ground_control_point,
                ref_normal=up_vector,
                max_distance=self.ground_compat_max_distance,
                max_angle_deg=self.ground_compat_max_angle_deg,
            )

            # Discard points marked road that are not ground-compatible.
            ground_compatible_indices = road_prior_indices[gc_mask]
            del gc_mask  # It is a submask only, prevent further usage.
            if self.verbose:
                logger.info(
                    f"{len(ground_compatible_indices)} ground-compatible points "
                    f"({100.0 * len(ground_compatible_indices) / max(len(road_prior_indices), 1):.1f}%)"
                )
            if self.export_per_frame_diagnostics:
                colors = np.zeros((len(points), 3), dtype=np.uint8)
                colors[ground_compatible_indices] = highlight_color
                save_ply(
                    str(frame_output_path) + "_ground_compatible_points.ply",
                    points,
                    normals=normals,
                    colors=colors,
                    logger=logger,
                )
            if len(ground_compatible_indices) < self.plane_min_eval_points:
                logger.warning(
                    f"Frame {idx} contains too few ground-compatible points ({len(ground_compatible_indices)} but "
                    f"{self.plane_min_eval_points} required), skipping segmentation"
                )
                continue

            # Randomly select a subset of points to evaluate plane hypotheses on (reduces the computational burden)
            eval_point_sub_indices = subsample(len(ground_compatible_indices), max_samples=self.plane_max_eval_points)
            eval_point_indices = ground_compatible_indices[eval_point_sub_indices]
            if self.verbose:
                logger.info(
                    f"{len(eval_point_indices)} of {len(ground_compatible_indices)} points "
                    f"sampled for evaluating plane hypotheses"
                )
            if self.export_per_frame_diagnostics:
                save_ply(
                    str(frame_output_path) + "_plane_eval_sample.ply",
                    points[eval_point_indices],
                    normals=normals[eval_point_indices],
                    logger=logger,
                )
            if len(eval_point_indices) < self.plane_min_eval_points:
                logger.warning(
                    f"Frame {idx} has too few plane evaluation samples ({len(eval_point_indices)}, "
                    f"{self.plane_min_eval_points} required), skipping segmentation"
                )
                continue

            # Robust plane fitting to only points presegmented as road that are also ground-compatible.
            # This is conservative, geared towards high precision rather than high recall, i.e.
            # may miss points (underdetection) but avoids false positives (avoids overdetection).
            start_time = perf_counter()
            plane, inlier_mask, ransac = detect_dominant_plane(
                points[ground_compatible_indices],  # Only using ground-compatible point for the fitting
                normals[ground_compatible_indices],
                distance_threshold=self.plane_max_distance,
                angle_threshold_deg=self.plane_max_angle_deg,
                num_hypotheses=self.num_plane_hypotheses,
                eval_point_indices=eval_point_sub_indices,
            )
            inlier_indices = ground_compatible_indices[inlier_mask]
            inlier_mask = np.zeros((len(points),), dtype=bool)
            inlier_mask[inlier_indices] = True
            if self.enable_plane_extension:
                # Find inliers in the full point cloud.
                # This is less conservative because the plane extends globally and points far from the road
                # may accidentally be compatible with the detected road plane, re-introducing some false positives.
                inlier_mask = ransac.find_inliers(plane, points, normals)
                inlier_indices = np.where(inlier_mask)[0]

            elapsed_time = perf_counter() - start_time
            if self.verbose:
                logger.info(
                    f"{self.num_plane_hypotheses} plane hypotheses evaluated against {len(eval_point_sub_indices)} points "
                    f"and winner has {len(inlier_indices)} inliers out of {len(ground_compatible_indices)} points "
                    f"({100.0 * len(inlier_indices) / max(len(ground_compatible_indices), 1):.1f}%), "
                    f"took {elapsed_time:.3f} s"
                )
            if self.export_per_frame_diagnostics:
                colors = np.zeros((len(points), 3), dtype=np.uint8)
                colors[inlier_indices] = highlight_color
                save_ply(
                    str(frame_output_path) + "_plane_segmentation.ply",
                    points,
                    normals=normals,
                    colors=colors,
                    logger=logger,
                )
                save_ply(
                    str(frame_output_path) + "_plane_inliers.ply",
                    points[inlier_indices],
                    normals=normals[inlier_indices],
                    logger=logger,
                )

                image = DominantPlaneRoadSegmenter._render_points(
                    centered_points, colors, camera_position=np.array([-10, 0, 20])
                )
                _save_image(str(frame_output_path) + "_plane_segmentation.png", image)

            if len(inlier_indices) < self.plane_min_inliers:
                logger.warning(
                    f"Frame {idx} has too few inliers ({len(inlier_indices)}) to trust the dominant plane, "
                    f"{self.plane_min_inliers} required, skipping segmentation"
                )
                continue

            # Accept segmentation result by storing it.
            # Replaces the default no-inliers mask already stored upfront.
            collected_road_masks[-1] = inlier_mask

        elapsed_time = perf_counter() - start_time_loop
        logger.info(f"Road segmentation for {len(collected_points)} lidar frames took {elapsed_time:.3f} s")

        if len(collected_points) == 0 or collected_points[0].size == 0:
            raise ValueError("No points in any of the lidar frames")

        logger.info(f"Merging {len(collected_points)} lidar point clouds")
        start_time = perf_counter()
        points = np.concatenate(collected_points)  # Raises ValueError for [] input
        normals = np.concatenate(collected_normals)
        road_mask = np.concatenate(collected_road_masks)
        road_prior_mask = np.concatenate(collected_road_prior_masks)
        elapsed_time = perf_counter() - start_time
        logger.info(f"Merged {len(points)} points from {len(collected_points)} spins in {elapsed_time:.3f} s")

        merged_colors = np.concatenate(collected_colors) if has_colors and collected_colors else None
        return points, normals, road_mask, road_prior_mask, merged_colors

    @staticmethod
    def _render_points(
        points: NDArray[np.float32],
        colors: NDArray[np.uint8],
        camera_target: NDArray[np.float32] = np.array([0, 0, 0], dtype=np.float32),
        camera_position: NDArray[np.float32] = np.array([0, 0, 100], dtype=np.float32),
        resolution_px: Tuple[int, int] = (1280, 1024),
        hfov_deg: float = 75.0,
        near_plane=1e-3,
        bgcolor: ArrayLike = np.array([100, 100, 100], dtype=np.uint8),
    ) -> PILImage.Image:
        """Simplified rendering of colored points into an image"""
        assert points.shape[1] == 3
        assert colors.shape == points.shape

        # Calculate world to camera transformation
        z_axis = camera_target - camera_position
        z_axis /= np.linalg.norm(z_axis)
        up_vector = np.array([0, 0, 1])
        x_axis = np.cross(z_axis, up_vector)
        x_axis_norm = np.linalg.norm(x_axis)
        # Handle camera looking straight up or down, as the cross product is null.
        x_axis = np.array([1, 0, 0]) if x_axis_norm < 1e-3 else x_axis / x_axis_norm
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)  # Just for numerical precision
        camera_to_world_rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
        assert np.allclose(camera_to_world_rotation @ camera_to_world_rotation.T, np.identity(3))

        # Transform points to camera space
        points_cam_space = (points - camera_position.reshape(1, 3)) @ camera_to_world_rotation

        # Cull points that are not in front of the near plane
        indices_front = np.where(points_cam_space[:, 2] > near_plane)[0]

        # Project points to the image
        width, height = resolution_px
        focal_length_px = width / 2 / np.tan(np.radians(hfov_deg) / 2)
        x = focal_length_px * points_cam_space[indices_front, 0] / points_cam_space[indices_front, 2] + width / 2.0
        y = focal_length_px * points_cam_space[indices_front, 1] / points_cam_space[indices_front, 2] + height / 2.0
        projects_inside = (x > 0.0) & (x < float(width)) & (y > 0.0) & (y < float(height))

        # Cull points that do not project within the image frame
        x = x[projects_inside]
        y = y[projects_inside]
        xi = x.astype(np.int32)
        yi = y.astype(np.int32)
        visible_point_indices = indices_front[projects_inside]

        colors = colors[visible_point_indices]

        image = np.empty(shape=(height, width, 3), dtype=np.uint8)
        image[:] = np.asarray(bgcolor)
        # Each point being splatted into a single pixel without any care to depth ordering
        image[yi, xi] = colors
        return PILImage.fromarray(image)


def downsample_3d_points_on_2d_grid(
    points: NDArray[np.float32],
    cell_size: float,
    colors: Optional[NDArray[np.uint8]] = None,
) -> Tuple[NDArray[np.float32], Optional[NDArray[np.uint8]]]:
    """Downsample points on a 2D grid in the XY plane.

    Constructs a 2D grid within the bounding box of the point cloud (discarding the Z coordinate in this step).
    A point is returned at the center of each occupied cell (producing a more regular output point distribution),
    and the output point will have the average z value of all input points falling into the cell (denoising effect).

    If colors are provided, they are averaged per cell in the same grouping used for Z averaging.
    """
    assert len(points.shape) == 2 and points.shape[1] == 3
    assert len(points) > 0
    assert cell_size > 0.0

    bbox_min = np.min(points[:, 0:2], axis=0)
    bbox_max = np.max(points[:, 0:2], axis=0) + cell_size * 1e-4  # Workaround to include boundary points

    # At least 1 cell along each dimension.
    cell_counts = np.maximum(np.ceil((bbox_max - bbox_min) / cell_size), [1, 1]).astype(int)
    scale = 1 / cell_size

    logger.info(
        f"bbox_size=({bbox_max[0] - bbox_min[0]:.3f}, {bbox_max[1] - bbox_min[1]:.3f}), "
        f"num_cells={tuple(cell_counts)}, "
        f"total_cells={np.prod(cell_counts) / 1e3:.1f}k"
    )

    # Point to cell assignments
    normalized_points = scale * (points[:, 0:2] - bbox_min)
    cell_indices = np.floor(normalized_points).astype(int)
    cell_ids = cell_indices[:, 1] * cell_counts[0] + cell_indices[:, 0]  # Linear index into cell per point

    # The cell ids group the points but cell ids are not sorted and non-contiguous.
    # Sort points and their cell assignments by increasing order of cell ID
    sorting_indices = np.argsort(cell_ids)
    sorted_cell_ids = cell_ids[sorting_indices]
    sorted_cell_indices = cell_indices[sorting_indices]
    sorted_points_z = points[sorting_indices, 2]

    # Split the points sorted by
    group_start_indices = np.unique(sorted_cell_ids, return_index=True)[1]
    points_per_group = np.split(sorted_points_z, group_start_indices[1:], axis=0)

    average_z_per_group = np.array([np.average(points_z) for points_z in points_per_group])
    group_cell_indices = sorted_cell_indices[group_start_indices]
    cell_centers = cell_size * (group_cell_indices + 0.5) + bbox_min

    downsampled_points = np.hstack((cell_centers, average_z_per_group.reshape(-1, 1)))

    # Average colors per cell using the same grouping as Z averaging.
    downsampled_colors = None
    if colors is not None:
        sorted_colors = colors[sorting_indices].astype(np.float32)
        colors_per_group = np.split(sorted_colors, group_start_indices[1:], axis=0)
        downsampled_colors = np.array([np.mean(c, axis=0) for c in colors_per_group], dtype=np.uint8)

    return downsampled_points, downsampled_colors


@dataclass
class DelaunayElevationMeshingAlgorithm:
    """Mesh a terrain/elevation-type 2.5D point cloud that does not contain vertical or overhanging structures

    Produces a mesh from input elevation point cloud by performing the following steps:
    1. Downsample the point cloud on a 2D grid laid over the XY plane (Bird's Eye View, BEV).
    2. Construct a Delaunay 2D mesh on the input points in the BEV (XY components of the points) and
       lift the 2D vertices to the 3D points in Z, so the mesh vertices become the downsampled points.
       A simple and relatively fast method to produce a watertight mesh (no holes) over elevation data.
       The meshed domain is the convex hull of the input points in the BEV.
       Noise and gross outlier (off-surface) points in the downsampled point cloud directly appear in the mesh.
       For this reason, the input is assumed to be free of gross outliers, otherwise they cause spikes in the mesh.
    3. A mesh smoothing is applied along the z-direction only to denoise the vertices as a post-processing step.
    """

    # TODO: Replace direct alignment of vertices to points with a linear least squares minimization of a fitting
    # and a smoothness term. This would allow to add extra points to the 2D mesh whose elevation is unknown,
    # e.g. extend the mesh smoothly at the edges to cover a larger area.

    enable_downsampling: bool = True
    voxel_size: float = 1.0
    min_points_per_voxel: int = 1
    smoothing_passes: int = 1
    elevation_axis: int = 2

    def build_mesh_from_points(
        self,
        points: NDArray[np.float32],
        colors: Optional[NDArray[np.uint8]] = None,
    ) -> Tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.float32], Optional[NDArray[np.uint8]]]:
        # Effective point cloud denoising and reduction of the number of points by orders of magnitude
        # to make 2D Delaunay meshing run faster.
        # 2D grid-based downsampling replaces voxel-based downsampling via pcu.downsample_point_cloud_on_voxel_grid()
        # because it produces a better quality mesh, free of the Z-quantization artifacts of the latter approach.
        # The additional regularization effect of clamping output points to cell-centers is beneficial
        # for smoothing algorithms that are not normalizing by vertex density.
        logger.info(f"Downsampling {len(points)} points on voxels (cell_size={self.voxel_size})")
        start_time = perf_counter()
        if self.enable_downsampling:
            vertices, vertex_colors = downsample_3d_points_on_2d_grid(points, self.voxel_size, colors=colors)
        else:
            vertices, vertex_colors = points, colors
        elapsed_time = perf_counter() - start_time
        logger.info(f"Downsampling produced {len(vertices)} points in {elapsed_time:.3f} s")

        logger.info(f"2D Delaunay triangulation over {len(vertices)} points")
        start_time = perf_counter()
        triangles = Delaunay(vertices[:, 0:2]).simplices
        elapsed_time = perf_counter() - start_time
        logger.info(f"2D Delaunay triangulation produced {len(triangles)} triangles in {elapsed_time:.3f} s")

        # Only want to smooth mesh vertices along the vertical direction.
        # Laplacian smoothing is not suitable as it also moves vertices parallel to the surface
        # and has a well known shrinking bias.
        smooth_vertices = vertices
        if self.smoothing_passes > 0:
            logger.info(f"Collecting vertex neighbors for {len(vertices)} vertices")
            start_time = perf_counter()
            vertex_neighbors, vertex_distances = get_vertex_neighbors(vertices, triangles)
            elapsed_time = perf_counter() - start_time
            logger.info(f"Vertex neighbors enumerated in {elapsed_time:.3f} s")

            start_time = perf_counter()
            logger.info(f"Mesh smoothing over {len(vertices)} vertices ({self.smoothing_passes} passes)")
            for i in range(self.smoothing_passes):
                smooth_vertices = mesh_smoothing_along_axis(
                    smooth_vertices, vertex_neighbors, vertex_distances, axis=self.elevation_axis
                )
            elapsed_time = perf_counter() - start_time
            logger.info(f"Mesh smoothing took {elapsed_time:.3f} s")

        return smooth_vertices, triangles, vertices, vertex_colors


def _save_image(image_path: str, image: PILImage.Image) -> None:
    logger.info(f"Saving {image_path}")
    image.save(image_path)


def get_mesh_edges(triangles: NDArray[np.int32]) -> NDArray[np.int32]:
    """Return a unique list of undirected edges on a mesh"""
    v0 = triangles[:, [0]]
    v1 = triangles[:, [1]]
    v2 = triangles[:, [2]]
    edges = np.vstack([np.hstack([v0, v1]), np.hstack([v1, v2]), np.hstack([v2, v0])])
    assert edges.shape == (3 * triangles.shape[0], 2)
    assert edges.dtype == triangles.dtype
    # Sort edges and make them unique
    swap = edges[:, 0] > edges[:, 1]
    edges[swap, :] = np.flip(edges[swap, :], axis=1)
    edges = np.unique(edges, axis=0)
    return edges


def calculate_edge_lengths(vertices: NDArray[np.float32], edges: NDArray[np.int32]) -> NDArray[np.float32]:
    assert vertices.shape[1] == 3
    assert edges.shape[1] == 2
    assert np.max(edges) < vertices.shape[0]
    return np.linalg.norm(vertices[edges[:, 0], :] - vertices[edges[:, 1], :], ord=2, axis=1)


def get_vertex_neighbors(
    vertices: NDArray[np.float32], triangles: NDArray[np.int32]
) -> Tuple[List[List[int]], List[List[float]]]:
    """List all adjacent vertices (1-ring neighbors) of every vertex

    Returns the list of indices of neighbors and the list of neighbor distances for every vertex.
    """
    num_vertices = len(vertices)
    assert vertices.shape[1] == 3
    assert triangles.shape[1] == 3
    assert np.max(triangles) < num_vertices
    edges = get_mesh_edges(triangles)
    edge_lengths = calculate_edge_lengths(vertices, edges)
    neighbors: List[List[int]] = [[] for i in range(num_vertices)]
    distances: List[List[float]] = [[] for i in range(num_vertices)]
    for edge_idx in range(len(edges)):
        i, j = edges[edge_idx, :]
        edge_length = edge_lengths[edge_idx]
        neighbors[i].append(j)
        neighbors[j].append(i)
        distances[i].append(edge_length)
        distances[j].append(edge_length)
    return neighbors, distances


def mesh_smoothing_along_axis(
    vertices: NDArray[np.float32],
    vertex_neighbors: List[List[int]],
    vertex_distances: List[List[float]],
    axis: int = 2,
    smoothness: float = 1.0,
) -> NDArray[np.float32]:
    """Smooths mesh along a single axis towards zero curvature, without moving vertices along the other two axes.

    Avoids the mesh shrinking bias of Laplacian smoothing, and keeps the vertices locked in the birds eye view.
    The height of each vertex is moved towards the weighted average height of the 1-ring neighbors.
    The weight is the inverse of the distance between the vertex and its neighbor, normalized to unit sum
    so that closer neighbors have higher influence.
    The amount of smoothness is controlled by a smoothness parameter linearly between the original height
    (smoothness=0.0) and the weighted average height (smoothness=1.0).
    """

    num_vertices = len(vertices)
    assert vertices.shape[1] == 3
    assert len(vertex_neighbors) == num_vertices
    assert len(vertex_distances) == num_vertices
    target_heights = np.empty((num_vertices,), dtype=np.float32)

    for i in range(num_vertices):
        neighbor_indices = vertex_neighbors[i]
        neighbor_distances = np.array(vertex_distances[i], dtype=np.float32)
        neighbor_heights = vertices[neighbor_indices, axis]
        # Reduce the influence of distant neighbors.
        # Works much better than simple average (a.k.a. mean curvature flow / Laplacian smoothing with uniform weights)
        neighbor_weights = 1.0 / neighbor_distances
        # neighbor_weights /= np.sum(neighbor_weights)  # np.average() does this normalization
        # TODO (a): Replace this with the vertical projection of the point to the plane regressed on its 1-ring.
        # TODO (b): Replace the whole meshing with a fitting using a fitting + smoothness term
        target_heights[i] = np.average(neighbor_heights, weights=neighbor_weights)

    smooth_vertices = vertices.copy()
    smooth_vertices[:, axis] = target_heights * smoothness + vertices[:, axis] * (1.0 - smoothness)
    return smooth_vertices


def get_nominal_ground_point_under_lidar(datasource: NCOREDataSource) -> NDArray[np.float32]:
    """Return a point on the nominal ground, specified in a reference frame with the lidar in the origin."""

    # TODO: Add missing sequence_id, sensor_id and lidar_frame_idx to PointCloud when it contains a single Lidar frame.
    # Until then, using the simplifying assumptions below.

    dataset_lidar_sensor_ids = datasource.lidar_sensors.keys()
    if len(dataset_lidar_sensor_ids) > 1:
        raise ValueError("Only a single lidar per dataset is supported currently")
    lidar_id = next(iter(dataset_lidar_sensor_ids))
    lidar_sensor = datasource.lidar_sensors[lidar_id]
    sensor_to_rig_transf = unpack_optional(lidar_sensor.T_sensor_rig)
    # The sensor position in the rig frame is in the last column if and only if the transformation is affine.
    if not np.allclose(sensor_to_rig_transf[3, 0:4], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("get_T_sensor_rig() does not return an affine transformation")
    sensor_height = sensor_to_rig_transf[2, 3]
    ground_point = np.array([0, 0, -sensor_height], dtype=np.float32)
    logger.info(
        f"Nominal ground point in a sensor-centric reference frame for lidar '{lidar_id}': {ground_point.tolist()} "
    )
    return ground_point


def xyz_to_spherical(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert the point cloud to spherical coordinates, assuming that points are centered and the z axis points up"""
    assert points.shape[1] == 3
    radii = np.linalg.norm(points, axis=1)
    azimuth = np.arctan2(points[:, 1], points[:, 0])  # Ranges [-pi, +pi]
    elevation = np.arcsin(points[:, 2] / radii)
    return azimuth, elevation, radii


def construct_lidar_spin_mesh(centered_points: np.ndarray):
    """Construct a triangle mesh over a set of input lidar points scanned from the origin."""
    azimuth, elevation, _ = xyz_to_spherical(centered_points)
    azimuth_elevation = np.stack([azimuth, elevation, np.zeros_like(azimuth)], axis=1)
    # TODO: scipy's Delaunay can take around 2 sec on ~300k points, find a faster impl. or method.
    triangles = Delaunay(azimuth_elevation[:, 0:2]).simplices
    return triangles, azimuth_elevation


def flip_normals_towards_viewpoint(points: np.ndarray, normals: np.ndarray, viewpoint: np.ndarray) -> np.ndarray:
    """Flip point normals to point towards a user-defined reference point (viewpoint)"""
    assert points.shape[1] == 3
    assert normals.shape == points.shape
    assert viewpoint.size == 3
    viewpoint = viewpoint.reshape((1, 3))
    needs_flipping = np.sum(normals * (viewpoint - points), axis=1) < 0.0
    flipped_normals = normals.copy()
    flipped_normals[needs_flipping] = -normals[needs_flipping]
    return flipped_normals


def compute_mesh_normals_in_lidar_spin(points: NDArray[np.float32], verbose=False) -> Tuple[np.ndarray, np.ndarray]:
    """Compute normals for points from a Lidar spin via triangle mesh construction over the scanning pattern.

    Input points are assumed to be in cartesian coordinates and scanned with a Lidar spinning at the origin.
    This function performs the following steps:
    1. Convert points to spherical coordinates (azimuth, elevation, radius).
    2. Construct a triangle mesh in azimuth-elevation (2D) space over the points as vertices.
    3. Compute vertex normals of the mesh.
    4. Flip normals to point outside of the surface, given the (ray) origin.
    """
    assert len(points.shape) == 2 and points.shape[1] == 3

    # Reconstruct mesh topology over the points based on the 2D scanning pattern in sperical coordinates.
    # This establishes a trivial point neighborhood system (as opposed to expensive 3D spatial queries).
    if verbose:
        logger.info(f"Reconstructing lidar spin mesh over {len(points)} points")
        start_time = perf_counter()
    triangles, azimuth_elevation = construct_lidar_spin_mesh(points)
    if verbose:
        elapsed_time = perf_counter() - start_time
        logger.info(f"{len(triangles)} triangles reconstructed in {elapsed_time:.3f} s")

    # Leverage the computed mesh topology to calculate point normals.
    if verbose:
        logger.info(f"Computing normals for {len(points)} mesh vertices")
        start_time = perf_counter()
    normals = pcu.estimate_mesh_vertex_normals(points, triangles)
    invalid_normals = np.any(np.isnan(normals), axis=1)
    normals[invalid_normals] = np.array([0, 0, 1], dtype=np.float32)  # Set NaN normals to something valid
    normals = flip_normals_towards_viewpoint(points, normals, np.zeros((1, 3), dtype=np.float32))
    if verbose:
        elapsed_time = perf_counter() - start_time
        logger.info(f"{len(normals)} normals computed in {elapsed_time:.3f} s")

    return normals, triangles


def find_ground_compatible_points(
    points: NDArray[np.float32],  # Input points to cull
    normals: NDArray[np.float32],  # Normals of the input points
    ground_control_point: NDArray[np.float32],  # A known point on the ground in a sensor-centric frame
    ref_normal: NDArray[np.float32],  # Reference normal for culling ground-compatible points
    max_distance: float,  # Distance tolerance w.r.t. the ground control points for plane hypotheses
    max_angle_deg: float,  # Angle tolerance w.r.t. the ref normal for plane hypotheses
) -> NDArray[np.bool_]:
    """Find all oriented points whose normal is within a certain angle tolerance from a reference normal and whose
    plane passes near a known ground control point within a given distance tolerance"""
    assert points.shape[1] == 3
    assert ref_normal.size == 3
    assert normals.shape == points.shape
    assert ground_control_point.size == 3
    assert max_distance > 0.0
    assert max_angle_deg > 0.0 and max_angle_deg <= 180.0
    assert np.allclose(np.sum(normals**2, axis=1), 1.0)  # Point normals expected to be unit-norm
    assert np.isclose(np.linalg.norm(ref_normal.flatten()), 1.0)  # Reference normal expected to be unit-norm

    plane_normals = normals
    plane_offsets = -np.sum(plane_normals * points, axis=1)
    # The distance is simple and can be calculated in batch because normals are all unit-norm.
    plane_distances_from_gcp = np.abs(np.sum(plane_normals * ground_control_point.flatten(), axis=1) + plane_offsets)

    cos_angles = np.sum(plane_normals * ref_normal.flatten(), axis=1)
    min_cos_angle = np.cos(np.radians(max_angle_deg))

    return np.logical_and(plane_distances_from_gcp <= max_distance, cos_angles > min_cos_angle)


def subsample(num_items, max_samples: int) -> np.ndarray:
    num_samples = min(max_samples, num_items)
    np.random.seed(GROUND_MESH_RANDOM_SEED)  # Makes runs deterministic
    return np.random.choice(num_items, num_samples, replace=False) if num_items > max_samples else np.arange(num_items)


def detect_dominant_plane(
    points: NDArray[np.float32],
    normals: NDArray[np.float32],
    distance_threshold: float,
    angle_threshold_deg: float,
    num_hypotheses: int,
    eval_point_indices: Optional[ArrayLike] = None,
) -> Tuple[Plane, NDArray[np.bool_], PlaneDetectorRansac]:
    ransac = PlaneDetectorRansac(distance_threshold, angle_threshold_deg)

    np.random.seed(GROUND_MESH_RANDOM_SEED)  # Makes runs deterministic

    # Generate plane hypotheses from individual points and their normals.
    # Assumes normals of decent quality but it is faster to evaluate than hypotheses from point triplets:
    # - Needs less hypotheses to achieve the same chance of an uncontaminated one (needs 1 inlier instead of 3).
    # - Enables prefiltering points based on geometric priors to avoid evaluating hypotheses that do not satisfy them.
    planes, _ = ransac.generate_planes_from_single_points(points, normals, num_planes=num_hypotheses)

    # Optionally subsample points for plane scoring because a spin can contain hundreds of thousands of points.
    # There must remain enough samples on the actual plane to be safely detectable.
    if eval_point_indices is None:
        eval_points = points
        eval_normals = normals
    else:
        indices = np.asarray(eval_point_indices)
        assert indices.size > 1
        eval_points = points[indices]
        eval_normals = normals[indices]

    plane_scores = np.zeros((len(planes), 1), dtype=np.float32)

    for plane_idx in range(len(planes)):
        plane = planes[plane_idx]
        score = ransac.evaluate_plane_hypothesis(plane, eval_points, eval_normals)
        assert not np.isnan(score)
        plane_scores[plane_idx] = score

    # Pick the plane with the highest score ~ most support from the points and normals.
    best_plane_idx = np.argmax(plane_scores)
    best_plane = planes[best_plane_idx]

    # Find inliers in the full set of input points
    is_inlier = ransac.find_inliers(best_plane, points, normals)

    # TODO: do at least 1 round of plane refitting (regression) to all inliers and recomputing inliers
    # because candidate planes are generated from single points whose normals are inaccurate,
    # although the best that has the most support is picked.

    return best_plane, is_inlier, ransac


def reconstruct_ground_mesh_from_points(
    point_cloud_generator: Generator[PointCloud, None, None],  # Input point cloud
    min_ray_length: float,
    ground_control_point: NDArray[np.float32],
    ground_compat_max_distance: float,
    ground_compat_max_angle_deg: float,
    num_plane_hypotheses: int,  # Number of plane hypotheses to generate for dominant plane detection
    plane_max_distance: float,
    plane_max_angle_deg: float,
    plane_min_eval_points: int,
    plane_max_eval_points: int,
    plane_min_inliers: int,
    enable_plane_extension: bool,
    voxel_size: float,
    min_points_per_voxel: int,
    smoothing_passes: int,
    export_per_frame_diagnostics: bool = False,
    output_path: Path = Path("."),
    points_to_world_transf: Optional[FrameConversion] = None,
    verbose: bool = False,
) -> Tuple[
    NDArray[np.float32],  # Mesh vertices after smoothing (V,3)
    NDArray[np.int32],  # Mesh triangles as vertex index triplets (T,3)
    NDArray[np.float32],  # Vertices before smoothing (V,3)
    NDArray[np.float32],  # Point positions for meshing (N,3)
    NDArray[np.float32],  # Point normals for meshing (N,3)
    NDArray[np.bool_],  # Road mask (N,), indicates for each point whether it has been identified as road internally
    NDArray[np.bool_],  # Initial road mask (N,) from the input point cloud
    Optional[NDArray[np.uint8]],  # Per-vertex RGB colors (V,3), or None if colors were not available
]:
    # Remove outliers from the road points by robust dominant plane fitting to the points per spin
    # such that plane hypotheses pass through the provided ground control point.
    road_segmenter = DominantPlaneRoadSegmenter(
        min_ray_length=min_ray_length,
        ground_control_point=ground_control_point,
        ground_compat_max_distance=ground_compat_max_distance,
        ground_compat_max_angle_deg=ground_compat_max_angle_deg,
        num_plane_hypotheses=num_plane_hypotheses,
        plane_max_distance=plane_max_distance,
        plane_max_angle_deg=plane_max_angle_deg,
        plane_min_eval_points=plane_min_eval_points,
        plane_max_eval_points=plane_max_eval_points,
        plane_min_inliers=plane_min_inliers,
        enable_plane_extension=enable_plane_extension,
        export_per_frame_diagnostics=export_per_frame_diagnostics,
        output_path=output_path,
        verbose=verbose,
    )
    points, normals, road_mask, initial_road_mask, point_colors = road_segmenter.segment_road(
        point_cloud_generator, transformation=points_to_world_transf
    )

    logger.info("Separating merged point cloud into road and non-road points")
    points_road = points[road_mask]
    colors_road = point_colors[road_mask] if point_colors is not None else None

    if len(points_road) < 3:
        raise ValueError(f"Too few road points ({len(points_road)}) to mesh")

    # Apply meshing to road points by assuming no outliers in the road segment at this point.
    meshing = DelaunayElevationMeshingAlgorithm(
        enable_downsampling=True,
        voxel_size=voxel_size,
        min_points_per_voxel=min_points_per_voxel,
        smoothing_passes=smoothing_passes,
        elevation_axis=2,
    )
    smooth_vertices, triangles, vertices, vertex_colors = meshing.build_mesh_from_points(
        points_road, colors=colors_road
    )

    return smooth_vertices, triangles, vertices, points, normals, road_mask, initial_road_mask, vertex_colors
