# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import logging
import os

from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from typing import List, Optional, Type

import numpy as np
import point_cloud_utils as pcu
import torch

from numpy.typing import NDArray
from omegaconf import DictConfig, ListConfig
from torch import nn

import ncore.data

from nre.config.model import SHGaussiansLayerConfig
from nre.config.trainer import TrainerConfig
from nre.datasets.ncore import NCOREDataSource
from nre.datasets.summary import DataSourceSummary
from nre.datasets.tracks import CuboidTracks
from nre.datasets.utils import compute_point_cloud_inside_tracks_mask
from nre.models.base import BaseModel
from nre.models.gaussians.utils import (
    RGB2SH,
    PLYGaussianLoader,
    sh_degree_to_specular_dim,
    subsample_points,
    track_random_initialization,
    uniform_sample_sphere,
)
from nre.models.utils import get_activation
from nre.utils.geometry import (
    se3_matrix_inverse,
    so3_matrix_to_quat,
    vector_align_rotation,
)
from nre.utils.knn import knn_points
from nre.utils.misc import sync_objects_or_raise, to_numpy, to_torch, unpack_optional
from nre.utils.profiling import ScopedTimer
from nre.utils.types import PointCloud
from nre.visualdebugger import get_visualdebugger


log = logging.getLogger(__name__)


class BaseInitialization(BaseModel, ABC):
    # no gaussians here to avoid circle import
    GAUSSIANS_INIT_VARIANTS: dict[str, Type[BaseInitialization]] = {}

    @staticmethod
    def register_to_gaussian_initialization_factory(name: str, cls: Type[BaseInitialization]) -> None:
        if name in BaseInitialization.GAUSSIANS_INIT_VARIANTS:
            raise KeyError(f"{name=} already in GAUSSIANS_INIT_VARIANTS.")
        BaseInitialization.GAUSSIANS_INIT_VARIANTS[name] = cls

    @staticmethod
    def factory(
        name: str,
        config: DictConfig,
        gaussian_config: SHGaussiansLayerConfig,
        trainer_config: TrainerConfig,
        labels_to_ignore: list[str],
        labels_to_use: list[str],
    ) -> BaseInitialization:
        return BaseInitialization.GAUSSIANS_INIT_VARIANTS[name](
            config, gaussian_config, trainer_config, labels_to_ignore, labels_to_use
        )

    def __init__(
        self,
        config: DictConfig,
        gaussian_config: SHGaussiansLayerConfig,
        trainer_config: TrainerConfig,
        labels_to_ignore: list[str],
        labels_to_use: list[str],
    ) -> None:
        super().__init__(config=config)

        self.gaussian_config = gaussian_config
        self.trainer_config = trainer_config
        self.labels_to_ignore = labels_to_ignore
        self.labels_to_use = labels_to_use
        self.gaussian_cuboid_ids: List[torch.Tensor] = []

        assert not (self.labels_to_ignore and self.labels_to_use), (
            f"[{self.__class__.__name__}] Only one of labels_to_ignore and labels_to_use should be specified"
        )

        self.debug_viz = self.gaussian_config.debug_viz
        if self.debug_viz:
            visualizer = get_visualdebugger()
            visualizer.set_properties(up="z_up", front="neg_x_front", navigation_style="free")

        self.debug_dir: Optional[str] = None
        if self.config.local_debug_dir is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.debug_dir = os.path.expanduser(os.path.join(self.config.local_debug_dir, timestamp))
            # Create all parent directories as needed
            os.makedirs(self.debug_dir, exist_ok=True)

    def initialize_from_datasource(self, datasource: DataSourceSummary, **kwargs) -> None:
        raise NotImplementedError

    def init_from_positions(
        self,
        positions: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
        observation_scale: Optional[torch.Tensor] = None,
    ) -> None:
        N = positions.shape[0]
        dtype = torch.float32
        density_activation_inv = get_activation(self.gaussian_config.density_activation, inverse=True)
        scale_activation_inv = get_activation(self.gaussian_config.scale_activation, inverse=True)

        # identity rotations
        rots = torch.zeros((N, 4), dtype=dtype, device=self.device)
        rots[:, 0] = 1.0  # they're quaternions stored in wxyz format

        if observation_scale is None:
            # Initialize isotropic scales if self.config.default_scale is a float, otherwise it should be a three element list
            assert isinstance(self.config.default_scale, float) or (
                isinstance(self.config.default_scale, ListConfig) and len(self.config.default_scale) == 3
            ), f"[{self.__class__.__name__}] default_scale must be a float or a 3-element list"

            observation_scale = (
                torch.full((N, 3), self.config.default_scale)
                if isinstance(self.config.default_scale, float)
                else torch.tensor(self.config.default_scale, device=self.device, dtype=torch.float32)
                .unsqueeze(0)
                .expand(positions.shape[0], -1)
            )

        scales = scale_activation_inv(observation_scale * self.config.scale_multiplier)

        # Repeat the scale in case only an isotropic scale was provided
        if scales.shape[1] == 1:
            scales = scales.repeat(1, 3)

        if normals is not None:
            mat = vector_align_rotation(
                torch.tensor([0, 0, 1], dtype=normals.dtype, device=self.device).repeat(normals.shape[0], 1),
                normals,
            )
            rots = so3_matrix_to_quat(mat)  # xyzw format
            rots = rots[:, [3, 0, 1, 2]]  # wxyz format
            if observation_scale.shape[1] == 1 or isinstance(self.config.default_scale, float):
                scales[..., 2] = scale_activation_inv(observation_scale * self.config.scale_multiplier / 10.0).squeeze(
                    -1
                )[..., 2]  # compress z-scale to bias Gaussians toward flatter geometry when using normals
            else:
                scales[..., 2] = scale_activation_inv(observation_scale * self.config.scale_multiplier).squeeze(-1)[
                    ..., 2
                ]  # use default scale if z-scale is provided

        # set density as a constant
        opacities = density_activation_inv(
            torch.full((N, 1), fill_value=self.config.default_density, dtype=dtype, device=self.device)
        )

        self.positions = nn.Parameter(positions.to(dtype=dtype, device=self.device))
        self.rotations = nn.Parameter(rots.to(dtype=dtype, device=self.device))
        self.scales = nn.Parameter(scales.to(dtype=dtype, device=self.device))
        self.densities = nn.Parameter(opacities.to(dtype=dtype, device=self.device))

    def init_additional_buffers(self, point_cloud: PointCloud) -> None:
        if self.gaussian_config.name in ["sh-gaussians", "rigid-gaussians", "deformable-gaussians"]:
            self.init_additional_buffers_SH(point_cloud)
        if self.debug_dir is not None:
            pcu.save_mesh_vnc(
                os.path.join(self.debug_dir, f"{self.gaussian_config.name}.ply"),
                to_numpy(point_cloud.xyz_end),
                to_numpy(point_cloud.normal) if point_cloud.normal is not None else None,
                to_numpy(point_cloud.color) if point_cloud.color is not None else None,
            )

    def init_additional_buffers_SH(self, point_cloud: PointCloud) -> None:
        dtype = torch.float32
        N = self.get_num_gaussians()

        # Initialize colors, constant if they weren't given
        if point_cloud.color is None:
            features_albedo = torch.rand((N, 3), dtype=dtype, device=self.device) / 255.0
        else:
            num_random_points = N - point_cloud.color.shape[0]

            features_albedo = torch.cat(
                [
                    to_torch(RGB2SH(to_numpy(point_cloud.color.float() / 255.0)), dtype=dtype, device=self.device),
                    # Check this part for random init
                    torch.rand((num_random_points, 3), dtype=dtype, device=self.device) / 255.0,
                ]
            )

        num_specular_dims = sh_degree_to_specular_dim(self.gaussian_config.progressive_training.max_n_features)
        features_specular = torch.zeros(N, num_specular_dims, device=self.device, dtype=dtype)
        self.features_albedo = nn.Parameter(features_albedo)
        self.features_specular = nn.Parameter(features_specular)

    def get_num_gaussians(self) -> int:
        return self.positions.shape[0]

    def filter_point_cloud_by_labels(self, point_cloud: PointCloud, datasource: DataSourceSummary) -> PointCloud:
        """
        Filter point cloud based on labels_to_use or labels_to_ignore.
        If neither is provided, return the point cloud as is.
        If both are provided, raise an error as this class requires only one of them to be provided.
        """

        if not (self.labels_to_use or self.labels_to_ignore):
            # No labels to filter by, return the point cloud as is
            return point_cloud
        else:
            assert bool(self.labels_to_ignore) ^ bool(self.labels_to_use), (
                f"[{self.__class__.__name__}] Only one of labels_to_ignore and labels_to_use should be specified"
            )

            assert point_cloud.semantic_class_id is not None, (
                f"{self.__class__.__name__}: Semantic class id is not present in the point cloud"
            )

            assert isinstance(datasource.datasource, NCOREDataSource), (
                f"[{self.__class__.__name__}] Only NCOREDataSource is supported"
            )

            sensor_type = unpack_optional(point_cloud.sensor_type)
            stuff_classes = unpack_optional(
                datasource.datasource.get_semantic_classes_map(
                    camera_semantics="camera" in sensor_type, lidar_semantics="lidar" in sensor_type
                )
            )

            if self.labels_to_use:
                keep_idx = torch.zeros_like(point_cloud.semantic_class_id, dtype=torch.bool)
                for label in self.labels_to_use:
                    label_idx = stuff_classes[label]
                    keep_idx = torch.logical_or(keep_idx, point_cloud.semantic_class_id == label_idx)
            else:
                keep_idx = torch.ones_like(point_cloud.semantic_class_id, dtype=torch.bool)
                for label in self.labels_to_ignore:
                    label_idx = stuff_classes[label]
                    keep_idx = torch.logical_and(keep_idx, point_cloud.semantic_class_id != label_idx)

            return point_cloud[keep_idx]


class RandomInitialization(BaseInitialization):
    def initialize_from_datasource(self, datasource: DataSourceSummary, **kwargs) -> None:
        """
        Initialize points randomly within a bounding box
        """

        # We create random points inside the bounds of the AABB
        aabb = datasource.get_aabb().to(self.device)

        positions = (
            torch.rand((self.config.num_points, 3), dtype=torch.float32, device=self.device) * aabb.get_extent()
            + aabb.blb
        )

        # Just initialize the points here, random colors will be initialzied in init_additional_buffers
        point_cloud = PointCloud(xyz_start=positions, xyz_end=positions)
        dist = (
            torch.sqrt(knn_points(positions.unsqueeze(0), positions.unsqueeze(0), K=2).dists[0, :, 1]).clamp_min(1e-3)
        ).unsqueeze(1)

        self.init_from_positions(positions=positions, observation_scale=dist)
        self.init_additional_buffers(point_cloud)


class SFMPointCloudInitialization(BaseInitialization):
    def initialize_from_datasource(self, datasource: DataSourceSummary, **kwargs) -> None:
        """
        Initialize points from SFM point cloud
        """
        pcs = [pc for pc in unpack_optional(datasource.get_point_clouds(device=self.device, color_type="generic-data"))]
        if not pcs:
            raise RuntimeError("sfm-point-cloud initialization received no points (color_type='generic-data').")
        point_cloud = PointCloud.collate_fn(pcs, device=self.device)
        # TODO(tylerz): shall we support the multiple camera sensors?
        #     (Qi): we should support multiple camera sensors in the future.
        assert isinstance(datasource.datasource, NCOREDataSource), (
            "Only NCOREDataSource is supported for sfm-point-cloud initialization"
        )
        assert len(datasource.datasource.camera_sensors) == 1, (
            "Only one camera sensor is currently supported for sfm-point-cloud initialization"
        )
        camera_sensor = list(datasource.datasource.camera_sensors.values())[0]
        camera_centers: list[np.ndarray] = []
        for idx in range(camera_sensor.frames_count):
            T_sensor_world_end = camera_sensor.get_frames_T_sensor_target(
                target_node="world",
                frame_indices=idx,
                frame_timepoint=ncore.data.FrameTimepoint.END,
            )  # 4x4
            camera_centers.append(T_sensor_world_end[:3, 3])
        camera_positions = torch.tensor(np.stack(camera_centers), device=self.device)
        dist = (
            torch.sqrt(knn_points(point_cloud.xyz_end.unsqueeze(0), camera_positions.unsqueeze(0)).dists)
            .clamp_min(1e-7)
            .squeeze(0)
        )
        scale = dist * self.config.scale_factor
        self.init_from_positions(positions=point_cloud.xyz_end, observation_scale=scale)
        self.init_additional_buffers(point_cloud)


class LidarRigTrajInitializationBase(BaseInitialization):
    @ScopedTimer(f"LidarRigTrajInitializationBase/get_point_clouds_and_ego_poses")
    def get_point_clouds_and_ego_poses(self, datasource: DataSourceSummary) -> tuple[PointCloud, torch.Tensor, float]:
        """
        Returns points and ego poses from the datasource, optionally filtering by class id.
        Subsamples the point cloud to the maximum size defined by `num_point_cloud_points` and returns a sparsity compensation factor
        that can be used to increase the scale of the remaining points.
        """

        assert isinstance(datasource.datasource, NCOREDataSource), (
            f"[{self.__class__.__name__}] Only NCOREDataSource is supported for from_lidar_rigtraj initialization"
        )

        # Function to read point clouds from the datasource
        def maybe_read_point_cloud():
            pcs = [
                pc
                for pc in unpack_optional(
                    datasource.get_point_clouds(
                        device=self.device,
                        lidar_ids=self.config.lidar_ids,
                        camera_ids=self.config.camera_ids,
                        non_dynamic_points_only=self.config.non_dynamic_points_only,
                        color_type="camera-rgb",
                        step_frame=self.config.step_frame,
                    )
                )
            ]

            assert len(pcs) > 0, "No point clouds were found in the dataset"
            point_cloud = PointCloud.collate_fn(pcs, device=self.device)

            # Filter the point cloud based on labels_to_use or labels_to_ignore
            if bool(self.labels_to_ignore) or bool(self.labels_to_use):
                point_cloud = self.filter_point_cloud_by_labels(point_cloud, datasource)

            original_n_points = point_cloud.n_points

            # subsample point cloud to desired size
            if point_cloud.n_points > self.config.num_point_cloud_points:
                random_idxs = torch.from_numpy(
                    np.random.choice(point_cloud.n_points, self.config.num_point_cloud_points, replace=False)
                ).to(self.device)
                point_cloud = point_cloud[random_idxs]

            return point_cloud, original_n_points

        # Actually read the point cloud
        point_cloud, original_n_points = maybe_read_point_cloud()

        # We need to use the original number of points (before subsampling) to compute the sparsity compensation factor
        sparsity_compensation = max(1, original_n_points / self.config.num_point_cloud_points)
        log.info(
            f"[{self.__class__.__name__}] Original number of points: {original_n_points}, number of points: {point_cloud.n_points}, sparsity compensation: {sparsity_compensation}"
        )

        # Get ego positions in NRE coordinates
        rig_trajectories = unpack_optional(datasource.get_rig_trajectories())
        T_world_to_nre_np, S_world_to_nre_np = rig_trajectories.world_to_nre.get_transformation_matrices()
        T_world_to_nre = torch.tensor(T_world_to_nre_np, device=self.device, dtype=torch.float32)
        S_world_to_nre = torch.tensor(S_world_to_nre_np, device=self.device, dtype=torch.float32)
        # Ego poses that map from camera_to_nre
        ego_c2nre = torch.cat(
            [
                (T_world_to_nre @ rig_traj.T_rig_worlds.float().to(self.device) @ S_world_to_nre)
                for rig_traj in rig_trajectories.rig_trajectories
            ]
        )

        return point_cloud, ego_c2nre, sparsity_compensation


class InitializationMixin(BaseInitialization):
    @abstractmethod
    def get_point_clouds_and_ego_poses(
        self, datasource: DataSourceSummary
    ) -> tuple[PointCloud, torch.Tensor, float]: ...

    def _initialize_from_datasource(self, datasource: DataSourceSummary, **kwargs) -> None:
        """
        Initialize from a combination of colored point clouds and random points.

        Random points are sampled from two strategies: near and far.
        Near points are sampled from a set of spheres of radius `near_radius` around each of the ego positions.
        Far points are sampled from a sphere of the size of the scene extent (centered around the ego positions in the middle of the sequence).

        Camera frusta are used to filter out invisible points.
        """
        assert isinstance(self, (LidarRigTrajInitialization, AccumulatedPointCloudInitialization)), (
            f"[{self.__class__.__name__}] This initialization is only supported for LidarRigTrajInitialization and AccumulatedPointCloudInitialization"
        )

        point_cloud, ego_c2nre, sparsity_compensation = self.get_point_clouds_and_ego_poses(datasource)
        ego_positions = ego_c2nre[:, :3, 3]

        scene_extent = datasource.get_aabb().get_extent().max().item()

        assert isinstance(datasource.datasource, NCOREDataSource), (
            f"[{self.__class__.__name__}] Only NCOREDataSource is supported for this initialization"
        )

        # Get all camera frusta
        def maybe_read_camera_frusta():
            return [
                camera_frustum.to(self.device)
                for camera_id in datasource.datasource.camera_ids
                for camera_frustum, _ in unpack_optional(
                    datasource.get_camera_frusta(
                        camera_id=camera_id, far_plane_depth=self.config.far_radius_factor * scene_extent / 2
                    )
                )
            ]

        camera_frusta = [camera_frustum.to(self.device) for camera_frustum in maybe_read_camera_frusta()]

        # sample random points
        random_pts_list: list[torch.Tensor] = []
        if self.debug_viz:
            visualizer = get_visualdebugger()
        else:
            visualizer = None

        # sample random points
        num_near_pts = self.config.num_near_points
        if num_near_pts > 0:  # distances uniformly from (0, 0.5 * scene_extent)
            near_pts = uniform_sample_sphere(num_near_pts, ego_positions.device) * scene_extent * 0.5
            if visualizer is not None:
                visualizer.add_point_cloud("random_near", near_pts.cpu().numpy(), radius=1e-3, point_render_mode="quad")
            random_pts_list.append(near_pts)

        num_far_pts = self.config.num_far_points
        if num_far_pts > 0:  # distances uniformly from (0, 10 * scene_extent)
            far_pts = (
                uniform_sample_sphere(
                    num_far_pts, ego_positions.device, inverse=True, radius=self.config.far_radius_factor
                )
                * scene_extent
                * 0.5
            )
            if visualizer is not None:
                visualizer.add_point_cloud("random_far", far_pts.cpu().numpy(), radius=1e-3, point_render_mode="quad")
            random_pts_list.append(far_pts)

        if num_near_pts > 0 or num_far_pts > 0:
            random_pts = torch.cat(random_pts_list, dim=0)

            # check visibility using camera frusta
            visibility_mask = (
                torch.vstack([camera_frustum.points_in_frustum(random_pts) for camera_frustum in camera_frusta]).sum(0)
                > 0
            )

            random_pts = random_pts[visibility_mask]
            if visualizer is not None:
                visualizer.add_point_cloud(
                    "random_points", random_pts.cpu().numpy(), radius=1e-3, point_render_mode="quad"
                )
                visualizer.add_point_cloud(
                    "ego_positions", ego_positions.cpu().numpy(), radius=1e-3, point_render_mode="quad"
                )
                visualizer.add_point_cloud(
                    "point_cloud",
                    point_cloud.xyz_end.cpu().numpy(),
                    radius=1e-3,
                    point_render_mode="quad",
                    colors_quantities={"colored": unpack_optional(point_cloud.color).cpu().numpy()},
                    enabled=True,
                )
                visualizer.show()

            # positions
            positions = torch.cat([point_cloud.xyz_end.to(self.device), random_pts])

            # normals
            normals = None
            if point_cloud.normal is not None:
                normals = torch.cat(
                    [
                        point_cloud.normal.to(self.device),
                        torch.tensor([0, 0, 1], dtype=point_cloud.normal.dtype, device=self.device).repeat(
                            random_pts.shape[0], 1
                        ),
                    ]
                )

            # get scales based on distance to ego poses
            dist_to_ego_positions = (
                torch.sqrt(knn_points(random_pts.unsqueeze(0), ego_positions.to(positions).unsqueeze(0)).dists)
                .clamp_min(1e-7)
                .squeeze(0)
            )
            observation_scale = torch.cat(
                [
                    unpack_optional(point_cloud.camera_footprint_scale).to(self.device).unsqueeze(-1)
                    * np.sqrt(sparsity_compensation),
                    dist_to_ego_positions * self.config.observation_scale_factor,
                ]
            )
        else:
            positions = point_cloud.xyz_end.to(self.device)
            normals = point_cloud.normal.to(self.device) if point_cloud.normal is not None else None
            observation_scale = unpack_optional(point_cloud.camera_footprint_scale).to(self.device).unsqueeze(
                -1
            ) * np.sqrt(sparsity_compensation)

        if self.config.get("init_with_normals", False):
            self.init_from_positions(positions=positions, normals=normals, observation_scale=observation_scale)
        else:
            self.init_from_positions(positions=positions, observation_scale=observation_scale)
        self.init_additional_buffers(point_cloud)


class LidarRigTrajInitialization(LidarRigTrajInitializationBase, InitializationMixin):
    def initialize_from_datasource(self, datasource: DataSourceSummary, **kwargs) -> None:
        """
        Initialize from a combination of colored point clouds and random points.

        Random points are sampled from two strategies: near and far.
        Near points are sampled from a set of spheres of radius `near_radius` around each of the ego positions.
        Far points are sampled from a sphere of the size of the scene extent (centered around the ego positions in the middle of the sequence).

        Camera frusta are used to filter out invisible points.
        """
        self._initialize_from_datasource(datasource, **kwargs)


class RoadInitializationMixin(BaseInitialization):
    @abstractmethod
    def get_point_clouds_and_ego_poses(
        self, datasource: DataSourceSummary
    ) -> tuple[PointCloud, torch.Tensor, float]: ...

    def _project_to_ego_z_plane(
        self,
        points: torch.Tensor,
        ego_c2nre: torch.Tensor,
        ego_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project points onto the local ego z-plane to encourage flat road geometry.

        For each point, finds the nearest ego pose, transforms the point into that ego's
        local coordinate frame, sets z to ``config.z_offset``, then transforms back to
        NRE world space.

        Args:
            points: World-space positions to project. Shape ``[N, 3]``.
            ego_c2nre: Ego-to-NRE SE(3) transforms. Shape ``[M, 4, 4]``.
            ego_positions: Ego translation vectors (``ego_c2nre[:, :3, 3]``). Shape ``[M, 3]``.

        Returns:
            A tuple of:
                - projected_points: Projected world-space positions. Shape ``[N, 3]``.
                - nearest_ego_indices: Index of the nearest ego pose per point. Shape ``[N]``.
        """
        nearest_ego_indices = (points[..., None, :] - ego_positions).norm(dim=-1).argmin(dim=-1)
        ego_c2nre_per_point = ego_c2nre[nearest_ego_indices]
        ego_nre2c_per_point = se3_matrix_inverse(ego_c2nre_per_point)

        local = (ego_nre2c_per_point[:, :3, :3] @ points[..., :, None] + ego_nre2c_per_point[:, :3, 3:4]).squeeze(-1)
        local[..., 2] = self.config.get("z_offset", 0)

        projected_points = (
            ego_c2nre_per_point[:, :3, :3] @ local[..., :, None] + ego_c2nre_per_point[:, :3, 3:4]
        ).squeeze(-1)
        return projected_points, nearest_ego_indices

    def _initialize_from_datasource(self, datasource: DataSourceSummary, **kwargs) -> None:
        """
        Initialize from colored point clouds and with HUGSIM's strategy which encourages a flatter road initialization
        by setting the height of each point to that of the closest ego position. This initialization is intended to be used
        in conjunction with a default_scale configuration that initializes the points to be flat disks
        (via a parameter such as [0.1, 0.1, 0.001])
        """

        assert isinstance(self, (LidarRigTrajRoadInitialization, AccumulatedPointCloudRoadInitialization)), (
            f"[{self.__class__.__name__}] This initialization is only supported for LidarRigTrajRoadInitialization and AccumulatedPointCloudRoadInitialization"
        )

        point_cloud, ego_c2nre, sparsity_compensation = self.get_point_clouds_and_ego_poses(datasource)
        ego_positions = ego_c2nre[:, :3, 3]  # [M, 3]
        positions = point_cloud.xyz_end.to(self.device)  # [N, 3]

        base_scale = (
            self.config.default_scale if isinstance(self.config.default_scale, float) else self.config.default_scale[0]
        )

        # Set the height of each point to that of the closest ego position (as in HUGSIM)
        # This encourages a flatter road initialization
        if self.config.project_to_z_offset:
            positions, pc_nearest_ego_indices = self._project_to_ego_z_plane(positions, ego_c2nre, ego_positions)
        else:
            pc_nearest_ego_indices = (positions[..., None, :] - ego_positions).norm(dim=-1).argmin(dim=-1)

        if point_cloud.camera_footprint_scale is not None:
            pc_scale = unpack_optional(point_cloud.camera_footprint_scale).to(self.device).unsqueeze(-1).clamp_min(
                base_scale
            ) * np.sqrt(sparsity_compensation)
        else:
            pc_scale = torch.full(
                (positions.shape[0], 1), base_scale, device=self.device, dtype=torch.float32
            ) * np.sqrt(sparsity_compensation)

        pc_normals = None
        if self.config.init_with_normals:
            if point_cloud.normal is not None:
                pc_normals = point_cloud.normal.to(self.device)
            else:
                normals_local = torch.tensor([0, 0, 1], dtype=positions.dtype, device=self.device).repeat(
                    positions.shape[0], 1
                )
                pc_normals = (ego_c2nre[pc_nearest_ego_indices][:, :3, :3] @ normals_local[..., :, None]).squeeze(-1)

        # initialize random points along the ego trajectory
        random_pts = None
        rand_scale = None
        rand_normals = None

        if self.config.num_random_points > 0:
            scene_extent = datasource.get_aabb().get_extent().max().item()
            num_random_pts = self.config.num_random_points

            # Get camera frusta for visibility filtering
            camera_frusta = [
                camera_frustum.to(self.device)
                for camera_id in getattr(datasource.datasource, "camera_ids", [])
                for camera_frustum, _ in unpack_optional(
                    datasource.get_camera_frusta(
                        camera_id=camera_id, far_plane_depth=self.config.far_radius_factor * scene_extent / 2
                    )
                )
            ]

            # Sample base positions uniformly from the ego trajectory
            # ego_positions shape is [M, 3]
            idx = torch.randint(0, ego_positions.shape[0], (num_random_pts,), device=self.device)
            base_pts = ego_positions[idx]

            road_width_radius = self.config.get("random_road_width_radius", 15.0)
            r = torch.sqrt(torch.rand(num_random_pts, 1, device=self.device)) * road_width_radius
            theta = torch.rand(num_random_pts, 1, device=self.device) * 2 * np.pi

            offset_x = r * torch.cos(theta)
            offset_y = r * torch.sin(theta)
            offset_z = torch.zeros_like(offset_x)

            random_pts = base_pts + torch.cat([offset_x, offset_y, offset_z], dim=1)

            visibility_mask = (
                torch.vstack([frustum.points_in_frustum(random_pts) for frustum in camera_frusta]).sum(0) > 0
            )
            random_pts = random_pts[visibility_mask]

            random_pts, rand_nearest_ego_indices = self._project_to_ego_z_plane(random_pts, ego_c2nre, ego_positions)

            dist_to_ego = (
                torch.sqrt(knn_points(random_pts.unsqueeze(0), ego_positions.unsqueeze(0)).dists)
                .clamp_min(1e-7)
                .squeeze(0)
            )
            rand_scale = (dist_to_ego * self.config.get("observation_scale_factor", 0.01)).clamp_min(base_scale)

            if self.config.init_with_normals:
                normals_local = torch.tensor([0, 0, 1], dtype=random_pts.dtype, device=self.device).repeat(
                    random_pts.shape[0], 1
                )
                rand_normals = (ego_c2nre[rand_nearest_ego_indices][:, :3, :3] @ normals_local[..., :, None]).squeeze(
                    -1
                )

        if random_pts is not None:
            assert rand_scale is not None
            all_positions = torch.cat([positions, random_pts], dim=0)
            observation_scale = torch.cat([pc_scale, rand_scale], dim=0)
            normals = (
                torch.cat([pc_normals, rand_normals], dim=0)
                if (pc_normals is not None and rand_normals is not None)
                else None
            )
        else:
            all_positions = positions
            observation_scale = pc_scale
            normals = pc_normals

        if not isinstance(self.config.default_scale, float):
            z_scale = torch.full(
                (all_positions.shape[0], 1), self.config.default_scale[2], device=self.device, dtype=torch.float32
            )
            observation_scale = torch.cat([observation_scale, observation_scale, z_scale], dim=1)

        if self.debug_viz:
            visualizer = get_visualdebugger()
            visualizer.add_point_cloud(
                "road_point_cloud",
                positions.cpu().numpy(),
                radius=1e-3,
                point_render_mode="quad",
                colors_quantities={"colored": unpack_optional(point_cloud.color).cpu().numpy()},
                enabled=True,
            )
            if random_pts is not None:
                visualizer.add_point_cloud(
                    "road_random_points",
                    random_pts.cpu().numpy(),
                    radius=1e-3,
                    point_render_mode="quad",
                    enabled=True,
                )
            visualizer.show()

        if normals is not None:
            self.init_from_positions(positions=all_positions, normals=normals, observation_scale=observation_scale)
        else:
            self.init_from_positions(positions=all_positions, observation_scale=observation_scale)

        self.init_additional_buffers(point_cloud)


class LidarRigTrajRoadInitialization(LidarRigTrajInitializationBase, RoadInitializationMixin):
    def initialize_from_datasource(self, datasource: DataSourceSummary, **kwargs) -> None:
        """
        Initialize from colored point clouds and with HUGSIM's strategy which encourages a flatter road initialization
        by setting the height of each point to that of the closest ego position. This initialization is intended to be used
        in conjunction with a default_scale configuration that initializes the points to be flat disks
        (via a parameter such as [0.1, 0.1, 0.001])
        """
        self._initialize_from_datasource(datasource, **kwargs)


class NoPointsFoundException(Exception):
    pass


class DynamicTracksInitialization(BaseInitialization):
    def initialize_from_datasource(self, datasource: DataSourceSummary, **kwargs) -> None:
        cuboid_tracks = kwargs.get("cuboid_tracks", None)
        assert isinstance(cuboid_tracks, CuboidTracks), (
            f"[{self.__class__.__name__}] cuboid_tracks is required and must be CuboidTracks"
        )

        # TODO filter out cuboid tracks ahead of time
        point_clouds: List[PointCloud] = []
        gaussian_cuboid_ids: List[torch.Tensor] = []
        gaussian_scales: List[torch.Tensor] = []
        observed_counts: List[int] = []

        track_point_clouds = list(
            unpack_optional(
                datasource.get_track_point_clouds(
                    cuboid_tracks=cuboid_tracks,
                    cuboid_dim_scale_factor=1.0,
                    lidar_ids=self.config.get("lidar_ids", None),
                    camera_ids=self.config.get("camera_ids", None),
                    return_color=True,
                    step_frame=self.config.get("step_frame", 1),
                    keep_all_track_poses=self.config.keep_all_track_poses,
                    device=self.device,
                )
            )
        )

        track_id_counts: dict[str, int] = defaultdict(int)
        track_pcs: dict[str, list[PointCloud]] = defaultdict(list)
        track_gids: dict[str, list[torch.Tensor]] = defaultdict(list)
        track_scales: dict[str, list[torch.Tensor]] = defaultdict(list)
        for track_point_cloud in track_point_clouds:
            point_cloud = track_point_cloud.point_cloud

            if self.config.symmetric_axis is not None:
                symmetric_point_cloud = point_cloud.flipped(axis=self.config.symmetric_axis)
                point_cloud = PointCloud.collate_fn([point_cloud, symmetric_point_cloud], device=self.device)

            track_id_counts[track_point_cloud.track_id] += point_cloud.n_points

            track_pcs[track_point_cloud.track_id].append(point_cloud.to(self.device))
            track_gids[track_point_cloud.track_id].append(
                torch.full(
                    (point_cloud.n_points, 1),
                    fill_value=cuboid_tracks.tracks_id.index(track_point_cloud.track_id),
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            track_scales[track_point_cloud.track_id].append(
                unpack_optional(point_cloud.camera_footprint_scale).unsqueeze(-1)
            )

        for track_id in cuboid_tracks.tracks_id:
            # Add tracks with no point clouds yielded to the list. This can happen to a track if:
            # 1. all its cuboids are skipped by step_frame
            # 2. all its cuboids are skipped by timestamp checks when keep_all_track_poses is false
            if track_id not in track_id_counts:
                track_id_counts[track_id] = 0

        # subsample the point clouds to desired size
        for track_id, count in track_id_counts.items():
            if count == 0:
                # Add random initialization for tracks with no detected points
                track_index = cuboid_tracks.tracks_id.index(track_id)
                cuboid_extent = cuboid_tracks.cuboids_dims[track_index].to(self.device)

                point_cloud, gaussian_cuboid_id, gaussian_scale = track_random_initialization(
                    self.config.num_point_cloud_points_per_track,
                    track_index,
                    cuboid_extent,
                    self.config.default_scale,
                    self.device,
                )

                point_clouds.append(point_cloud)
                gaussian_cuboid_ids.append(gaussian_cuboid_id)
                gaussian_scales.append(gaussian_scale)
                observed_counts.append(
                    0
                )  # These are random points, so no need to track the observed count for sparsity compensation
            elif count > self.config.num_point_cloud_points_per_track:
                # uniformly sample points from each point cloud
                original_observed_counts = [pc.n_points for pc in track_pcs[track_id]]
                subsampled_track_pcs, subsampled_track_gids, subsampled_track_scales, subsampled_observed_counts = (
                    subsample_points(
                        self.config.num_point_cloud_points_per_track,
                        track_pcs[track_id],
                        track_gids[track_id],
                        track_scales[track_id],
                        original_observed_counts,
                    )
                )

                point_clouds.extend(subsampled_track_pcs)
                gaussian_cuboid_ids.extend(subsampled_track_gids)
                gaussian_scales.extend(subsampled_track_scales)
                observed_counts.extend(subsampled_observed_counts)
            else:
                point_clouds.extend(track_pcs[track_id])
                gaussian_cuboid_ids.extend(track_gids[track_id])
                gaussian_scales.extend(track_scales[track_id])
                observed_counts.extend([pc.n_points for pc in track_pcs[track_id]])

                if self.config.get("fill_with_random_points", False):
                    num_random_points = self.config.num_point_cloud_points_per_track - count
                    # Add random initialization for tracks
                    track_index = cuboid_tracks.tracks_id.index(track_id)
                    cuboid_extent = cuboid_tracks.cuboids_dims[track_index].to(self.device)

                    point_cloud, gaussian_cuboid_id, gaussian_scale = track_random_initialization(
                        num_random_points,
                        track_index,
                        cuboid_extent,
                        self.config.default_scale,
                        self.device,
                    )

                    point_clouds.append(point_cloud)
                    gaussian_cuboid_ids.append(gaussian_cuboid_id)
                    gaussian_scales.append(gaussian_scale)
                    observed_counts.append(
                        0
                    )  # These are random points, so no need to track the observed count for sparsity compensation

        if len(point_clouds) == 0:
            raise NoPointsFoundException("No point clouds were found in the dataset")

        if self.config.num_point_cloud_points_in_layer is not None:
            # Further subsample the point clouds to the desired number of points in the layer
            point_clouds, gaussian_cuboid_ids, gaussian_scales, observed_counts = subsample_points(
                self.config.num_point_cloud_points_in_layer,
                point_clouds,
                gaussian_cuboid_ids,
                gaussian_scales,
                observed_counts,
            )

        point_cloud = PointCloud.collate_fn(point_clouds, device=self.device)

        # Sparsity compensation
        observation_scales = []
        for scale, observed_count in zip(gaussian_scales, observed_counts):
            sparsity_compensation = max(1, np.sqrt(observed_count / scale.shape[0])) if observed_count > 0 else 1
            observation_scales.append(scale * sparsity_compensation)

        self.init_from_positions(
            point_cloud.xyz_end,
            observation_scale=torch.cat(observation_scales),
        )
        self.init_additional_buffers(point_cloud)
        self.gaussian_cuboid_ids = gaussian_cuboid_ids


class AccumulatedPointCloudInitializationBase(BaseInitialization):
    @staticmethod
    def load_ply_with_segment_id(
        filename: str,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.uint8], Optional[NDArray[np.uint8]]]:
        ply_data = pcu.load_triangle_mesh(filename).vertex_data
        ply_metadata = ply_data.custom_attributes

        xyz = ply_data.positions
        normal = ply_data.normals
        color = ply_data.colors
        segment_id = ply_metadata["segment_id"].astype(np.uint8) if "segment_id" in ply_metadata else None

        return xyz, normal, color, segment_id

    def get_point_clouds_and_ego_poses(self, datasource: DataSourceSummary) -> tuple[PointCloud, torch.Tensor, float]:
        """
        Returns points loaded from a ply file and ego poses from the datasource, optionally filtering by class id.
        Subsamples the point cloud to the maximum size defined by `num_point_cloud_points` and returns a sparsity compensation factor
        that can be used to increase the scale of the remaining points.
        """

        assert isinstance(datasource.datasource, (NCOREDataSource)), (
            f"[{self.__class__.__name__}] Only NCOREDataSource is supported for from_accumulated_point_cloud initialization"
        )

        # Get ego positions in NRE coordinates
        rig_trajectories = unpack_optional(datasource.get_rig_trajectories())
        T_world_to_nre_np, S_world_to_nre_np = rig_trajectories.world_to_nre.get_transformation_matrices()
        T_world_to_nre = torch.tensor(T_world_to_nre_np, device=self.device, dtype=torch.float32)
        S_world_to_nre = torch.tensor(S_world_to_nre_np, device=self.device, dtype=torch.float32)
        # Ego poses that map from camera_to_nre
        ego_c2nre = torch.cat(
            [
                (T_world_to_nre @ rig_traj.T_rig_worlds.float().to(self.device) @ S_world_to_nre)
                for rig_traj in rig_trajectories.rig_trajectories
            ]
        )

        def maybe_read_point_cloud():
            xyz_e_world, normal_world, color, segment_id = self.load_ply_with_segment_id(self.config.point_cloud_path)

            xyz_e_nre = datasource.datasource.world_to_nre.transform_points(xyz_e_world)

            normal_nre = None
            if normal_world is not None:
                # Rotate normal from world to NRE coordinates, multiply from the right with the transpose is equivalent to multiply from the left
                normal_nre = normal_world @ datasource.datasource.world_to_nre.matrix[:3, :3].T
                normal_nre = normal_nre / np.maximum(np.linalg.norm(normal_nre, axis=1)[..., np.newaxis], 1e-5)

            xyz_e_nre = to_torch(xyz_e_nre, device=str(self.device), dtype=torch.float32)
            normal_nre = (
                to_torch(normal_nre, device=str(self.device), dtype=torch.float32) if normal_nre is not None else None
            )
            color = to_torch(color[:, :3], device=str(self.device), dtype=torch.uint8) if color is not None else None
            segment_id = (
                to_torch(segment_id, device=str(self.device), dtype=torch.uint8) if segment_id is not None else None
            )

            # get scales based on distance to ego poses
            dist_to_ego_positions = (
                torch.sqrt(knn_points(xyz_e_nre.unsqueeze(0), ego_c2nre[:, :3, 3].unsqueeze(0)).dists)
                .clamp_min(1e-7)
                .squeeze(0)
                .squeeze(-1)
            )
            scale = dist_to_ego_positions * self.config.get("observation_scale_factor", 0.01)

            point_cloud = PointCloud(
                xyz_start=xyz_e_nre,
                xyz_end=xyz_e_nre,
                flags=None,
                color=color,
                normal=normal_nre,
                semantic_class_id=segment_id,
                camera_footprint_scale=scale,
                sensor_type=["camera"],
            )

            # Filter the point cloud based on labels_to_use or labels_to_ignore
            if bool(self.labels_to_ignore) or bool(self.labels_to_use):
                point_cloud = self.filter_point_cloud_by_labels(point_cloud, datasource)
            if self.config.non_dynamic_points_only:
                assert isinstance(datasource.datasource, NCOREDataSource), (
                    "non_dynamic_points_only requires NCOREDataSource"
                )
                cuboidtracks_dynamic = datasource.datasource.get_cuboid_tracks(dynamic_only=True, world_frame=False)
                dynamic_class_ids = None
                if point_cloud.semantic_class_id is not None:
                    sensor_type = unpack_optional(point_cloud.sensor_type)
                    dynamic_class_ids = datasource.datasource.get_dynamic_semantic_class_ids(
                        sensor_type=list(sensor_type),
                        dynamic_class_names=list(datasource.datasource.camera_point_cloud_dynamic_classes),
                    )

                inside_dynamic_tracks_mask = compute_point_cloud_inside_tracks_mask(
                    point_cloud=point_cloud,
                    time_range_us=datasource.datasource.time_range_us,
                    cuboidtracks_dynamic=cuboidtracks_dynamic,
                    track_padding_m=datasource.datasource.valid_points_cuboid_track_params.track_padding_m,
                    batch_size=20,
                    tqdm_disabled=datasource.datasource.tqdm_disabled,
                    restrict_to_class_ids=dynamic_class_ids,
                )
                point_cloud = point_cloud[~inside_dynamic_tracks_mask]

            original_n_points = point_cloud.n_points

            # subsample point cloud to desired size
            if point_cloud.n_points > self.config.num_point_cloud_points:
                random_idxs = torch.from_numpy(
                    np.random.choice(point_cloud.n_points, self.config.num_point_cloud_points, replace=False)
                ).to(self.device)
                point_cloud = point_cloud[random_idxs]

            return point_cloud, original_n_points

        # Actually read the point cloud
        point_cloud, original_n_points = maybe_read_point_cloud()

        # We need to use the original number of points (before subsampling) to compute the sparsity compensation factor
        sparsity_compensation = max(1, original_n_points / self.config.num_point_cloud_points)
        log.info(
            f"[{self.__class__.__name__}] Original number of points: {original_n_points}, number of points: {point_cloud.n_points}, sparsity compensation: {sparsity_compensation}"
        )

        return point_cloud, ego_c2nre, sparsity_compensation


class AccumulatedPointCloudInitialization(AccumulatedPointCloudInitializationBase, InitializationMixin):
    def initialize_from_datasource(self, datasource: DataSourceSummary, **kwargs) -> None:
        """
        Initialize from a combination of colored point clouds and random points.

        The point cloud is loaded from a ply file and then transformed to NRE coordinates.
        Random points are sampled from two strategies: near and far.
        Near points are sampled from a set of spheres of radius `near_radius` around each of the ego positions.
        Far points are sampled from a sphere of the size of the scene extent (centered around the ego positions in the middle of the sequence).

        Camera frusta are used to filter out invisible points.
        """
        self._initialize_from_datasource(datasource, **kwargs)


class AccumulatedPointCloudRoadInitialization(AccumulatedPointCloudInitializationBase, RoadInitializationMixin):
    def initialize_from_datasource(self, datasource: DataSourceSummary, **kwargs) -> None:
        """
        Initialize from colored point clouds, which are loaded from a ply file, and with HUGSIM's strategy which encourages a flatter road
        initialization by setting the height of each point to that of the closest ego position. This initialization is intended to be used
        in conjunction with a default_scale configuration that initializes the points to be flat disks
        (via a parameter such as [0.1, 0.1, 0.001])
        """
        self._initialize_from_datasource(datasource, **kwargs)


class NRMPlyInitialization(BaseInitialization):
    def get_point_clouds_and_ego_poses(self, datasource: DataSourceSummary, road: bool) -> dict[str, torch.Tensor]:
        """
        Returns points and other gaussian properties loaded from a ply file, filtered by a road mask.
        Randomly subsamples the point cloud to the maximum size defined by `num_point_cloud_points`.
        The input pointcloud can be generated by running nrm with the export_ply config option.

        This initialization assumes the following:
        - scale and density are activated.
        - position and rotation are in world space, and must be converted to NRE space.
        - rotation is a quaternion in wxyz format.
        - features_albedo is in SH, not RGB.
        """

        assert isinstance(datasource.datasource, (NCOREDataSource)), (
            f"[{self.__class__.__name__}] Only NCOREDataSource is supported for from_accumulated_point_cloud initialization"
        )

        loaded_ply = PLYGaussianLoader(self.config.path, device=str(self.device))

        # Map from world space to NRE space
        loaded_ply.transform(
            torch.FloatTensor(datasource.datasource.world_to_nre.matrix).to(self.device),
        )

        log.info(f"[{self.__class__.__name__}] Original number of points: {loaded_ply.positions.shape[0]}")
        # PRUNING
        # Remove points with infinite densities
        mask = torch.isfinite(loaded_ply.densities.squeeze())
        log.info(f"[{self.__class__.__name__}] Kept {mask.sum().item()} finite points")

        # Remove points based on road bool (road layer or not)
        assert loaded_ply.road_mask is not None, "Road mask is required for initialization"
        if road:
            mask &= loaded_ply.road_mask
            log.info(f"[{self.__class__.__name__}] Kept {mask.sum().item():,} road points")
        else:
            mask &= ~loaded_ply.road_mask
            log.info(f"[{self.__class__.__name__}] Kept {mask.sum().item():,} non-road points")

        # Remove points based on density
        if self.config.density_pruning.enabled:
            # Only keep the top-k points with the highest densities, where k = self.config.density_pruning.keep
            number_gaussians_to_keep = self.config.density_pruning.keep
            current_n_points = mask.sum().item()
            if number_gaussians_to_keep is not None and number_gaussians_to_keep < current_n_points:
                # Get the indices of the points in the mask that are True
                idx_in_full = mask.nonzero(as_tuple=True)[0]
                # Get the densities of the masked-in points
                masked_densities = loaded_ply.densities.squeeze()[mask]
                # Find the top-k densities and their indices within masked_densities
                # Compute the indices to be removed: smallest densities (not in topk)
                removed_indices_in_masked = torch.topk(
                    masked_densities, current_n_points - number_gaussians_to_keep, largest=False, dim=0
                ).indices
                # Map back to original mask
                mask[idx_in_full[removed_indices_in_masked]] = False
                log.info(f"[{self.__class__.__name__}] Kept {mask.sum().item()} topk density points")
            # only keep points with densities greater than the threshold
            if self.config.density_pruning.density_threshold is not None:
                mask &= (
                    get_activation("sigmoid", inverse=False)(loaded_ply.densities.squeeze())
                    > self.config.density_pruning.density_threshold
                )
                log.info(
                    f"[{self.__class__.__name__}] Kept {mask.sum().item()} density threshold ({self.config.density_pruning.density_threshold}) points"
                )

        # randomly remove points if there are too many
        original_n_points = mask.sum().item()
        if original_n_points > self.config.num_point_cloud_points:
            true_indices = mask.nonzero(as_tuple=False).flatten()
            removal_count = int(original_n_points) - self.config.num_point_cloud_points
            removal_idcs = true_indices[torch.randperm(int(original_n_points), device=self.device)[:removal_count]]
            mask[removal_idcs] = False
            log.info(f"[{self.__class__.__name__}] Kept {mask.sum().item()} points after random subsampling")
        sparsity_compensation = max(1, original_n_points / self.config.num_point_cloud_points)
        log.info(
            f"[{self.__class__.__name__}] Original number of points before random sampling: {original_n_points}, new number of points: {mask.sum().item()}, sparsity compensation: {sparsity_compensation}"
        )

        densities = loaded_ply.densities[mask, :]
        scales = loaded_ply.scales[mask, :]
        positions = loaded_ply.positions[mask, :]
        rotations = loaded_ply.rotations[mask, :]
        features_albedo = loaded_ply.features_albedo[mask, :]  # albedo features are in SH not RGB

        if loaded_ply.features_specular is not None:
            features_specular = loaded_ply.features_specular[mask, :]
        else:
            num_specular_dims = sh_degree_to_specular_dim(self.gaussian_config.progressive_training.max_n_features)
            features_specular = torch.zeros(positions.shape[0], num_specular_dims, device=self.device)

        return {
            "positions": positions,
            "rotations": rotations,
            "densities": densities,
            "scales": scales,
            "features_albedo": features_albedo,
            "features_specular": features_specular,
        }

    def initialize_from_datasource(self, datasource: DataSourceSummary, road: bool = False, **kwargs) -> None:
        """
        Initialize from one or more ply files obtained from NRM.
        """

        point_cloud_dict = self.get_point_clouds_and_ego_poses(datasource, road)
        self.positions = nn.Parameter(point_cloud_dict["positions"])
        self.rotations = nn.Parameter(point_cloud_dict["rotations"])
        self.scales = nn.Parameter(point_cloud_dict["scales"])
        self.densities = nn.Parameter(point_cloud_dict["densities"])
        self.features_albedo = nn.Parameter(point_cloud_dict["features_albedo"])
        self.features_specular = nn.Parameter(point_cloud_dict["features_specular"])


class NRMPlyRoadInitialization(NRMPlyInitialization):
    def initialize_from_datasource(self, datasource: DataSourceSummary, road: bool = True, **kwargs) -> None:
        """
        Initialize from one or more ply files obtained from NRM, but with only the points labeled as road
        """
        super().initialize_from_datasource(datasource, road=True, **kwargs)


class PlyInitialization(BaseInitialization):
    def initialize_from_datasource(self, datasource: DataSourceSummary, **kwargs) -> None:
        """
        Initialize from a ply file (ie: one obtained from NRM). The ply positions are assumed to be in world space.
        """

        loaded_ply = PLYGaussianLoader(self.config.path, device=str(self.device))

        if self.config.translate_world_to_nre:
            assert isinstance(datasource.datasource, NCOREDataSource), (
                f"[{self.__class__.__name__}] Only NCOREDataSource is supported for world to nre translation"
            )
            # Map from world space to NRE space
            loaded_ply.transform(
                torch.FloatTensor(datasource.datasource.world_to_nre.matrix).to(self.device),
            )

        self.positions = nn.Parameter(loaded_ply.positions)
        self.rotations = nn.Parameter(loaded_ply.rotations)
        self.scales = nn.Parameter(loaded_ply.scales)
        self.densities = nn.Parameter(loaded_ply.densities)
        self.features_albedo = nn.Parameter(loaded_ply.features_albedo)
        if loaded_ply.features_specular is not None:
            self.features_specular = nn.Parameter(loaded_ply.features_specular)
        else:
            num_specular_dims = sh_degree_to_specular_dim(self.gaussian_config.progressive_training.max_n_features)
            self.features_specular = nn.Parameter(
                torch.zeros(self.positions.shape[0], num_specular_dims, device=self.device)
            )


class PlyTracksInitialization(BaseInitialization):
    def initialize_from_datasource(self, datasource: DataSourceSummary, **kwargs) -> None:
        """
        Initialize specified tracks with ply files and use a fallback initializer for remaining tracks
        """
        cuboid_tracks = kwargs.get("cuboid_tracks", None)
        assert isinstance(cuboid_tracks, CuboidTracks), (
            f"[{self.__class__.__name__}] cuboid_tracks is required and must be CuboidTracks"
        )

        track_ids = set(self.config.track_ids.keys())
        ply_track_ids = []
        fallback_track_ids = []

        for i, track_id in enumerate(cuboid_tracks.tracks_id):
            if track_id in track_ids:
                ply_track_ids.append(i)
            else:
                fallback_track_ids.append(i)

        positions: List[torch.Tensor] = []
        rotations: List[torch.Tensor] = []
        scales: List[torch.Tensor] = []
        densities: List[torch.Tensor] = []
        features_albedo: List[torch.Tensor] = []
        features_specular: List[torch.Tensor] = []
        gaussian_cuboid_ids: List[torch.Tensor] = []

        if len(fallback_track_ids) > 0:
            log.info(f"Initializing {len(fallback_track_ids)} fallback tracks using {self.config.fallback.name}")
            try:
                fallback_tracks = CuboidTracks.Ops.subset_from_indices(cuboid_tracks, fallback_track_ids)
                fallback = BaseInitialization.factory(
                    self.config.fallback.name,
                    self.config.fallback,
                    self.gaussian_config,
                    self.trainer_config,
                    self.labels_to_ignore,
                    self.labels_to_use,
                )
                fallback.initialize_from_datasource(datasource, cuboid_tracks=fallback_tracks)

                positions.append(fallback.positions)
                rotations.append(fallback.rotations)
                scales.append(fallback.scales)
                densities.append(fallback.densities)
                features_albedo.append(fallback.features_albedo)
                features_specular.append(fallback.features_specular)
                gaussian_cuboid_ids.append(
                    torch.IntTensor(fallback_track_ids).to(self.device)[torch.cat(fallback.gaussian_cuboid_ids)]
                )
            except NoPointsFoundException:
                log.warning("No points found for fallback initialization")
                pass

        object_completion_tracks = CuboidTracks.Ops.subset_from_indices(cuboid_tracks, ply_track_ids)
        transform = torch.FloatTensor(self.config.transform).to(self.device)
        scale_activation = get_activation(self.gaussian_config.scale_activation, inverse=False)
        scale_activation_inv = get_activation(self.gaussian_config.scale_activation, inverse=True)

        for track_index, track_id in enumerate(object_completion_tracks.tracks_id):
            loaded_ply = PLYGaussianLoader(self.config.track_ids[track_id], device=str(self.device))

            loaded_ply.transform(transform)
            if self.config.scale_to_track:
                original_dims = object_completion_tracks.cuboids_dims[track_index]
                # Prefer dataset-configured padding if available; fallback to init/gaussian config or default
                padding_values = None
                if isinstance(datasource.datasource, NCOREDataSource):
                    # First, try lidar-points padding
                    padding_values = list(
                        getattr(
                            datasource.datasource.valid_points_cuboid_track_params,
                            "track_padding_m",
                            [],
                        )
                    )

                if not padding_values:
                    raise ValueError(f"No padding values found for track {track_id}")

                padding_tensor = torch.tensor(padding_values, dtype=original_dims.dtype, device=original_dims.device)
                adjusted_dims = torch.clamp(original_dims - padding_tensor, min=1e-6)
                scale_factor = adjusted_dims.norm().item()

                loaded_ply.scale(
                    scale_factor,
                    scale_activation,
                    scale_activation_inv,
                )
            positions.append(loaded_ply.positions)
            rotations.append(loaded_ply.rotations)
            scales.append(loaded_ply.scales)
            densities.append(loaded_ply.densities)
            features_albedo.append(loaded_ply.features_albedo)
            if loaded_ply.features_specular is not None:
                features_specular.append(loaded_ply.features_specular)
            else:
                num_specular_dims = sh_degree_to_specular_dim(self.gaussian_config.progressive_training.max_n_features)
                features_specular.append(
                    torch.zeros(
                        loaded_ply.features_albedo.shape[0],
                        num_specular_dims,
                        device=self.device,
                    )
                )
            gaussian_cuboid_ids.append(
                torch.full(
                    (loaded_ply.features_albedo.shape[0], 1),
                    fill_value=ply_track_ids[track_index],
                    dtype=torch.int32,
                    device=self.device,
                )
            )

        self.positions = nn.Parameter(torch.cat(positions))
        self.rotations = nn.Parameter(torch.cat(rotations))
        self.scales = nn.Parameter(torch.cat(scales))
        self.densities = nn.Parameter(torch.cat(densities))
        self.features_albedo = nn.Parameter(torch.cat(features_albedo))
        self.features_specular = nn.Parameter(torch.cat(features_specular))
        self.gaussian_cuboid_ids = gaussian_cuboid_ids


def compute_vertex_normals(vertices: NDArray[np.float32], faces: NDArray[np.int32]) -> NDArray[np.float32]:
    """Compute area-weighted vertex normals from a triangle mesh."""
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = face_normals / np.maximum(norms, 1e-8)

    vertex_normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_normals)
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals = vertex_normals / np.maximum(norms, 1e-8)
    return vertex_normals


class LidarGroundMeshRoadInitialization(BaseInitialization):
    """Initialize road Gaussians from a reconstructed colored ground mesh.

    Builds a Delaunay ground mesh from LiDAR point clouds at init time,
    using the same reconstruction pipeline as the ground-mesh artifact export.
    Mesh vertices supply Gaussian positions and (optionally) per-vertex RGB
    colors; face normals orient Gaussians to lie flat on the road surface.
    """

    def _build_ground_mesh_init_state(self, ncore_ds: NCOREDataSource) -> tuple[NDArray, NDArray, NDArray | None]:
        from nre.utils.io.ground_mesh import (
            get_nominal_ground_point_under_lidar,
            reconstruct_ground_mesh_from_points,
        )

        log.info(f"[{self.__class__.__name__}] Building colored ground mesh for road initialization …")

        # Always request colors for initialization: projected camera RGB improves SH starting values
        # vs random init. The artifact export path has a separate `colored` flag that controls the
        # output format, which is a different concern.
        pc_gen = ncore_ds.get_point_clouds(
            device=torch.device("cpu"),
            color_type="camera-rgb",
            non_dynamic_points_only=True,
            step_frame=self.config.get("step_frame", 1),
        )

        nre_to_world = ncore_ds.world_to_nre.inverse()

        smooth_vertices, triangles, _, _, _, _, _, vertex_colors = reconstruct_ground_mesh_from_points(
            point_cloud_generator=pc_gen,
            min_ray_length=self.config.get("min_ray_length", 0.1),
            ground_control_point=get_nominal_ground_point_under_lidar(ncore_ds),
            ground_compat_max_distance=self.config.get("ground_compat_max_distance", 0.5),
            ground_compat_max_angle_deg=self.config.get("ground_compat_max_angle_deg", 60.0),
            num_plane_hypotheses=self.config.get("num_plane_hypotheses", 100),
            plane_max_distance=self.config.get("plane_max_distance", 0.3),
            plane_max_angle_deg=self.config.get("plane_max_angle_deg", 30),
            plane_min_eval_points=self.config.get("plane_min_eval_points", 1000),
            plane_max_eval_points=self.config.get("plane_max_eval_points", 10000),
            plane_min_inliers=self.config.get("plane_min_inliers", 1000),
            enable_plane_extension=self.config.get("enable_plane_extension", False),
            voxel_size=self.config.get("voxel_size", 0.1),
            min_points_per_voxel=self.config.get("min_points_per_voxel", 1),
            smoothing_passes=self.config.get("smoothing_passes", 10),
            points_to_world_transf=nre_to_world,
            verbose=False,
        )

        log.info(
            f"[{self.__class__.__name__}] Ground mesh: {len(smooth_vertices)} vertices, {len(triangles)} triangles, "
            f"colors={'yes' if vertex_colors is not None else 'no'}"
        )

        # Transform mesh vertices from world space back to NRE space
        positions_nre = ncore_ds.world_to_nre.transform_points(smooth_vertices)
        normals_world = compute_vertex_normals(smooth_vertices, triangles)
        normals_nre = normals_world @ ncore_ds.world_to_nre.matrix[:3, :3].T
        normals_nre = normals_nre / np.maximum(np.linalg.norm(normals_nre, axis=1, keepdims=True), 1e-8)

        if self.debug_dir is not None:
            from nre.utils.io.ply import save_ply

            mesh_path = os.path.join(self.debug_dir, "ground_mesh_road_init.ply")
            save_ply(
                mesh_path,
                positions_nre,
                triangles,
                normals=normals_nre,
                colors=vertex_colors,
                logger=log,
            )

        positions_nre = np.asarray(positions_nre, dtype=np.float32)
        normals_nre = np.asarray(normals_nre, dtype=np.float32)
        if vertex_colors is not None:
            vertex_colors = np.asarray(vertex_colors, dtype=np.uint8)

        return positions_nre, normals_nre, vertex_colors

    def initialize_from_datasource(self, datasource: DataSourceSummary, **kwargs) -> None:
        ncore_ds = datasource.datasource
        assert isinstance(ncore_ds, NCOREDataSource), f"[{self.__class__.__name__}] Only NCOREDataSource is supported"

        num_random = self.config.get("num_random_points", None)
        assert num_random is None or num_random == 0, (
            f"[{self.__class__.__name__}] does not support random points (got num_random_points={num_random}). "
            f"Set num_random_points to 0 or remove it from the config."
        )

        # Build the road mesh once on rank 0, then share the CPU init state with all ranks.
        positions_nre, normals_nre, vertex_colors = sync_objects_or_raise(
            lambda: self._build_ground_mesh_init_state(ncore_ds)
        )

        positions = torch.from_numpy(positions_nre).float().to(self.device)

        # Build flat-disk scales
        default_scale = self.config.get("default_scale", [0.1, 0.1, 0.001])
        if isinstance(default_scale, (list, ListConfig)):
            scale_tensor = torch.tensor(default_scale, device=self.device, dtype=torch.float32)
            observation_scale = scale_tensor.unsqueeze(0).expand(positions.shape[0], -1)
        else:
            observation_scale = torch.full((positions.shape[0], 3), default_scale, device=self.device)

        normals_tensor = torch.from_numpy(normals_nre).float().to(self.device)

        if isinstance(default_scale, (list, ListConfig)):
            effective_scale = [s * self.config.scale_multiplier for s in default_scale]
        else:
            effective_scale = default_scale * self.config.scale_multiplier
        log.info(
            f"[{self.__class__.__name__}] scale_multiplier={self.config.scale_multiplier}, "
            f"default_scale={list(default_scale)}, effective_scale={effective_scale}, "
            f"num_positions={positions.shape[0]}"
        )

        self.init_from_positions(
            positions=positions,
            normals=normals_tensor,
            observation_scale=observation_scale,
        )

        # Build a PointCloud so that init_additional_buffers_SH converts
        # mesh vertex colors via RGB2SH.
        mesh_color = None
        if vertex_colors is not None:
            mesh_color = torch.from_numpy(vertex_colors).to(self.device)
        point_cloud = PointCloud(
            xyz_start=positions,
            xyz_end=positions,
            color=mesh_color,
        )
        self.init_additional_buffers(point_cloud)


class CameraGroundMeshRoadInitialization(BaseInitialization):
    """Initialize road Gaussians from a ground mesh built from camera depth point clouds.

    Uses camera metric-depth point clouds with semantic road labels to build a
    Delaunay ground mesh.  Unlike the LiDAR-based ``LidarGroundMeshRoadInitialization``,
    this skips the LiDAR-specific normal estimation and RANSAC road segmentation,
    relying on camera semantic labels (``ROAD_SEMANTIC``) to identify road points
    and feeding them directly into ``DelaunayElevationMeshingAlgorithm``.
    """

    def _load_mesh_from_file(self, mesh_path: str) -> tuple[NDArray, NDArray, NDArray | None]:
        """Load a ground mesh PLY and return (vertices, faces, vertex_colors_or_None).

        The mesh is expected to be in **world space**.  Per-vertex normals in the
        file (if any) are ignored — normals are always recomputed from faces.
        """
        log.info(f"[{self.__class__.__name__}] Loading ground mesh from {mesh_path}")
        mesh_data = pcu.load_triangle_mesh(mesh_path)
        vertices = mesh_data.vertex_data.positions.astype(np.float32)
        faces = mesh_data.face_data.vertex_ids.astype(np.int32)

        vertex_colors = None
        raw_colors = getattr(mesh_data.vertex_data, "colors", None)
        if raw_colors is not None:
            vertex_colors = raw_colors[:, :3].astype(np.uint8)

        log.info(
            f"[{self.__class__.__name__}] Loaded mesh: {len(vertices)} vertices, {len(faces)} faces, "
            f"colors={'yes' if vertex_colors is not None else 'no'}"
        )
        return vertices, faces, vertex_colors

    def _build_mesh_from_fused_point_cloud(
        self, ncore_ds: NCOREDataSource
    ) -> tuple[NDArray[np.float32], NDArray[np.int32], "NDArray[np.uint8] | None"]:
        """Build a Delaunay ground mesh from a fused point cloud PLY with segment_id.

        Loads the accumulated point cloud, filters road points by semantic class
        (``segment_id``), then runs Delaunay elevation meshing with voxel-based
        color averaging.

        Returns ``(smooth_vertices, triangles, vertex_colors)`` in **world space**.
        """
        from nre.utils.io.ground_mesh import DelaunayElevationMeshingAlgorithm

        point_cloud_path = self.config.point_cloud_path
        log.info(f"[{self.__class__.__name__}] Loading fused point cloud from {point_cloud_path}")

        ply_data = pcu.load_triangle_mesh(point_cloud_path).vertex_data
        ply_metadata = ply_data.custom_attributes

        xyz_world = ply_data.positions.astype(np.float32)
        color = ply_data.colors
        assert "segment_id" in ply_metadata, f"[{self.__class__.__name__}] segment_id is not present in the point cloud"
        segment_id = ply_metadata["segment_id"]

        log.info(
            f"[{self.__class__.__name__}] Loaded {len(xyz_world)} points, colors={'yes' if color is not None else 'no'}"
        )

        stuff_classes = unpack_optional(ncore_ds.get_semantic_classes_map(camera_semantics=True, lidar_semantics=False))
        assert self.labels_to_use, (
            f"[{self.__class__.__name__}] labels_to_use (from layer class_labels) must be set "
            f"to identify road points in the fused point cloud"
        )
        road_mask = np.zeros(len(xyz_world), dtype=bool)
        for label in self.labels_to_use:
            if label in stuff_classes:
                road_mask |= segment_id == stuff_classes[label]
                log.info(f"[{self.__class__.__name__}] Label '{label}' → class_id={stuff_classes[label]}")
            else:
                log.warning(f"[{self.__class__.__name__}] Label '{label}' not found in semantic classes")

        road_points = xyz_world[road_mask]
        road_colors = None
        if color is not None:
            road_colors = color[road_mask][:, :3].astype(np.uint8)

        log.info(
            f"[{self.__class__.__name__}] Filtered {len(road_points)} road points from {len(xyz_world)} total, "
            f"colors={'yes' if road_colors is not None else 'no'}"
        )

        if len(road_points) < 3:
            raise ValueError(
                f"[{self.__class__.__name__}] Too few road points ({len(road_points)}) to mesh. "
                f"Check point_cloud_path and semantic labels."
            )

        meshing = DelaunayElevationMeshingAlgorithm(
            enable_downsampling=True,
            voxel_size=self.config.voxel_size,
            min_points_per_voxel=self.config.min_points_per_voxel,
            smoothing_passes=self.config.smoothing_passes,
            elevation_axis=2,
        )
        smooth_vertices, triangles, _, vertex_colors = meshing.build_mesh_from_points(road_points, colors=road_colors)

        return smooth_vertices, triangles, vertex_colors

    def initialize_from_datasource(self, datasource: DataSourceSummary, **kwargs) -> None:
        ncore_ds = datasource.datasource
        assert isinstance(ncore_ds, NCOREDataSource), f"[{self.__class__.__name__}] Only NCOREDataSource is supported"

        num_random = self.config.get("num_random_points", None)
        assert num_random is None or num_random == 0, (
            f"[{self.__class__.__name__}] does not support random points (got num_random_points={num_random}). "
            f"Set num_random_points to 0 or remove it from the config."
        )

        mesh_path = self.config.get("mesh_path", None)
        if mesh_path is not None:
            smooth_vertices, triangles, vertex_colors = self._load_mesh_from_file(mesh_path)
        else:
            assert self.config.get("point_cloud_path", None) is not None, (
                f"[{self.__class__.__name__}] Either mesh_path or point_cloud_path must be set"
            )
            smooth_vertices, triangles, vertex_colors = self._build_mesh_from_fused_point_cloud(ncore_ds)

        if len(smooth_vertices) < 3 or len(triangles) == 0:
            raise ValueError(
                f"[{self.__class__.__name__}] Loaded mesh is degenerate "
                f"({len(smooth_vertices)} vertices, {len(triangles)} faces). "
                f"Check the input file."
            )

        log.info(
            f"[{self.__class__.__name__}] Ground mesh: {len(smooth_vertices)} vertices, {len(triangles)} triangles, "
            f"colors={'yes' if vertex_colors is not None else 'no'}"
        )

        positions_nre = ncore_ds.world_to_nre.transform_points(smooth_vertices)
        normals_world = compute_vertex_normals(smooth_vertices, triangles)
        normals_nre = normals_world @ ncore_ds.world_to_nre.matrix[:3, :3].T
        normals_nre = normals_nre / np.maximum(np.linalg.norm(normals_nre, axis=1, keepdims=True), 1e-8)

        if self.debug_dir is not None:
            from nre.utils.io.ply import save_ply

            debug_mesh_path = os.path.join(self.debug_dir, "camera_ground_mesh_road_init.ply")
            save_ply(
                debug_mesh_path,
                positions_nre,
                triangles,
                normals=normals_nre,
                colors=vertex_colors,
                logger=log,
            )

        positions = torch.from_numpy(positions_nre).float().to(self.device)
        normals_tensor = torch.from_numpy(normals_nre).float().to(self.device)

        default_scale = self.config.get("default_scale", [0.1, 0.1, 0.001])
        if isinstance(default_scale, (list, ListConfig)):
            scale_tensor = torch.tensor(default_scale, device=self.device, dtype=torch.float32)
        else:
            scale_tensor = torch.full((3,), default_scale, device=self.device, dtype=torch.float32)
        observation_scale = scale_tensor.unsqueeze(0).expand(positions.shape[0], -1)

        effective_scale = (scale_tensor * self.config.scale_multiplier).tolist()
        log.info(
            f"[{self.__class__.__name__}] scale_multiplier={self.config.scale_multiplier}, "
            f"default_scale={scale_tensor.tolist()}, effective_scale={effective_scale}, "
            f"num_positions={positions.shape[0]}"
        )

        self.init_from_positions(
            positions=positions,
            normals=normals_tensor,
            observation_scale=observation_scale,
        )

        mesh_color = None
        if vertex_colors is not None:
            mesh_color = torch.from_numpy(vertex_colors).to(self.device)
        point_cloud = PointCloud(
            xyz_start=positions,
            xyz_end=positions,
            color=mesh_color,
        )
        self.init_additional_buffers(point_cloud)


BaseInitialization.register_to_gaussian_initialization_factory("random", RandomInitialization)
BaseInitialization.register_to_gaussian_initialization_factory("sfm-point-cloud", SFMPointCloudInitialization)
BaseInitialization.register_to_gaussian_initialization_factory("lidar-rig-trajectory", LidarRigTrajInitialization)
BaseInitialization.register_to_gaussian_initialization_factory(
    "lidar-rig-trajectory-road", LidarRigTrajRoadInitialization
)
BaseInitialization.register_to_gaussian_initialization_factory("lidar-dynamic-tracks", DynamicTracksInitialization)
BaseInitialization.register_to_gaussian_initialization_factory("camera-dynamic-tracks", DynamicTracksInitialization)
BaseInitialization.register_to_gaussian_initialization_factory(
    "accumulated-point-cloud", AccumulatedPointCloudInitialization
)
BaseInitialization.register_to_gaussian_initialization_factory(
    "accumulated-point-cloud-road", AccumulatedPointCloudRoadInitialization
)
BaseInitialization.register_to_gaussian_initialization_factory("nrm-ply", NRMPlyInitialization)
BaseInitialization.register_to_gaussian_initialization_factory("nrm-road-ply", NRMPlyRoadInitialization)
BaseInitialization.register_to_gaussian_initialization_factory("ply", PlyInitialization)
BaseInitialization.register_to_gaussian_initialization_factory("ply-tracks", PlyTracksInitialization)
BaseInitialization.register_to_gaussian_initialization_factory(
    "lidar-ground-mesh-road", LidarGroundMeshRoadInitialization
)
BaseInitialization.register_to_gaussian_initialization_factory(
    "camera-ground-mesh-road", CameraGroundMeshRoadInitialization
)
