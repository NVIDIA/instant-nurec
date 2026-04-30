# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import types
import unittest

from typing import Optional, cast

import numpy as np
import pytest
import torch

from numpy.typing import NDArray
from omegaconf import OmegaConf

import nre.models.gaussians.initializations as initializations_module

from nre.config.model import SHGaussiansLayerConfig
from nre.config.trainer import TrainerConfig
from nre.datasets.base import BaseDataSource
from nre.datasets.ncore import NCOREDataSource
from nre.datasets.summary import DataSourceSummary
from nre.datasets.tracks import CuboidTracks, TrackFlags
from nre.models.gaussians.initializations import (
    AccumulatedPointCloudInitializationBase,
    CameraGroundMeshRoadInitialization,
    DynamicTracksInitialization,
    LidarGroundMeshRoadInitialization,
    LidarRigTrajInitialization,
    LidarRigTrajRoadInitialization,
    NoPointsFoundException,
    compute_vertex_normals,
    pcu,
)
from nre.utils.geometry import quat_to_so3_matrix
from nre.utils.types import (
    CameraFrustum,
    FrameConversion,
    PointCloud,
    RigTrajectories,
    TrackPointCloud,
)


class _StubNcore(NCOREDataSource):
    def __init__(
        self,
        camera_ids: list[str],
        point_clouds: list[PointCloud],
        track_point_clouds: Optional[dict[str, TrackPointCloud]] = None,
        tracks_skip: Optional[set[str]] = None,
    ):
        # Intentionally skip parent init; provide only required members
        self.camera_ids = camera_ids
        self._pcs = point_clouds
        self._track_point_clouds = track_point_clouds or {}
        self._tracks_skip = tracks_skip or set()
        self.world_to_nre = FrameConversion.from_origin_scale_axis(
            target_origin=np.zeros(3, dtype=np.float32), target_scale=1.0, target_axis=[0, 1, 2]
        )

    def get_semantic_classes_map(self, camera_semantics: bool, lidar_semantics: bool):
        # Minimal mapping covering camera/lidar; the caller ignores which is used beyond presence
        return {"road": 1, "vehicle": 2, "sky": 3}

    def get_point_clouds(
        self,
        device: torch.device,
        lidar_ids=None,
        camera_ids=None,
        valid_points_only=True,
        non_dynamic_points_only=True,
        color_type=None,
        step_frame: int = 1,
        visualize: bool = False,
        force: bool = True,
    ):
        def generator():
            for pc in self._pcs:
                yield pc

        return generator()

    def get_camera_frusta(
        self,
        camera_id: str | None = None,
        near_plane_depth: float = 0.1,
        far_plane_depth: float = 150.0,
        step_frame: int = 1,
    ):
        # Simple axis-aligned frustum-like box corners
        near_z = 0.1
        far_z = 2.0
        corners = torch.tensor(
            [
                [-1.0, -1.0, near_z],
                [-1.0, 1.0, near_z],
                [1.0, 1.0, near_z],
                [1.0, -1.0, near_z],
                [-1.0, -1.0, far_z],
                [-1.0, 1.0, far_z],
                [1.0, 1.0, far_z],
                [1.0, -1.0, far_z],
            ],
            dtype=torch.float32,
        )

        def generator():
            yield CameraFrustum(corners=corners.clone()), 0

        return generator()

    def get_track_point_clouds(
        self,
        cuboid_tracks,
        cuboid_dim_scale_factor: float = 1.0,
        lidar_ids: Optional[list[str]] = None,
        camera_ids: Optional[list[str]] = None,
        return_color: bool = False,
        step_frame: int = 1,
        keep_all_track_poses: bool = False,
        device: torch.device = torch.device("cuda"),
    ):
        def generator():
            for track_id, track_pc in self._track_point_clouds.items():
                if track_id in self._tracks_skip:
                    continue
                yield track_pc

        return generator()


class _FakeFrustum:
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def points_in_frustum(self, points: torch.Tensor) -> torch.Tensor:
        # Accept all points
        return torch.ones(points.shape[0], dtype=torch.bool, device=points.device)

    def to(self, device: torch.device):
        self.device = device
        return self


class _TrajObj:
    def __init__(self, rigs: list[torch.Tensor]):
        self.world_to_nre = FrameConversion.from_origin_scale_axis(
            target_origin=np.zeros(3, dtype=np.float32), target_scale=1.0, target_axis=[0, 1, 2]
        )

        class _Rig:
            def __init__(self, T_seq: torch.Tensor):
                self.T_rig_worlds = T_seq

        self.rig_trajectories = [_Rig(T) for T in rigs]


def _make_summary(
    pc_list: list[PointCloud],
    rig_T_list: list[torch.Tensor],
    camera_ids: list[str],
    datasource_override: Optional[BaseDataSource] = None,
) -> DataSourceSummary:
    state = DataSourceSummary.State(
        n_frames_per_camera=np.zeros(1, dtype=np.int32),
        n_frames_per_lidar=np.zeros(0, dtype=np.int32),
        sequence_tracks_all=None,
        sequence_tracks_dynamic=None,
        rig_trajectories=cast(Optional[RigTrajectories], _TrajObj(rig_T_list)),
        xform_matrices=np.zeros((0, 4), dtype=np.float32),
        aabb_blb=np.array([[-0.5, -0.5, -0.5]], dtype=np.float32),
        aabb_trf=np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
    )
    ds = (
        datasource_override
        if datasource_override is not None
        else _StubNcore(camera_ids=camera_ids, point_clouds=pc_list)
    )
    return DataSourceSummary(datasource=ds, state=state)


def _make_gaussian_cfg():
    # Minimal gaussian config for tests
    # Use DictConfig for nested configs since they're typed as Any
    return SHGaussiansLayerConfig(
        name="sh-gaussians",
        density_activation="sigmoid",
        scale_activation="exp",
        rotation_activation="normalize",
        progressive_training=OmegaConf.create({"max_n_features": 16}),
        particle=OmegaConf.create({}),
        debug_viz=False,
    )


def _make_trainer_cfg() -> TrainerConfig:
    return TrainerConfig(
        world_size=1,
        num_nodes=1,
        device_count=1,
        relative_lr=False,
        relative_schedule=False,
        relative_num_workers=False,
        batch_size_scaling_factor=1.0,
        training_step_scaling_factor=1.0,
        max_epochs=1,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        precision="32",
    )


class TestBaseInitialization(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def test_filter_point_cloud_by_labels(self):
        # 5 points with class ids: [road(1), vehicle(2), sky(3), road(1), vehicle(2)]
        ids = torch.tensor([1, 2, 3, 1, 2], dtype=torch.uint8, device=self.device)
        xyz = torch.zeros((5, 3), dtype=torch.float32, device=self.device)
        pc = PointCloud(xyz_start=xyz, xyz_end=xyz, semantic_class_id=ids, sensor_type=["camera"])

        rig_T = torch.eye(4, device=self.device, dtype=torch.float32).unsqueeze(0)
        summary = _make_summary(pc_list=[pc], rig_T_list=[rig_T], camera_ids=["cam0"])

        def make_model(labels_to_use: list[str], labels_to_ignore: list[str]):
            init_cfg = OmegaConf.create(
                {
                    "default_scale": 0.2,
                    "default_density": 0.5,
                    "scale_multiplier": 1.0,
                    "local_debug_dir": None,
                }
            )
            return LidarRigTrajInitialization(
                config=init_cfg,
                gaussian_config=_make_gaussian_cfg(),
                trainer_config=_make_trainer_cfg(),
                labels_to_ignore=labels_to_ignore,
                labels_to_use=labels_to_use,
            ).to(self.device)

        # labels_to_use
        model_use = make_model(labels_to_use=["road", "sky"], labels_to_ignore=[])
        out_use = model_use.filter_point_cloud_by_labels(pc, summary)
        self.assertTrue(out_use == pc[torch.tensor([0, 2, 3], dtype=torch.long, device=self.device)])

        # labels_to_ignore
        model_ignore = make_model(labels_to_use=[], labels_to_ignore=["vehicle"])  # remove id==2
        out_ignore = model_ignore.filter_point_cloud_by_labels(pc, summary)
        self.assertTrue(out_ignore == pc[torch.tensor([0, 2, 3], dtype=torch.long, device=self.device)])

        # neither labels_to_use nor labels_to_ignore are provided
        model_none = make_model(labels_to_use=[], labels_to_ignore=[])
        out_none = model_none.filter_point_cloud_by_labels(pc, summary)
        self.assertTrue(out_none == pc)

        # error case 1: both labels_to_use and labels_to_ignore specified (assert XOR in method)
        model_both = make_model(labels_to_use=["road"], labels_to_ignore=[])
        model_both.labels_to_ignore = ["vehicle"]  # mutate after construction to bypass __init__ assert
        with self.assertRaises(AssertionError):
            _ = model_both.filter_point_cloud_by_labels(pc, summary)

        # error case 2: missing semantic_class_id in point cloud
        pc_no_semantic = PointCloud(xyz_start=xyz, xyz_end=xyz, sensor_type=["camera"])
        with self.assertRaises(AssertionError):
            _ = model_use.filter_point_cloud_by_labels(pc_no_semantic, summary)

        # error case 3: unsupported datasource type (not NCOREDataSource)
        class _NonNcoreDatasource:
            def __init__(self, camera_ids: list[str]):
                self.camera_ids = camera_ids

            def get_semantic_classes_map(self, camera_semantics: bool, lidar_semantics: bool):
                return {"road": 1, "vehicle": 2, "sky": 3}

        bad_summary = DataSourceSummary(
            datasource=cast(BaseDataSource, _NonNcoreDatasource(camera_ids=["cam0"])), state=summary.state
        )
        with self.assertRaises(AssertionError):
            _ = model_use.filter_point_cloud_by_labels(pc, bad_summary)


class TestInitializationMixin(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def test_lidar_rig_traj_initialization_positions_and_buffers(self):
        torch.manual_seed(0)

        # Create a small colored point cloud
        n_pc = 10
        xyz = torch.randn(n_pc, 3, device=self.device, dtype=torch.float32) * 0.1
        color = (torch.linspace(0, 255, n_pc, device=self.device).unsqueeze(1).repeat(1, 3)).to(torch.uint8)
        normal = torch.nn.functional.normalize(torch.randn(n_pc, 3, device=self.device), dim=-1)
        scale = torch.full((n_pc,), 0.05, device=self.device)
        pc = PointCloud(
            xyz_start=xyz, xyz_end=xyz, color=color, normal=normal, camera_footprint_scale=scale, sensor_type=["camera"]
        )

        # Two identity rig frames
        rig_T = torch.eye(4, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
        summary = _make_summary(pc_list=[pc], rig_T_list=[rig_T], camera_ids=["cam0", "cam1"])

        init_cfg = OmegaConf.create(
            {
                "num_near_points": 5,
                "num_far_points": 3,
                "far_radius_factor": 10.0,
                "observation_scale_factor": 0.01,
                "lidar_ids": None,
                "camera_ids": None,
                "step_frame": 1,
                "point_cloud_device": str(self.device),
                "num_point_cloud_points": 100,
                "default_scale": 0.2,
                "default_density": 0.5,
                "local_debug_dir": None,
                "scale_multiplier": 1.0,
                "non_dynamic_points_only": True,
            }
        )

        model = LidarRigTrajInitialization(
            config=init_cfg,
            gaussian_config=SHGaussiansLayerConfig.model_validate(_make_gaussian_cfg()),
            trainer_config=_make_trainer_cfg(),
            labels_to_ignore=[],
            labels_to_use=[],
        ).to(self.device)

        model.initialize_from_datasource(summary)

        # Expected: original points + near + far (filtered kept all)
        self.assertLessEqual(model.positions.shape[0], n_pc + init_cfg.num_near_points + init_cfg.num_far_points)
        self.assertEqual(model.rotations.shape[0], model.positions.shape[0])
        self.assertEqual(model.scales.shape[0], model.positions.shape[0])
        self.assertEqual(model.densities.shape, (model.positions.shape[0], 1))
        # SH buffers created
        self.assertEqual(model.features_albedo.shape[0], model.positions.shape[0])
        self.assertEqual(model.features_albedo.shape[1], 3)
        self.assertGreater(model.features_specular.shape[1], 0)


class TestRoadInitializationMixin(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def test_flatten_road_z_offset(self):
        # Build a small point cloud at varying z
        n_pc = 6
        xyz = torch.stack(
            [
                torch.linspace(-0.1, 0.1, n_pc, device=self.device),
                torch.linspace(-0.2, 0.2, n_pc, device=self.device),
                torch.linspace(-1.0, 1.0, n_pc, device=self.device),
            ],
            dim=1,
        )
        pc = PointCloud(
            xyz_start=xyz, xyz_end=xyz, camera_footprint_scale=torch.ones(n_pc, dtype=torch.float32, device=self.device)
        )

        # Identity ego poses
        rig_T = torch.eye(4, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)
        summary = _make_summary(pc_list=[pc], rig_T_list=[rig_T], camera_ids=["cam0"])

        init_cfg = OmegaConf.create(
            {
                "project_to_z_offset": True,
                "z_offset": 0.0,
                "num_point_cloud_points": 100,
                "num_random_points": 10,
                "far_radius_factor": 10.0,
                "observation_scale_factor": 0.01,
                "default_scale": [0.1, 0.1, 0.001],
                "default_density": 0.5,
                "local_debug_dir": None,
                "scale_multiplier": 1.0,
                "lidar_ids": None,
                "camera_ids": None,
                "step_frame": 1,
                "point_cloud_device": str(self.device),
                "non_dynamic_points_only": True,
                "init_with_normals": False,
            }
        )

        model = LidarRigTrajRoadInitialization(
            config=init_cfg,
            gaussian_config=SHGaussiansLayerConfig.model_validate(_make_gaussian_cfg()),
            trainer_config=_make_trainer_cfg(),
            labels_to_ignore=[],
            labels_to_use=[],
        ).to(self.device)

        model.initialize_from_datasource(summary)

        # All z coordinates should be set to z_offset in world (identity transforms)
        self.assertTrue(
            torch.allclose(model.positions[:, 2], torch.full((n_pc,), init_cfg.z_offset, device=self.device))
        )

    def test_init_with_normals(self):
        n_pc = 10
        xyz = torch.stack(
            [
                torch.linspace(-0.1, 0.1, n_pc, device=self.device),
                torch.linspace(-0.2, 0.2, n_pc, device=self.device),
                torch.linspace(-1.0, 1.0, n_pc, device=self.device),
            ],
            dim=1,
        )
        # ~15 degree slope normal (upward)
        angle_deg = 15.0
        angle_rad = angle_deg * np.pi / 180.0
        slope_normal = torch.tensor(
            [np.sin(angle_rad), 0.0, np.cos(angle_rad)], dtype=torch.float32, device=self.device
        ).repeat(n_pc, 1)  # [n_pc, 3]

        pc1 = PointCloud(
            xyz_start=xyz,
            xyz_end=xyz,
            normal=slope_normal,
            sensor_type=["camera"],
            camera_footprint_scale=torch.ones(n_pc, dtype=torch.float32, device=self.device),
        )  # with normals
        pc2 = PointCloud(
            xyz_start=xyz,
            xyz_end=xyz,
            sensor_type=["camera"],
            camera_footprint_scale=torch.ones(n_pc, dtype=torch.float32, device=self.device),
        )  # without normals

        # Create ego poses corresponding to the slope normal
        # Create rotation matrix: rotate around y-axis by 15 degrees
        # so that ego's local z-axis [0,0,1] transforms to [sin(15°), 0, cos(15°)]
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotation_matrix = torch.tensor(
            [[cos_a, 0, sin_a, 0], [0, 1, 0, 0], [-sin_a, 0, cos_a, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
            device=self.device,
        )

        rig_T = rotation_matrix.unsqueeze(0).repeat(3, 1, 1)
        summary1 = _make_summary(pc_list=[pc1], rig_T_list=[rig_T], camera_ids=["cam0"])
        summary2 = _make_summary(pc_list=[pc2], rig_T_list=[rig_T], camera_ids=["cam0"])

        init_cfg = OmegaConf.create(
            {
                "project_to_z_offset": True,
                "z_offset": 0.0,
                "num_point_cloud_points": 100,
                "num_random_points": 10,
                "far_radius_factor": 10.0,
                "observation_scale_factor": 0.01,
                "default_scale": [0.1, 0.1, 0.001],
                "default_density": 0.5,
                "local_debug_dir": None,
                "scale_multiplier": 1.0,
                "lidar_ids": None,
                "camera_ids": None,
                "step_frame": 1,
                "point_cloud_device": str(self.device),
                "non_dynamic_points_only": True,
                "init_with_normals": True,
            }
        )

        model1 = LidarRigTrajRoadInitialization(
            config=init_cfg,
            gaussian_config=_make_gaussian_cfg(),
            trainer_config=_make_trainer_cfg(),
            labels_to_ignore=[],
            labels_to_use=[],
        ).to(self.device)
        model1.initialize_from_datasource(summary1)

        model2 = LidarRigTrajRoadInitialization(
            config=init_cfg,
            gaussian_config=_make_gaussian_cfg(),
            trainer_config=_make_trainer_cfg(),
            labels_to_ignore=[],
            labels_to_use=[],
        ).to(self.device)
        model2.initialize_from_datasource(summary2)

        def get_z_axis(rotations: torch.Tensor) -> torch.Tensor:
            rots = rotations[:, [1, 2, 3, 0]]  # wxyz to xyzw
            return quat_to_so3_matrix(rots)[:, :, 2]

        # All z-axis should be aligned with the slope normal
        z_axis1 = get_z_axis(model1.rotations)
        cos_similarity1 = torch.sum(z_axis1 * slope_normal, dim=-1)
        self.assertTrue(
            torch.allclose(cos_similarity1, torch.ones((n_pc,), device=self.device), atol=1e-4),  # ~0.8 degree
        )

        z_axis2 = get_z_axis(model2.rotations)
        cos_similarity2 = torch.sum(z_axis2 * slope_normal, dim=-1)
        self.assertTrue(
            torch.allclose(cos_similarity2, torch.ones((n_pc,), device=self.device), atol=1e-4),  # ~0.8 degree
        )


class TestAccumulatedPointCloudInitializationBase(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def test_load_ply_with_segment_id(self):
        class _Mesh:
            def __init__(self, positions, normals, colors, custom_attributes):
                self.vertex_data = types.SimpleNamespace(
                    positions=positions,
                    normals=normals,
                    colors=colors,
                    custom_attributes=custom_attributes,
                )

        def _fake_load_triangle_mesh(filename: str):
            N = 5
            pos = np.stack([np.arange(N), np.zeros(N), np.zeros(N)], axis=1).astype(np.float32)
            normals = np.tile(np.array([[0, 0, 1]], dtype=np.float32), (N, 1))
            colors = (np.linspace(0, 255, N)[:, None].repeat(3, axis=1)).astype(np.uint8)
            if filename.endswith("withseg.ply"):
                custom = {"segment_id": np.arange(N, dtype=np.uint8)}
            else:
                custom = {}
            return _Mesh(pos, normals, colors, custom)

        # Monkeypatch
        orig_loader = pcu.load_triangle_mesh
        pcu.load_triangle_mesh = _fake_load_triangle_mesh
        try:
            # With segment_id
            xyz, normal, color, seg = AccumulatedPointCloudInitializationBase.load_ply_with_segment_id(
                "/tmp/withseg.ply"
            )
            self.assertEqual(xyz.shape, (5, 3))
            self.assertEqual(normal.shape, (5, 3))
            self.assertEqual(color.shape, (5, 3))
            self.assertIsNotNone(seg)
            self.assertEqual(seg.shape, (5,))  # type: ignore[union-attr]
            self.assertEqual(xyz.dtype, np.float32)
            self.assertEqual(normal.dtype, np.float32)
            self.assertEqual(color.dtype, np.uint8)
            self.assertEqual(seg.dtype, np.uint8)  # type: ignore[union-attr]

            # Without segment_id
            xyz2, normal2, color2, seg2 = AccumulatedPointCloudInitializationBase.load_ply_with_segment_id(
                "/tmp/noseg.ply"
            )
            self.assertEqual(xyz2.shape, (5, 3))
            self.assertEqual(normal2.shape, (5, 3))
            self.assertEqual(color2.shape, (5, 3))
            self.assertIsNone(seg2)
        finally:
            pcu.load_triangle_mesh = orig_loader  # restore

    def test_get_point_clouds_and_ego_poses(self):
        class _Mesh:
            def __init__(self, positions, normals, colors, segment_id):
                self.vertex_data = types.SimpleNamespace(
                    positions=positions,
                    normals=normals,
                    colors=colors,
                    custom_attributes={"segment_id": segment_id} if segment_id is not None else {},
                )

        def _fake_load_triangle_mesh(filename: str):
            N = 8
            pos = np.stack([np.linspace(0, 0.7, N), np.zeros(N), np.zeros(N)], axis=1).astype(np.float32)
            normals = np.tile(np.array([[0, 0, 1]], dtype=np.float32), (N, 1))
            colors = (np.linspace(0, 255, N)[:, None].repeat(3, axis=1)).astype(np.float32)
            seg = np.arange(N, dtype=np.uint8)
            return _Mesh(pos, normals, colors, seg)

        # Monkeypatch
        orig_loader = pcu.load_triangle_mesh
        pcu.load_triangle_mesh = _fake_load_triangle_mesh
        try:
            # Minimal subclass to access base method
            class _Accum(AccumulatedPointCloudInitializationBase):
                def initialize_from_datasource(self, datasource, **kwargs):  # pragma: no cover
                    raise NotImplementedError

            init_cfg = OmegaConf.create(
                {
                    "point_cloud_path": "/tmp/fake.ply",
                    "num_point_cloud_points": 100,
                    "observation_scale_factor": 0.02,
                    "default_scale": 0.2,
                    "default_density": 0.5,
                    "local_debug_dir": None,
                    "scale_multiplier": 1.0,
                    "non_dynamic_points_only": False,
                }
            )

            model = _Accum(
                config=init_cfg,
                gaussian_config=_make_gaussian_cfg(),
                trainer_config=_make_trainer_cfg(),
                labels_to_ignore=[],
                labels_to_use=[],
            ).to(self.device)

            # Identity ego poses
            rig_T = torch.eye(4, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
            summary = _make_summary(pc_list=[], rig_T_list=[rig_T], camera_ids=["cam0"])

            pc, ego_c2nre, sparsity_comp = model.get_point_clouds_and_ego_poses(summary)

            # Shapes and basic values
            self.assertIsInstance(pc, PointCloud)
            self.assertEqual(pc.n_points, 8)
            self.assertTrue(torch.allclose(ego_c2nre, rig_T))
            self.assertEqual(sparsity_comp, 1)  # 8 < num_point_cloud_points
            # Camera footprint scale equals distance to origin (x-axis) * factor
            assert pc.camera_footprint_scale is not None
            expected_scale = (
                torch.tensor(np.linspace(0, 0.7, 8), dtype=torch.float32, device=self.device)
                * init_cfg.observation_scale_factor
            )
            self.assertTrue(torch.allclose(pc.camera_footprint_scale, expected_scale, atol=1e-6))
        finally:
            pcu.load_triangle_mesh = orig_loader  # restore


class TestDynamicTracksInitialization(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.gaussian_cfg = _make_gaussian_cfg()
        self.trainer_cfg = _make_trainer_cfg()
        self.identity_rig = torch.eye(4, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _make_dynamic_model(self, **cfg_overrides):
        base_cfg = {
            "num_point_cloud_points_per_track": 4,
            "num_point_cloud_points_in_layer": None,
            "default_scale": 0.2,
            "fill_with_random_points": False,
            "keep_all_track_poses": False,
            "symmetric_axis": None,
            "lidar_ids": None,
            "camera_ids": None,
            "step_frame": 1,
            "default_density": 0.5,
            "scale_multiplier": 1.0,
            "local_debug_dir": None,
        }
        base_cfg.update(cfg_overrides)
        init_cfg = OmegaConf.create(base_cfg)
        return DynamicTracksInitialization(
            config=init_cfg,
            gaussian_config=self.gaussian_cfg,
            trainer_config=self.trainer_cfg,
            labels_to_ignore=[],
            labels_to_use=[],
        ).to(self.device)

    def _make_track_point_cloud(self, track_id: str) -> TrackPointCloud:
        n_pts = 3
        xyz = torch.full((n_pts, 3), 0.0, dtype=torch.float32, device=self.device)
        scale = torch.full((n_pts,), 0.05, dtype=torch.float32, device=self.device)
        color = torch.full((n_pts, 3), 128, dtype=torch.uint8, device=self.device)
        pc = PointCloud(
            xyz_start=xyz,
            xyz_end=xyz,
            color=color,
            camera_footprint_scale=scale,
            sensor_type=None,
        )
        return TrackPointCloud(track_id=track_id, point_cloud=pc)

    def _make_cuboid_tracks(self, track_ids: list[str]) -> CuboidTracks:
        tracks_poses = []
        tracks_timestamps = []
        cuboid_dims = []
        label_classes = []
        flags = []
        for idx, _ in enumerate(track_ids):
            pose = np.eye(4, dtype=np.float32)
            pose[:3, 3] = np.array([idx * 0.1, 0.0, 0.0], dtype=np.float32)
            tracks_poses.append(np.stack([pose, pose], axis=0))
            tracks_timestamps.append(np.array([0, 1], dtype=np.int64))
            cuboid_dims.append(np.ones(3, dtype=np.float32) * 0.5)
            label_classes.append("vehicle")
            flags.append(TrackFlags.NONE)

        return CuboidTracks.Factory.from_numpy(
            tracks_id=track_ids,
            tracks_poses=tracks_poses,
            tracks_timestamps_us=tracks_timestamps,
            tracks_label_class=label_classes,
            tracks_flags=flags,
            cuboids_dims=cuboid_dims,
            device=self.device,
        )

    def _count_gaussians_per_track(self, model: DynamicTracksInitialization, n_tracks: int) -> torch.Tensor:
        gaussian_ids = torch.cat(model.gaussian_cuboid_ids, dim=0).squeeze(-1).cpu()
        return torch.bincount(gaussian_ids, minlength=n_tracks)

    def test_track_all_skipped(self):
        track_ids = ["0", "1"]
        cuboid_tracks = self._make_cuboid_tracks(track_ids)
        pcs = {
            "0": self._make_track_point_cloud("0"),
            "1": self._make_track_point_cloud("1"),
        }
        datasource = _StubNcore(
            camera_ids=["cam0"],
            point_clouds=[],
            track_point_clouds=pcs,
            tracks_skip={"0", "1"},
        )

        model = self._make_dynamic_model(step_frame=3)
        summary = _make_summary(
            pc_list=[],
            rig_T_list=[self.identity_rig],
            camera_ids=["cam0"],
            datasource_override=cast(BaseDataSource, datasource),
        )
        model.initialize_from_datasource(summary, cuboid_tracks=cuboid_tracks)

        counts = self._count_gaussians_per_track(model, len(track_ids))
        self.assertEqual(counts[0].item(), model.config.num_point_cloud_points_per_track)
        self.assertEqual(counts[1].item(), model.config.num_point_cloud_points_per_track)

    def test_empty_cuboid_tracks_raise_no_points_found(self):
        cuboid_tracks = self._make_cuboid_tracks([])
        datasource = _StubNcore(camera_ids=["cam0"], point_clouds=[], track_point_clouds={})
        summary = _make_summary(
            pc_list=[],
            rig_T_list=[self.identity_rig],
            camera_ids=["cam0"],
            datasource_override=cast(BaseDataSource, datasource),
        )
        model = self._make_dynamic_model()

        with self.assertRaises(NoPointsFoundException):
            model.initialize_from_datasource(summary, cuboid_tracks=cuboid_tracks)


class TestComputeVertexNormals(unittest.TestCase):
    """Tests for compute_vertex_normals."""

    def test_flat_triangle(self):
        """Single XY-plane triangle -> all normals point in +Z."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        normals = compute_vertex_normals(vertices, faces)

        assert normals.shape == (3, 3)
        for i in range(3):
            np.testing.assert_allclose(normals[i], [0, 0, 1], atol=1e-6)

    def test_shared_vertex_coplanar(self):
        """Two coplanar triangles sharing an edge -> normals still [0, 0, 1]."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
        normals = compute_vertex_normals(vertices, faces)

        assert normals.shape == (4, 3)
        for i in range(4):
            np.testing.assert_allclose(normals[i], [0, 0, 1], atol=1e-6)

    def test_unit_length(self):
        """Normals on a tilted triangle are unit length."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        normals = compute_vertex_normals(vertices, faces)

        assert normals.shape == (3, 3)
        lengths = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-6)


def _make_simple_mesh(n_verts: int = 4, with_colors: bool = True) -> tuple[NDArray, NDArray, "NDArray | None"]:
    """Return a flat quad mesh (2 triangles) in the XY plane for testing."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)[:n_verts]
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    if n_verts < 4:
        faces = faces[:1]
    colors = np.tile(np.array([[128, 64, 32]], dtype=np.uint8), (n_verts, 1)) if with_colors else None
    return vertices, faces, colors


def _make_camera_gmr_model(
    device: torch.device,
    labels_to_use: list[str] | None = None,
    **cfg_overrides,
) -> CameraGroundMeshRoadInitialization:
    base_cfg = {
        "num_random_points": 0,
        "default_density": 0.99,
        "default_scale": [0.1, 0.1, 0.001],
        "scale_multiplier": 1.0,
        "point_cloud_path": None,
        "mesh_path": None,
        "voxel_size": 0.1,
        "smoothing_passes": 1,
        "min_points_per_voxel": 1,
        "local_debug_dir": None,
    }
    base_cfg.update(cfg_overrides)
    return CameraGroundMeshRoadInitialization(
        config=OmegaConf.create(base_cfg),
        gaussian_config=_make_gaussian_cfg(),
        trainer_config=_make_trainer_cfg(),
        labels_to_ignore=[],
        labels_to_use=labels_to_use or ["road"],
    ).to(device)


def _make_lidar_gmr_model(
    device: torch.device,
    **cfg_overrides,
) -> LidarGroundMeshRoadInitialization:
    base_cfg = {
        "num_random_points": 0,
        "default_density": 0.99,
        "default_scale": [0.1, 0.1, 0.001],
        "scale_multiplier": 1.0,
        "step_frame": 1,
        "local_debug_dir": None,
    }
    base_cfg.update(cfg_overrides)
    return LidarGroundMeshRoadInitialization(
        config=OmegaConf.create(base_cfg),
        gaussian_config=_make_gaussian_cfg(),
        trainer_config=_make_trainer_cfg(),
        labels_to_ignore=[],
        labels_to_use=[],
    ).to(device)


class TestLidarGroundMeshRoadInitialization(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def test_initialize_from_datasource_consumes_broadcast_state(self):
        verts, _, colors = _make_simple_mesh(with_colors=True)
        normals = np.tile(np.array([[0, 0, 1]], dtype=np.float32), (len(verts), 1))
        rig_T = torch.eye(4, device=self.device, dtype=torch.float32).unsqueeze(0)
        summary = _make_summary(pc_list=[], rig_T_list=[rig_T], camera_ids=["cam0"])
        model = _make_lidar_gmr_model(self.device)

        state: dict[str, object] = {}

        def _fake_sync_objects_or_raise(func, master_rank=0):
            state["master_rank"] = master_rank
            return func()

        def _fake_build_ground_mesh_init_state(ncore_ds):
            state["datasource"] = ncore_ds
            return verts, normals, colors

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(initializations_module, "sync_objects_or_raise", _fake_sync_objects_or_raise)
            monkeypatch.setattr(model, "_build_ground_mesh_init_state", _fake_build_ground_mesh_init_state)
            model.initialize_from_datasource(summary)

        expected_pos = torch.from_numpy(verts).float().to(self.device)
        self.assertTrue(torch.allclose(model.positions, expected_pos, atol=1e-6))

        identity_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        for rotation in model.rotations:
            self.assertTrue(torch.allclose(rotation, identity_rot, atol=1e-5))

        self.assertEqual(state["master_rank"], 0)
        self.assertIs(state["datasource"], summary.datasource)
        self.assertTrue((model.features_albedo != 0).any())
        self.assertTrue(torch.allclose(model.features_albedo[0], model.features_albedo[1], atol=1e-6))

    def test_initialize_routes_mesh_build_through_sync_objects(self):
        verts, _, _ = _make_simple_mesh(with_colors=False)
        normals = np.tile(np.array([[0, 0, 1]], dtype=np.float32), (len(verts), 1))
        rig_T = torch.eye(4, device=self.device, dtype=torch.float32).unsqueeze(0)
        summary = _make_summary(pc_list=[], rig_T_list=[rig_T], camera_ids=["cam0"])
        model = _make_lidar_gmr_model(self.device)

        state = {"sync_calls": 0, "build_calls": 0, "in_sync": False}

        def _fake_sync_objects_or_raise(func, master_rank=0):
            self.assertEqual(master_rank, 0)
            state["sync_calls"] += 1
            state["in_sync"] = True
            try:
                return func()
            finally:
                state["in_sync"] = False

        def _fake_build_ground_mesh_init_state(_ncore_ds):
            self.assertTrue(state["in_sync"])
            state["build_calls"] += 1
            return verts, normals, None

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(initializations_module, "sync_objects_or_raise", _fake_sync_objects_or_raise)
            monkeypatch.setattr(model, "_build_ground_mesh_init_state", _fake_build_ground_mesh_init_state)
            model.initialize_from_datasource(summary)

        self.assertEqual(state["sync_calls"], 1)
        self.assertEqual(state["build_calls"], 1)
        self.assertEqual(model.positions.shape[0], len(verts))

    def test_build_ground_mesh_init_state_coerces_payload_dtypes(self):
        verts, faces, colors = _make_simple_mesh(with_colors=True)
        ncore_ds = _StubNcore(camera_ids=["cam0"], point_clouds=[])
        model = _make_lidar_gmr_model(self.device)

        import nre.utils.io.ground_mesh as ground_mesh_module

        orig_gcp = ground_mesh_module.get_nominal_ground_point_under_lidar
        orig_reconstruct = ground_mesh_module.reconstruct_ground_mesh_from_points
        try:
            ground_mesh_module.get_nominal_ground_point_under_lidar = lambda _ds: np.zeros(3, dtype=np.float32)

            def _fake_reconstruct_ground_mesh_from_points(**_kwargs):
                return (
                    verts.astype(np.float64),
                    faces,
                    None,
                    None,
                    None,
                    None,
                    None,
                    colors.astype(np.uint32),
                )

            ground_mesh_module.reconstruct_ground_mesh_from_points = _fake_reconstruct_ground_mesh_from_points

            positions_nre, normals_nre, vertex_colors = model._build_ground_mesh_init_state(ncore_ds)
        finally:
            ground_mesh_module.get_nominal_ground_point_under_lidar = orig_gcp
            ground_mesh_module.reconstruct_ground_mesh_from_points = orig_reconstruct

        self.assertEqual(positions_nre.dtype, np.float32)
        self.assertEqual(positions_nre.ndim, 2)
        self.assertEqual(normals_nre.dtype, np.float32)
        self.assertEqual(normals_nre.shape, positions_nre.shape)
        self.assertIsNotNone(vertex_colors)
        self.assertEqual(cast(NDArray, vertex_colors).dtype, np.uint8)

    def test_build_ground_mesh_init_state_preserves_none_vertex_colors(self):
        verts, faces, _ = _make_simple_mesh(with_colors=False)
        ncore_ds = _StubNcore(camera_ids=["cam0"], point_clouds=[])
        model = _make_lidar_gmr_model(self.device)

        import nre.utils.io.ground_mesh as ground_mesh_module

        orig_gcp = ground_mesh_module.get_nominal_ground_point_under_lidar
        orig_reconstruct = ground_mesh_module.reconstruct_ground_mesh_from_points
        try:
            ground_mesh_module.get_nominal_ground_point_under_lidar = lambda _ds: np.zeros(3, dtype=np.float32)

            def _fake_reconstruct_ground_mesh_from_points(**_kwargs):
                return verts.astype(np.float64), faces, None, None, None, None, None, None

            ground_mesh_module.reconstruct_ground_mesh_from_points = _fake_reconstruct_ground_mesh_from_points

            positions_nre, normals_nre, vertex_colors = model._build_ground_mesh_init_state(ncore_ds)
        finally:
            ground_mesh_module.get_nominal_ground_point_under_lidar = orig_gcp
            ground_mesh_module.reconstruct_ground_mesh_from_points = orig_reconstruct

        self.assertEqual(positions_nre.dtype, np.float32)
        self.assertEqual(normals_nre.dtype, np.float32)
        self.assertIsNone(vertex_colors)


class TestCameraGroundMeshRoadInitialization(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # -- _load_mesh_from_file ---------------------------------------------------

    def test_load_mesh_from_file_with_colors(self):
        verts, faces, colors = _make_simple_mesh(with_colors=True)

        class _MeshData:
            vertex_data = types.SimpleNamespace(
                positions=verts, colors=np.c_[colors, np.zeros((len(verts), 1), dtype=np.uint8)]
            )
            face_data = types.SimpleNamespace(vertex_ids=faces)

        orig = pcu.load_triangle_mesh
        pcu.load_triangle_mesh = lambda _path: _MeshData()
        try:
            model = _make_camera_gmr_model(self.device)
            v, f, c = model._load_mesh_from_file("/fake/mesh.ply")
            np.testing.assert_array_equal(v, verts)
            np.testing.assert_array_equal(f, faces)
            assert c is not None
            assert colors is not None
            np.testing.assert_array_equal(c, colors)
        finally:
            pcu.load_triangle_mesh = orig

    def test_load_mesh_from_file_without_colors(self):
        verts, faces, _ = _make_simple_mesh(with_colors=False)

        class _MeshData:
            vertex_data = types.SimpleNamespace(positions=verts, colors=None)
            face_data = types.SimpleNamespace(vertex_ids=faces)

        orig = pcu.load_triangle_mesh
        pcu.load_triangle_mesh = lambda _path: _MeshData()
        try:
            model = _make_camera_gmr_model(self.device)
            v, f, c = model._load_mesh_from_file("/fake/mesh.ply")
            np.testing.assert_array_equal(v, verts)
            np.testing.assert_array_equal(f, faces)
            self.assertIsNone(c)
        finally:
            pcu.load_triangle_mesh = orig

    # -- _build_mesh_from_fused_point_cloud ------------------------------------

    def test_build_mesh_filters_road_by_labels_to_use(self):
        np.random.seed(42)
        N = 10
        xyz = np.random.randn(N, 3).astype(np.float32)
        colors = np.arange(N * 4, dtype=np.uint8).reshape(N, 4)
        seg = np.array([1, 1, 2, 1, 3, 2, 1, 3, 2, 2], dtype=np.int32)
        road_mask = seg == 1  # road=1 in _StubNcore
        expected_road_xyz = xyz[road_mask]
        expected_road_colors = colors[road_mask][:, :3]

        class _MeshData:
            vertex_data = types.SimpleNamespace(
                positions=xyz,
                colors=colors,
                custom_attributes={"segment_id": seg},
            )

        orig = pcu.load_triangle_mesh
        pcu.load_triangle_mesh = lambda _path: _MeshData()

        captured_args: dict = {}

        def _fake_build_mesh(self_meshing, points, colors=None):  # type: ignore[override]
            captured_args["points"] = points
            captured_args["colors"] = colors
            verts, faces, clrs = _make_simple_mesh(with_colors=colors is not None)
            normals = np.zeros_like(verts)
            return verts, faces, normals, clrs

        import nre.utils.io.ground_mesh as gm_mod

        orig_build = getattr(gm_mod.DelaunayElevationMeshingAlgorithm, "build_mesh_from_points", None)
        gm_mod.DelaunayElevationMeshingAlgorithm.build_mesh_from_points = _fake_build_mesh  # type: ignore[assignment]
        try:
            ncore_ds = _StubNcore(camera_ids=["cam0"], point_clouds=[])
            model = _make_camera_gmr_model(
                self.device,
                labels_to_use=["road"],
                point_cloud_path="/fake/fused.ply",
            )
            model._build_mesh_from_fused_point_cloud(ncore_ds)

            # Verify the correct road points were passed to the meshing algorithm
            np.testing.assert_array_equal(
                captured_args["points"],
                expected_road_xyz,
            )
            np.testing.assert_array_equal(
                captured_args["colors"],
                expected_road_colors,
            )
        finally:
            pcu.load_triangle_mesh = orig
            if orig_build is not None:
                gm_mod.DelaunayElevationMeshingAlgorithm.build_mesh_from_points = orig_build

    def test_build_mesh_asserts_on_empty_labels_to_use(self):
        N = 5
        xyz = np.zeros((N, 3), dtype=np.float32)
        seg = np.ones(N, dtype=np.int32)

        class _MeshData:
            vertex_data = types.SimpleNamespace(
                positions=xyz,
                colors=None,
                custom_attributes={"segment_id": seg},
            )

        orig = pcu.load_triangle_mesh
        pcu.load_triangle_mesh = lambda _path: _MeshData()
        try:
            ncore_ds = _StubNcore(camera_ids=["cam0"], point_clouds=[])
            model = _make_camera_gmr_model(
                self.device,
                labels_to_use=[],
                point_cloud_path="/fake/fused.ply",
            )
            model.labels_to_use = []
            with self.assertRaises(AssertionError):
                model._build_mesh_from_fused_point_cloud(ncore_ds)
        finally:
            pcu.load_triangle_mesh = orig

    def test_build_mesh_too_few_road_points_raises(self):
        N = 5
        xyz = np.zeros((N, 3), dtype=np.float32)
        seg = np.array([2, 2, 2, 2, 1], dtype=np.int32)

        class _MeshData:
            vertex_data = types.SimpleNamespace(
                positions=xyz,
                colors=None,
                custom_attributes={"segment_id": seg},
            )

        orig = pcu.load_triangle_mesh
        pcu.load_triangle_mesh = lambda _path: _MeshData()
        try:
            ncore_ds = _StubNcore(camera_ids=["cam0"], point_clouds=[])
            model = _make_camera_gmr_model(
                self.device,
                labels_to_use=["road"],
                point_cloud_path="/fake/fused.ply",
            )
            with self.assertRaises(ValueError):
                model._build_mesh_from_fused_point_cloud(ncore_ds)
        finally:
            pcu.load_triangle_mesh = orig

    # -- initialize_from_datasource --------------------------------------------

    def test_initialize_from_datasource_with_mesh_path(self):
        verts, faces, colors = _make_simple_mesh(with_colors=True)
        n_verts = len(verts)

        class _MeshData:
            vertex_data = types.SimpleNamespace(
                positions=verts, colors=np.c_[colors, np.zeros((n_verts, 1), dtype=np.uint8)]
            )
            face_data = types.SimpleNamespace(vertex_ids=faces)

        orig = pcu.load_triangle_mesh
        pcu.load_triangle_mesh = lambda _path: _MeshData()
        try:
            rig_T = torch.eye(4, device=self.device, dtype=torch.float32).unsqueeze(0)
            summary = _make_summary(pc_list=[], rig_T_list=[rig_T], camera_ids=["cam0"])

            model = _make_camera_gmr_model(self.device, mesh_path="/fake/mesh.ply")
            model.initialize_from_datasource(summary)

            # Positions should equal input vertices (identity world_to_nre)
            expected_pos = torch.from_numpy(verts).float().to(self.device)
            self.assertTrue(
                torch.allclose(model.positions, expected_pos, atol=1e-6),
                "Positions should match input mesh vertices under identity transform",
            )

            # Flat XY mesh → normals are [0,0,1] → rotations should be identity (wxyz)
            identity_rot = torch.tensor([1.0, 0, 0, 0], device=self.device)
            for i in range(n_verts):
                self.assertTrue(
                    torch.allclose(model.rotations[i], identity_rot, atol=1e-5),
                    "Flat XY mesh should produce identity rotations",
                )

            # Colors provided → features_albedo should come from RGB2SH, not random
            self.assertTrue(
                (model.features_albedo != 0).any(),
                "features_albedo should be non-zero when vertex colors are provided",
            )
            # All vertices have the same color → all albedo rows should be identical
            self.assertTrue(
                torch.allclose(model.features_albedo[0], model.features_albedo[1], atol=1e-6),
                "Identical vertex colors should produce identical SH features",
            )
        finally:
            pcu.load_triangle_mesh = orig

    def test_initialize_from_datasource_with_point_cloud_path(self):
        verts, faces, colors = _make_simple_mesh(with_colors=True)

        model = _make_camera_gmr_model(
            self.device,
            labels_to_use=["road"],
            point_cloud_path="/fake/fused.ply",
        )

        model._build_mesh_from_fused_point_cloud = lambda ncore_ds: (verts, faces, colors)

        rig_T = torch.eye(4, device=self.device, dtype=torch.float32).unsqueeze(0)
        summary = _make_summary(pc_list=[], rig_T_list=[rig_T], camera_ids=["cam0"])

        model.initialize_from_datasource(summary)

        # Same pipeline as mesh_path; verify positions match and colors flow through
        expected_pos = torch.from_numpy(verts).float().to(self.device)
        self.assertTrue(
            torch.allclose(model.positions, expected_pos, atol=1e-6),
            "point_cloud_path should produce same positions as mesh_path for identical geometry",
        )
        self.assertTrue(
            (model.features_albedo != 0).any(),
            "features_albedo should be non-zero when vertex colors are provided",
        )

    def test_initialize_rejects_random_points(self):
        rig_T = torch.eye(4, device=self.device, dtype=torch.float32).unsqueeze(0)
        summary = _make_summary(pc_list=[], rig_T_list=[rig_T], camera_ids=["cam0"])

        model = _make_camera_gmr_model(self.device, num_random_points=5)
        with self.assertRaises(AssertionError):
            model.initialize_from_datasource(summary)

    def test_initialize_rejects_degenerate_mesh(self):
        """Mesh with <3 vertices or 0 faces should raise ValueError early."""
        degenerate_verts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        degenerate_faces = np.zeros((0, 3), dtype=np.int32)

        class _MeshData:
            vertex_data = types.SimpleNamespace(positions=degenerate_verts, colors=None)
            face_data = types.SimpleNamespace(vertex_ids=degenerate_faces)

        orig = pcu.load_triangle_mesh
        pcu.load_triangle_mesh = lambda _path: _MeshData()
        try:
            rig_T = torch.eye(4, device=self.device, dtype=torch.float32).unsqueeze(0)
            summary = _make_summary(pc_list=[], rig_T_list=[rig_T], camera_ids=["cam0"])

            model = _make_camera_gmr_model(self.device, mesh_path="/fake/bad.ply")
            with self.assertRaises(ValueError):
                model.initialize_from_datasource(summary)
        finally:
            pcu.load_triangle_mesh = orig


if __name__ == "__main__":
    unittest.main()
