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
import math

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Type, cast

import lietorch as lt
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_scatter

from matplotlib import collections as mc
from omegaconf import DictConfig
from torch import nn

from nre.config.trainer import TrainerConfig
from nre.datasets.tracks import CuboidTracks
from nre.models.base import BaseModel
from nre.models.nn_extensions import module_call_type
from nre.utils.geometry import so3_matrix_to_quat
from nre.utils.optim import OptimizerLRSchedulerConfig, configure_optimizers
from nre.utils.packed_ops import packed_max, packed_min
from nre.utils.trainer import adjust_step_for_world_size
def get_visualdebugger():
    class _Null:
        def __getattr__(self, _n):
            return lambda *a, **k: None
    return _Null()


log = logging.getLogger(__name__)


class BaseTracksCalib(BaseModel):
    @staticmethod
    def factory(
        name: str, config: DictConfig, trainer_config: TrainerConfig, cuboid_tracks: CuboidTracks
    ) -> BaseTracksCalib:
        variants: dict[str, Type[BaseTracksCalib]] = {
            "direct-tracks-calib": DirectTracksCalib,
            "static-tracks-calib": StaticTracksCalib,
            "unicycle-tracks-calib": UnicycleTracksCalib,
            "skip-tracks-calib": SkipTracksCalib,
        }
        variant_class = variants[name]
        return variant_class(config, trainer_config, cuboid_tracks)

    @dataclass(slots=False, kw_only=True)
    class CuboidTracksVisualizeData:
        """
        Data stored for visualizing cuboid tracks calibration.
        """

        init: Optional[CuboidTracks] = None
        current: Optional[CuboidTracks] = None

    def __init__(self, config: DictConfig, trainer_config: TrainerConfig, cuboid_tracks: CuboidTracks):
        super().__init__(config)
        self.trainer_config = trainer_config

        self.visualize: bool = self.config.visualize
        # Volatile cache the cuboid tracks just for visualization
        self.vis_data = BaseTracksCalib.CuboidTracksVisualizeData()

        if hasattr(self.config, "start_global_step"):
            self.config.start_global_step = adjust_step_for_world_size(trainer_config, self.config.start_global_step)
            log.info(f"BaseTracksCalib: start_global_step={self.config.start_global_step}")

    def save_tracks_for_visualize(self, tracks: CuboidTracks) -> None:
        if self.visualize:
            self.vis_data.current = tracks

            if self.vis_data.init is None:
                self.vis_data.init = CuboidTracks.Ops.clone(tracks)

    @abstractmethod
    def forward(
        self,
        tracks: CuboidTracks,
    ) -> CuboidTracks: ...

    __call__ = module_call_type(forward)


class DirectTracksCalib(BaseTracksCalib):
    """Incorporates calibration parameters (6-DoF object delta poses) directly in the outputs"""

    gradient_mask: torch.Tensor

    def __init__(self, config: DictConfig, trainer_config: TrainerConfig, cuboid_tracks: CuboidTracks):
        super().__init__(config, trainer_config, cuboid_tracks)

        # Called during class initialization
        assert self.config.start_global_step >= 0, (
            f"{self.__class__.__name__}: require positive global start step or zero"
        )

        n_tracks_poses: int = cuboid_tracks.tracks_poses.shape[0]

        self.tracks_delta_q = nn.Parameter(
            torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device, dtype=torch.float32).repeat(n_tracks_poses, 1)
        )
        self.tracks_delta_t = nn.Parameter(torch.zeros(n_tracks_poses, 3, device=self.device, dtype=torch.float32))

        self.gradient_mask = nn.Buffer(
            torch.ones(n_tracks_poses, device=self.device, dtype=torch.bool), persistent=False
        )
        for pose_start, pose_count in cuboid_tracks.tracks_packinfo.cpu().numpy():
            if self.config.fix_first_pose:
                self.gradient_mask[pose_start] = False
            if self.config.fix_last_pose:
                self.gradient_mask[pose_start + pose_count - 1] = False

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        if global_step >= self.config.start_global_step:
            # unfreeze parameters if estimated
            self.tracks_delta_q.requires_grad_(True)
            self.tracks_delta_t.requires_grad_(True)
        else:
            self.tracks_delta_q.requires_grad_(False)
            self.tracks_delta_t.requires_grad_(False)

        with torch.no_grad():
            if getattr(system, "logger_enabled", True):
                # Logging
                LOG_TO_PROG_BAR = self.config.log_to_prog_bar

                delta_pose = self.get_tracks_delta_transform()
                delta_pose_vec = delta_pose.vec()
                avg_delta_t = delta_pose_vec[:, :3].mean(dim=0).norm()
                avg_delta_angle = 2 * torch.acos(delta_pose_vec[:, 6:].mean(dim=0).norm()) * 180 / np.pi

                system.log_dict(
                    {
                        "tracks_calib/delta_t": avg_delta_t,
                        "tracks_calib/delta_angle": avg_delta_angle,
                    },
                    prog_bar=LOG_TO_PROG_BAR,
                )

            # Visualize the optimized tracks
            if self.visualize and self.vis_data.current is not None:
                optimized_tracks = self.forward(self.vis_data.current)
                optimized_tracks.visualize(1, show=False)
                get_visualdebugger().update()

        return {}  # no new / additional parameters

    def get_tracks_delta_transform(self) -> lt.SE3:
        if self.gradient_mask.any():
            current_tracks_delta_q = torch.where(
                self.gradient_mask.view(-1, 1), self.tracks_delta_q, self.tracks_delta_q.detach()
            )
            current_tracks_delta_t = torch.where(
                self.gradient_mask.view(-1, 1), self.tracks_delta_t, self.tracks_delta_t.detach()
            )
        else:
            current_tracks_delta_q = self.tracks_delta_q
            current_tracks_delta_t = self.tracks_delta_t

        return lt.SE3.InitFromVec(
            torch.cat(
                [current_tracks_delta_t, current_tracks_delta_q / current_tracks_delta_q.norm(dim=1, keepdim=True)],
                dim=1,
            )
        )

    def forward(
        self,
        tracks: CuboidTracks,
    ) -> CuboidTracks:
        self.save_tracks_for_visualize(tracks)

        # Transform with optimized delta poses
        current_tracks_delta_transform = self.get_tracks_delta_transform()
        tracks = CuboidTracks.Ops.transform_with_delta_poses(tracks, current_tracks_delta_transform, left_multiply=True)

        return tracks

    __call__ = module_call_type(forward)

    def configure_optimizers(self, name_prefix: str = "") -> list[OptimizerLRSchedulerConfig]:
        return configure_optimizers(self.config, self.trainer_config, self, name_prefix)


class StaticTracksCalib(BaseTracksCalib):
    """
    Replace the object poses with a mean pose that is obtained during initialization
    Since it makes no sense to optimize static poses via rendering loss, the model does not contain optimizable parameters
    """

    static_poses: lt.SE3
    min_timestamp_us: int
    max_timestamp_us: int

    TIMESTAMP_EXTEND_US: int = 1_000_000_000  # 1000s of extension

    def __init__(self, config: DictConfig, trainer_config: TrainerConfig, cuboid_tracks: CuboidTracks):
        super().__init__(config, trainer_config, cuboid_tracks)

        pose_starts, pose_counts = cuboid_tracks.tracks_packinfo[:, 0], cuboid_tracks.tracks_packinfo[:, 1]
        n_tracks = len(cuboid_tracks.tracks_id)
        pose_track_idx = torch.arange(n_tracks, device=cuboid_tracks.device).repeat_interleave(pose_counts)

        # Compute the mean pose for each track.
        # This is an iterative algorithm taken from https://ethaneade.com/lie.pdf.
        # TODO[JH]: consider exposing the SE3-sample mean computation as a utility function (potentially with extensions to other moments)
        pose_rotation = lt.SO3.InitFromVec(cuboid_tracks.tracks_poses.vec()[:, 3:])
        pose_translation = cuboid_tracks.tracks_poses.vec()[:, :3]

        # Use the first pose as the initial value
        mu_q = pose_rotation[pose_starts]
        for _ in range(4):
            v = (mu_q.inv()[pose_track_idx] * pose_rotation).log()
            mu_q = mu_q * lt.SO3.exp(torch_scatter.scatter_mean(v, pose_track_idx, dim=0, dim_size=n_tracks))

        mu_t = torch_scatter.scatter_mean(pose_translation, pose_track_idx, dim=0, dim_size=n_tracks)

        self.static_poses = lt.SE3(torch.cat([mu_t, mu_q.vec()], dim=1))

        self.min_timestamp_us = cast(int, cuboid_tracks.tracks_timestamps_us.min().item()) - self.TIMESTAMP_EXTEND_US
        self.max_timestamp_us = cast(int, cuboid_tracks.tracks_timestamps_us.max().item()) + self.TIMESTAMP_EXTEND_US

    def forward(
        self,
        tracks: CuboidTracks,
    ) -> CuboidTracks:
        assert self.static_poses.shape[0] == tracks.n_tracks, "Mismatch in number of tracks"
        return CuboidTracks.Ops.freeze(
            tracks,
            torch.full((tracks.n_tracks,), self.min_timestamp_us, device=tracks.device, dtype=torch.int64),
            torch.full((tracks.n_tracks,), self.max_timestamp_us, device=tracks.device, dtype=torch.int64),
            self.static_poses,
        )


class UnicycleTracksCalib(BaseTracksCalib):
    """
    Approximate the car motion via a unicycle on the ground
    Taken from: HUGS: Holistic Urban 3D Scene Understanding via Gaussian Splatting

    Ground plane/cuboid-local coordinates: x-right, y-forward, z-up
    Possible cuboid coordinates: x-forward, y-left, z-up (user can specify)
    T_world_cuboid = T_world_ground * T_ground_cuboid_local * T_cuboid_local_cuboid
    """

    xy: nn.Parameter  # (n_track_poses, 2)                - xy position
    theta: nn.Parameter  # (n_track_poses, )                 - orientation in radians
    velocity: nn.Parameter  # (n_track_poses - n_tracks, )      - velocity
    omega: nn.Parameter  # (n_track_poses - n_tracks, )      - angular velocity
    time_delta_s: torch.Tensor  # (n_track_poses - n_tracks, )      - time delta between poses
    time_delta_v_s: torch.Tensor  # (n_track_poses - n_tracks, )    - time delta between velocities

    # Note that we might want to penalize rotation around the local z-axis.
    ground_q: nn.Buffer  # (n_tracks, 4)                     - ground plane orientation (z-up, y-forward, x-right)
    ground_t: nn.Buffer  # (n_tracks, 3)                     - ground plane center

    off_one_x_mapping: torch.Tensor  # (n_track_poses - n_tracks, ) - points to the index in xy and theta
    off_two_v_mapping: torch.Tensor  # (n_track_poses - n_tracks * 3, ) - points to the index in velocity and omega
    tracks_idx: torch.Tensor  # (n_track_poses, )                 - index of the corresponding tracks index

    def __init__(self, config: DictConfig, trainer_config: TrainerConfig, cuboid_tracks: CuboidTracks):
        super().__init__(config, trainer_config, cuboid_tracks)

        assert self.config.start_global_step >= -1, (
            f"{self.__class__.__name__}: require global start step >=0 or -1 (that never starts)"
        )

        self.initial_xy: Optional[torch.Tensor] = None

        # T_cuboid_local_cuboid transforms from x-right (cuboid-local) to x-forward (cuboid)
        cuboid_x = np.asarray(self.config.cuboid_tracks.forward_axis)
        cuboid_z = np.asarray(self.config.cuboid_tracks.up_axis)
        cuboid_y = np.cross(cuboid_z, cuboid_x)
        cuboid_q = so3_matrix_to_quat(np.stack([cuboid_x, cuboid_y, cuboid_z], axis=1))

        self.Tclc = lt.SE3.InitFromVec(
            torch.from_numpy(np.concatenate([np.zeros(3), cuboid_q])).float().to(self.device).unsqueeze(0)
        ).inv()

        tracks_xys: list[torch.Tensor] = []
        tracks_thetas: list[torch.Tensor] = []
        tracks_velocities: list[torch.Tensor] = []
        tracks_omegas: list[torch.Tensor] = []
        tracks_time_deltas: list[torch.Tensor] = []
        tracks_time_delta_vs: list[torch.Tensor] = []

        tracks_ground_qs: list[torch.Tensor] = []
        tracks_ground_ts: list[torch.Tensor] = []
        tracks_idx: list[torch.Tensor] = []
        tracks_off_one_x_mapping: list[torch.Tensor] = []
        tracks_off_two_v_mapping: list[torch.Tensor] = []

        cuboid_local_up_dir = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device)
        cuboid_local_front_dir = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        current_x_idx: int = 0
        current_v_idx: int = 0

        for pose_start, pose_count in cuboid_tracks.tracks_packinfo.cpu().numpy():
            assert pose_count > 1, "Need at least 2 poses to enable unicycle model"
            all_poses = cuboid_tracks.tracks_poses[pose_start : pose_start + pose_count]
            all_timestamps_us = cuboid_tracks.tracks_timestamps_us[pose_start : pose_start + pose_count]

            time_delta_s = (all_timestamps_us[1:] - all_timestamps_us[:-1]).float() / 1e6

            # all_poses is now T_world_cuboid * T_cuboid_local_cuboid^(-1)
            all_poses = all_poses * self.Tclc.inv()

            positions = all_poses.translation()[:, :3]
            up_dirs = all_poses.act(cuboid_local_up_dir.view(1, -1))[:, :3]
            front_dirs = all_poses.act(cuboid_local_front_dir.view(1, -1))[:, :3]

            # Determine ground plane local coordinates
            ground_z = torch.mean(up_dirs, dim=0)
            ground_z = ground_z / ground_z.norm()

            ground_center = torch.mean(positions, dim=0)
            ground_x = front_dirs[0]

            ground_y = torch.cross(ground_z, ground_x)
            ground_y = ground_y / ground_y.norm()

            ground_x = torch.cross(ground_y, ground_z)

            ground_rotation_matrix = torch.stack([ground_x, ground_y, ground_z], dim=1)
            tracks_ground_q = so3_matrix_to_quat(ground_rotation_matrix)

            # Transform translations and front_dirs to local coordinates
            #   by first obtaining T_ground_cuboid_local, and then projecting it to 2D.
            ground_inv_t = -ground_rotation_matrix.t().mv(ground_center)

            tracks_xy = (positions.mm(ground_rotation_matrix) + ground_inv_t)[:, :2]
            local_front_dirs = front_dirs.mm(ground_rotation_matrix)[:, :2]
            local_front_dirs = local_front_dirs / local_front_dirs.norm(dim=1, keepdim=True)
            # (note that theta is relative to x-axis, not front-dirs, i.e. y)
            tracks_theta = torch.atan2(local_front_dirs[:, 1], local_front_dirs[:, 0])
            tracks_omega = (tracks_theta[1:] - tracks_theta[:-1]) / time_delta_s

            # Approximating initial velocity
            tracks_velocity = torch.norm(tracks_xy[1:] - tracks_xy[:-1], dim=1) / time_delta_s
            tracks_time_delta_v = (time_delta_s[1:] + time_delta_s[:-1]) / 2
            tracks_time_delta_v = torch.nn.functional.pad(tracks_time_delta_v, (0, 1), "constant", 0)

            tracks_xys.append(tracks_xy)
            tracks_thetas.append(tracks_theta)
            tracks_velocities.append(tracks_velocity)
            tracks_omegas.append(tracks_omega)
            tracks_time_deltas.append(time_delta_s)
            tracks_time_delta_vs.append(tracks_time_delta_v)

            tracks_ground_qs.append(tracks_ground_q.view(1, -1))
            tracks_ground_ts.append(ground_center.view(1, -1))

            tracks_idx.append(torch.full((pose_count,), len(tracks_idx), device=self.device).long())

            if pose_count > 1:
                tracks_off_one_x_mapping.append(torch.arange(pose_count - 1, device=self.device).long() + current_x_idx)
            if pose_count > 2:
                tracks_off_two_v_mapping.append(torch.arange(pose_count - 3, device=self.device).long() + current_v_idx)

            assert current_x_idx == pose_start, "Mismatch in pose start"
            current_x_idx += pose_count
            current_v_idx += pose_count - 1

        self.xy = nn.Parameter(torch.cat(tracks_xys, dim=0))
        self.theta = nn.Parameter(torch.cat(tracks_thetas, dim=0))
        self.velocity = nn.Parameter(torch.cat(tracks_velocities, dim=0))
        self.omega = nn.Parameter(torch.cat(tracks_omegas, dim=0))
        self.time_delta_s = torch.cat(tracks_time_deltas, dim=0)
        self.time_delta_v_s = torch.cat(tracks_time_delta_vs, dim=0)
        self.ground_q = nn.Buffer(torch.cat(tracks_ground_qs, dim=0))
        self.ground_t = nn.Buffer(torch.cat(tracks_ground_ts, dim=0))
        self.tracks_idx = torch.cat(tracks_idx, dim=0)
        self.off_one_x_mapping = torch.cat(tracks_off_one_x_mapping, dim=0)
        self.off_two_v_mapping = torch.cat(tracks_off_two_v_mapping, dim=0)

    def save_tracks_for_visualize(self, tracks: CuboidTracks) -> None:
        super().save_tracks_for_visualize(tracks)
        if self.visualize:
            if self.initial_xy is None:
                self.initial_xy = self.xy.detach().clone()

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        if global_step >= self.config.start_global_step >= 0:
            # unfreeze parameters if estimated
            self.xy.requires_grad_(True)
            self.theta.requires_grad_(True)
            self.velocity.requires_grad_(True)
            self.omega.requires_grad_(True)
        else:
            self.xy.requires_grad_(False)
            self.theta.requires_grad_(False)
            self.velocity.requires_grad_(False)
            self.omega.requires_grad_(False)

        with torch.no_grad():
            if self.vis_data.current is not None:
                optimized_tracks = self.forward(self.vis_data.current)
                optimized_tracks.visualize(1, show=False)
                self.visualize_unicycle()
                get_visualdebugger().update()

        return {}  # no new / additional parameters

    def visualize_unicycle(self):
        num_tracks = min(self.ground_q.size(0), 16)

        if not hasattr(self, "figure"):
            plt.ion()
            num_rows = math.ceil(math.sqrt(num_tracks))
            num_cols = math.ceil(num_tracks / num_rows)
            figure, axes = plt.subplots(nrows=num_rows, ncols=num_cols)
            self.figure = figure
            self.all_quiver_arrows = []
            self.all_scatter_points = []
            self.all_connections = []
            redraw = False

        else:
            redraw = True

        xy, theta, tracks_idx = (
            self.xy.detach().cpu().numpy(),
            self.theta.detach().cpu().numpy(),
            self.tracks_idx.detach().cpu().numpy(),
        )
        velocity = self.velocity.detach().cpu().numpy()
        time_delta_s = self.time_delta_s.cpu().numpy()
        initial_xy = self.initial_xy.cpu().numpy()

        v_idx = 0
        for track_idx in range(num_tracks):
            track_xy = xy[tracks_idx == track_idx]
            initial_track_xy = initial_xy[tracks_idx == track_idx]
            angle = theta[tracks_idx == track_idx]
            v = velocity[v_idx : v_idx + len(track_xy) - 1]
            dt = time_delta_s[v_idx : v_idx + len(track_xy) - 1]

            if redraw:
                self.all_scatter_points[track_idx].set_offsets(track_xy)
                self.all_quiver_arrows[track_idx].set_offsets(track_xy[:-1])
                self.all_quiver_arrows[track_idx].set_UVC(v * np.cos(angle[:-1]) * dt, v * np.sin(angle[:-1]) * dt)
                self.all_connections[track_idx].set_paths(np.stack([initial_track_xy, track_xy], axis=1))

            else:
                ax = axes.ravel()[track_idx]
                ax.plot(initial_track_xy[:, 0], initial_track_xy[:, 1], linestyle="--", color="gray")
                conn = mc.LineCollection(
                    np.stack([initial_track_xy, track_xy], axis=1), color="red", linestyle="-", linewidth=0.2
                )
                scatter_points = ax.scatter(track_xy[:, 0], track_xy[:, 1])
                quiver_arrows = ax.quiver(
                    track_xy[:-1, 0],
                    track_xy[:-1, 1],
                    v * np.cos(angle[:-1]) * dt,
                    v * np.sin(angle[:-1]) * dt,
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                )
                ax.add_collection(conn)
                self.all_scatter_points.append(scatter_points)
                self.all_quiver_arrows.append(quiver_arrows)
                self.all_connections.append(conn)
                ax.set_title(f"Track {track_idx}")
            v_idx += len(v)

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def _get_xy_theta(self) -> tuple[torch.Tensor, torch.Tensor]:
        gradient_mask = torch.ones(self.xy.size(0), device=self.device, dtype=torch.bool)

        if self.config.fix_first_pose:
            mask = torch.zeros(self.xy.size(0), device=self.device, dtype=torch.bool)
            mask[self.off_one_x_mapping] = True
            gradient_mask = gradient_mask & mask

        if self.config.fix_last_pose:
            mask = torch.zeros(self.xy.size(0), device=self.device, dtype=torch.bool)
            mask[self.off_one_x_mapping + 1] = True
            gradient_mask = gradient_mask & mask

        if gradient_mask.any():
            current_xy = torch.where(gradient_mask.view(-1, 1), self.xy, self.xy.detach())
            current_theta = torch.where(gradient_mask, self.theta, self.theta.detach())

        else:
            current_xy = self.xy
            current_theta = self.theta

        return current_xy, current_theta

    def get_pose(self) -> lt.SE3:
        T_world_ground = lt.SE3.InitFromVec(
            torch.cat([self.ground_t, self.ground_q / self.ground_q.norm(dim=1, keepdim=True)], dim=1)
        )[self.tracks_idx]

        current_xy, current_theta = self._get_xy_theta()

        zero_column = torch.zeros(self.xy.size(0), device=self.device).view(-1, 1)
        T_ground_cuboid_local = lt.SE3.InitFromVec(
            torch.cat(
                [
                    current_xy,
                    zero_column,
                    zero_column,
                    zero_column,
                    torch.sin(current_theta / 2).view(-1, 1),
                    torch.cos(current_theta / 2).view(-1, 1),
                ],
                dim=1,
            )
        )
        return T_world_ground * T_ground_cuboid_local * self.Tclc

    def get_consistency(self) -> torch.Tensor:
        """
        Consistency loss that enforces all the parameters in the unicycle model harmonizes.

        Return (n_track_poses - n_tracks, 3) tensor, each column represents:
            1. Consistency in x-direction
            2. Consistency in y-direction
            3. Consistency in theta-direction
        """

        current_xy, current_theta = self._get_xy_theta()

        vdw = self.velocity / (self.omega + 1e-6)
        diff_xy = current_xy[self.off_one_x_mapping + 1] - current_xy[self.off_one_x_mapping]
        consistency_x = diff_xy[:, 0] - vdw * (
            torch.sin(current_theta[self.off_one_x_mapping + 1]) - torch.sin(current_theta[self.off_one_x_mapping])
        )
        consistency_y = diff_xy[:, 1] + vdw * (
            torch.cos(current_theta[self.off_one_x_mapping + 1]) - torch.cos(current_theta[self.off_one_x_mapping])
        )
        consistency_theta = (
            current_theta[self.off_one_x_mapping + 1]
            - current_theta[self.off_one_x_mapping]
            - self.omega * self.time_delta_s
        )
        return torch.stack([consistency_x, consistency_y, consistency_theta], dim=1)

    def get_smoothness(self) -> torch.Tensor:
        """
        Smoothness loss that enforces constant acceleration and angular acceleration

        Return (n_track_poses - n_tracks * 3, 2) tensor, each column represents:
            1. Acceleration
            2. Angular acceleration
        """
        t_0: torch.Tensor = self.off_two_v_mapping
        t_1: torch.Tensor = self.off_two_v_mapping + 1
        t_2: torch.Tensor = self.off_two_v_mapping + 2

        # Linear acceleration computation
        acc_plus = (self.velocity[t_2] - self.velocity[t_1]) / self.time_delta_v_s[t_1]
        acc_minus = (self.velocity[t_1] - self.velocity[t_0]) / self.time_delta_v_s[t_0]

        # Angular acceleration computation
        ang_acc_plus = (self.omega[t_2] - self.omega[t_1]) / self.time_delta_v_s[t_1]
        ang_acc_minus = (self.omega[t_1] - self.omega[t_0]) / self.time_delta_v_s[t_0]

        return torch.stack([acc_plus - acc_minus, ang_acc_plus - ang_acc_minus], dim=1)

    def forward(
        self,
        tracks: CuboidTracks,
    ) -> CuboidTracks:
        self.save_tracks_for_visualize(tracks)
        tracks = CuboidTracks.Ops.clone(tracks)
        tracks.tracks_poses = self.get_pose()
        return tracks

    __call__ = module_call_type(forward)

    def configure_optimizers(self, name_prefix: str = "") -> list[OptimizerLRSchedulerConfig]:
        return configure_optimizers(self.config, self.trainer_config, self, name_prefix)


class SkipTracksCalib(BaseTracksCalib):
    def __init__(self, config: DictConfig, trainer_config: TrainerConfig, cuboid_tracks: CuboidTracks):
        super().__init__(config, trainer_config, cuboid_tracks)

    def forward(
        self,
        tracks: CuboidTracks,
    ) -> CuboidTracks:
        # return inputs as is
        return tracks

    __call__ = module_call_type(forward)


class CompositeTracksCalib(BaseTracksCalib):
    """Routes calibration based on whether tracks are original or inserted.

    Used for inference-time asset insertion/edit flows.
    """

    def __init__(
        self,
        config: DictConfig,
        trainer_config: TrainerConfig,
        original_calib: BaseTracksCalib,
        inserted_track_ids: set[str],
        cuboid_tracks: CuboidTracks,
    ):
        # Avoid BaseTracksCalib.__init__ to prevent repeatedly mutating shared config
        # (e.g. start_global_step world-size adjustment) on chained insertions.
        BaseModel.__init__(self, config)
        self.trainer_config = trainer_config
        self.original_calib = original_calib
        self.inserted_track_ids = set(inserted_track_ids)

    def forward(self, tracks: CuboidTracks) -> CuboidTracks:
        # Edge case: no inserted tracks.
        if not self.inserted_track_ids:
            return self.original_calib(tracks)

        ordered_all_ids = tracks.tracks_id
        inserted_track_ids_ordered = [track_id for track_id in ordered_all_ids if track_id in self.inserted_track_ids]

        # No inserted track is present in this specific track set.
        if not inserted_track_ids_ordered:
            return self.original_calib(tracks)

        # Edge case: all tracks are inserted; equivalent to SkipCalib behavior.
        if len(inserted_track_ids_ordered) == len(ordered_all_ids):
            return tracks

        original_track_ids_ordered = [
            track_id for track_id in ordered_all_ids if track_id not in self.inserted_track_ids
        ]
        original_tracks = CuboidTracks.Ops.subset_from_tracks_id(tracks, original_track_ids_ordered)
        inserted_tracks = CuboidTracks.Ops.subset_from_tracks_id(tracks, inserted_track_ids_ordered)

        calibrated_original = self.original_calib(original_tracks)
        merged_tracks = CuboidTracks.Ops.concatenate([calibrated_original, inserted_tracks])

        # Preserve the original tracks order so gaussian_cuboid_ids keep a stable track-index mapping.
        reordered_tracks = CuboidTracks.Ops.subset_from_tracks_id(merged_tracks, ordered_all_ids)
        assert reordered_tracks.tracks_id == ordered_all_ids, (
            "CompositeTracksCalib: output track order must match input track order"
        )
        return reordered_tracks

    __call__ = module_call_type(forward)
