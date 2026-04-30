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

import gzip
import json
import warnings

from typing import Any, Optional, cast

import msgpack
import torch

from libs.nrend.renderer import Renderer  # type: ignore
from ncore.data import (
    ConcreteCameraModelParametersUnion,
    ConcreteLidarModelParametersUnion,
)
from nre.datasets.tracks import CuboidTracks
from nre.models.base import BaseModel
from nre.utils.batch import FrameMeta, RenderingData
from nre.utils.misc import get_union_types
from nre.utils.types import (
    ExtraSignal,
    GaussiansRenderReturn,
)


class NRenderableModel:
    _rendered_model_json_path: Optional[str]
    _renderer_settings: dict[str, Any] | Renderer.Hint
    _nrend_renderer: Optional[Renderer]

    def __init__(self):
        self._rendered_model_json_path = None
        self._renderer_settings = Renderer.Hint.DEFAULT
        self._nrend_renderer = None

    def parse_extra_ray_signals(self, extra_ray_signals: torch.Tensor, lidar_extra_ray_signals: bool) -> ExtraSignal:
        raise NotImplementedError

    def model_json_dict(self) -> dict[str, Any]:
        if self._rendered_model_json_path:
            with gzip.open(self._rendered_model_json_path, "r") as f:
                return msgpack.unpackb(f.read())
        else:
            assert isinstance(self, BaseModel), "Only BaseModel are NRenderable"
            return self.serialize_to_json_dict()

    def renderer_settings(self):
        return self._renderer_settings

    def setup_nrend(
        self,
        track_ids: list[str] = [],
        rendered_model_json_path: Optional[str] = None,
        renderer_hint: Renderer.Hint = Renderer.Hint.DEFAULT,
        renderer_settings_json_path: Optional[str] = None,
        log_level: Renderer.LogLevel = Renderer.LogLevel.ERROR,
        profiling_frequency: float = 0.0,
    ) -> None:
        assert isinstance(self, BaseModel), "Only BaseModel are NRenderable"
        self._rendered_model_json_path = rendered_model_json_path
        # Render settings logic :
        # - if renderer_hint is DEFAULT :
        #   - if renderer_settings_path is provided, use the renderer settings from the file
        #   - else use the renderer settings from the config
        # - if renderer_hint is not DEFAULT, use the default renderer settings from the renderer_hint
        if renderer_hint == Renderer.Hint.DEFAULT:
            if renderer_settings_json_path is not None:
                with open(renderer_settings_json_path, "r") as f:
                    self._renderer_settings = json.load(f)
            else:
                renderer_config = getattr(self.config, "renderer", None)
                if renderer_config is not None:
                    from nre.config.base_schema import config_to_primitive

                    self._renderer_settings = cast(Any, config_to_primitive(renderer_config))
                else:
                    self._renderer_settings = {}
        else:
            self._renderer_settings = renderer_hint
        renderer = Renderer(
            self.model_json_dict(),
            render_settings=self._renderer_settings,
            track_instances_uid_map=track_ids,
            log_level=log_level,
            profiling_frequency=profiling_frequency,
        )
        self._nrend_renderer = renderer if renderer.valid() else None
        if self._nrend_renderer is None:
            warnings.warn(f"[{self.__class__.__name__}] model is not nrenderable !")
        # For LiDAR rendering, we also need to set the n_bins_elevation and max_pts_per_tile
        if isinstance(self._renderer_settings, dict):
            lidar_tiling = self._renderer_settings.get("tiling", {}).get("lidar", {})
            lidar_tile_size_elevation = lidar_tiling.get("tile_size_elevation", 16)
            lidar_tile_size_azimuth = lidar_tiling.get("tile_size_azimuth", 16)
            self._lidar_max_pts_per_tile = lidar_tile_size_elevation * lidar_tile_size_azimuth
            self._lidar_n_bins_elevation = lidar_tiling.get("n_bins_elevation", 16)
        else:
            warnings.warn(
                f"[{self.__class__.__name__}] renderer_hint testing : using default LiDAR tiling values [16, 16] may lead to inconsistencies"
            )
            # Use default values when renderer_settings is a Renderer.Hint enum
            self._lidar_max_pts_per_tile = 16 * 16
            self._lidar_n_bins_elevation = 16

    def has_nrend(self) -> bool:
        return self._nrend_renderer is not None

    def render_nrend_sensor_rays_with_poses(
        self,
        frame_idx: int,
        rendering_data: RenderingData,
        frame_meta: FrameMeta,
        num_active_track_instances,
        active_track_instances_ids,
        active_track_instances_start_pose,
        active_track_instances_end_pose,
    ) -> GaussiansRenderReturn:
        device = rendering_data.rays.device

        rays_timestamps_us = (
            rendering_data.rays_timestamps_us[0] if rendering_data.rays_timestamps_us is not None else None
        )  # [H, W, 1]
        sensor_parameters: ConcreteCameraModelParametersUnion | ConcreteLidarModelParametersUnion = (
            rendering_data.sensor_model_parameters[0]
        )
        poses_tquat_sensor = rendering_data.poses_tquat_startend[0]
        rendering_camera_sensor = isinstance(sensor_parameters, get_union_types(ConcreteCameraModelParametersUnion))
        rendering_lidar_sensor = isinstance(sensor_parameters, get_union_types(ConcreteLidarModelParametersUnion))
        if rendering_camera_sensor:
            camera_parameters = cast(ConcreteCameraModelParametersUnion, sensor_parameters)
            rays = rendering_data.rays[0]  # [H, W, 6]
            height, width = rays.shape[:2]
            ray_origins, ray_directions = torch.split(rays, [3, 3], dim=-1)
            sensor_parameters_data = Renderer.prepare_and_cache_camera_model(
                camera_parameters,
                device=device,
            )
            # ray direction lengths encode ray spreads
            if rendering_data._rays_footprints is not None:
                assert rendering_data._rays_footprints.shape == (
                    1,
                    height,
                    width,
                    1,
                ), f"footprint.shape: {rendering_data._rays_footprints.shape}"
                ray_directions = ray_directions * rendering_data._rays_footprints[0]
        elif rendering_lidar_sensor:
            lidar_parameters = cast(ConcreteLidarModelParametersUnion, sensor_parameters)
            # Be Careful: NRend expect lidar rays to be organized in the column-major, instead of row-major as in camera rays.
            # This means we need to transpose the lidar rays from shape (height, width) to
            # (width, height) ((column, row)) before passing into NRend (and transpose back afterwards).
            rays = rendering_data.rays[0].transpose(0, 1)  # [W, H, 6]
            height, width = rays.shape[:2]
            ray_origins, ray_directions = torch.split(rays, [3, 3], dim=-1)
            assert isinstance(self, BaseModel), "Only BaseModel are NRenderable (need config)"
            renderer_config = getattr(self.config, "renderer", None)
            assert renderer_config is not None, "renderer config required for lidar"
            ray_directions = ray_directions * renderer_config.antialiasing.lidar_divergence
            sensor_parameters_data = Renderer.prepare_and_cache_lidar_model(
                lidar_parameters,
                device=device,
                n_bins_elevation=self._lidar_n_bins_elevation,
                max_pts_per_tile=self._lidar_max_pts_per_tile,
            ).parameters

            # Sanity check to ensure the lidar model is properly initialized with all the required information
            assert sensor_parameters_data.has_rolling_shutter_info()
            assert sensor_parameters_data.has_tiling_info()
        else:
            raise ValueError(f"Unsupported model parameters type: {type(rendering_data.sensor_model_parameters[0])}")

        assert self._nrend_renderer is not None, f"[{self.__class__.__name__}] nrend renderer is not initialized"
        # Use CPU copy to avoid GPU->CPU sync when calling .item()
        timestamps_tensor = rendering_data.timestamps_startend_us_cpu[0, :]  # (2,) - on CPU
        frame_start_timestamp = int(timestamps_tensor[0].item())
        frame_end_timestamp = int(timestamps_tensor[1].item())
        radiance_opacity, distance, normal, extra_ray_signals, _ = self._nrend_renderer.render(
            frame_idx,
            rendering_data.w,
            rendering_data.h,
            frame_start_timestamp,
            frame_end_timestamp,
            ray_origins.contiguous(),
            ray_directions.contiguous(),
            rays_timestamps_us,
            sensor_parameters_data,
            torch.stack(
                [
                    torch.tensor(frame_meta.unique_sensor_idx, device=device),
                    torch.tensor(frame_meta.unique_frame_idx, device=device),
                ],
                dim=0,
            ),
            poses_tquat_sensor[0, :] if poses_tquat_sensor is not None else None,
            poses_tquat_sensor[1, :] if poses_tquat_sensor is not None else None,
            num_active_track_instances,
            active_track_instances_ids,
            active_track_instances_start_pose,
            active_track_instances_end_pose,
        )

        has_radiance = radiance_opacity.shape[-1] > 1
        if has_radiance:
            radiance, opacity = torch.split(radiance_opacity, [3, 1], dim=2)
        else:
            opacity = radiance_opacity

        has_normal = normal.shape[-1] > 0

        assert isinstance(self, BaseModel), "Only BaseModel are NRenderable"

        if rendering_lidar_sensor:
            # transpose the output back from (width, height) to (height, width) for lidar rendering
            if has_radiance:
                radiance = radiance.transpose(0, 1)
            if has_normal:
                normal = normal.transpose(0, 1)
            opacity = opacity.transpose(0, 1)
            distance = distance.transpose(0, 1)
            if extra_ray_signals.shape[-1] > 0:
                extra_ray_signals = extra_ray_signals.transpose(0, 1)

        return GaussiansRenderReturn(
            opacity=opacity.reshape((-1,)),
            distance=distance.reshape((-1,)),
            normal=normal.reshape((-1, 3)) if has_normal else None,
            rgb=radiance.reshape((-1, 3)) if has_radiance else None,
            # FIXME : parse the extra-signals from the model config
            extra_ray_signals=self.parse_extra_ray_signals(extra_ray_signals.reshape(-1, c), rendering_lidar_sensor)
            if (c := extra_ray_signals.shape[-1]) > 0
            else None,
        )

    def render_nrend_sensor_rays(
        self,
        default_frame_idx: int,
        rendering_data: RenderingData,
        frame_meta: FrameMeta,
        tracks: Optional[CuboidTracks] = None,
    ) -> GaussiansRenderReturn:
        frame_idx = default_frame_idx
        # Use CPU copy to avoid GPU->CPU sync when calling .item()
        timestamps_tensor = rendering_data.timestamps_startend_us_cpu[0, :]  # (2,) - on CPU
        frame_start_timestamp = int(timestamps_tensor[0].item())
        frame_end_timestamp = int(timestamps_tensor[1].item())

        if tracks is not None:
            (
                num_active_track_instances,
                active_track_instances_ids,
                active_track_instances_start_pose,
                active_track_instances_end_pose,
            ) = tracks.frame_poses_interpolation(
                torch.tensor([frame_start_timestamp, frame_end_timestamp], device="cuda", dtype=torch.int64)
            )
        else:
            num_active_track_instances = torch.zeros((1,), dtype=torch.int32)
            active_track_instances_ids = None
            active_track_instances_start_pose = None
            active_track_instances_end_pose = None

        return self.render_nrend_sensor_rays_with_poses(
            frame_idx,
            rendering_data,
            frame_meta,
            int(num_active_track_instances[0]),
            active_track_instances_ids,
            active_track_instances_start_pose,
            active_track_instances_end_pose,
        )

    def nrend_profilings(self) -> dict[str, float]:
        return {} if self._nrend_renderer is None else self._nrend_renderer.collect_profilings()

    def get_updated_cuboid_tracks(self) -> Optional[CuboidTracks]:
        """
        Get the updated cuboid tracks of the model.
        """
        return None
