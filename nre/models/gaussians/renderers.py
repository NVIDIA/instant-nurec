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

import copy
import logging
import math

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Callable, List, NamedTuple, Optional, Type, cast

import gsplat
import torch

from omegaconf import OmegaConf

from libs.geometry.kernels.pose import se3pose_to_inverse_matrix

# C++ / CUDA libs
from libs.nrend.renderer import DifferentiableRenderer, Renderer  # type: ignore
from ncore.data import (
    BivariateWindshieldModelParameters,
    ConcreteCameraModelParametersUnion,
    ConcreteExternalDistortionParametersUnion,
    ConcreteLidarModelParametersUnion,
    FThetaCameraModelParameters,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    ReferencePolynomial,
    RowOffsetStructuredSpinningLidarModelParameters,
    ShutterType,
)
from nre.config.base_schema import config_to_primitive
from nre.config.model import BaseRendererConfig, NRendRendererConfig, SensorOutputConfig
from nre.models.base import BaseModel
from nre.utils.batch import FrameMeta, RenderingData
from nre.utils.misc import get_union_types
from nre.utils.prober import get_global_prober
from nre.utils.profiling import ScopedTimer
from nre.utils.types import (
    ExtraSignal,
    GaussiansRenderReturn,
)


class BaseGaussianRenderer(ABC):
    def __init__(self, config: BaseRendererConfig, model: BaseModel) -> None:
        self.config = config

    @abstractmethod
    def render(
        self,
        rendering_data: RenderingData,
        gaussian_parameters: dict[str, torch.Tensor],
        n_active_features: int,
        extra_ray_signal_infos: tuple[list[str], list[int], list[Callable]],
        frame_meta: Optional[List[FrameMeta]] = None,
    ) -> GaussiansRenderReturn: ...

    def render_with_deferred_bp(
        self,
        rendering_data: RenderingData,
        gaussian_parameters: dict[str, torch.Tensor],
        n_active_features: int,
        extra_ray_signal_infos: tuple[list[str], list[int], list[Callable]] = ([], [], []),
        frame_meta: Optional[List[FrameMeta]] = None,
    ) -> GaussiansRenderReturn:
        """
        During training time, when multiple frames are rendered in a single feed-forward pass, the number of intermediate parameters needed to be cached for
        gradient computation can be very large (e.g. the model parameters in the nrend renderer), resulting in large training memory usage.

        Deferred back-propogation is a memory-efficient way to compute backward gradient, first proposed in: **ARF: Artistic Radiance Fields** (https://arxiv.org/abs/2206.06360).
        The idea is to do a no_grad forward pass first. When gradients are needed, we re-do a grad_enabled forward pass followed immediately by a backward pass.
        In ARF, the above process is repeated for each patch separately, while for 3DGS, we do it for each image.
        Ref: GRM: https://github.com/camenduru/GRM/blob/68a7226e9f05a1bcee5d9624dc7336b06c4dd3ed/model/render/deferred_bp.py

        Implementation-wise, this is essentially the same as checkpointing. We just need to flatten/unflatten the dataclasses to list of tensors.
        """
        assert not (rendering_data.rays.requires_grad or rendering_data.poses_tquat_startend.requires_grad), (
            "Gradients w.r.t rendering data are not supported for now."
        )
        kwargs_keys: list[str] = list(gaussian_parameters.keys())

        def render_fn(*args: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
            parameters_dict = dict(zip(kwargs_keys, args))
            return self.render(
                rendering_data, parameters_dict, n_active_features, extra_ray_signal_infos, frame_meta
            ).to_tuple()

        ## when use_reentrant=False:
        # - The forward pass still runs with gradient enabled, but with a hook that replaces ctx.saved_tensors as "weakref" to save memory
        # of immediate activations.
        # - During backward pass, unpacking ctx.saved_tensor will trigger the function to run again until all needed saved_tensors are available.
        # - The function execution can end earlier (through throwing an _StopRecomputationError).
        # - However, the nrend.render implementation stores too much tensors in, e.g., ctx.undifferentiable_parameters and those are carried in the
        # forward pass, so memory usage there is still high. We have a new checkpoint-friendly implementation for this.
        ## when use_reentrant=True:
        # - During forward pass the function runs with torch.no_grad(), hence no computation graph is built, nor is ctx.undifferentiable_parameters kept,
        # so the memory is under control.
        # - Note that this deprecated though, and has limitations in e.g. torch.compile'd backward pass, or nested input tensors.
        # - Hence we need to "flatten" the input dataclasses into tensor tuples as arguments here.
        result = torch.utils.checkpoint.checkpoint(
            render_fn, *(gaussian_parameters[key] for key in kwargs_keys), use_reentrant=True
        )
        return GaussiansRenderReturn.from_tuple(result)

    @staticmethod
    def factory(name: str, config: BaseRendererConfig, model: BaseModel) -> BaseGaussianRenderer:
        """Factory method to create renderer instances.

        Available renderers:
        - NRend variants (current):
          - 3dgrt-optix-nrend: NRend with OptiX ray tracing
          - 3dgrt-rejection-optix-nrend: NRend with OptiX and rejection sampling
          - 3dgut-nrend: NRend with 3DGUT (distortion, rolling shutter)
          - 3dgs-nrend: NRend standard 3DGS

        - GSplat variants (new):
          - 3dgs-gsplat: GSplat standard 3DGS (no distortion, pose gradients supported)
          - 3dgut-gsplat: GSplat with 3DGUT (distortion, rolling shutter, NO pose gradients)

        Args:
            name: Renderer variant name
            config: Renderer configuration
            model: Model instance

        Returns:
            BaseGaussianRenderer instance
        """
        variants: dict[str, Type[BaseGaussianRenderer]] = {
            "3dgrt-optix-nrend": Gaussian3DNRenderer,
            "3dgrt-rejection-optix-nrend": Gaussian3DNRenderer,
            "3dgut-nrend": Gaussian3DNRenderer,
            "3dgs-nrend": Gaussian3DNRenderer,
        }
        if name not in variants:
            raise ValueError(f"Unknown renderer variant: {name}. Available variants: {list(variants.keys())}")
        return variants[name](config, model)

    def collect_profilings(self) -> dict[str, float]:
        return {}


class Gaussian3DNRenderer(BaseGaussianRenderer):
    model_state_dict_prefix: str = ""

    config: NRendRendererConfig  # type: ignore[assignment] # Narrow type

    def __init__(self, config: NRendRendererConfig, model: BaseModel) -> None:
        super().__init__(config, model)

        self.config = config

        # Get the model configuration from the model instance
        model_dict = model.serialize_to_json_dict(with_state_dict=False)

        # Force fourier_features_dim to 1
        model_dict["nre_data"]["config"]["fourier_features_dim"] = 1

        # Create the renderer
        self.nrend_diff_renderer = DifferentiableRenderer(
            model=model_dict,
            render_settings=cast(Any, config_to_primitive(self.config)),
            log_level=Renderer.LogLevel(self.config.log_level),
            profiling_frequency=self.config.profiling.frequency,
        )

        # Setup split rendering interface : preparation / rendering
        self.nrend_diff_prepare_renderer: Optional[DifferentiableRenderer] = None

        if self.config.prepare_before_render:
            # Use the default renderer to prepare the local particles
            self.nrend_diff_prepare_renderer = self.nrend_diff_renderer
            # Since the preparation handle feature computation the global renderer has to use a different model
            model_dict["nre_data"]["config"]["particle"]["radiance_sph_degree"] = 0
            model_dict["nre_data"]["config"]["particle"]["extra_signal_sph_degree"] = 0
            model_dict["nre_data"]["config"]["particle"]["camera_extra_signal_sph_degree"] = 0
            model_dict["nre_data"]["config"]["particle"]["lidar_extra_signal_sph_degree"] = 0
            self.nrend_diff_renderer = DifferentiableRenderer(
                model=model_dict,
                render_settings=cast(Any, config_to_primitive(self.config)),
                log_level=Renderer.LogLevel(self.config.log_level),
                profiling_frequency=self.config.profiling.frequency,
            )

        # For LiDAR rendering, we also need to set the n_bins_elevation and max_pts_per_tile
        lidar_tiling = self.config.tiling.lidar
        self._lidar_max_pts_per_tile = lidar_tiling.tile_size_elevation * lidar_tiling.tile_size_azimuth
        self._lidar_n_bins_elevation = lidar_tiling.n_bins_elevation
        self._lidar_resolution_elevation = lidar_tiling.resolution_elevation
        self._lidar_densification_factor_azimuth = lidar_tiling.densification_factor_azimuth

        # Set the enable_ray_based_culling flag
        enable_ray_based_culling = self.config.culling.enable_ray_based_culling
        logging.getLogger(__name__).info(f"Gaussian3DNRenderer: enable_ray_based_culling: {enable_ray_based_culling}")

    def render(
        self,
        rendering_data: RenderingData,
        gaussian_parameters: dict[str, torch.Tensor],
        n_active_features: int,
        extra_ray_signal_infos: tuple[list[str], list[int], list[Callable]],
        frame_meta: Optional[List[FrameMeta]] = None,
    ) -> GaussiansRenderReturn:
        assert rendering_data.b == 1, "Only single-frame batch is supported"
        if frame_meta is not None:
            assert len(frame_meta) == 1, "Only single-frame batch is supported"

        sensor_parameters: ConcreteCameraModelParametersUnion | ConcreteLidarModelParametersUnion = (
            rendering_data.sensor_model_parameters[0]
        )
        rendering_lidar_sensor = isinstance(sensor_parameters, get_union_types(ConcreteLidarModelParametersUnion))
        rendering_camera_sensor = isinstance(sensor_parameters, get_union_types(ConcreteCameraModelParametersUnion))

        # pack the parameters
        with ScopedTimer("Gaussian3DNRenderer/render/pack_parameters"):
            num_particles = gaussian_parameters["positions"].shape[0]
            particle_density = torch.concat(
                [
                    gaussian_parameters["positions"],
                    gaussian_parameters["densities"],
                    gaussian_parameters["rotations"],
                    gaussian_parameters["scales"],
                    torch.zeros_like(gaussian_parameters["densities"]),
                ],
                dim=1,
            )
            particle_radiance = gaussian_parameters["features"].reshape(
                num_particles, gaussian_parameters["features"].shape[-1] // 3, 3
            )
            particle_extra_signal = gaussian_parameters["extra_signal"].reshape(
                num_particles, gaussian_parameters["extra_signal"].shape[-1]
            )

            if rendering_camera_sensor:
                particle_active_sensor_extra_signal_key = ".particle_camera_extra_signal"
                particle_active_sensor_extra_signal = gaussian_parameters["camera_extra_signal"].reshape(
                    num_particles, gaussian_parameters["camera_extra_signal"].shape[-1]
                )
                particle_inactive_sensor_extra_signal_key = ".particle_lidar_extra_signal"
                particle_inactive_sensor_extra_signal = torch.empty(
                    0, device=particle_active_sensor_extra_signal.device
                )
            elif rendering_lidar_sensor:
                particle_active_sensor_extra_signal_key = ".particle_lidar_extra_signal"
                particle_active_sensor_extra_signal = gaussian_parameters["lidar_extra_signal"].reshape(
                    num_particles, gaussian_parameters["lidar_extra_signal"].shape[-1]
                )
                particle_inactive_sensor_extra_signal_key = ".particle_camera_extra_signal"
                particle_inactive_sensor_extra_signal = torch.empty(
                    0, device=particle_active_sensor_extra_signal.device
                )

        # update the model parameters
        with ScopedTimer("Gaussian3DNRenderer/render/update_model_parameters"):
            model_parameters_to_update = {
                self.model_state_dict_prefix + ".particles_number": torch.tensor(
                    [num_particles], dtype=torch.uint32, device="cpu"
                ),
                self.model_state_dict_prefix + ".active_sh_degree": torch.tensor(
                    [n_active_features], dtype=torch.int32, device="cpu"
                ),
                self.model_state_dict_prefix + ".particle_density": particle_density,
                self.model_state_dict_prefix + ".particle_radiance": particle_radiance,
                self.model_state_dict_prefix + ".particle_extra_signal": particle_extra_signal,
                self.model_state_dict_prefix
                + particle_active_sensor_extra_signal_key: particle_active_sensor_extra_signal,
                self.model_state_dict_prefix
                + particle_inactive_sensor_extra_signal_key: particle_inactive_sensor_extra_signal,
            }
            if self.nrend_diff_prepare_renderer is not None:
                self.nrend_diff_prepare_renderer.update_model_parameters(model_parameters_to_update)  # type: ignore
            else:
                self.nrend_diff_renderer.update_model_parameters(model_parameters_to_update)

        with ScopedTimer("Gaussian3DNRenderer/render/prepare_and_cache_sensor_model"):
            frame_idx = 0  # default_frame_idx
            frame_start_timestamp, frame_end_timestamp = rendering_data.timestamps_startend_us_cpu[0].tolist()
            frame_start_timestamp = int(frame_start_timestamp)
            frame_end_timestamp = int(frame_end_timestamp)

            rays_timestamps_us: Optional[torch.Tensor] = None
            if rendering_camera_sensor:
                camera_parameters = cast(ConcreteCameraModelParametersUnion, sensor_parameters)
                rays = rendering_data.rays[0]  # [H, W, 6]
                height, width = rays.shape[:2]
                ray_origins, ray_directions = torch.split(rays, [3, 3], dim=-1)
                sensor_parameters_data = DifferentiableRenderer.prepare_and_cache_camera_model(
                    camera_parameters,
                    device=ray_origins.device,
                )
                ray_origins = ray_origins.contiguous()
                ray_directions = ray_directions.contiguous()
                # ray direction lengths encode ray spreads
                if rendering_data._rays_footprints is not None:
                    assert rendering_data._rays_footprints.shape == (
                        1,
                        height,
                        width,
                        1,
                    ), f"footprint.shape: {rendering_data._rays_footprints.shape}"
                    ray_directions = ray_directions * rendering_data._rays_footprints[0]
                if rendering_data.rays_timestamps_us is not None:
                    rays_timestamps_us = rendering_data.rays_timestamps_us[0].contiguous().reshape(-1)
            elif rendering_lidar_sensor:
                lidar_parameters = cast(ConcreteLidarModelParametersUnion, sensor_parameters)
                # Be Careful: NRend expect lidar rays to be organized in the column-major, instead of row-major as in camera rays.
                # This means we need to transpose the lidar rays from shape (height, width) to
                # (width, height) ((column, row)) before passing into NRend (and transpose back afterwards).
                rays = rendering_data.rays[0].transpose(0, 1)  # [W, H, 6]
                height, width = rays.shape[:2]
                ray_origins, ray_directions = torch.split(rays, [3, 3], dim=-1)
                sensor_parameters_data = DifferentiableRenderer.prepare_and_cache_lidar_model(
                    lidar_parameters,
                    device=ray_origins.device,
                    n_bins_elevation=self._lidar_n_bins_elevation,
                    max_pts_per_tile=self._lidar_max_pts_per_tile,
                    resolution_elevation=self._lidar_resolution_elevation,
                    densification_factor_azimuth=self._lidar_densification_factor_azimuth,
                ).parameters
                # FIXME : move the default lidar divergence to the sensor ray footprint
                ray_directions = ray_directions * self.config.antialiasing.lidar_divergence
                # Sanity check to ensure the lidar model is properly initialized with all the required information
                assert sensor_parameters_data.has_rolling_shutter_info()
                assert sensor_parameters_data.has_tiling_info()
            else:
                raise ValueError(f"Unsupported sensor type: {type(sensor_parameters)}")

            if frame_meta is not None:
                sensor_frame_idx = torch.tensor(
                    [frame_meta[0].unique_sensor_idx, frame_meta[0].unique_frame_idx],
                    dtype=torch.int32,
                    device=ray_origins.device,
                )
            else:
                sensor_frame_idx = None

            poses_tquat_start, poses_tquat_end = torch.unbind(rendering_data.poses_tquat_startend[0], dim=0)  # (7,)

        if self.nrend_diff_prepare_renderer is not None:
            with ScopedTimer("Gaussian3DNRenderer/render/prepare"):
                (
                    local_particles_density,
                    local_particles_radiance,
                    local_particles_extra_signal,
                    local_particles_sensor_extra_signal,
                    _,  # scene_data unused in prepare_scene path
                ) = self.nrend_diff_prepare_renderer.render(
                    frame_id=frame_idx,
                    frame_width=width,
                    frame_height=height,
                    frame_start_timestamp=frame_start_timestamp,
                    frame_end_timestamp=frame_end_timestamp,
                    rays_origin=ray_origins.contiguous(),
                    rays_direction=ray_directions.contiguous(),
                    rays_timestamp=rays_timestamps_us,
                    frames_sensor_model=sensor_parameters_data,
                    frames_sensor_ids=sensor_frame_idx,
                    frames_sensor_start_pose=poses_tquat_start,
                    frames_sensor_end_pose=poses_tquat_end,
                    prepare_scene=True,
                )
                # TODO : for MGPU support, gather all local particles into global particles
                (
                    global_particles_density,
                    global_particles_radiance,
                    global_particles_extra_signal,
                    global_particles_sensor_extra_signal,
                ) = (
                    local_particles_density,
                    local_particles_radiance,
                    local_particles_extra_signal,
                    local_particles_sensor_extra_signal,
                )
                # update renderer parameters with the global particles
                model_parameters_to_update = {
                    self.model_state_dict_prefix + ".particles_number": torch.tensor(
                        [global_particles_density.shape[0]], dtype=torch.uint32, device="cpu"
                    ),
                    self.model_state_dict_prefix + ".particle_density": global_particles_density,
                    self.model_state_dict_prefix + ".particle_radiance": global_particles_radiance,
                    self.model_state_dict_prefix + ".particle_extra_signal": global_particles_extra_signal,
                    self.model_state_dict_prefix
                    + particle_active_sensor_extra_signal_key: global_particles_sensor_extra_signal,
                    self.model_state_dict_prefix
                    + particle_inactive_sensor_extra_signal_key: particle_inactive_sensor_extra_signal,
                }
                self.nrend_diff_renderer.update_model_parameters(model_parameters_to_update)

        with ScopedTimer("Gaussian3DNRenderer/render/render"):
            radiance_opacity, distance, normal, extra_ray_signals, scene_data = self.nrend_diff_renderer.render(
                frame_id=frame_idx,
                frame_width=width,
                frame_height=height,
                frame_start_timestamp=frame_start_timestamp,
                frame_end_timestamp=frame_end_timestamp,
                rays_origin=ray_origins.contiguous(),
                rays_direction=ray_directions.contiguous(),
                rays_timestamp=rays_timestamps_us,
                frames_sensor_model=sensor_parameters_data,  # sensor model parameters
                frames_sensor_ids=sensor_frame_idx,
                frames_sensor_start_pose=poses_tquat_start,  # sensor to world
                frames_sensor_end_pose=poses_tquat_end,  # sensor to world
                checkpoint_friendly_backward=self.config.checkpoint_friendly_backward,
            )
            has_radiance = radiance_opacity.shape[-1] > 1
            if has_radiance:
                radiance, opacity = torch.split(radiance_opacity, [3, 1], dim=2)
            else:
                opacity = radiance_opacity

            has_normal = normal.shape[-1] > 0
            parsed_scene_data = self.nrend_diff_renderer.parse_scene_data(scene_data)

        with ScopedTimer("Gaussian3DNRenderer/render/transpose_output"):
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
                rgb=radiance.reshape(-1, 3).contiguous() if has_radiance else None,
                opacity=opacity.flatten(),
                distance=distance.flatten(),  # ray distance from NRend
                normal=normal.reshape(-1, 3) if has_normal else None,  # ray normal from NRend
                extra_ray_signals=ExtraSignal.from_packed_tensor(
                    extra_ray_signals.reshape(-1, c), extra_ray_signal_infos
                )
                if (c := extra_ray_signals.shape[-1]) > 0
                else None,
                visibility=parsed_scene_data.get("visibility", None),
                cumulated_weights=parsed_scene_data.get("cumulated_weights", None),
            )

    def collect_profilings(self) -> dict[str, float]:
        profilings_dict = {} if self.nrend_diff_renderer is None else self.nrend_diff_renderer.collect_profilings()
        if self.nrend_diff_prepare_renderer is not None:
            profilings_dict.update(self.nrend_diff_prepare_renderer.collect_profilings())
        return profilings_dict


