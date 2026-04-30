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
from libs.geometry.kernels.quaternion import quat_slerp

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
from nre.config.model import BaseRendererConfig, GSplatRendererConfig, NRendRendererConfig, SensorOutputConfig
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
            # NRend variants
            "3dgrt-optix-nrend": Gaussian3DNRenderer,
            "3dgrt-rejection-optix-nrend": Gaussian3DNRenderer,
            "3dgut-nrend": Gaussian3DNRenderer,
            "3dgs-nrend": Gaussian3DNRenderer,
            # GSplat variants
            "3dgs-gsplat": GSplatRenderer,  # Standard 3DGS: with_ut=False, pose grads supported
            "3dgut-gsplat": GSplatRenderer,  # 3DGUT: with_ut=True, NO pose grads
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


class ExternalDistortionAdapter(ABC):
    """Abstract base class for external distortion adapters.

    Provides a unified interface for different external distortion models
    to simplify GSplat rendering code through polymorphism.
    """

    def __init__(self, external_distortion_params: ConcreteExternalDistortionParametersUnion, device: torch.device):
        self.external_distortion_params = external_distortion_params
        self.device = device

    @abstractmethod
    def get_external_distortion_name(self) -> gsplat.ExternalDistortionModelMeta:
        """Get external distortion name.

        Returns:
            External distortion name
        """
        pass

    @abstractmethod
    def get_distortion_params(self) -> gsplat.ExternalDistortionModelParameters:
        """Get distortion parameters for gsplat."""
        pass

    @staticmethod
    def factory(
        sensor_params: ConcreteExternalDistortionParametersUnion, device: torch.device
    ) -> ExternalDistortionAdapter:
        """Factory method to create appropriate external distortion adapter.

        Args:
            sensor_params: External distortion parameters
            device: Torch device

        Returns:
            External distortion adapter instance
        """
        if isinstance(sensor_params, BivariateWindshieldModelParameters):
            return BivariateWindshieldAdapter(sensor_params, device)
        else:
            raise ValueError(f"Unsupported external distortion model: {type(sensor_params)}")


class BivariateWindshieldAdapter(ExternalDistortionAdapter):
    """Adapter for bivariate windshield external distortion model."""

    def get_external_distortion_name(self) -> gsplat.ExternalDistortionModelMeta:
        return "bivariate-windshield"

    def get_distortion_params(self) -> gsplat.ExternalDistortionModelParameters:
        params = cast(BivariateWindshieldModelParameters, self.external_distortion_params)

        match params.reference_poly:
            case ReferencePolynomial.FORWARD:
                reference_poly = gsplat.rendering.ExternalDistortionReferencePolynomial.FORWARD
            case ReferencePolynomial.BACKWARD:
                reference_poly = gsplat.rendering.ExternalDistortionReferencePolynomial.BACKWARD
            case _:
                raise ValueError(f"Unsupported reference polynomial type: {params.reference_poly}")

        assert len(params.horizontal_poly) <= gsplat.rendering.BivariateWindshieldModelParameters.MAX_COEFFS
        assert len(params.vertical_poly) <= gsplat.rendering.BivariateWindshieldModelParameters.MAX_COEFFS
        assert len(params.horizontal_poly_inverse) <= gsplat.rendering.BivariateWindshieldModelParameters.MAX_COEFFS
        assert len(params.vertical_poly_inverse) <= gsplat.rendering.BivariateWindshieldModelParameters.MAX_COEFFS

        # NOTE: We explicitly set the device to "cpu" here becauses gsplat will create
        # std::array's on the host from the tensors to then send in constant cache of
        # the kernels. We don't want to copy the tensor to the GPU here because it would
        # mean a DtoH copy and then another HtoD copy back to the GPU to call the kernel,
        # effectively creating a needless round-trip copy.
        gsplat_params = gsplat.rendering.BivariateWindshieldModelParameters()
        gsplat_params.reference_poly = reference_poly  # type: ignore[attr-defined]
        gsplat_params.horizontal_poly = torch.tensor(  # type: ignore[attr-defined]
            params.horizontal_poly, dtype=torch.float32, device="cpu"
        )
        gsplat_params.vertical_poly = torch.tensor(  # type: ignore[attr-defined]
            params.vertical_poly, dtype=torch.float32, device="cpu"
        )
        gsplat_params.horizontal_poly_inverse = torch.tensor(  # type: ignore[attr-defined]
            params.horizontal_poly_inverse, dtype=torch.float32, device="cpu"
        )
        gsplat_params.vertical_poly_inverse = torch.tensor(  # type: ignore[attr-defined]
            params.vertical_poly_inverse, dtype=torch.float32, device="cpu"
        )
        return gsplat_params


class SensorModelParametersAdapter(ABC):
    """Abstract base class for camera model adapters.

    Provides a unified interface for different camera models (pinhole, fisheye, f-theta)
    to simplify GSplat rendering code through polymorphism.
    """

    def __init__(self, sensor_params: ConcreteCameraModelParametersUnion, device: torch.device):
        self.sensor_params = sensor_params
        self.device = device

    @staticmethod
    def factory(
        sensor_params: ConcreteCameraModelParametersUnion, device: torch.device
    ) -> SensorModelParametersAdapter:
        """Factory method to create appropriate camera adapter.

        Args:
            sensor_params: Camera model parameters
            device: Torch device

        Returns:
            Camera adapter instance
        """
        if isinstance(sensor_params, OpenCVPinholeCameraModelParameters):
            return PinholeSensorParametersAdapter(sensor_params, device)
        elif isinstance(sensor_params, OpenCVFisheyeCameraModelParameters):
            return FisheyeSensorParametersAdapter(sensor_params, device)
        elif isinstance(sensor_params, FThetaCameraModelParameters):
            return FThetaSensorParametersAdapter(sensor_params, device)
        else:
            raise TypeError(f"Unsupported camera model: {type(sensor_params)}")

    @abstractmethod
    def get_camera_model_name(self) -> gsplat.CameraModel:
        """Get gsplat camera model name."""
        pass

    @abstractmethod
    def get_gsplat_sensor_params(self) -> dict[str, Any]:
        """Get distortion parameters for gsplat."""
        pass


class PinholeSensorParametersAdapter(SensorModelParametersAdapter):
    """Adapter for OpenCV pinhole camera model."""

    def get_camera_model_name(self) -> gsplat.CameraModel:
        return "pinhole"

    def get_gsplat_sensor_params(self) -> dict[str, Any]:
        params = cast(OpenCVPinholeCameraModelParameters, self.sensor_params)
        radial = (
            torch.tensor(list(params.radial_coeffs), dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        )  # (1, 1, 6)

        tangential = (
            torch.tensor(list(params.tangential_coeffs), dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )  # (1, 1, 2)

        fx, fy = params.focal_length[0], params.focal_length[1]
        cx, cy = params.principal_point[0], params.principal_point[1]
        Ks = (
            torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        return {"radial_coeffs": radial, "tangential_coeffs": tangential, "Ks": Ks}


class FisheyeSensorParametersAdapter(SensorModelParametersAdapter):
    """Adapter for OpenCV fisheye camera model."""

    def get_camera_model_name(self) -> gsplat.CameraModel:
        return "fisheye"

    def get_gsplat_sensor_params(self) -> dict[str, Any]:
        params = cast(OpenCVFisheyeCameraModelParameters, self.sensor_params)
        radial = (
            torch.tensor(list(params.radial_coeffs), dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        )  # (1, 1, 4)

        fx, fy = params.focal_length[0], params.focal_length[1]
        cx, cy = params.principal_point[0], params.principal_point[1]
        Ks = (
            torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        return {"radial_coeffs": radial, "Ks": Ks}


class FThetaSensorParametersAdapter(SensorModelParametersAdapter):
    """Adapter for F-Theta camera model."""

    def get_camera_model_name(self) -> gsplat.CameraModel:
        return "ftheta"

    def get_gsplat_sensor_params(self) -> dict[str, Any]:
        params = cast(FThetaCameraModelParameters, self.sensor_params)

        # Map polynomial type
        match params.reference_poly:
            case FThetaCameraModelParameters.PolynomialType.PIXELDIST_TO_ANGLE:
                reference_poly = gsplat.rendering.FThetaPolynomialType.PIXELDIST_TO_ANGLE
            case FThetaCameraModelParameters.PolynomialType.ANGLE_TO_PIXELDIST:
                reference_poly = gsplat.rendering.FThetaPolynomialType.ANGLE_TO_PIXELDIST
            case _:
                raise ValueError(f"Unsupported polynomial type: {params.reference_poly}")

        ftheta_params = gsplat.rendering.FThetaCameraDistortionParameters(
            reference_poly=reference_poly,
            pixeldist_to_angle_poly=tuple(params.pixeldist_to_angle_poly.tolist()),
            angle_to_pixeldist_poly=tuple(params.angle_to_pixeldist_poly.tolist()),
            max_angle=float(params.max_angle),
            linear_cde=tuple(params.linear_cde.tolist()),
        )

        # Extract focal length from first-order coefficient
        fx = params.angle_to_pixeldist_poly[1]
        fy = params.angle_to_pixeldist_poly[1]
        cx, cy = params.principal_point[0], params.principal_point[1]
        Ks = (
            torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        return {"ftheta_coeffs": ftheta_params, "Ks": Ks}


class GSplatRenderer(BaseGaussianRenderer):
    """GSplat rendering implementation for Gaussian models.

    Implements the BaseGaussianRenderer interface using the gsplat library
    for 3D Gaussian Splatting with automatic differentiation via PyTorch autograd.
    """

    config: GSplatRendererConfig  # type: ignore[assignment] # Narrow type

    def __init__(self, config: GSplatRendererConfig, model: BaseModel) -> None:
        super().__init__(config, model)
        self.config = config

        # Parse output settings into a per-sensor dict so a lidar variant can be
        # added alongside without touching this structure.
        def _parse_outputs(sensor_outputs: SensorOutputConfig) -> dict:
            return {
                "enable_features": sensor_outputs.enable_features,
                "enable_normals": sensor_outputs.enable_normals,
                "enable_extended_features": sensor_outputs.enable_extended_features,
                "enable_sensor_extended_features": sensor_outputs.enable_sensor_features,
            }

        self._camera_outputs = _parse_outputs(config.outputs.camera)
        self._lidar_outputs = _parse_outputs(config.outputs.lidar)

        # Warn about 3DGUT pose gradient limitation
        if config.name == "3dgut-gsplat" and not config.use_rays:
            logging.getLogger(__name__).warning(
                "GSplatRenderer with 3DGUT and without ray input does NOT support gradients "
                "w.r.t. camera/track poses. Pose optimization workflows (DirectTracksCalib, "
                "FreePoseCalib) will not work with 3dgut-gsplat. "
                "Use 3dgut-nrend for pose optimization. "
            )

        # Validate and extract tile_size (gsplat requires square tiles)
        camera_tiling = config.tiling.camera
        tile_width = camera_tiling.tile_width
        tile_height = camera_tiling.tile_height
        assert tile_width == tile_height, (
            f"GSplat requires square tiles, got tile_width={tile_width}, tile_height={tile_height}"
        )
        self.tile_size = tile_width

        # Validate render mode if specified
        render_mode = config.render.mode
        if render_mode == "kbuffer":
            k_buffer_size = config.render.k_buffer_size
            assert k_buffer_size == 0, (
                f"GSplat Phase 1 only supports k_buffer_size=0 (equivalent to classic blending). "
                f"Got k_buffer_size={k_buffer_size}. See docs/gsplat_integration.md Phase 3.7 for k-buffer support."
            )

        # Storage for last rendering meta (for absgrad access by densification strategies)
        self.last_rendering_meta: Optional[dict] = None

        self.use_rays: bool = config.use_rays
        # Log once per process; composite creates camera + lidar renderer with same config
        _log_key = "_logged_rays" if self.use_rays else "_logged_sensor_model"
        if not getattr(GSplatRenderer, _log_key, False):
            if self.use_rays:
                logging.getLogger(__name__).info("GSplatRenderer: rasterization using rays as input")
            else:
                logging.getLogger(__name__).info("GSplatRenderer: rasterization using sensor model as input")
            setattr(GSplatRenderer, _log_key, True)  # type: ignore[attr-defined]

        # Load model config from serialized model dict (same source as nrend C++ uses).
        model_dict = model.serialize_to_json_dict(with_state_dict=False)
        model_config = model_dict.get("nre_data", {}).get("config", {})
        self._saturate_radiance: bool = model_config.get("saturate_radiance", True)

        # Load extra signal SH degrees from the model's particle config.
        # SH degrees must match across enabled signals because gsplat uses a single degree.
        particle_config = model_config.get("particle", {})
        extra_signal_dim = particle_config.get("extra_signal_dim", 0)
        camera_extra_signal_dim = particle_config.get("camera_extra_signal_dim", 0)

        extra_sph_degree: Optional[int] = (
            particle_config.get("extra_signal_sph_degree", 0) if extra_signal_dim > 0 else None
        )
        camera_extra_sph_degree: Optional[int] = (
            particle_config.get("camera_extra_signal_sph_degree", 0) if camera_extra_signal_dim > 0 else None
        )
        if extra_sph_degree is not None and camera_extra_sph_degree is not None:
            assert extra_sph_degree == camera_extra_sph_degree, (
                f"GSplat requires a single SH degree for all extra signals, but "
                f"extra_signal_sph_degree={extra_sph_degree} != camera_extra_signal_sph_degree={camera_extra_sph_degree}"
            )
        validated_degree = extra_sph_degree if extra_sph_degree is not None else camera_extra_sph_degree
        # gsplat uses None to mean "no SH evaluation" (degree 0)
        self._extra_signals_sh_degree: Optional[int] = validated_degree if validated_degree else None

        # Lidar extra signal SH degree (read from model config, same as camera path above).
        lidar_extra_signal_dim = particle_config.get("lidar_extra_signal_dim", 0)
        lidar_sph_degree: Optional[int] = (
            particle_config.get("lidar_extra_signal_sph_degree", 0) if lidar_extra_signal_dim > 0 else None
        )
        self._lidar_extra_signals_sh_degree: Optional[int] = lidar_sph_degree if lidar_sph_degree else None

    def render(
        self,
        rendering_data: RenderingData,
        gaussian_parameters: dict[str, torch.Tensor],
        n_active_features: int,
        extra_ray_signal_infos: tuple[list[str], list[int], list[Callable]],
        frame_meta: Optional[List[FrameMeta]] = None,
    ) -> GaussiansRenderReturn:
        """Render Gaussians using gsplat.

        Camera extra signals (common + camera_extra_signal) are supported when
        enable_extended_features and enable_features are True; they are passed to
        gsplat as extra_signals and unpacked into ExtraSignal. Lidar extra signals
        are not yet supported (follow-up).

        Args:
            rendering_data: Ray bundle with camera parameters and poses
            gaussian_parameters: Dict with means, quats, scales, densities, features,
                and optionally extra_signal, camera_extra_signal (for camera path)
            n_active_features: Number of active SH features
            extra_ray_signal_infos: Extra signal (names, dims, activations) for unpacking
            frame_meta: Frame metadata (optional)

        Returns:
            GaussiansRenderReturn with RGB, ray distance, opacity, normals, and
            extra_ray_signals when camera extra is enabled
        """
        assert rendering_data.b == 1, "Only single-frame batch supported"

        with ScopedTimer("GSplatRenderer/render"):
            # Extract sensor parameters and set sensor-specific state (camera vs lidar)
            sensor_params = rendering_data.sensor_model_parameters[0]
            device = rendering_data.rays[0].device
            is_camera = isinstance(sensor_params, get_union_types(ConcreteCameraModelParametersUnion))

            out = self._camera_outputs if is_camera else self._lidar_outputs
            enable_features: bool = out["enable_features"]
            enable_normals: bool = out["enable_normals"]
            enable_extended_features: bool = out["enable_extended_features"]
            enable_sensor_extended_features: bool = out["enable_sensor_extended_features"]

            near_plane = float(self.config.culling.near_clip_distance)
            far_plane = 1e10
            external_distortion_params = None
            viewmat = None
            viewmat_rs = None
            camera_model = None

            if is_camera:
                camera_params = cast(ConcreteCameraModelParametersUnion, sensor_params)
                width = rendering_data.w
                height = rendering_data.h
                # Create camera adapter using polymorphism
                sensor_adapter = SensorModelParametersAdapter.factory(camera_params, device)

                external_distortion_adapter = (
                    ExternalDistortionAdapter.factory(camera_params.external_distortion_parameters, device)
                    if camera_params.external_distortion_parameters is not None
                    else None
                )

                # Get camera model info from adapter
                camera_model = sensor_adapter.get_camera_model_name()
                gsplat_sensor_params = sensor_adapter.get_gsplat_sensor_params()

                external_distortion_params = (
                    external_distortion_adapter.get_distortion_params()
                    if external_distortion_adapter is not None
                    else None
                )

                # Build camera matrices using adapter (start-of-frame pose).
                viewmat = self._build_viewmat(rendering_data, device)

                if self.use_rays:
                    # TODO: add ray footprint support to gsplat
                    # TODO: add support for ray timestamp input to gsplat
                    gsplat_sensor_params["rays"] = rendering_data.rays[0]  # (H, W, 6)
                use_3dgut = self.config.name == "3dgut-gsplat"
                rolling_shutter_type = gsplat.RollingShutterType.GLOBAL
                if self._has_rolling_shutter(camera_params):
                    viewmat_rs = self._build_rolling_shutter_matrix(rendering_data, device)
                    rolling_shutter_type = self._get_rolling_shutter_type(camera_params)

                far_plane = float(self.config.culling.far_clip_distance_camera)

                rasterize_mode = self.config.rasterize_mode

            else:
                lidar_params = cast(ConcreteLidarModelParametersUnion, sensor_params)
                if not isinstance(lidar_params, RowOffsetStructuredSpinningLidarModelParameters):
                    raise NotImplementedError(
                        "GSplatRenderer supports only camera and RowOffsetStructuredSpinningLidarModelParameters lidar."
                    )
                lidar_tiling = self.config.tiling.lidar
                max_pts_per_tile = lidar_tiling.tile_size_elevation * lidar_tiling.tile_size_azimuth
                lidar_coeffs = self._build_lidar_params(
                    lidar_params,
                    device=device,
                    n_bins_elevation=lidar_tiling.n_bins_elevation,
                    max_pts_per_tile=max_pts_per_tile,
                    resolution_elevation=lidar_tiling.resolution_elevation,
                    densification_factor_azimuth=lidar_tiling.densification_factor_azimuth,
                )
                width = lidar_coeffs.n_columns
                height = lidar_coeffs.n_rows

                camera_model = cast(gsplat.CameraModel, "lidar")
                use_3dgut = True
                gsplat_sensor_params = {
                    "Ks": torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                    "lidar_coeffs": lidar_coeffs,
                }

                # For lidar, use mid-frame
                viewmat = self._build_viewmat(rendering_data, device, True)

                viewmat_rs = self._build_rolling_shutter_matrix(rendering_data, device)
                # TODO: this is currently ignored with Lidar, but gsplat expects it as a
                # mandatory parameter. But for giving viewmat_rs (which lidar needs), the
                # shutter type must be different from GLOBAL.
                rolling_shutter_type = gsplat.RollingShutterType.ROLLING_LEFT_TO_RIGHT

                if self.use_rays:
                    # TODO: add ray footprint support to gsplat
                    # TODO: add support for ray timestamp input to gsplat
                    rays = rendering_data.rays[0]  # (H, W, 6)
                    rays = torch.cat(
                        [
                            rays[..., :3],  # origin
                            rays[..., 3:] * self.config.antialiasing.lidar_divergence,  # direction
                        ],
                        dim=-1,
                    )

                    gsplat_sensor_params["rays"] = rays

                far_plane = float(self.config.culling.far_clip_distance_lidar)

                # Lidar must not apply view-dependent compensation to opacity
                rasterize_mode = "classic"

            # Extract Gaussian parameters
            gaussian_params = self._extract_gaussian_parameters(gaussian_parameters, device)

            # Determine rendering mode: RGB-only vs SH
            # RGB-only: n_active_features=0 and colors are (N, 3) - matches nrend's automatic behavior
            # SH mode: n_active_features>0 and colors are (N, K, 3)
            colors_shape = gaussian_params.colors.shape
            is_rgb_only = colors_shape[-1] == 3 and len(colors_shape) == 2  # (N, 3) shape

            # Validate n_active_features against stored features (only for SH mode)
            if enable_features and not is_rgb_only:
                n_stored_bands = colors_shape[1]  # (N, K, 3) -> K bands
                n_requested_bands = (n_active_features + 1) ** 2
                assert n_requested_bands <= n_stored_bands, (
                    f"Feature dimension mismatch: requested sh_degree={n_active_features} "
                    f"({n_requested_bands} bands), but stored features only have {n_stored_bands} bands "
                    f"(sh_degree={int(math.sqrt(n_stored_bands)) - 1}). "
                    f"Cannot use more features than stored in model."
                )

            # Calculate eps2d from min_projected_ray_radius
            # eps2d=0 for lidar (no Mip-Splatting dilation needed).
            proj = self.config.projection
            min_ray_radius = proj.min_projected_ray_radius
            eps2d = 0.0 if not is_camera else min_ray_radius**2

            if use_3dgut:
                ut_params = gsplat.rendering.UnscentedTransformParameters(
                    alpha=float(proj.ut_alpha),
                    beta=float(proj.ut_beta),
                    kappa=float(proj.ut_kappa),
                    in_image_margin_factor=float(proj.image_margin_factor),
                    require_all_sigma_points_valid=proj.ut_require_all_sigma_points,
                )

            else:
                if self.use_rays:
                    raise NotImplementedError("Ray input in 3dGS mode isn't supported")

                if camera_model == "ftheta":
                    raise NotImplementedError("GSplat doesn't support ftheta camera model 3dGS mode")

                if rolling_shutter_type != gsplat.RollingShutterType.GLOBAL:
                    raise NotImplementedError(
                        "GSplat doesn't support rolling shutter type other than GLOBAL in 3dGS mode"
                    )

                # No support for distortion coefficients in 3dGS mode
                gsplat_sensor_params_3dgs = {"Ks": gsplat_sensor_params["Ks"]}
                gsplat_sensor_params = gsplat_sensor_params_3dgs
                external_distortion_params = None

                ut_params = None
                viewmat_rs = None

            # Gradients and memory optimization parameters
            sparse_grad = self.config.sparse_grad
            absgrad = self.config.absgrad
            packed = self.config.packed

            # Validate sparse_grad requires packed mode
            if sparse_grad or packed:
                if use_3dgut:
                    logging.getLogger(__name__).warning(
                        "sparse_grad requested but 3DGUT in gsplat currently doesn't support packed=True. "
                    )
                    sparse_grad = False
                    packed = False

            # Determine render mode based on output config (NRend pattern)
            # - RGB-D: Render RGB and depth (default when features enabled)
            # - D: Render depth only (when features disabled - optimization)
            render_mode: gsplat.RenderMode = "RGB+D" if enable_features else "D"
            if use_3dgut:
                # Returns along-the-ray distance instead of z-depth
                render_mode = "RGB-d" if enable_features else "d"

            global_z_order = not use_3dgut  # True for 3DGS (z-depth), False for 3DGUT (Euclidean distance)

            # Match nrend's automatic RGB vs SH handling:
            # - RGB-only mode (Kelvin/Celsius): sh_degree=None, colors are (N, 3)
            # - SH mode: sh_degree=n_active_features, colors are (N, K, 3)
            sh_degree_param = None if is_rgb_only else (n_active_features if enable_features else None)

            extra_signal_dim = gaussian_parameters["extra_signal"].shape[-1]
            camera_extra_signal_dim = gaussian_parameters["camera_extra_signal"].shape[-1]

            extra_signals = None
            extra_signals_sh_degree = None
            parts = []
            if enable_extended_features and extra_signal_dim > 0:
                parts.append(gaussian_parameters["extra_signal"])
            if enable_sensor_extended_features:
                if is_camera:
                    camera_extra_signal_dim = gaussian_parameters["camera_extra_signal"].shape[-1]
                    if camera_extra_signal_dim > 0:
                        parts.append(gaussian_parameters["camera_extra_signal"])
                else:
                    lidar_extra_signal_dim = gaussian_parameters["lidar_extra_signal"].shape[-1]
                    if lidar_extra_signal_dim > 0:
                        parts.append(gaussian_parameters["lidar_extra_signal"])
            if parts:
                extra_signals = torch.cat(parts, dim=-1).unsqueeze(0)  # [1, N, E]
                if not is_camera:
                    extra_signals_sh_degree = self._lidar_extra_signals_sh_degree
                else:
                    extra_signals_sh_degree = self._extra_signals_sh_degree

            # Call gsplat rasterization
            with ScopedTimer("GSplatRenderer/render/rasterization"):
                render_colors, render_alphas, meta = gsplat.rasterization(
                    means=gaussian_params.means.unsqueeze(0),  # (1, N, 3)
                    quats=gaussian_params.quats.unsqueeze(0),  # (1, N, 4)
                    scales=gaussian_params.scales.unsqueeze(0),  # (1, N, 3)
                    opacities=gaussian_params.opacities.unsqueeze(0),  # (1, N)
                    colors=gaussian_params.colors.unsqueeze(0),
                    viewmats=viewmat.unsqueeze(0).unsqueeze(0),  # (1, 1, 4, 4)
                    width=width,
                    height=height,
                    near_plane=near_plane,
                    far_plane=far_plane,
                    radius_clip=float(self.config.radius_clip),
                    eps2d=eps2d,
                    sh_degree=sh_degree_param,
                    tile_size=self.tile_size,
                    render_mode=render_mode,
                    rasterize_mode=rasterize_mode,
                    camera_model=camera_model,
                    global_z_order=global_z_order,
                    **gsplat_sensor_params,
                    external_distortion_coeffs=external_distortion_params,
                    ut_params=ut_params,
                    rolling_shutter=rolling_shutter_type,
                    viewmats_rs=viewmat_rs.unsqueeze(0).unsqueeze(0) if viewmat_rs is not None else None,
                    # 3DGUT mode: with_ut=True (distortion/RS, NO pose grads)
                    # Standard mode: with_ut=False (pose grads supported)
                    # Use 3dgs-gsplat for pose optimization workflows
                    with_ut=use_3dgut,
                    with_eval3d=use_3dgut,
                    # Compute normals if enabled (requires with_eval3d=True)
                    return_normals=enable_normals and use_3dgut,
                    packed=packed,  # Memory efficient
                    sparse_grad=sparse_grad,  # COO sparse layout gradients (Experimental)
                    absgrad=absgrad,  # Absolute gradients for splitting
                    extra_signals=extra_signals,
                    extra_signals_sh_degree=extra_signals_sh_degree,
                )

                # Store meta dict for densification strategy access (if absgrad enabled)
                # Strategies can access meta["means2d"].absgrad after backward pass
                if absgrad:
                    self.last_rendering_meta = meta

            # Parse outputs
            with ScopedTimer("GSplatRenderer/render/parse_outputs"):
                opacity = render_alphas[0, 0, :, :, 0]  # (H, W)

                # Extract based on render mode
                if enable_features:
                    # RGB+d/D mode: render_colors has [R, G, B, D] in channel dimension
                    rgb = render_colors[0, 0, :, :, :3]  # (H, W, 3)
                    distance = render_colors[0, 0, :, :, 3]  # (H, W) depth or hit distance
                else:
                    # D/d mode: render_colors only has depth
                    rgb = None
                    distance = render_colors[0, 0, :, :, 0]  # (H, W) depth or hit distance

                # Conditionally compute/return outputs based on config (NRend pattern)
                # Check config.outputs.camera.enable_* flags

                # Normals: Extract from meta if computed
                # Normal computation follows nRend: canonical normal (0,0,1) transformed by
                # Gaussian rotation, flipped if facing away from ray
                normal = None
                if enable_normals:
                    render_normals = meta.get("normals")
                    if render_normals is not None:
                        # Shape: (1, 1, H, W, 3) -> (H, W, 3)
                        normal = render_normals[0, 0, :, :, :]

                if extra_signals is not None:  # we passed extra signals to gsplat
                    render_extra_signals = meta.get("render_extra_signals")
                    assert render_extra_signals is not None, "gsplat should have returned meta.render_extra_signals"
                    extra_signals = render_extra_signals[0, 0, :, :, :]  # (H, W, E)

                if extra_signals is not None:
                    E = extra_signals.shape[-1]
                    extra_ray_signals = ExtraSignal.from_packed_tensor(
                        extra_signals.reshape(-1, E), extra_ray_signal_infos
                    )
                else:
                    extra_ray_signals = None

                # Visibility: derive from gsplat radii (>0 means the Gaussian
                # projects into the camera frustum).  Required by losses that
                # set visibility_filter=True (e.g. gaussian_scale/density).
                visibility = None
                radii = meta.get("radii")
                if radii is not None:
                    # radii can be (N,) or (N, 2); reduce to per-Gaussian bool
                    if radii.ndim > 1:
                        radii = radii.amax(dim=-1)
                    # gsplat may return radii as (B, C, N) or (B, C, N, 2) but we
                    # always render single batch / single camera and the losses expect
                    # (N,) visibility so flatten to (N,).
                    visibility = (radii > 0).float()
                    if visibility is not None:
                        visibility = visibility.reshape(-1).contiguous()

                # Saturate RGB output to [0, 1] if configured (matches nrend SaturateRadiance behavior)
                if rgb is not None and self._saturate_radiance:
                    rgb = rgb.clamp(0.0, 1.0)

                return GaussiansRenderReturn(
                    rgb=rgb.reshape(-1, 3) if rgb is not None else None,
                    opacity=opacity.reshape(-1),
                    distance=distance.reshape(-1),
                    normal=normal.reshape(-1, 3) if normal is not None else None,
                    extra_ray_signals=extra_ray_signals,
                    visibility=visibility,
                )

    class _GaussianParameters(NamedTuple):
        """Gaussian parameters extracted for rendering (private to GSplatRenderer).

        Attributes:
            means: Gaussian centers (N, 3)
            quats: Quaternion rotations in wxyz format (N, 4)
            scales: Gaussian scales (N, 3)
            opacities: Gaussian opacities post-sigmoid (N,)
            colors: Spherical harmonic coefficients (N, K, 3)
        """

        means: torch.Tensor
        quats: torch.Tensor
        scales: torch.Tensor
        opacities: torch.Tensor
        colors: torch.Tensor

    def _extract_gaussian_parameters(
        self, gaussian_params: dict[str, torch.Tensor], device: torch.device
    ) -> _GaussianParameters:
        """Extract and prepare Gaussian parameters for gsplat.

        Note: gaussian_params from collect_gaussian_parameters() contains
        post-activated values via get_*() methods:
        - positions: raw (no activation)
        - rotations: rotation_activation applied (default: normalize)
        - scales: scale_activation applied (default: exp)
        - densities: density_activation applied (default: sigmoid)

        Args:
            gaussian_params: Dict with Gaussian parameters
            device: Target device for tensors

        Returns:
            _GaussianParameters named tuple with validated and prepared parameters

        Raises:
            AssertionError: If input shapes are invalid
        """
        # Validate required keys
        required_keys = ["positions", "rotations", "scales", "densities", "features"]
        for key in required_keys:
            assert key in gaussian_params, f"Missing required parameter: {key}"

        # Extract parameters (already post-activated)
        means = gaussian_params["positions"]  # (N, 3) raw positions
        quats = gaussian_params["rotations"]  # (N, 4) wxyz, normalized
        scales = gaussian_params["scales"]  # (N, 3) exp-activated
        opacities = gaussian_params["densities"]  # (N,) or (N, 1) sigmoid-activated
        features = gaussian_params["features"]  # (N, K*3) SH coefficients

        # Validate shapes
        n_gaussians = means.shape[0]
        assert means.ndim == 2 and means.shape[1] == 3, f"Expected positions shape (N, 3), got {means.shape}"
        assert quats.ndim == 2 and quats.shape == (n_gaussians, 4), (
            f"Expected rotations shape ({n_gaussians}, 4), got {quats.shape}"
        )
        assert scales.ndim == 2 and scales.shape == (n_gaussians, 3), (
            f"Expected scales shape ({n_gaussians}, 3), got {scales.shape}"
        )
        # Densities can be (N,) or (N, 1) - squeeze to (N,)
        if opacities.ndim == 2 and opacities.shape[1] == 1:
            opacities = opacities.squeeze(1)
        assert opacities.ndim == 1 and opacities.shape[0] == n_gaussians, (
            f"Expected densities shape ({n_gaussians},) or ({n_gaussians}, 1), got {opacities.shape}"
        )
        assert features.ndim == 2 and features.shape[0] == n_gaussians, (
            f"Expected features shape ({n_gaussians}, K*3), got {features.shape}"
        )
        assert features.shape[1] % 3 == 0, (
            f"Features dimension must be divisible by 3 (SH bands), got {features.shape[1]}"
        )

        # Handle RGB-only vs SH mode to match nrend's automatic behavior
        # RGB-only: features are (N, 3) - pass directly for sh_degree=None
        # SH mode: features are (N, K*3) where K>1 - reshape to (N, K, 3)
        num_sh_bands = features.shape[1] // 3
        if num_sh_bands == 1:
            # RGB-only mode: keep as (N, 3) for gsplat's RGB rendering
            colors = features
        else:
            # SH mode: reshape to (N, K, 3) for gsplat's SH rendering
            colors = features.reshape(n_gaussians, num_sh_bands, 3)

        return self._GaussianParameters(
            means=means.to(device),
            quats=quats.to(device),
            scales=scales.to(device),
            opacities=opacities.to(device),
            colors=colors.to(device),
        )

    def _build_lidar_params(
        self,
        ncore_params: RowOffsetStructuredSpinningLidarModelParameters,
        *,
        device: torch.device,
        n_bins_elevation: int,
        max_pts_per_tile: int,
        resolution_elevation: int,
        densification_factor_azimuth: int,
    ) -> gsplat.RowOffsetStructuredSpinningLidarModelParametersExt:
        """Build gsplat lidar acceleration structures directly from ncore parameters."""

        def _inner(
            gsplat_params: gsplat.RowOffsetStructuredSpinningLidarModelParameters,
            n_bins_elevation: int,
            max_pts_per_tile: int,
            resolution_elevation: int,
            densification_factor_azimuth: int,
        ) -> gsplat.RowOffsetStructuredSpinningLidarModelParametersExt:
            angles_to_columns_map = gsplat.compute_lidar_angles_to_columns_map(gsplat_params)
            tiling = gsplat.compute_lidar_tiling(
                gsplat_params,
                n_bins_elevation=n_bins_elevation,
                max_pts_per_tile=max_pts_per_tile,
                resolution_elevation=resolution_elevation,
                densification_factor_azimuth=densification_factor_azimuth,
            )
            return gsplat.RowOffsetStructuredSpinningLidarModelParametersExt(
                gsplat_params, angles_to_columns_map, tiling
            )

        if not hasattr(GSplatRenderer._build_lidar_params, "_inner"):
            GSplatRenderer._build_lidar_params._inner = lru_cache(maxsize=16)(_inner)  # type: ignore[attr-defined]

        gsplat_spin = (
            gsplat.SpinningDirection.CLOCKWISE
            if ncore_params.spinning_direction == "cw"
            else gsplat.SpinningDirection.COUNTER_CLOCKWISE
        )
        gsplat_params = gsplat.RowOffsetStructuredSpinningLidarModelParameters(
            row_elevations_rad=torch.tensor(ncore_params.row_elevations_rad, dtype=torch.float32, device=device),
            column_azimuths_rad=torch.tensor(ncore_params.column_azimuths_rad, dtype=torch.float32, device=device),
            row_azimuth_offsets_rad=torch.tensor(
                ncore_params.row_azimuth_offsets_rad, dtype=torch.float32, device=device
            ),
            spinning_frequency_hz=ncore_params.spinning_frequency_hz,
            spinning_direction=gsplat_spin,
            fov_eps_factor=4,
        )

        return GSplatRenderer._build_lidar_params._inner(  # type: ignore[attr-defined]
            gsplat_params,
            n_bins_elevation,
            max_pts_per_tile,
            resolution_elevation,
            densification_factor_azimuth,
        )

    def _build_viewmat(
        self, rendering_data: RenderingData, device: torch.device, use_midframe: bool = False
    ) -> torch.Tensor:
        """Build gsplat-compatible viewmat matrix.

        Args:
            rendering_data: Rendering data containing sensor poses
            device: Target device
            use_midframe: If True, interpolate start/end poses at t=0.5 to match
                mid-frame projection pose.

        Returns:
            viewmat: (4, 4) world-to-camera transformation matrix
        """

        if use_midframe:
            pose_start = rendering_data.poses_tquat_startend[0, 0]  # (7,)
            pose_end = rendering_data.poses_tquat_startend[0, 1]  # (7,)
            t_mid = 0.5 * (pose_start[:3] + pose_end[:3])
            q_mid = quat_slerp(pose_start[3:].unsqueeze(0), pose_end[3:].unsqueeze(0), 0.5).squeeze(0)
            pose_tquat = torch.cat([t_mid, q_mid]).unsqueeze(0)  # (1, 7)
        else:
            pose_tquat = rendering_data.poses_tquat_startend[0, 0].unsqueeze(0)  # (1, 7)

        # Convert tquat to inverted 4x4 matrix (world-to-camera)
        viewmat = se3pose_to_inverse_matrix(pose_tquat[..., :3], pose_tquat[..., 3:])

        # Prober integration - save data for testing the viewmat computation
        if (
            prober_result := get_global_prober()(
                0,  # step
                "gsplat_viewmat",
                poses_tquat=pose_tquat,
                viewmat_grad=viewmat,
            )
        ) is not None:
            # Connect the gradient probing to the rest of the computation graph
            (viewmat,) = prober_result

        return viewmat.squeeze(0).to(device)

    def _has_rolling_shutter(self, sensor_params: ConcreteCameraModelParametersUnion) -> bool:
        """Check if sensor has rolling shutter enabled."""
        if hasattr(sensor_params, "shutter_type"):
            return sensor_params.shutter_type != ShutterType.GLOBAL
        return False

    def _get_rolling_shutter_type(self, sensor_params: ConcreteCameraModelParametersUnion) -> gsplat.RollingShutterType:
        """Get rolling shutter type for gsplat.

        Returns:
            gsplat.RollingShutterType enum value
        """
        if not hasattr(sensor_params, "shutter_type"):
            return gsplat.RollingShutterType.GLOBAL

        # Map ncore ShutterType to gsplat RollingShutterType
        shutter_map = {
            ShutterType.GLOBAL: gsplat.RollingShutterType.GLOBAL,
            ShutterType.ROLLING_TOP_TO_BOTTOM: gsplat.RollingShutterType.ROLLING_TOP_TO_BOTTOM,
            ShutterType.ROLLING_LEFT_TO_RIGHT: gsplat.RollingShutterType.ROLLING_LEFT_TO_RIGHT,
            ShutterType.ROLLING_BOTTOM_TO_TOP: gsplat.RollingShutterType.ROLLING_BOTTOM_TO_TOP,
            ShutterType.ROLLING_RIGHT_TO_LEFT: gsplat.RollingShutterType.ROLLING_RIGHT_TO_LEFT,
        }
        return shutter_map.get(sensor_params.shutter_type, gsplat.RollingShutterType.GLOBAL)

    def _build_rolling_shutter_matrix(self, rendering_data: RenderingData, device: torch.device) -> torch.Tensor:
        """Build end viewmat for rolling shutter.

        Returns:
            viewmat_rs: (4, 4) world-to-camera matrix at frame end
        """
        # Extract sensor-to-NRE pose (end)
        pose_tquat_end = rendering_data.poses_tquat_startend[0, 1].unsqueeze(0)  # (1, 7)

        # Convert tquat to inverted 4x4 matrix (world-to-camera)
        viewmat_rs = se3pose_to_inverse_matrix(pose_tquat_end[..., :3], pose_tquat_end[..., 3:])

        return viewmat_rs.squeeze(0).to(device)

    def collect_profilings(self) -> dict[str, float]:
        """Collect profiling information."""
        return {}
