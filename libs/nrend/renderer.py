# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import copy
import importlib
import logging
import os
import warnings

from enum import IntEnum
from functools import lru_cache
from typing import Any, Optional, Tuple

import msgpack
import torch

from libs.vren.interface import (  # type: ignore
    VrenConcreteCameraModelParametersUnion,
    to_vren,
    vren,
)
from libs.vren.lidars import (  # type: ignore
    PreprocessedLidarModel,
    VrenLidarModelParametersUnion,
    preprocess_lidar,
)
from ncore.data import (
    ConcreteCameraModelParametersUnion,
    ConcreteLidarModelParametersUnion,
    FThetaCameraModelParameters,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    RowOffsetStructuredSpinningLidarModelParameters,
)
from nre.utils.profiling import ScopedTimer


# initialize the nrend plugin
_nrend_plugin = None

# libnrend requires cuda CC >= 8.0 - skipping import
if torch.cuda.get_device_capability() >= (8, 0):
    # import tinycudann for settings its rtc_dir
    tcnn_mod = importlib.import_module("tinycudann")

    def import_module_safe(module_name):
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            return None

    # Try loading the NRend Python extension from known module paths
    _nrend_plugin = import_module_safe("libs.nrend.libnrend_cc")
    if _nrend_plugin is None:
        if not torch.cuda.is_available():
            raise EnvironmentError("Unknown compute capability. Ensure PyTorch with CUDA support is installed.")

        warnings.warn("NRend plugin libnrend_cc not found. Ensure it is on PYTHONPATH.")

    def _set_rtc_directories():
        if _nrend_plugin is None:
            return

        # adding tinycudann directory
        if tcnn_mod is not None:
            rtc_dir = os.path.join(os.path.dirname(tcnn_mod.__file__), "rtc")
            include_dir = os.path.join(rtc_dir, "include")
            if os.path.isdir(include_dir):
                _nrend_plugin.set_rtc_include_dir(include_dir, False, False)
            else:
                warnings.warn(f"tiny-cuda-nn rtc include directory not found at {include_dir}, skipping")
        # adding optix jit headers directory
        try:
            # try bazel runfiles API first
            from python.runfiles import runfiles

            r = runfiles.Create()
            assert r is not None, "Runfiles API not available."

            optix_include_dir = r.Rlocation("optix_dev/optix-dev-9.0.0/include/optix_device.h")
            if optix_include_dir is not None:
                optix_include_dir = os.path.dirname(optix_include_dir)
        except (ImportError, ModuleNotFoundError, AttributeError, TypeError, AssertionError):
            # Fallback to non-bazel case
            optix_include_dir = os.path.join("..", "optix_dev/optix-dev-9.0.0/include")

        assert optix_include_dir, "OptiX include directory not found."

        _nrend_plugin.set_rtc_include_dir(str(os.path.abspath(optix_include_dir)), True, False)

        # base jit directory
        extra_rtc_dir = os.path.join(os.path.dirname(__file__), "rtc")
        if not os.path.isdir(extra_rtc_dir):
            extra_rtc_dir = os.path.dirname(__file__)
        assert os.path.isdir(extra_rtc_dir)

        # adding nrend potentially obfuscated headers extra directory
        extra_include_dir = str(os.path.join(extra_rtc_dir, "obfuscated/include"))
        if not os.path.exists(extra_include_dir):  # try the un-obfuscated version
            extra_include_dir = str(os.path.join(extra_rtc_dir, "include"))
        assert os.path.isdir(extra_include_dir)
        _nrend_plugin.set_rtc_include_dir(extra_include_dir, False, True)

        # adding the nrend jit cache directory
        cache_base = os.environ.get("TEST_TMPDIR") or os.path.join(os.path.expanduser("~"), ".cache")
        nrend_cache_dir = os.path.join(cache_base, "nrend", "ptx_cache")
        os.makedirs(nrend_cache_dir, exist_ok=True)
        _nrend_plugin.set_rtc_cache_dir(str(nrend_cache_dir))

    _set_rtc_directories()


def assert_on_device(tensor: Optional[torch.Tensor], device: torch.device) -> None:
    if tensor is None:
        return
    if tensor.device != device:
        raise ValueError(f"Expected {tensor.device=} to be equal to {device=}")


class Renderer:
    """Expose NRendererWrapper"""

    class Hint(IntEnum):
        DEFAULT = 0
        FASTEST = 1
        FAST = 2
        QUALITY = 3
        HIGHEST_QUALITY = 4
        FAST_QUALITY = 5
        QUALITY_FAST = 6

    class LogLevel(IntEnum):
        FATAL = 0
        ERROR = 1
        WARNING = 2
        INFO = 3
        DEBUG = 4

    @staticmethod
    def logging_level(log_level: LogLevel):
        match log_level:
            case Renderer.LogLevel.DEBUG:
                return logging.DEBUG
            case Renderer.LogLevel.INFO:
                return logging.INFO
            case Renderer.LogLevel.WARNING:
                return logging.WARNING
            case Renderer.LogLevel.ERROR:
                return logging.ERROR
            case _:
                return logging.FATAL

    _cached_model_state_dict: Optional[dict[str, torch.Tensor]]

    def _setup_nrenderer_wrapper(
        self,
        model: dict[str, Any],
        render_settings: dict[str, Any] | Hint = Hint.DEFAULT,
        track_instances_uid_map: list[str] = [],
        log_level: LogLevel = LogLevel.ERROR,
        profiling_frequency: float = 0.0,
        differentiable: bool = False,
    ):
        assert _nrend_plugin, f"NRend library not initialized."

        self.logger = logging.getLogger("NREND")
        self.logger.setLevel(Renderer.logging_level(log_level))
        if log_level >= Renderer.LogLevel.DEBUG:
            printable_model = copy.deepcopy(model)
            if "nre_data" in printable_model and "state_dict" in printable_model["nre_data"]:
                printable_model["nre_data"]["state_dict"] = printable_model["nre_data"]["state_dict"].keys()
            self.logger.debug(f"{'DIFFERENTIABLE' if differentiable else ''} MODEL : {printable_model}")
            self.logger.debug(f"RENDER SETTINGS : {render_settings}")
            self.logger.debug(f"TRACK INSTANCES UID MAP : {track_instances_uid_map}")

        has_render_settings = isinstance(render_settings, dict)

        # extract rendering flags from render_settings (settings not read by NRend)
        outputs_settings = (render_settings.get("outputs") or {}) if has_render_settings else {}  # type: ignore
        compute_normals = True
        scene_settings = outputs_settings.get("scene") or {}
        compute_scene_cumulated_weights = scene_settings.get("enable_cumulated_weights", False)
        compute_scene_visibility = scene_settings.get("enable_visibility", False)

        # Store scene data configuration for parsing
        self._scene_data_config = {
            "enable_cumulated_weights": compute_scene_cumulated_weights,
            "enable_visibility": compute_scene_visibility,
        }

        self._nrenderer_wrapper = _nrend_plugin.NRendererWrapper(
            msgpack.packb(model),
            msgpack.packb(render_settings if has_render_settings else {}),
            track_instances_uid_map,
            Renderer.Hint.DEFAULT if has_render_settings else render_settings,
            int(log_level),
            profiling_frequency,
            differentiable,
            compute_normals,
            compute_scene_cumulated_weights,
            compute_scene_visibility,
        )

        self._cached_model_state_dict = None

    @ScopedTimer("NRenderer.init")
    def __init__(
        self,
        model: dict[str, Any],
        render_settings: dict[str, Any] | Hint = Hint.DEFAULT,
        track_instances_uid_map: list[str] = [],
        log_level: LogLevel = LogLevel.ERROR,
        profiling_frequency: float = 0.0,
    ):
        """
        Initialize the engine from a model bytes data

        Args:
          model : dictionary containing the serialized model data
          render_settings : dictionary containing the renderer settings
          track_instances_uid : list of track instance uid to be rendered (N_tracks [str])
          log_level : [0=FATAL,1=ERROR, 2=WARNING, 3=INFO, 4=DEBUG]
        """
        self._setup_nrenderer_wrapper(
            model,
            render_settings,
            track_instances_uid_map,
            log_level,
            profiling_frequency,
            differentiable=False,
        )

    def valid(self) -> bool:
        return self._nrenderer_wrapper.valid()

    @ScopedTimer("NRenderer.render")
    def render(
        self,
        frame_id: int,
        frame_width: int,
        frame_height: int,
        frame_start_timestamp: int,
        frame_end_timestamp: int,
        rays_origin: torch.Tensor,
        rays_direction: torch.Tensor,
        rays_timestamp: Optional[torch.Tensor] = None,
        frames_sensor_model: Optional[vren.CameraModelParameters | VrenLidarModelParametersUnion] = None,
        frames_sensor_ids: Optional[torch.Tensor] = None,
        frames_sensor_start_pose: Optional[torch.Tensor] = None,
        frames_sensor_end_pose: Optional[torch.Tensor] = None,
        num_active_track_instances: int = 0,
        active_track_instances_ids: Optional[torch.Tensor] = None,
        active_track_instances_start_pose: Optional[torch.Tensor] = None,
        active_track_instances_end_pose: Optional[torch.Tensor] = None,
        tile: Optional[Tuple[int, int, int, int]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Render a view

        Args:
          frame_id : unique frame identifier
          frame_width : width of the frame
          frame_height : height of the frame
          frame_start_timestamp: timestamp of the frame at the begining of the capture
          frame_end_timestamp: timestamp of the frame at the end of the capture
          rays_origin : contiguous float tensor containing the 3d position of the rays origin [HxWx3]
          rays_direction : contiguous float tensor containing the 3d rays direction (norm is the ray spread) [HxWx3]
          rays_timestamp : contiguous float tensor containing the rays timestamp (in [frame_start_timestamp, frame_end_timestamp]) [HxWx1]
          frames_sensor_model : variant describing the sensor model
          frames_sensor_ids : contiguous int tensor containing the frame sensor ids (sensor id, sensor start frame id, sensor end frame id) [2]
          frames_sensor_start_pose : contiguous float tensor containing  position and rotation (quaternion) of the frame sensor at frame_start_timestamp [7]
          frames_sensor_end_pose : contiguous float tensor containing  position and rotation (quaternion) of the frame sensor at frame_end_timestamp [7]
          num_active_track_instances : number of active instances for the current frame
          active_track_instances_ids : contiguous int tensor containing the num_active_tracks map idx (into the initialized track_ids) and instance ids of the active tracks [num_active_tracksx2]
          active_track_instances_start_pose : contiguous float tensor containing  position and rotation (quaternion) of the active tracks at frame_start_timestamp [num_active_tracksx7]
          active_track_instances_end_pose : contiguous float tensor containing position and rotation (quaternion) of the active tracks at frame_end_timestamp [num_active_tracksx7]
          tile : 2D offset and 2D size of the tile (or crop) to be render, if None the full frame is rendered

        Returns:
          rays_radiance_density : HW4 float tensor containing the RGB radiance and the density
          rays_hit_distance : HW1 float tensor containing the final hit distance of the ray
          rays_hit_normal : HW3 float tensor containing the final hit normal of the ray
          rays_extra_signals : HWN float tensor containing the extra signals (N varies according to the model)
          scene_data : ExD float tensor containing the scene data (E=num elements, D=data dim per element)
        """
        assert _nrend_plugin, f"NRend library not initialized."

        # check that all input tensors are on the same device
        device = rays_origin.device
        assert_on_device(rays_direction, device)
        assert_on_device(rays_timestamp, device)
        assert_on_device(frames_sensor_ids, device)
        assert_on_device(frames_sensor_start_pose, device)
        assert_on_device(frames_sensor_end_pose, device)
        assert_on_device(active_track_instances_ids, device)
        assert_on_device(active_track_instances_start_pose, device)
        assert_on_device(active_track_instances_end_pose, device)

        rays_radiance_density, rays_hit_distance, rays_hit_normal, rays_extra_signals, scene_data, _, success = (
            self._nrenderer_wrapper.render(
                frame_id,
                frame_width,
                frame_height,
                (0, 0, frame_width, frame_height) if tile is None else tile,
                frame_start_timestamp,
                frame_end_timestamp,
                rays_origin,
                rays_direction,
                (rays_timestamp if rays_timestamp is not None else torch.empty((0,), dtype=torch.int64)),
                frames_sensor_model if frames_sensor_model else _nrend_plugin.NRendererSensorProjectionModel(),
                (frames_sensor_ids if frames_sensor_ids is not None else torch.empty((0,), dtype=torch.int32)),
                (frames_sensor_start_pose if frames_sensor_start_pose is not None else torch.empty((0,))),
                (frames_sensor_end_pose if frames_sensor_end_pose is not None else torch.empty((0,))),
                num_active_track_instances,
                (active_track_instances_ids if active_track_instances_ids is not None else torch.empty((0,))),
                (
                    active_track_instances_start_pose
                    if active_track_instances_start_pose is not None
                    else torch.empty((0,))
                ),
                (active_track_instances_end_pose if active_track_instances_end_pose is not None else torch.empty((0,))),
            )
        )
        assert success, "NRenderer.render failed."
        return rays_radiance_density, rays_hit_distance, rays_hit_normal, rays_extra_signals, scene_data

    def parse_scene_data(self, scene_data_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Parse a raw scene_data tensor according to the renderer's configuration.

        Args:
            scene_data_tensor: Raw scene data tensor of shape (n_elements, n_features)

        Returns:
            Dictionary with keys "cumulated_weights" and/or "visibility" mapping to tensors
        """
        result: dict[str, torch.Tensor] = {}

        if scene_data_tensor.numel() == 0:
            return result

        col_idx = 0

        if self._scene_data_config["enable_cumulated_weights"]:
            result["cumulated_weights"] = scene_data_tensor[:, col_idx]
            col_idx += 1

        if self._scene_data_config["enable_visibility"]:
            result["visibility"] = scene_data_tensor[:, col_idx]
            col_idx += 1

        return result

    @ScopedTimer("NRender.update_model_parameters")
    def update_model_parameters(self, parameters: dict[str, torch.Tensor], deep_copy: bool = False):
        """
        Update the model parameters tensors

        Args:
          parameters: dictionnary containing the model parameters to update, indexed by their key in the state dictionnary
          deep_copy: if true  : the parameters are copied into internal host buffer
                     if false : the parameters are only attached. In other words a reference to the input buffers is used and the client is responsible of maintaining its availability
        """
        self._cached_model_state_dict = parameters
        success = self._nrenderer_wrapper.update_model_parameters(parameters, deep_copy)
        assert success, "NRenderer.update_model_parameters failed."

    def collect_profilings(self) -> dict[str, float]:
        """
        Collect all profilings
        """
        return self._nrenderer_wrapper.collect_profilings()

    @ScopedTimer("NRenderer.prepare_and_cache_camera_model")
    @staticmethod
    def prepare_and_cache_camera_model(
        ncore_camera_model_parameters_outer: ConcreteCameraModelParametersUnion, device: torch.device
    ) -> VrenConcreteCameraModelParametersUnion:
        def inner(
            ncore_camera_model_parameters_json: str, ncore_camera_model_parameter_type: str, device: torch.device
        ) -> tuple[VrenConcreteCameraModelParametersUnion, torch.Tensor | None, torch.Tensor | None]:
            # deserialize NCore camera model parameters
            ncore_camera_model_parameters: ConcreteCameraModelParametersUnion
            if ncore_camera_model_parameter_type == FThetaCameraModelParameters.type():
                ncore_camera_model_parameters = FThetaCameraModelParameters.from_json(
                    ncore_camera_model_parameters_json
                )
            elif ncore_camera_model_parameter_type == OpenCVPinholeCameraModelParameters.type():
                ncore_camera_model_parameters = OpenCVPinholeCameraModelParameters.from_json(
                    ncore_camera_model_parameters_json
                )
            elif ncore_camera_model_parameter_type == OpenCVFisheyeCameraModelParameters.type():
                ncore_camera_model_parameters = OpenCVFisheyeCameraModelParameters.from_json(
                    ncore_camera_model_parameters_json
                )
            else:
                raise Exception(
                    f"Unsupported camera model type: {ncore_camera_model_parameter_type} (extend enumeration here?)"
                )

            # Convert ncore to vren camera model parameters
            vren_camera_model_parameters = to_vren(ncore_camera_model_parameters)

            # Handle special case of attached external distortions if necessary
            if (external_distortion_parameters := ncore_camera_model_parameters.external_distortion_parameters) is None:
                return vren_camera_model_parameters, None, None

            # Note: this is hardcoding the parameters for the single BivariateWindshieldModelParameters external distortion type currently
            # - needs to be extended once there are more types of external distortion
            if external_distortion_parameters.type() != "bivariate-windshield":
                raise Exception(f"Unsupported external distortion type: {external_distortion_parameters.type()}")

            vren_camera_model_parameters.external_distortion_parameters.preprocess_ws_paramters(
                horizontal_poly_torch := torch.tensor(
                    external_distortion_parameters.horizontal_poly,
                    device=device,
                    dtype=torch.float32,
                ),
                vertical_poly_torch := torch.tensor(
                    external_distortion_parameters.vertical_poly,
                    device=device,
                    dtype=torch.float32,
                ),
            )

            # Make sure to return vren model parameters as well as the tensors that the model references
            return vren_camera_model_parameters, horizontal_poly_torch, vertical_poly_torch

        # LRU cache is used to hold onto the horizontal_poly_torch and vertical_poly_torch tensors of the external distortion, if present,
        # since their data is pointed-to and used by the sensor_parameters_data.external_distortion_parameters when a windshield model is present.
        # Note: The lru_cache caches all the return variables of "inner"
        if not hasattr(Renderer.prepare_and_cache_camera_model, "inner"):
            Renderer.prepare_and_cache_camera_model.inner = lru_cache(maxsize=16)(  # type: ignore[attr-defined]
                inner
            )

        vren_sensor_parameters_data, _, _ = Renderer.prepare_and_cache_camera_model.inner(  # type: ignore[attr-defined]
            # serialize camera model parameters into a hash-able form
            ncore_camera_model_parameters_outer.to_json(),
            ncore_camera_model_parameters_outer.type(),
            device,
        )

        return vren_sensor_parameters_data

    @ScopedTimer("NRenderer.prepare_and_cache_lidar_model")
    @staticmethod
    def prepare_and_cache_lidar_model(
        ncore_lidar_model_parameters_outer: ConcreteLidarModelParametersUnion,
        *,
        n_bins_elevation: int,
        max_pts_per_tile: int,  # Must match the renderer's configuration
        resolution_elevation: int = 1600,
        densification_factor_azimuth: int = 8,
        device: torch.device,
    ) -> PreprocessedLidarModel:
        def inner(
            ncore_lidar_model_parameters_json: str, ncore_lidar_model_parameter_type: str, device: torch.device
        ) -> PreprocessedLidarModel:
            # deserialize NCore lidar model parameters
            ncore_lidar_model_parameters: ConcreteLidarModelParametersUnion
            if ncore_lidar_model_parameter_type == RowOffsetStructuredSpinningLidarModelParameters.type():
                ncore_lidar_model_parameters = RowOffsetStructuredSpinningLidarModelParameters.from_json(
                    ncore_lidar_model_parameters_json
                )
            else:
                raise Exception(
                    f"Unsupported lidar model type: {ncore_lidar_model_parameter_type} (extend enumeration here?)"
                )
            return preprocess_lidar(
                ncore_lidar_model_parameters,
                n_bins_elevation=n_bins_elevation,
                max_pts_per_tile=max_pts_per_tile,
                resolution_elevation=resolution_elevation,
                densification_factor_azimuth=densification_factor_azimuth,
                device=device,
            )

        if not hasattr(Renderer.prepare_and_cache_lidar_model, "inner"):
            Renderer.prepare_and_cache_lidar_model.inner = lru_cache(maxsize=16)(inner)  # type: ignore[attr-defined]

        return Renderer.prepare_and_cache_lidar_model.inner(  # type: ignore[attr-defined]
            # serialize lidar model parameters into a hash-able form
            ncore_lidar_model_parameters_outer.to_json(),
            ncore_lidar_model_parameters_outer.type(),
            device,
        )


class DifferentiableRenderer(Renderer):
    """Expose NRendererWrapper differentiable interface"""

    class _CkptFriendlyAutoGrad(torch.autograd.Function):
        """
        - Store all heavy tensors not directly under ctx attributes, but under saved_tensors,
        this is helpful using non-reentrant checkpointing so that forward ctx-attribute would not carry too much data.
        - Since we cannot serialize forward_context, we recompute it in the backward pass.
        """

        @ScopedTimer("DifferentiableRenderer._CkptFriendlyAutoGrad.forward")
        @staticmethod
        def forward(
            ctx,
            rays_origin: torch.Tensor,
            rays_direction: torch.Tensor,
            model_state_dict_keys: list[str],
            render_undifferentiable_parameters: tuple,
            *model_state_dict_tensors,
        ):
            renderer_wrapper = render_undifferentiable_parameters[0]
            forward_model_state_dict = render_undifferentiable_parameters[16]
            rays_timestamp = render_undifferentiable_parameters[6]

            # This dict is put directly under ctx so be mindful it should be small.
            small_undifferentiable_parameters_dict = {
                "frame_id": render_undifferentiable_parameters[1],
                "frame_width": render_undifferentiable_parameters[2],
                "frame_height": render_undifferentiable_parameters[3],
                "tile": render_undifferentiable_parameters[15],
                "frame_start_timestamp": render_undifferentiable_parameters[4],
                "frame_end_timestamp": render_undifferentiable_parameters[5],
                "frames_sensor_model": render_undifferentiable_parameters[7],
                "frames_sensor_ids": render_undifferentiable_parameters[8],
                "frames_sensor_start_pose": render_undifferentiable_parameters[9],
                "frames_sensor_end_pose": render_undifferentiable_parameters[10],
                "num_active_track_instances": render_undifferentiable_parameters[11],
                "active_track_instances_ids": render_undifferentiable_parameters[12],
                "active_track_instances_start_pose": render_undifferentiable_parameters[13],
                "active_track_instances_end_pose": render_undifferentiable_parameters[14],
            }

            (
                rays_radiance_density,
                rays_hit_distance,
                rays_hit_normal,
                rays_extra_signals,
                scene_data,
                _,
                success,
            ) = renderer_wrapper.render(
                small_undifferentiable_parameters_dict["frame_id"],
                small_undifferentiable_parameters_dict["frame_width"],
                small_undifferentiable_parameters_dict["frame_height"],
                small_undifferentiable_parameters_dict["tile"],
                small_undifferentiable_parameters_dict["frame_start_timestamp"],
                small_undifferentiable_parameters_dict["frame_end_timestamp"],
                rays_origin,
                rays_direction,
                rays_timestamp,
                small_undifferentiable_parameters_dict["frames_sensor_model"],
                small_undifferentiable_parameters_dict["frames_sensor_ids"],
                small_undifferentiable_parameters_dict["frames_sensor_start_pose"],
                small_undifferentiable_parameters_dict["frames_sensor_end_pose"],
                small_undifferentiable_parameters_dict["num_active_track_instances"],
                small_undifferentiable_parameters_dict["active_track_instances_ids"],
                small_undifferentiable_parameters_dict["active_track_instances_start_pose"],
                small_undifferentiable_parameters_dict["active_track_instances_end_pose"],
            )
            assert success, "DifferentiableNRenderer.render forward failed."

            ctx.renderer_wrapper = renderer_wrapper
            ctx.small_undifferentiable_parameters_dict = small_undifferentiable_parameters_dict
            ctx.model_state_dict_keys = model_state_dict_keys

            if forward_model_state_dict is not None:
                ctx.forward_model_state_dict_keys, forward_model_state_dict_values = zip(
                    *forward_model_state_dict.items()
                )
            else:
                ctx.forward_model_state_dict_keys, forward_model_state_dict_values = (), ()

            ctx.save_for_backward(
                *tuple(
                    [model_state_dict_tensors[k] for k in range(len(model_state_dict_keys))]
                    + [
                        rays_origin,
                        rays_direction,
                        rays_timestamp,
                        *forward_model_state_dict_values,
                    ]
                )
            )
            ctx.mark_non_differentiable(scene_data)
            return rays_radiance_density, rays_hit_distance, rays_hit_normal, rays_extra_signals, scene_data

        @ScopedTimer("DifferentiableRenderer._CkptFriendlyAutoGrad.backward")
        @staticmethod
        def backward(
            ctx,
            rays_radiance_density_grad,
            rays_hit_distance_grad,
            rays_hit_normal_grad,
            rays_extra_signals_grad,
            _scene_data_grad,
        ):
            # unpack hook is called
            ctx_saved_tensors = ctx.saved_tensors

            # differentiable state dict
            renderer_wrapper = ctx.renderer_wrapper
            num_model_state_dict_keys = len(ctx.model_state_dict_keys)
            model_state_dict = {
                ctx.model_state_dict_keys[k]: ctx_saved_tensors[k] for k in range(len(ctx.model_state_dict_keys))
            }

            # non-differentiable state dict
            (
                rays_origin,
                rays_direction,
                rays_timestamp,
                *forward_model_state_dict_values,
            ) = ctx_saved_tensors[num_model_state_dict_keys:]

            # Reset the model parameters to the forward values
            if ctx.forward_model_state_dict_keys:
                renderer_wrapper.update_model_parameters(
                    {
                        ctx.forward_model_state_dict_keys[k]: forward_model_state_dict_values[k]
                        for k in range(len(ctx.forward_model_state_dict_keys))
                    },
                    False,
                )

            small_undifferentiable_parameters_dict = ctx.small_undifferentiable_parameters_dict

            # Forward pass, mainly to obtain the forward_ctx
            (
                rays_radiance_density,
                rays_hit_distance,
                rays_hit_normal,
                rays_extra_signals,
                scene_data,
                forward_ctx,
                success,
            ) = renderer_wrapper.render(
                small_undifferentiable_parameters_dict["frame_id"],
                small_undifferentiable_parameters_dict["frame_width"],
                small_undifferentiable_parameters_dict["frame_height"],
                small_undifferentiable_parameters_dict["tile"],
                small_undifferentiable_parameters_dict["frame_start_timestamp"],
                small_undifferentiable_parameters_dict["frame_end_timestamp"],
                rays_origin,
                rays_direction,
                rays_timestamp,
                small_undifferentiable_parameters_dict["frames_sensor_model"],
                small_undifferentiable_parameters_dict["frames_sensor_ids"],
                small_undifferentiable_parameters_dict["frames_sensor_start_pose"],
                small_undifferentiable_parameters_dict["frames_sensor_end_pose"],
                small_undifferentiable_parameters_dict["num_active_track_instances"],
                small_undifferentiable_parameters_dict["active_track_instances_ids"],
                small_undifferentiable_parameters_dict["active_track_instances_start_pose"],
                small_undifferentiable_parameters_dict["active_track_instances_end_pose"],
            )
            assert success, "DifferentiableNRenderer.render re-forward in backward failed."

            # Backward pass
            (
                rays_origin_grad,
                rays_direction_grad,
                model_state_dict_grad,
                success,
            ) = renderer_wrapper.render_backward(
                small_undifferentiable_parameters_dict["frame_id"],
                small_undifferentiable_parameters_dict["frame_width"],
                small_undifferentiable_parameters_dict["frame_height"],
                small_undifferentiable_parameters_dict["tile"],
                small_undifferentiable_parameters_dict["frame_start_timestamp"],
                small_undifferentiable_parameters_dict["frame_end_timestamp"],
                rays_origin,
                rays_direction,
                model_state_dict,
                rays_timestamp,
                small_undifferentiable_parameters_dict["frames_sensor_model"],
                small_undifferentiable_parameters_dict["frames_sensor_ids"],
                small_undifferentiable_parameters_dict["frames_sensor_start_pose"],
                small_undifferentiable_parameters_dict["frames_sensor_end_pose"],
                small_undifferentiable_parameters_dict["num_active_track_instances"],
                small_undifferentiable_parameters_dict["active_track_instances_ids"],
                small_undifferentiable_parameters_dict["active_track_instances_start_pose"],
                small_undifferentiable_parameters_dict["active_track_instances_end_pose"],
                rays_radiance_density,
                rays_radiance_density_grad,
                rays_hit_distance,
                rays_hit_distance_grad,
                rays_hit_normal,
                rays_hit_normal_grad,
                rays_extra_signals,
                rays_extra_signals_grad,
                forward_ctx,
            )
            assert success, "DifferentiableNRenderer.render backward failed."
            forward_ctx.reset()
            return tuple(
                [rays_origin_grad, rays_direction_grad, None, None]
                + [model_state_dict_grad[k] for k in ctx.model_state_dict_keys]
            )

    class _AutoGrad(torch.autograd.Function):
        @ScopedTimer("DifferentiableRenderer._AutoGrad.forward")
        @staticmethod
        def forward(
            ctx,
            rays_origin: torch.Tensor,
            rays_direction: torch.Tensor,
            model_state_dict_keys: list[str],
            render_undifferentiable_parameters: tuple,
            *model_state_dict_tensors,
        ):
            (
                rays_radiance_density,
                rays_hit_distance,
                rays_hit_normal,
                rays_extra_signals,
                scene_data,
                render_forward_ctx,
                success,
            ) = render_undifferentiable_parameters[0].render(
                render_undifferentiable_parameters[1],  # frame_id
                render_undifferentiable_parameters[2],  # frame_width
                render_undifferentiable_parameters[3],  # frame_height
                render_undifferentiable_parameters[15],  # tile
                render_undifferentiable_parameters[4],  # frame_start_timestamp
                render_undifferentiable_parameters[5],  # frame_end_timestamp
                rays_origin,
                rays_direction,
                render_undifferentiable_parameters[6],  # rays_timestamp
                render_undifferentiable_parameters[7],  # frames_sensor_model
                render_undifferentiable_parameters[8],  # frames_sensor_ids
                render_undifferentiable_parameters[9],  # frames_sensor_start_pose
                render_undifferentiable_parameters[10],  # frames_sensor_end_pose
                render_undifferentiable_parameters[11],  # num_active_track_instances
                render_undifferentiable_parameters[12],  # active_track_instances_ids
                render_undifferentiable_parameters[13],  # active_track_instances_start_pose
                render_undifferentiable_parameters[14],  # active_track_instances_end_pose
            )
            assert success, "DifferentiableNRenderer.render forward failed."
            ctx.render_undifferentiable_parameters = render_undifferentiable_parameters
            ctx.model_state_dict_keys = model_state_dict_keys
            ctx.render_forward_ctx = render_forward_ctx
            ctx.save_for_backward(
                *tuple(
                    [model_state_dict_tensors[k] for k in range(len(model_state_dict_keys))]
                    + [
                        rays_origin,
                        rays_direction,
                        rays_radiance_density,
                        rays_hit_distance,
                        rays_hit_normal,
                        rays_extra_signals,
                    ]
                )
            )
            # scene_data is not differentiable, mark it for autograd
            ctx.mark_non_differentiable(scene_data)
            return rays_radiance_density, rays_hit_distance, rays_hit_normal, rays_extra_signals, scene_data

        @ScopedTimer("DifferentiableRenderer._AutoGrad.backward")
        @staticmethod
        def backward(
            ctx,
            rays_radiance_density_grad,
            rays_hit_distance_grad,
            rays_hit_normal_grad,
            rays_extra_signals_grad,
            _scene_data_grad,
        ):
            renderer_wrapper = ctx.render_undifferentiable_parameters[0]

            # Reset the model parameters to the forward values
            forward_model_state_dict = ctx.render_undifferentiable_parameters[16]
            if forward_model_state_dict is not None:
                renderer_wrapper.update_model_parameters(forward_model_state_dict, False)

            num_model_state_dict_keys = len(ctx.model_state_dict_keys)
            model_state_dict = {
                ctx.model_state_dict_keys[k]: ctx.saved_tensors[k] for k in range(len(ctx.model_state_dict_keys))
            }
            (
                rays_origin_grad,
                rays_direction_grad,
                model_state_dict_grad,
                success,
            ) = renderer_wrapper.render_backward(
                ctx.render_undifferentiable_parameters[1],  # frame_id
                ctx.render_undifferentiable_parameters[2],  # frame_width
                ctx.render_undifferentiable_parameters[3],  # frame_height
                ctx.render_undifferentiable_parameters[15],  # tile
                ctx.render_undifferentiable_parameters[4],  # frame_start_timestamp
                ctx.render_undifferentiable_parameters[5],  # frame_end_timestamp
                ctx.saved_tensors[num_model_state_dict_keys + 0],  # rays_origin
                ctx.saved_tensors[num_model_state_dict_keys + 1],  # rays_direction
                model_state_dict,
                ctx.render_undifferentiable_parameters[6],  # rays_timestamp
                ctx.render_undifferentiable_parameters[7],  # frames_sensor_model
                ctx.render_undifferentiable_parameters[8],  # frames_sensor_ids
                ctx.render_undifferentiable_parameters[9],  # frames_sensor_start_pose
                ctx.render_undifferentiable_parameters[10],  # frames_sensor_end_pose
                ctx.render_undifferentiable_parameters[11],  # num_active_track_instances
                ctx.render_undifferentiable_parameters[12],  # active_track_instances_ids
                ctx.render_undifferentiable_parameters[13],  # active_track_instances_start_pose
                ctx.render_undifferentiable_parameters[14],  # active_track_instances_end_pose
                ctx.saved_tensors[num_model_state_dict_keys + 2],  # rays_radiance_density
                rays_radiance_density_grad,
                ctx.saved_tensors[num_model_state_dict_keys + 3],  # rays_hit_distance
                rays_hit_distance_grad,
                ctx.saved_tensors[num_model_state_dict_keys + 4],  # rays_hit_normal
                rays_hit_normal_grad,
                ctx.saved_tensors[num_model_state_dict_keys + 5],  # rays_extra_signals
                rays_extra_signals_grad,
                ctx.render_forward_ctx,
            )
            assert success, "DifferentiableNRenderer.render backward failed."
            ctx.render_forward_ctx.reset()
            return tuple(
                [rays_origin_grad, rays_direction_grad, None, None]
                + [model_state_dict_grad[k] for k in ctx.model_state_dict_keys]
            )

    class _PrepareSceneAutoGrad(torch.autograd.Function):
        @ScopedTimer("DifferentiableRenderer._PrepareSceneAutoGrad.forward")
        @staticmethod
        def forward(
            ctx,
            model_state_dict_keys: list[str],
            render_undifferentiable_parameters: tuple,
            *model_state_dict_tensors,
        ):
            (
                scene_density,
                scene_features,
                scene_extended_features,
                scene_sensor_extended_features,
                scene_data,
                forward_ctx,
                success,
            ) = render_undifferentiable_parameters[0].prepare_scene(
                render_undifferentiable_parameters[1],  # frame_id
                render_undifferentiable_parameters[2],  # frame_width
                render_undifferentiable_parameters[3],  # frame_height
                render_undifferentiable_parameters[15],  # tile
                render_undifferentiable_parameters[4],  # frame_start_timestamp
                render_undifferentiable_parameters[5],  # frame_end_timestamp
                render_undifferentiable_parameters[7],  # frames_sensor_model
                render_undifferentiable_parameters[9],  # frames_sensor_start_pose
                render_undifferentiable_parameters[10],  # frames_sensor_end_pose
                render_undifferentiable_parameters[11],  # num_active_track_instances
                render_undifferentiable_parameters[12],  # active_track_instances_ids
                render_undifferentiable_parameters[13],  # active_track_instances_start_pose
                render_undifferentiable_parameters[14],  # active_track_instances_end_pose
            )
            assert success, "DifferentiableNRenderer.prepare_scene forward failed."
            ctx.render_undifferentiable_parameters = render_undifferentiable_parameters
            ctx.model_state_dict_keys = model_state_dict_keys
            ctx.forward_ctx = forward_ctx
            ctx.save_for_backward(
                *tuple(
                    [model_state_dict_tensors[k] for k in range(len(model_state_dict_keys))]
                    + [
                        scene_features,
                        scene_extended_features,
                        scene_sensor_extended_features,
                    ]
                )
            )
            return scene_density, scene_features, scene_extended_features, scene_sensor_extended_features, scene_data

        @ScopedTimer("DifferentiableRenderer._PrepareSceneAutoGrad.backward")
        @staticmethod
        def backward(
            ctx,
            scene_density_grad,
            scene_features_grad,
            scene_extended_features_grad,
            scene_sensor_extended_features_grad,
            scene_data_grad,  # unused, scene_data is not differentiable
        ):
            renderer_wrapper = ctx.render_undifferentiable_parameters[0]

            # Reset the model parameters to the forward values
            forward_model_state_dict = ctx.render_undifferentiable_parameters[16]
            if forward_model_state_dict is not None:
                renderer_wrapper.update_model_parameters(forward_model_state_dict, False)

            num_model_state_dict_keys = len(ctx.model_state_dict_keys)
            model_state_dict = {
                ctx.model_state_dict_keys[k]: ctx.saved_tensors[k] for k in range(len(ctx.model_state_dict_keys))
            }
            (
                model_state_dict_grad,
                success,
            ) = renderer_wrapper.prepare_scene_backward(
                ctx.render_undifferentiable_parameters[1],  # frame_id
                ctx.render_undifferentiable_parameters[2],  # frame_width
                ctx.render_undifferentiable_parameters[3],  # frame_height
                ctx.render_undifferentiable_parameters[15],  # tile
                ctx.render_undifferentiable_parameters[4],  # frame_start_timestamp
                ctx.render_undifferentiable_parameters[5],  # frame_end_timestamp
                model_state_dict,
                ctx.render_undifferentiable_parameters[7],  # frames_sensor_model
                ctx.render_undifferentiable_parameters[9],  # frames_sensor_start_pose
                ctx.render_undifferentiable_parameters[10],  # frames_sensor_end_pose
                ctx.render_undifferentiable_parameters[11],  # num_active_track_instances
                ctx.render_undifferentiable_parameters[12],  # active_track_instances_ids
                ctx.render_undifferentiable_parameters[13],  # active_track_instances_start_pose
                ctx.render_undifferentiable_parameters[14],  # active_track_instances_end_pose
                scene_density_grad,
                ctx.saved_tensors[num_model_state_dict_keys + 0],  # scene_features
                scene_features_grad,
                ctx.saved_tensors[num_model_state_dict_keys + 1],  # scene_extended_features
                scene_extended_features_grad,
                ctx.saved_tensors[num_model_state_dict_keys + 2],  # scene_sensor_extended_features
                scene_sensor_extended_features_grad,
                ctx.forward_ctx,
            )
            assert success, "DifferentiableNRenderer.prepare_scene backward failed."
            ctx.forward_ctx.reset()
            return tuple([None, None] + [model_state_dict_grad[k] for k in ctx.model_state_dict_keys])

    @ScopedTimer("NDiffRenderer.init")
    def __init__(
        self,
        model,
        render_settings: dict[str, Any] | None = None,
        track_instances_uid_map: list[str] = [],
        log_level: Renderer.LogLevel = Renderer.LogLevel.ERROR,
        profiling_frequency: float = 0.0,
    ):
        if render_settings is None:
            render_settings = {}
        super()._setup_nrenderer_wrapper(
            model,
            render_settings,
            track_instances_uid_map,
            log_level,
            profiling_frequency,
            differentiable=True,
        )

    @ScopedTimer("NDiffRenderer.render")
    def render(
        self,
        frame_id: int,
        frame_width: int,
        frame_height: int,
        frame_start_timestamp: int,
        frame_end_timestamp: int,
        rays_origin: torch.Tensor,
        rays_direction: torch.Tensor,
        rays_timestamp: Optional[torch.Tensor] = None,
        frames_sensor_model: Optional[vren.CameraModelParameters | VrenLidarModelParametersUnion] = None,
        frames_sensor_ids: Optional[torch.Tensor] = None,
        frames_sensor_start_pose: Optional[torch.Tensor] = None,
        frames_sensor_end_pose: Optional[torch.Tensor] = None,
        num_active_track_instances: int = 0,
        active_track_instances_ids: Optional[torch.Tensor] = None,
        active_track_instances_start_pose: Optional[torch.Tensor] = None,
        active_track_instances_end_pose: Optional[torch.Tensor] = None,
        tile: Optional[Tuple[int, int, int, int]] = None,
        prepare_scene: bool = False,
        checkpoint_friendly_backward: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Differentially render a view wrt the rays origin and direction and all model parameters requiring gradients that are currently attached through update_model

        Args:
          NO_DIFF frame_id : unique frame identifier
          NO_DIFF frame_width : width of the frame
          NO_DIFF frame_height : height of the frame
          NO_DIFF frame_start_timestamp: timestamp of the frame at the begining of the capture
          NO_DIFF frame_end_timestamp: timestamp of the frame at the end of the capture
          rays_origin : contiguous float tensor containing the 3d position of the rays origin [HxWx3]
          rays_direction : contiguous float tensor containing the 3d normalized rays direction [HxWx3]
          NO_DIFF rays_timestamp : contiguous float tensor containing the rays timestamp (in [frame_start_timestamp, frame_end_timestamp]) [HxWx1]
          NO_DIFF frames_sensor_model : variant describing the sensor model
          NO_DIFF frames_sensor_ids : contiguous int tensor containing the frame sensor ids (sensor id, sensor start frame id, sensor end frame id) [2]
          NO_DIFF frames_sensor_start_pose : contiguous float tensor containing  position and rotation (quaternion) of the frame sensor at frame_start_timestamp [7]
          NO_DIFF frames_sensor_end_pose : contiguous float tensor containing  position and rotation (quaternion) of the frame sensor at frame_end_timestamp [7]
          NO_DIFF num_active_track_instances : number of active instances for the current frame
          NO_DIFF active_track_instances_ids : contiguous int tensor containing the num_active_tracks map idx (into the initialized track_ids) and instance ids of the active tracks [num_active_tracksx2]
          NO_DIFF active_track_instances_start_pose : contiguous float tensor containing  position and rotation (quaternion) of the active tracks at frame_start_timestamp [num_active_tracksx7]
          NO_DIFF active_track_instances_end_pose : contiguous float tensor containing position and rotation (quaternion) of the active tracks at frame_end_timestamp [num_active_tracksx7]
          NO_DIFF tile : 2D offset and 2D size of the tile (or crop) to be render, if None the full frame is rendered
          NO_DIFF prepare_scene : if true, the scene is prepared for distributed scene rendering
          NO_DIFF checkpoint_friendly_backward : if true, use an alternative implementation of the render autograd function.

        Returns:
          rays_radiance_density : HW4 float tensor containing the RGB radiance and the density
          rays_hit_distance : HW1 float tensor containing the final hit distance of the ray
          rays_hit_normal : HW3 float tensor containing the final hit normal of the ray
          rays_extra_signals : HWN float tensor containing the extra signals (N varies according to the model)
        """
        assert _nrend_plugin, f"NRend library not initialized."

        differentiable_parameters_keys = (
            [state_dict_key for state_dict_key, tensor in self._cached_model_state_dict.items() if tensor.requires_grad]
            if self._cached_model_state_dict is not None
            else []
        )
        differentiable_parameters_tensors = (
            [self._cached_model_state_dict[state_dict_key] for state_dict_key in differentiable_parameters_keys]
            if self._cached_model_state_dict is not None
            else []
        )
        undifferentiable_args = (
            self._nrenderer_wrapper,
            frame_id,
            frame_width,
            frame_height,
            frame_start_timestamp,
            frame_end_timestamp,
            (
                rays_timestamp
                if rays_timestamp is not None
                else torch.empty((0,), device=rays_origin.device, dtype=torch.int64)
            ),
            frames_sensor_model if frames_sensor_model else _nrend_plugin.NRendererSensorProjectionModel(),
            (
                frames_sensor_ids
                if frames_sensor_ids is not None
                else torch.empty((0,), device=rays_origin.device, dtype=torch.int32)
            ),
            (frames_sensor_start_pose if frames_sensor_start_pose is not None else torch.empty((0,))),
            (frames_sensor_end_pose if frames_sensor_end_pose is not None else torch.empty((0,))),
            num_active_track_instances,
            (
                active_track_instances_ids
                if active_track_instances_ids is not None
                else torch.empty((0,), device=rays_origin.device)
            ),
            (
                active_track_instances_start_pose
                if active_track_instances_start_pose is not None
                else torch.empty((0,), device=rays_origin.device)
            ),
            (
                active_track_instances_end_pose
                if active_track_instances_end_pose is not None
                else torch.empty((0,), device=rays_origin.device)
            ),
            (0, 0, frame_width, frame_height) if tile is None else tile,
            self._cached_model_state_dict.copy() if self._cached_model_state_dict is not None else None,
        )

        if prepare_scene:
            return DifferentiableRenderer._PrepareSceneAutoGrad.apply(
                differentiable_parameters_keys, undifferentiable_args, *differentiable_parameters_tensors
            )
        else:
            functor = (
                DifferentiableRenderer._CkptFriendlyAutoGrad
                if checkpoint_friendly_backward
                else DifferentiableRenderer._AutoGrad
            )
            return functor.apply(
                rays_origin,
                rays_direction,
                differentiable_parameters_keys,
                undifferentiable_args,
                *differentiable_parameters_tensors,
            )
