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

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Tuple, Type

import lietorch as lt
import torch

from omegaconf import DictConfig
from torch import nn

import nre.models.gaussians.collect as collect

from libs.gaussian_mcmc.interface import gaussian_mcmc  # type: ignore
from ncore.data import RowOffsetStructuredSpinningLidarModelParameters
from ncore.sensors import RowOffsetStructuredSpinningLidarModel
from nre.config.model import (
    DeformableGaussiansLayerConfig,
    ElasticLoadedAHLayerConfig,
    LayerConfigType,
    RigidGaussiansLayerConfig,
    SHGaussiansLayerConfig,
)
from nre.config.trainer import TrainerConfig
from nre.datasets.summary import DataSourceSummary
from nre.datasets.tracks import CuboidTracks
from nre.models.base import BaseModel
from nre.models.composite import LayerTrackIds
from nre.models.gaussians.initializations import BaseInitialization
from nre.models.gaussians.utils import (
    Asset,
    sh_degree_to_specular_dim,
    write_ply_3dgrt,
    write_ply_3dgs,
)
from nre.models.input_embedding import (
    BaseInputEmbedding,
    HolisticRemapTimeInputEmbedding,
    IndividualRemapTimeInputEmbedding,
    IndividualStepTimeInputEmbedding,
)
from nre.models.object_feature_volume import HashGridObjectFeatureVolume
from nre.models.tracks_calib import BaseTracksCalib, CompositeTracksCalib, DirectTracksCalib
from nre.models.utils import ExpActivation, concat_rays_timestamps, get_activation, update_module_step
from nre.utils.batch import DataAndRenderingBatch, RenderingData
from nre.utils.geometry import quat_mult_xyzw, quat_to_so3_matrix
from nre.utils.misc import all_gather_tensor_list, unpack_optional
from nre.utils.optim import OptimizerLRSchedulerConfig, configure_optimizers
from nre.utils.prober import get_global_prober
from nre.utils.profiling import ScopedTimer
from nre.utils.trainer import adjust_step_for_world_size
from nre.utils.types import AABB3D, CuboidTracksData, ModelInput, SceneContractor, TracksData


log = logging.getLogger(__name__)
prober = get_global_prober()


class GaussianExportFormat(str, Enum):
    _3DGS = "3dgs"
    _3DGRT = "3dgrt"


@lru_cache(maxsize=16)
def compute_stable_merging_indices(full_length: int, world_size: int, device: torch.device) -> torch.Tensor:
    # Because the gaussian parameters are distributed as [world_rank::world_size], we might want to re-order the
    # parameters back to the original order after all-gather to produce a consistent state_dict. This is done by
    # creating a mapping from the original indices to the all-gathered indices.
    full_indices = torch.arange(full_length, device=device)
    local_indices = []
    for rank in range(world_size):
        local_indices.append(full_indices[rank::world_size])
    merged_indices = torch.cat(local_indices)
    return torch.argsort(merged_indices)


@ScopedTimer("distributed_all_gather_gaussian_parameters")
def distributed_all_gather_gaussian_parameters(
    local_parameters: dict[str, torch.Tensor], keys: list[str] | None = None, stable: bool = False
) -> dict[str, torch.Tensor]:
    # If distributed training is not initialized, or the local parameters are empty, we can just return the local parameters
    if not local_parameters or not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return local_parameters

    keys = list(local_parameters.keys()) if keys is None else keys
    assert len(keys) > 0, "keys must be non-empty"

    world_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # reshape all parameters to [num_gaussians, D] and record the shapes
    length = local_parameters[keys[0]].shape[0]
    for key in keys:
        assert local_parameters[key].shape[0] == length, f"{key} {local_parameters[key].shape[0]} != {length}"
    shapes = [local_parameters[key].shape for key in keys]
    dtypes = [local_parameters[key].dtype for key in keys]

    # Handle empty case - compute the feature dimension explicitly
    if length == 0:
        # For empty tensors, calculate feature_dim from the original shape
        # shape is [0, d1, d2, ...], so feature_dim is d1 * d2 * ...
        params = []
        for key in keys:
            feature_dim = 1
            for dim in local_parameters[key].shape[1:]:
                feature_dim *= dim
            params.append(local_parameters[key].reshape(0, feature_dim))
    else:
        params = [local_parameters[key].reshape(length, -1) for key in keys]  # local parameters after reshape

    # All gather the parameters from all ranks. `parameters_all_ranks` is a list of tensors,
    # each tensor in it is a GS attribute with the shape [all_gaussians, D]
    full_params = all_gather_tensor_list(world_size, params)
    full_length = len(full_params[0])

    # recover the original shape of the parameters, and overwrite the parameter dict.
    full_parameters = local_parameters
    for key, shape, dtype, param in zip(keys, shapes, dtypes, full_params):
        full_parameters[key] = param.reshape(full_length, *shape[1:]).to(dtype)

    if stable:
        indices = compute_stable_merging_indices(full_length, world_size, torch.device("cuda"))
        for key in keys:
            full_parameters[key] = full_parameters[key][indices, ...]

    return full_parameters


class BaseGaussianModel(BaseModel, ABC):
    # Core Gaussian parameters
    # Positions of the 3D Gaussians (x, y, z) [n_gaussians, 3]
    positions: nn.Parameter | nn.UninitializedParameter
    # Rotation of each Gaussian represented as a unit quaternion [n_gaussians, 4]
    rotations: nn.Parameter | nn.UninitializedParameter
    # Anisotropic scale of each Gaussian [n_gaussians, 3]
    scales: nn.Parameter | nn.UninitializedParameter
    # Density of each Gaussian [n_gaussians, 1]
    densities: nn.Parameter | nn.UninitializedParameter

    # Extra signal parameters (sensor-specific)
    # Common extra signals for both camera and lidar [n_gaussians, extra_signal_dim]
    extra_signal: nn.Parameter | nn.UninitializedParameter
    # Camera-specific extra signals [n_gaussians, camera_extra_signal_dim]
    camera_extra_signal: nn.Parameter | nn.UninitializedParameter
    # Lidar-specific extra signals [n_gaussians, lidar_extra_signal_dim]
    lidar_extra_signal: nn.Parameter | nn.UninitializedParameter

    scene_extent: float

    GAUSSIANS_VARIANTS: dict[str, Type[BaseGaussianModel]] = {}

    config: SHGaussiansLayerConfig  # type: ignore[assignment] # All gaussian models have SH config fields

    @staticmethod
    def register_to_gaussians_factory(name: str, cls: Type[BaseGaussianModel]) -> None:
        if name in BaseGaussianModel.GAUSSIANS_VARIANTS:
            raise KeyError(f"{name=} already in GAUSSIANS_VARIANTS.")
        BaseGaussianModel.GAUSSIANS_VARIANTS[name] = cls

    # Currently we only have one variant, but we anticipate exploring more in the future
    @staticmethod
    def factory(
        name: str,
        config: SHGaussiansLayerConfig,
        trainer_config: TrainerConfig,
        datasource: DataSourceSummary,
        init_from_datasource: bool,
        initializer: Optional[BaseInitialization] = None,
        precision: Optional[int] = None,
        cuboid_tracks: Optional[CuboidTracks] = None,
        start_end_timestamp_us: Optional[Tuple[int, int]] = None,
    ) -> BaseGaussianModel:
        return BaseGaussianModel.GAUSSIANS_VARIANTS[name](
            config=config,
            trainer_config=trainer_config,
            datasource=datasource,
            init_from_datasource=init_from_datasource,
            initializer=initializer,
            precision=precision,
            cuboid_tracks=cuboid_tracks,
            start_end_timestamp_us=start_end_timestamp_us,
        )

    def __init__(
        self,
        config: SHGaussiansLayerConfig,
        trainer_config: TrainerConfig,
        datasource: DataSourceSummary,
        init_from_datasource: bool,
        initializer: Optional[BaseInitialization] = None,
        precision: Optional[int] = None,
        cuboid_tracks: Optional[CuboidTracks] = None,
        start_end_timestamp_us: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__(config=config.to_dictconfig())
        self.config = config  # type: ignore[assignment]  # Override with typed config
        self.trainer_config = trainer_config

        self.density_activation = get_activation(self.config.density_activation)
        self.density_activation_inv = get_activation(self.config.density_activation, inverse=True)
        self.scale_activation = get_activation(self.config.scale_activation)
        self.scale_activation_inv = get_activation(self.config.scale_activation, inverse=True)
        self.rotation_activation = get_activation(self.config.rotation_activation)
        self.scene_extent = datasource.get_aabb().get_extent().max().item()

        # The packing process creates mapping dictionaries that specify how signals are
        # arranged in the parameter tensors, enabling efficient rendering for different
        # sensor types.

        # Step 1: Initialize packing info dictionaries for Gaussian-level signals
        # These track how signals are packed and activated within each Gaussian's parameter tensors
        self.extra_signal_param_infos: tuple[dict[str, int], list[int]] = (
            {},
            [],
        )  # Common signals: ({signal_name: index}, [signal_dims])
        self.camera_extra_signal_param_infos: tuple[dict[str, int], list[int]] = (
            {},
            [],
        )  # Camera signals: ({signal_name: index}, [signal_dims])
        self.lidar_extra_signal_param_infos: tuple[dict[str, int], list[int]] = (
            {},
            [],
        )  # Lidar signals: ({signal_name: index}, [signal_dims])

        extra_signal_configs = self.config.extra_signal
        if extra_signal_configs is not None:
            # Track index for each signal type
            start_index = 0
            camera_start_index = 0
            lidar_start_index = 0

            # First pass: Pack signals into their respective parameter tensors
            for signal_name, params in extra_signal_configs.items():
                signal_dim = params["n_signal_dim"]
                sensor_type = params["sensor_type"]
                signal_activation = get_activation(params["activation"])

                if sensor_type == "common":
                    # Common signals are stored in the extra_signal parameter tensor
                    self.extra_signal_param_infos[0][str(signal_name)] = start_index
                    self.extra_signal_param_infos[1].append(signal_dim)
                    start_index += 1
                elif sensor_type == "camera":
                    # Camera-specific signals are stored in the camera_extra_signal parameter tensor
                    self.camera_extra_signal_param_infos[0][str(signal_name)] = camera_start_index
                    self.camera_extra_signal_param_infos[1].append(signal_dim)
                    camera_start_index += 1
                elif sensor_type == "lidar":
                    # Lidar-specific signals are stored in the lidar_extra_signal parameter tensor
                    self.lidar_extra_signal_param_infos[0][str(signal_name)] = lidar_start_index
                    self.lidar_extra_signal_param_infos[1].append(signal_dim)
                    lidar_start_index += 1
                else:
                    raise ValueError(f"Invalid sensor_type '{sensor_type}' for signal '{signal_name}'. ")

        # Step 2: Create ray-level signal packing info for rendering
        # These dictionaries specify how signals are arranged in the final rendered output
        # for each sensor type. This is different from Gaussian-level packing because:
        # - Common signals are duplicated for both camera and lidar rendering
        # - Sensor-specific signals are only included in their respective outputs
        self.camera_extra_ray_signal_infos: tuple[list[str], list[int], list[Callable]] = ([], [], [])
        self.lidar_extra_ray_signal_infos: tuple[list[str], list[int], list[Callable]] = ([], [], [])

        if extra_signal_configs is not None:
            # First pass: Add common signals to both camera and lidar ray outputs
            for signal_name, params in extra_signal_configs.items():
                if params["sensor_type"] == "common":
                    signal_dim = params["n_signal_dim"]
                    signal_activation = get_activation(params["activation"])
                    # Common signals appear in both camera and lidar ray outputs
                    self.camera_extra_ray_signal_infos[0].append(str(signal_name))
                    self.camera_extra_ray_signal_infos[1].append(signal_dim)
                    self.camera_extra_ray_signal_infos[2].append(signal_activation)

                    self.lidar_extra_ray_signal_infos[0].append(str(signal_name))
                    self.lidar_extra_ray_signal_infos[1].append(signal_dim)
                    self.lidar_extra_ray_signal_infos[2].append(signal_activation)

            # Second pass: Add sensor-specific signals to their respective ray outputs
            for signal_name, params in extra_signal_configs.items():
                sensor_type = params["sensor_type"]
                signal_dim = params["n_signal_dim"]
                signal_activation = get_activation(params["activation"])
                if sensor_type == "camera":
                    # Camera-specific signals only appear in camera ray outputs
                    self.camera_extra_ray_signal_infos[0].append(str(signal_name))
                    self.camera_extra_ray_signal_infos[1].append(signal_dim)
                    self.camera_extra_ray_signal_infos[2].append(signal_activation)
                elif sensor_type == "lidar":
                    # Lidar-specific signals only appear in lidar ray outputs
                    self.lidar_extra_ray_signal_infos[0].append(str(signal_name))
                    self.lidar_extra_ray_signal_infos[1].append(signal_dim)
                    self.lidar_extra_ray_signal_infos[2].append(signal_activation)

        # We need to wait for the datasource and torch.distributed context to be initialized. We will re-initializing
        # particles from datasource and validate the model's parameters later
        self.positions = nn.UninitializedParameter()
        self.rotations = nn.UninitializedParameter()
        self.scales = nn.UninitializedParameter()
        self.densities = nn.UninitializedParameter()
        self.extra_signal = nn.UninitializedParameter()
        self.camera_extra_signal = nn.UninitializedParameter()
        self.lidar_extra_signal = nn.UninitializedParameter()

        # Defer initialization of Gaussians buffers and validate the model's parameters.
        def _maybe_initialize_gaussians():
            if init_from_datasource:
                assert initializer is not None and datasource is not None, (
                    f"{self.__class__.__name__}: initializer and datasource must be provided when init_from_datasource is True"
                )
                self.initialize_gaussians_from_datasource(
                    initializer=unpack_optional(initializer),
                    datasource=datasource,
                    cuboid_tracks=cuboid_tracks,
                )
                self._validate_fields()

        self._maybe_initialize_gaussians = _maybe_initialize_gaussians

        # Misc
        self.overriden_parameters: Dict[str, nn.Parameter] = {}
        self.overriden_buffers: Dict[str, nn.Buffer] = {}
        self.overriden_state: Dict[str, Any] = {}

    def maybe_initialize_gaussians(self) -> None:
        self._maybe_initialize_gaussians()

    def _get_slice_boundaries(self) -> tuple[int, int]:
        # In distributed training, each rank only hosts a subset of Gaussians.
        if self.trainer_config.world_size > 1:
            assert torch.distributed.is_initialized()
            assert torch.distributed.get_world_size() == self.trainer_config.world_size, (
                f"{torch.distributed.get_world_size()} != {self.trainer_config.world_size}"
            )
            world_rank = torch.distributed.get_rank()
            world_size = self.trainer_config.world_size
        else:
            world_rank = 0
            world_size = 1
        return world_rank, world_size

    def configure_sharded_params_and_buffers(self) -> list[str]:
        ret = super().configure_sharded_params_and_buffers()
        BaseModel.mark_as_sharded(self.positions)
        BaseModel.mark_as_sharded(self.rotations)
        BaseModel.mark_as_sharded(self.scales)
        BaseModel.mark_as_sharded(self.densities)
        BaseModel.mark_as_sharded(self.extra_signal)
        BaseModel.mark_as_sharded(self.camera_extra_signal)
        BaseModel.mark_as_sharded(self.lidar_extra_signal)
        return ret + [
            "positions",
            "rotations",
            "scales",
            "densities",
            "extra_signal",
            "camera_extra_signal",
            "lidar_extra_signal",
        ]

    def initialize_gaussians_from_datasource(
        self,
        initializer: BaseInitialization,
        datasource: DataSourceSummary,
        cuboid_tracks: Optional[CuboidTracks] = None,
    ) -> None:
        with ScopedTimer(f"BaseGaussianModel/{initializer.__class__.__name__}/initialize_from_datasource"):
            # Because initializer is not attached to any module, we need to manually move it to the correct device.
            initializer.to(self.device).initialize_from_datasource(datasource, cuboid_tracks=cuboid_tracks)

        world_rank, world_size = self._get_slice_boundaries()

        self.positions = nn.Parameter(initializer.positions[world_rank::world_size].contiguous())
        self.rotations = nn.Parameter(initializer.rotations[world_rank::world_size].contiguous())
        self.scales = nn.Parameter(initializer.scales[world_rank::world_size].contiguous())
        self.densities = nn.Parameter(initializer.densities[world_rank::world_size].contiguous())
        self.extra_signal = nn.Parameter(
            torch.rand(
                self.positions.shape[0],
                self.config.particle.extra_signal_dim,
                dtype=self.positions.dtype,
                device=self.device,
            )
        )
        self.camera_extra_signal = nn.Parameter(
            torch.rand(
                self.positions.shape[0],
                self.config.particle.camera_extra_signal_dim,
                dtype=self.positions.dtype,
                device=self.device,
            )
        )
        self.lidar_extra_signal = nn.Parameter(
            torch.rand(
                self.positions.shape[0],
                self.config.particle.lidar_extra_signal_dim * self.get_lidar_extra_signal_sph_dim(),
                dtype=self.positions.dtype,
                device=self.device,
            )
        )

    def configure_optimizers(self, name_prefix: str = "") -> list[OptimizerLRSchedulerConfig]:
        """
        Configure the optimizers of the Gaussian model. We keep a reference to the optimizers such that the strategies
        can modify them, but also pass it to the global optimizers to perform steps/backwards.
        """

        optimizers = configure_optimizers(self.config.to_dictconfig(), self.trainer_config, self, name_prefix)

        for opt in optimizers:
            for param_group in opt["optimizer"].param_groups:
                if "positions" in param_group["name"] and self.config.scale_pos_lr_by_scene_extent:
                    param_group["lr"] *= 0.5 * self.scene_extent  # Multiply the position lr by the scene scale
                    if "lr_scheduler" in opt:
                        opt["lr_scheduler"]["scheduler"].base_lrs[0] *= 0.5 * self.scene_extent

        # Make a reference to the optimizer such that the strategies will be able to modify it
        self.optimizers = optimizers

        readable = []
        for entry in optimizers:
            optimizer = entry["optimizer"]
            opt_class = optimizer.__class__.__name__
            group_names = [pg.get("name", f"group{idx}") for idx, pg in enumerate(optimizer.param_groups)]
            readable.append(f"{opt_class}[" + ", ".join(map(str, group_names)) + "]")
        log.info(f"Create optimizer for {self.__class__.__name__}: " + "; ".join(readable))

        return optimizers

    def _validate_fields(self) -> None:
        """
        Validate that all the per-Gaussian fields have the correct shape
        """
        num_gaussians = self.get_num_gaussians()
        assert self.positions.shape == (num_gaussians, 3)
        assert self.densities.shape == (num_gaussians, 1)
        assert self.rotations.shape == (num_gaussians, 4)
        assert self.scales.shape == (num_gaussians, 3)
        assert self.extra_signal.shape == (
            num_gaussians,
            self.config.particle.extra_signal_dim,
        )
        assert self.camera_extra_signal.shape == (
            num_gaussians,
            self.config.particle.camera_extra_signal_dim,
        )
        assert self.lidar_extra_signal.shape == (
            num_gaussians,
            self.config.particle.lidar_extra_signal_dim * self.get_lidar_extra_signal_sph_dim(),
        )

    #
    # Implementation of what is necessary for parameter collection.
    #
    @staticmethod
    def get_frame_timestamp(rendering_data: RenderingData) -> int:
        assert rendering_data.timestamps_startend_us_cpu is not None
        assert rendering_data.timestamps_startend_us_cpu.shape == (1, 2)
        assert rendering_data.timestamps_startend_us_cpu.dtype == torch.int64
        return int(
            (
                rendering_data.timestamps_startend_us_cpu[0, 0].item()
                + rendering_data.timestamps_startend_us_cpu[0, 1].item()
            )
            // 2
        )

    @dataclass(slots=True)
    class CollectionContext:
        rendering_data: RenderingData
        is_training_batch: bool
        tracks_edit: Optional[LayerTrackIds.Edit]

        additional_data: dict[str, Tuple[torch.Tensor, bool]] = field(default_factory=dict)

    def get_parameters(self) -> dict[str, torch.Tensor]:
        return {
            "positions": self.positions,
            "rotations": self.rotations,
            "scales": self.scales,
            "densities": self.densities,
            "extra_signal": self.extra_signal,
            "camera_extra_signal": self.camera_extra_signal,
            "lidar_extra_signal": self.lidar_extra_signal,
        }

    def get_layer_config(self) -> collect.LayerConfigBase:
        if self.config.rotation_activation == "normalize":
            rotation_activation = collect.RotationActivation.NORMALIZE
        else:
            raise NotImplementedError(f"Invalid rotation activation: {self.config.rotation_activation}")

        if self.config.scale_activation == "exp":
            scale_activation = collect.ScaleActivation.EXP
        else:
            raise NotImplementedError(f"Invalid scale activation: {self.config.scale_activation}")

        if self.config.density_activation == "sigmoid":
            density_activation = collect.DensityActivation.SIGMOID
        else:
            raise NotImplementedError(f"Invalid density activation: {self.config.density_activation}")

        return collect.LayerConfigBase(
            rotation_activation=rotation_activation,
            scale_activation=scale_activation,
            density_activation=density_activation,
        )

    def get_layer_data(
        self, context: CollectionContext, gathered_parameters: dict[str, torch.Tensor]
    ) -> collect.LayerDataBase:
        return collect.LayerDataBase(
            positions=gathered_parameters["positions"],
            rotations=gathered_parameters["rotations"],
            scales=gathered_parameters["scales"],
            densities=gathered_parameters["densities"],
            extra_signal=gathered_parameters["extra_signal"],
            camera_extra_signal=gathered_parameters["camera_extra_signal"],
            lidar_extra_signal=gathered_parameters["lidar_extra_signal"],
        )

    def post_process(self, offset: int, results: collect.CollectorResult, layer_data: collect.LayerDataBase) -> None:
        pass

    #
    # End of implementation of what is necessary for parameter collection.
    #

    def get_scales(self, preactivation=False) -> torch.Tensor:
        assert isinstance(self.scales, nn.Parameter), "should be initialized"
        if preactivation:
            return self.scales
        else:
            return self.scale_activation(self.scales)

    # Most of the NRE codebase assumes quaternions are in xyzw format, but the NRend 3DGUT rasterizer
    # assumes wxyz format as in the original 3DGS implementation
    def get_rotations(self, quaternion_format: Literal["xyzw", "wxyz"], preactivation=False) -> torch.Tensor:
        assert isinstance(self.rotations, nn.Parameter), "should be initialized"
        if preactivation:
            rotations = self.rotations
        else:
            rotations = self.rotation_activation(self.rotations)

        match quaternion_format:
            case "xyzw":
                # Rotations are stored internally in wxyz format, since NRend uses this convention and directly
                # reads the state_dict during inference mode
                xyz = rotations[:, 1:4]
                w = rotations[:, 0:1]
                return torch.cat([xyz, w], dim=-1)
            case "wxyz":
                return rotations
            case _:
                raise NotImplementedError(f"quaternion_format {quaternion_format} not supported")

    def get_positions(self) -> nn.Parameter:
        assert isinstance(self.positions, nn.Parameter), "should be initialized"
        return self.positions

    def get_densities(self, preactivation=False) -> nn.Parameter:
        assert isinstance(self.densities, nn.Parameter), "should be initialized"
        if preactivation:
            return self.densities
        else:
            return self.density_activation(self.densities)

    def get_covariance(self) -> torch.Tensor:
        # The quaternions are stored in the wxyz format so we can avoid the conversion on pytorch side
        quaternion_format: Literal["xyzw", "wxyz"] = "wxyz"

        scales = self.get_scales()
        quats = self.get_rotations(quaternion_format=quaternion_format)

        return gaussian_mcmc.quat_scale_to_covariance(quats, scales, quaternion_format)

    def get_lidar_extra_signal_sph_dim(self) -> int:
        return (self.config.particle.lidar_extra_signal_sph_degree + 1) ** 2

    def get_extra_signal(self, sensor_type: str = "common") -> torch.Tensor:
        if sensor_type == "common":
            return self.extra_signal
        elif sensor_type == "camera":
            return self.camera_extra_signal
        elif sensor_type == "lidar":
            return self.lidar_extra_signal
        else:
            raise ValueError(f"Invalid sensor type: {sensor_type}")

    def get_extra_signal_by_key(self, key: str) -> torch.Tensor:
        if key in self.extra_signal_param_infos[0]:
            assert isinstance(self.extra_signal, nn.Parameter), "extra_signal should be initialized"
            return torch.split(self.extra_signal, self.extra_signal_param_infos[1], dim=-1)[
                self.extra_signal_param_infos[0][key]
            ]
        elif key in self.camera_extra_signal_param_infos[0]:
            assert isinstance(self.camera_extra_signal, nn.Parameter), "camera_extra_signal should be initialized"
            return torch.split(self.camera_extra_signal, self.camera_extra_signal_param_infos[1], dim=-1)[
                self.camera_extra_signal_param_infos[0][key]
            ]
        elif key in self.lidar_extra_signal_param_infos[0]:
            assert isinstance(self.lidar_extra_signal, nn.Parameter), "lidar_extra_signal should be initialized"
            return torch.split(self.lidar_extra_signal, self.lidar_extra_signal_param_infos[1], dim=-1)[
                self.lidar_extra_signal_param_infos[0][key]
            ]
        else:
            raise ValueError(f"Invalid signal key: {key}")

    def get_num_gaussians(self) -> int:
        return self.positions.shape[0]

    def get_additional_buffers(self) -> dict[str, torch.Tensor]:
        return {}

    def remove_gaussians_in_trajectories(
        self, trajectories: List[torch.Tensor], cuboid_dims: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Removes gaussians along trajectories within the specified cuboid dims.
        Note that the override is only meant to be used at inference time and should be reverted
        by calling restore_training_parameters before resuming training.
        Args:
            trajectories: List of N trajectories. Each trajectory has a variable number of 4x4 poses
            cuboid_dims: List of N cuboid dims. Gaussians within cuboid_dim of each trajectory pose will be removed
        Return:
            Mask indicating which of the original gaussians remain.
        """
        include_mask = torch.ones_like(self.positions[..., 0], dtype=torch.bool)

        # We can probably calculate the mask in one batched operation, but this might cause OOMs
        # depending on the number of gaussians and trajectories.
        for trajectory, cuboid_dim in zip(trajectories, cuboid_dims):
            positions_trajectory = (self.positions @ trajectory[:, :3, :3]) - (
                trajectory[:, :3, :3].permute(0, 2, 1) @ trajectory[:, :3, 3:]
            ).permute(0, 2, 1)
            include_mask = torch.logical_and(
                include_mask, (positions_trajectory.abs() > 0.5 * cuboid_dim).any(dim=-1).all(dim=0)
            )

        self.save_training_parameters()
        self.positions = nn.Parameter(self.positions[include_mask])
        self.scales = nn.Parameter(self.scales[include_mask])
        self.rotations = nn.Parameter(self.rotations[include_mask])
        self.densities = nn.Parameter(self.densities[include_mask])
        self.extra_signal = nn.Parameter(self.extra_signal[include_mask])

        return include_mask

    def save_training_parameters(self):
        """
        Should be called before validation-time modifications such as remove_gaussians_in_trajectories
        and track_ply_override. restore_training_parameters should then be called before resuming
        training.
        """
        if len(self.overriden_parameters) > 0:
            return  # training parameters have already been stored
        self.overriden_parameters["positions"] = self.positions
        self.overriden_parameters["scales"] = self.scales
        self.overriden_parameters["rotations"] = self.rotations
        self.overriden_parameters["densities"] = self.densities
        self.overriden_parameters["extra_signal"] = self.extra_signal

    def restore_training_parameters(self) -> None:
        if len(self.overriden_parameters) > 0:
            self.positions = self.overriden_parameters["positions"]
            self.scales = self.overriden_parameters["scales"]
            self.rotations = self.overriden_parameters["rotations"]
            self.densities = self.overriden_parameters["densities"]
            self.extra_signal = self.overriden_parameters["extra_signal"]
            self.overriden_parameters = {}
            self.overriden_buffers = {}
            self.overriden_state = {}

    def get_number_of_gaussians_per_track(self) -> dict[str, int]:
        """
        Returns the number of gaussians, if tracks are available this is returned per track.
        """
        return {
            "<NO_TRACK>": self.get_num_gaussians(),
        }

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        for p in self.parameters():
            if p.is_cuda:
                p.record_stream(stream)
        for b in self.buffers():
            if b.is_cuda:
                b.record_stream(stream)


class SHGaussianModel(BaseGaussianModel):
    # Feature vector of the 0th order SH coefficients [n_gaussians, 3] (We split it into two due to different learning rates)
    features_albedo: nn.Parameter | nn.UninitializedParameter
    # Features of the higher order SH coefficients [n_gaussians, 3]
    features_specular: nn.Parameter | nn.UninitializedParameter

    n_active_features: int

    config: SHGaussiansLayerConfig  # type: ignore[assignment] # Narrow type from BaseLayerConfig

    def __init__(
        self,
        config: SHGaussiansLayerConfig,
        trainer_config: TrainerConfig,
        datasource: DataSourceSummary,
        init_from_datasource: bool,
        initializer: Optional[BaseInitialization],
        precision: Optional[int] = None,
        cuboid_tracks: Optional[CuboidTracks] = None,
        start_end_timestamp_us: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.fourier_features_dim = unpack_optional(config.fourier_features_dim)
        self.start_end_timestamp_us = start_end_timestamp_us
        super().__init__(
            config=config,
            trainer_config=trainer_config,
            datasource=datasource,
            init_from_datasource=init_from_datasource,
            initializer=initializer,
            precision=precision,
            cuboid_tracks=cuboid_tracks,
        )
        self.n_active_features = 1

        self.config = copy.deepcopy(self.config)
        self.config.progressive_training.increase_frequency = adjust_step_for_world_size(
            trainer_config, self.config.progressive_training.increase_frequency
        )
        log.info(
            "SHGaussianModel/progressive_training: increase_frequency=%d increase_step=%d",
            self.config.progressive_training.increase_frequency,
            self.config.progressive_training.increase_step,
        )

        self.time_embed = None
        if unpack_optional(self.config.fourier_features_dim) > 1:
            log.info(
                f"SHGaussianModel/time_embed: Fourier features enabled with dim={unpack_optional(self.config.fourier_features_dim)}"
            )
            assert self.config.time_embed is not None, "time_embed must be provided when fourier_features_dim > 1"
            self.time_embed = BaseInputEmbedding.factory(
                self.config.time_embed.name,
                self.config.time_embed,
                trainer_config,
                torch.tensor(self.start_end_timestamp_us),
                cuboid_tracks,
            )

        # We need to wait for the datasource and torch.distributed context to be initialized. We will re-initializing
        # particles from datasource and validate the model's parameters later
        self.features_albedo = nn.UninitializedParameter()
        self.features_specular = nn.UninitializedParameter()

    def configure_sharded_params_and_buffers(self) -> list[str]:
        ret = super().configure_sharded_params_and_buffers()
        BaseModel.mark_as_sharded(self.features_albedo)
        BaseModel.mark_as_sharded(self.features_specular)
        return ret + ["features_albedo", "features_specular"]

    def initialize_gaussians_from_datasource(
        self,
        initializer: BaseInitialization,
        datasource: DataSourceSummary,
        cuboid_tracks: Optional[CuboidTracks] = None,
    ) -> None:
        super().initialize_gaussians_from_datasource(
            initializer=initializer,
            datasource=datasource,
            cuboid_tracks=cuboid_tracks,
        )

        assert initializer is not None, (
            f"{self.__class__.__name__}: initializer must be provided when init_from_datasource is True"
        )

        world_rank, world_size = self._get_slice_boundaries()

        if self.fourier_features_dim > 1:
            features_albedo = torch.zeros(self.get_num_gaussians(), self.fourier_features_dim, 3, device=self.device)
            features_albedo[:, 0, :] = initializer.features_albedo[world_rank::world_size]
            self.features_albedo = nn.Parameter(features_albedo.contiguous())
        else:
            self.features_albedo = nn.Parameter(initializer.features_albedo[world_rank::world_size].contiguous())

        self.features_specular = nn.Parameter(initializer.features_specular[world_rank::world_size].contiguous())

    def get_albedo_sh_dim(self) -> int:
        return 3

    def get_specular_sh_dim(self) -> int:
        return sh_degree_to_specular_dim(self.config.progressive_training.max_n_features)

    def _validate_fields(self) -> None:
        """Validate the base parameters"""
        super()._validate_fields()
        num_gaussians = self.get_num_gaussians()

        albedo_sh_dim = self.get_albedo_sh_dim()
        if self.fourier_features_dim > 1:
            assert self.features_albedo.shape == (num_gaussians, self.fourier_features_dim, albedo_sh_dim)
        else:
            assert self.features_albedo.shape == (num_gaussians, albedo_sh_dim)
        specular_sh_dims = self.get_specular_sh_dim()
        assert self.features_specular.shape == (num_gaussians, specular_sh_dims)

    #
    # Implementation of what is necessary for parameter collection.
    #
    def get_parameters(self) -> dict[str, torch.Tensor]:
        return {
            **super().get_parameters(),
            "features_albedo": self.features_albedo,
            "features_specular": self.features_specular,
        }

    def get_layer_config(self) -> collect.LayerConfigSH:
        embed_config: collect.EmbeddingConfig | None = None
        fourier_features_dim = unpack_optional(self.config.fourier_features_dim)
        if fourier_features_dim > 1:
            if type(self.time_embed) == IndividualRemapTimeInputEmbedding:
                embed_config = collect.IndividualRemapTimeInputEmbeddingConfig(
                    timestamps_us_ranges=self.time_embed.timestamps_us_ranges,
                    remap_min=self.time_embed.remap_min,
                    remap_max=self.time_embed.remap_max,
                )
            elif type(self.time_embed) == HolisticRemapTimeInputEmbedding:
                if type(self.time_embed.timestamps_us_min) != int:
                    raise ValueError(
                        f"timestamps_us_min must be an integer, got {type(self.time_embed.timestamps_us_min)}"
                    )
                if type(self.time_embed.timestamps_us_max) != int:
                    raise ValueError(
                        f"timestamps_us_max must be an integer, got {type(self.time_embed.timestamps_us_max)}"
                    )
                embed_config = collect.HolisticRemapTimeInputEmbeddingConfig(
                    timestamps_us_min=self.time_embed.timestamps_us_min,
                    timestamps_us_max=self.time_embed.timestamps_us_max,
                    remap_min=self.time_embed.remap_min,
                    remap_max=self.time_embed.remap_max,
                )
            elif type(self.time_embed) == IndividualStepTimeInputEmbedding:
                embed_config = collect.IndividualStepTimeInputEmbeddingConfig(
                    timestamps_us_ranges=self.time_embed.timestamps_us_ranges,
                    n_steps=self.time_embed.n_steps,
                    n_dims=self.time_embed.n_dims,
                )
            else:
                raise NotImplementedError(f"Invalid time embed: {type(self.time_embed)}")

        layer_config = super().get_layer_config()
        return collect.LayerConfigSH(
            rotation_activation=layer_config.rotation_activation,
            scale_activation=layer_config.scale_activation,
            density_activation=layer_config.density_activation,
            fourier_features_dim=fourier_features_dim,
            embed_config=embed_config,
        )

    def get_layer_data(
        self, context: BaseGaussianModel.CollectionContext, gathered_parameters: dict[str, torch.Tensor]
    ) -> collect.LayerDataSH:
        embed_data: collect.EmbeddingData | None = None
        if unpack_optional(self.config.fourier_features_dim) > 1:
            if type(self.time_embed) == IndividualRemapTimeInputEmbedding:
                embed_data = collect.IndividualRemapTimeInputEmbeddingData(
                    instance_idx=gathered_parameters["gaussian_cuboid_ids"],
                )
            elif type(self.time_embed) == HolisticRemapTimeInputEmbedding:
                embed_data = None
            elif type(self.time_embed) == IndividualStepTimeInputEmbedding:
                embed_data = collect.IndividualStepTimeInputEmbeddingData(
                    instance_idx=gathered_parameters["gaussian_cuboid_ids"],
                    u=self.time_embed.u,
                    beta=self.time_embed.beta,
                )
            else:
                raise NotImplementedError(f"Invalid time embed: {type(self.time_embed)}")
        else:
            embed_data = None

        layer_data = super().get_layer_data(context, gathered_parameters)
        return collect.LayerDataSH(
            positions=layer_data.positions,
            rotations=layer_data.rotations,
            scales=layer_data.scales,
            densities=layer_data.densities,
            extra_signal=layer_data.extra_signal,
            camera_extra_signal=layer_data.camera_extra_signal,
            lidar_extra_signal=layer_data.lidar_extra_signal,
            features_albedo=gathered_parameters["features_albedo"],
            features_specular=gathered_parameters["features_specular"],
            embed_data=embed_data,
        )

    # This is a copy from the model's collector, it can be used to access certain
    # "global" functions.
    collector: collect.GaussianParameterCollector

    #
    # End of implementation of what is necessary for parameter collection.
    #

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        additional_parameters: dict[str, torch.Tensor] = {}
        if self.time_embed is not None:
            additional_parameters |= update_module_step(self.time_embed, epoch, global_step, system)
        return additional_parameters

    def update_step_train_batch_end(
        self, epoch: int, global_step: int, batch: DataAndRenderingBatch, system, **kwargs
    ) -> None:
        # Every N its we increase the levels of SH up to a maximum degree
        if global_step > 0 and global_step % self.config.progressive_training.increase_frequency == 0:
            self.increase_number_of_active_features()

        self.log_extra_parameters(system, epoch, global_step)

    @ScopedTimer("SHGaussianModel/log_extra_parameters")
    def log_extra_parameters(self, system, epoch: int, global_step: int) -> None:
        system.log("n_active_features", self.n_active_features)

        for optimizer in self.optimizers:
            for param_group in optimizer["optimizer"].param_groups:
                if "name" in param_group:
                    system.log(f"lr/{param_group['name']}", param_group["lr"])

    def configure_optimizers(self, name_prefix: str = "") -> list[OptimizerLRSchedulerConfig]:
        """Returns a list of module-owned configured optimizers (optimizers paired with an optional LR scheduler),
        which will be stepped in the main training loop, allowing the module to interact with it's owned optimizers"""

        optimizers = super().configure_optimizers(name_prefix=name_prefix)
        if self.time_embed is not None:
            optimizers += configure_optimizers(
                self.config.time_embed, self.trainer_config, self.time_embed, name_prefix
            )

        return optimizers

    @ScopedTimer("SHGaussianModel/increase_number_of_active_features")
    def increase_number_of_active_features(self) -> None:
        """Increase the number of optimizable featues (features_specular)"""
        self.n_active_features = min(
            self.config.progressive_training.max_n_features,
            self.n_active_features + self.config.progressive_training.increase_step,
        )

    def get_extra_state(self) -> dict[str, Any] | None:
        return {"n_active_features": self.n_active_features}

    def set_extra_state(self, state: dict[str, Any] | None) -> None:
        if state is not None and "n_active_features" in state:
            self.n_active_features = int(state["n_active_features"])

    def remove_gaussians_in_trajectories(
        self, trajectories: List[torch.Tensor], cuboid_dims: List[torch.Tensor]
    ) -> torch.Tensor:
        include_mask = super().remove_gaussians_in_trajectories(trajectories, cuboid_dims)
        self.features_albedo = nn.Parameter(self.features_albedo[include_mask])
        self.features_specular = nn.Parameter(self.features_specular[include_mask])
        return include_mask

    def save_training_parameters(self):
        first_save = len(self.overriden_parameters) == 0
        super().save_training_parameters()
        if first_save:
            self.overriden_parameters["features_albedo"] = self.features_albedo
            self.overriden_parameters["features_specular"] = self.features_specular

    def restore_training_parameters(self) -> None:
        if len(self.overriden_parameters) > 0:
            self.features_albedo = self.overriden_parameters["features_albedo"]
            self.features_specular = self.overriden_parameters["features_specular"]
            super().restore_training_parameters()

    def export_ply(
        self,
        export_dir: Path,
        format: GaussianExportFormat = GaussianExportFormat._3DGS,
        percentage_gaussians: float = 100,
    ) -> None:
        """
        Export Gaussian model as PLY files in the specified format.

        For 3DGS format, it should be compatible with the original 3DGS implementation but differences
        between 3DGS/3DGUT/3DGRT rendering will cause slight differences when rendered with
        3rd-party 3DGS viewers.
        NB : extra_signal is not exported.

        Args:
            export_dir: Directory path where the PLY file will be saved as 'model.ply'.
            format: Export format for the Gaussian data. Available options:
                - GaussianExportFormat._3DGS: Original 3D Gaussian Splatting format
                - GaussianExportFormat._3DGRT: 3DGRT format for visualization
            percentage_gaussians: Percentage of Gaussians to export (0, 100]. Only used
                for _3DGRT format. Defaults to 100 (export all Gaussians).
        """
        if format == GaussianExportFormat._3DGS:
            write_ply_3dgs(
                export_dir / "model.ply",
                self.positions,
                self.rotations,
                self.scales,
                self.densities,
                self.features_albedo,
                self.features_specular,
            )
        elif format == GaussianExportFormat._3DGRT:
            write_ply_3dgrt(
                export_dir / "model.ply",
                self.get_positions(),
                self.get_rotations(quaternion_format="xyzw"),
                self.get_scales(),
                self.get_densities(),
                percentage_gaussians=percentage_gaussians,
            )


class RigidGaussianModel(SHGaussianModel):
    cuboid_tracks: CuboidTracks
    tracks_calib: BaseTracksCalib

    gaussian_cuboid_ids: nn.Buffer | nn.UninitializedBuffer
    global_step: int

    config: RigidGaussiansLayerConfig  # type: ignore[assignment] # Narrow type from SHGaussiansLayerConfig

    def __init__(
        self,
        config: RigidGaussiansLayerConfig,
        trainer_config: TrainerConfig,
        datasource: DataSourceSummary,
        init_from_datasource: bool,
        initializer: Optional[BaseInitialization],
        precision: Optional[int] = None,
        cuboid_tracks: Optional[CuboidTracks] = None,
        start_end_timestamp_us: Optional[Tuple[int, int]] = None,
    ) -> None:
        assert cuboid_tracks is not None, "cuboid_tracks must be provided for DynamicGaussians"
        self.cuboid_tracks = cuboid_tracks
        self.global_step = 0

        super().__init__(
            config=config,
            trainer_config=trainer_config,
            datasource=datasource,
            init_from_datasource=init_from_datasource,
            initializer=initializer,
            precision=precision,
            cuboid_tracks=cuboid_tracks,
            start_end_timestamp_us=start_end_timestamp_us,
        )

        self.tracks_calib = BaseTracksCalib.factory(
            config.tracks_calib.name, config.tracks_calib, trainer_config, cuboid_tracks
        )

        self.track_albedos = (
            nn.Parameter(
                torch.eye(4, device=self.device)[:3].unsqueeze(0).repeat(unpack_optional(cuboid_tracks).n_tracks, 1, 1)
            )
            if self.config.optimize_track_albedo
            else None
        )

        self.track_scales = (
            nn.Parameter(
                self.scale_activation_inv(torch.ones(unpack_optional(cuboid_tracks).n_tracks, 3, device=self.device))
            )
            if self.config.optimize_track_scale
            else None
        )

        # Config flag for nearest neighbor track selection (LiDAR only)
        self.nearest_neighbor_track_for_lidar = self.config.nearest_neighbor_track_for_lidar

        # We need to wait for the datasource and torch.distributed context to be initialized. We will re-initializing
        # particles from datasource and validate the model's parameters later
        self.gaussian_cuboid_ids = nn.UninitializedBuffer()

        # Config to specify if the Gaussians are static.
        self.is_static = self.config.is_static

    def create_tracks_edit_from_cuboid_tracks(self, edited_cuboid_tracks: CuboidTracks) -> LayerTrackIds.Edit:
        # Filter tracks that are not modeled in this layer
        existing_cuboid_tracks = CuboidTracks.Ops.subset_from_mask(
            edited_cuboid_tracks,
            torch.tensor([t in unpack_optional(self.cuboid_tracks.tracks_id) for t in edited_cuboid_tracks.tracks_id]),
        )

        # Build the mapping from the edited_cuboid_tracks to the original cuboid_tracks
        assert self.cuboid_tracks.tracks_id is not None, "track_ids should be initialized before enabling editing"
        edited_mapping = torch.tensor(
            [self.cuboid_tracks.tracks_id.index(t) for t in existing_cuboid_tracks.tracks_id],
            dtype=torch.long,
            device=existing_cuboid_tracks.device,
        )

        return LayerTrackIds.Edit(cuboid_tracks=existing_cuboid_tracks, mapping=edited_mapping)

    def configure_sharded_params_and_buffers(self) -> list[str]:
        ret = super().configure_sharded_params_and_buffers()
        BaseModel.mark_as_sharded(self.gaussian_cuboid_ids)
        return ret + ["gaussian_cuboid_ids"]

    def initialize_gaussians_from_datasource(
        self,
        initializer: BaseInitialization,
        datasource: DataSourceSummary,
        cuboid_tracks: Optional[CuboidTracks] = None,
    ) -> None:
        super().initialize_gaussians_from_datasource(
            initializer=initializer,
            datasource=datasource,
            cuboid_tracks=cuboid_tracks,
        )

        assert initializer is not None, (
            f"{self.__class__.__name__}: initializer must be provided when init_from_datasource is True"
        )

        world_rank, world_size = self._get_slice_boundaries()

        self.gaussian_cuboid_ids = nn.Buffer(
            torch.vstack(initializer.gaussian_cuboid_ids).data.squeeze(-1)[world_rank::world_size].contiguous()
        )

    def _validate_fields(self) -> None:
        """Validate the base parameters"""
        super()._validate_fields()
        num_gaussians = self.get_num_gaussians()

        assert self.gaussian_cuboid_ids.shape == (num_gaussians,)

    def get_additional_buffers(self) -> dict[str, torch.Tensor]:
        additional_buffers = super().get_additional_buffers()
        additional_buffers |= {
            "gaussian_cuboid_ids": self.gaussian_cuboid_ids,
        }
        return additional_buffers

    def remove_gaussians_in_trajectories(
        self, trajectories: List[torch.Tensor], cuboid_dims: List[torch.Tensor]
    ) -> torch.Tensor:
        include_mask = super().remove_gaussians_in_trajectories(trajectories, cuboid_dims)
        self.gaussian_cuboid_ids = nn.Buffer(self.gaussian_cuboid_ids[include_mask])
        return include_mask

    #
    # Implementation of what is necessary for parameter collection.
    #
    def get_parameters(self) -> dict[str, torch.Tensor]:
        return {
            **super().get_parameters(),
            "gaussian_cuboid_ids": self.gaussian_cuboid_ids,
        }

    def get_layer_config(self) -> collect.LayerConfigRigid:
        layer_config = super().get_layer_config()
        assert isinstance(layer_config, collect.LayerConfigSH)
        return collect.LayerConfigRigid(
            rotation_activation=layer_config.rotation_activation,
            scale_activation=layer_config.scale_activation,
            density_activation=layer_config.density_activation,
            fourier_features_dim=layer_config.fourier_features_dim,
            embed_config=layer_config.embed_config,
        )

    def get_layer_data(
        self, context: BaseGaussianModel.CollectionContext, gathered_parameters: dict[str, torch.Tensor]
    ) -> collect.LayerDataRigid:
        # Apply tracks edit and calibration.
        # This might filter gathered_parameters based on tracks edit
        tracks_poses, tracks_timestamps, tracks_packinfo, cuboid_dims, tracks_ids = self._get_calibrated_tracks(
            context, gathered_parameters
        )

        layer_data = super().get_layer_data(context, gathered_parameters)
        assert isinstance(layer_data, collect.LayerDataSH)

        # Apply track scale.  Note that this could be done in the merged kernel,
        # but it's not enabled by default, so we might optimize it later.
        if self.track_scales is not None:
            assert type(self.scale_activation) == ExpActivation
            track_scales = self.track_scales[tracks_ids]
            activated_track_scales = self.scale_activation(track_scales)
            layer_data.positions = layer_data.positions * activated_track_scales
            # scales are not activated here, we can just add them
            # exp(s1) * exp(s2) = exp(s1 + s2)
            layer_data.scales = layer_data.scales + track_scales

        # Perform track interpolation.
        interpolated_pose_vec, interpolated_mask = self._get_interpolated_poses(
            context.rendering_data,
            tracks_poses,
            tracks_timestamps,
            tracks_packinfo,
            cuboid_dims,
        )

        context.additional_data["interpolated_mask"] = (interpolated_mask, False)

        return collect.LayerDataRigid(
            positions=layer_data.positions,
            rotations=layer_data.rotations,
            scales=layer_data.scales,
            densities=layer_data.densities,
            extra_signal=layer_data.extra_signal,
            camera_extra_signal=layer_data.camera_extra_signal,
            lidar_extra_signal=layer_data.lidar_extra_signal,
            features_albedo=layer_data.features_albedo,
            features_specular=layer_data.features_specular,
            embed_data=layer_data.embed_data,
            poses=interpolated_pose_vec,
            keep_mask=interpolated_mask,
            tracks_ids=tracks_ids,
        )

    def _get_calibrated_tracks(
        self, context: BaseGaussianModel.CollectionContext, gaussian_nodes_parameters: dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tracks_edit = context.tracks_edit
        if tracks_edit is not None and tracks_edit.mapping.shape[0] > 0:
            # When tracks_edit is on, we remove Gaussians which are not in tracks in
            # the mapping.
            with ScopedTimer("RigidGaussianModel/edit_mapping"):
                # The gaussian_cuboid_ids are the indices of the gaussians in the original cuboid tracks.
                # We create a new pack_info that is the same as the original, but points into the new tracks.
                # This way we don't need to update the gaussian_cuboid_ids.
                #
                # We might consider storing the pack_info inside the tracks_edit object already in this format.
                new_pack_info = torch.zeros_like(self.cuboid_tracks.tracks_packinfo)
                new_pack_info[tracks_edit.mapping] = tracks_edit.cuboid_tracks.tracks_packinfo
                new_cuboid_dims = torch.zeros_like(self.cuboid_tracks.cuboids_dims)
                new_cuboid_dims[tracks_edit.mapping] = tracks_edit.cuboid_tracks.cuboids_dims

                # Only keep the gaussians that are in the tracks in the mapping.
                #
                # Note that we could use an approach like when they are out of the tracks timestamps range,
                # and just zero their opacity.  This would keep the buffer size constant and
                # would avoid the CPU/GPU sync.
                #
                # We leave the exploration of this optimization for later.
                gaussian_cuboid_ids = gaussian_nodes_parameters["gaussian_cuboid_ids"]
                device = gaussian_cuboid_ids.device
                n_og_tracks = self.cuboid_tracks.n_tracks
                tracks_keep_mask = torch.zeros((n_og_tracks,), dtype=torch.bool, device=device)
                tracks_keep_mask[tracks_edit.mapping] = True

                keep_mask = tracks_keep_mask[gaussian_cuboid_ids]
                all_indices = torch.arange(gaussian_cuboid_ids.shape[0], device=gaussian_cuboid_ids.device)
                # NOTE: This is a CPU-GPU sync.
                indices_to_keep = all_indices[keep_mask]
                updated_gaussian_nodes_parameters = {
                    k: v[indices_to_keep] for k, v in gaussian_nodes_parameters.items()
                }
                gaussian_nodes_parameters.update(updated_gaussian_nodes_parameters)

            # Use the edited tracks for the rest of this function.
            tracks_poses = tracks_edit.cuboid_tracks.tracks_poses.vec()
            tracks_timestamps = tracks_edit.cuboid_tracks.tracks_timestamps_us
            tracks_packinfo = new_pack_info
            cuboid_dims = new_cuboid_dims
        else:
            if type(self.tracks_calib) == DirectTracksCalib:
                # Use collector to get the calibrated tracks.
                # Eventually this could be merged with parameter collection.
                tracks_calib_data = collect.DirectTracksCalibData(
                    tracks_poses=self.cuboid_tracks.tracks_poses.vec(),
                    gradient_mask=self.tracks_calib.gradient_mask,
                    tracks_delta_q=self.tracks_calib.tracks_delta_q,
                    tracks_delta_t=self.tracks_calib.tracks_delta_t,
                )
                calibrated_tracks = self.collector.calibrate_tracks_poses(tracks_calib_data)

                tracks_poses = calibrated_tracks
                tracks_timestamps = self.cuboid_tracks.tracks_timestamps_us
                tracks_packinfo = self.cuboid_tracks.tracks_packinfo
                cuboid_dims = self.cuboid_tracks.cuboids_dims
            else:
                cuboid_tracks = self.tracks_calib(self.cuboid_tracks)
                tracks_poses = cuboid_tracks.tracks_poses.vec()
                tracks_timestamps = cuboid_tracks.tracks_timestamps_us
                tracks_packinfo = cuboid_tracks.tracks_packinfo
                cuboid_dims = cuboid_tracks.cuboids_dims

        gaussian_cuboid_ids = gaussian_nodes_parameters["gaussian_cuboid_ids"]
        return tracks_poses, tracks_timestamps, tracks_packinfo, cuboid_dims, gaussian_cuboid_ids

    def _get_interpolated_poses(
        self,
        rendering_data: RenderingData,
        tracks_poses: torch.Tensor,
        tracks_timestamps: torch.Tensor,
        tracks_packinfo: torch.Tensor,
        cuboid_dims: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_tracks = tracks_packinfo.shape[0]

        # If the rigid Gaussians are static, we don't need to interpolate the poses
        if self.is_static:
            return tracks_poses[::2], torch.ones(n_tracks, dtype=torch.bool, device=tracks_poses.device)

        ## Raycasting based timestamp estimation
        # Get accurate timestamps for Camera and LiDAR, if rays timestamps are available
        with ScopedTimer("RigidGaussianModel/get_tracks_timestamps_estimation_data"):
            tracks_timestamps_data: collect.TracksTimestampsData
            default_timestamp = self.get_frame_timestamp(rendering_data)
            if rendering_data.rays_timestamps_us is not None and n_tracks > 0:
                rays = rendering_data.rays.view(-1, 6)
                rays_timestamps = rendering_data.rays_timestamps_us.view(-1)
                tracks_timestamps_data = collect.TracksTimestampsEstimationData(
                    rays=rays,
                    rays_timestamps=rays_timestamps,
                    cuboids_dimensions=cuboid_dims,
                    default_timestamp=default_timestamp,
                )
            else:
                # No ray timestamps available - fallback to use single timestamp for all tracks
                tracks_timestamps_data = collect.TracksTimestampsGlobalData(
                    timestamp=default_timestamp,
                )

        with ScopedTimer("RigidGaussianModel/get_tracks_poses"):
            # Detect if this is a LiDAR sensor (for nearest neighbor selection)
            is_lidar = False
            if rendering_data.sensor_model_parameters is not None and len(rendering_data.sensor_model_parameters) > 0:
                sensor_model = rendering_data.sensor_model_parameters[0]
                # Check for the Parameters type (runtime data)
                is_lidar = isinstance(
                    sensor_model,
                    (RowOffsetStructuredSpinningLidarModel, RowOffsetStructuredSpinningLidarModelParameters),
                )
            nearest_neighbor = is_lidar and self.nearest_neighbor_track_for_lidar

            tracks_interpolation_data = collect.TracksInterpolationData(
                tracks_poses=tracks_poses,
                tracks_timestamps=tracks_timestamps,
                tracks_packinfo=tracks_packinfo,
                timestamps_data=tracks_timestamps_data,
                nearest_neighbor=nearest_neighbor,
            )

            # Timestamp estimation and pose sampling strategy:
            # - tracks_timestamp_us contains either projection-based per-track estimates (if ray timestamps
            #   are available) or falls back to frame mid-time for all tracks.
            # - For LiDAR: Use nearest neighbor snapping to the closest annotated pose sample. This allows
            #   for use of high-confidence/high-FPS cuboid annotations when available.
            #   Works with both projection-based estimates and frame mid-time fallback.
            interpolated_pose_vec, interpolated_mask = self.collector.interpolate_tracks_poses(
                tracks_interpolation_data
            )

            # Regardless of whether we are using nearest neighbor or interpolation, we filter the
            # gaussians based on the and interpolated_mask.
            #
            # For each track, the interpolated_mask will be True if the track timestamp that was used
            # for interpolation falls within the timestamps associated with that track's poses.  If it
            # doesn't, it means we are trying to interpolate outside of the available poses, so the
            # mask will be False for this track, and the gaussians belonging to it will be filtered out.
            return interpolated_pose_vec, interpolated_mask

    def post_process(self, offset: int, results: collect.CollectorResult, layer_data: collect.LayerDataBase) -> None:
        # Apply track albedo.  Note that this could be done in the merged kernel,
        # but it's not enabled by default, so we might optimize it later.
        if self.track_albedos is not None:
            assert isinstance(layer_data, collect.LayerDataRigid)
            gaussian_cuboid_ids = layer_data.tracks_ids
            nb_gaussians = gaussian_cuboid_ids.shape[0]
            features = results.features[offset : offset + nb_gaussians]
            features_albedo = features[..., :3]

            track_albedos = self.track_albedos[gaussian_cuboid_ids]
            transformed_features_albedo = (
                track_albedos
                @ (torch.cat([features_albedo, torch.ones_like(features_albedo[..., :1])], -1).unsqueeze(-1))
            ).squeeze(-1)
            features = torch.cat([transformed_features_albedo, features[..., 3:]], -1)
            results.features[offset : offset + nb_gaussians] = features

    #
    # End of implementation of what is necessary for parameter collection.
    #

    def configure_optimizers(self, name_prefix: str = "") -> list[OptimizerLRSchedulerConfig]:
        """Returns a list of module-owned configured optimizers (optimizers paired with an optional LR scheduler),
        which will be stepped in the main training loop, allowing the module to interact with it's owned optimizers"""

        optimizers = super().configure_optimizers(name_prefix)

        optimizers += self.tracks_calib.configure_optimizers(name_prefix=name_prefix)

        if self.track_albedos is not None:
            optimizers += configure_optimizers(self.config.track_albedo, self.trainer_config, self, name_prefix)

        if self.track_scales is not None:
            optimizers += configure_optimizers(self.config.track_scale, self.trainer_config, self, name_prefix)

        readable = []
        for entry in optimizers:
            optimizer = entry["optimizer"]
            opt_class = optimizer.__class__.__name__
            group_names = [pg.get("name", f"group{idx}") for idx, pg in enumerate(optimizer.param_groups)]
            readable.append(f"{opt_class}[" + ", ".join(map(str, group_names)) + "]")
        log.info(f"Create optimizer for {self.__class__.__name__}: " + "; ".join(readable))
        return optimizers

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        self.global_step = global_step
        return update_module_step(self.tracks_calib, epoch, global_step, system)

    def get_track_poses_and_dims(self, track_id: str) -> Optional[Tuple[lt.SE3, torch.Tensor]]:
        """
        Gets the poses and cuboid dimensions of a track id if present
        """
        if track_id not in self.cuboid_tracks.tracks_id:
            return None

        cur_track = CuboidTracks.Ops.subset_from_tracks_id(self.tracks_calib(self.cuboid_tracks), [track_id])
        assert cur_track.n_tracks == 1, f"Unexpected number of tracks: {cur_track.n_tracks}"
        return cur_track.tracks_poses, cur_track.cuboids_dims[0]

    def _compute_scale_factor(
        self,
        cuboid_dims: torch.Tensor,
        dims_offset: torch.Tensor,
    ) -> float:
        """
        Computes the scale factor for an asset based on cuboid dimensions and offset.

        Args:
            cuboid_dims: The dimensions of the cuboid [x, y, z]
            dims_offset: The offset to apply to cuboid dimensions

        Returns:
            scale_factor: The computed scale factor as a scalar float
        """
        log.info(f"Applying dims_offset: {dims_offset.tolist()} to asset cuboids_dims: {cuboid_dims.tolist()}")
        return (cuboid_dims - dims_offset).max().item()

    def track_ply_override(
        self,
        track_id: str,
        asset: Asset,
        transform: torch.Tensor,
        dims_offset: torch.Tensor,
    ) -> bool:
        """
        Overrides the gaussians for a track id with gaussians from an asset.
        Note that the override is only meant to be used at inference time and should be reverted
        by calling restore_training_parameters before resuming training.
        """
        if track_id not in self.cuboid_tracks.tracks_id:
            log.warning(
                f"Failed to replace track {track_id}, not found in {self.__class__.__name__}'s cuboid_tracks, skipping"
            )
            return False

        loaded_ply = asset
        track_index = self.cuboid_tracks.tracks_id.index(track_id)
        mask = self.gaussian_cuboid_ids != track_index
        loaded_ply.transform(transform)

        if loaded_ply.cuboids_dims is not None:
            scale_factor = self._compute_scale_factor(loaded_ply.cuboids_dims, dims_offset)
        else:
            scale_factor = self._compute_scale_factor(self.cuboid_tracks.cuboids_dims[track_index], dims_offset)

        loaded_ply.scale(
            scale_factor,
            self.scale_activation,
            self.scale_activation_inv,
        )

        self.save_training_parameters()

        if self.track_albedos is not None:
            self.track_albedos[track_index] = torch.eye(4, device=self.track_albedos.device)[:3]

        if self.track_scales is not None:
            self.track_scales[track_index] = self.scale_activation_inv(torch.ones(3), device=self.track_scales.device)

        # Handle features_albedo shape mismatch for temporal Fourier features - fix needed for asset harvester ply's (lower dimension than car2sim dynamic gaussians)
        if self.features_albedo.shape[1:] != loaded_ply.features_albedo.shape[1:]:
            # Shape mismatch detected - model expects [N, fourier_features_dim, 3], but PLY has [N, 3]
            if self.fourier_features_dim > 1:
                # Expand loaded PLY features to match the expected shape
                expanded_features_albedo = torch.zeros(
                    loaded_ply.features_albedo.shape[0],
                    self.fourier_features_dim,
                    3,
                    dtype=self.features_albedo.dtype,
                    device=self.features_albedo.device,
                )
                expanded_features_albedo[:, 0, :] = loaded_ply.features_albedo
                loaded_features_albedo = expanded_features_albedo
            else:
                # This shouldn't happen, but handle it anyway
                loaded_features_albedo = loaded_ply.features_albedo
        else:
            # No shape mismatch, use features as-is
            loaded_features_albedo = loaded_ply.features_albedo

        self.positions = nn.Parameter(torch.cat([self.positions[mask], loaded_ply.positions]))
        self.scales = nn.Parameter(torch.cat([self.scales[mask], loaded_ply.scales]))
        self.rotations = nn.Parameter(torch.cat([self.rotations[mask], loaded_ply.rotations]))
        self.densities = nn.Parameter(torch.cat([self.densities[mask], loaded_ply.densities]))
        self.features_albedo = nn.Parameter(torch.cat([self.features_albedo[mask], loaded_features_albedo]))
        self.features_specular = nn.Parameter(
            torch.cat(
                [
                    self.features_specular[mask],
                    loaded_ply.features_specular
                    if loaded_ply.features_specular is not None
                    else torch.zeros(
                        loaded_ply.features_albedo.shape[0],
                        self.features_specular.shape[1],
                        dtype=self.features_specular.dtype,
                        device=self.device,
                    ),
                ]
            )
        )
        # FIXME : extra_signal is not imported from the ply file
        self.extra_signal = nn.Parameter(
            torch.cat(
                [
                    self.extra_signal[mask],
                    torch.zeros(
                        loaded_ply.positions.shape[0],
                        self.extra_signal.shape[1],
                        dtype=self.extra_signal.dtype,
                        device=self.extra_signal.device,
                    ),
                ]
            )
        )

        camera_signal = self.get_extra_signal("camera")
        if camera_signal.shape[0] > 0:
            self.camera_extra_signal = nn.Parameter(
                torch.cat(
                    [
                        camera_signal[mask],
                        torch.zeros(
                            loaded_ply.positions.shape[0],
                            camera_signal.shape[1],
                            dtype=camera_signal.dtype,
                            device=camera_signal.device,
                        ),
                    ]
                )
            )

        lidar_signal = self.get_extra_signal("lidar")
        if lidar_signal.shape[0] > 0:
            self.lidar_extra_signal = nn.Parameter(
                torch.cat(
                    [
                        lidar_signal[mask],
                        torch.zeros(
                            loaded_ply.positions.shape[0],
                            lidar_signal.shape[1],
                            dtype=lidar_signal.dtype,
                            device=lidar_signal.device,
                        ),
                    ]
                )
            )

        self.gaussian_cuboid_ids = nn.Buffer(
            torch.cat(
                [
                    self.gaussian_cuboid_ids[mask],
                    torch.full(
                        (loaded_ply.features_albedo.shape[0],),
                        track_index,
                        dtype=self.gaussian_cuboid_ids.dtype,
                        device=self.device,
                    ),
                ]
            )
        )
        log.info(f"Completed replacement of track {track_id}")
        return True

    def insert_asset(self, track_id: str, asset: Asset, cuboid_tracks: CuboidTracks, dims_offset: torch.Tensor) -> bool:
        """
        Inserts a new track_id with gaussians derived from 'asset'.
        Note that the insert is only meant to be used at inference time and should be reverted
        by calling restore_training_parameters before resuming training.
        """
        transform_matrix = [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        transform: torch.Tensor = torch.FloatTensor(transform_matrix).to("cuda")

        # Model's self.cuboid_tracks must be updated before this function
        track_index = self.cuboid_tracks.tracks_id.index(track_id)

        loaded_ply = asset
        loaded_ply.transform(transform)

        scale_factor = self._compute_scale_factor(cuboid_tracks.cuboids_dims[0], dims_offset)
        loaded_ply.scale(
            scale_factor,
            self.scale_activation,
            self.scale_activation_inv,
        )
        if self.track_albedos is not None:
            # Expand tensor to accommodate new track
            self.track_albedos = nn.Parameter(
                torch.cat([self.track_albedos, torch.eye(4, device=self.track_albedos.device)[:3].unsqueeze(0)])
            )
        if self.track_scales is not None:
            # Expand tensor to accommodate new track
            self.track_scales = nn.Parameter(
                torch.cat(
                    [self.track_scales, self.scale_activation_inv(torch.ones(1, 3, device=self.track_scales.device))]
                )
            )

        # Handle features_albedo shape mismatch for temporal Fourier features - fix needed for asset harvester ply's (lower dimension than car2sim dynamic gaussians)
        if self.features_albedo.shape[1:] != loaded_ply.features_albedo.shape[1:]:
            # Shape mismatch detected - model expects [N, fourier_features_dim, 3], but PLY has [N, 3]
            if self.fourier_features_dim > 1:
                # Expand loaded PLY features to match the expected shape
                expanded_features_albedo = torch.zeros(
                    loaded_ply.features_albedo.shape[0],
                    self.fourier_features_dim,
                    3,
                    dtype=self.features_albedo.dtype,
                    device=self.features_albedo.device,
                )
                expanded_features_albedo[:, 0, :] = loaded_ply.features_albedo
                loaded_features_albedo = expanded_features_albedo
            else:
                # This shouldn't happen, but handle it anyway
                loaded_features_albedo = loaded_ply.features_albedo
        else:
            # No shape mismatch, use features as-is
            loaded_features_albedo = loaded_ply.features_albedo

        self.positions = nn.Parameter(torch.cat([self.positions, loaded_ply.positions]))
        self.scales = nn.Parameter(torch.cat([self.scales, loaded_ply.scales]))
        self.rotations = nn.Parameter(torch.cat([self.rotations, loaded_ply.rotations]))
        self.densities = nn.Parameter(torch.cat([self.densities, loaded_ply.densities]))
        self.features_albedo = nn.Parameter(torch.cat([self.features_albedo, loaded_features_albedo]))
        self.features_specular = nn.Parameter(
            torch.cat(
                [
                    self.features_specular,
                    loaded_ply.features_specular
                    if loaded_ply.features_specular is not None
                    else torch.zeros(
                        loaded_ply.features_albedo.shape[0],
                        self.features_specular.shape[1],
                        dtype=self.features_specular.dtype,
                        device=self.device,
                    ),
                ]
            )
        )
        # FIXME : extra_signal is not imported from the ply file
        self.extra_signal = nn.Parameter(
            torch.cat(
                [
                    self.extra_signal,
                    torch.zeros(
                        loaded_ply.positions.shape[0],
                        self.extra_signal.shape[1],
                        dtype=self.extra_signal.dtype,
                        device=self.extra_signal.device,
                    ),
                ]
            )
        )

        camera_signal = self.get_extra_signal("camera")
        if camera_signal.shape[0] > 0:
            self.camera_extra_signal = nn.Parameter(
                torch.cat(
                    [
                        camera_signal,
                        torch.zeros(
                            loaded_ply.positions.shape[0],
                            camera_signal.shape[1],
                            dtype=camera_signal.dtype,
                            device=camera_signal.device,
                        ),
                    ]
                )
            )

        lidar_signal = self.get_extra_signal("lidar")
        if lidar_signal.shape[0] > 0:
            self.lidar_extra_signal = nn.Parameter(
                torch.cat(
                    [
                        lidar_signal,
                        torch.zeros(
                            loaded_ply.positions.shape[0],
                            lidar_signal.shape[1],
                            dtype=lidar_signal.dtype,
                            device=lidar_signal.device,
                        ),
                    ]
                )
            )

        self.gaussian_cuboid_ids = nn.Buffer(
            torch.cat(
                [
                    self.gaussian_cuboid_ids,
                    torch.full(
                        (loaded_ply.features_albedo.shape[0],),
                        track_index,
                        dtype=self.gaussian_cuboid_ids.dtype,
                        device=self.device,
                    ),
                ]
            )
        )
        log.info(f"Completed insertion of track {track_id}")

        return True

    def update_tracks_calib_and_time_embed(self, cuboid_tracks: CuboidTracks, inserted_track_ids: set[str]) -> None:
        """
        Updates tracks_calib to handle newly inserted tracks while preserving learned calibration.

        Args:
            cuboid_tracks: Combined tracks (original + inserted)
            inserted_track_ids: Set of newly inserted track IDs
        """

        if isinstance(self.tracks_calib, CompositeTracksCalib):
            original_calib = self.tracks_calib.original_calib
            inserted_track_ids = self.tracks_calib.inserted_track_ids | inserted_track_ids
        else:
            original_calib = self.tracks_calib

        self.tracks_calib = CompositeTracksCalib(
            config=self.config.tracks_calib,
            trainer_config=self.trainer_config,
            original_calib=original_calib,
            inserted_track_ids=inserted_track_ids,
            cuboid_tracks=cuboid_tracks,
        )

        # Recreate time_embed with updated tracks if using temporal features
        if unpack_optional(self.config.fourier_features_dim) > 1 and self.time_embed is not None:
            self.time_embed = BaseInputEmbedding.factory(
                self.config.time_embed.name,
                self.config.time_embed,
                self.trainer_config,
                torch.tensor(self.start_end_timestamp_us),
                cuboid_tracks,
            )

    def save_training_parameters(self):
        first_save = len(self.overriden_parameters) == 0
        super().save_training_parameters()
        if first_save:
            assert len(self.overriden_buffers) == 0, (
                "overriden_buffers should be empty if overriden_parameters was empty"
            )
            self.overriden_buffers["gaussian_cuboid_ids"] = self.gaussian_cuboid_ids
            self.overriden_state["cuboid_tracks"] = self.cuboid_tracks
            self.overriden_state["tracks_calib"] = self.tracks_calib
            self.overriden_state["time_embed"] = self.time_embed
            if self.track_albedos is not None:
                self.overriden_parameters["track_albedos"] = self.track_albedos
            if self.track_scales is not None:
                self.overriden_parameters["track_scales"] = self.track_scales
            self.overriden_parameters["camera_extra_signal"] = self.camera_extra_signal
            self.overriden_parameters["lidar_extra_signal"] = self.lidar_extra_signal

    def restore_training_parameters(self) -> None:
        if len(self.overriden_parameters) > 0:
            self.cuboid_tracks = self.overriden_state["cuboid_tracks"]
            self.tracks_calib = self.overriden_state["tracks_calib"]
            self.time_embed = self.overriden_state["time_embed"]
            self.gaussian_cuboid_ids = self.overriden_buffers["gaussian_cuboid_ids"]
            if "track_albedos" in self.overriden_parameters:
                self.track_albedos = self.overriden_parameters["track_albedos"]
            if "track_scales" in self.overriden_parameters:
                self.track_scales = self.overriden_parameters["track_scales"]
            self.camera_extra_signal = self.overriden_parameters["camera_extra_signal"]
            self.lidar_extra_signal = self.overriden_parameters["lidar_extra_signal"]
            super().restore_training_parameters()

    def get_extra_state(self) -> dict[str, Any] | None:
        extra_state = super().get_extra_state() or {}
        extra_state["n_tracks"] = self.cuboid_tracks.n_tracks
        return extra_state

    def set_extra_state(self, state: dict[str, Any] | None) -> None:
        super().set_extra_state(state)
        if (state is not None) and "n_tracks" in state:
            assert self.cuboid_tracks.n_tracks == state["n_tracks"], (
                "State contains a different number of tracks than the model requires"
            )

    def export_ply(
        self,
        export_dir: Path,
        format: GaussianExportFormat = GaussianExportFormat._3DGS,
        percentage_gaussians: float = 100,
    ) -> None:
        """
        Export Gaussian model as PLY files in the specified format.

        For 3DGS format, it should be compatible with the original 3DGS implementation but differences
        between 3DGS/3DGUT/3DGRT rendering will cause slight differences when rendered with
        3rd-party 3DGS viewers.
        NB : extra_signal is not exported.

        Args:
            export_dir: Directory path where the PLY file will be saved as 'model.ply'.
            format: Export format for the Gaussian data. Available options:
                - GaussianExportFormat._3DGS: Original 3D Gaussian Splatting format
                - GaussianExportFormat._3DGRT: 3DGRT format for visualization
            percentage_gaussians: Percentage of Gaussians to export (0, 100]. Only used
                for _3DGRT format. Defaults to 100 (export all Gaussians).
        """
        for track_index, track_id in enumerate(self.cuboid_tracks.tracks_id):
            track_mask = self.gaussian_cuboid_ids == track_index
            if track_mask.any():
                if format == GaussianExportFormat._3DGS:
                    write_ply_3dgs(
                        export_dir / f"{track_id}.ply",
                        self.positions[track_mask],
                        self.rotations[track_mask],
                        self.scales[track_mask],
                        self.densities[track_mask],
                        self.features_albedo[track_mask],
                        self.features_specular[track_mask],
                    )
                elif format == GaussianExportFormat._3DGRT:
                    write_ply_3dgrt(
                        export_dir / f"{track_id}.ply",
                        self.get_positions(),
                        self.get_rotations(quaternion_format="xyzw"),
                        self.get_scales(),
                        self.get_densities(),
                        percentage_gaussians=percentage_gaussians,
                    )

    def get_number_of_gaussians_per_track(self) -> dict[str, int]:
        """
        Returns the number of gaussians, if tracks are available this is returned per track.
        """
        return {
            self.cuboid_tracks.tracks_id[track_idx]: count.item()
            for track_idx, count in zip(*torch.unique(self.gaussian_cuboid_ids, return_counts=True))
        }


class DeformationNetwork(nn.Module):
    config: DictConfig
    feature_volume: HashGridObjectFeatureVolume

    def __init__(
        self,
        config: DictConfig,
        trainer_config: TrainerConfig,
        precision: int,
        cuboid_tracks: CuboidTracks,
    ):
        super().__init__()
        self.config = config
        self.trainer_config = trainer_config

        self.feature_volume = HashGridObjectFeatureVolume(
            self.config.feature_volume, trainer_config, precision, cuboid_tracks
        )

    def forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass with pre-computed concatenated embeddings.

        Args:
            embeddings: Concatenated embeddings [N, 3 + instance_dim + time_dim]

        Returns:
            Dictionary with deformation parameters (positions, rotations, scales)
        """
        out = self.feature_volume(embeddings)

        # NOTE : TCNN MLP outputs at least 16 floats
        out_dim = out.shape[-1]
        assert out_dim >= 10, f"Deformation network output dimension must be at least 10, got {out_dim}"
        positions, rotations, scales, _ = torch.split(out.float(), [3, 4, 3, out_dim - 10], dim=-1)
        ret = {"positions": positions}
        if self.config.deform_rotations:
            ret["rotations"] = rotations
        if self.config.deform_scales:
            ret["scales"] = scales

        return ret

    def configure_optimizers(self, name_prefix: str = "") -> list[OptimizerLRSchedulerConfig]:
        """Returns a list of module-owned configured optimizers (optimizers paired with an optional LR scheduler),
        which will be stepped in the main training loop, allowing the module to interact with it's owned optimizers"""

        return configure_optimizers(self.config, self.trainer_config, self, name_prefix)


class DeformableGaussianModel(RigidGaussianModel):
    deform_network: DeformationNetwork

    config: DeformableGaussiansLayerConfig  # type: ignore[assignment] # Narrow type from RigidGaussiansLayerConfig

    def __init__(
        self,
        config: DeformableGaussiansLayerConfig,
        trainer_config: TrainerConfig,
        datasource: DataSourceSummary,
        init_from_datasource: bool,
        initializer: Optional[BaseInitialization],
        precision: Optional[int] = None,
        cuboid_tracks: Optional[CuboidTracks] = None,
        start_end_timestamp_us: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__(
            config,
            trainer_config,
            datasource,
            init_from_datasource,
            initializer,
            precision,
            cuboid_tracks,
            start_end_timestamp_us,
        )

        self.config = self.config.model_copy(
            update={
                "deformnet_start_iteration": adjust_step_for_world_size(
                    trainer_config, self.config.deformnet_start_iteration
                )
            },
            deep=True,
        )
        log.info(f"DeformableGaussianModel: deformnet_start_iteration={self.config.deformnet_start_iteration}")

        aabb_extent = self.cuboid_tracks.cuboids_dims * (1.0 + self.config.cuboid_tracks_expand)
        self.aabb_extent = nn.Buffer(aabb_extent, persistent=True)
        aabb_extent = torch.max(aabb_extent, dim=1, keepdim=True).values.repeat(1, 3)
        self.aabbs = AABB3D(blb=-aabb_extent / 2.0, trf=aabb_extent / 2.0)
        self._scene_contractor = SceneContractor(None, self.aabbs, is_single=False, is_merf=False)

        self.deform_network = DeformationNetwork(
            self.config.deform_network,
            trainer_config,
            unpack_optional(precision),
            cuboid_tracks=self.cuboid_tracks,
        )

        # Frame step used to regularize the smoothness of the deformation field
        gaps = self.cuboid_tracks.tracks_timestamps_us[1:] - self.cuboid_tracks.tracks_timestamps_us[:-1]
        gaps = gaps[gaps > 0]  # To ignore negative timestamps when we switch tracks
        self.frame_step = gaps.median().item()

        self.use_deform_network = False

        # Track indices of replaced tracks. For asset replacement, we need to exclude the replaced tracks from the deformation computation
        # as the network would not be trained on some of the positions / rotations of the asset
        self._replaced_track_indices: set[int] = set()

    def get_scene_contractor(self) -> SceneContractor:
        return self._scene_contractor

    def track_ply_override(
        self,
        track_id: str,
        asset: Asset,
        transform: torch.Tensor,
        dims_offset: torch.Tensor,
    ) -> bool:
        """
        Overrides the gaussians for a track id with gaussians from an asset.
        Replaced tracks are excluded from deformation computation.
        """
        if track_id not in self.cuboid_tracks.tracks_id:
            log.warning(
                f"Failed to replace track {track_id}, not found in {self.__class__.__name__} cuboid_tracks, skipping"
            )
            return False

        track_index = self.cuboid_tracks.tracks_id.index(track_id)

        replaced = super().track_ply_override(track_id, asset, transform, dims_offset)
        if not replaced:
            return False

        self._replaced_track_indices.add(track_index)
        return True

    def restore_training_parameters(self) -> None:
        self._replaced_track_indices.clear()
        super().restore_training_parameters()

    def _compute_deform_embeddings(
        self,
        xyzs: torch.Tensor,
        instance_idx: torch.Tensor,
        timestamp_us: int,
        timestamps_delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Helper to convert positions and metadata to concatenated embeddings for deformation network.

        Args:
            xyzs: Positions [N, 3]
            instance_idx: Instance indices [N]
            timestamp_us: Single timestamp (microseconds) for all samples
            timestamps_delta: Optional per-sample timestamp delta [N] in microseconds.
                            If provided, outputs 3 embeddings stacked along batch dimension.

        Returns:
            Concatenated embeddings tensor:
            - Without timestamps_delta: [N, 3 + instance_dim + time_dim]
            - With timestamps_delta: [3*N, 3 + instance_dim + time_dim] stacked as [t, t-delta, t+delta]
        """
        from nre.models.input_embedding import (
            IndividualRemapTimeInputEmbedding,
            WeightedInstanceInputEmbedding,
        )

        instance_embedding = self.deform_network.feature_volume.instance_input_embedding
        time_embedding = self.deform_network.feature_volume.time_input_embedding

        if isinstance(instance_embedding, WeightedInstanceInputEmbedding):
            assert instance_embedding.embedding_dim == 1, (
                f"Collector only supports instance embedding dimension of 1, got {instance_embedding.embedding_dim}"
            )
            instance_embedding_weights = collect.WeightedInstanceInputEmbeddingData(
                instance_embedding_weights=instance_embedding.embedding.weight
            )
        else:
            raise NotImplementedError(
                f"Unsupported instance embedding type: {type(instance_embedding).__name__}. "
                f"Supported types: WeightedInstanceInputEmbedding"
            )

        time_embedding_config: collect.EmbeddingConfig
        if type(time_embedding) == IndividualRemapTimeInputEmbedding:
            time_embedding_config = collect.IndividualRemapTimeInputEmbeddingConfig(
                timestamps_us_ranges=time_embedding.timestamps_us_ranges,
                remap_min=time_embedding.remap_min,
                remap_max=time_embedding.remap_max,
            )
        else:
            raise NotImplementedError(
                f"Unsupported time embedding type: {type(time_embedding).__name__}. "
                f"Supported types: IndividualRemapTimeInputEmbedding"
            )

        sc = self.get_scene_contractor()
        scene_contractor = collect.SceneContractorData(
            aabb_blb=sc.aabb.blb,
            aabb_trf=sc.aabb.trf,
            degree=sc.degree,
            is_merf=sc.is_merf,
        )

        input_embedding_data = collect.InputEmbeddingData(
            xyzs=xyzs,
            instance_idx=instance_idx.to(torch.int32),
            timestamps_us=timestamp_us,
            timestamps_delta=timestamps_delta,
            scene_contractor=scene_contractor,
            instance_embedding_weights=instance_embedding_weights,
            time_embedding_config=time_embedding_config,
        )

        return self.collector.prepare_input_embeddings(input_embedding_data)

    #
    # Implementation of what is necessary for parameter collection.
    #
    def get_layer_config(self) -> collect.LayerConfigDeformable:
        layer_config = super().get_layer_config()
        assert isinstance(layer_config, collect.LayerConfigSH)
        return collect.LayerConfigDeformable(
            rotation_activation=layer_config.rotation_activation,
            scale_activation=layer_config.scale_activation,
            density_activation=layer_config.density_activation,
            fourier_features_dim=layer_config.fourier_features_dim,
            embed_config=layer_config.embed_config,
        )

    def get_layer_data(
        self, context: BaseGaussianModel.CollectionContext, gathered_parameters: dict[str, torch.Tensor]
    ) -> collect.LayerDataDeformable:
        base_layer_data = super().get_layer_data(context, gathered_parameters)

        deform_positions = None
        deform_rotations = None

        gaussian_cuboid_ids = base_layer_data.tracks_ids
        if self.use_deform_network and gaussian_cuboid_ids.numel() > 0:
            with ScopedTimer("DeformableGaussianModel/deform_network"):
                frame_timestamp_us = self.get_frame_timestamp(context.rendering_data)
                base_positions = base_layer_data.positions

                if context.is_training_batch:
                    numel_deform = base_positions.shape[0]
                    delta_range = max(2, int(self.config.smoothness_frame_steps * self.frame_step))
                    timestamp_delta = torch.randint(
                        1, delta_range, (numel_deform,), device=base_positions.device, dtype=torch.int64
                    )

                    # Compute embeddings with time deltas in a single call
                    # Output shape: [3*N, 3 + instance_dim + time_dim]
                    # Rows ordered as: [emb(t), emb(t-delta), emb(t+delta)]
                    embeddings = self._compute_deform_embeddings(
                        xyzs=base_positions,
                        instance_idx=gaussian_cuboid_ids,
                        timestamp_us=frame_timestamp_us,
                        timestamps_delta=timestamp_delta,
                    )
                    deform_parameters_batch = self.deform_network(embeddings)
                    deform_parameters = dict[str, torch.Tensor]()
                    for k, v in deform_parameters_batch.items():
                        deform_parameters[k] = v[:numel_deform]

                    position_deform = deform_parameters["positions"]
                    position_deform_before = deform_parameters_batch["positions"][numel_deform : numel_deform * 2]
                    position_deform_after = deform_parameters_batch["positions"][numel_deform * 2 :]

                    deform_smoothness = 0.5 * (position_deform_before + position_deform_after) - position_deform

                    # Pass the mask as additional data to avoid CUDA sync from boolean indexing.
                    # The mask will be used in the loss for masked reduction.
                    interpolated_mask = context.additional_data["interpolated_mask"][0]
                    keep_mask = interpolated_mask[gaussian_cuboid_ids].float().unsqueeze(1)
                    context.additional_data["deform_smoothness"] = (deform_smoothness, True)  # [N, 3]
                    context.additional_data["deform_smoothness_mask"] = (keep_mask, True)  # [N, 1]
                else:
                    embeddings = self._compute_deform_embeddings(
                        xyzs=base_positions,
                        instance_idx=gaussian_cuboid_ids,
                        timestamp_us=frame_timestamp_us,
                    )
                    deform_parameters = self.deform_network(embeddings)

                    # Zero out deformation for replaced tracks (rigid behavior)
                    if self._replaced_track_indices:
                        replaced_ids = torch.tensor(
                            list(self._replaced_track_indices),
                            device=gaussian_cuboid_ids.device,
                            dtype=gaussian_cuboid_ids.dtype,
                        )
                        replaced_mask = torch.isin(gaussian_cuboid_ids, replaced_ids)
                        deform_parameters["positions"][replaced_mask] = 0.0
                        deform_parameters["rotations"][replaced_mask] = 0.0

                # We only support deformed positions and rotations, that's the only thing used in our configs.
                # We assume we always want positions and rotations, and never scales.  We could make this
                # configurable if we wanted.
                assert "positions" in deform_parameters, "Deform positions must be in deform_parameters"
                assert self.config.deform_network.deform_rotations, "Deform rotations must be enabled"
                assert "rotations" in deform_parameters, "Deform rotations must be in deform_parameters"
                assert self.config.deform_network.rotations_from_identity, "Rotations must be from identity"
                assert not self.config.deform_network.deform_scales, "Deform scales must be disabled"
                assert "scales" not in deform_parameters, "Deform scales must not be in deform_parameters"

                deform_positions = deform_parameters["positions"].to(base_positions)
                deform_rotations = deform_parameters["rotations"].to(base_layer_data.rotations)

        return collect.LayerDataDeformable(
            positions=base_layer_data.positions,
            rotations=base_layer_data.rotations,
            scales=base_layer_data.scales,
            densities=base_layer_data.densities,
            extra_signal=base_layer_data.extra_signal,
            camera_extra_signal=base_layer_data.camera_extra_signal,
            lidar_extra_signal=base_layer_data.lidar_extra_signal,
            features_albedo=base_layer_data.features_albedo,
            features_specular=base_layer_data.features_specular,
            embed_data=base_layer_data.embed_data,
            poses=base_layer_data.poses,
            keep_mask=base_layer_data.keep_mask,
            tracks_ids=base_layer_data.tracks_ids,
            deform_positions=deform_positions,
            deform_rotations=deform_rotations,
        )

    #
    # End of implementation of what is necessary for parameter collection.
    #

    def configure_optimizers(self, name_prefix: str = "") -> list[OptimizerLRSchedulerConfig]:
        """Returns a list of module-owned configured optimizers (optimizers paired with an optional LR scheduler),
        which will be stepped in the main training loop, allowing the module to interact with it's owned optimizers"""

        return super().configure_optimizers(name_prefix) + self.deform_network.configure_optimizers(name_prefix)

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        if global_step >= self.config.deformnet_start_iteration:
            self.use_deform_network = True
            self.deform_network.feature_volume.encoding.update_step_train_batch_start(
                epoch, global_step - self.config.deformnet_start_iteration, system
            )

        return update_module_step(self.tracks_calib, epoch, global_step, system)

    def get_extra_state(self) -> dict[str, Any] | None:
        extra_state = super().get_extra_state()
        if extra_state is None:
            extra_state = {}
        extra_state["use_deform_network"] = self.use_deform_network
        return extra_state

    def set_extra_state(self, state: dict[str, Any] | None) -> None:
        if (state is not None) and "use_deform_network" in state:
            self.use_deform_network = state["use_deform_network"]
        super().set_extra_state(state)


class ProjectiveTransformDeformation(nn.Module):
    """
    Per-track linear deformation field.
    Each track has its own learnable transform (3x3 or 4x4) and optional per-frame translation.
    """

    def __init__(
        self,
        config: DictConfig,
        trainer_config: TrainerConfig,
        num_tracks: int,
        deform_matrix_size: int = 3,
    ) -> None:
        super().__init__()
        self.config = config
        self.trainer_config = trainer_config
        self.num_tracks = num_tracks
        self.deform_matrix_size = deform_matrix_size

        if deform_matrix_size == 3:
            self.W = torch.nn.Parameter(torch.eye(3).unsqueeze(0).repeat(num_tracks, 1, 1))
        elif deform_matrix_size == 4:
            self.W = torch.nn.Parameter(torch.eye(4).unsqueeze(0).repeat(num_tracks, 1, 1))
        else:
            raise NotImplementedError

        log.info(
            f"ProjectiveTransformDeformation: W shape: {self.W.shape}, num_tracks: {num_tracks}, deform_matrix_size: {deform_matrix_size}"
        )

    def forward(
        self,
        x: torch.Tensor,
        track_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input coordinates (N, 3)
            track_ids: Track indices per gaussian (N,)
        Returns:
            Deformed coordinates (N, 3)
        """

        W = self.W[track_ids]

        if self.deform_matrix_size == 4:
            x_1 = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
            xw = torch.bmm(W, x_1.unsqueeze(-1)).squeeze(-1)
            denom = xw[..., [-1]].clamp_min(1e-8)
            x_deformed = (xw / denom)[..., :-1]
        elif self.deform_matrix_size == 3:
            x_deformed = torch.bmm(W, x.unsqueeze(-1)).squeeze(-1)
        else:
            raise NotImplementedError

        return x_deformed

    def configure_optimizers(self, name_prefix: str = "") -> list[OptimizerLRSchedulerConfig]:
        """Configure optimizers for the deformation field."""

        return configure_optimizers(self.config, self.trainer_config, self, name_prefix)


class DeformableRigidAssetModel(RigidGaussianModel):
    """
    RigidGaussianModel with simple LinearDeformation integration.
    This prevents individual points from moving freely while maintaining spatial coherence.
    """

    config: ElasticLoadedAHLayerConfig  # type: ignore[assignment] # Narrow type from RigidGaussiansLayerConfig

    def __init__(
        self,
        config: ElasticLoadedAHLayerConfig,
        trainer_config: TrainerConfig,
        datasource: DataSourceSummary,
        init_from_datasource: bool,
        initializer: Optional[BaseInitialization],
        precision: Optional[int] = None,
        cuboid_tracks: Optional[CuboidTracks] = None,
        start_end_timestamp_us: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__(
            config,
            trainer_config,
            datasource,
            init_from_datasource,
            initializer,
            precision,
            cuboid_tracks,
            start_end_timestamp_us,
        )

        log.info(
            f"DeformableRigidAssetModel: initializing deformation network with num_tracks: {self.cuboid_tracks.n_tracks}, {self.cuboid_tracks.tracks_id}"
        )

        self.log_every_n_steps = self.config.log_every_n_steps or 100
        # Initialize the implicit per-track deformation field
        self.deformation_network = ProjectiveTransformDeformation(
            config=self.config.deformation_network,
            trainer_config=self.trainer_config,
            num_tracks=self.cuboid_tracks.n_tracks,
            deform_matrix_size=3 if self.config.deform_matrix_size is None else self.config.deform_matrix_size,
        )
        self.use_deformation = True
        # One-time transition logs
        self._masking_started_logged = False
        self._masking_ended_logged = False
        self._deform_started_logged = False
        self._deform_ended_logged = False
        # Resume handling
        self._resumed_from_ckpt = False
        self._first_update_done = False

        # Global compact track id mapping (raw ids -> [0..n_tracks-1]) created lazily
        self._global_track_ids_sorted: torch.Tensor | None = None  # Stored as a plain Tensor when initialized

        # Typing-only declarations to satisfy static type checkers; buffers are registered later
        # These annotations do not create runtime attributes and will not interfere with register_buffer
        self.gaussians_hit_counter: torch.Tensor  # type: ignore[assignment]
        self.gaussians_hit_mask: torch.Tensor  # type: ignore[assignment]

    def get_extra_state(self) -> dict[str, Any] | None:
        extra_state = super().get_extra_state()
        if extra_state is None:
            extra_state = {}
        extra_state["use_deformation"] = self.use_deformation
        # Persist transition flags so banners don't re-fire on resume
        extra_state["masking_started_logged"] = self._masking_started_logged
        extra_state["masking_ended_logged"] = self._masking_ended_logged
        extra_state["deform_started_logged"] = self._deform_started_logged
        extra_state["deform_ended_logged"] = self._deform_ended_logged
        return extra_state

    def set_extra_state(self, state: dict[str, Any] | None) -> None:
        if state is not None:
            if "use_deformation" in state:
                self.use_deformation = state["use_deformation"]
            # Restore flags and mark resume
            self._masking_started_logged = bool(state.get("masking_started_logged", False))
            self._masking_ended_logged = bool(state.get("masking_ended_logged", False))
            self._deform_started_logged = bool(state.get("deform_started_logged", False))
            self._deform_ended_logged = bool(state.get("deform_ended_logged", False))
            self._resumed_from_ckpt = True
        super().set_extra_state(state)

    #
    # Implementation of what is necessary for parameter collection.
    #
    def get_layer_data(
        self, context: BaseGaussianModel.CollectionContext, gathered_parameters: dict[str, torch.Tensor]
    ) -> collect.LayerDataRigid:
        layer_data = super().get_layer_data(context, gathered_parameters)
        gaussian_cuboid_ids = layer_data.tracks_ids
        if not self.use_deformation or gaussian_cuboid_ids.numel() == 0:
            # Don't apply deformation network to this layer.
            return layer_data

        with ScopedTimer("DeformableRigidAssetModel/deformation_network"):
            base_positions = layer_data.positions
            track_ids = gaussian_cuboid_ids
            # Safety: ensure indices are within deformation network range
            if track_ids.numel() > 0:
                max_idx = int(track_ids.max().item())
                if max_idx >= self.deformation_network.num_tracks:
                    raise RuntimeError(
                        f"Deformation track_ids out of range: max={max_idx} >= num_tracks={self.deformation_network.num_tracks}"
                    )
            with ScopedTimer("DeformableRigidAssetModel/run_network"):
                deformed_positions = self.deformation_network(
                    base_positions,
                    track_ids,
                )
            delta = (deformed_positions - base_positions).abs()
            mean_delta = delta.mean().item()
            max_delta = delta.max().item() if delta.numel() > 0 else 0.0
            n_tracks_in_batch = torch.unique(gaussian_cuboid_ids).numel()
            log.debug(
                f"DeformableRigidAssetModel: deformation applied | mean_abs_delta={mean_delta:.6f} "
                f"max_abs_delta={max_delta:.6f} "
                f"tracks_in_batch={n_tracks_in_batch}"
            )

            # Adding cast to float32 to avoid type mismatch errors in the collector
            # collector expects float32 tensors
            # But torch.bmm may return float16 tensors
            layer_data.positions = deformed_positions.to(torch.float32)

        return layer_data

    #
    # End of implementation of what is necessary for parameter collection.
    #

    def configure_optimizers(self, name_prefix: str = "") -> list[OptimizerLRSchedulerConfig]:
        """Configure optimizers for both base model and deformation field."""

        log.info(f"[DeformableRigidAssetModel] Configuring optimizers (name_prefix='{name_prefix}')")

        # Get base optimizers
        base_optimizers = super().configure_optimizers(name_prefix)
        log.info(f"[DeformableRigidAssetModel] Got {len(base_optimizers)} base optimizers from parent class")

        # Use the utility function to create the proper optimizer configuration
        log.info(f"[DeformableRigidAssetModel] Configuring deformation network optimizers...")
        deformation_optimizers = self.deformation_network.configure_optimizers(name_prefix=name_prefix)

        log.info(f"[DeformableRigidAssetModel] Got {len(deformation_optimizers)} deformation network optimizers")

        # Combine base optimizers with deformation optimizers
        base_optimizers.extend(deformation_optimizers)
        log.info(f"[DeformableRigidAssetModel] Total optimizers: {len(base_optimizers)}")

        # Print detailed summary of all optimizers and their parameter groups
        log.info(f"\n{'=' * 80}")
        log.info(f"[DeformableRigidAssetModel] Optimizer Configuration Summary")
        log.info(f"{'=' * 80}")
        for idx, opt_config in enumerate(base_optimizers):
            optimizer = opt_config["optimizer"]
            has_scheduler = "lr_scheduler" in opt_config
            scheduler_name = opt_config["lr_scheduler"]["scheduler"].__class__.__name__ if has_scheduler else "None"
            log.info(f"\nOptimizer {idx}: {optimizer.__class__.__name__} (scheduler: {scheduler_name})")
            for pg_idx, param_group in enumerate(optimizer.param_groups):
                pg_name = param_group.get("name", f"group_{pg_idx}")
                lr = param_group["lr"]
                num_params = len(param_group["params"])
                log.info(f"  [{pg_name}] initial_lr={lr:.6f}, num_params={num_params}")
        log.info(f"{'=' * 80}\n")

        return base_optimizers

    def get_additional_buffers(self) -> dict[str, torch.Tensor]:
        additional_buffers = super().get_additional_buffers()
        # Persist masking buffers only for ElasticLoadedAH
        if hasattr(self, "gaussians_hit_counter") and isinstance(self.gaussians_hit_counter, torch.Tensor):
            additional_buffers["gaussians_hit_counter"] = self.gaussians_hit_counter
        if hasattr(self, "gaussians_hit_mask") and isinstance(self.gaussians_hit_mask, torch.Tensor):
            additional_buffers["gaussians_hit_mask"] = self.gaussians_hit_mask
        return additional_buffers

    # Handle masking buffers on checkpoint load so they are not treated as unexpected keys
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        policy = self.config.restriction_policy
        counter_key = prefix + "gaussians_hit_counter"
        mask_key = prefix + "gaussians_hit_mask"

        has_counter = counter_key in state_dict
        has_mask = mask_key in state_dict

        if policy == "vis_frequency_drop":
            # Require presence for resume; register buffers so assignment succeeds
            if not has_counter or not has_mask:
                # Let super report missing if strict, but give clearer error
                raise RuntimeError(
                    f"Checkpoint is missing required masking buffers: present(counter={has_counter}, mask={has_mask})."
                )
            if "gaussians_hit_counter" not in self._buffers:
                self.register_buffer(
                    "gaussians_hit_counter",
                    state_dict[counter_key].to(torch.long),
                    persistent=True,
                )
            if "gaussians_hit_mask" not in self._buffers:
                val = state_dict[mask_key]
                val_bool = val if val.dtype == torch.bool else (val > 0)
                self.register_buffer(
                    "gaussians_hit_mask",
                    val_bool.to(torch.bool),
                    persistent=True,
                )
        else:
            raise NotImplementedError(f"Restriction policy {policy} not implemented")

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

        # Re-attach autograd hooks after loading (hooks are not serialized in checkpoints)
        if policy == "vis_frequency_drop":
            try:
                self.setup_grad_hooks()
            except Exception:
                # If this fails, better to raise to avoid silent masking stalls
                raise

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        # One-time resume-aware initialization for transition banners
        if not self._first_update_done:
            self._first_update_done = True
            if self._resumed_from_ckpt:
                # If resuming after masking end, suppress end banner
                if self.config.restriction_policy == "vis_frequency_drop":
                    masking_end_step = int(self.config.masking_end_step or 3000)
                    if global_step >= masking_end_step:
                        self._masking_started_logged = True
                        self._masking_ended_logged = True
                # Always keep deformation enabled across resumes
                self._deform_started_logged = True
                self._deform_ended_logged = False
                self.use_deformation = True
        # Log learning rates every 100 steps for the first 1000 steps, then every 1000 steps
        if global_step % self.log_every_n_steps == 0:
            log.debug(f"\n{'=' * 80}")
            log.debug(f"[DeformableRigidAssetModel] Learning Rates at Step {global_step}")
            log.debug(f"{'=' * 80}")
            for idx, opt_config in enumerate(self.optimizers):
                optimizer = opt_config["optimizer"]
                opt_name = f"Optimizer {idx}"
                log.debug(f"\n  {opt_name}:")
                for pg_idx, param_group in enumerate(optimizer.param_groups):
                    pg_name = param_group.get("name", f"group_{pg_idx}")
                    lr = param_group["lr"]
                    num_params = len(param_group["params"])
                    log.debug(f"    [{pg_name}] lr={lr:.6f}, num_params={num_params}")
            log.debug(f"{'=' * 80}\n")

        # Track masking start/end events (vis_frequency_drop policy)
        if self.config.restriction_policy == "vis_frequency_drop":
            masking_end_step = int(self.config.masking_end_step or 3000)
            if not self._masking_started_logged and global_step == 0:
                self._masking_started_logged = True
                log.debug(
                    f"[DeformableRigidAssetModel] ✅ Masking STARTED at step 0 (policy=vis_frequency_drop, will end at step {masking_end_step})"
                )
            if (not self._masking_ended_logged) and (global_step >= masking_end_step):
                self._masking_ended_logged = True
                # Optional quick stats snapshot
                try:
                    n = int(self.gaussians_hit_mask.numel()) if hasattr(self, "gaussians_hit_mask") else -1
                    masked = int(self.gaussians_hit_mask.sum().item()) if hasattr(self, "gaussians_hit_mask") else -1
                    log.debug(
                        f"[DeformableRigidAssetModel] 🛑 Masking ENDED at step {global_step} (n={n}, masked={masked})"
                    )
                except Exception:
                    log.debug(f"[DeformableRigidAssetModel] 🛑 Masking ENDED at step {global_step}")

        # Ensure deformation is always enabled and announce once if not already
        if not self._deform_started_logged:
            self._deform_started_logged = True
            self._deform_ended_logged = False
            self.use_deformation = True
            try:
                W = self.deformation_network.W
                log.debug(
                    f"[DeformableRigidAssetModel] ✅ Deformation ENABLED (always-on) at step {global_step} "
                    f"(W shape={tuple(W.shape)}, min={float(W.min()):.4f}, max={float(W.max()):.4f})"
                )
            except Exception:
                log.info(f"[DeformableRigidAssetModel] Deformation network failed to initialize at step {global_step}")

        if self.config.restriction_policy == "vis_frequency_drop":
            # Track current step for gated logging in hooks
            self._current_step = int(global_step)
            # Log mask stats at the first step after a resume (global_step may reset depending on restore)
            if global_step == 0:
                try:
                    n = int(self.gaussians_hit_mask.numel())
                    active = int((~self.gaussians_hit_mask).sum().item())
                    masked = int(self.gaussians_hit_mask.sum().item())
                    if hasattr(self, "gaussians_hit_counter"):
                        hit_sum = int(self.gaussians_hit_counter.sum().item())
                        hit_min = int(self.gaussians_hit_counter.min().item())
                        hit_max = int(self.gaussians_hit_counter.max().item())
                    else:
                        hit_sum, hit_min, hit_max = -1, -1, -1
                    log.info(
                        f"[MaskResumeStats][before-step] n={n}, masked={masked}, active={active}, "
                        f"hit_counter_sum={hit_sum}, hit_counter_min={hit_min}, hit_counter_max={hit_max}"
                    )
                except Exception:
                    raise RuntimeError("Error in logging mask stats at global step 0")
            # Use masking_end_step for determining when to stop updating the hit mask
            # This should match when albedo starts training (step 3000 by default)
            masking_end_step = getattr(self.config, "masking_end_step", 3000)
            if global_step > 0 and global_step < masking_end_step:
                if (global_step % self.log_every_n_steps) == 0:
                    denom = float(max(1, int(self.gaussians_hit_counter.numel())))
                    stepf = float(max(1, int(global_step)))
                    log.debug(
                        f"[DeformableRigidAssetModel] gaussian hit frequency: {self.gaussians_hit_counter.sum() / denom / stepf}"
                    )
                # Create a boolean mask tensor using explicit Tensor ops for mypy
                self.gaussians_hit_mask = torch.lt(
                    self.gaussians_hit_counter.to(torch.float32) / float(global_step), 0.2
                )

                if (global_step % self.log_every_n_steps) == 0:
                    try:
                        n = int(self.gaussians_hit_mask.numel())
                        active = int((~self.gaussians_hit_mask).sum().item())
                        masked = int(self.gaussians_hit_mask.sum().item())
                        hit_sum = int(self.gaussians_hit_counter.sum().item())
                        hit_min = int(self.gaussians_hit_counter.min().item())
                        hit_max = int(self.gaussians_hit_counter.max().item())
                        log.debug(
                            f"[MaskResumeStats][after-update] step={global_step} n={n}, masked={masked}, active={active}, "
                            f"hit_counter_sum={hit_sum}, hit_counter_min={hit_min}, hit_counter_max={hit_max}"
                        )
                    except Exception:
                        raise RuntimeError(f"Error in logging mask stats at global step {global_step}")
        else:
            raise NotImplementedError(
                f"Restriction policy {self.config.restriction_policy or 'hardcoded_masking'} not implemented"
            )

        return update_module_step(self.tracks_calib, epoch, global_step, system)

    def setup_grad_hooks(self):
        policy = self.config.restriction_policy
        if policy == "vis_frequency_drop":
            positions = self.get_positions()
            num_gaussians, _ = positions.shape
            # Register persistent buffers once during initialization
            if "gaussians_hit_counter" not in self._buffers:
                self.register_buffer(
                    "gaussians_hit_counter",
                    torch.zeros((num_gaussians,), dtype=torch.long, device=self.device),
                    persistent=True,
                )
            if "gaussians_hit_mask" not in self._buffers:
                self.register_buffer(
                    "gaussians_hit_mask",
                    torch.ones((num_gaussians,), dtype=torch.bool, device=self.device),
                    persistent=True,
                )

            def albedo_hook(grad):
                grad_norm = torch.linalg.norm(grad[:, 0], axis=1)
                self.gaussians_hit_counter[grad_norm > 4e-7] += 1

                # Throttled debug logs for gradient stats
                if hasattr(self, "_current_step") and (self._current_step % self.log_every_n_steps) == 0:
                    log.debug(
                        f"[DeformableRigidAssetModel] grad_norm quantile 0.95: {torch.quantile(grad_norm, q=0.95)}"
                    )
                    log.debug(f"[DeformableRigidAssetModel] grad_norm quantile 0.9: {torch.quantile(grad_norm, q=0.9)}")
                    log.debug(f"[DeformableRigidAssetModel] grad_norm quantile 0.8: {torch.quantile(grad_norm, q=0.8)}")
                    log.debug(f"[DeformableRigidAssetModel] grad_norm quantile 0.5: {torch.quantile(grad_norm, q=0.5)}")
                    log.debug(
                        f"[DeformableRigidAssetModel] gaussians_hit_mask sum: {self.gaussians_hit_mask.sum() / len(grad_norm)}"
                    )

                if grad is not None:
                    grad[self.gaussians_hit_mask] = 0.0
                return grad

            def freeze_grad_hook(grad):
                if grad is not None:
                    grad[self.gaussians_hit_mask] = 0.0
                return grad

            self.positions.register_hook(freeze_grad_hook)
            self.rotations.register_hook(freeze_grad_hook)
            self.densities.register_hook(freeze_grad_hook)
            self.scales.register_hook(freeze_grad_hook)
            self.features_albedo.register_hook(albedo_hook)
            self.features_specular.register_hook(freeze_grad_hook)
        else:
            raise NotImplementedError(
                f"Restriction policy {self.config.get('restiction_policy', 'hardcoded_masking')} not implemented"
            )

    def initialize_gaussians_from_datasource(
        self,
        initializer: BaseInitialization,
        datasource: DataSourceSummary,
        cuboid_tracks: Optional[CuboidTracks] = None,
    ) -> None:
        super().initialize_gaussians_from_datasource(
            initializer=initializer,
            datasource=datasource,
            cuboid_tracks=cuboid_tracks,
        )

        self.setup_grad_hooks()


BaseGaussianModel.register_to_gaussians_factory("sh-gaussians", SHGaussianModel)
BaseGaussianModel.register_to_gaussians_factory("rigid-gaussians", RigidGaussianModel)
BaseGaussianModel.register_to_gaussians_factory("deformable-gaussians", DeformableGaussianModel)
BaseGaussianModel.register_to_gaussians_factory("elastic-loaded-ah", DeformableRigidAssetModel)
