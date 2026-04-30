# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from abc import ABC, abstractmethod
from typing import List, Optional

import torch

from omegaconf import DictConfig
from torch import nn

from nre.config.model import CalibConfig
from nre.config.trainer import TrainerConfig
from nre.datasets.summary import DataSourceSummary
from nre.models.base import BaseModel
from nre.models.nn_extensions import module_call_type
from nre.utils.batch import (
    CameraFreePoseViewGeometry,
    DataBatch,
    LidarFreePoseViewGeometry,
    RenderingBatch,
    RenderingData,
)
from nre.utils.optim import OptimizerLRSchedulerConfig, configure_optimizers
from nre.utils.trainer import adjust_step_for_world_size
from nre.utils.types import RigTrajectories


log = logging.getLogger(__name__)


class BaseCalib(BaseModel, ABC):
    """Base class for all calibration models"""

    config: CalibConfig  # type: ignore[assignment] # Override type annotation from BaseModel

    def __init__(self, config: CalibConfig, trainer_config: TrainerConfig):
        super().__init__(config.to_dictconfig())
        self.config = config  # type: ignore[assignment]  # Override with typed config
        self.trainer_config = trainer_config

        self.enabled: bool = config.lidar.enabled or config.camera.enabled
        if self.enabled:
            self.start_global_step: int = adjust_step_for_world_size(trainer_config, config.start_global_step)
            self.skip_first_pose_delta: bool = config.skip_first_pose_delta

            log.info(f"BaseCalib: start_global_step={self.start_global_step}")

    @staticmethod
    def factory(
        name: str, config: CalibConfig, trainer_config: TrainerConfig, datasource: DataSourceSummary
    ) -> BaseCalib:
        if name == "free-pose-calib" or name == "skip-calib":
            if (rig_trajectories := datasource.get_rig_trajectories()) is not None:
                return FreePoseCalib.from_rig_trajectories(rig_trajectories, config, trainer_config)
            else:
                raise RuntimeError(f"Rig_trajectories required for {name} calibration")
        else:
            raise TypeError(f"Unknown calib {name=}.")

    @property
    @abstractmethod
    def params(self) -> List[nn.Parameter]: ...

    @abstractmethod
    def forward(
        self, data_batch: DataBatch, skip_calib: bool = False, global_step_for_prober: Optional[int] = None
    ) -> RenderingBatch: ...

    def configure_optimizers(self, name_prefix: str = "") -> list[OptimizerLRSchedulerConfig]:
        return configure_optimizers(self.config.to_dictconfig(), self.trainer_config, self, name_prefix)

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        if self.enabled and global_step >= self.start_global_step:
            # unfreeze parameters if enabled
            for param in self.params:
                param.requires_grad_(True)
        else:
            self.requires_grad_(False)  # freeze all unconditionally, as start step not reached yet
        return {}  # no new / additional parameters

    __call__ = module_call_type(forward)


class FreePoseCalib(BaseCalib):
    """Free pose calibration"""

    def __init__(
        self,
        config: CalibConfig,
        trainer_config: TrainerConfig,
        lidar_view_geometry: Optional[LidarFreePoseViewGeometry] = None,
        camera_view_geometry: Optional[CameraFreePoseViewGeometry] = None,
    ):
        super().__init__(config, trainer_config)
        self.lidar_view_geometry = lidar_view_geometry
        self.camera_view_geometry = camera_view_geometry
        self.enable_torch_compile: bool = config.enable_torch_compile

        calib_enabled_camera = camera_view_geometry is not None and camera_view_geometry.enable_calib
        calib_enabled_lidar = lidar_view_geometry is not None and lidar_view_geometry.enable_calib
        log.info(f"FreePoseCalib: calib {'ON' if calib_enabled_camera else 'OFF'} for cameras")
        log.info(f"FreePoseCalib: calib {'ON' if calib_enabled_lidar else 'OFF'} for lidars")
        log.info(f"FreePoseCalib: enable_torch_compile={self.enable_torch_compile}")

        assert lidar_view_geometry is not None or camera_view_geometry is not None, (
            "At least one view geometry must be provided"
        )

    @property
    def params(self) -> List[nn.Parameter]:
        parameters: list[nn.Parameter] = []
        if self.camera_view_geometry is not None and self.camera_view_geometry.enable_calib:
            assert self.camera_view_geometry.embeds is not None
            parameters.extend(list(self.camera_view_geometry.embeds.parameters()))
        if self.lidar_view_geometry is not None and self.lidar_view_geometry.enable_calib:
            assert self.lidar_view_geometry.embeds is not None
            parameters.extend(list(self.lidar_view_geometry.embeds.parameters()))
        return parameters

    def forward(
        self, data_batch: DataBatch, skip_calib: bool = False, global_step_for_prober: Optional[int] = None
    ) -> RenderingBatch:
        rendering_data_camera: Optional[RenderingData] = None
        if data_batch.camera is not None:
            assert self.camera_view_geometry is not None
            skip_camera_calib = skip_calib or (
                self.config.skip_first_pose_delta and data_batch.camera.meta[0].unique_frame_idx == 0
            )
            rendering_data_camera = self.camera_view_geometry.to_rendering_data(
                data_batch.camera,
                cache_sensor_params=True,
                skip_calib=skip_camera_calib,
                global_step_for_prober=global_step_for_prober,
                enable_torch_compile=self.enable_torch_compile,
            )

        rendering_data_lidar: Optional[RenderingData] = None
        if data_batch.lidar is not None:
            assert self.lidar_view_geometry is not None
            skip_lidar_calib = skip_calib or (
                self.config.skip_first_pose_delta and data_batch.lidar.meta[0].unique_frame_idx == 0
            )
            rendering_data_lidar = self.lidar_view_geometry.to_rendering_data(
                data_batch.lidar,
                cache_sensor_params=True,
                skip_calib=skip_lidar_calib,
                global_step_for_prober=global_step_for_prober,
                enable_torch_compile=self.enable_torch_compile,
            )

        return RenderingBatch(camera=rendering_data_camera, lidar=rendering_data_lidar)

    @staticmethod
    def from_rig_trajectories(
        rig_trajectories: RigTrajectories,
        config: CalibConfig,
        trainer_config: TrainerConfig,
    ) -> FreePoseCalib:
        """Constructs a FreePoseCalib from rig-trajectories"""
        return FreePoseCalib(
            config=config,
            trainer_config=trainer_config,
            lidar_view_geometry=LidarFreePoseViewGeometry.from_rig_trajectories(
                rig_trajectories, enable_calib=config.lidar.enabled
            ),
            camera_view_geometry=CameraFreePoseViewGeometry.from_rig_trajectories(
                rig_trajectories, enable_calib=config.camera.enabled
            ),
        )
