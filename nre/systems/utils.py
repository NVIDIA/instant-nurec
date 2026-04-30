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

from typing import TYPE_CHECKING, Union

import torch

from omegaconf import DictConfig

from nre.datamodules.base import BaseDataModule


if TYPE_CHECKING:
    from libs.losses.orchestration.loss_aggregator import LossAggregator
    from nre.models.base import BaseModel


def update_module_step(
    m: Union[BaseModel, LossAggregator], epoch: int, global_step: int, system, **kwargs
) -> dict[str, torch.Tensor]:
    additional_parameters = m.update_step_train_batch_start(epoch, global_step, system, **kwargs)
    return additional_parameters


def update_datamodule_epoch(dm: BaseDataModule, epoch: int, system, **kwargs) -> None:
    dm.update_epoch(epoch, system, **kwargs)


def system_config_compatibility_check(config: DictConfig) -> None:
    """
    Performs system-level compatibility checks of the system's config. If settings are not compatible an assert is triggered.
    """

    # Check that every sampler config contains update_n_epochs
    if samplers_config := config.dataset.get("samplers"):
        for batch_sampler, sampler_config in samplers_config.items():
            if not hasattr(sampler_config, "update_n_epochs"):
                raise ValueError(f"Required key 'update_n_epochs' is missing from {batch_sampler} config.")
            elif type(update_interval := sampler_config.get("update_n_epochs")) is not int or (update_interval < 0):
                raise TypeError(f"Key 'update_n_epochs' in {batch_sampler} must be a non-negative integer.")

    # Check that the sum of all ratios of all samplers is 1.0
    if samplers_config := config.dataset.get("samplers"):
        camera_rays_ratio = lidar_rays_ratio = 0
        n_camera_sampler = n_lidar_samplers = 0
        # Some samplers can return both camera and lidar rays so we need to check both
        for batch_sampler in samplers_config.values():
            if (ratio_camera_sampler := batch_sampler.get("ratio_camera_samples", 0)) > 0:
                camera_rays_ratio += ratio_camera_sampler
                n_camera_sampler += 1

            if (ratio_lidar_sampler := batch_sampler.get("ratio_lidar_samples", 0)) > 0:
                lidar_rays_ratio += ratio_lidar_sampler
                n_lidar_samplers += 1

        if any([batch_sampler.camera_pixel_sampler.name == "image-crop" for batch_sampler in samplers_config.values()]):
            assert n_camera_sampler == 1, "Image crop sampler cannot be combined with other camera ray samplers"

        if config.dataset.n_train_sample_camera_rays and abs(camera_rays_ratio - 1.0) > 0.01:
            raise ValueError(
                f"Sum of the batch ratios of all camera batch samplers has to be 1.0 (got {camera_rays_ratio}, tolerance 0.01)"
            )

        if config.dataset.n_train_sample_lidar_rays and abs(lidar_rays_ratio - 1.0) > 0.01:
            raise ValueError(
                f"Sum of the batch ratios of all lidar batch samplers has to be 1.0 (got {lidar_rays_ratio}, tolerance 0.01)"
            )

    if hasattr(config.model, "gaussians"):
        if config.model.gaussians.name == "sh-gaussians":
            # Checks for the gradient based densification process
            assert samplers_config is not None and any(
                [batch_sampler.camera_pixel_sampler.name == "image-crop" for batch_sampler in samplers_config.values()]
            ), "not compatible with gradient-based densification for 3D Gaussians"
            assert config.trainer.precision == 32, "not compatible with gradient-based densification for 3D Gaussians"
