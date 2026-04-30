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

from abc import abstractmethod
from typing import List, Type, TypeVar

import torch
import torch.nn as nn

from omegaconf import DictConfig

from nre.config.model import (
    BasePostProcessingConfig,
    BilateralGridPostProcessingConfig,
    DINOv2PEPostProcessingConfig,
    PPISPPostProcessingConfig,
)
from nre.config.trainer import TrainerConfig
from nre.models.base import BaseModel
from nre.models.nn_extensions import module_call_type
from nre.models.post_processings.bilateral_grid.bilateral_grid import BilateralGrid
from nre.models.post_processings.bilateral_grid.bilateral_grid_cuda import BilateralGridCUDA
from nre.models.post_processings.ppisp import PPISP, BasePPISP, PPISPSlang
from nre.utils.misc import unpack_optional
from nre.utils.optim import OptimizerLRSchedulerConfig, configure_optimizers
from nre.utils.types import ExtraSignal, GaussiansCompositeReturn


logger = logging.getLogger(__name__)

from nre.utils.prober import get_global_prober


# Use the global prober instance
prober = get_global_prober()


class BasePostProcessing(BaseModel):
    variants: dict[str, Type[BasePostProcessing]] = {}
    config: BasePostProcessingConfig  # type: ignore[assignment] # Override type annotation from BaseModel

    def __init__(
        self,
        config: BasePostProcessingConfig,
        trainer_config: TrainerConfig,
        n_frames_per_camera: List[int],
        n_frames_per_lidar: List[int],
    ):
        super().__init__(config.to_dictconfig())
        self.config = config  # type: ignore[assignment] # Override with typed config
        self.trainer_config = trainer_config
        logger.info(f"{self.__call__.__name__}/n_frames_per_camera: {n_frames_per_camera}")
        logger.info(f"{self.__call__.__name__}/n_frames_per_lidar: {n_frames_per_lidar}")

    @classmethod
    def factory(
        cls,
        name: str,
        config: BasePostProcessingConfig,
        trainer_config: TrainerConfig,
        n_frames_per_camera: List[int],
        n_frames_per_lidar: List[int],
    ) -> BasePostProcessing:
        if name not in cls.variants:
            raise ValueError(f"Unknown variant: {name}")
        variant_class = cls.variants[name]
        return variant_class(
            config=config,
            trainer_config=trainer_config,
            n_frames_per_camera=n_frames_per_camera,
            n_frames_per_lidar=n_frames_per_lidar,
        )

    @classmethod
    def register_variant(cls, name: str, variant: Type[BasePostProcessing]) -> None:
        cls.variants[name] = variant

    @abstractmethod
    def forward(
        self,
        results: GaussiansCompositeReturn,
        rays: torch.Tensor | None = None,  # [N, 6]
        coords_xy: torch.Tensor | None = None,  # [N, 2]
        unique_frame_idx: int | None = None,
        unique_sensor_idx: int | None = None,
        global_step: int = 0,
    ) -> None: ...

    __call__ = module_call_type(forward)


class DinoV2PEPostProcessing(BasePostProcessing):
    """
    Positional embedding post-processing for DINOv2.
    """

    config: DINOv2PEPostProcessingConfig

    def __init__(
        self,
        config: DINOv2PEPostProcessingConfig,
        trainer_config: TrainerConfig,
        n_frames_per_camera: List[int],
        n_frames_per_lidar: List[int],
    ):
        super().__init__(config, trainer_config, n_frames_per_camera, n_frames_per_lidar)

        # Globally-shared post-processing for all cameras.
        self.n_dims = int(config.n_dims)
        self.learnable_pe_map = nn.Parameter(
            0.05
            * torch.randn(
                1,
                self.n_dims // 2,
                self.config.pe_height,
                self.config.pe_width,
            ),
            requires_grad=True,
        )
        # positional embedding head to decode the PE map to create some correalation.
        self.pe_head = nn.Sequential(
            nn.Conv2d(self.n_dims // 2, self.n_dims, 1),
        )

    def forward(
        self,
        results: GaussiansCompositeReturn,
        rays: torch.Tensor | None = None,  # [N, 6]
        coords_xy: torch.Tensor | None = None,  # [N, 2]
        unique_frame_idx: int | None = None,
        unique_sensor_idx: int | None = None,
        global_step: int = 0,
    ) -> None:
        if not self.training and self.config.test_no_pe_map:
            return

        assert (
            results.rendered_cam is not None
            and results.rendered_cam.extra_ray_signals is not None
            and results.rendered_cam.extra_ray_signals.dinov2_feats is not None
        ), "`extra_ray_signals.dinov2_feats` should not be empty if `self.signal_type` is `dinov2_feats` "

        if rays is not None:
            assert coords_xy is not None, (
                f"[{self.__class__.__name__}] No camera meta information found for camera rays!"
            )

            pe_map = self.pe_head(self.learnable_pe_map)
            pe_ray = (
                (
                    torch.nn.functional.grid_sample(
                        pe_map,
                        coords_xy.reshape(1, 1, -1, 2) * 2 - 1,
                        align_corners=False,
                        mode="bilinear",
                        padding_mode="border",
                    )
                )
                .squeeze(2)
                .squeeze(0)
                .permute(1, 0)
            )

            dinov2_feats = results.rendered_cam.extra_ray_signals.dinov2_feats
            results.rendered_cam.extra_ray_signals.dinov2_feats = dinov2_feats + pe_ray


BasePostProcessing.register_variant("dinov2-pe-post-processing", DinoV2PEPostProcessing)


def create_bilateral_grid(config: BilateralGridPostProcessingConfig, n_frames: int):
    logger.info(f"Creating bilateral grid with {n_frames} frames")
    use_cuda = getattr(config, "use_cuda", True)
    CurrentBilateralGrid = BilateralGridCUDA if use_cuda else BilateralGrid
    return CurrentBilateralGrid(
        n_frames,
        config.width,
        config.height,
        config.depth,
        getattr(config, "enable_gridsize1_optimization", True),
    )


class BilateralGridPerFrame(BasePostProcessing):
    """
    Implementation of "Bilateral Guided Radiance Field Processing (Bilarf)".
    This approach follows the published work and learns one bilateral grid per frame.
    """

    config: BilateralGridPostProcessingConfig

    def __init__(
        self,
        config: BilateralGridPostProcessingConfig,
        trainer_config: TrainerConfig,
        n_frames_per_camera: List[int],
        n_frames_per_lidar: List[int],
    ):
        super().__init__(config, trainer_config, n_frames_per_camera, n_frames_per_lidar)

        self.bilateral_grid = create_bilateral_grid(self.config, sum(n_frames_per_camera))

        # This is used so that we don't apply temporal variation loss across frames belonging to different cameras
        self.same_camera_as_next_frame = nn.Buffer(
            torch.ones(sum(n_frames_per_camera) - 1, dtype=torch.bool), persistent=False
        )
        offset = 0
        for camera_n_frames in n_frames_per_camera[:-1]:
            self.same_camera_as_next_frame[offset + camera_n_frames - 1] = 0
            offset += camera_n_frames

    def forward(
        self,
        results: GaussiansCompositeReturn,
        rays: torch.Tensor | None = None,  # [N, 6]
        coords_xy: torch.Tensor | None = None,  # [N, 2]
        unique_frame_idx: int | None = None,
        unique_sensor_idx: int | None = None,
        global_step: int = 0,
    ) -> None:
        if rays is None:
            return

        assert results.rendered_cam is not None, "rendered_cam should not be None if rays is not None"
        if results.rendered_cam.extra_ray_signals is None:
            results.rendered_cam.extra_ray_signals = ExtraSignal()

        if results.rendered_cam.rgb is None:
            return
        rgb = unpack_optional(results.rendered_cam.rgb)

        # We check if rgb_before_post_processing is None since we potentially apply bilateral
        # grids both at on per-frame or per-camera level, and we want to store the rgb values
        # before any sort of post-processing takes place
        if results.rendered_cam.extra_ray_signals.rgb_before_post_processing is None:
            results.rendered_cam.extra_ray_signals.rgb_before_post_processing = rgb.clone()

        # Apply to camera rays.
        assert coords_xy is not None, f"{self.__class__.__name__} requires coords_xy"
        assert unique_frame_idx is not None, f"{self.__class__.__name__} requires unique_frame_idx"
        frame_idcs = torch.full((rgb.shape[0],), unique_frame_idx, dtype=torch.int32, device=rgb.device)
        results.rendered_cam.rgb = self.bilateral_grid(rgb, coords_xy, frame_idcs)

        if (
            prober_result := prober(
                global_step,
                "bilateral_grid_per_frame",
                rgb=rgb,
                coords_xy=coords_xy,
                grid_idcs=frame_idcs,
                bilateral_grid=self.bilateral_grid.grid,
                output_rgb_grad=results.rendered_cam.rgb,
            )
        ) is not None:
            # Connect the gradient probing to the rest of the computation graph
            (results.rendered_cam.rgb,) = prober_result

    def configure_optimizers(self, name_prefix: str = "") -> List[OptimizerLRSchedulerConfig]:
        return configure_optimizers(self.config.to_dictconfig(), self.trainer_config, self, name_prefix)


BasePostProcessing.register_variant("bilateral-grid-per-frame", BilateralGridPerFrame)


class BilateralGridPerCamera(BasePostProcessing):
    """
    Learn one bilateral grid per camera.
    """

    config: BilateralGridPostProcessingConfig

    def __init__(
        self,
        config: BilateralGridPostProcessingConfig,
        trainer_config: TrainerConfig,
        n_frames_per_camera: List[int],
        n_frames_per_lidar: List[int],
    ):
        super().__init__(config, trainer_config, n_frames_per_camera, n_frames_per_lidar)

        self.num_cameras = len(n_frames_per_camera)
        if self.num_cameras == 1:
            logger.warning(
                "Single camera training with per-camera bilateral grids may reduce quality! It's recommended to disable per-camera bilateral grids in this configuration."
            )

        self.bilateral_grid = create_bilateral_grid(self.config, self.num_cameras)

    def forward(
        self,
        results: GaussiansCompositeReturn,
        rays: torch.Tensor | None = None,  # [N, 6]
        coords_xy: torch.Tensor | None = None,  # [N, 2]
        unique_frame_idx: int | None = None,
        unique_sensor_idx: int | None = None,
        global_step: int = 0,
    ) -> None:
        if rays is None:
            return

        assert results.rendered_cam is not None, "rendered_cam should not be None if rays_cam is not None"
        if results.rendered_cam.extra_ray_signals is None:
            results.rendered_cam.extra_ray_signals = ExtraSignal()

        if results.rendered_cam.rgb is None:
            return
        rgb = unpack_optional(results.rendered_cam.rgb)

        # We check if rgb_before_post_processing is None since we potentially apply bilateral
        # grids both at on per-frame or per-camera level, and we want to store the rgb values
        # before any sort of post-processing takes place
        if results.rendered_cam.extra_ray_signals.rgb_before_post_processing is None:
            results.rendered_cam.extra_ray_signals.rgb_before_post_processing = rgb.clone()

        assert unique_sensor_idx is not None, f"{self.__class__.__name__} requires unique_sensor_idx"
        assert unique_sensor_idx < self.num_cameras, "Camera indices out of bounds"
        sensor_idcs = torch.full((rgb.shape[0],), unique_sensor_idx, dtype=torch.int32, device=rgb.device)
        assert coords_xy is not None, f"{self.__class__.__name__} requires coords_xy"
        transformed_rgb = self.bilateral_grid(rgb, coords_xy, sensor_idcs)
        results.rendered_cam.rgb = transformed_rgb

        if (
            prober_result := prober(
                global_step,
                "bilateral_grid_per_camera",
                rgb=rgb,
                coords_xy=coords_xy,
                grid_idcs=sensor_idcs,
                bilateral_grid=self.bilateral_grid.grid,
                output_rgb_grad=results.rendered_cam.rgb,
            )
        ) is not None:
            # Connect the gradient probing to the rest of the computation graph
            (results.rendered_cam.rgb,) = prober_result

    def configure_optimizers(self, name_prefix: str = "") -> List[OptimizerLRSchedulerConfig]:
        return configure_optimizers(self.config.to_dictconfig(), self.trainer_config, self, name_prefix)


BasePostProcessing.register_variant("bilateral-grid-per-camera", BilateralGridPerCamera)

BilateralGridT = TypeVar("BilateralGridT", BilateralGridPerFrame, BilateralGridPerCamera)


class PPISPPostProcessing(BasePostProcessing):
    """
    Post-processing module based on physically plausible ISP pipeline.
    """

    config: PPISPPostProcessingConfig

    def __init__(
        self,
        config: PPISPPostProcessingConfig,
        trainer_config: TrainerConfig,
        n_frames_per_camera: list[int],
        n_frames_per_lidar: list[int],
    ):
        super().__init__(config, trainer_config, n_frames_per_camera, n_frames_per_lidar)

        self.num_cameras = len(n_frames_per_camera)
        self.num_frames = sum(n_frames_per_camera)

        self.ppisp: BasePPISP
        if self.config.use_slang:
            self.ppisp = PPISPSlang(self.device, n_frames_per_camera, self.num_cameras, self.num_frames)
        else:
            self.ppisp = PPISP(self.device, n_frames_per_camera, self.num_cameras, self.num_frames)

        # if per_frame = disabled, we forward -1 frame index to ppisp
        self.per_frame_ppisp_enabled = self.config.per_frame_ppisp_enabled

    def forward(
        self,
        results: GaussiansCompositeReturn,
        rays: torch.Tensor | None = None,  # [N, 6]
        coords_xy: torch.Tensor | None = None,  # [N, 2]
        unique_frame_idx: int | None = None,
        unique_sensor_idx: int | None = None,
        global_step: int = 0,
    ) -> None:
        if rays is None:
            return

        assert results.rendered_cam is not None, "rendered_cam should not be None if rays_cam is not None"
        if results.rendered_cam.extra_ray_signals is None:
            results.rendered_cam.extra_ray_signals = ExtraSignal()

        if results.rendered_cam.rgb is None:
            return
        rgb = unpack_optional(results.rendered_cam.rgb)

        # We check if rgb_before_post_processing is None since we potentially apply bilateral
        # grids both at on per-frame or per-camera level, and we want to store the rgb values
        # before any sort of post-processing takes place
        if results.rendered_cam.extra_ray_signals.rgb_before_post_processing is None:
            results.rendered_cam.extra_ray_signals.rgb_before_post_processing = rgb.clone()

        # Extract image coordinates of each ray in [0, 1]
        assert coords_xy is not None, f"{self.__class__.__name__} requires coords_xy"

        # Extract camera and frame indices of each ray
        # Camera indices need to be cast to int16 so that they can be used for indexing.
        assert unique_sensor_idx is not None, f"{self.__class__.__name__} requires unique_sensor_idx"
        assert unique_sensor_idx < self.num_cameras, "Camera indices out of bounds"
        unique_camera_idcs = torch.full((rgb.shape[0],), unique_sensor_idx, dtype=torch.int16, device=rgb.device)

        assert unique_frame_idx is not None, f"{self.__class__.__name__} requires unique_frame_idx"
        assert unique_frame_idx < self.num_frames, "Frame indices out of bounds"
        if not self.per_frame_ppisp_enabled:
            unique_frame_idx = -1  # -1 skips frame-specific optimizations
        frame_idcs = torch.full((rgb.shape[0],), unique_frame_idx, dtype=torch.int32, device=rgb.device)

        # Apply ISP and write processed rgb to output
        transformed_rgb = self.ppisp(rgb, coords_xy, unique_camera_idcs, frame_idcs)

        if (
            prober_result := prober(
                global_step,
                "ppisp",
                rgb=rgb,
                coords_xy=coords_xy,
                unique_camera_idcs=unique_camera_idcs,
                frame_idcs=frame_idcs,
                output_rgb_grad=transformed_rgb,
            )
        ) is not None:
            # Connect the gradient probing to the rest of the computation graph
            (transformed_rgb,) = prober_result

        results.rendered_cam.rgb = transformed_rgb

    def configure_optimizers(self, name_prefix: str = "") -> List[OptimizerLRSchedulerConfig]:
        return configure_optimizers(self.config.to_dictconfig(), self.trainer_config, self, name_prefix)


BasePostProcessing.register_variant("ppisp-post-processing", PPISPPostProcessing)
