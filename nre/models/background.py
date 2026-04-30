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
from enum import Enum
from typing import Any, Optional, Tuple, Type, Union

import cv2
import numpy as np
import nvdiffrast.torch as dr
import tinycudann as tcnn
import torch

from omegaconf import DictConfig
from simple_lama_inpainting import SimpleLama
from torch import nn

from nre.config.base_schema import config_to_primitive
from nre.config.model import (
    BackgroundColorConfig,
    BaseBackgroundConfig,
    ExtraSignalSkyMlpBackgroundConfig,
    SkyEnvMapBackgroundConfig,
    SkyMlpBackgroundConfig,
)
from nre.config.trainer import TrainerConfig
from nre.models.base import BaseModel
from nre.models.nn_extensions import module_call_type
from nre.models.utils import eval_tcnn_network, get_activation
from nre.utils.geometry import cartesian_to_spherical
from nre.utils.misc import linear_to_srgb, srgb_to_linear, unpack_optional
from nre.utils.trainer import adjust_step_for_world_size
from nre.utils.types import ExtraSignal, GaussiansCompositeReturn


class BaseBackground(BaseModel, ABC):
    composite_in_linear_space: bool
    config: BaseBackgroundConfig  # type: ignore[assignment] # Override type annotation from BaseModel

    @staticmethod
    def factory(name: str, config: BaseBackgroundConfig, trainer_config: TrainerConfig) -> BaseBackground:
        variants: dict[str, Type[BaseBackground]] = {
            "sky-mlp": SkyMlpBackground,
            "sky-env-map": SkyEnvMapBackground,
            "extra-signal-sky-mlp": ExtraSignalSkyMlpBackground,
            "background-color": BackgroundColor,
            "skip-background": SkipBackground,
        }
        return variants[name](config, trainer_config)

    def __init__(self, config: BaseBackgroundConfig, trainer_config: TrainerConfig) -> None:
        super().__init__(config.to_dictconfig())
        self.config = config  # type: ignore[assignment] # Override with typed config
        self.trainer_config = trainer_config
        self.composite_in_linear_space = self.config.composite_in_linear_space

    def combine_rgb_colors(self, results: GaussiansCompositeReturn, background_rgb: torch.Tensor) -> None:
        """Compute the linear combination of RGB colors within the volume and background"""
        assert results.rendered_cam is not None, "rendered_cam should not be None"
        assert results.rendered_cam.rgb is not None, "rendered_cam.rgb should not be None"
        if results.rendered_cam.extra_ray_signals is None:
            results.rendered_cam.extra_ray_signals = ExtraSignal()

        if self.composite_in_linear_space:
            rgb_linear = srgb_to_linear(results.rendered_cam.rgb)
            background_rgb_linear = srgb_to_linear(background_rgb)
            background_rgb_linear_scaled = background_rgb_linear * (1.0 - results.rendered_cam.opacity[..., None])
            results.rendered_cam.rgb = linear_to_srgb(rgb_linear + background_rgb_linear_scaled)
            results.rendered_cam.extra_ray_signals.rgb_background = background_rgb_linear
        else:
            background_color = background_rgb * (1.0 - results.rendered_cam.opacity[..., None])
            results.rendered_cam.rgb = results.rendered_cam.rgb + background_color
            results.rendered_cam.extra_ray_signals.rgb_background = background_rgb

    @abstractmethod
    def forward(
        self,
        rays_d: torch.Tensor,
        results: GaussiansCompositeReturn,
        is_training: bool,
    ) -> None: ...

    __call__ = module_call_type(forward)


class EnvMapType(Enum):
    EQUIRECTANGULAR = "equirectangular"
    CUBEMAP = "cubemap"


class SkyEnvMapBackground(BaseBackground):
    """
    Models the background with an environment map which can either be modeled as a cubemap or with equirectangular projection.
    We also support inpainting unobserved regions of the scene (which we track via gradient updates) through Lama.
    """

    config: SkyEnvMapBackgroundConfig  # narrow from BaseBackgroundConfig

    def __init__(self, config: SkyEnvMapBackgroundConfig, trainer_config: TrainerConfig):
        super().__init__(config, trainer_config)

        self.activation = (
            get_activation("saturate") if getattr(config, "saturate_radiance", True) else get_activation("relu")
        )

        self.envmap_type = EnvMapType(config.envmap_type)

        self.min_grad_updates = adjust_step_for_world_size(trainer_config, self.config.min_grad_updates)
        logging.getLogger(__name__).info(f"SkyEnvMapBackground: min_grad_updates={self.min_grad_updates}")

        match self.envmap_type:
            case EnvMapType.EQUIRECTANGULAR:
                self.textures = torch.nn.Parameter(
                    torch.full((1, self.config.height, self.config.width, 3), 0.5), requires_grad=True
                )
            case EnvMapType.CUBEMAP:
                assert self.config.height == self.config.width, (
                    f"[{self.__class__.__name__}] Height and width must be identical when using cubemap"
                )
                self.textures = torch.nn.Parameter(
                    torch.full((1, 6, self.config.height, self.config.width, 3), 0.5), requires_grad=True
                )

        self.texture_grads = torch.nn.Buffer(torch.zeros_like(self.textures.squeeze(0)[..., 0]))

        # Tracks how many times the environment map has been updated. We generally don't want to inpaint the environment map
        # until the scene has converged somewhat
        self.n_grad_updates = 1  # Keep as a plain int. Using nn.Buffer(torch.tensor(...)) causes a GPU to CPU sync during conditional checks (e.g., if/assert).

        # NRE: x forward, y left, z up
        # OpenGL: x right, y up, z back
        # This is not required for volume rendering correctness but makes it easier to reason about nvdiffrast
        # and inpainting
        self.to_opengl = nn.Buffer(torch.FloatTensor([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]), persistent=False)

        self.original_textures = (
            nn.Buffer(torch.zeros_like(self.textures), persistent=False) if self.config.should_inpaint else None
        )

    def get_extra_state(self) -> dict[str, Any]:
        return {"n_grad_updates": self.n_grad_updates}

    def set_extra_state(self, state_dict: dict[str, Any]) -> None:
        if "n_grad_updates" in state_dict:
            self.n_grad_updates = state_dict["n_grad_updates"]
        else:
            logging.getLogger(__name__).warning(
                f"[{self.__class__.__name__}] Key 'n_grad_updates' not found in state_dict. "
                "This may occur when loading checkpoints from older versions. Using default value of 1."
            )
            self.n_grad_updates = 1

    def forward(
        self,
        rays_d: torch.Tensor,
        results: GaussiansCompositeReturn,
        is_training: bool,
    ) -> None:
        assert results.rendered_cam is not None, "rendered_cam should not be None"
        rays_d = rays_d @ self.to_opengl.T

        match self.envmap_type:
            case EnvMapType.EQUIRECTANGULAR:
                phi, theta, _ = cartesian_to_spherical(rays_d)
                uv = torch.cat([(phi / torch.pi * 2 - 1).unsqueeze(-1), theta.unsqueeze(-1) / torch.pi], -1)
                background_rgb = dr.texture(
                    self.textures,
                    uv.unsqueeze(0).unsqueeze(0),
                    filter_mode="linear",
                    boundary_mode="wrap",
                )
            case EnvMapType.CUBEMAP:
                background_rgb = dr.texture(
                    self.textures,
                    rays_d.float().unsqueeze(0).unsqueeze(0),
                    filter_mode="linear",
                    boundary_mode="cube",
                )

        assert isinstance(background_rgb, torch.Tensor)
        background_rgb = self.activation(background_rgb.squeeze(0).squeeze(0))

        if is_training:
            if self.n_grad_updates > self.min_grad_updates:
                grad = torch.autograd.grad(
                    background_rgb,
                    self.textures,
                    (1 - results.rendered_cam.opacity)
                    .unsqueeze(-1)
                    .expand(*([-1] * len(results.rendered_cam.opacity.shape)), 3),
                    retain_graph=True,
                )[0].detach()

                # This breaks in DDP training. We need to perform a average reduction on the gradient.
                # TODO: This is a hacky fix. We need to find a better way to handle this.
                if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                    grad = grad / torch.distributed.get_world_size()
                    torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM)

                self.texture_grads.copy_(torch.maximum(grad.squeeze(0).mean(dim=-1), self.texture_grads))

            self.n_grad_updates += 1

        self.combine_rgb_colors(results, background_rgb)

    def maybe_inpaint(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Potentially inpaints the textures and returns the inpainted textures and mask for logging.
        The inpainted textures will then be used until restore_original_textures is called (typically when
        resuming training).
        """

        if not self._should_inpaint:
            return None

        unpack_optional(self.original_textures).copy_(self.textures)
        match self.envmap_type:
            case EnvMapType.EQUIRECTANGULAR:
                texture_grads = self.texture_grads.data
            case EnvMapType.CUBEMAP:
                texture_grads = self._flatten_cube(self.texture_grads.data)

        inpaint_mask = texture_grads < self.config.inpaint_threshold

        kernel = np.ones((self.config.inpaint_kernel_size, self.config.inpaint_kernel_size), np.uint8)
        inpaint_mask = torch.BoolTensor(cv2.dilate(inpaint_mask.byte().cpu().numpy(), kernel, iterations=1))

        textures = self.flattened_textures().clamp(0, 1)
        simple_lama = SimpleLama(textures.device)
        inpainted_textures = (
            simple_lama.model(
                textures.permute(2, 0, 1).unsqueeze(0),
                inpaint_mask.to(textures.device).unsqueeze(0).unsqueeze(0).float(),
            )
            .detach()
            .squeeze(0)
            .permute(1, 2, 0)
        )

        match self.envmap_type:
            case EnvMapType.EQUIRECTANGULAR:
                self.textures.copy_(inpainted_textures.unsqueeze(0))
            case EnvMapType.CUBEMAP:
                inpainted_texture_list = []
                for row in range(2):
                    for col in range(3):
                        inpainted_texture_list.append(
                            inpainted_textures[
                                row * self.textures.shape[2] : (row + 1) * self.textures.shape[2],
                                col * self.textures.shape[3] : (col + 1) * self.textures.shape[3],
                            ]
                        )
                self.textures.copy_(torch.stack(inpainted_texture_list).unsqueeze(0))

        return inpainted_textures, inpaint_mask

    def restore_original_textures(self) -> None:
        if self._should_inpaint:
            self.textures.copy_(unpack_optional(self.original_textures))

    def flattened_textures(self) -> torch.Tensor:
        textures = self.textures.squeeze(0)
        match self.envmap_type:
            case EnvMapType.EQUIRECTANGULAR:
                return textures
            case EnvMapType.CUBEMAP:
                return self._flatten_cube(textures)

    @property
    def _should_inpaint(self) -> bool:
        return self.config.should_inpaint and self.n_grad_updates > self.min_grad_updates

    def _flatten_cube(self, tensor: torch.Tensor) -> torch.Tensor:
        """Flattens a (6, H, W, ...) cube map into a (2 * H, 3 * W, ...) tensor"""
        return torch.cat([torch.cat([x for x in tensor[:3]], dim=1), torch.cat([x for x in tensor[3:]], dim=1)], dim=0)

    __call__ = module_call_type(forward)


class SkyMlpBackground(BaseBackground):
    config: SkyMlpBackgroundConfig  # narrow from BaseBackgroundConfig

    def __init__(self, config: SkyMlpBackgroundConfig, trainer_config: TrainerConfig):
        super().__init__(config, trainer_config)

        self.dir_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config=config_to_primitive(self.config.dir_encoding_config),
            dtype=torch.float,
            seed=self.config.seed,
        )
        self.dir_encoding.jit_fusion = self.config.enable_jit_if_supported and tcnn.supports_jit_fusion()

        self.sky_mlp = tcnn.Network(
            n_input_dims=self.dir_encoding.n_output_dims,
            n_output_dims=3,
            network_config=config_to_primitive(self.config.mlp_network_config),
            seed=self.config.seed,
        )
        self.sky_mlp.jit_fusion = self.config.enable_jit_if_supported and tcnn.supports_jit_fusion()

    def forward(
        self,
        rays_d: torch.Tensor,
        results: GaussiansCompositeReturn,
        is_training: bool,
    ) -> None:
        dir_encoding = self.dir_encoding((rays_d.reshape(-1, 3) + 1) / 2)  # rays_d assumed to be normalized
        background_rgb = eval_tcnn_network(self.sky_mlp, dir_encoding).float()

        self.combine_rgb_colors(results, background_rgb)

    __call__ = module_call_type(forward)


class ExtraSignalSkyMlpBackground(SkyMlpBackground):
    config: ExtraSignalSkyMlpBackgroundConfig  # narrow from SkyMlpBackgroundConfig

    def __init__(self, config: ExtraSignalSkyMlpBackgroundConfig, trainer_config: TrainerConfig):
        super().__init__(config, trainer_config)

        self.signal_type = self.config.extra_signal.type

        self.extra_signal_sky_mlp = tcnn.Network(
            n_input_dims=self.dir_encoding.n_output_dims,
            n_output_dims=self.config.extra_signal.n_output_dims,
            network_config=config_to_primitive(self.config.extra_signal.mlp_network_config),
            seed=self.config.seed,
        )
        self.extra_signal_sky_mlp.jit_fusion = self.config.enable_jit_if_supported and tcnn.supports_jit_fusion()

    def forward(
        self,
        rays_d: torch.Tensor,
        results: GaussiansCompositeReturn,
        is_training: bool,
    ) -> None:
        super().forward(rays_d, results, is_training)

        # Input order: [v, h]
        dir_encoding = self.dir_encoding((rays_d.reshape(-1, 3) + 1) / 2)  # rays_d assumed to be normalized
        background_extra_signal = eval_tcnn_network(self.extra_signal_sky_mlp, dir_encoding).float()

        assert results.rendered_cam is not None and results.rendered_cam.extra_ray_signals is not None
        match self.signal_type:
            case "semantic_logits":
                results.rendered_cam.extra_ray_signals.semantic_logits = (
                    results.rendered_cam.extra_ray_signals.semantic_logits
                    + background_extra_signal * (1.0 - results.rendered_cam.opacity[..., None])
                )
            case "dinov2_feats":
                results.rendered_cam.extra_ray_signals.dinov2_feats = (
                    results.rendered_cam.extra_ray_signals.dinov2_feats
                    + background_extra_signal * (1.0 - results.rendered_cam.opacity[..., None].detach())
                )
                # Background is actually the mask.
                results.rendered_cam.extra_ray_signals.dinov2_mask = results.rendered_cam.opacity > 0.8
            case _:
                raise ValueError(f"[{self.__class__.__name__}] unknown signal type {self.signal_type}")

    __call__ = module_call_type(forward)


class BackgroundColor(BaseBackground):
    config: BackgroundColorConfig  # narrow from BaseBackgroundConfig

    def __init__(self, config: BackgroundColorConfig, trainer_config: TrainerConfig):
        super().__init__(config, trainer_config)

        self.background_color_type = self.config.color

        assert self.background_color_type in [
            "white",
            "black",
            "random",
        ], f"[{self.__class__.__name__}] Background color must be one of 'white', 'black', 'random'"

        self.color: Optional[nn.Buffer]
        if self.background_color_type == "white":
            self.color = nn.Buffer(torch.ones((3,), dtype=torch.float32, device=self.device), persistent=False)
        elif self.background_color_type == "black":
            self.color = nn.Buffer(torch.zeros((3,), dtype=torch.float32, device=self.device), persistent=False)
        else:
            self.color = None

        self.random_eval_color: Optional[nn.Buffer]
        if self.background_color_type == "random":
            self.random_eval_color = nn.Buffer(torch.FloatTensor(self.config.random_eval_color), persistent=False)
            assert self.random_eval_color.min() >= 0 and self.random_eval_color.max() <= 1, (
                "random_eval_color values should be between 0 and 1"
            )
        else:
            self.random_eval_color = None

    def forward(
        self,
        rays_d: torch.Tensor,
        results: GaussiansCompositeReturn,
        is_training: bool,
    ) -> None:
        if self.background_color_type == "random":
            if is_training:
                color = torch.rand((3,), dtype=torch.float32, device=self.device)
            else:
                color = unpack_optional(self.random_eval_color)
        else:
            color = unpack_optional(self.color)

        self.combine_rgb_colors(results, color)

    __call__ = module_call_type(forward)


class SkipBackground(BaseBackground):
    def forward(
        self,
        rays_d: torch.Tensor,
        results: GaussiansCompositeReturn,
        is_training: bool,
    ) -> None:
        pass

    __call__ = module_call_type(forward)
