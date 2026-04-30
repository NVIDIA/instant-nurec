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

import contextlib
import logging

from abc import ABC
from typing import Any, Optional, Type

import numpy as np
import tinycudann as tcnn
import torch
import torch.nn as nn

from omegaconf import DictConfig

from nre.config.trainer import TrainerConfig
from nre.models.base import BaseModel
from nre.models.nn_extensions import module_call_type
from nre.utils.misc import precision_to_dtype, unpack_optional, zip_nested_dict
from nre.utils.trainer import adjust_step_for_world_size
from nre.utils.types import ModelInput, ModuleRef, SceneContractor


log = logging.getLogger(__name__)


class TcnnMultiLevelEncoding(nn.Module):
    """
    Encapsulates various hash-grid variants (progressive, fused, LoD) with or without MLP,
    also includes common TCNN module configurations and setups.
    """

    n_input_dims: int  # The number of input dimensions
    n_pos_dims: int = 3  # The number of position dimensions
    n_output_dims: int  # The number of output dimensions

    n_levels: int  # The number of levels
    n_active_levels: int  # The number of currently active levels
    n_features_per_level: int  # The number of features per level
    grid_base_res: float  # The base resolution / scale of the grid
    grid_max_res: float  # The max resolution / scale of the grid
    grid_res_list: np.ndarray  # The list of resolution / scale per level

    encoding_has_level_input: bool  # Whether the encoding takes `level` as input (True for LoD or Progressive)
    encoding_config: dict  # The final composed encoding dict
    encoding_lod_config: Optional[dict] = None  # Configs for encoding LoD support
    encoding_prog_config: Optional[dict] = None  # Configs for encoding progressive support

    mlp_network_config: Optional[dict] = None  # The network configuration dict
    network_n_input_dims: Optional[int] = None  # The number of input dimensions of the network
    network_n_neurons: Optional[int] = None  # The number of neurons of the hidden layers of the network
    network_n_hidden_layers: Optional[int] = None  # The number of hidden layers of the network

    is_fused: bool  # Whether the current encoding is fused-hash-grid-with-mlp or just hash-grid
    is_progressive: bool  # Whether the current encoding has progressive behavior
    is_lod: bool  # Whether the current encoding has LoD support

    def __init__(
        self,
        n_input_dims: int,
        encoding_config: dict,
        trainer_config: TrainerConfig,
        n_pos_dims: int = 3,
        n_output_dims: Optional[int] = None,
        encoding_interp_config: Optional[dict] = None,
        encoding_lod_config: Optional[dict] = None,
        encoding_prog_config: Optional[dict] = None,
        mlp_network_config: Optional[dict] = None,
        mlp_network_input_include_xyz: bool = True,
        dtype: Optional[torch.dtype] = None,
        seed: int = 42,
        enable_jit_if_supported: bool = True,
        double_backward_skip_input_grad: bool = True,
    ) -> None:
        super().__init__()

        self.n_input_dims = n_input_dims
        self.n_pos_dims = n_pos_dims
        self.encoding_lod_config = encoding_lod_config
        self.encoding_prog_config = encoding_prog_config
        self.encoding_has_level_input = self.encoding_lod_config is not None or self.encoding_prog_config is not None
        self.mlp_network_config = mlp_network_config
        self.mlp_network_input_include_xyz = mlp_network_input_include_xyz

        self.is_fused = self.mlp_network_config is not None
        self.is_progressive = self.encoding_prog_config is not None
        self.is_lod = self.encoding_lod_config is not None

        assert isinstance(encoding_config, dict), f"[{self.__class__.__name__}] encoding_config needs to be a dict"

        self.n_levels = int(encoding_config["n_levels"])
        self.n_features_per_level = int(encoding_config["n_features_per_level"])
        self.grid_base_res = float(encoding_config["base_resolution"])
        self.grid_max_res = float(encoding_config["max_resolution"])

        # Set `per_level_scale`
        log_per_level_scale = np.log(self.grid_max_res / self.grid_base_res) / (self.n_levels - 1)
        encoding_config["per_level_scale"] = np.exp(log_per_level_scale)

        self.grid_res_list = self.grid_base_res * np.exp(log_per_level_scale * np.arange(self.n_levels))

        # Set `base_scale` for Permtuo lattice
        if encoding_config["otype"] == "Permuto":
            encoding_config["base_scale"] = float(self.grid_base_res)

        if encoding_interp_config is not None:
            interp_type = encoding_interp_config["interp_type"]
            match interp_type:
                case "linear":
                    encoding_config["interpolation"] = "Linear"
                case "smooth":
                    encoding_config["interpolation"] = "Smoothstep"
                case "smooth_straight_through" | "smooth_st":
                    # Smoothstep + straight through; Both pos. and pos_deriv. are modified.
                    encoding_config["interpolation"] = "LinearSmoothGrad"
                    encoding_config["interpolation_param"] = encoding_interp_config["lambda_"]
                case "smooth_straight_through_v2" | "smooth_st_v2":
                    # Smoothstep + straight through; Only pos_deriv. is modified.
                    encoding_config["interpolation"] = "LinearSmoothGrad2"
                    encoding_config["interpolation_param"] = encoding_interp_config["lambda_"]
                case _:
                    raise ValueError(f"[{self.__class__.__name__}] unknown interp_type {interp_type}")

        # Optionally, wrap encoding config with LoD / level input support
        if self.encoding_has_level_input:
            encoding_config = {
                "n_dims_to_encode": self.n_input_dims + 1,
                "otype": "MultiLevelEncodingLoD",
                "lod_type": (
                    "Soft"
                    if self.encoding_lod_config is not None and self.encoding_lod_config["continuous"] is True
                    else "Hard"
                ),
                "base": encoding_config,
            }
        else:
            encoding_config["n_dims_to_encode"] = self.n_input_dims

        if mlp_network_config is not None:
            assert isinstance(mlp_network_config, dict), (
                f"[{self.__class__.__name__}] mlp_network_config needs to be a dict"
            )
            # Encoding + Network
            # Optionally, concatenate original input to the encodinng output (i.e. network input)
            if self.mlp_network_input_include_xyz:
                encoding_config = {
                    "otype": "Composite",
                    "reduction": "Concatenation",  # Concatenate the identity output with permuto encoding output
                    "nested": [
                        encoding_config,
                        {
                            "n_dims_to_encode": self.n_pos_dims,
                            "otype": "Identity",
                            # From [0,1] to [-1,1]
                            "scale": 2.0,
                            "offset": -1.0,
                        },
                    ],
                }

            self.n_output_dims = unpack_optional(n_output_dims)
            self.network_n_input_dims = (
                self.n_levels * self.n_features_per_level + self.n_pos_dims * self.mlp_network_input_include_xyz
            )
            self.network_n_neurons = mlp_network_config["n_neurons"]
            self.network_n_hidden_layers = mlp_network_config["n_hidden_layers"]

            self.tcnn_module = tcnn.NetworkWithInputEncoding(
                n_input_dims=self.n_input_dims
                + 1 * self.encoding_has_level_input
                + self.n_pos_dims * self.mlp_network_input_include_xyz,
                n_output_dims=self.n_output_dims,
                encoding_config=encoding_config,
                network_config=mlp_network_config,
                seed=seed,
            )

            self.encoding_config = encoding_config
            self.mlp_network_config = mlp_network_config
        else:
            # Encoding only
            self.tcnn_module = tcnn.Encoding(
                n_input_dims=self.n_input_dims + 1 * self.encoding_has_level_input,
                encoding_config=encoding_config,
                dtype=dtype,
                seed=seed,
            )

            self.encoding_config = encoding_config
            self.n_output_dims = self.tcnn_module.n_output_dims

        # Misc setup for the created TCNN model
        self.tcnn_module.jit_fusion = enable_jit_if_supported and tcnn.supports_jit_fusion()
        self.tcnn_module.backward_backward_input_no_input_grad = double_backward_skip_input_grad

        self.tcnn_hyperparams = self.tcnn_module.native_tcnn_module.hyperparams()

        # Progressive setup
        if encoding_prog_config:
            self.n_initial_levels = encoding_prog_config["n_initial_levels"]
            assert self.n_initial_levels <= self.n_levels, (
                f"[{self.__class__.__name__}]: Number of init levels must be smaller or equal to number of all levels"
            )

            self.update_frequency = adjust_step_for_world_size(trainer_config, encoding_prog_config["update_frequency"])
            log.info(f"TcnnMultiLevelEncoding/progressive_training: update_frequency={self.update_frequency}")

            self.n_levels_per_update = encoding_prog_config["n_levels_per_update"]
            assert self.n_levels_per_update > 0 and isinstance(self.n_levels_per_update, int), (
                f"[{self.__class__.__name__}]: Update number of levels per step must be a positive integer number"
            )

        if self.encoding_prog_config:
            self.n_active_levels = self.n_initial_levels
        else:
            self.n_active_levels = self.n_levels

    def forward(self, points_01: torch.Tensor, levels: Optional[torch.Tensor | float | int] = None):
        if points_01.numel() == 0:
            return torch.empty([*points_01.shape[:-1], self.tcnn_module.n_output_dims], device=points_01.device)

        n_active_levels = self.n_active_levels

        tcnn_inputs = []
        if self.encoding_has_level_input:
            shape = [*points_01.shape[:-1], 1]
            if isinstance(levels, torch.Tensor):
                levels_01 = levels.clamp(-1, n_active_levels) / self.n_levels
                levels_01 = levels_01.expand(shape)
            elif isinstance(levels, (float, int)):
                levels_01 = torch.full(
                    shape,
                    float(np.clip(float(levels), -1, n_active_levels) / self.n_levels),
                    dtype=points_01.dtype,
                    device=points_01.device,
                )
            elif levels is None:
                levels_01 = torch.full(
                    shape, n_active_levels / self.n_levels, dtype=points_01.dtype, device=points_01.device
                )
            else:
                raise ValueError(
                    f"[{self.__class__.__name__}] Invalid argument type for levels. Should be a tensor or a number"
                )
            tcnn_inputs = [points_01, levels_01]
        else:
            assert levels is None, "`level` input is not supported"
            tcnn_inputs = [points_01]

        if self.mlp_network_config is not None and self.mlp_network_input_include_xyz is True:
            tcnn_inputs.append(points_01[..., : self.n_pos_dims])

        return self.tcnn_module(torch.cat(tcnn_inputs, dim=-1) if len(tcnn_inputs) > 1 else tcnn_inputs[0]).float()

    def get_extra_state(self) -> Any:
        # export extra states which cannot be trivially recomputed outside of NRE
        return {
            "n_input_dims": self.n_input_dims,
            "n_levels": self.n_levels if self.encoding_has_level_input else -1,
            "n_active_levels": self.n_active_levels if self.encoding_has_level_input else 0,
            "encoding_config": self.encoding_config,
        }

    def set_extra_state(self, state: Any) -> None:
        pass

    __call__ = module_call_type(forward)

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        if self.encoding_prog_config:
            step = global_step // self.update_frequency
            self.n_active_levels = min(self.n_initial_levels + step * self.n_levels_per_update, self.n_levels)

        return {}

    @contextlib.contextmanager
    def backward_input_only(self):
        """
        Due to the everlasting pytorch issue in autograd engine (https://github.com/pytorch/pytorch/issues/56500), \
        the `only_inputs` argument in `torch.autograd.grad()` does not work on custom autograd.Function's backward().

        When we call `autograd.grad(..., only_inputs=True)` that invokes the underlying modules' backward(), \
        our custom autograd Function (typically, TCNN's `_module_function_backward`) \
        has no information that we should only compute input's gradients, \
        so the param's gradients will still be computed which is at a high temporal and memory cost.

        To avoid this and still allows for param gradient computation in backward() calls, \
        we have to temporarily set a flag to the native module and restore it later.
        
        An example on using this function:
        >>> with self.backward_input_only():
        >>>     nablas = torch.autograd.grad(sdf, points, torch.ones_like(sdf), create_graph=True, only_inputs=True)[0]
        And you will still be able to backward() normally in training.
        >>> loss = ... + eikonal_loss(nablas)
        >>> loss.backward()
        """
        try:
            # Set backward_input_only to True here to skip param grad computation when only computing input gradients
            self.tcnn_module.backward_input_only = True
            yield
        finally:
            # Set backward_input_only to False here to not break the normal training.
            self.tcnn_module.backward_input_only = False

    @staticmethod
    def build_from_listed_configs(
        n_input_dims: int,
        encoding_config: dict,
        trainer_config: TrainerConfig,
        n_pos_dims: int = 3,
        n_output_dims: Optional[int] = None,
        encoding_interp_config: Optional[dict] = None,
        encoding_lod_config: Optional[dict] = None,
        encoding_prog_config: Optional[dict] = None,
        mlp_network_config: Optional[dict] = None,
        mlp_network_input_include_xyz: bool = True,
        dtype: Optional[torch.dtype] = None,
        seed: int = 42,
        enable_jit_if_supported: bool = True,
        double_backward_skip_input_grad: bool = True,
    ) -> list["TcnnMultiLevelEncoding"]:
        """Allows for listed multiple configs in `encoding_config` and `mlp_network_config`"""

        if isinstance(encoding_config["otype"], list):
            encoding_config_list = list(zip_nested_dict(encoding_config))
        else:
            encoding_config_list = [encoding_config]
        n_listed_configs = len(encoding_config_list)

        def _unfold_potentially_listed(cfg: Optional[dict] = None, cfg_name: str = "") -> list[dict] | list[None]:
            if cfg is not None and isinstance(next(iter(unpack_optional(cfg).values())), list):
                cfg_list = list(zip_nested_dict(unpack_optional(cfg)))
                assert len(cfg_list) == n_listed_configs, (
                    f"In-consistent listed configs for `encoding_config_list` (length={n_listed_configs}) "
                    f"and `{cfg_name}` (length={len(cfg_list)})"
                )
            else:
                cfg_list = [cfg] * n_listed_configs  # type: ignore
            return cfg_list

        encoding_prog_config_list = _unfold_potentially_listed(encoding_prog_config, "encoding_prog_config")
        encoding_lod_config_list = _unfold_potentially_listed(encoding_lod_config, "encoding_lod_config")
        mlp_network_config_list = _unfold_potentially_listed(mlp_network_config, "mlp_network_config")

        modules = []
        for enc_config, prog_config, lod_config, mlp_config in zip(
            encoding_config_list, encoding_prog_config_list, encoding_lod_config_list, mlp_network_config_list
        ):
            modules.append(
                TcnnMultiLevelEncoding(
                    n_input_dims=n_input_dims,
                    encoding_config=enc_config,
                    trainer_config=trainer_config,
                    n_pos_dims=n_pos_dims,
                    n_output_dims=n_output_dims,
                    encoding_lod_config=lod_config,
                    encoding_prog_config=prog_config,
                    encoding_interp_config=encoding_interp_config,
                    mlp_network_config=mlp_config,
                    mlp_network_input_include_xyz=mlp_network_input_include_xyz,
                    dtype=dtype,
                    seed=seed,
                    enable_jit_if_supported=enable_jit_if_supported,
                    double_backward_skip_input_grad=double_backward_skip_input_grad,
                )
            )
        return modules


class BaseFeatureVolume(BaseModel, ABC):
    # keep outside to allow registering object feature volumes from a separate source file.
    # add HashGrid and SkipHashGrid at the end of this file
    VARIANTS: dict[str, Type[BaseFeatureVolume]] = {}

    @staticmethod
    def register_to_factory(name: str, cls: Type[BaseFeatureVolume]) -> None:
        if name in BaseFeatureVolume.VARIANTS:
            raise KeyError(f"{name=} already in VARIANTS.")
        BaseFeatureVolume.VARIANTS[name] = cls

    @staticmethod
    def factory(name: str, config: DictConfig, precision: str | int) -> BaseFeatureVolume:
        return BaseFeatureVolume.VARIANTS[name](config, precision=precision)

    device: torch.device
    dtype: torch.dtype

    def __init__(self, config: DictConfig, precision: str | int):
        super().__init__(config)
        self.dtype = precision_to_dtype(precision)

    @contextlib.contextmanager
    def backward_input_only(self):
        """
        Due to the everlasting pytorch issue in autograd engine (https://github.com/pytorch/pytorch/issues/56500), \
        the `only_inputs` argument in `torch.autograd.grad()` does not work on custom autograd.Function's backward().

        When we call `autograd.grad(..., only_inputs=True)` that invokes the underlying modules' backward(), \
        our custom autograd Function (typically, TCNN's `_module_function_backward`) \
        has no information that we should only compute input's gradients, \
        so the param's gradients will still be computed which is at a high temporal and memory cost.

        To avoid this and still allows for param gradient computation in backward() calls, \
        we have to temporarily set a flag to the native module and restore it later.
        
        An example on using this function:
        >>> with self.backward_input_only():
        >>>     nablas = torch.autograd.grad(sdf, points, torch.ones_like(sdf), create_graph=True, only_inputs=True)[0]
        And you will still be able to backward() normally in training.
        >>> loss = ... + eikonal_loss(nablas)
        >>> loss.backward()
        """
        try:
            # Set backward_input_only to True here to skip param grad computation when only computing input gradients
            yield
        finally:
            # Set backward_input_only to False here to not break the normal training.
            pass
