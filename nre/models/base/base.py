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

import inspect

from typing import Any, Iterable, Iterator

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from omegaconf import DictConfig, OmegaConf

from nre import __version__
from nre.config import BaseConfigSchema
from nre.utils.batch import DataAndRenderingBatch
from nre.utils.misc import map_optional, strip_none_from_config


class BaseModel(nn.Module):
    post_processings: Iterable[nn.Module]

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config

        # In PyTorch, any registered buffer in an nn.Module will be automatically moved to the correct device when `to(device)` is
        # called on the module. Thus, we can create a transient marker to track the device of the model. Because it's a registered
        # buffer, it is moved alongside the rest of the model. Thus, we can reliably query self._device_indicator.device to determine
        # which device the model currently resides on.
        self._device_indicator = nn.Buffer(torch.tensor(0), persistent=False)

    @property
    def device(self) -> torch.device:
        return self._device_indicator.device

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        """
        Hook function for system, invoked at system.on_train_batch_start(), i.e. the very start of each train step.
        Useful for e.g. update_module_step()
        """
        return {}

    def update_step_train_batch_end(
        self, epoch: int, global_step: int, batch: DataAndRenderingBatch, system, **kwargs
    ) -> None:
        """
        Hook function for system, invoked at system.on_train_batch_end(), i.e. the very end of each train step.
        """
        pass

    def on_train_from_scratch_start(self, system, **kwargs) -> None:
        """
        Hook function for system, invoked at system.on_train_start() only when system.resume is False.
        Useful for e.g. setting scalers, shape initialization process for sdf models, etc.
        """
        pass

    def train(self, mode: bool = True) -> BaseModel:
        return super().train(mode=mode)

    def eval(self) -> BaseModel:
        return super().eval()

    def serialize_to_legacy_nrend_dict(
        self, scale: float = 1.0, offset: npt.NDArray[np.float32] = np.array([0.0, 0.0, 0.0])
    ) -> dict[str, Any]:
        raise NotImplementedError

    @torch.no_grad()
    def serialize_to_json_dict(self, with_state_dict: bool = True) -> dict[str, Any]:
        def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
            # default conversion to half for single precision tensor
            tensor = tensor.to(dtype=torch.float16) if tensor.dtype == torch.float32 else tensor
            return tensor.flatten().cpu().numpy().tobytes()

        def tensor_shape_to_list(tensor: torch.Tensor) -> list[int]:
            return list(tensor.size())

        # Convert config to plain dict for JSON serialization
        if isinstance(self.config, BaseConfigSchema):
            # Typed config (Pydantic), need deep conversion for nested DictConfigs in Any fields
            omegaconf_dictconfig = self.config.to_dictconfig()
            config_dict = OmegaConf.to_container(omegaconf_dictconfig, resolve=False)
        else:
            # Untyped config (DictConfig)
            config_dict = OmegaConf.to_container(self.config)

        config_dict = strip_none_from_config(config_dict)

        # refers to for versioning and format details
        json_dict = {
            "nre_data": {
                "version": map_optional(__version__, lambda v: v.semantic_string()),
                "model": "nre",  # base nre model
                "config": config_dict,
                "state_dict": {f".{key}": value for key, value in self.state_dict().items()} if with_state_dict else {},
            }
        }

        assert isinstance(json_dict["nre_data"]["state_dict"], dict)

        if with_state_dict:
            # add shape entries for every tensor
            json_dict["nre_data"]["state_dict"].update(
                {
                    key + ".shape": tensor_shape_to_list(value)
                    for key, value in json_dict["nre_data"]["state_dict"].items()
                    if isinstance(value, torch.Tensor) and isinstance(key, str)
                }
            )
            # convert tensor to bytes
            json_dict["nre_data"]["state_dict"] = {
                key: tensor_to_bytes(value) if isinstance(value, torch.Tensor) else value
                for key, value in json_dict["nre_data"]["state_dict"].items()
            }

        # remove unserializable object
        # TODO : this is not filtering pybind object.
        def serializable_value(value) -> bool:
            return not inspect.isclass(value)

        if with_state_dict:
            json_dict["nre_data"]["state_dict"] = {
                key: value for key, value in json_dict["nre_data"]["state_dict"].items() if serializable_value(value)
            }

        return json_dict

    def get_base_model_children(self) -> Iterator[BaseModel]:
        """Yields all children that are instances of BaseModel"""
        for child in self.children():
            if isinstance(child, BaseModel):
                yield child

    def configure_sharded_params_and_buffers(self) -> list[str]:
        """Set the "sharded" flag for the parameters and buffers that are distributed across ranks, and then return the names of the parameters and buffers that are sharded."""
        return []

    @staticmethod
    def mark_as_sharded(param: torch.Tensor) -> None:
        setattr(param, "_sharded", True)

    @staticmethod
    def is_sharded(param: torch.Tensor) -> bool:
        return hasattr(param, "_sharded") and param._sharded
