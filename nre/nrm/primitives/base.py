# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Literal, Self, TypeVar

import torch

from omegaconf import DictConfig, OmegaConf

from nre.config.version import get_version
from nre.models.gaussians.renderers import BaseGaussianRenderer
from nre.models.nrenderable import NRenderableModel
from nre.nrm.config.models import PrimitiveExportPreprocessConfig
from nre.utils.batch import DataAndRenderingBatch
from nre.utils.misc import map_optional, strip_none_from_config
from nre.utils.types import (
    Checkpoint,
    RigTrajectories,
)


class BaseNRMPrimitive(NRenderableModel):
    """
    Base class for all renderable primitives reconstructed by an NRM.
    """

    @abstractmethod
    def state_dict_and_config(self) -> tuple[dict[str, Any], DictConfig]:
        """Used for serialization to be used in nrend."""

    @abstractmethod
    def get_checkpoint(self) -> Checkpoint:
        """Used for serialization to be used in render."""

    @torch.no_grad()
    def serialize_to_json_dict(self, with_state_dict: bool = True) -> dict[str, Any]:
        """Used only during initialization time, and/or test time with nrend."""

        state_dict, config = self.state_dict_and_config()
        config_dict = strip_none_from_config(OmegaConf.to_container(config))
        json_dict = {
            "nre_data": {
                "version": map_optional(get_version(), lambda v: v.semantic_string()),
                "model": "nre",
                # Empty to be filled by the derived class
                "config": config_dict,
                "state_dict": {f".{key}": value for key, value in state_dict.items()} if with_state_dict else {},
            }
        }

        if with_state_dict:
            assert isinstance(json_dict["nre_data"]["state_dict"], dict)
            # add shape entries for every tensor
            json_dict["nre_data"]["state_dict"].update(
                {
                    key + ".shape": list(value.size())
                    for key, value in json_dict["nre_data"]["state_dict"].items()
                    if isinstance(value, torch.Tensor) and isinstance(key, str)
                }
            )

            # convert tensor to bytes
            def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
                # default conversion to half for single precision tensor
                tensor = tensor.to(dtype=torch.float16) if tensor.dtype == torch.float32 else tensor
                return tensor.flatten().cpu().numpy().tobytes()

            json_dict["nre_data"]["state_dict"] = {
                key: tensor_to_bytes(value) if isinstance(value, torch.Tensor) else value
                for key, value in json_dict["nre_data"]["state_dict"].items()
            }

        return json_dict

    @abstractmethod
    def device(self) -> torch.device: ...

    @abstractmethod
    def rigid_transform(self, T_new: torch.Tensor) -> Self: ...

    @abstractmethod
    def preprocess_for_export(
        self,
        context_batch: DataAndRenderingBatch,
        config: PrimitiveExportPreprocessConfig,
        context_rig: RigTrajectories | None = None,
    ) -> Self:
        """
        Filter and preprocess the primitive for export (e.g. density/sky/road masking).
        Called per chunk after forward; when merging is enabled, merge will then apply
        rigid_transform to align chunks. Implementations must not apply rigid_transform.
        context_rig is optional; Celsius may require it when project_to_z_offset is True.
        """

    @abstractmethod
    def __len__(self) -> int: ...


NRMPrimitiveType = TypeVar("NRMPrimitiveType", bound=BaseNRMPrimitive)


class BaseGaussiansNRMPrimitive(BaseNRMPrimitive):
    """
    Predict-only carrier for Gaussian primitives. The renderer is unused but the
    field is retained so downstream type annotations and merge helpers can keep
    propagating it as None.
    """

    gaussians_renderer: BaseGaussianRenderer | None

    def __init__(
        self,
        gaussians_renderer: BaseGaussianRenderer | None,
        checkpointing: Literal["render", "all", "none"] = "render",
        shared_gaussian_parameters: bool = False,
    ):
        self.gaussians_renderer = gaussians_renderer
        self.checkpointing = checkpointing
        self.shared_gaussian_parameters = shared_gaussian_parameters
