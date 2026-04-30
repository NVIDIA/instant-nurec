# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from omegaconf import DictConfig, OmegaConf
from pydantic import (
    BaseModel as PydanticBaseModel,
)
from pydantic import (
    Field,  # reexport
)


__all__ = ["BaseConfigSchema", "Field", "config_to_primitive"]


def config_to_primitive(config, resolve=True):
    """Convert OmegaConf/DictConfig or Pydantic model to plain Python dict/list.

    Safe to import from nre.models etc. (no nre.config.nre).
    """
    if config is None:
        return None
    if isinstance(config, BaseConfigSchema):
        return config.model_dump(exclude_none=True)
    if isinstance(config, PydanticBaseModel):
        return config.model_dump()
    if isinstance(config, dict):
        return dict(config)
    if isinstance(config, list):
        return list(config)
    return OmegaConf.to_container(config, resolve=resolve)


class BaseConfigSchema(PydanticBaseModel):
    """
    The base class for NRE config structs, acts very similar to python's dataclasses.
    We extend it with an extra method which converts a typed config into an untyped one
    for use with modules which have not yet been reworked.
    """

    def to_dictconfig(self) -> DictConfig:
        # TODO: remove once the entire repo has transitioned to strongly typed config.
        return DictConfig(self.model_dump())

    def __hash__(self) -> int:
        return hash((type(self), self.__repr__()))
