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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Optional, Type, Union

import torch

from omegaconf import DictConfig
from torch import nn

from nre.config.model import BaseStrategyConfig
from nre.models.base import BaseModel
from nre.models.gaussians.gaussians_model import BaseGaussianModel
from nre.models.nn_extensions import TypedModuleDict
from nre.utils.batch import DataAndRenderingBatch
from nre.utils.trainer import TrainerConfig


if TYPE_CHECKING:
    from nre.models.gaussians.gaussians_model import BaseGaussianModel  # pycena: skip


class BaseGaussianStrategy(nn.Module, ABC):
    """
    Base class for Gaussian Strategies. Strategies are responsible for the densification and pruning of the Gaussians.

    Currently there are two main methods that are called to interact with a Strategy:
        - `update_step_train_batch_end`: called at the end of the train batch, useful for periodic updates to the gaussians (densify, prune, density resets etc.)

    """

    GAUSSIAN_STRATEGY_VARIANTS: dict[str, Type[BaseGaussianStrategy]] = {}

    def __init__(
        self,
        config: BaseStrategyConfig,
        trainer_config: TrainerConfig,
        init_from_datasource: bool,
        gaussians_nodes: TypedModuleDict[BaseGaussianModel],
    ) -> None:
        super().__init__()
        self.config = config
        self.init_from_datasource = init_from_datasource
        self.last_reset_epoch: Optional[int] = None

    @torch.no_grad()
    def maybe_initialize_buffers(self, gaussians_nodes: TypedModuleDict[BaseGaussianModel]) -> None: ...

    @abstractmethod
    def update_step_train_batch_end(
        self,
        epoch: int,
        global_step: int,
        batch: DataAndRenderingBatch,
        system,
        gaussians_nodes: TypedModuleDict[BaseGaussianModel],
        **kwargs,
    ) -> None: ...

    @staticmethod
    def register_to_gaussian_strategy_factory(name: str, cls: Type[BaseGaussianStrategy]) -> None:
        if name in BaseGaussianStrategy.GAUSSIAN_STRATEGY_VARIANTS:
            raise KeyError(f"{name=} already in GAUSSIAN_STRATEGY_VARIANTS.")
        BaseGaussianStrategy.GAUSSIAN_STRATEGY_VARIANTS[name] = cls

    @staticmethod
    def factory(
        name: str,
        config: BaseStrategyConfig,
        trainer_config: TrainerConfig,
        init_from_datasource: bool,
        gaussians_nodes: TypedModuleDict[BaseGaussianModel],
    ) -> BaseGaussianStrategy:
        return BaseGaussianStrategy.GAUSSIAN_STRATEGY_VARIANTS[name](
            config, trainer_config, init_from_datasource, gaussians_nodes
        )

    @torch.no_grad()
    def _update_param_with_optimizer(
        self,
        gaussian_model: BaseGaussianModel,
        update_param_fn: Callable[[str, torch.Tensor], torch.Tensor],
        update_optimizer_fn: Callable[[str, str, torch.Tensor], torch.Tensor],
        names: Union[list[str], None] = None,
    ) -> None:
        """Update the parameters and the state in the optimizers using the provided lambda functions.

        Args:
            update_param_fn: A function that takes the name of the parameter and the parameter itself,
                and returns the new parameter.
            optimizer_fn: A function that takes the key of the optimizer state and the state value,
                and returns the new state value.
            names: A list of key names to update. If None, update all. Default: None.
        """
        for optim_sched in gaussian_model.optimizers:
            optimizer = optim_sched["optimizer"]
            for i, param_group in enumerate(optimizer.param_groups):
                if "name" in param_group:
                    name = param_group["name"].split(".")[-1]
                    if (names is None) or (name in names):
                        p = param_group["params"][0]
                        p_state = optimizer.state[p]
                        del optimizer.state[p]
                        for key in p_state.keys():
                            v = p_state[key]
                            p_state[key] = update_optimizer_fn(name, key, v)
                        p_new = update_param_fn(name, p)
                        # We need to preserve the "sharded" flag
                        if BaseModel.is_sharded(p):
                            BaseModel.mark_as_sharded(p_new)
                        # Set the new parameter in the optimizer
                        optimizer.param_groups[i]["params"] = [p_new]
                        optimizer.state[p_new] = p_state
                        setattr(gaussian_model, name, p_new)

        torch.cuda.empty_cache()
        # Also update additional non-optimizable buffers
        for name, b in gaussian_model.get_additional_buffers().items():
            if (names is None) or (name in names):
                b_new = update_param_fn(name, b)
                # We need to preserve the "sharded" flag
                if BaseModel.is_sharded(b):
                    BaseModel.mark_as_sharded(b_new)
                # Set the new parameter in the model
                setattr(gaussian_model, name, b_new)

    def _check_step_condition(self, step: int, start: int, end: int, freq: int) -> bool:
        """Checks if an operation should occur for the given step."""
        if step > start and step < end and step % freq == 0:
            return True
        return False
