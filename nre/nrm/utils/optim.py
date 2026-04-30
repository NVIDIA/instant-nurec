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

import logging
import math

from typing import Iterator, Optional, Type

import torch

from omegaconf import DictConfig
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import (
    LinearLR,
    LRScheduler,
    _warn_get_lr_called_within_step,
)
from torch.optim.optimizer import Optimizer

from nre.config.base_schema import BaseConfigSchema
from nre.config.optim import OptimizerConfig, SchedulerConfig
from nre.utils.optim import (
    LRSchedulerConfigType,
    LRSchedulerTypeUnion,
    OptimizerLRSchedulerConfig,
    StepFunLRSchedulerMixin,
    get_model_parameters,
    optim_fns,
)


log = logging.getLogger(__name__)


def mark_parameter_no_weight_decay(param: torch.nn.Parameter) -> None:
    """
    Mark a parameter as not requiring weight decay by setting an attribute within the parameter.
    """
    setattr(param, "_nrm_no_weight_decay", True)


def check_parameter_no_weight_decay(param: torch.nn.Parameter) -> bool:
    """
    Check if a parameter is marked as not requiring weight decay.
    """
    flag = getattr(param, "_nrm_no_weight_decay", False)
    assert isinstance(flag, bool), "Attribute '_nrm_no_weight_decay' must be a boolean"
    return flag


def _separate_no_weight_decay_parameters(optimizer: Optimizer, model: torch.nn.Module) -> None:
    """
    Separate specific parameters from other parameters and set weight_decay=0 for them.
    This modifies the optimizer's param_groups in-place.

    Args:
        optimizer: The optimizer to modify
        model: The model to search for parameter names
    """
    # Create a mapping from parameter object to parameter name for efficient lookup
    param_to_name = {param: name for name, param in model.named_parameters()}

    new_param_groups = []

    for group in optimizer.param_groups:
        # Separate no weight decay and other parameters
        no_weight_decay_params = []
        other_params = []

        for param in group["params"]:
            param_name = param_to_name.get(param)
            if param_name and check_parameter_no_weight_decay(param):
                no_weight_decay_params.append(param)
            else:
                other_params.append(param)

        # Create separate groups for bias and non-bias parameters
        if other_params:
            other_group = group.copy()
            other_group["params"] = other_params
            new_param_groups.append(other_group)

        if no_weight_decay_params:
            no_weight_decay_group = group.copy()
            no_weight_decay_group["params"] = no_weight_decay_params
            no_weight_decay_group["weight_decay"] = 0.0  # Remove weight decay for bias terms
            # Update the name to indicate this is a no weight decay group
            if "name" in no_weight_decay_group:
                no_weight_decay_group["name"] = no_weight_decay_group["name"] + "_no_weight_decay"
            new_param_groups.append(no_weight_decay_group)

    # Replace the optimizer's param_groups
    optimizer.param_groups = new_param_groups


def parse_scheduler_config(config: SchedulerConfig, optimizer: Optimizer) -> LRSchedulerConfigType:
    def init_scheduler(name: str, *args, **kwargs) -> LRSchedulerTypeUnion:
        if name in _StepFunSchedulers:
            return _StepFunSchedulers[name](*args, **kwargs)
        elif name in _ProgressBasedSchedulers:
            return _ProgressBasedSchedulers[name](*args, **kwargs)
        elif hasattr(lr_scheduler, name):
            return getattr(lr_scheduler, name)(*args, **kwargs)
        else:
            raise NotImplementedError(f"Scheduler {name} not implemented")

    interval: str = getattr(config, "interval", "epoch")
    assert interval in ["epoch", "step"]

    if config.name == "SequentialLR":
        # Process nested scheduler configs
        assert config.schedulers is not None, "SequentialLR requires nested 'schedulers'"
        nested_schedulers = []
        for nested_config in config.schedulers:
            nested_schedulers.append(parse_scheduler_config(nested_config, optimizer)["scheduler"])

        return {
            "scheduler": init_scheduler(
                config.name,
                optimizer,
                nested_schedulers,
                milestones=config.milestones,
            ),
            "interval": interval,
        }
    else:
        return {"scheduler": init_scheduler(config.name, optimizer, **config.args), "interval": interval}


def parse_optimizer(
    config: OptimizerConfig,
    model: torch.nn.Module,
    name_prefix: str = "",
) -> Optional[Optimizer]:
    """Instantiate optimizer for a given model and (optional) config-provided parameter group parametrizations

    Skip instantiation of optimizer if *no* parameter groups would be associated with the new instance
    (e.g., for optimizers that would only affect frozen model components)

    Args:
        config: Optimizer configuration
        model: The model to optimize
        name_prefix: Prefix for parameter names
    """

    params: list[dict] | Iterator[torch.nn.Parameter]
    if getattr(config, "params", None) is not None:
        # parse config-provided parameter group parametrizations and associated with model
        params = get_model_parameters(model, config, name_prefix)

        if not len(params):
            # don't initialize an optimizer if no parameter groups are going to be associated with it
            return None
    else:
        # config doesn't provide parameter groups parametrizations, use all parameter groups from model as is
        params = model.parameters()

    optimizer = optim_fns[config.name](params, **config.args)

    # Separate bias parameters from weight decay
    _separate_no_weight_decay_parameters(optimizer, model)

    return optimizer


def configure_optimizers(
    config: BaseConfigSchema | DictConfig,
    model: torch.nn.Module | torch.nn.ModuleList,
    name_prefix: str = "",
) -> list[OptimizerLRSchedulerConfig]:
    """
    Construct PL parameters for a single optimizer (stored in 'system.optimizer' config),
    associated with the model's parameter groups (with optional lr_scheduler if present).

    Note: config accepts BaseConfigSchema | DictConfig during the transition to typed configs.
    The NRM caller (base.py) still passes DictConfig. Will be migrated in a follow-up MR.
    """

    ret: list[OptimizerLRSchedulerConfig] = []

    if hasattr(config, "optimizer") and config.optimizer is not None:
        optimizer_config = config.optimizer
        scheduler_config = getattr(config, "scheduler", None)

        if (optim := parse_optimizer(optimizer_config, model, name_prefix)) is not None:
            if scheduler_config is not None:
                lr_sched = parse_scheduler_config(scheduler_config, optim)
                ret.append({"optimizer": optim, "lr_scheduler": lr_sched})
            else:
                ret.append({"optimizer": optim})

    return ret


class ProgressBasedLRScheduler(LRScheduler):
    """
    A base class for learning rate schedulers that are based on training progress (e.g., percentage of the current epochs/steps).
    This class is intended to be subclassed for specific implementations.
    """

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        super().__init__(optimizer, last_epoch)
        self._current_epoch: int = 0
        self._total_epochs: int = 0
        self._current_local_step: int = 0
        self._total_local_steps: int = 0

    def set_progress(self, epoch: int, total_epochs: int, local_step: int, total_local_steps: int) -> None:
        """
        Set the current progress of the scheduler -- to be called in system's manual training_step()
        """
        self._current_epoch = epoch
        self._total_epochs = total_epochs
        self._current_local_step = local_step
        self._total_local_steps = total_local_steps


class CosineWithWarmupPBScheduler(ProgressBasedLRScheduler):
    """
    A learning rate scheduler that combines a warmup phase with a cosine annealing schedule.
    The warmup phase linearly increases the learning rate from 0 to the initial learning rate,
    followed by a cosine annealing phase.

    Args:
        warmup_factor (float): The factor by which the learning rate is multiplied at the beginning of the warmup phase.
        warmup_steps (int): The number of steps for the warmup phase.
        cosine_factor (float): The factor by which the learning rate is multiplied at the end of the cosine annealing phase.
        cosine_factor_progress (float): At which training progress (in terms of percentage) should the LR reach the `cosine_factor * initial_lr` value.
    this is by default 1.0, meaning that we will reach the `cosine_factor * initial_lr` at the end of training (i.e. 100%).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_factor: float,
        warmup_steps: int,
        cosine_factor: float,
        cosine_factor_progress: float = 1.0,
        last_epoch: int = -1,
    ):
        super().__init__(optimizer, last_epoch)
        self.warmup_factor = warmup_factor
        self.warmup_steps = warmup_steps
        self.cosine_factor = cosine_factor
        self.cosine_factor_progress = cosine_factor_progress

    def get_lr(self) -> list[float]:
        _warn_get_lr_called_within_step(self)

        # We cannot know what is the progress at the beginning of training, so we assume a small learning rate (i.e. ignore this single step).
        if self._step_count <= 1:
            return [group["initial_lr"] * 1e-6 for group in self.optimizer.param_groups]

        assert self._total_local_steps > 0 and self._total_epochs > 0, "Must call set_progress before get_lr()"

        current_global_step = self._current_epoch * self._total_local_steps + self._current_local_step
        if current_global_step < self.warmup_steps:
            # Linear warmup phase
            warmup_progress = current_global_step / (self.warmup_steps - 1)
            return [
                group["initial_lr"] * (self.warmup_factor + (1 - self.warmup_factor) * warmup_progress)
                for group in self.optimizer.param_groups
            ]

        else:
            # Cosine annealing phase
            cosine_progress = (current_global_step - self.warmup_steps) / (
                self._total_epochs * self._total_local_steps - self.warmup_steps
            )
            # Make sure we never bounce back in the cosine function
            cosine_progress = min(cosine_progress / self.cosine_factor_progress, 1.0)
            # cosine_factor = self.cosine_factor * (1 - cosine_progress) + self.cosine_factor_progress * cosine_progress
            return [
                group["initial_lr"]
                * (self.cosine_factor + (1 - self.cosine_factor) * (1 + math.cos(math.pi * cosine_progress)) / 2)
                for group in self.optimizer.param_groups
            ]


_ProgressBasedSchedulers: dict[str, Type[ProgressBasedLRScheduler]] = {
    cls.__name__: cls for cls in (CosineWithWarmupPBScheduler,)
}


class StepFunCosineAnnealingLR(StepFunLRSchedulerMixin):
    """
    Modified version of `torch.optim.lr_scheduler.CosineAnnealingLR`.
    Support either of:
    - Setting an absolute final lr for all the optimizer groups using `eta_min`
    - Setting a relative final lr factor based on the different initial lrs of different optimizer groups using `min_factor`
    """

    T_max: float
    eta_mins: list[float]

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: Optional[float] = None,
        min_factor: Optional[float] = None,
        last_epoch: int = -1,
        update_every_n_steps: int = 1,
    ):
        last_epoch = (last_epoch // update_every_n_steps) if last_epoch > 0 else last_epoch
        self.update_every_n_steps = update_every_n_steps
        LRScheduler.__init__(self, optimizer, last_epoch)

        # Adding the numerator with (update_every_n_steps-1) to prevent over flooring too much in the division
        self.T_max = max((T_max + update_every_n_steps - 1) // update_every_n_steps, 1)
        if eta_min is None:
            assert min_factor is not None, "Please specify `min_factor` if `eta_min` is not provided"
            self.eta_mins = [(float(group["initial_lr"]) * min_factor) for group in optimizer.param_groups]
        else:
            assert min_factor is None, "Please do not specify `min_factor` since `eta_min` is provided"
            self.eta_mins = [eta_min for _ in optimizer.param_groups]

    def get_lr(self):
        _warn_get_lr_called_within_step(self)

        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        elif self._step_count == 1 and self.last_epoch > 0:
            return [
                eta_min + (base_lr - eta_min) * (1 + math.cos((self.last_epoch) * math.pi / self.T_max)) / 2
                for base_lr, eta_min, group in zip(self.base_lrs, self.eta_mins, self.optimizer.param_groups)
            ]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group["lr"] + (base_lr - eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, eta_min, group in zip(self.base_lrs, self.eta_mins, self.optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max))
            * (group["lr"] - eta_min)
            + eta_min
            for eta_min, group in zip(self.eta_mins, self.optimizer.param_groups)
        ]

    def _get_closed_form_lr(self):
        return [
            eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base_lr, eta_min in zip(self.base_lrs, self.eta_mins)
        ]


class StepFunLinearLR(StepFunLRSchedulerMixin, LinearLR):
    def __init__(
        self,
        optimizer: Optimizer,
        start_factor: float = 1.0 / 3,
        end_factor: float = 1.0,
        total_iters: int = 5,
        last_epoch: int = -1,
        update_every_n_steps: int = 1,
    ):
        last_epoch = (last_epoch // update_every_n_steps) if last_epoch > 0 else last_epoch
        self.update_every_n_steps = update_every_n_steps
        # Adding the numerator with (update_every_n_steps-1) to prevent over flooring too much in the division
        total_iters = max((total_iters + update_every_n_steps - 1) // update_every_n_steps, 1)
        super().__init__(optimizer, start_factor, end_factor, total_iters, last_epoch)


_StepFunSchedulers: dict[str, Type[LRScheduler]] = {
    cls.__name__: cls for cls in (StepFunCosineAnnealingLR, StepFunLinearLR)
}
