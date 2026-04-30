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

import copy
import logging
import math

from typing import Callable, Iterator, NotRequired, Optional, Required, Type, TypedDict

import apex
import lietorch as lt
import numpy as np
import torch

from omegaconf import DictConfig
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import (
    ExponentialLR,
    LinearLR,
    LRScheduler,
    PolynomialLR,
    ReduceLROnPlateau,
    _warn_get_lr_called_within_step,
)
from torch.optim.optimizer import Optimizer

from nre.config.base_schema import BaseConfigSchema
from nre.config.optim import OptimizerConfig, SchedulerConfig
from nre.config.trainer import TrainerConfig
from nre.utils.trainer import (
    adjust_betas_for_batch_size,
    adjust_gamma_for_world_size,
    adjust_learning_rate_for_batch_size,
    adjust_step_for_world_size,
)


log = logging.getLogger(__name__)

# Types parameterizing optimizers / lr schedulers

LRSchedulerTypeUnion = LRScheduler | ReduceLROnPlateau


class LRSchedulerConfigType(TypedDict, total=False):
    """A typed dictionary representing an initialized LRScheduler for consumption by PL"""

    scheduler: Required[LRSchedulerTypeUnion]
    name: str | None
    interval: str
    frequency: int
    reduce_on_plateau: bool
    monitor: str | None
    strict: bool


class OptimizerLRSchedulerConfig(TypedDict):
    """A typed dictionary representing an initialized optimizer with (optional) initialized LRScheduler for consumption by PL"""

    optimizer: Optimizer
    lr_scheduler: NotRequired[LRSchedulerConfigType]


optim_fns: dict[str, Callable] = {
    "fused_adam": apex.optimizers.FusedAdam,
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
}


def get_model_parameters(model: torch.nn.Module, config: OptimizerConfig, name_prefix: str = "") -> list[dict]:
    params = []
    name_prefix = (name_prefix + ".") if name_prefix else ""

    def getattr_recursive(m: object, attr: str) -> object:
        for name in attr.split("."):
            m = getattr(m, name)
        return m

    def append_param_recursive(param_name: str, param: object, param_options: dict):
        """Recursively register parameter group if parameter options contains 'args' key - otherwise recurse down the model's hierarchy"""
        if param_args := param_options.get("args"):
            match param:
                case None:
                    # intentionally skip a parameter group if it is not initialized to a value - this
                    # will result in optimizers not being instantiated if none of it's parameters is
                    # set to an optimizable value (e.g., if all parameters should be unconditionally frozen)
                    pass
                case torch.nn.Parameter():
                    params.append({"params": [param], "name": name_prefix + param_name, **param_args})
                case torch.nn.ParameterList():
                    params.append({"params": list(param), "name": name_prefix + param_name, **param_args})
                case torch.nn.ParameterDict():
                    params.append({"params": list(param.values()), "name": name_prefix + param_name, **param_args})
                case torch.nn.Module():
                    module_parameters = filter(lambda p: p.requires_grad, param.parameters())
                    n_params = sum([np.prod(p.size(), dtype=int) for p in module_parameters])

                    if n_params > 0:
                        params.append(
                            {"params": list(param.parameters()), "name": name_prefix + param_name, **param_args}
                        )
                case lt.LieGroupParameter():
                    params.append({"params": [param], "name": name_prefix + param_name, **param_args})
        else:
            # deepen one level to sub-parameters within current model level
            for param_subname, param_suboptions in param_options.items():
                append_param_recursive(
                    param_name + "." + param_subname, getattr_recursive(param, param_subname), param_suboptions
                )

    # process top-level list of parameter groups within config
    assert config.params is not None
    for param_name, param_options in config.params.items():
        if param_options is not None:
            append_param_recursive(param_name, getattr_recursive(model, param_name), param_options)

    return params


def adjust_param_group_learning_rates(
    trainer_config: TrainerConfig, param_options: dict, config_name: str, group_name: str = ""
) -> None:
    """Recursively adjust learning rates in parameter group configurations to handle both flat and nested structures."""
    if param_args := param_options.get("args"):
        # Flat structure: has direct args with lr
        if "lr" in param_args:
            param_args["lr"] = adjust_learning_rate_for_batch_size(trainer_config, param_args["lr"])
            log.info("Optimizer/%s/%s: lr=%f", config_name, group_name, param_args["lr"])
    else:
        # Nested structure: recurse into sub-parameters
        for subkey, suboptions in param_options.items():
            if isinstance(suboptions, dict):
                sub_group_name = f"{group_name}.{subkey}" if group_name else subkey
                adjust_param_group_learning_rates(trainer_config, suboptions, config_name, sub_group_name)


def parse_scheduler_config(
    config: SchedulerConfig,
    optimizer: Optimizer,
    trainer_config: TrainerConfig,
    recursion_depth: int = 0,
) -> LRSchedulerConfigType:
    log.info("%sScheduler/%s", "  " * recursion_depth, config.name)

    # This function is used to instantiate the scheduler.
    def init_scheduler(interval: str, name: str, *args, **kwargs) -> LRSchedulerTypeUnion:
        """
        In this function, we first adjust the scheduler parameters based on the distributed world size,
        then we instantiate the scheduler.
        """

        logfunc = lambda msg: log.info("  └─%s", msg)

        if name in _StepFunSchedulers:
            kwargs["interval"] = interval
            kwargs["trainer"] = trainer_config
            return _StepFunSchedulers[name](*args, **kwargs)

        elif name in ["ExponentialLR"]:
            if "gamma" in kwargs:
                kwargs["gamma"] = adjust_gamma_for_world_size(trainer_config, kwargs["gamma"])
                logfunc(f"gamma={kwargs['gamma']}")

        elif name in ["ConstantLR", "LinearLR", "PolynomialLR"]:
            if "total_iters" in kwargs:
                kwargs["total_iters"] = adjust_step_for_world_size(trainer_config, kwargs["total_iters"])
                logfunc(f"total_iters={kwargs['total_iters']}")

        elif name in ["OneCycleLR"]:
            if "steps_per_epoch" in kwargs:
                kwargs["steps_per_epoch"] = adjust_step_for_world_size(trainer_config, kwargs["steps_per_epoch"])
                logfunc(f"steps_per_epoch={kwargs['steps_per_epoch']}")

            if "total_steps" in kwargs:
                kwargs["total_steps"] = adjust_step_for_world_size(trainer_config, kwargs["total_steps"])
                logfunc(f"total_steps={kwargs['total_steps']}")

            if "max_lr" in kwargs:
                if isinstance(kwargs["max_lr"], list):
                    kwargs["max_lr"] = [
                        adjust_learning_rate_for_batch_size(trainer_config, lr) for lr in kwargs["max_lr"]
                    ]
                else:
                    kwargs["max_lr"] = adjust_learning_rate_for_batch_size(trainer_config, kwargs["max_lr"])
                logfunc(f"max_lr={kwargs['max_lr']}")

        elif name in ["CosineAnnealingLR"]:
            if "T_max" in kwargs:
                kwargs["T_max"] = adjust_step_for_world_size(trainer_config, kwargs["T_max"])
                logfunc(f"T_max={kwargs['T_max']}")

        elif name in ["SequentialLR"]:
            if "milestones" in kwargs:
                if interval == "step":
                    kwargs["milestones"] = [
                        adjust_step_for_world_size(trainer_config, milestone) for milestone in kwargs["milestones"]
                    ]
                logfunc(f"milestones={kwargs['milestones']}")

        # Now we instantiate the scheduler
        if hasattr(lr_scheduler, name):
            return getattr(lr_scheduler, name)(*args, **kwargs)
        else:
            raise NotImplementedError(f"Scheduler {name} not implemented")

    # Construct the top-level scheduler and build the nested schedulers recursively.
    interval: str = getattr(config, "interval", "epoch")
    assert interval in ["epoch", "step"]

    if config.name == "SequentialLR":
        assert config.schedulers is not None, "SequentialLR requires nested 'schedulers'"
        return {
            "scheduler": init_scheduler(
                interval,
                config.name,
                optimizer,
                schedulers=[
                    parse_scheduler_config(conf, optimizer, trainer_config, recursion_depth + 1)["scheduler"]
                    for conf in config.schedulers
                ],
                milestones=config.milestones,
            ),
            "interval": interval,
        }
    elif config.name == "ChainedScheduler":
        assert config.schedulers is not None, "ChainedScheduler requires nested 'schedulers'"
        return {
            "scheduler": getattr(lr_scheduler, config.name)(
                [
                    parse_scheduler_config(conf, optimizer, trainer_config, recursion_depth + 1)["scheduler"]
                    for conf in config.schedulers
                ]
            ),
            "interval": interval,
        }
    else:
        return {"scheduler": init_scheduler(interval, config.name, optimizer, **config.args), "interval": interval}


def parse_optimizer(
    config: OptimizerConfig,
    model: torch.nn.Module,
    trainer_config: TrainerConfig,
    name_prefix: str = "",
) -> Optional[Optimizer]:
    """Instantiate optimizer for a given model and (optional) config-provided parameter group parametrizations

    Skip instantiation of optimizer if *no* parameter groups would be associated with the new instance
    (e.g., for optimizers that would only affect frozen model components)
    """

    config = config.model_copy(deep=True) if isinstance(config, BaseConfigSchema) else copy.deepcopy(config)

    if config.name == "fused_adam":
        config.args["lr"] = adjust_learning_rate_for_batch_size(trainer_config, config.args["lr"])
        log.info("Optimizer/%s: lr=%f", config.name, config.args["lr"])

        config.args["betas"] = adjust_betas_for_batch_size(trainer_config, config.args["betas"])
        log.info("Optimizer/%s: betas=%s", config.name, config.args["betas"])

        config_params = getattr(config, "params", None)
        if config_params is not None:
            for key, group in config_params.items():
                adjust_param_group_learning_rates(trainer_config, group, config.name, key)
    else:
        raise NotImplementedError(f"Optimizer {config.name} not supported")

    params: list[dict] | list[torch.nn.Parameter]
    if config_params is not None:
        # parse config-provided parameter group parametrizations and associated with model
        params = get_model_parameters(model, config, name_prefix)
    else:
        # config doesn't provide parameter groups parametrizations, use all parameter groups from model as is
        params = list(model.parameters())

    if not len(params):
        # don't initialize an optimizer if no parameter groups are going to be associated with it
        return None

    return optim_fns[config.name](params, **config.args)


def configure_optimizers(
    config: BaseConfigSchema | DictConfig,
    trainer_config: TrainerConfig,
    model: torch.nn.Module | torch.nn.ModuleList,
    name_prefix: str = "",
) -> list[OptimizerLRSchedulerConfig]:
    """
    Construct PL parameters for single optimizer (stored in 'system.optimizer' config) or
    multiple optimizers (stored in 'system.optimizers' config list), associated with the
    model's parameter groups (both with optional lr_schedulers if present)

    Note: config accepts BaseConfigSchema | DictConfig during the transition to typed configs.
    Some callers (e.g. calib.py, base.py, post_processing.py) still pass DictConfig via
    .to_dictconfig(). The inner functions (parse_optimizer, parse_scheduler_config) are typed
    for OptimizerConfig/SchedulerConfig but work with DictConfig at runtime thanks to duck typing.
    These callers will be migrated in follow-up MRs.
    """

    ret: list[OptimizerLRSchedulerConfig] = []

    def parse_append_single_optimizer(
        optimizer_config: OptimizerConfig,
        scheduler_config: SchedulerConfig | None,
    ) -> None:
        """
        Parses and instantiate an optimizer and associated parameter group configuration
        of a single optimizer, and appends it to the output list of optimizer dictionaries.
        """
        if (optim := parse_optimizer(optimizer_config, model, trainer_config, name_prefix)) is None:
            return

        if scheduler_config is not None:
            lr_scheduler = parse_scheduler_config(scheduler_config, optim, trainer_config)
            ret.append({"optimizer": optim, "lr_scheduler": lr_scheduler})
        else:
            ret.append({"optimizer": optim})

    if hasattr(config, "optimizers") and config.optimizers is not None:
        # multiple optimizers case
        for optimizer_config in config.optimizers:
            parse_append_single_optimizer(optimizer_config, getattr(optimizer_config, "scheduler", None))

    elif hasattr(config, "optimizer") and config.optimizer is not None:
        # single optimizer case
        parse_append_single_optimizer(config.optimizer, getattr(config, "scheduler", None))

    return ret


class StepFunLRSchedulerMixin(LRScheduler):
    """
    A mixin class for `LRScheduler` subclasses that discretize the continuous lr functions into step functions.
    Overrides the internal step mechanisms with `update_every_n_steps`.
    """

    update_every_n_steps: int

    def _initial_step(self):
        self._step_count_actual = 0
        super()._initial_step()

    def step(self, epoch: Optional[int] = None):
        if epoch is None:
            self._step_count_actual += 1
        else:
            self._step_count_actual = epoch

        if self._step_count_actual % self.update_every_n_steps == 0:
            epoch_internal = (epoch // self.update_every_n_steps) if epoch is not None else None
            super().step(epoch_internal)


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
        interval: str,
        trainer: TrainerConfig,
        eta_min: Optional[float] = None,
        min_factor: Optional[float] = None,
        last_epoch: int = -1,
        update_every_n_steps: int = 1,
    ):
        if interval == "step":
            T_max = adjust_step_for_world_size(trainer, T_max)
            update_every_n_steps = adjust_step_for_world_size(trainer, update_every_n_steps)

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


class StepFunExponentialLR(StepFunLRSchedulerMixin, ExponentialLR):
    def __init__(
        self,
        optimizer: Optimizer,
        gamma: float,
        interval: str,
        trainer: TrainerConfig,
        last_epoch: int = -1,
        update_every_n_steps: int = 1,
    ):
        if interval == "step":
            update_every_n_steps = adjust_step_for_world_size(trainer, update_every_n_steps)

        last_epoch = (last_epoch // update_every_n_steps) if last_epoch > 0 else last_epoch
        self.update_every_n_steps = update_every_n_steps
        super().__init__(optimizer, gamma, last_epoch)


class StepFunPolynomialLR(StepFunLRSchedulerMixin, PolynomialLR):
    def __init__(
        self,
        optimizer: Optimizer,
        interval: str,
        trainer: TrainerConfig,
        total_iters: int = 5,
        power: float = 1.0,
        last_epoch: int = -1,
        update_every_n_steps: int = 1,
    ):
        if interval == "step":
            update_every_n_steps = adjust_step_for_world_size(trainer, update_every_n_steps)
            total_iters = adjust_step_for_world_size(trainer, total_iters)

        last_epoch = (last_epoch // update_every_n_steps) if last_epoch > 0 else last_epoch
        self.update_every_n_steps = update_every_n_steps
        # Adding the numerator with (update_every_n_steps-1) to prevent over flooring too much in the division
        total_iters = max((total_iters + update_every_n_steps - 1) // update_every_n_steps, 1)
        super().__init__(optimizer, total_iters, power, last_epoch)


class StepFunLinearLR(StepFunLRSchedulerMixin, LinearLR):
    def __init__(
        self,
        optimizer: Optimizer,
        interval: str,
        trainer: TrainerConfig,
        start_factor: float = 1.0 / 3,
        end_factor: float = 1.0,
        total_iters: int = 5,
        last_epoch: int = -1,
        update_every_n_steps: int = 1,
    ):
        if interval == "step":
            update_every_n_steps = adjust_step_for_world_size(trainer, update_every_n_steps)
            total_iters = adjust_step_for_world_size(trainer, total_iters)

        last_epoch = (last_epoch // update_every_n_steps) if last_epoch > 0 else last_epoch
        self.update_every_n_steps = update_every_n_steps
        # Adding the numerator with (update_every_n_steps-1) to prevent over flooring too much in the division
        total_iters = max((total_iters + update_every_n_steps - 1) // update_every_n_steps, 1)
        super().__init__(optimizer, start_factor, end_factor, total_iters, last_epoch)


_StepFunSchedulers: dict[str, Type[LRScheduler]] = {
    cls.__name__: cls for cls in (StepFunCosineAnnealingLR, StepFunPolynomialLR, StepFunExponentialLR, StepFunLinearLR)
}
