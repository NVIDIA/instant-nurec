# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Layer 3: Orchestration - Centralized aggregator for all loss functions."""

import logging

from typing import Any

import torch

# Import loss modules to trigger their registration decorators.
# This ensures all losses are available when LossAggregator is used.
import libs.losses.models.loss_fns  # noqa: F401
import libs.losses.models.primitive_losses  # noqa: F401
import libs.losses.models.render_losses  # noqa: F401

from libs.losses.models.base_losses import BaseLoss, BasePrimitiveLoss, BaseRenderLoss, SlangBaseLoss
from libs.losses.models.losses_module import ModuleLosses
from libs.losses.models.registry import make_loss
from libs.losses.orchestration.config import LossAggregatorReturn, LossConfig, LossReturn
from nre.datasets.base import BaseDataset
from nre.models.base import BaseModel
from nre.nrm.models.base import BaseNRMSupervisionPack
from nre.nrm.primitives.base import BaseNRMPrimitive
from nre.utils.batch import DataAndRenderingBatch
from nre.utils.profiling import ScopedTimer
from nre.utils.trainer import TrainerConfig
from nre.utils.types import GaussiansCompositeReturn


log = logging.getLogger(__name__)


class LossAggregator:
    """Centralized aggregator for all the loss functions."""

    losses: list[BaseLoss]
    in_cuda: ModuleLosses  # Layer 3 as explained in the docs/architecture/modules-losses.md file.

    def __init__(self, config: LossConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        # Optional override to disable all fused CUDA losses used for testing
        force_disable_cuda: bool = kwargs.pop("force_disable_cuda", False)
        self.losses = []
        self.in_cuda = ModuleLosses()
        for loss_type, loss_config in config.items():
            loss_name = str(loss_type)
            full_name = f"{loss_name}_{loss_config.fn}_{loss_config.reduce.name}"
            if not force_disable_cuda and full_name in self.in_cuda.available:
                self.in_cuda.losses.append(SlangBaseLoss(loss_name, loss_config, trainer_config))
            else:
                self.losses.append(make_loss(loss_name, loss_config, trainer_config))

    def __call__(
        self,
        *,  # Enforce keyword arguments
        step: int,
        model: BaseModel,
        results: GaussiansCompositeReturn | None = None,
        target: DataAndRenderingBatch | None = None,
        primitive: BaseNRMPrimitive | None = None,
        supervision_pack: BaseNRMSupervisionPack | None = None,
        context: DataAndRenderingBatch | None = None,
    ) -> LossAggregatorReturn:
        ret: dict[str, LossReturn] = {}

        if self.in_cuda.losses:
            with ScopedTimer("LossAggregator/ModuleLosses"):
                ret = self.in_cuda(step, model, results, target)

        for loss in self.losses:
            with ScopedTimer(f"LossAggregator/{loss.name}"):
                if loss.should_run_fn(step):
                    if isinstance(loss, BaseRenderLoss):
                        assert results is not None and target is not None, (
                            f"Loss {loss.name} requires results and target!"
                        )
                        loss_ret = loss(results=results, target=target, model=model)

                    elif isinstance(loss, BasePrimitiveLoss):
                        assert primitive is not None and context is not None and supervision_pack is not None, (
                            f"Loss {loss.name} is a BasePrimitiveLoss and requires primitive, context, and supervision_pack!"
                        )
                        primitive_loss_kwargs: dict[str, Any] = dict(
                            primitive=primitive,
                            supervision_pack=supervision_pack,
                            context=context,
                            model=model,
                        )
                        if loss.needs_target:
                            primitive_loss_kwargs["target"] = target
                        loss_ret = loss(**primitive_loss_kwargs)

                    else:
                        raise ValueError(f"Unknown loss type {type(loss)}")

                    if loss_ret is not None:
                        ret[loss.name] = loss_ret

        with ScopedTimer("LossAggregator/reduce"):
            return LossAggregatorReturn(ret)

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        additional_params: dict[str, torch.Tensor] = {}
        for loss in self.losses + self.in_cuda.losses:
            additional_params |= loss.update_step_train_batch_start(epoch, global_step, system, **kwargs)
        return additional_params

    def initialize(self, train_dataset: BaseDataset) -> None:
        for loss in self.losses + self.in_cuda.losses:
            loss.initialize(train_dataset)


__all__ = ["LossAggregator"]
