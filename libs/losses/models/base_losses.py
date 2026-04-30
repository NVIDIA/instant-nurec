# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Layer 2: Base classes for loss modules."""

import logging

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any, Callable, Optional, cast

import omegaconf
import torch
import torch.nn as nn

from libs.losses.models.lambda_schedulers.lambda_schedulers import BaseLambdaScheduler
from libs.losses.models.lambda_schedulers.registry import make as make_lambda_scheduler
from libs.losses.models.reduce_functions.registry import make as make_reduce_fn
from libs.losses.models.registry import make_loss_fn
from libs.losses.models.utils import get_mask_semantic
from libs.losses.orchestration.config import LossItemConfig, LossReturn, RayReduceFn
from nre.datasets.base import BaseDataset
from nre.models.base import BaseModel
from nre.nrm.models.base import BaseNRMSupervisionPack
from nre.nrm.primitives.base import BaseNRMPrimitive
from nre.utils.batch import CameraFrameLabels, DataAndRenderingBatch, LidarFrameLabels
from nre.utils.misc import unpack_optional
from nre.utils.trainer import TrainerConfig, adjust_step_for_world_size
from nre.utils.types import GaussiansCompositeReturn


log = logging.getLogger(__name__)


class BaseLoss(nn.Module, ABC):
    name: str
    lambda_: float
    loss_fn: Callable[..., torch.Tensor]
    reduce_fn: RayReduceFn

    start_step: int

    lambda_scheduler: Optional[BaseLambdaScheduler]

    def __init__(self, loss_name: str, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__()

        self.name = f"{loss_name}_{config.fn}_{config.reduce.name}"
        self.lambda_ = config.lambda_
        self.loss_fn = make_loss_fn(config.fn)
        self.reduce_fn = make_reduce_fn(config.reduce.name, config.reduce)

        # If not specified start applying the loss from step 0
        self.start_step = 0
        if config.start_step is not None:
            self.start_step = adjust_step_for_world_size(trainer_config, config.start_step)

        self.visibility_filter = bool(config.visibility_filter)
        self.occlusion_aware = bool(config.occlusion_aware)

        log.info(f"BaseLoss/{self.name}: start_step={self.start_step}")

        # If not specified keep lambda constant
        self.lambda_scheduler = None
        if config.lambda_scheduler is not None:
            self.lambda_scheduler = make_lambda_scheduler(
                config.lambda_scheduler.name, config.lambda_scheduler, trainer_config, config.lambda_
            )

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        # Currently we only support a linear fadeout LambdaScheduler
        if self.lambda_scheduler is not None:
            self.lambda_ = self.lambda_scheduler(epoch=epoch, global_step=global_step)

        return {}

    def initialize(self, train_dataset: BaseDataset) -> None:
        pass

    def should_run_fn(self, step: int) -> bool:
        return self.lambda_ > 0.0 and step >= self.start_step

    def apply_loss_fn(
        self,
        *args: Any,
        reduce_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> LossReturn:
        """
        Apply loss-specific function to arguments and create return
        """
        return LossReturn(
            name=self.name,
            lambda_=self.lambda_,
            value=self.loss_fn(*args, **kwargs),
            reduce_fn=self.reduce_fn,
            reduce_mask=reduce_mask,
        )


class BaseRenderLoss(BaseLoss):
    """
    Losses that are applied to the rendered results given ground truth.
    """

    @abstractmethod
    def forward(
        self,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]: ...


class BasePrimitiveLoss(BaseLoss):
    """Loss is applied to context batch and a reconstructed NRM primitive."""

    needs_target: bool = False

    @abstractmethod
    def forward(
        self,
        primitive: BaseNRMPrimitive,
        supervision_pack: BaseNRMSupervisionPack,
        context: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]: ...


class BaseLossWithConfidence(BaseLoss):
    """
    Confidence prediction loss.

    Confidence-weighting is active only when both ``confidence`` (per-call tensor) and
    ``confidence_alpha_`` (config float) are set; if either is ``None`` the unweighted base loss is
    returned unchanged. This lets callers route a possibly-unused ``confidence`` tensor through a
    shared code path -- e.g. primitives that always predict a confidence head, even when paired
    with a loss instance that is configured without ``confidence_alpha_``.
    """

    confidence_alpha_: float | None = None

    def __init__(self, loss_name: str, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__(loss_name, config, trainer_config, **kwargs)
        self.confidence_alpha_ = config.confidence_alpha_

    def apply_loss_fn(
        self,
        *args,
        confidence: torch.Tensor | None = None,
        reduce_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> LossReturn:
        """
        Apply the base loss and, when ``confidence`` and ``self.confidence_alpha_`` are both set,
        reweight the per-element loss by ``confidence`` with a ``-alpha * log(confidence)``
        regularizer. If either is ``None``, the unweighted base loss is returned unchanged (see
        class docstring for why this is a silent no-op).
        """
        loss_return = super().apply_loss_fn(*args, reduce_mask=reduce_mask, **kwargs)

        if confidence is None or self.confidence_alpha_ is None:
            return loss_return

        loss_value = loss_return.value

        ndim_a = confidence.ndim
        assert confidence.shape[:ndim_a] == loss_value.shape[:ndim_a], (
            f"Confidence and value must have the same prefix shape, but got {confidence.shape} and {loss_value.shape}."
        )
        confidence = confidence.view(*confidence.shape, *[1] * (loss_value.ndim - ndim_a))
        loss_value = loss_value * confidence - self.confidence_alpha_ * torch.log(confidence)

        return replace(loss_return, value=loss_value, _reduced_value=None)


class BaseLossWithSemanticWeights(BaseLossWithConfidence):
    """
    Losses that will parse additional mask_semantic_classes and semantic_lambdas keys into class variables.
    """

    def __init__(self, loss_name: str, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__(loss_name, config, trainer_config, **kwargs)
        self.mask_semantic_classes = config.mask_semantic_classes or []
        self.semantic_lambdas = config.semantic_lambdas or []
        assert len(self.mask_semantic_classes) == len(self.semantic_lambdas), (
            "mask_semantic_classes: list[str] and semantic_lambdas: list[float] should be of same length"
        )

        # This indicates whether the loss_value is per-image or per-ray.
        self.per_image_loss = config.per_image_loss or False

    @staticmethod
    def _masked_value(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert value.shape[: mask.ndim] == mask.shape, (
            f"Value and mask must have the same prefix shape, but got {value.shape} and {mask.shape}."
        )
        mask = mask.view(*mask.shape, *[1] * (value.ndim - mask.ndim))
        return value * mask

    def apply_loss_fn(
        self,
        *args: Any,
        frame_labels: LidarFrameLabels | CameraFrameLabels | None = None,
        frame_labels_mask: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
        reduce_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> LossReturn:
        """
        Apply loss-specific function to arguments and create return
        """
        loss_return = super().apply_loss_fn(*args, confidence=confidence, reduce_mask=reduce_mask, **kwargs)
        loss_value = loss_return.value

        if len(self.mask_semantic_classes) > 0:
            assert frame_labels is not None, "frame_labels should be provided when mask_semantic_classes is not empty"
            semantic_loss_values: list[torch.Tensor] = []
            for semantic_class, semantic_lambda in zip(self.mask_semantic_classes, self.semantic_lambdas):
                mask_semantic = get_mask_semantic(frame_labels, semantic_class)

                if self.per_image_loss:
                    # loss_value's shape is [B, *], and mask_semantic's shape is [N], reduce to [B]
                    assert frame_labels_mask is None, "frame_labels_mask should be None when per_image_loss is True"
                    num_images = unpack_optional(frame_labels.b)
                    mask_semantic = mask_semantic.view(num_images, -1).float().mean(dim=1) > 0

                elif frame_labels_mask is not None:
                    mask_semantic = mask_semantic[frame_labels_mask]

                # By convention, we zero-out non-masked region instead of masking it mainly to keep the denominator.
                # This also makes sure that the loss is not NaN if the mask is empty.
                semantic_loss_values.append(semantic_lambda * self._masked_value(loss_value, mask_semantic.squeeze(-1)))
            loss_value = cast(torch.Tensor, sum(semantic_loss_values))

        return replace(loss_return, value=loss_value, _reduced_value=None)


class SlangBaseLoss(BaseLoss):
    """Concrete implementation of BaseLoss for Slang-based losses."""

    def __init__(self, loss_name: str, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs: Any) -> None:
        super().__init__(loss_name, config, trainer_config, **kwargs)

        # Store all custom config attributes dynamically (e.g., layer_name, road_z_scale, layer_lambdas)
        # This allows Slang losses to access the same attributes as their Python counterparts
        for key, value in config.model_dump().items():
            if key not in ("fn", "lambda_", "reduce", "start_step", "lambda_scheduler"):
                setattr(self, str(key), value)


__all__ = [
    "BaseLoss",
    "BaseRenderLoss",
    "BasePrimitiveLoss",
    "BaseLossWithSemanticWeights",
    "SlangBaseLoss",
]
