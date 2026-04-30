# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss configuration and return types.

Uses nre.config.base_schema.BaseConfigSchema as base (no other nre imports there, so no circular deps).
Config classes first; loss return/reduction types at the end (so config is a single file for stable import order).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Protocol, TypeVar

import torch

from pydantic import Field, RootModel

from nre.config.base_schema import BaseConfigSchema


class ReduceConfig(BaseConfigSchema):
    """Configuration for loss reduction functions."""

    name: str = Field(description="Reduction function type")
    quantile: float | None = None


class LambdaSchedulerConfig(BaseConfigSchema):
    """Configuration for lambda scheduler."""

    name: str = Field(description="Lambda scheduler name")
    start: int
    end: int
    update_interval: str
    update_frequency: int
    lambda_end: float


class LossItemConfig(BaseConfigSchema):
    """Configuration for a single loss item."""

    class MaskConfig(BaseConfigSchema):
        mode: str
        value: float

    fn: str = Field(description="Loss function name (l1, mse, ssim, cross_entropy, etc.)")
    lambda_: float
    reduce: ReduceConfig = Field(description="Configuration for reduction function")

    # Optional common fields
    start_step: int | None = None
    lambda_scheduler: LambdaSchedulerConfig | None = None
    visibility_filter: bool = False
    occlusion_aware: bool = False

    mask_semantic_classes: list[str] | None = None
    semantic_lambdas: list[float] | None = None

    confidence_alpha_: float | None = None

    per_image_loss: bool | None = Field(default=None, description="Whether loss is computed per-image or per-ray")

    # SSIM
    window_size: int | None = None
    channel: int | None = None
    mask: MaskConfig | None = None

    # PPISP
    lambdas: dict[str, float] | None = None

    # Road Gaussians
    layer_name: str | None = None
    n_samples: int | None = None
    grid_len: float | None = None
    min: float | None = None
    range: float | None = None
    rotation_lambda: float | None = None

    # BackgroundInTrackGaussian
    layer_names: list[str] | None = None
    density_logits_min: float = -20.0

    # Gaussian_scale, gaussian_density
    layer_lambdas: dict[str, float] | None = None

    # Gaussian Z Scale
    road_z_scale: float | None = None

    # Node Semantic Gaussians
    layer_labels_to_use: dict[str, list[str]] | None = None
    layer_labels_to_exclude: dict[str, list[str]] | None = None

    # Gaussian Flatten
    max_to_median_ratio_threshold: float | None = None
    axes_type: str | None = None

    # Distance and Normal losses
    allow_missing_supervision: bool | None = None
    min_distance: float | None = None
    max_distance: float | None = None
    normalize_by_opacity: bool | None = None

    # bce_clipped, bce_truncated
    clip_opacity: float | None = None

    # LPIPS
    limit_max_resolution: int | None = None

    # Primitive sky distance
    sky_distance: float | None = None

    # Primitive geometry-related (e.g. distance, depth, pointmap) loss
    scale_steps: list[int] | None = None
    scale_lambdas: list[float] | None = None
    use_z_depth: bool = Field(default=False, description="Whether to use z-depth for the geometry loss")

    # Primitive mask loss
    semantic_class: str | None = None

    # Primitive sky cubemap loss
    masked_region: Literal["skip", "average", "smooth"] = Field(
        default="average", description="How to handle masked regions"
    )

    # Primitive velocity loss
    velocity_mask_threshold: float | None = Field(
        default=None,
        description="Threshold for the velocity mask (loss only applied to region whose gt value > threshold). If None, no mask is applied.",
    )


class LossConfig(RootModel[dict[str, LossItemConfig]]):
    """
    Top-level loss configuration.

    Maps loss names to their configurations.
    """

    def __getitem__(self, key: str) -> LossItemConfig:
        """Allow dict-like access: loss_config['rgb']"""
        return self.root[key]

    def items(self):
        """Allow dict.items() usage."""
        return self.root.items()

    def keys(self):
        """Allow dict.keys() usage."""
        return self.root.keys()

    def values(self):
        """Allow dict.values() usage."""
        return self.root.values()

    def get(self, key: str, default=None):
        """Allow dict.get() usage."""
        return self.root.get(key, default)


# -----------------------------------------------------------------------------
# Loss return and reduction types (single file with config for stable order)
# -----------------------------------------------------------------------------


class RayReduceFn(Protocol):
    """Interface of a reduction function that can reduce to a single value."""

    def __call__(self, value: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Reduces inputs to single value."""
        ...


@dataclass(slots=True, kw_only=True)
class LossReturn:
    """
    Unified loss return type.

    - name: name of the loss (e.g. 'rgb_huber_mean')
    - lambda_: coefficient to weight the reduced loss value with
    - value: numerical value of the loss, not reduced; for per-ray losses
      value is only defined for valid rays (e.g. shape (n_valid_cam_rays,))
    - reduce_fn: function to reduce loss value with
    - reduce_mask: optional masked reduction
    """

    name: str
    lambda_: float
    value: torch.Tensor
    reduce_fn: RayReduceFn
    reduce_mask: Optional[torch.Tensor] = None
    _reduced_value: torch.Tensor | None = None

    @property
    def reduced_value(self) -> torch.Tensor:
        assert self._reduced_value is not None
        return self._reduced_value

    def __post_init__(self) -> None:
        assert callable(self.reduce_fn)
        if self._reduced_value is None:
            self._reduced_value = self.reduce()

    def reduce(self) -> torch.Tensor:
        """Performs loss value reduction and lambda-weighting."""
        return self.lambda_ * self.reduce_fn(self.value, reduce_mask=self.reduce_mask)


@dataclass(slots=True)
class LossAggregatorReturn:
    """Contains all the individual loss returns."""

    loss_returns: dict[str, LossReturn]
    total_value: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.total_value = self.reduce()

    def reduce(self) -> torch.Tensor:
        """Performs loss value reduction and lambda-weighting for all loss returns."""
        return torch.sum(torch.stack([lr.reduced_value for lr in self.loss_returns.values()]))

    @torch.no_grad()
    def log(self, logging_fn: Callable[..., None]) -> None:
        """Logging of all loss_return values and lambda coeffs."""
        for name, value in self.loss_returns.items():
            logging_fn(f"loss/{name}", value.reduced_value)
            logging_fn(f"train_params/{name}_lambda", value.lambda_)


@dataclass(slots=True)
class LossAggregatorBatchReturn:
    """
    Batch version of the LossAggregatorReturn.

    - batch_loss_returns: list of computed losses, one per batch element
    - total_value: total loss for the batch (precomputed)
    - extra_fields: optional batch step data (e.g. for visualizer)
    """

    batch_loss_returns: list[LossAggregatorReturn]
    total_value: torch.Tensor = field(init=False)
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.total_value = self.reduce()

    def reduce(self) -> torch.Tensor:
        """Mean over batch to be consistent with multi-node DDP."""
        return torch.mean(torch.stack([lr.total_value for lr in self.batch_loss_returns]))

    @torch.no_grad()
    def log(self, logging_fn: Callable[..., None], sync_dist: bool = False) -> None:
        """Logging of all loss_return values and lambda coeffs."""
        loss_values: dict[str, list[torch.Tensor]] = defaultdict(list)
        loss_lambdas: dict[str, list[float]] = defaultdict(list)

        for loss_returns in self.batch_loss_returns:
            for name, value in loss_returns.loss_returns.items():
                loss_values[name].append(value.reduced_value)
                loss_lambdas[name].append(value.lambda_)

        for name, values in loss_values.items():
            logging_fn(
                f"loss/{name}",
                torch.mean(torch.stack(values)),
                batch_size=len(values),
                sync_dist=sync_dist,
            )
            mean_lambda = sum(loss_lambdas[name]) / len(loss_lambdas[name])
            logging_fn(f"train_params/{name}_lambda", mean_lambda, batch_size=len(values))


LossReturnType = TypeVar("LossReturnType", LossAggregatorReturn, LossAggregatorBatchReturn)
