# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

from typing import Any, Literal

from pydantic import model_validator

from nre.config.base_schema import BaseConfigSchema, Field


# Scheduler names that are known to be supported. The code also accepts any name from
# torch.optim.lr_scheduler via hasattr fallback, so this is not exhaustive.
COMPOSITE_SCHEDULER_NAMES = {"SequentialLR", "ChainedScheduler"}


class SchedulerConfig(BaseConfigSchema):
    """Configuration for a PyTorch learning rate scheduler.

    Supports simple schedulers (ExponentialLR, CosineAnnealingLR, etc.) and
    composite schedulers (SequentialLR, ChainedScheduler) via the `schedulers`
    and `milestones` fields.
    """

    name: str = Field(description="Name of the scheduler class (e.g. 'ExponentialLR', 'CosineAnnealingLR').")
    interval: Literal["epoch", "step"] = Field(
        default="epoch", description="Whether to step the scheduler per epoch or per training step."
    )
    args: dict[str, Any] = Field(
        default_factory=dict, description="Keyword arguments forwarded to the scheduler constructor."
    )

    # For composite schedulers (SequentialLR, ChainedScheduler)
    schedulers: list[SchedulerConfig] | None = Field(
        default=None, description="Nested scheduler configs for composite schedulers (SequentialLR, ChainedScheduler)."
    )
    milestones: list[int] | None = Field(
        default=None, description="Step milestones for SequentialLR to switch between nested schedulers."
    )

    @model_validator(mode="after")
    def _validate_composite_fields(self) -> SchedulerConfig:
        if self.name in COMPOSITE_SCHEDULER_NAMES:
            if self.schedulers is None:
                raise ValueError(f"{self.name} requires 'schedulers' to be set")
            if self.name == "SequentialLR":
                if self.milestones is None:
                    raise ValueError("SequentialLR requires 'milestones' to be set")
                if len(self.milestones) != len(self.schedulers) - 1:
                    raise ValueError(
                        f"SequentialLR: len(milestones) must equal len(schedulers) - 1, "
                        f"got {len(self.milestones)} milestones for {len(self.schedulers)} schedulers"
                    )
                if self.milestones != sorted(self.milestones):
                    raise ValueError(f"SequentialLR: milestones must be non-decreasing, got {self.milestones}")
        else:
            if self.schedulers is not None:
                raise ValueError(f"'schedulers' is only valid for composite schedulers, not '{self.name}'")
            if self.milestones is not None:
                raise ValueError(f"'milestones' is only valid for SequentialLR, not '{self.name}'")
        return self


class OptimizerConfig(BaseConfigSchema):
    """Configuration for a PyTorch optimizer."""

    name: Literal["fused_adam", "adam", "sgd"] = Field(
        description="Name of the optimizer class. Must match a key in optim_fns."
    )
    args: dict[str, Any] = Field(
        default_factory=dict, description="Keyword arguments forwarded to the optimizer constructor (lr, betas, etc.)."
    )
    params: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Parameter group overrides: maps group names to their argument dicts. "
            "Kept as dict[str, Any] because the nesting varies (some groups have sub-groups)."
        ),
    )
    scheduler: SchedulerConfig | None = Field(
        default=None, description="Optional learning rate scheduler attached to this optimizer."
    )


class TrackOptimConfig(BaseConfigSchema):
    """Configuration for track-level optimization (track_albedo, track_scale)."""

    start_global_step: int = Field(default=0, description="Global training step at which this optimization begins.")
    optimizer: OptimizerConfig = Field(description="Optimizer config for this track parameter.")
    scheduler: SchedulerConfig | None = Field(default=None, description="Optional scheduler for this track parameter.")
