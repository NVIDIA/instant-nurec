# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

import pytest

from pydantic import ValidationError

from nre.config.optim import OptimizerConfig, SchedulerConfig, TrackOptimConfig


def test_simple_optimizer() -> None:
    config = OptimizerConfig.model_validate(
        {
            "name": "fused_adam",
            "args": {"lr": 0.01, "eps": 1e-15, "betas": [0.9, 0.99]},
        }
    )
    assert config.name == "fused_adam"
    assert config.args["lr"] == 0.01
    assert config.params is None
    assert config.scheduler is None


def test_optimizer_with_params_and_scheduler() -> None:
    config = OptimizerConfig.model_validate(
        {
            "name": "fused_adam",
            "args": {"lr": 0.01, "eps": 1e-15, "betas": [0.9, 0.99]},
            "params": {
                "positions": {"args": {"lr": 0.00016}},
                "densities": {"args": {"lr": 0.05}},
            },
            "scheduler": {
                "name": "ExponentialLR",
                "interval": "step",
                "args": {"gamma": 0.9999},
            },
        }
    )
    assert config.name == "fused_adam"
    assert config.params is not None
    assert config.params["positions"]["args"]["lr"] == 0.00016
    assert config.scheduler is not None
    assert config.scheduler.name == "ExponentialLR"
    assert config.scheduler.interval == "step"


def test_sequential_lr_scheduler() -> None:
    config = SchedulerConfig.model_validate(
        {
            "name": "SequentialLR",
            "interval": "step",
            "milestones": [250, 350],
            "schedulers": [
                {"name": "ConstantLR", "args": {"factor": 0.0, "total_iters": 250}},
                {
                    "name": "LinearLR",
                    "interval": "step",
                    "args": {"start_factor": 0.01, "end_factor": 1.0, "total_iters": 100},
                },
                {
                    "name": "StepFunCosineAnnealingLR",
                    "interval": "step",
                    "args": {"T_max": 10000, "min_factor": 0.0333, "update_every_n_steps": 1},
                },
            ],
        }
    )
    assert config.name == "SequentialLR"
    assert config.milestones == [250, 350]
    assert config.schedulers is not None
    assert len(config.schedulers) == 3
    assert config.schedulers[0].name == "ConstantLR"
    assert config.schedulers[1].name == "LinearLR"
    assert config.schedulers[2].name == "StepFunCosineAnnealingLR"


def test_track_optim_config() -> None:
    config = TrackOptimConfig.model_validate(
        {
            "start_global_step": 500,
            "optimizer": {
                "name": "fused_adam",
                "args": {"lr": 1e-5, "eps": 1e-15, "betas": [0.9, 0.99]},
            },
            "scheduler": {
                "name": "SequentialLR",
                "interval": "step",
                "milestones": [500, 600],
                "schedulers": [
                    {"name": "ConstantLR", "args": {"factor": 0.0, "total_iters": 500}},
                    {"name": "LinearLR", "args": {"start_factor": 0.01, "end_factor": 1.0, "total_iters": 100}},
                    {"name": "ExponentialLR", "args": {"gamma": 0.999}},
                ],
            },
        }
    )
    assert config.start_global_step == 500
    assert config.optimizer.name == "fused_adam"
    assert config.scheduler is not None
    assert config.scheduler.name == "SequentialLR"


def test_scheduler_validation_rejects_schedulers_on_simple() -> None:
    """Simple schedulers must not have 'schedulers' or 'milestones' fields."""
    with pytest.raises(ValidationError, match="only valid for composite"):
        SchedulerConfig.model_validate(
            {"name": "ExponentialLR", "args": {"gamma": 0.99}, "schedulers": [{"name": "ConstantLR"}]}
        )


def test_scheduler_validation_rejects_milestones_on_simple() -> None:
    with pytest.raises(ValidationError, match="only valid for SequentialLR"):
        SchedulerConfig.model_validate({"name": "ExponentialLR", "args": {"gamma": 0.99}, "milestones": [100]})


def test_scheduler_validation_requires_schedulers_on_sequential() -> None:
    with pytest.raises(ValidationError, match="requires 'schedulers'"):
        SchedulerConfig.model_validate({"name": "SequentialLR", "milestones": [100]})


def test_scheduler_validation_requires_milestones_on_sequential() -> None:
    with pytest.raises(ValidationError, match="requires 'milestones'"):
        SchedulerConfig.model_validate(
            {
                "name": "SequentialLR",
                "schedulers": [{"name": "ConstantLR", "args": {"factor": 0.0, "total_iters": 100}}],
            }
        )


def test_scheduler_validation_milestones_count_mismatch() -> None:
    """len(milestones) must equal len(schedulers) - 1."""
    with pytest.raises(ValidationError, match="len\\(milestones\\) must equal len\\(schedulers\\) - 1"):
        SchedulerConfig.model_validate(
            {
                "name": "SequentialLR",
                "milestones": [100, 200, 300],
                "schedulers": [
                    {"name": "ConstantLR", "args": {"factor": 0.0, "total_iters": 100}},
                    {"name": "LinearLR", "args": {"start_factor": 0.01, "end_factor": 1.0, "total_iters": 100}},
                ],
            }
        )


def test_scheduler_validation_milestones_not_increasing() -> None:
    with pytest.raises(ValidationError, match="non-decreasing"):
        SchedulerConfig.model_validate(
            {
                "name": "SequentialLR",
                "milestones": [300, 100],
                "schedulers": [
                    {"name": "ConstantLR", "args": {"factor": 0.0, "total_iters": 100}},
                    {"name": "LinearLR", "args": {"start_factor": 0.01, "end_factor": 1.0, "total_iters": 100}},
                    {"name": "ExponentialLR", "args": {"gamma": 0.99}},
                ],
            }
        )


def test_chained_scheduler() -> None:
    config = SchedulerConfig.model_validate(
        {
            "name": "ChainedScheduler",
            "schedulers": [
                {"name": "ExponentialLR", "args": {"gamma": 0.99}},
                {"name": "ConstantLR", "args": {"factor": 0.5, "total_iters": 10}},
            ],
        }
    )
    assert config.name == "ChainedScheduler"
    assert config.schedulers is not None
    assert len(config.schedulers) == 2


def test_optimizer_name_literal() -> None:
    """OptimizerConfig.name must be one of the supported optimizer names."""
    with pytest.raises(ValidationError, match="Input should be 'fused_adam', 'adam' or 'sgd'"):
        OptimizerConfig.model_validate({"name": "unsupported_optimizer"})


def test_optimizer_to_dictconfig() -> None:
    """Verify round-trip through to_dictconfig for backward compatibility."""
    config = OptimizerConfig.model_validate(
        {
            "name": "fused_adam",
            "args": {"lr": 0.01},
            "params": {"positions": {"args": {"lr": 0.00016}}},
        }
    )
    dc = config.to_dictconfig()
    assert dc.name == "fused_adam"
    assert dc.args.lr == 0.01
    assert dc.params.positions.args.lr == 0.00016
