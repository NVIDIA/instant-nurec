# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

import pytest

from pydantic import ValidationError

from nre.config.model import DensifyConfig, GsplatStrategyConfig, PruneConfig, ResetDensityConfig


REQUIRED_FIELDS = {"frequency": 100, "start_iteration": 500, "end_iteration": 15_000}


def test_densify_config_defaults() -> None:
    config = DensifyConfig(**REQUIRED_FIELDS)
    assert config.frequency == 100
    assert config.start_iteration == 500
    assert config.end_iteration == 15_000
    assert config.clone_grad_threshold == 0.0002
    assert config.split_grad_threshold == 0.0002
    assert config.relative_size_threshold == 0.01
    assert config.split_n_gaussians == 2


def test_prune_config_defaults() -> None:
    config = PruneConfig(**REQUIRED_FIELDS)
    assert config.density_threshold == 0.01


def test_reset_density_config_defaults() -> None:
    config = ResetDensityConfig(**REQUIRED_FIELDS)
    assert config.new_max_density == 0.01


@pytest.mark.parametrize("cls", [DensifyConfig, PruneConfig, ResetDensityConfig])
@pytest.mark.parametrize("missing", ["frequency", "start_iteration", "end_iteration"])
def test_required_fields_are_enforced(cls: type, missing: str) -> None:
    args = {k: v for k, v in REQUIRED_FIELDS.items() if k != missing}
    with pytest.raises(ValidationError):
        cls(**args)


@pytest.mark.parametrize("cls", [DensifyConfig, PruneConfig, ResetDensityConfig])
def test_frequency_must_be_positive(cls: type) -> None:
    args = {**REQUIRED_FIELDS, "frequency": 0}
    with pytest.raises(ValidationError):
        cls(**args)


@pytest.mark.parametrize("cls", [DensifyConfig, PruneConfig, ResetDensityConfig])
@pytest.mark.parametrize("field", ["start_iteration", "end_iteration"])
def test_iteration_fields_reject_negative(cls: type, field: str) -> None:
    args = {**REQUIRED_FIELDS, field: -1}
    with pytest.raises(ValidationError):
        cls(**args)


def test_prune_density_threshold_is_bounded() -> None:
    with pytest.raises(ValidationError):
        PruneConfig(**REQUIRED_FIELDS, density_threshold=-0.1)
    with pytest.raises(ValidationError):
        PruneConfig(**REQUIRED_FIELDS, density_threshold=1.5)


def test_reset_density_new_max_density_is_bounded() -> None:
    with pytest.raises(ValidationError):
        ResetDensityConfig(**REQUIRED_FIELDS, new_max_density=-0.1)
    with pytest.raises(ValidationError):
        ResetDensityConfig(**REQUIRED_FIELDS, new_max_density=1.5)


def test_gsplat_strategy_config_all_none_path() -> None:
    """Validates the all-None path that the rest of the codebase relies on."""
    config = GsplatStrategyConfig.model_validate({"name": "gsplat"})
    assert config.densify is None
    assert config.prune is None
    assert config.reset_density is None


def test_gsplat_strategy_config_with_sub_configs() -> None:
    config = GsplatStrategyConfig.model_validate(
        {
            "name": "gsplat",
            "densify": REQUIRED_FIELDS,
            "prune": REQUIRED_FIELDS,
            "reset_density": REQUIRED_FIELDS,
        }
    )
    assert config.densify is not None and config.densify.frequency == 100
    assert config.prune is not None and config.prune.frequency == 100
    assert config.reset_density is not None and config.reset_density.frequency == 100
