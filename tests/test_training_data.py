# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

import instant_nurec.datasets.datamodule as predict_datamodule_module
import instant_nurec.datasets.mixture as mixture_module
from instant_nurec.config_schema.dataset import NCoreInstantNuRecDatasetConfig
from instant_nurec.datasets.datamodule import InstantNuRecDataModule
from instant_nurec.datasets.instantnurec_base import BaseInstantNuRecIndexableDataset, InstantNuRecDataError
from instant_nurec.datasets.instantnurec_ncore import NCoreInstantNuRecDataset
from instant_nurec.datasets.mixture import NCoreMixtureDataset
from instant_nurec.training.data import KelvinTrainingDataModule


def test_waymo_lowercase_cyclist_is_unconditionally_dynamic() -> None:
    assert "cyclist" in NCoreInstantNuRecDataset.UNCONDITIONALLY_DYNAMIC_LABELS


def test_dataset_rng_matches_bazel_sha256_epoch_item_seed(monkeypatch) -> None:
    dataset = object.__new__(NCoreInstantNuRecDataset)
    dataset._rng_epoch = 7
    dataset._epoch = 7
    monkeypatch.setenv("PL_GLOBAL_SEED", "38")

    actual = dataset._get_rng(11).integers(0, 2**31, size=8)
    digest = hashlib.sha256(b"7_11_38").digest()
    expected = np.random.default_rng(int.from_bytes(digest[:8], "big")).integers(0, 2**31, size=8)
    np.testing.assert_array_equal(actual, expected)

    # Reinitialization makes an item stable; changing the train RNG epoch changes it.
    np.testing.assert_array_equal(actual, dataset._get_rng(11).integers(0, 2**31, size=8))
    dataset.set_rng_epoch(8)
    assert not np.array_equal(actual, dataset._get_rng(11).integers(0, 2**31, size=8))


def test_dataset_explicit_seed_works_without_lightning_environment(monkeypatch) -> None:
    dataset = object.__new__(NCoreInstantNuRecDataset)
    dataset._rng_epoch = -1
    dataset._global_seed = 38
    monkeypatch.delenv("PL_GLOBAL_SEED", raising=False)

    actual = dataset._get_rng(11).integers(0, 2**31, size=8)
    digest = hashlib.sha256(b"-1_11_38").digest()
    expected = np.random.default_rng(int.from_bytes(digest[:8], "big")).integers(0, 2**31, size=8)

    np.testing.assert_array_equal(actual, expected)


def _bare_ncore_retry_dataset() -> NCoreInstantNuRecDataset:
    dataset = object.__new__(NCoreInstantNuRecDataset)
    dataset._global_seed = 38
    dataset._rng_epoch = 7
    dataset._epoch = 7
    dataset._retry_on_error = True
    dataset.ncore_json_paths = [f"sequence-{index}.json" for index in range(32)]
    dataset.num_samples_per_sequence = 1
    return dataset


def test_ncore_training_retry_is_deterministic_and_prediction_fails_loud(monkeypatch) -> None:
    dataset = _bare_ncore_retry_dataset()
    attempts: list[int] = []

    def fail_one_index(batch_idx: int, rng: np.random.Generator):
        del rng
        attempts.append(batch_idx)
        if batch_idx == 5:
            raise InstantNuRecDataError("short sequence")
        return batch_idx

    monkeypatch.setattr(dataset, "getitem_allow_exceptions", fail_one_index)
    first_result = dataset[5]
    first_attempts = attempts.copy()
    attempts.clear()
    second_result = dataset[5]

    assert first_result == second_result
    assert attempts == first_attempts
    assert first_attempts[0] == 5
    assert len(first_attempts) == len(set(first_attempts)) == 2

    dataset._retry_on_error = False
    attempts.clear()
    with pytest.raises(InstantNuRecDataError, match="short sequence"):
        dataset[5]
    assert attempts == [5]


def test_ncore_training_retry_stops_after_exactly_ten_unique_attempts(monkeypatch) -> None:
    dataset = _bare_ncore_retry_dataset()
    attempts: list[int] = []

    def always_fail(batch_idx: int, rng: np.random.Generator):
        del rng
        attempts.append(batch_idx)
        raise InstantNuRecDataError("still broken")

    monkeypatch.setattr(dataset, "getitem_allow_exceptions", always_fail)

    with pytest.raises(InstantNuRecDataError, match="tried out 10 attempts"):
        dataset[5]

    assert len(attempts) == len(set(attempts)) == 10


def test_predict_datamodule_passes_config_seed_without_lightning_environment(monkeypatch) -> None:
    seen = {}

    class FakeDataset:
        def __init__(self, config, **kwargs) -> None:
            del config
            seen.update(kwargs)

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            return index

    monkeypatch.delenv("PL_GLOBAL_SEED", raising=False)
    monkeypatch.setattr(predict_datamodule_module, "NCoreInstantNuRecDataset", FakeDataset)
    dataset_config = SimpleNamespace(
        camera_subsampler=SimpleNamespace(frame_width=64, frame_height=48),
        frame_batch_sampler=SimpleNamespace(n_frames_per_sample=4),
    )
    config = SimpleNamespace(
        seed=38,
        dataset=SimpleNamespace(predict=dataset_config),
        system=SimpleNamespace(predict_num_workers=0, predict_batch_size=1),
    )

    InstantNuRecDataModule(config).predict_dataloader()

    assert seen["global_seed"] == 38


def test_training_datamodule_enables_retries_for_direct_and_mixture_datasets(monkeypatch) -> None:
    seen: list[tuple[str, dict]] = []

    def fake_ncore_init(self, config, **kwargs) -> None:
        del self, config
        seen.append(("ncore", kwargs))

    def fake_mixture_init(self, config, **kwargs) -> None:
        del self, config
        seen.append(("mixture", kwargs))

    monkeypatch.setattr(NCoreInstantNuRecDataset, "__init__", fake_ncore_init)
    monkeypatch.setattr(NCoreMixtureDataset, "__init__", fake_mixture_init)
    datamodule = KelvinTrainingDataModule(SimpleNamespace(seed=38))

    direct_config = NCoreInstantNuRecDatasetConfig(ncore_json_paths=["unused.json"])
    datamodule._make_dataset(direct_config)
    datamodule._make_dataset(SimpleNamespace())

    assert seen[0][0] == "ncore"
    assert seen[0][1]["retry_on_error"] is True
    assert seen[1] == (
        "mixture",
        {
            "global_seed": 38,
            "retry_on_error": True,
        },
    )


def test_mixture_matches_bazel_length_routing_and_epoch_contract(monkeypatch) -> None:
    class FakeDataset:
        created: list["FakeDataset"] = []

        def __init__(self, config, **kwargs) -> None:
            del kwargs
            self.name = config.name
            self.length = config.length
            self.rng_epochs: list[int] = []
            self.epochs: list[int] = []
            self.created.append(self)

        def __len__(self) -> int:
            return self.length

        def __getitem__(self, index: int):
            return self.name, index

        def set_rng_epoch(self, value: int) -> None:
            self.rng_epochs.append(value)

        def set_epoch(self, value: int) -> None:
            self.epochs.append(value)

    monkeypatch.setattr(mixture_module, "NCoreInstantNuRecDataset", FakeDataset)
    monkeypatch.setenv("PL_GLOBAL_SEED", "38")

    def component(name: str, length: int, ratio: float):
        dataset_config = SimpleNamespace(
            name=name,
            length=length,
            camera_subsampler=SimpleNamespace(frame_width=1, frame_height=1),
            frame_batch_sampler=SimpleNamespace(n_frames_per_sample=1),
        )
        return SimpleNamespace(sample_ratio=ratio, config=dataset_config)

    config = SimpleNamespace(
        mixture={
            "undersampled": component("under", 4, 0.5),
            "oversampled": component("over", 3, 2.0),
            "disabled": component("off", 9, 0.0),
        }
    )
    dataset = NCoreMixtureDataset(config)

    # int(4 * .5) + int(3 * 2) and zero-ratio components are omitted.
    assert len(dataset) == 8
    assert [item.name for item in dataset.datasets] == ["undersampled", "oversampled"]
    assert len(FakeDataset.created) == 2

    dataset.set_rng_epoch(7)
    dataset.set_epoch(9)
    assert all(item.rng_epochs == [7] for item in FakeDataset.created)
    assert all(item.epochs == [9] for item in FakeDataset.created)

    # Oversampling preserves one ordered pass, then draws only the excess.
    assert [dataset[index] for index in range(2, 5)] == [("over", 0), ("over", 1), ("over", 2)]
    extra = [dataset[index] for index in range(5, 8)]
    assert all(name == "over" and 0 <= index < 3 for name, index in extra)

    # SHA-derived per-item RNG makes both undersampling and excess draws stable.
    assert dataset[0] == dataset[0]
    assert dataset[7] == dataset[7]


class _RetryingFakeNCoreDataset(BaseInstantNuRecIndexableDataset[tuple[str, int]]):
    created: list["_RetryingFakeNCoreDataset"] = []

    def __init__(self, config, *, global_seed: int | None = None, retry_on_error: bool = False, **kwargs) -> None:
        del kwargs
        self.name = config.name
        self.length = config.length
        self.fail_indices = config.fail_indices
        self.fail_all = config.fail_all
        self._global_seed = 38 if global_seed is None else global_seed
        self._rng_epoch = -1
        self._epoch = -1
        self._retry_on_error = retry_on_error
        self.attempts: list[int] = []
        self.created.append(self)

    def __len__(self) -> int:
        return self.length

    def set_rng_epoch(self, value: int) -> None:
        self._rng_epoch = value

    def set_epoch(self, value: int) -> None:
        self._epoch = value

    def _get_rng(self, batch_idx: int) -> np.random.Generator:
        digest = hashlib.sha256(f"{self._rng_epoch}_{batch_idx}_{self._global_seed}".encode()).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], "big"))

    def getitem_allow_exceptions(self, batch_idx: int, rng: np.random.Generator) -> tuple[str, int]:
        del rng
        self.attempts.append(batch_idx)
        if self.fail_all or batch_idx in self.fail_indices:
            raise InstantNuRecDataError(f"bad component item {batch_idx}")
        return self.name, batch_idx


def _retry_mixture_config(*, fail_indices: set[int] | None = None, fail_all: bool = False):
    dataset_config = SimpleNamespace(
        name="component",
        length=32,
        fail_indices=set() if fail_indices is None else fail_indices,
        fail_all=fail_all,
        camera_subsampler=SimpleNamespace(frame_width=1, frame_height=1),
        frame_batch_sampler=SimpleNamespace(n_frames_per_sample=1),
    )
    return SimpleNamespace(
        mixture={
            "component": SimpleNamespace(
                sample_ratio=1.0,
                config=dataset_config,
            )
        }
    )


def test_mixture_component_retries_before_outer_mixture(monkeypatch) -> None:
    _RetryingFakeNCoreDataset.created.clear()
    monkeypatch.setattr(mixture_module, "NCoreInstantNuRecDataset", _RetryingFakeNCoreDataset)
    dataset = NCoreMixtureDataset(
        _retry_mixture_config(fail_indices={0}),
        global_seed=38,
        retry_on_error=True,
    )
    outer_attempts: list[int] = []
    outer_getitem = dataset.getitem_allow_exceptions

    def trace_outer(batch_idx: int, rng: np.random.Generator):
        outer_attempts.append(batch_idx)
        return outer_getitem(batch_idx, rng)

    monkeypatch.setattr(dataset, "getitem_allow_exceptions", trace_outer)

    result = dataset[0]
    component = _RetryingFakeNCoreDataset.created[0]

    assert outer_attempts == [0]
    assert component._retry_on_error is True
    assert component.attempts[0] == 0
    assert len(component.attempts) == len(set(component.attempts)) == 2
    assert result == ("component", component.attempts[-1])


def test_mixture_retries_after_component_exhausts_ten_attempts(monkeypatch, caplog) -> None:
    caplog.set_level("CRITICAL")
    _RetryingFakeNCoreDataset.created.clear()
    monkeypatch.setattr(mixture_module, "NCoreInstantNuRecDataset", _RetryingFakeNCoreDataset)
    dataset = NCoreMixtureDataset(
        _retry_mixture_config(fail_all=True),
        global_seed=38,
        retry_on_error=True,
    )
    outer_attempts: list[int] = []
    outer_getitem = dataset.getitem_allow_exceptions

    def trace_outer(batch_idx: int, rng: np.random.Generator):
        outer_attempts.append(batch_idx)
        return outer_getitem(batch_idx, rng)

    monkeypatch.setattr(dataset, "getitem_allow_exceptions", trace_outer)

    with pytest.raises(InstantNuRecDataError, match="tried out 10 attempts"):
        dataset[0]

    component = _RetryingFakeNCoreDataset.created[0]
    assert len(outer_attempts) == len(set(outer_attempts)) == 10
    assert len(component.attempts) == 100
