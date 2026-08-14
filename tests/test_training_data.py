# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np

import instant_nurec.datasets.datamodule as predict_datamodule_module
import instant_nurec.datasets.mixture as mixture_module
from instant_nurec.datasets.datamodule import InstantNuRecDataModule
from instant_nurec.datasets.instantnurec_ncore import NCoreInstantNuRecDataset
from instant_nurec.datasets.mixture import NCoreMixtureDataset


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
