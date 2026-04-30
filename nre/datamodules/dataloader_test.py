# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import hashlib

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import pytorch_lightning as pl
import torch

from python.runfiles import runfiles

import nre.datamodules

from nre.config.nre import NREConfig
from nre.config.parse import parse_typed_config


RUNFILES = runfiles.Create()


@pytest.fixture
def small_prod_config() -> str:
    return "apps/AV/NV/3dgut_dynamic.yaml"


@pytest.fixture
def small_dataset_path() -> Path:
    path = Path(
        RUNFILES.Rlocation(
            "test_data_ncore/cf5ff7f6-5c82-11ed-806f-00044bf655de_1667597307250262_1667597318349978_1667597307250262_1667597308250262.json"
        ),
    )
    if not path.exists():
        raise AssertionError(
            f"Test dataset not found. This is an issue with your filesystem/test suite, not the code under test. Missing {path=}"
        )
    return path


def hash_tensor(tensor: torch.Tensor) -> str:
    """Compute a hash of a torch.Tensor object"""
    tensor_bytes = tensor.numpy().tobytes()
    return hashlib.md5(tensor_bytes).hexdigest()


def compute_object_hashes(obj: Any) -> list[str]:
    """Returns a list of hashes of each torch.Tensor in the specified object"""
    hashes = []
    if obj is not None and not isinstance(obj, (int, float, str, bool)):
        if isinstance(obj, torch.Tensor):
            hashes.append(hash_tensor(obj))
        elif isinstance(obj, np.ndarray):
            hashes.append(hash_tensor(torch.from_numpy(obj)))
        elif isinstance(obj, list):
            for value in obj:
                hashes.extend(compute_object_hashes(value))
        else:
            attributes = sorted(dir(obj))
            variables = [attr for attr in attributes if not callable(getattr(obj, attr)) and not attr.startswith("__")]
            for name in variables:
                value = getattr(obj, name)
                hashes.extend(compute_object_hashes(value))
    return hashes


def hash_object_tensors(obj: Any) -> str:
    """Recursively find all torch.Tensor's in the object and compute a combined hash of their values"""
    hashes = compute_object_hashes(obj)
    return hashlib.md5("".join(hashes).encode()).hexdigest()


def run_deterministic_dataloader(config: NREConfig, datamodule: nre.datamodules.BaseDataModule) -> list[str]:
    # Reset the RNG seed to ensure dataloader starts in deterministic state
    pl.seed_everything(config.seed)
    hashes = []
    for batch_idx, batch in enumerate(datamodule.train_dataloader()):
        hash = hash_object_tensors(batch)
        print(f"TEST: batch {batch_idx}: {hash}")
        hashes.append(hash)
    return hashes


def test_deterministic_consecutive_runs(
    small_prod_config: str, small_dataset_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Test to validate that consecutive runs of the dataloader produces deterministics results"""

    output_root = tmp_path_factory.mktemp(small_prod_config.replace("/", "_"))
    n_samples_per_epoch = 2

    # Ensure the path is a *quoted* string for Hydra compatibility with bazel's `~`-separated paths
    small_dataset_path_str = '"{0}"'.format(small_dataset_path)

    hydra_args = [
        f"out_dir={output_root}",
        f"dataset.path={small_dataset_path_str}",
        f"dataset.n_samples_per_epoch={n_samples_per_epoch}",
        f"dataset.n_train_sample_lidar_rays=0",
        f"mode=train",
        f"logger.offline=true",
        f"logger.run_id=out",
    ]

    config = parse_typed_config(config_name=small_prod_config, hydra_args=hydra_args)
    datamodule = nre.datamodules.make(config.datamodule.name, config)

    print("TEST: starting dataloader run: 0")
    hashes0 = run_deterministic_dataloader(config, datamodule)

    print("TEST: starting dataloader run: 1")
    hashes1 = run_deterministic_dataloader(config, datamodule)

    assert len(hashes0) == len(hashes1) == n_samples_per_epoch
    assert hashes0 == hashes1
    for hash in hashes0:
        assert len(hash) == 32
