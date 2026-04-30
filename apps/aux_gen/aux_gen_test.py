# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import unittest

from pathlib import Path

import pytest
import torch

from click.testing import CliRunner
from python.runfiles import runfiles

from apps.aux_gen.ncore_aux_data import cli
from apps.aux_gen.utils import _quantile


RUNFILES = runfiles.Create()


@pytest.fixture
def small_dataset_path() -> Path:
    path = Path(
        RUNFILES.Rlocation(
            "test_data_ncore/cf5ff7f6-5c82-11ed-806f-00044bf655de_1667597307250262_1667597318349978_1667597307250262_1667597308250262.zarr.itar"
        ),
    )
    if not path.exists():
        raise AssertionError(
            f"Test dataset not found. This is an issue with your filesystem/test suite, not the code under test. Missing {path=}"
        )

    return path


def test_ncore_aux_data(small_dataset_path: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Integration test to validate execution of ncore_aux_data cli on small test data set"""

    print("Output root:", output_root := tmp_path_factory.mktemp("aux_gen_test"))

    # Use deprecated --shard-file-pattern: test_data_ncore provides a single .zarr.itar shard,
    # not a V3/V4 sequence meta file. Prefer --dataset-path when a meta file is available.
    result = CliRunner().invoke(
        cli,
        [
            f"--shard-file-pattern={small_dataset_path}",
            "--camera-id=camera_front_wide_120fov",
            f"--output-dir={output_root}",
            "--store-meta",
            "offset",
            "--sequence-duration-sec=0.4",  # restrict to few frames only to terminate more quickly
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0


def test_ncore_aux_data_no_ego_mask_creates_no_egomask_file(
    small_dataset_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """With --no-ego-mask, no *.aux.egomask.* file should be created (fix for mistaken empty store)."""
    output_root = tmp_path_factory.mktemp("aux_gen_test_no_egomask")
    # Use deprecated --shard-file-pattern (test data is single shard; see test_ncore_aux_data).
    result = CliRunner().invoke(
        cli,
        [
            f"--shard-file-pattern={small_dataset_path}",
            "--camera-id=camera_front_wide_120fov",
            f"--output-dir={output_root}",
            "--store-meta",
            "--no-ego-mask",
            "offset",
            "--sequence-duration-sec=0.4",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    egomask_files = list(output_root.rglob("*egomask*"))
    assert not egomask_files, f"Expected no egomask files with --no-ego-mask, found: {egomask_files}"


class TestQuantileFunction(unittest.TestCase):
    def test_torch_quantile(self) -> None:
        a = torch.randn(2, 3)
        x = torch.quantile(a, 0.90, dim=1, keepdim=True)
        y = _quantile(a, 0.90, dim=1, keepdim=True)
        torch.testing.assert_close(x, y)

        a = torch.randn(200, 300)
        x = torch.quantile(a, 0.90)
        y = _quantile(a, 0.90)
        torch.testing.assert_close(x, y)

    def test_torch_quantile_large(self) -> None:
        # If this test fails after upgrading torch, it is probably safe to remove the _quantile function.
        a = torch.randn(4096**2 + 1)
        with self.assertRaises(RuntimeError):
            torch.quantile(a, 0.90)
        _quantile(a, 0.90)
