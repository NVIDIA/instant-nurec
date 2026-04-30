# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from pathlib import Path

import pytest

from click.testing import CliRunner
from python.runfiles import runfiles

from nre.nrm.run import main


RUNFILES = runfiles.Create()


@pytest.fixture
def small_nrm_dataset_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    data_path = Path(
        RUNFILES.Rlocation(
            "test_data_ncore/cf5ff7f6-5c82-11ed-806f-00044bf655de_1667597307250262_1667597318349978_1667597307250262_1667597308250262.json"
        ),
    )
    if not data_path.exists():
        raise AssertionError(
            f"Test dataset not found. This is an issue with your filesystem/test suite, not the code under test. Missing {data_path=}"
        )
    nrm_data_root = tmp_path_factory.mktemp("nrm_data")
    with (nrm_data_root / "train.lst").open("w") as f:
        f.write(f"{data_path.resolve()}\n")

    return nrm_data_root


@pytest.mark.parametrize(
    "config_name",
    [
        "configs/nrm/apps/celsius_ci.yaml",
        "configs/nrm/apps/kelvin_ci.yaml",
    ],
)
def test_training(config_name: str, small_nrm_dataset_path: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Smoke test to make sure NRM training runs without crashing."""

    print("Output root:", output_root := tmp_path_factory.mktemp(config_name.replace("/", "_")))

    # Ensure the path is a *quoted* string for Hydra compatibility with bazel's `~`-separated paths
    small_nrm_dataset_path_train_lst_str = '"{0}"'.format(str(small_nrm_dataset_path / "train.lst"))

    common_args = [
        f"--config-name={config_name}",
        f"dataset.train.ncore_json_list_path={small_nrm_dataset_path_train_lst_str}",
        f"dataset.val.ncore_json_list_path={small_nrm_dataset_path_train_lst_str}",
        f"dataset.test.ncore_json_list_path={small_nrm_dataset_path_train_lst_str}",
        f"out_dir={output_root}",
        "logger=wandb",
        "logger.offline=true",
        "logger.run_id=out",  # fixes the output subdir, avoids generating a random hash
        # reduce GPU memory consumption to prevent GPU OOM issues in CI machines
        "system.device_count=1",
        "system.num_nodes=1",
    ]

    # Don't use/download pretrained weights for Kelvin CI test
    if config_name == "configs/nrm/apps/kelvin_ci.yaml":
        common_args.append("~model.init_weights_paths")

    result = CliRunner().invoke(main, common_args, catch_exceptions=False)
    assert result.exit_code == 0
