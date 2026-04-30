# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import glob

import pytest

from nre.config.parse import parse_typed_config


so_app_configs = glob.glob("configs/apps/**/[!_]*.yaml", recursive=True)

# Exclude any "options" subdirectories, as these contain config groups (not fully formed configs).
so_app_configs = [conf for conf in so_app_configs if "/options/" not in conf]


@pytest.mark.parametrize("config_name", so_app_configs)
def test_all_so_configs_are_valid(config_name: str) -> None:
    """
    A simple test which checks if all app configs are gramatically valid w.r.t. the config schema.
    It does not check if they actually lead to correct code execution.

    This test checks SO (scene optimization) configs.
    """

    hydra_args = [
        "dataset.path=/does/not/matter",
        "out_dir=/does/not/matter",
    ]

    parse_typed_config(config_name=config_name, hydra_args=hydra_args)
