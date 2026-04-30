# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
import math
import os

from pathlib import Path
from typing import Optional

import hydra

from omegaconf import DictConfig, OmegaConf, open_dict

from nre.config.base_schema import BaseConfigSchema, config_to_primitive
from nre.repo_root import __reporoot__
from nre.utils.misc import rank_zero_only


# ============ Register OmegaConf Recolvers ============= #
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("calc_exp_lr_decay_rate", lambda factor, n: factor ** (1.0 / n))
OmegaConf.register_new_resolver("add", lambda a, b: a + b)
OmegaConf.register_new_resolver("sub", lambda a, b: a - b)
OmegaConf.register_new_resolver("mul", lambda a, b: a * b)
OmegaConf.register_new_resolver("div", lambda a, b: a / b)
OmegaConf.register_new_resolver("divceil", lambda a, b: int(math.ceil(a / b)))
OmegaConf.register_new_resolver("power", lambda a, b: a**b)
OmegaConf.register_new_resolver("sqrt", lambda a: float(math.sqrt(a)))
OmegaConf.register_new_resolver("min", lambda a, b: min(int(a), int(b)))
OmegaConf.register_new_resolver("dirname", lambda p: os.path.dirname(p))
OmegaConf.register_new_resolver("basename", lambda p: os.path.basename(p))
OmegaConf.register_new_resolver("join_path", lambda a, b: os.path.join(a, b))
OmegaConf.register_new_resolver("sum_values", lambda d: sum([v for v in d.values()]))
OmegaConf.register_new_resolver("int", lambda a: int(a))
OmegaConf.register_new_resolver("load_json", lambda path: OmegaConf.create(json.load(open(path))))


OmegaConf.register_new_resolver("and_then", lambda condition, then, or_else: then if condition else or_else)
OmegaConf.register_new_resolver("len", lambda p: len(p))
OmegaConf.register_new_resolver("invalid_key", lambda p, k: k not in p)
# ======================================================= #


@rank_zero_only
def dump_config(path: str, config: BaseConfigSchema) -> None:
    save_config = config.to_dictconfig()

    # Add the hydra specific part that prevents creating new directories automatically
    with open_dict(save_config):
        save_config["hydra"] = {"output_subdir": None, "job": {"chdir": False}, "run": {"dir": "."}}

    # Add the macro that tells hydra that the path to the config should not be included in the dict structure
    with open(path, "w") as fp:
        fp.write("# @package _global_ \n\n")
        OmegaConf.save(config=save_config, f=fp)


def parse_untyped_config(
    config_name: str, hydra_args: list[str] | tuple[str], config_dir: str = "./configs"
) -> DictConfig:
    """
    Main config loading functionality.

    Hydra initialization expects a relative config path wrt to the config directory.
    We manually handle the cases where an absolute path or a relative path to the repo root are given for convenience
    """
    hydra_args_list: list[str] = list(hydra_args)  # convert to list in case we have a tuple

    config_dir_path = Path(config_dir)
    config_name_path = Path(str(config_name))

    # Convert config_dir into an absolute path
    if config_dir_path.is_absolute():
        # Nothing to do,
        config_registry_dir = str(config_dir_path)
    else:
        # This enables defining relative configs from repo root (to enable autocomplete)
        if config_name_path.is_relative_to(config_dir_path):
            config_name_path = config_name_path.relative_to(config_dir_path)

        # Relative config_dirs are relative to repository's root
        config_registry_dir = str(__reporoot__ / config_dir_path)

    del config_dir  # we'll use config_registry_dir from now on

    if config_name_path.is_absolute():
        # Use config_name's directory as config_dir
        config_dir = str(config_name_path.parent)
        # Now config name is simply the file name
        config_name = config_name_path.parts[-1]

        # We still want to add NuRec's config registry to the search path
        hydra_args_list.append(f"hydra.searchpath=[{config_registry_dir}]")
    else:
        # Config name must be relative to config_registry_dir
        config_dir = config_registry_dir
        config_name = str(config_name_path)

    # Using Hydra context below for proper Hydra cleanup, so that subsequent calls to main()
    # e.g. from multiple tests or other scripts work properly.
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = hydra.compose(config_name=str(config_name), overrides=hydra_args_list)

    OmegaConf.resolve(config)

    return config


def assert_no_out_dir_override_in_resume(hydra_args: list[str]) -> None:
    """Checks hydra overrides to prevent changing the out_dir if resume is set and mode=train"""

    resume_true = any([param.startswith("resume=") for param in hydra_args])
    # Don't fail when using resume=auto with wandb sweeps
    resume_auto = any([(param.startswith("resume=") and "auto" in param) for param in hydra_args])
    mode_is_train = any([(param.startswith("mode=") and "train" in param) for param in hydra_args])
    if (
        (resume_true and not resume_auto)
        and mode_is_train
        and any([param.startswith("out_dir=") for param in hydra_args])
    ):
        raise AssertionError("out_dir cannot be changed when resuming training")


