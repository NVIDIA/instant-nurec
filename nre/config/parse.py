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

from omegaconf import OmegaConf, open_dict

from nre.config.base_schema import BaseConfigSchema
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


