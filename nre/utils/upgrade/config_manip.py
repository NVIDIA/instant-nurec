# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Any

from omegaconf import DictConfig, OmegaConf


def exists_config_key(cfg: DictConfig, dotted_key: str) -> bool:
    """Checks if a config key exists anywhere in the config tree"""
    # (OmegaConf.select(cfg, dotted_key) is not None) confuses a missing key and a key with a None value unfortunately.
    # Walking down the tree instead.
    keys = dotted_key.split(".")
    node = cfg
    for key in keys[:-1]:  # Walk down until the parent of the final key
        if key not in node:
            return False
        node = node[key]
    return keys[-1] in node


def insert_config_key(cfg: DictConfig, dotted_key: str, value: Any) -> None:
    """Inserts a config key anywhere in the config tree in place while also creating missing parent nodes

    If the key already existed, it will be overwritten with the provided value.
    """
    keys = dotted_key.split(".")
    node = cfg
    for key in keys[:-1]:  # Walk down until the parent of the final key
        if key not in node or node[key] is None:
            node[key] = {}  # Create missing dicts
        node = node[key]
    node[keys[-1]] = value


def remove_config_key(cfg: DictConfig, dotted_key: str) -> None:
    """Removes a config key anywhere in the config tree in place while leaving the parent nodes intact

    If the key is not found, the function does nothing.
    """

    if not exists_config_key(cfg, dotted_key):
        return

    keys = dotted_key.split(".")
    node = cfg
    for key in keys[:-1]:  # Walk down until the parent of the final key
        node = node[key]
    del node[keys[-1]]


def copy_config_key(cfg: DictConfig, source_dotted_key: str, target_dotted_key: str, default_value: Any = None) -> None:
    """Copies a config key from anywhere to anywhere else in the config tree in place while creating missing
    target path nodes.

    If any node on the path to the target key does not exist, it will be automatically created as a dict.
    If the source key is not found, the target key will be populated with the default value.
    """
    assert source_dotted_key != target_dotted_key, "Source and target keys cannot be the same"
    value = OmegaConf.select(cfg, source_dotted_key, default=default_value)
    insert_config_key(cfg, target_dotted_key, value)


def move_config_key(cfg: DictConfig, source_dotted_key: str, target_dotted_key: str, default_value: Any = None) -> None:
    """Moves a config key from anywhere to anywhere else in the config tree in place while creating missing
    target path nodes.

    If any node on the path to the target key does not exist, it will be automatically created as a dict.
    If the source key is not found, the target key will be populated with the default value.
    """
    assert source_dotted_key != target_dotted_key, "Source and target keys cannot be the same"
    copy_config_key(cfg, source_dotted_key, target_dotted_key, default_value)
    remove_config_key(cfg, source_dotted_key)
