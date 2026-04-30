# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import unittest

from pathlib import Path

from python.runfiles import runfiles

from nre.config.nre import NREConfig
from nre.config.parse import parse_untyped_config
from nre.utils.upgrade import upgrade_config


# Required to see log.info() messages from invoked upgrade functions from this test.
logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s", level=logging.INFO, force=True)

RUNFILES = runfiles.Create()


class TestConfigBackwardCompatibility(unittest.TestCase):
    """Tests that a config from 25.05 is consistent with the current config"""

    def setUp(self):
        self.path = Path(RUNFILES.Rlocation("test_config_25_05/parsed.yaml"))
        if not self.path.exists():
            raise AssertionError(
                f"Test config not found. This is an issue with your filesystem/test suite, not the code under test. Missing {self.path=}"
            )

    def test_config_25_05_can_be_parsed(self):
        # Load YAML as config
        config = parse_untyped_config(config_name=str(self.path), hydra_args=[])
        self.assertIn("version", config)
        self.assertIn("version_string", config.version)
        self.assertEqual(config.version.version_string, "0.2.573-f0508837")

    def test_config_25_05_can_be_upgraded(self):
        # Try to load the config by upgrading it to the current version
        # We only care that parse_typed_config executes without failing

        old_config = parse_untyped_config(config_name=str(self.path), hydra_args=[])
        try:
            upgraded_config = upgrade_config(old_config)
            upgraded_config = NREConfig.model_validate(upgraded_config)

            assert upgraded_config.version is not None
            self.assertNotEqual(old_config.version.version_string, upgraded_config.version.version_string)
        except Exception as e:
            self.fail(f"Failed to upgrade config 25.05: {e}")
