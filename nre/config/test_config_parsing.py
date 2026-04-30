# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import tempfile
import unittest

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from nre.config.parse import parse_untyped_config


def write_yaml_config(config_data: dict, file_path: str | Path) -> None:
    """
    Write a nested dictionary as a YAML config file.

    Args:
        config_data: The configuration data as a nested dictionary
        file_path: Path where the YAML file should be written
    """

    file_path = Path(file_path)

    # Convert dict to OmegaConf for proper YAML serialization
    omega_config = OmegaConf.create(config_data)

    # Make sure the subdirectories exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the config with Hydra global package annotation
    with open(file_path, "w") as fp:
        fp.write("# @package _global_\n\n")
        OmegaConf.save(config=omega_config, f=fp)


class TestConfigParsing(unittest.TestCase):
    """Tests for the absolute config path fix that handles nested path hierarchies."""

    def setUp(self):
        """Set up test fixtures with temporary config files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create a test config content
        self.test_config_content = {
            "test_key": "test_value",
            "nested": {"config": "content"},
            "system": {"mode": "test"},
        }

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_absolute_config_path_extraction(self):
        """Test that absolute config paths properly extract the leaf config node."""
        # Create a test config file with actual YAML content using utility function
        config_file = self.temp_path / "test_config.yaml"
        write_yaml_config(self.test_config_content, str(config_file))

        # Assert that we're testing with an absolute path
        self.assertTrue(config_file.is_absolute(), f"config_file must be absolute, got: {config_file}")

        # Test the actual function behavior
        config = parse_untyped_config(config_name=str(config_file), hydra_args=[])

        # Verify that the config contains the actual content, not the nested structure
        self.assertEqual(config.test_key, "test_value")
        self.assertEqual(config.nested.config, "content")
        self.assertEqual(config.system.mode, "test")

    def test_deeply_nested_absolute_path(self):
        """Test extraction from a deeply nested absolute path."""
        # Create a deep directory structure
        deep_path = self.temp_path / "a" / "very" / "deep" / "nested" / "path"
        config_file = deep_path / "deep_config.yaml"

        # Create test config data and write it using the utility function
        config_data = {"deep": "config", "value": 42}
        write_yaml_config(config_data, str(config_file))

        # Assert that we're testing with an absolute path
        self.assertTrue(config_file.is_absolute(), f"config_file must be absolute, got: {config_file}")

        # Test the actual function behavior
        config = parse_untyped_config(config_name=str(config_file), hydra_args=[])

        # Verify extraction works for deep nesting
        self.assertEqual(config.deep, "config")
        self.assertEqual(config.value, 42)

    def test_absolute_path_with_hydra_args(self):
        """Test that absolute paths work correctly with hydra arguments."""
        config_file = self.temp_path / "config_with_args.yaml"

        # Create a config with values that can be overridden
        config_data = {
            "param": "default_value",
            "param2": "not_to_be_changed",
            "other": "static_value",
            "nested": {"override_me": "original", "keep_me": "unchanged"},
        }
        write_yaml_config(config_data, str(config_file))

        # Assert that we're testing with an absolute path
        self.assertTrue(config_file.is_absolute(), f"config_file must be absolute, got: {config_file}")

        # Test with Hydra override arguments - this tests real functionality
        hydra_args = ["param=overridden_value", "nested.override_me=new_value"]
        config = parse_untyped_config(config_name=str(config_file), hydra_args=hydra_args)

        # Verify that Hydra overrides work correctly with absolute paths
        self.assertEqual(config.param, "overridden_value")  # Should be overridden
        self.assertEqual(config.param2, "not_to_be_changed")  # Should remain unchanged
        self.assertEqual(config.other, "static_value")  # Should remain unchanged
        self.assertEqual(config.nested.override_me, "new_value")  # Nested override
        self.assertEqual(config.nested.keep_me, "unchanged")  # Should remain unchanged

    def test_relative_config_path(self):
        """Test parsing a config with a relative path using config_dir parameter."""
        # Create a subdirectory in temp_path for the config file
        config_subdir = self.temp_path / "relative_configs"

        # Create a test config file in the subdirectory
        config_file = config_subdir / "relative_config.yaml"
        config_data = {"relative_test": True, "config_dir_test": "success", "nested": {"data": "from_relative_config"}}
        write_yaml_config(config_data, str(config_file))

        # Test with relative config name and config_dir pointing to the subdirectory
        config = parse_untyped_config(config_name="relative_config.yaml", hydra_args=[], config_dir=str(config_subdir))

        # Verify that the config was loaded correctly
        self.assertEqual(config.relative_test, True)
        self.assertEqual(config.config_dir_test, "success")
        self.assertEqual(config.nested.data, "from_relative_config")

    def test_nested_relative_config(self):
        """Test that hydra overrides work correctly with relative config paths."""
        # Create a config directory
        config_dir = self.temp_path / "relative_configs" / "nested"

        # Create a test config file with values that can be overridden
        config_file = config_dir / "nested_config.yaml"
        config_data = {
            "param": "original_value",
            "param2": "not_to_be_changed",
            "number": 42,
            "override_me": "default",
            "nested": {"value": "original_nested", "keep": "unchanged", "number": 100},
            "list_param": ["item1", "item2"],
        }
        write_yaml_config(config_data, str(config_file))

        # Test with hydra override arguments using relative config path
        hydra_args = [
            "param=overridden_value",
            "number=999",
            "override_me=new_value",
            "nested.value=new_nested_value",
            "nested.number=200",
            "list_param=[new_item1,new_item2,new_item3]",
        ]

        config = parse_untyped_config(
            config_name="nested/nested_config.yaml", hydra_args=hydra_args, config_dir=str(config_dir.parent)
        )

        # Verify that all overrides work correctly with relative configs
        self.assertEqual(config.param, "overridden_value")
        self.assertEqual(config.param2, "not_to_be_changed")
        self.assertEqual(config.number, 999)
        self.assertEqual(config.override_me, "new_value")
        self.assertEqual(config.nested.value, "new_nested_value")
        self.assertEqual(config.nested.number, 200)
        self.assertEqual(config.nested.keep, "unchanged")  # Should remain unchanged
        self.assertEqual(config.list_param, ["new_item1", "new_item2", "new_item3"])

    def test_default_config_dir(self):
        """Test loading a config from default config_dir and overriding seed parameter."""
        # Test with the real ncore_ds.yaml config from the configs directory

        config = parse_untyped_config(
            config_name="tests/ncore_ds.yaml",
            hydra_args=["seed=123"],
        )

        # Verify that the seed was overridden correctly
        self.assertEqual(config.seed, 123)

        # Also verify that other base config values are loaded correctly
        self.assertEqual(config.mode, "trainval")  # from base.yaml
        self.assertEqual(config.out_dir, ".")  # from ncore_ds.yaml


if __name__ == "__main__":
    unittest.main()
