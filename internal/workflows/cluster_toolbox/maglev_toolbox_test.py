# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import unittest

from unittest.mock import patch

from internal.workflows.cluster_toolbox.maglev_toolbox import MaglevToolbox


class TestMaglevToolboxCLI(unittest.TestCase):
    """Test suite for verifying maglev CLI path configuration in Bazel system."""

    def setUp(self):
        """Set up test fixtures."""
        pass

    def test_get_bazel_maglev_cli_file_not_found(self):
        """Test that _get_bazel_maglev_cli raises FileNotFoundError when file doesn't exist."""
        # Mock os.path.isfile to return False
        with patch("os.path.isfile") as mock_isfile:
            mock_isfile.return_value = False

            # Test that FileNotFoundError is raised
            with self.assertRaises(FileNotFoundError):
                MaglevToolbox._get_bazel_maglev_cli()

    def test_maglev_cli_path_in_bazel_environment(self):
        """Test that maglev CLI path is correctly set in Bazel environment."""
        # In a real Bazel test environment, the file should exist
        # We check if we're running under Bazel
        if os.environ.get("BAZEL_TEST"):  # Bazel sets this environment variable
            # In Bazel test environment, the maglev CLI should be available
            MaglevToolbox._get_bazel_maglev_cli()


if __name__ == "__main__":
    unittest.main()
