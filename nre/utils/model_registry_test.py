# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Unit tests for the model registry module."""

import tempfile
import unittest

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import requests

from nre.utils.model_registry import (
    GitLabModelRegistry,
    ModelRegistry,
    ModelRegistryError,
    NgcModelRegistry,
    create_model_registry,
)


class TestModelRegistry(unittest.TestCase):
    """Test cases for the base ModelRegistry class."""

    class MockModelRegistry(ModelRegistry):
        """Mock implementation for unit testing base class."""

        @property
        def domain(self) -> str:
            return "mock.example.com"

        @staticmethod
        def api_key_hint() -> str:
            return "Mock API Key (starting with 'mock-')"

        def _validate_url(self, url: str) -> bool:
            # Only accept URLs starting with 'mock://'
            return url.startswith("mock://")

        def _download_to_file(self, filename: str) -> str:
            """Mock implementation that just returns a predefined path"""
            return "mock_download_path"

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()

    def test_init_valid_url(self):
        """Test initialization with a valid URL."""
        registry = self.MockModelRegistry(
            model_url="mock://valid/url", model_cache_dir=self.test_dir, api_key="test_key"
        )
        self.assertEqual(registry.model_url, "mock://valid/url")
        self.assertEqual(registry.model_cache_dir, self.test_dir)

    def test_init_invalid_url(self):
        """Test initialization with an invalid URL."""
        with self.assertRaises(ModelRegistryError) as cm:
            self.MockModelRegistry(model_url="invalid://url", model_cache_dir=self.test_dir, api_key="test_key")
        self.assertIn("Invalid model URL", str(cm.exception))

    def test_verify_cached_file(self):
        """Test the _verify_cached_file method."""
        # Test with valid file
        valid_file = self.test_dir / "valid_file"
        valid_file.write_text("x" * 2048)  # 2KB file, above MIN_FILE_SIZE

        # Test with invalid file
        invalid_file = self.test_dir / "invalid_file"
        invalid_file.write_text("x" * 512)  # 512B file, below MIN_FILE_SIZE

        registry = self.MockModelRegistry(
            model_url="mock://valid/url", model_cache_dir=self.test_dir, api_key="test_key"
        )

        self.assertTrue(registry._verify_cached_file(valid_file))
        self.assertFalse(registry._verify_cached_file(invalid_file))
        self.assertFalse(registry._verify_cached_file(self.test_dir / "nonexistent"))

    def test_verify_downloaded_file(self):
        """Test the _verify_downloaded_file method."""
        # Create a test file
        test_file = self.test_dir / "test_file"
        content = "x" * 2048  # 2KB file
        test_file.write_text(content)

        registry = self.MockModelRegistry(
            model_url="mock://valid/url", model_cache_dir=self.test_dir, api_key="test_key"
        )

        # Test with correct size
        self.assertTrue(registry._verify_downloaded_file(test_file, len(content)))

        # Test with incorrect size
        self.assertFalse(registry._verify_downloaded_file(test_file, len(content) + 100))

        # Test with nonexistent file
        self.assertFalse(registry._verify_downloaded_file(self.test_dir / "nonexistent", 2048))

    def test_get_model_with_cached_file(self):
        """Test get_model with an existing cached file."""
        # Create a mock file in the cache
        cached_file = self.test_dir / "model.pt"
        cached_file.write_text("x" * 2048)  # Large enough to pass verification

        # Create a mock registry that tracks if download was called
        class TrackingRegistry(self.MockModelRegistry):
            download_called = False

            def _download_to_file(self, filename: str) -> str:
                self.download_called = True
                return "mock_download_path"

        registry = TrackingRegistry(
            model_url="mock://valid/model.pt", model_cache_dir=self.test_dir, api_key="test_key"
        )

        # Test with existing valid file
        result = registry.get_model()
        self.assertEqual(result, str(cached_file))  # Should use cached file
        self.assertFalse(registry.download_called)

        # Test with invalid file (too small)
        cached_file.write_text("x" * 512)  # Too small to pass verification

        registry.download_called = False
        result = registry.get_model()
        self.assertEqual(result, "mock_download_path")  # Should download new file
        self.assertTrue(registry.download_called)

    def test_get_model_fresh_download(self):
        """Test get_model with a fresh download."""

        # Setup registry with a mock download method
        class DownloadingRegistry(self.MockModelRegistry):
            def _download_to_file(self, filename: str) -> str:
                return str(self.model_cache_dir / filename)

        registry = DownloadingRegistry(
            model_url="mock://valid/model.pt", model_cache_dir=self.test_dir, api_key="test_key"
        )

        # Test a fresh download
        with (
            patch("pathlib.Path.exists", return_value=False),
            patch("pathlib.Path.mkdir"),
            patch("pathlib.Path.is_dir", return_value=True),
        ):
            result = registry.get_model()  # Should call _download_to_file
            expected_file = self.test_dir / "model.pt"
            self.assertEqual(result, str(expected_file))

    def test_get_credential_from_netrc_success(self):
        """Test successful credential retrieval from netrc."""
        # Mock netrc
        with patch("netrc.netrc") as mock_netrc_class:
            mock_netrc = MagicMock()
            mock_netrc_class.return_value = mock_netrc
            mock_netrc.authenticators.return_value = ("user", None, "test_credential")

            # Test retrieval
            credential = ModelRegistry._get_credential_from_netrc("test.domain.com")
            self.assertEqual(credential, "test_credential")
            mock_netrc.authenticators.assert_called_once_with("test.domain.com")

    def test_get_credential_from_netrc_missing(self):
        """Test credential retrieval when netrc file is missing."""
        # Mock netrc file not found
        with patch("netrc.netrc", side_effect=FileNotFoundError("No .netrc file")):
            with self.assertRaises(ModelRegistryError) as cm:
                ModelRegistry._get_credential_from_netrc("test.domain.com")
            self.assertIn("Failed to read .netrc file", str(cm.exception))

    def test_empty_api_key(self):
        """Test initialization with empty API key."""
        with patch("pathlib.Path.mkdir"), patch("pathlib.Path.exists", return_value=False):
            registry = NgcModelRegistry(
                model_url="https://api.ngc.nvidia.com/v2/models/test/model",
                model_cache_dir=self.test_dir,
                api_key="",
            )
            with self.assertRaises(ModelRegistryError) as cm:
                registry._download_to_file("test_file.pt")
            self.assertIn("API key cannot be empty", str(cm.exception))


class TestNgcModelRegistry(unittest.TestCase):
    """Test cases for the NgcModelRegistry class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

        # Mock requests Session
        self.session_patcher = patch("requests.Session")
        self.mock_session_class = self.session_patcher.start()
        self.mock_session = MagicMock()
        self.mock_session_class.return_value = self.mock_session
        self.addCleanup(self.session_patcher.stop)

        # Mock netrc
        self.netrc_patcher = patch("netrc.netrc")
        self.mock_netrc_class = self.netrc_patcher.start()
        self.mock_netrc = MagicMock()
        self.mock_netrc_class.return_value = self.mock_netrc
        self.addCleanup(self.netrc_patcher.stop)

        self.registry = NgcModelRegistry(
            model_url="https://api.ngc.nvidia.com/v2/models/test/model",
            model_cache_dir=self.test_dir,
            api_key="nvapi-test_api_key",  # non-legacy API key
        )

        self.registry_legacy = NgcModelRegistry(
            model_url="https://api.ngc.nvidia.com/v2/models/test/model",
            model_cache_dir=self.test_dir,
            api_key="test_api_key",  # legacy API key
        )

    def tearDown(self):
        """Clean up test artifacts."""
        self.temp_dir.cleanup()

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        self.assertEqual(self.registry.model_url, "https://api.ngc.nvidia.com/v2/models/test/model")
        self.assertEqual(self.registry.api_key, "nvapi-test_api_key")
        # Session should not be initialized yet
        self.assertIsNone(self.registry.session)
        # Verify session headers are not set yet
        self.mock_session.headers.update.assert_not_called()

    def test_lazy_session_initialization(self):
        """Test that session is lazily initialized during _download_to_file."""
        # Create a new registry without an API key
        registry = NgcModelRegistry(
            model_url="https://api.ngc.nvidia.com/v2/models/test/model",
            model_cache_dir=self.test_dir,
            api_key=None,
        )

        # Mock the API key retrieval
        with patch.object(registry, "_get_api_key", return_value="nvapi-test_api_key"):
            # Mock the download process
            with patch("requests.Session") as mock_session:
                mock_response = MagicMock()
                mock_response.headers = {"content-length": "100"}
                mock_response.iter_content.return_value = [b"x" * 100]
                mock_response.raise_for_status.return_value = None
                mock_session.return_value.get.return_value = mock_response
                with (
                    patch("builtins.open", mock_open()),
                    patch.object(registry, "_verify_downloaded_file", return_value=True),
                ):
                    # Call _download_to_file to trigger lazy initialization
                    registry._download_to_file("test_file.pt")
                    # Verify session was created with correct headers
                    mock_session.return_value.headers.update.assert_called_once_with(
                        {
                            "Accept": "application/json",
                            "Content-Type": "application/json",
                        }
                    )
                    self.assertEqual(mock_session.return_value.auth.token, "nvapi-test_api_key")

    def test_lazy_api_key_initialization(self):
        """Test that API key is lazily initialized during _download_to_file."""
        # Create a new registry without an API key
        registry = NgcModelRegistry(
            model_url="https://api.ngc.nvidia.com/v2/models/test/model",
            model_cache_dir=self.test_dir,
            api_key=None,
        )

        # Mock the API key retrieval
        with patch.object(registry, "_get_api_key", return_value="nvapi-test_api_key"):
            # Mock the download process
            with patch("requests.Session") as mock_session:
                mock_response = MagicMock()
                mock_response.headers = {"content-length": "100"}
                mock_response.iter_content.return_value = [b"x" * 100]
                mock_response.raise_for_status.return_value = None
                mock_session.return_value.get.return_value = mock_response
                with (
                    patch("builtins.open", mock_open()),
                    patch.object(registry, "_verify_downloaded_file", return_value=True),
                ):
                    # Call _download_to_file to trigger lazy initialization
                    registry._download_to_file("test_file.pt")
                    # Verify API key was set
                    self.assertEqual(registry.api_key, "nvapi-test_api_key")

    def test_empty_api_key(self):
        """Test initialization with empty API key."""
        registry = NgcModelRegistry(
            model_url="https://api.ngc.nvidia.com/v2/models/test/model",
            model_cache_dir=self.test_dir,
            api_key="",
        )
        with self.assertRaises(ModelRegistryError) as cm:
            registry._download_to_file("test_file.pt")
        self.assertIn("API key cannot be empty", str(cm.exception))

    def test_validate_url_valid(self):
        """Test URL validation with valid NGC URL."""
        self.assertTrue(self.registry._validate_url("https://api.ngc.nvidia.com/v2/models/test/model"))

    def test_validate_url_invalid(self):
        """Test URL validation with invalid URL."""
        self.assertFalse(self.registry._validate_url("https://invalid-url.com"))

    def test_domain_property(self):
        """Test the domain property returns the correct value."""
        self.assertEqual(self.registry.domain, "api.ngc.nvidia.com")

    def test_get_api_key_from_env_var(self):
        """Test API key retrieval from NGC_API_KEY environment variable."""

        # Mock os.environ.get to return a test API key only for NGC_API_KEY
        def mock_get_env(key, default=None):
            if key == "NGC_API_KEY":
                return "nvapi-env_api_key"
            return default

        # Create a new registry without an API key
        with patch("os.environ.get", side_effect=mock_get_env):
            # Also mock netrc to ensure it's not used
            with patch.object(ModelRegistry, "_get_credential_from_netrc", return_value=None):
                registry = NgcModelRegistry(
                    model_url="https://api.ngc.nvidia.com/v2/models/test/model",
                    model_cache_dir=self.test_dir,
                    api_key=None,
                )

                self.assertEqual(registry._get_api_key(), "nvapi-env_api_key")

    def test_api_key_precedence(self):
        """Test that API key precedence is correctly followed: direct param > env var > netrc."""

        # Set up mocks for all three methods
        def mock_get_env(key, default=None):
            if key == "NGC_API_KEY":
                return "nvapi-env_api_key"
            return default

        # Test direct param takes precedence
        with patch("os.environ.get", side_effect=mock_get_env):
            with patch.object(ModelRegistry, "_get_credential_from_netrc", return_value="nvapi-netrc_api_key"):
                registry = NgcModelRegistry(
                    model_url="https://api.ngc.nvidia.com/v2/models/test/model",
                    model_cache_dir=self.test_dir,
                    api_key="nvapi-direct_api_key",
                )
                # Force API key initialization by calling _download_to_file
                with patch("requests.Session") as mock_session:
                    mock_response = MagicMock()
                    mock_response.headers = {"content-length": "100"}
                    mock_response.iter_content.return_value = [b"x" * 100]
                    mock_response.raise_for_status.return_value = None
                    mock_session.return_value.get.return_value = mock_response
                    with (
                        patch("builtins.open", mock_open()),
                        patch.object(registry, "_verify_downloaded_file", return_value=True),
                    ):
                        registry._download_to_file("test_file.pt")
                        self.assertEqual(registry.api_key, "nvapi-direct_api_key")

        # Test env var takes precedence over netrc
        with patch("os.environ.get", side_effect=mock_get_env):
            with patch.object(ModelRegistry, "_get_credential_from_netrc", return_value="nvapi-netrc_api_key"):
                registry = NgcModelRegistry(
                    model_url="https://api.ngc.nvidia.com/v2/models/test/model",
                    model_cache_dir=self.test_dir,
                    api_key=None,
                )
                # Force API key initialization by calling _download_to_file
                with patch("requests.Session") as mock_session:
                    mock_response = MagicMock()
                    mock_response.headers = {"content-length": "100"}
                    mock_response.iter_content.return_value = [b"x" * 100]
                    mock_response.raise_for_status.return_value = None
                    mock_session.return_value.get.return_value = mock_response
                    with (
                        patch("builtins.open", mock_open()),
                        patch.object(registry, "_verify_downloaded_file", return_value=True),
                    ):
                        registry._download_to_file("test_file.pt")
                        self.assertEqual(registry.api_key, "nvapi-env_api_key")

    def test_extract_org_team_from_model_url_valid(self):
        """Test extraction of org and team from model URL."""
        registry = NgcModelRegistry(
            model_url="https://api.ngc.nvidia.com/v2/org/my_org/team/my_team/models/test/model",
            model_cache_dir=self.test_dir,
            api_key=None,
        )
        self.assertEqual(registry._extract_org_team_from_model_url(), ("my_org", "my_team"))

    def test_extract_org_team_from_model_url_invalid(self):
        """Test extraction of org and team from model URL."""
        registry = NgcModelRegistry(
            model_url="https://api.ngc.nvidia.com/v2/models/test/model",
            model_cache_dir=self.test_dir,
            api_key=None,
        )
        with self.assertRaises(ModelRegistryError) as cm:
            registry._extract_org_team_from_model_url()
        self.assertIn(
            "Invalid NGC URL format, cannot extract org/team: https://api.ngc.nvidia.com/v2/models/test/model",
            str(cm.exception),
        )

    def test_request_session_token_from_legacy_api_key_success(self):
        """Test successful session token request from legacy API key."""
        registry = NgcModelRegistry(
            model_url="https://api.ngc.nvidia.com/v2/org/my_org/team/my_team/models/test/model",
            model_cache_dir=self.test_dir,
            api_key=None,
        )

        # Mock the requests.get call
        with patch("requests.get") as mock_get:
            # Setup mock response
            mock_response = MagicMock()
            mock_response.json.return_value = {"token": "my_session_token"}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            # Test the method
            result = registry._request_session_token_from_legacy_api_key("legacy_api_key", "my_org", "my_team")

            # Verify the result
            self.assertEqual(result, "my_session_token")

            # Verify the request was made correctly
            mock_get.assert_called_once_with(
                "https://authn.nvidia.com/token?service=ngc&scope=group/ngc:my_org&group/ngc:my_org/my_team",
                auth=("$oauthtoken", "legacy_api_key"),
                headers={"Accept": "application/json"},
                timeout=30,
            )

    def test_request_session_token_from_legacy_api_key_no_token_in_response(self):
        """Test session token request when response doesn't contain token."""
        registry = NgcModelRegistry(
            model_url="https://api.ngc.nvidia.com/v2/org/my_org/team/my_team/models/test/model",
            model_cache_dir=self.test_dir,
            api_key=None,
        )

        # Mock the requests.get call with response missing token
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {"error": "invalid_request"}  # No "token" field
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            # Test that it raises ModelRegistryError
            with self.assertRaises(ModelRegistryError) as cm:
                registry._request_session_token_from_legacy_api_key("legacy_api_key", "my_org", "my_team")

            self.assertIn("No token found in authentication response", str(cm.exception))

    def test_request_session_token_from_legacy_api_key_request_failure(self):
        """Test session token request when HTTP request fails."""
        registry = NgcModelRegistry(
            model_url="https://api.ngc.nvidia.com/v2/org/my_org/team/my_team/models/test/model",
            model_cache_dir=self.test_dir,
            api_key=None,
        )

        # Mock the requests.get call to raise an exception
        with patch("requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.RequestException("Connection error")

            # Test that it raises ModelRegistryError
            with self.assertRaises(ModelRegistryError) as cm:
                registry._request_session_token_from_legacy_api_key("legacy_api_key", "my_org", "my_team")

            self.assertIn("Failed to exchange legacy API key for session token", str(cm.exception))

    def test_request_session_token_from_legacy_api_key_invalid_json(self):
        """Test session token request when response has invalid JSON."""
        registry = NgcModelRegistry(
            model_url="https://api.ngc.nvidia.com/v2/org/my_org/team/my_team/models/test/model",
            model_cache_dir=self.test_dir,
            api_key=None,
        )

        # Mock the requests.get call with invalid JSON response
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            # Test that it raises ModelRegistryError
            with self.assertRaises(ModelRegistryError) as cm:
                registry._request_session_token_from_legacy_api_key("legacy_api_key", "my_org", "my_team")

            self.assertIn("Invalid response format from NGC authentication service", str(cm.exception))

    def test_request_session_token_from_api_key_legacy_token(self):
        """Test _request_session_token_from_api_key with legacy API key (non-PAT)."""
        registry = NgcModelRegistry(
            model_url="https://api.ngc.nvidia.com/v2/org/my_org/team/my_team/models/test/model",
            model_cache_dir=self.test_dir,
            api_key=None,
        )

        # Mock the requests.get call that happens in _request_session_token_from_legacy_api_key
        with patch("requests.get") as mock_get:
            # Setup mock response for the token exchange
            mock_response = MagicMock()
            mock_response.json.return_value = {"token": "exchanged_session_token"}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            # Test with legacy API key (doesn't start with "nvapi-")
            result = registry._request_session_token_from_api_key("legacy_api_key")

            # Verify the HTTP request was made correctly
            mock_get.assert_called_once_with(
                "https://authn.nvidia.com/token?service=ngc&scope=group/ngc:my_org&group/ngc:my_org/my_team",
                auth=("$oauthtoken", "legacy_api_key"),
                headers={"Accept": "application/json"},
                timeout=30,
            )

            # Verify the result
            self.assertEqual(result, "exchanged_session_token")


class TestGitLabModelRegistry(unittest.TestCase):
    """Test cases for the GitLabModelRegistry class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

        # Mock requests Session
        self.session_patcher = patch("requests.Session")
        self.mock_session_class = self.session_patcher.start()
        self.mock_session = MagicMock()
        self.mock_session_class.return_value = self.mock_session
        self.addCleanup(self.session_patcher.stop)

        # Mock netrc
        self.netrc_patcher = patch("netrc.netrc")
        self.mock_netrc_class = self.netrc_patcher.start()
        self.mock_netrc = MagicMock()
        self.mock_netrc_class.return_value = self.mock_netrc
        self.addCleanup(self.netrc_patcher.stop)

        self.registry = GitLabModelRegistry(
            model_url="https://gitlab-master.nvidia.com/projects/models/test/model.tar.gz",
            model_cache_dir=self.test_dir,
            api_key="test_api_key",
        )

    def tearDown(self):
        """Clean up test artifacts."""
        self.temp_dir.cleanup()

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        self.assertEqual(self.registry.model_url, "https://gitlab-master.nvidia.com/projects/models/test/model.tar.gz")
        self.assertEqual(self.registry.api_key, "test_api_key")
        # Session should not be initialized yet
        self.assertIsNone(self.registry.session)
        # Verify session headers are not set yet
        self.mock_session.headers.update.assert_not_called()

    def test_lazy_session_initialization(self):
        """Test that session is lazily initialized during _download_to_file."""
        # Create a new registry without an API key
        registry = GitLabModelRegistry(
            model_url="https://gitlab-master.nvidia.com/projects/models/test/model.tar.gz",
            model_cache_dir=self.test_dir,
            api_key=None,
        )

        # Mock the API key retrieval
        with patch.object(registry, "_get_api_key", return_value="test_api_key"):
            # Mock the download process
            with patch("requests.Session") as mock_session:
                mock_response = MagicMock()
                mock_response.headers = {"content-length": "100"}
                mock_response.iter_content.return_value = [b"x" * 100]
                mock_response.raise_for_status.return_value = None
                mock_session.return_value.get.return_value = mock_response
                with (
                    patch("builtins.open", mock_open()),
                    patch.object(registry, "_verify_downloaded_file", return_value=True),
                ):
                    # Call _download_to_file to trigger lazy initialization
                    registry._download_to_file("test_file.pt")
                    # Verify session was created with correct headers
                    mock_session.return_value.headers.update.assert_called_once_with(
                        {
                            "Accept": "application/json",
                            "Content-Type": "application/json",
                        }
                    )
                    self.assertEqual(mock_session.return_value.auth.token, "test_api_key")

    def test_lazy_api_key_initialization(self):
        """Test that API key is lazily initialized during _download_to_file."""
        # Create a new registry without an API key
        registry = GitLabModelRegistry(
            model_url="https://gitlab-master.nvidia.com/projects/models/test/model.tar.gz",
            model_cache_dir=self.test_dir,
            api_key=None,
        )

        # Mock the API key retrieval
        with patch.object(registry, "_get_api_key", return_value="test_api_key"):
            # Mock the download process
            with patch("requests.Session") as mock_session:
                mock_response = MagicMock()
                mock_response.headers = {"content-length": "100"}
                mock_response.iter_content.return_value = [b"x" * 100]
                mock_response.raise_for_status.return_value = None
                mock_session.return_value.get.return_value = mock_response
                with (
                    patch("builtins.open", mock_open()),
                    patch.object(registry, "_verify_downloaded_file", return_value=True),
                ):
                    # Call _download_to_file to trigger lazy initialization
                    registry._download_to_file("test_file.pt")
                    # Verify API key was set
                    self.assertEqual(registry.api_key, "test_api_key")

    def test_empty_api_key(self):
        """Test initialization with empty API key."""
        registry = GitLabModelRegistry(
            model_url="https://gitlab-master.nvidia.com/projects/models/test/model.tar.gz",
            model_cache_dir=self.test_dir,
            api_key="",
        )
        with self.assertRaises(ModelRegistryError) as cm:
            registry._download_to_file("test_file.pt")
        self.assertIn("API key cannot be empty", str(cm.exception))

    def test_validate_url_valid(self):
        """Test URL validation with valid GitLab URL."""
        self.assertTrue(
            self.registry._validate_url("https://gitlab-master.nvidia.com/projects/models/test/model.tar.gz")
        )

    def test_validate_url_invalid(self):
        """Test URL validation with invalid URL."""
        self.assertFalse(self.registry._validate_url("https://invalid-url.com"))

    def test_domain_property(self):
        """Test the domain property returns the correct value."""
        self.assertEqual(self.registry.domain, "gitlab-master.nvidia.com")


class TestModelRegistryFactory(unittest.TestCase):
    """Test cases for the model registry factory function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()

    def test_create_ngc_registry(self):
        """Test creation of NgcModelRegistry from factory."""
        registry = create_model_registry(
            model_url="https://api.ngc.nvidia.com/v2/models/test/model",
            model_cache_dir=self.test_dir,
            api_key="test_api_key",
        )
        self.assertIsInstance(registry, NgcModelRegistry)
        self.assertEqual(registry.model_url, "https://api.ngc.nvidia.com/v2/models/test/model")
        self.assertEqual(registry.api_key, "test_api_key")

    def test_create_gitlab_registry(self):
        """Test creation of GitLabModelRegistry from factory."""
        registry = create_model_registry(
            model_url="https://gitlab-master.nvidia.com/projects/models/test/model.tar.gz",
            model_cache_dir=self.test_dir,
            api_key="test_api_key",
        )
        self.assertIsInstance(registry, GitLabModelRegistry)
        self.assertEqual(registry.model_url, "https://gitlab-master.nvidia.com/projects/models/test/model.tar.gz")
        self.assertEqual(registry.api_key, "test_api_key")

    def test_unsupported_domain(self):
        """Test factory with unsupported domain."""
        with self.assertRaises(ModelRegistryError) as cm:
            create_model_registry(
                model_url="https://unsupported-domain.com/model",
                model_cache_dir=self.test_dir,
                api_key="test_api_key",
            )
        self.assertIn("Unsupported model registry domain", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
