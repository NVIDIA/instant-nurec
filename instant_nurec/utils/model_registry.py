# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for downloading models from various model registries."""

import logging
import netrc
import os
import re

from abc import ABC, abstractmethod
from pathlib import Path
from typing import NoReturn, Optional, Type, TypeVar
from urllib.parse import urlparse

import requests

from requests.auth import AuthBase
from tqdm import tqdm

from instant_nurec.utils.misc import unpack_optional


logger = logging.getLogger(__name__)

ExceptionType = TypeVar("ExceptionType", bound=Exception)


def partial_api_key(api_key: str, clear_view_length: int) -> str:
    """Helper function to partially mask an API key for logging.

    Args:
        api_key: The API key to mask.
        clear_view_length: Length of the API key to show unmasked.

    WARNING: if the API key is shorter than or equal to clear_view_length,
             the full API key will be returned unmasked.

    Returns:
        str: The partially masked API key.
    """

    if len(api_key) <= clear_view_length:
        return api_key

    return api_key[:clear_view_length] + "*" * (len(api_key) - clear_view_length)


def log_and_raise(exception: Type[ExceptionType], message: str, *args, **kwargs) -> NoReturn:
    """Helper function to log an error and raise an exception.

    Args:
        exception: Exception class to raise
        message: Error message to log and include in exception
        *args: Format args for the message
        **kwargs: Additional kwargs for exception

    Raises:
        ExceptionType: The specified exception type
    """
    formatted_message = message % args if args else message
    logger.error(formatted_message)
    raise exception(formatted_message, **kwargs)


class ModelRegistryError(Exception):
    """Base exception for model registry errors."""

    pass


class ModelRegistry(ABC):
    """Abstract base class for model registry implementations.

    This class defines the interface for downloading models from different model
    registries. Each specific registry implementation must inherit from this class
    and implement the download method.

    Attributes:
        model_url (str): URL where the model can be downloaded from.
        model_cache_dir (Path): Directory where downloaded models should be cached.
        api_key (str): API key for authenticating with the registry. If not provided,
            it will be retrieved from the .netrc file.
    """

    MIN_FILE_SIZE = 1024  # 1KB minimum file size
    CHUNK_SIZE = 8192  # 8KB chunks for downloading

    def __init__(self, model_url: str, model_cache_dir: Path, api_key: Optional[str] = None) -> None:
        """Initialize the model registry.

        Args:
            model_url: URL where the model can be downloaded from.
            model_cache_dir: Directory where downloaded models should be cached.

        Raises:
            ModelRegistryError: If the model URL is invalid.
        """
        logger.debug("Initializing model registry with URL: %s and cache dir: %s", model_url, model_cache_dir)
        if not self._validate_url(model_url):
            log_and_raise(ModelRegistryError, "Invalid model URL: %s", model_url)
        self.model_url = model_url
        self.model_cache_dir = model_cache_dir
        logger.debug("Model registry initialized successfully")
        self.api_key = api_key
        self.session: Optional[requests.Session] = None
        super().__init__()

    def _get_api_key(self, api_key: Optional[str] = None) -> Optional[str]:
        """Get the API key for the model registry.

        The API key is resolved in the following order of precedence:
        1. Directly provided api_key parameter
        2. .netrc file credentials for the registry domain

        Args:
            api_key: Optional API key provided directly.

        Returns:
            Optional[str]: The API key for the model registry.

        Raises:
            ModelRegistryError: If no API key could be found or if an empty API key was provided.
        """
        # Validate API key if provided
        if api_key is not None and not api_key.strip():
            log_and_raise(ModelRegistryError, "API key cannot be empty")

        # Get API key from netrc if not provided
        if not api_key:
            logger.debug("No API key provided, attempting to get it from the .netrc file.")
            try:
                api_key = ModelRegistry._get_credential_from_netrc(self.domain)
                logger.debug("Successfully retrieved API key from the .netrc file.")
            except Exception as e:
                log_and_raise(ModelRegistryError, "Failed to get API key from .netrc file: %s", str(e))

        if not api_key:
            log_and_raise(ModelRegistryError, "API key must be provided either directly or via .netrc file")

        return api_key

    def _get_session(self, session_token: str) -> requests.Session:
        """Get a session for the model registry.

        Returns:
            requests.Session: A session for the model registry.
        """

        session = requests.Session()
        session.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

        class BearerAuth(AuthBase):
            """Custom authentication handler for OAuth2 Bearer tokens.

            Args:
                token: The OAuth2 bearer token to use for authentication
            """

            def __init__(self, token):
                if not token or not token.strip():
                    raise ValueError("Token cannot be empty")

                self.token = token

            def __call__(self, r):
                r.headers["Authorization"] = f"Bearer {self.token}"
                return r

        session.auth = BearerAuth(session_token)

        logger.debug("Session initialized with authentication headers")

        return session

    def _request_session_token_from_api_key(self, api_key: str) -> str:
        """The session token is the API key in most cases, e.g. for NGCModelRegistry this could be different if a legacy access token is used.

        Args:
            api_key: The API key to use for authentication.

        Returns:
            str: A session token for the model registry.
        """
        if not api_key:
            log_and_raise(ModelRegistryError, "API key cannot be empty")

        return api_key

    @property
    @abstractmethod
    def domain(self) -> str:
        """Get the domain of the model registry.

        Returns:
            str: The domain of the model registry.
        """
        pass

    @staticmethod
    @abstractmethod
    def api_key_hint() -> str:
        """Get backend-dependent hint for the API source / key format.

        Returns:
            str: A hint for the API source / key format.
        """
        pass

    def _validate_url(self, url: str) -> bool:
        """Validate if the URL is valid for this registry.

        Args:
            url: The URL to validate.

        Returns:
            bool: True if the URL is valid for this registry.
        """
        logger.debug("Validating URL: %s", url)
        try:
            parsed = urlparse(url)
            is_valid = parsed.netloc == self.domain
            if not is_valid:
                logger.warning("Invalid URL domain: %s (expected %s)", parsed.netloc, self.domain)
            return is_valid
        except Exception as e:
            logger.warning("Failed to parse URL: %s - %s", url, str(e))
            return False

    def _verify_cached_file(self, file_path: Path) -> bool:
        """Verify if a cached file meets minimum size requirements.

        Args:
            file_path: Path to the cached file.

        Returns:
            bool: True if the file meets minimum size requirements.
        """
        logger.debug("Verifying cached file: %s", file_path)

        try:
            actual_size = file_path.stat().st_size
        except OSError as e:
            logger.warning("Failed to get file size: %s", str(e))
            return False

        if actual_size < self.MIN_FILE_SIZE:
            logger.warning(
                "Cached file size (%d bytes) is below minimum required size (%d bytes)", actual_size, self.MIN_FILE_SIZE
            )
            return False

        return True

    def _verify_downloaded_file(self, file_path: Path, expected_size: int) -> bool:
        """Verify if a newly downloaded file matches the expected size.

        Args:
            file_path: Path to the downloaded file.
            expected_size: Expected file size from Content-Length header.

        Returns:
            bool: True if the file size matches expected size.
        """
        logger.debug("Verifying downloaded file: %s", file_path)

        try:
            actual_size = file_path.stat().st_size
        except OSError as e:
            logger.warning("Failed to get file size: %s", str(e))
            return False

        if actual_size != expected_size:
            logger.warning("Downloaded file size mismatch: expected %d bytes, got %d bytes", expected_size, actual_size)
            return False

        return True

    def _download_to_file(self, filename: str) -> str:
        """Download a file from the registry to the cache.

        This method handles the complete download process, including:
        - Preparing the download request using the registry-specific implementation
        - Creating necessary directories
        - Performing the actual download with progress tracking
        - Verifying the downloaded file

        Args:
            filename: The target filename for the downloaded file

        Returns:
            str: Path to the downloaded file

        Raises:
            ModelRegistryError: If the download fails
        """
        logger.info(f"Starting model download from {self.__class__.__name__}")

        # Create cache directory if it doesn't exist
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

        # Create the full path to the cached file
        cached_file = self.model_cache_dir / filename

        try:
            # Lazy initialization of API key
            if self.api_key is None:
                self.api_key = self._get_api_key()
            if not self.api_key:
                log_and_raise(ModelRegistryError, "API key cannot be empty")

            session_token = self._request_session_token_from_api_key(self.api_key)

            # Lazy initialization of session
            if self.session is None:
                self.session = self._get_session(session_token)

            logger.info("Downloading model from %s", self.model_url)  # Log URL without credentials
            response = self.session.get(self.model_url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            if total_size == 0:
                log_and_raise(ModelRegistryError, "Server did not provide Content-Length header")

            with open(cached_file, "wb") as f:
                # Create a progress bar with unit='B' (bytes) and unit_scale=True to show sizes in MB/GB
                with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Downloading {filename}") as pbar:
                    for chunk in response.iter_content(chunk_size=self.CHUNK_SIZE):
                        if chunk:  # filter out keep-alive chunks
                            f.write(chunk)
                            pbar.update(len(chunk))

            # Verify the downloaded file matches expected size
            if not self._verify_downloaded_file(cached_file, total_size):
                log_and_raise(ModelRegistryError, "Downloaded file is invalid: %s", cached_file)

            logger.info("Model downloaded successfully to %s", cached_file)
            return str(cached_file)
        except requests.exceptions.RequestException as e:
            # In case of a request error, provide a hint about the API key to assist users with authentication issues
            log_and_raise(
                ModelRegistryError,
                "Failed to download model: %s\nAPI key: '%s'\nAPI key hint: %s",
                str(e),
                # Don't log full API key for security reasons
                partial_api_key(unpack_optional(self.api_key), 8),
                self.api_key_hint(),
            )
        except Exception as e:
            log_and_raise(ModelRegistryError, "Unexpected error while downloading model: %s", str(e))

    def get_model(self) -> str:
        """Download a model or use cached version if available.

        This is the main method that should be called by clients to get a model.

        Returns:
            str: Path to the downloaded model file.

        Raises:
            ModelRegistryError: If the download fails.
        """
        # Create cache directory if it doesn't exist
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured cache directory exists: %s", self.model_cache_dir)

        # Parse the model URL to get the filename
        parsed_url = urlparse(self.model_url)
        filename = os.path.basename(parsed_url.path)
        cached_file = self.model_cache_dir / filename
        logger.debug("Target cache file: %s", cached_file)

        # Check for existing file in cache
        if cached_file.exists():
            logger.debug("Found existing file in cache")
            if self._verify_cached_file(cached_file):
                logger.info("Using cached model file: %s", cached_file)
                return str(cached_file)
            else:
                logger.warning("Cached file is invalid, will redownload: %s", cached_file)
                cached_file.unlink()

        # If we got here, we need to download the file
        return self._download_to_file(filename)

    @staticmethod
    def _get_credential_from_netrc(domain: str) -> Optional[str]:
        """Get credential from the .netrc file for a specific domain.

        Args:
            domain (str): The domain to get credentials for.

        Returns:
            str: Credential if found in .netrc, None otherwise.

        Raises:
            ModelRegistryError: If there's an error reading the .netrc file.
        """
        logger.debug(f"Attempting to get credentials from the .netrc file for {domain}.")
        try:
            auth = netrc.netrc().authenticators(domain)
            if not auth:
                logger.debug("No authentication found in .netrc")
                return None
            credential = auth[2]
            if not credential:
                logger.debug("No credential found in .netrc")
                return None
            logger.debug(f"Successfully retrieved credential from the .netrc file for {domain}.")
            return credential
        except FileNotFoundError as e:
            logger.error("Failed to read .netrc file: %s", str(e))
            log_and_raise(ModelRegistryError, "Failed to read .netrc file: %s", str(e))


class NgcModelRegistry(ModelRegistry):
    """Implementation for downloading models from NVIDIA's NGC registry.

    This class handles downloading models from NVIDIA's NGC registry. It supports
    caching downloaded models to avoid redundant downloads.

    Attributes:
        api_key (str): API key for authenticating with NGC. If not provided, it will be
            retrieved from the NGC_API_KEY environment variable or from the .netrc file.
    """

    NGC_DOMAIN = "api.ngc.nvidia.com"
    API_KEY_HINT = (
        "NGC Personal Key with catalog rights (starting with 'nvapi-' - see https://ngc.nvidia.com/setup/api-keys)\n"
        + "NGC Legacy API key with catalog rights (not starting with 'nvapi-' - see https://ngc.nvidia.com/setup/api-keys)"
    )
    NGC_AUTH_DOMAIN = "authn.nvidia.com"

    @property
    def domain(self) -> str:
        return self.NGC_DOMAIN

    @staticmethod
    def api_key_hint() -> str:
        return NgcModelRegistry.API_KEY_HINT

    def _get_api_key(self, api_key: Optional[str] = None) -> Optional[str]:
        """Get the API key for the NGC model registry.

        The API key is resolved in the following order of precedence:
        1. Directly provided api_key parameter
        2. NGC_API_KEY environment variable
        3. .netrc file credentials for the registry domain

        Args:
            api_key: Optional API key provided directly.

        Returns:
            Optional[str]: The API key for the NGC registry.

        Raises:
            ModelRegistryError: If no API key could be found or if an empty API key was provided.
        """
        logger.debug("Received api_key: %s.", api_key)
        # Validate API key if provided
        if api_key is not None and not api_key.strip():
            log_and_raise(ModelRegistryError, "API key cannot be empty")

        # Get API key from NGC_API_KEY environment variable if not provided
        if not api_key:
            logger.debug("No API key provided directly, attempting to get it from NGC_API_KEY environment variable.")
            api_key = os.environ.get("NGC_API_KEY")
            if api_key:
                logger.debug("Successfully retrieved API key from NGC_API_KEY environment variable.")

        # Get API key from .netrc file if not provided
        api_key = super()._get_api_key(api_key)

        # Check if api-key has the expected form of a NGC Personal Key and warn otherwise
        if api_key and (not api_key.startswith("nvapi-")):
            logger.warning(
                # Don't log full API key for security reasons
                f"The NGC API key '{partial_api_key(api_key, 8)}' does not appear to be a valid NGC Personal Key (expected to be 'nvapi-' prefixed) "
                "and might result in authentication issues [legacy NGC API keys are likely unsupported]."
            )

        return api_key

    def _extract_org_team_from_model_url(self) -> tuple[str, str]:
        """Extract org and team from self.model_url.

        Returns:
            tuple: (org, team) extracted from URL

        Raises:
            ModelRegistryError: If URL format is invalid
        """
        pattern = r"https://api\.ngc\.nvidia\.com/v2/org/([^/]+)/team/([^/]+)/.*"
        match = re.match(pattern, self.model_url)
        if not match:
            log_and_raise(ModelRegistryError, "Invalid NGC URL format, cannot extract org/team: %s", self.model_url)

        org, team = match.groups()
        logger.debug("Extracted org='%s', team='%s' from URL", org, team)
        return org, team

    def _request_session_token_from_legacy_api_key(self, legacy_api_key: str, org: str, team: str) -> str:
        """Exchange legacy API key for session token via NGC authentication service.

        Args:
            legacy_api_key: The legacy NGC API key
            org: Organization name (e.g., 'nvstaging')
            team: Team name (e.g., 'nre')

        Returns:
            str: Session token for authenticated requests

        Raises:
            ModelRegistryError: If token exchange fails
        """
        logger.debug("Exchanging legacy API key for session token")

        # Construct the scope parameter for the authentication request
        scope = f"group/ngc:{org}&group/ngc:{org}/{team}"
        auth_url = f"https://{self.NGC_AUTH_DOMAIN}/token?service=ngc&scope={scope}"

        try:
            # Make the token exchange request
            response = requests.get(
                auth_url, auth=("$oauthtoken", legacy_api_key), headers={"Accept": "application/json"}, timeout=30
            )
            response.raise_for_status()

            # Extract the token from the response
            token_data = response.json()
            session_token = token_data.get("token")

            if not session_token:
                log_and_raise(ModelRegistryError, "No token found in authentication response")

            logger.debug("Successfully exchanged legacy API key for session token")
            return session_token

        except requests.exceptions.RequestException as e:
            log_and_raise(ModelRegistryError, "Failed to exchange legacy API key for session token: %s", str(e))
        except (KeyError, ValueError) as e:
            log_and_raise(ModelRegistryError, "Invalid response format from NGC authentication service: %s", str(e))

    def _request_session_token_from_api_key(self, api_key: str) -> str:
        """Get a session token for the NGC model registry with proper authentication.

        This method handles the NGC-specific authentication flow:
        1. Gets the legacy API key
        2. Extracts org/team from the model URL
        3. Exchanges legacy API key for session token

        Returns:
            session_token: A session token configured for NGC authentication
        """

        # Potential session token is the API key/personal access token (PAT)
        potential_session_token = super()._request_session_token_from_api_key(api_key)

        # PATs start with "nvapi-", legacy access tokens don't.
        if potential_session_token.startswith("nvapi-"):
            # PAT is the session token
            return potential_session_token
        else:
            # Legacy access tokens need to request session tokens for a given org and team
            org, team = self._extract_org_team_from_model_url()
            return self._request_session_token_from_legacy_api_key(potential_session_token, org, team)


def create_model_registry(model_url: str, model_cache_dir: Path, api_key: Optional[str] = None) -> ModelRegistry:
    """Factory: only the NGC registry domain is supported in the standalone."""
    parsed_url = urlparse(model_url)
    domain = parsed_url.netloc
    if domain == NgcModelRegistry.NGC_DOMAIN:
        return NgcModelRegistry(model_url=model_url, model_cache_dir=model_cache_dir, api_key=api_key)
    log_and_raise(ModelRegistryError, "Unsupported model registry domain: %s", domain)
