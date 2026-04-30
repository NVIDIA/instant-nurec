# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import logging
import re

from concurrent import futures
from pathlib import Path
from typing import List, NamedTuple

from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


log = logging.getLogger(__name__)

# RE to parse a content-range header
CONTENT_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")

DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_WORKERS = 16


class ProbeResult(NamedTuple):
    """Result of probing a file for range request support."""

    is_range_supported: bool
    """True if the file supports range requests, False otherwise."""

    file_size: int
    """Size of the file in bytes."""


class ContentRange(NamedTuple):
    """
    Result of parsing a Content-Range header

    Example of the header value: 'bytes 42-1980/1742413650'
    This means that the size is 1742413650 bytes, and the range starts at 42 and ends at 1980.
    The end is not inclusive, so the last byte is 1979.
    """

    start: int
    """Start byte of the range."""

    end: int
    """End byte of the range."""

    size: int
    """Size of the range."""


class ChunkRange(NamedTuple):
    """Represents a range of bytes to download as a chunk."""

    start: int
    end: int


class Downloader:
    """
    A parallel file downloader that efficiently downloads files from URLs.

    This class provides functionality to download files from URLs, automatically choosing
    between parallel and sequential download strategies based on server capabilities.

    Public Methods:
        download(url: str, output_path: Path) -> int

    Configuration options:
        - chunk_size: Size of each download chunk (default: 8MB)
        - max_workers: Maximum number of parallel download workers (default: 16)
        - max_retries: Maximum number of retry attempts for failed downloads (default: 3)
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_workers: int = DEFAULT_MAX_WORKERS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        """
        Initialize the Downloader with configuration parameters.

        Args:
            chunk_size (int): Size of each download chunk in bytes. This determines how large each
                parallel download segment will be. Default: 8MB (8 * 1024 * 1024 bytes)
            max_workers (int): Maximum number of parallel download workers. This controls how many
                concurrent downloads can be performed. Default: 16 workers
            max_retries (int): Maximum number of retry attempts for failed downloads. This applies
                to both individual chunk downloads and full file downloads. Default: 3 retries
        """
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.max_retries = max_retries

    def download(self, url: str, output_path: Path) -> int:
        """
        Download a file from a URL in parallel chunks.

        Using HTTP range requests if supported.
        If range requests are not supported, download the file sequentially.

        URL must be presigned or public.

        Args:
            url (str): The URL to download from
            output_path (Path): The path to save the downloaded file
        Returns:
            int: The total number of bytes downloaded
        """
        if self.max_workers < 2:
            return self._download_file_sequential(url, output_path)

        probe_result = self._probe_file(url)

        if not probe_result.is_range_supported:
            return self._download_file_sequential(url, output_path)

        return self._download_file_concurrent(url, output_path, probe_result.file_size)

    @property
    def session(self) -> Session:
        """
        Create a new requests session with a retry strategy

        Note: do not reuse session for concurrent downloads, create a new one instead to utilize
        multiple connections.
        """
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=2,
            status_forcelist=[  # HTTP status codes to retry on
                429,  # Too many requests
                500,  # Internal server error
                502,  # Bad gateway
                503,  # Service unavailable
                504,  # Gateway timeout
            ],
            connect=self.max_retries,  # Retry on connection errors (including DNS resolution)
            read=self.max_retries,  # Retry on read timeouts
            other=self.max_retries,  # Retry on other issues like SSL errors
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)

        session = Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        return session

    def _probe_file(self, url: str) -> ProbeResult:
        """
        Probe a file URL to determine if it supports HTTP range requests.

        Args:
            url (str): URL of the file to probe

        Returns:
            ProbeResult: A named tuple containing is_range_supported and file_size
        """
        #  Check if URL is presigned by checking for X-Amz-Signature query parameter
        if "X-Amz-Signature" in url:
            # A presigned URL would fail HEAD probe, so we use GET
            return self._probe_file_get(url)

        try:
            return self._probe_file_head(url)
        except Exception as e:
            log.warning(f"HEAD probe failed: {e}")
            return self._probe_file_get(url)

    def _probe_file_head(self, url: str) -> ProbeResult:
        """
        Probe a file to determine if it supports HTTP range requests using HEAD.

        Args:
            url (str): URL of the file to probe

        Returns:
            ProbeResult: A named tuple containing is_range_supported and file_size
        """
        resp = self.session.head(url, allow_redirects=True)
        resp.raise_for_status()

        return ProbeResult(
            is_range_supported=resp.headers.get("accept-ranges") == "bytes",
            file_size=int(resp.headers.get("content-length", 0)),
        )

    def _probe_file_get(self, url: str) -> ProbeResult:
        """
        Probe a file to determine if it supports HTTP range requests using GET.

        This is a fallback for presigned URLs.

        Args:
            url (str): URL of the file to probe

        Returns:
            ProbeResult: A named tuple containing is_range_supported and file_size
        """
        # Try to get first byte of the file
        resp = self.session.get(url, headers={"Range": "bytes=0-0"}, allow_redirects=True)
        resp.raise_for_status()

        log.debug(f"Probe GET {url} {resp.status_code} {resp.headers}")

        # accept-ranges must be bytes
        if resp.headers.get("accept-ranges") != "bytes":
            raise Exception("Range not supported")

        # parse range header
        range_header = resp.headers.get("content-range")
        if range_header is None:
            raise Exception("Content-Range header not found")

        content_range = self._parse_content_range(range_header)

        log.debug(
            f"Probe GET {url} {resp.status_code=} {resp.headers=} {content_range.start=} {content_range.end=} {content_range.size=}"
        )

        return ProbeResult(is_range_supported=True, file_size=content_range.size)

    def _parse_content_range(self, range_value: str) -> ContentRange:
        """
        Parse a content-range header value (e.g. 'bytes 42-1980/1742413650').

        Args:
            range_value (str): content-range header value from the probe response
        Returns:
            ContentRange: named tuple containing start, end, and size values
        """

        match = CONTENT_RANGE_RE.match(range_value)
        if match is None:
            raise Exception(f"Invalid content-range header: {range_value}")

        return ContentRange(start=int(match.group(1)), end=int(match.group(2)), size=int(match.group(3)))

    def _calculate_chunks(self, file_size: int) -> List[ChunkRange]:
        """
        Split the work for up to max_workers, so each worker download one chunk.

        Args:
            file_size (int): size of the file to download
        Returns:
            List[ChunkRange]: list of ChunkRange representing the range of each chunk
        """
        # Calculate chunk sizes
        chunk_count = min(self.max_workers, max(1, file_size // self.chunk_size))
        chunk_size = file_size // chunk_count
        chunks = [ChunkRange(i * chunk_size, min((i + 1) * chunk_size - 1, file_size - 1)) for i in range(chunk_count)]

        # chunk_size * chunk_count may be less than file_size, so we need to adjust the last chunk
        # to account for the missing bytes (up to chunk_count bytes)
        chunks[-1] = ChunkRange(chunks[-1].start, file_size - 1)

        return chunks

    # Function to download a single chunk
    def _download_chunk(self, url: str, output: Path, index: int, start: int, end: int) -> int:
        """
        Download a single chunk of the file and write it to the output file at the correct position.

        Args:
            url (str):     URL of the file to download
            output (Path): path to save the downloaded file
            index (int):   index of the chunk
            start (int):   start byte of the chunk
            end (int):     end byte of the chunk
        Returns:
            int: number of bytes downloaded, e.g. total size of the chunk
        """
        log.debug(f"Downloading chunk #{index:02d} {start}-{end} ({end - start + 1} B)")
        headers = {"Range": f"bytes={start}-{end}"}
        resp = self.session.get(url, headers=headers, stream=True, allow_redirects=True)
        resp.raise_for_status()

        # the content length of the chunk must match the expected chunk size
        content_length = int(resp.headers.get("content-length", 0))
        if content_length != end - start + 1:
            log.error(f"Content-Length mismatch for chunk #{index:02d}: {content_length} != {end - start + 1}")
            raise ValueError(f"Content-Length mismatch for chunk #{index:02d}: {content_length} != {end - start + 1}")

        # Write the chunk to the output file at the correct position
        with open(output, "r+b") as f:
            f.seek(start)
            f.write(resp.content)

        return end - start + 1

    def _download_file_concurrent(self, url: str, output: Path, file_size: int) -> int:
        """
        Download a file in parallel chunks.

        Args:
            url (str):       URL of the file to download
            output (Path):   path to save the downloaded file
            file_size (int): size of the file to download
        Returns:
            int: number of bytes downloaded, e.g. total size of the file
        """

        chunks = self._calculate_chunks(file_size)

        # Create the empty output file with the correct size
        with open(output, "wb") as f:
            f.truncate(file_size)

        # Download chunks in parallel
        total_bytes_downloaded = 0
        with futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            download_chunk_futures = [
                executor.submit(self._download_chunk, url, output, index, start, end)
                for index, (start, end) in enumerate(chunks)
            ]

            # Wait for completion and collect results
            for future in futures.as_completed(download_chunk_futures):
                try:
                    total_bytes_downloaded += future.result()
                except Exception as e:
                    log.error(f"Chunk download failed: {e}")
                    raise

        return total_bytes_downloaded

    def _download_file_sequential(self, url: str, output: Path) -> int:
        """
        Fallback for sequential download.

        Args:
            url (str):     URL of the file to download
            output (Path): path to save the downloaded file
        Returns:
            int: number of bytes downloaded, e.g. total size of the file
        """
        total_bytes_downloaded = 0
        with self.session.get(url, stream=True, allow_redirects=True) as r:
            r.raise_for_status()
            file_size = int(r.headers.get("content-length", 0))
            log.info(f"Downloading {url} sequentially ({file_size / 1024 / 1024:.1f} MB)")

            with open(output, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    total_bytes_downloaded += len(chunk)

        return total_bytes_downloaded


################################################################################
# Scene ID validation
################################################################################

# RE to validate a scene ID
# - Alphanumeric characters
# - Underscores
# - Hyphens
# - dots
SCENE_ID_PATTERN: re.Pattern = re.compile(r"^[.a-zA-Z0-9_-]+$")


def check_safe_scene_id(scene_id: str) -> bool:
    """
    Validate scene_id to be safe for a filename

    Allow letters, numbers, underscores, and hyphens
    """
    global SCENE_ID_PATTERN

    return SCENE_ID_PATTERN.match(scene_id) is not None
