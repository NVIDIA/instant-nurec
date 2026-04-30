# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import io
import tempfile

from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest
import requests

from requests.structures import CaseInsensitiveDict

from nre.grpc.downloader import ContentRange, Downloader, ProbeResult, check_safe_scene_id


# Shortcut for chunk size
CS = 1024 * 1024


@pytest.fixture
def downloader():
    """Fixture that provides a Downloader instance with custom configuration."""
    return Downloader(chunk_size=CS, max_workers=8)


@pytest.fixture
def mock_probe_file_get(downloader: Downloader):
    """
    Fixture to mock the _probe_file_get method returning the given probe result or raising the given exception using the side_effect parameter
    """

    def __patcher(is_range_supported: bool = False, file_size: int = 0, side_effect=None):
        if side_effect is None:
            return patch.object(
                downloader,
                "_probe_file_get",
                return_value=ProbeResult(is_range_supported=is_range_supported, file_size=file_size),
            )
        else:
            return patch.object(
                downloader,
                "_probe_file_get",
                side_effect=side_effect,
            )

    return __patcher


@pytest.fixture
def mock_probe_file_head(downloader: Downloader):
    """
    Fixture to mock the _probe_file_head method returning the given probe result or raising the given exception using the side_effect parameter
    """

    def __patcher(is_range_supported: bool = False, file_size: int = 0, side_effect=None):
        if side_effect is None:
            return patch.object(
                downloader,
                "_probe_file_head",
                return_value=ProbeResult(is_range_supported=is_range_supported, file_size=file_size),
            )
        else:
            return patch.object(
                downloader,
                "_probe_file_head",
                side_effect=side_effect,
            )

    return __patcher


class TestConfiguration:
    def test_default_values(self, downloader: Downloader):
        """Test that Downloader initializes with correct default values when no config is provided."""
        downloader = Downloader()

        assert downloader.chunk_size == 8 * 1024 * 1024
        assert downloader.max_workers == 16

    def test_custom_values(self):
        """Test that Downloader initializes with correct custom values when config is provided."""
        downloader = Downloader(chunk_size=1024 * 1024, max_workers=8)

        assert downloader.chunk_size == 1024 * 1024
        assert downloader.max_workers == 8


class TestCalculateChunks:
    def test_calculate_chunks_xsmall_file(self, downloader: Downloader):
        """Test that _calculate_chunks returns a single chunk for files smaller than chunk size."""
        # Smaller than chunk size

        chunks = downloader._calculate_chunks(downloader.chunk_size - 1)

        assert len(chunks) == 1

    @pytest.mark.parametrize(
        "file_size,expected_num_chunks",
        [
            # one chunk
            (CS, 1),
            (CS - 1, 1),
            (CS + 1, 1),
            (int(CS * 1.5), 1),
            (CS * 2 - 1, 1),
            # two chunks
            (CS * 2, 2),
            (CS * 2 + 1, 2),
            (int(CS * 2.5), 2),
            (CS * 3 - 1, 2),
            # three chunks
            (CS * 3, 3),
            (CS * 3 + 1, 3),
            (int(CS * 3.5), 3),
            (CS * 4 - 1, 3),
        ],
    )
    def test_calculate_chunks_small_file(self, downloader: Downloader, file_size: int, expected_num_chunks: int):
        """Test that _calculate_chunks correctly divides files into appropriate number of chunks.

        Verifies that:
        1. The number of chunks is correct for various file sizes
        2. Chunks are contiguous (no gaps or overlaps)
        3. All chunks except the last have the same size
        4. The last chunk ends at the file size
        """
        # _calculate_chunks adjust the chunk size/count trying to use as many workers as possible
        chunks = downloader._calculate_chunks(file_size)

        assert len(chunks) == expected_num_chunks

        # Get the chunk size from the first chunk
        chunk_size = chunks[0][1] - chunks[0][0] + 1

        # Previous chunk end
        # -1 to account for the fact that the first chunk starts at 0
        prev_end = -1

        # Check chunk offsets
        for i, (start, end) in enumerate(chunks):
            # Second and further chunks start where the previous chunk ended
            assert start == prev_end + 1

            # all chunks are of the same size, except the last one
            if i < len(chunks) - 1:
                assert end - start + 1 == chunk_size
            else:
                # last chunk may be bigger to account for the missing bytes (up to chunk_count bytes)
                assert end - start + 1 <= chunk_size + (len(chunks) - 1)

            prev_end = end

        # last chunk ends at the file size
        assert chunks[-1][1] == file_size - 1


class TestContentRangeParsing:
    """Test class for testing the _parse_content_range method of the Downloader class."""

    @pytest.mark.parametrize(
        "range_header,expected",
        [
            ("bytes 42-1980/1742413650", (42, 1980, 1742413650)),
            ("bytes 0-0/2233481788", (0, 0, 2233481788)),
        ],
    )
    def test_parse_content_range_header(self, downloader: Downloader, range_header: str, expected: ContentRange):
        """Test parsing of valid content-range headers."""
        result = downloader._parse_content_range(range_header)
        assert result == ContentRange(start=expected[0], end=expected[1], size=expected[2])

    @pytest.mark.parametrize(
        "range_header",
        [
            "x-bytes 42-1980/5555555555555",  # invalid header name
            "bytes 42-1980/",  # missing length
            "bytes 42-1980/1742413650/1742413650",  # too many arguments
            "bytes aaa-1980/2025",  # invalid start
            "bytes  42-/2025",  # invalid end
            "bytes \r\n42-1980/2025",  # multi-line
        ],
    )
    def test_parse_content_range_header_invalid(self, downloader: Downloader, range_header: str):
        """Test that _parse_content_range raises an exception for invalid content-range headers."""
        with pytest.raises(Exception):
            downloader._parse_content_range(range_header)


class TestSession:
    def test_session_is_valid(self, downloader: Downloader):
        """Test that the downloader has a valid session attribute."""
        assert downloader.session is not None

    def test_session_is_not_reused(self, downloader: Downloader):
        """Test that each call to session creates a new session object rather than reusing the previous one."""
        s1 = downloader.session
        s2 = downloader.session

        assert s1 is not s2


def make_mock_response(status_code: int = 200, headers: Optional[dict] = None, content: Optional[bytes] = None):
    """
    Helper function to create a mock response with the given headers and status code.
    """

    mock_response = requests.Response()
    mock_response.status_code = status_code

    if headers is None:
        mock_response.headers = CaseInsensitiveDict()
    else:
        mock_response.headers = CaseInsensitiveDict(data=headers)

    if content is not None:
        mock_response.raw = io.BytesIO(content)

    return mock_response


class TestProbeFileHead:
    """
    Test for Downloader._probe_file_head

    The test uses mocks to simulate the response from the server.

    The safety-switch: the actual https://example.com/file.txt does not exists,
    so the response will be a 404 error if the mock is not set correctly.
    """

    def test_accept_ranges_header_bytes(self, downloader: Downloader):
        """
        Test that the probe_file_head method returns the correct probe result when the file is range supported.
        """

        mock_response = make_mock_response(
            headers={
                "accept-ranges": "bytes",
                "content-length": "42",
            }
        )

        with patch.object(requests.Session, "head", return_value=mock_response) as mock_session_head:
            result = downloader._probe_file_head("https://example.com/file.txt")
            mock_session_head.assert_called_once_with("https://example.com/file.txt", allow_redirects=True)
            assert result == ProbeResult(is_range_supported=True, file_size=42)

    def test_accept_ranges_header_missing(self, downloader: Downloader):
        """
        Test that the probe_file_head method returns the correct probe result when the file is not range supported.
        """

        mock_response = make_mock_response(headers={"content-length": "42"})

        with patch.object(requests.Session, "head", return_value=mock_response) as mock_session_head:
            result = downloader._probe_file_head("https://example.com/file.txt")
            mock_session_head.assert_called_once_with("https://example.com/file.txt", allow_redirects=True)
            assert result == ProbeResult(is_range_supported=False, file_size=42)

    def test_head_method_not_supported(self, downloader: Downloader):
        """
        Test that the probe_file_head method returns the correct probe result when HEAD method is not supported.
        """

        # Mock response
        mock_response = make_mock_response(status_code=405, headers={"allow": "GET"})

        with patch.object(requests.Session, "head", return_value=mock_response) as mock_session_head:
            # expect HTTPError to be raised
            with pytest.raises(requests.HTTPError):
                downloader._probe_file_head("https://example.com/file.txt")

            mock_session_head.assert_called_once_with("https://example.com/file.txt", allow_redirects=True)


class TestProbeFileGet:
    """
    Tests for Downloader._probe_file_get

    The test uses mocks to simulate the response from the server.

    The safety-switch: the actual https://example.com/file.txt does not exists,
    so the response will be a 404 error if the mock is not set correctly.
    """

    def test_content_range_header_bytes(self, downloader: Downloader):
        """
        Test that the probe_file_get method returns the correct probe result when the file is range supported.

        _probe_file_get is expected to request one byte from the beginning of the file (Range header is set to 0-0).

        The expected reponse headers:
            - accept-ranges: bytes
            - content-range: bytes 0-0/FILE_SIZE
            - content-length: 1
        """

        mock_response = make_mock_response(
            headers={
                "accept-ranges": "bytes",
                "content-range": "bytes 0-0/42",
                "content-length": "1",
            }
        )

        with patch.object(requests.Session, "get", return_value=mock_response) as mock_session_get:
            result = downloader._probe_file_get("https://example.com/file.txt")
            mock_session_get.assert_called_once_with(
                "https://example.com/file.txt", allow_redirects=True, headers={"Range": "bytes=0-0"}
            )
            assert result == ProbeResult(is_range_supported=True, file_size=42)

    def test_accept_ranges_header_is_missing(self, downloader: Downloader):
        """
        Test that the probe_file_get method returns the correct probe result when the file is not range supported.
        """

        mock_response = make_mock_response(
            headers={
                # accept-ranges is missing
                "content-range": "bytes 0-0/42",
                "content-length": "1",
            }
        )

        with patch.object(requests.Session, "get", return_value=mock_response) as mock_session_get:
            # expected the exception to be raised
            with pytest.raises(Exception) as e:
                downloader._probe_file_get("https://example.com/file.txt")

            mock_session_get.assert_called_once_with(
                "https://example.com/file.txt", allow_redirects=True, headers={"Range": "bytes=0-0"}
            )

            assert str(e.value) == "Range not supported"

    def test_accept_ranges_header_is_junk(self, downloader: Downloader):
        """
        Test that the probe_file_get method returns the correct probe result when the file is not range supported.
        """

        mock_response = make_mock_response(
            headers={
                "accept-ranges": "x-test-not-bytes",
                "content-range": "bytes 0-0/42",
                "content-length": "1",
            }
        )

        with patch.object(requests.Session, "get", return_value=mock_response) as mock_session_get:
            # expected the exception to be raised
            with pytest.raises(Exception) as e:
                downloader._probe_file_get("https://example.com/file.txt")

            mock_session_get.assert_called_once_with(
                "https://example.com/file.txt", allow_redirects=True, headers={"Range": "bytes=0-0"}
            )

            assert str(e.value) == "Range not supported"

    def test_content_range_header_is_missing(self, downloader: Downloader):
        """
        Test that the probe_file_get method returns the correct probe result when the content-range header is missing.
        """

        mock_response = make_mock_response(
            headers={
                "accept-ranges": "bytes",
                # content-range is missing
                "content-length": "1",
            }
        )

        with patch.object(requests.Session, "get", return_value=mock_response) as mock_session_get:
            # expected the exception to be raised
            with pytest.raises(Exception) as e:
                downloader._probe_file_get("https://example.com/file.txt")

            mock_session_get.assert_called_once_with(
                "https://example.com/file.txt", allow_redirects=True, headers={"Range": "bytes=0-0"}
            )

            assert str(e.value) == "Content-Range header not found"


class TestProbeFile:
    def test_probe_presigned_url_with_get(self, downloader: Downloader, mock_probe_file_get, mock_probe_file_head):
        """
        Test that the probe_file_get method returns the correct probe result when the file is a presigned URL.

        Ensure _probe_file_get is called with the correct arguments
        """

        url = "https://example.com/file.txt?X-Amz-Signature-fake=signature"

        with (
            mock_probe_file_head(True, 42) as probe_file_head,
            mock_probe_file_get(True, 42) as probe_file_get,
        ):
            result = downloader._probe_file(url)

            probe_file_get.assert_called_once_with(url)
            probe_file_head.assert_not_called()

            assert result == ProbeResult(is_range_supported=True, file_size=42)

        with (
            mock_probe_file_head(False, 0) as probe_file_head,
            mock_probe_file_get(False, 0) as probe_file_get,
        ):
            result = downloader._probe_file(url)

            probe_file_get.assert_called_once_with(url)
            probe_file_head.assert_not_called()

            assert result == ProbeResult(is_range_supported=False, file_size=0)

    def test_probe_head_raises_exception(self, downloader: Downloader, mock_probe_file_get, mock_probe_file_head):
        """
        Test that the probe_file_head method raises an exception when the file is not range supported.
        """

        url = "https://example.com/file.txt"

        # Mock _probe_file_head to raise an exception
        with (
            mock_probe_file_head(side_effect=Exception("x-head-function-fails")) as probe_file_head,
            mock_probe_file_get(True, 42) as probe_file_get,
        ):
            result = downloader._probe_file(url)

            probe_file_get.assert_called_once_with(url)
            probe_file_head.assert_called_once_with(url)

            assert result == ProbeResult(is_range_supported=True, file_size=42)

        with (
            mock_probe_file_head(side_effect=Exception("x-head-function-fails")) as probe_file_head,
            mock_probe_file_get(False, 0) as probe_file_get,
        ):
            result = downloader._probe_file(url)

            probe_file_get.assert_called_once_with(url)
            probe_file_head.assert_called_once_with(url)

            assert result == ProbeResult(is_range_supported=False, file_size=0)


########################################################
# Tests for low-level downloading methods
########################################################


@pytest.fixture
def temp_file_1024(downloader: Downloader):
    """
    Fixture to create a temporary file of the fixed size
    """

    size = 1024  # 1 KB

    with tempfile.NamedTemporaryFile(delete=False, mode="wb") as f:
        f.write(b"1" * size)
        f.flush()

        yield Path(f.name)


class TestDownloadChunk:
    """
    Test for Downloader._download_chunk
    """

    def test_basic(self, downloader: Downloader, temp_file_1024: Path):
        """
        Download a single chunk of the file and verify the content

        Chunk size is 42, offset is 10 (bytes from 10 to 52)

        """
        chunk_size = 42
        offset = 43

        # mock session.get
        mock_response = make_mock_response(
            headers={
                "accept-ranges": "bytes",
                "content-range": "bytes 10-52/1024",
                "content-length": "42",
            },
            content=b"X" * chunk_size,
        )

        chunk_start = offset
        chunk_end = offset + chunk_size - 1

        range_header = f"bytes={offset}-{offset + chunk_size - 1}"

        with patch.object(requests.Session, "get", return_value=mock_response) as mock_session_get:
            result = downloader._download_chunk(
                "https://example.com/file.txt", temp_file_1024, 0, chunk_start, chunk_end
            )

            mock_session_get.assert_called_once_with(
                "https://example.com/file.txt", stream=True, allow_redirects=True, headers={"Range": range_header}
            )

            assert result == 42

            # check the content of the file
            with open(temp_file_1024, "rb") as f:
                # expected the content to be 1111...1111XXX....XXX1111...1111
                #                                       ^ offset  ^ chunk_end
                assert f.read() == b"1" * offset + b"X" * chunk_size + b"1" * (1024 - offset - chunk_size)


class TestDownloadFileSequential:
    """
    Test for Downloader._download_file_sequential
    """

    def test_basic(self, downloader: Downloader, temp_file_1024: Path):
        """
        Download a file sequentially and verify the content
        """

        # mock session.get
        mock_response = make_mock_response(
            headers={
                "content-length": "1024",
            },
            content=b"X" * 1024,
        )

        with patch.object(requests.Session, "get", return_value=mock_response) as mock_session_get:
            result = downloader._download_file_sequential("https://example.com/file.txt", temp_file_1024)

            mock_session_get.assert_called_once_with("https://example.com/file.txt", stream=True, allow_redirects=True)

            assert result == 1024

            # check the content of the file
            with open(temp_file_1024, "rb") as f:
                assert f.read() == b"X" * 1024


########################################################
# Fixtures to mock low-level downloader methods
# to test a higher-level logic
########################################################

FILE_SIZE_100M = 1024 * 1024 * 100


@pytest.fixture
def temp_filename():
    """
    Fixture to generate a unique filename

    The filename is guaranteed to be unique and not exist on the file system.

    The test must not create a file.
    """

    file_path = Path(tempfile.mktemp())

    yield file_path

    if file_path.exists():
        file_path.unlink()


@pytest.fixture
def file_size():
    """
    Fixture to provide an expected file size

    File is neither expected to exist on the file system, nor to be created by the test.
    """

    return 1024 * 1024 * 100


@pytest.fixture
def mock_probe_file_range_not_supported(downloader: Downloader, file_size):
    """
    Fixture to mock the _download_file_sequential method
    """

    return patch.object(
        downloader, "_probe_file", return_value=ProbeResult(is_range_supported=False, file_size=file_size)
    )


@pytest.fixture
def mock_probe_file_range_supported(downloader: Downloader, file_size):
    """
    Fixture to mock the _download_file_sequential method
    """

    return patch.object(
        downloader, "_probe_file", return_value=ProbeResult(is_range_supported=True, file_size=file_size)
    )


@pytest.fixture
def mock_download_file_sequential(downloader: Downloader, file_size):
    """
    Fixture to mock the _download_file_sequential method
    """

    return patch.object(downloader, "_download_file_sequential", return_value=file_size)


@pytest.fixture
def mock_download_file_concurrent(downloader: Downloader, file_size):
    """
    Fixture to mock the _download_file_concurrent method
    """

    return patch.object(downloader, "_download_file_concurrent", return_value=file_size)


class TestDownloadFileConcurrent:
    """
    Test for Downloader._download_file_concurrent

    The suite validates that _download_file_concurrent calls _download_chunk with the correct arguments.
    """

    def test_download_chunks_are_called_correctly(self, downloader: Downloader, temp_filename: Path):
        """
        Ensure that _download_chunk is called with the correct arguments when downloading a file concurrently.

        Note: The test does not fetch any data from the network.
        It only validates the arguments passed to _download_chunk.

        Validates:
            - the file is created
            - the file size matches
            - the number of chunks is correct
            - the chunk arguments are correct according to pre-calculated chunks
        """

        downloader.chunk_size = 1024  # 1k
        test_file_size = 1024 * 100  # 100k

        chunks = downloader._calculate_chunks(test_file_size)

        def __download_chunk_check(url: str, output: Path, index: int, start: int, end: int) -> int:
            """
            Mock _download_chunk to return the correct number of bytes downloaded

            Validates:
                - the arguments are correct according to pre-calculated chunks
            """

            assert url == "https://example.com/file.txt"
            assert output == temp_filename

            assert index in range(len(chunks))
            assert start == chunks[index][0]
            assert end == chunks[index][1]

            return end - start + 1

        with (
            patch.object(downloader, "_download_chunk", wraps=__download_chunk_check) as download_chunk,
        ):
            result = downloader._download_file_concurrent("https://example.com/file.txt", temp_filename, test_file_size)

            assert result == test_file_size
            # Use call_args_list instead of call_count because MagicMock.call_count
            # uses non-atomic `self.call_count += 1` which races under ThreadPoolExecutor.
            assert len(download_chunk.call_args_list) == len(chunks)

            # file is created
            assert temp_filename.exists()
            assert temp_filename.stat().st_size == test_file_size


class TestDownload:
    def test_sequential_with_one_worker(
        self,
        downloader: Downloader,
        temp_filename: Path,
        mock_probe_file_range_not_supported,
        mock_download_file_sequential,
        mock_download_file_concurrent,
        file_size,
    ):
        """
        Sequential download with one worker

        Validates:
            - sequential download is called
            - concurrent download is not called
            - probe_file is not called
        """

        downloader.max_workers = 1

        with (
            mock_probe_file_range_not_supported as probe_file,
            mock_download_file_sequential as download_file_sequential,
            mock_download_file_concurrent as download_file_concurrent,
        ):
            result = downloader.download("https://example.com/file.txt", temp_filename)

            probe_file.assert_not_called()
            download_file_sequential.assert_called_once_with("https://example.com/file.txt", temp_filename)
            download_file_concurrent.assert_not_called()

            assert result == file_size

    def test_sequential_with_multiple_workers(
        self,
        downloader: Downloader,
        temp_filename: Path,
        mock_probe_file_range_not_supported,
        mock_download_file_sequential,
        mock_download_file_concurrent,
        file_size,
    ):
        """
        Sequential download with multiple workers when the file is not range supported

        Validates:
            - sequential download is called
            - concurrent download is not called
        """

        with (
            mock_probe_file_range_not_supported as probe_file,
            mock_download_file_sequential as download_file_sequential,
            mock_download_file_concurrent as download_file_concurrent,
        ):
            result = downloader.download("https://example.com/file.txt", temp_filename)

            probe_file.assert_called_once_with("https://example.com/file.txt")
            download_file_sequential.assert_called_once_with("https://example.com/file.txt", temp_filename)
            download_file_concurrent.assert_not_called()

            assert result == file_size

    def test_concurrent_basic(
        self,
        downloader: Downloader,
        temp_filename: Path,
        mock_probe_file_range_supported,
        mock_download_file_sequential,
        mock_download_file_concurrent,
        file_size,
    ):
        """
        Concurrent download with multiple workers when the file is range supported

        Validates:
            - sequential download is not called
            - concurrent download is called with the correct arguments
        """

        with (
            mock_probe_file_range_supported as probe_file,
            mock_download_file_sequential as download_file_sequential,
            mock_download_file_concurrent as download_file_concurrent,
        ):
            result = downloader.download("https://example.com/file.txt", temp_filename)

            probe_file.assert_called_once_with("https://example.com/file.txt")
            download_file_sequential.assert_not_called()
            download_file_concurrent.assert_called_once_with("https://example.com/file.txt", temp_filename, file_size)

            assert result == file_size


class TestSafeSceneId:
    """
    Test for Downloader.check_safe_scene_id
    """

    @pytest.mark.parametrize(
        "scene_id",
        [
            "my_scene_id",
            "my_scene_id_123",
            # UUIDs are valid scene IDs
            "90b2cfcb-bbf3-419c-b8a5-2c2bb332baf6",
            # Some valid scene IDs from the real world
            "d11c0e82-f689-4401-b2e5-0ea2a6a08060_130284183747_130304183747_clipgt-d2647214-13a9-4bb0-aaad-9e6cca41cf21.usdz",
            "clipgt-c49a3cc0-9708-4e74-bc0c-60d6afb6a26b",
        ],
    )
    def test_safe_scene_id(self, scene_id: str):
        assert check_safe_scene_id(scene_id)

    @pytest.mark.parametrize(
        "scene_id",
        [
            "",
            " /slashes/are/bad",
            "../../x-bad",
            "../../etc/x-test-passwd",
            "spaces are bad",
            "escaped%20spaces%20are%20bad",
            "speci@l!char$@are#bad",
            "&amp-are-bad",
        ],
    )
    def test_unsafe_scene_id(self, scene_id: str):
        assert not check_safe_scene_id(scene_id)
