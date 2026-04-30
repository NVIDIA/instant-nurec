# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import shutil
import tempfile

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from upath import UPath


def parse_universal_path(path: str, s3_block_size_mb: int = 50, s3_cache_type: str = "readahead") -> UPath:
    """
    Parse a universal path into a UPath object.

    Args:
        path: The path to parse. Example paths include:
            - s3@pdx-team-ncore://your-bucket/file.txt [This would access file.txt under your bucket, using the pdx-team-ncore profile for S3]
            - s3://your-bucket/file.txt [This would access file.txt under your bucket, using the default profile for S3]
            - /path/to/local/file.txt [This would access file.txt in the local filesystem]
            - (Other protocols supported by upath, e.g. https://example.com/file.txt, should also work, but not thoroughly tested)

        s3_block_size_mb: The block size for downloading in MB if path protocol is s3.
        s3_cache_type: Default cache type for the file descriptor if path protocol is s3.
            Note this is different from simplecache or filecache provided for the upath itself, where cache during the lifecycle
            of the upath, instead of the file descriptor.

    Returns:
        A UPath object.
    """

    # First test if path is in the format of <protocol>://<path>
    try:
        protocol, base_path = path.split("://", 1)
    except ValueError:
        # https://github.com/fsspec/universal_pathlib?tab=readme-ov-file#local-paths-and-url-paths
        # If path does not come with a protocol, UPath(path) will return PosixUPath/WindowsUPath
        # while UPath("file://" + path) will return FilePath instance that supports the wb+ protocol.
        return UPath("file://" + path)

    upath_kwargs: dict[str, Any] = {}
    if protocol.startswith("s3"):
        # For S3 paths, test if matches the format of s3@<profile>://<path>
        if "@" in protocol:
            _, profile = protocol.split("@", 1)
            upath_kwargs["profile"] = profile
            upath_kwargs["default_block_size"] = 1024 * 1024 * s3_block_size_mb
            upath_kwargs["default_cache_type"] = s3_cache_type
            path = f"s3://{base_path}"

    # Otherwise, directly use upath to parse it
    return UPath(path, **upath_kwargs)


@contextmanager
def local_temp_file(universal_path: UPath) -> Generator[Path, None, None]:
    """
    Creates a temporary file from a universal path. This is a context manager that is
    supposed to be used as follows:

    with local_temp_file(universal_path) as temp_path:
        ...

    The temporary file is deleted after the context manager is exited.
    """

    assert universal_path.is_file(), f"Path {universal_path} is not a file"
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / universal_path.name
        with universal_path.open("rb") as fsrc, temp_path.open("wb") as fdst:
            shutil.copyfileobj(fsrc, fdst)
        yield temp_path
