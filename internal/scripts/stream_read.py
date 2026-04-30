# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
HTTP utilities for secure response handling.

Provides streaming read functionality to safely handle large HTTP responses
while protecting against memory exhaustion attacks (CVE-2025-13836).
"""

# Streaming read settings to handle large responses safely (CVE-2025-13836)
URLLIB_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB per chunk
URLLIB_MAX_SIZE = 1024 * 1024 * 1024  # 1 GiB maximum total


def stream_read(response, max_size=URLLIB_MAX_SIZE, chunk_size=URLLIB_CHUNK_SIZE):
    """
    Read response data in chunks, rejecting responses over max_size.

    Args:
        response: HTTP response object with a read() method
        max_size: Maximum allowed total size (default 1 GiB)
        chunk_size: Size of each chunk to read (default 8 MiB)

    Returns:
        bytes: Complete response data

    Raises:
        ValueError: If response exceeds max_size
    """
    chunks = []
    total_size = 0
    while True:
        chunk = response.read(chunk_size)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > max_size:
            raise ValueError(f"Response size exceeds maximum allowed ({max_size} bytes)")
        chunks.append(chunk)
    return b"".join(chunks)
