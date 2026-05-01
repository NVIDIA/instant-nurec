# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from upath import UPath


def parse_universal_path(path: str) -> UPath:
    """Parse a local path into a UPath object.

    Predict-only standalone never reads from S3 / s3@profile paths, so the
    s3-specific cache/block-size kwargs that NRE supported are gone.
    """
    try:
        path.split("://", 1)
    except ValueError:
        pass
    if "://" not in path:
        # https://github.com/fsspec/universal_pathlib?tab=readme-ov-file#local-paths-and-url-paths
        # Without a protocol prefix, UPath(path) returns PosixUPath/WindowsUPath
        # while UPath("file://" + path) returns a FilePath instance that supports
        # the wb+ protocol used by ncore writers.
        return UPath("file://" + path)
    return UPath(path)
