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

from upath import UPath


def parse_universal_path(path: str) -> UPath:
    """Parse a local path into a UPath object."""
    if "://" not in path:
        # https://github.com/fsspec/universal_pathlib?tab=readme-ov-file#local-paths-and-url-paths
        # Without a protocol prefix, UPath(path) returns PosixUPath/WindowsUPath
        # while UPath("file://" + path) returns a FilePath instance that supports
        # the wb+ protocol used by ncore writers.
        return UPath("file://" + path)
    return UPath(path)
