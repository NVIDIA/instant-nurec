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

"""Branch-coverage tests for instant_nurec.utils.files.parse_universal_path."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from instant_nurec.utils.files import parse_universal_path


def test_local_absolute_path_gets_file_protocol_prefix():
    out = parse_universal_path("/tmp/foo")
    # The "file://" prefix is the load-bearing detail: ncore writers need
    # FilePath (via the file:// scheme) for the wb+ protocol.
    assert str(out) == "file:///tmp/foo"


def test_local_relative_path_gets_file_protocol_prefix_and_is_resolved():
    """Relative paths get file:// prefix; UPath then resolves to abs CWD path."""
    out = parse_universal_path("relative/path/foo.json")
    s = str(out)
    assert s.startswith("file:///")
    assert s.endswith("/relative/path/foo.json")


def test_s3_path_passes_through_unchanged():
    out = parse_universal_path("s3://bucket/key")
    assert str(out) == "s3://bucket/key"


def test_https_path_passes_through_unchanged():
    out = parse_universal_path("https://example.com/foo")
    assert str(out) == "https://example.com/foo"


def test_explicit_file_path_passes_through_unchanged():
    """A path that already has file:// shouldn't get a second protocol prefix."""
    out = parse_universal_path("file:///tmp/foo")
    assert str(out) == "file:///tmp/foo"


def test_empty_string_gets_file_protocol_prefix():
    """Empty string takes the no-protocol branch, gets file:// prefix; UPath
    treats `file://` as the CWD."""
    out = parse_universal_path("")
    assert str(out).startswith("file:///")


def test_returned_object_is_a_upath_instance():
    from upath import UPath

    out = parse_universal_path("/tmp/foo")
    assert isinstance(out, UPath)
