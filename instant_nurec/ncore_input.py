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

"""Resolve ``--ncore-path`` into a flat list of absolute, validated JSON paths.

Two input shapes are supported:

* ``.json`` — a single ncorev4 sequence metadata file (NuRec-aligned).
* ``.lst`` — one path per line, each pointing at a sequence ``.json``.
  Lines may be absolute or relative-to-the-LST-file's directory.
  ``#``-prefixed and blank lines are skipped.
"""

from __future__ import annotations

from pathlib import Path


def resolve_ncore_paths(arg: Path) -> list[Path]:
    """Resolve ``arg`` into a flat list of absolute, existing JSON paths.

    Raises ``ValueError`` for any unrecognised suffix, missing file,
    non-JSON LST entry, or empty LST.
    """
    suffix = arg.suffix
    if suffix == ".json":
        resolved = arg.expanduser().resolve()
        if not resolved.is_file():
            raise ValueError(f"--ncore-path {arg!s}: not an existing JSON file")
        return [resolved]

    if suffix == ".lst":
        lst = arg.expanduser().resolve()
        if not lst.is_file():
            raise ValueError(f"--ncore-path {arg!s}: not an existing LST file")
        lst_dir = lst.parent
        out: list[Path] = []
        for lineno, raw in enumerate(lst.read_text().splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # 1. expand ~ (so ~/foo becomes /home/user/foo and now counts as absolute)
            p = Path(line).expanduser()
            # 2. anchor relative paths to the LST file's directory
            if not p.is_absolute():
                p = lst_dir / p
            # 3. canonicalize (follows symlinks; collapses ./ and ../)
            p = p.resolve()
            # 4. validate type + existence
            if p.suffix != ".json":
                raise ValueError(
                    f"--ncore-path {arg!s} line {lineno}: "
                    f"{raw!r} → {p!s} is not a .json file"
                )
            if not p.is_file():
                raise ValueError(
                    f"--ncore-path {arg!s} line {lineno}: "
                    f"{raw!r} → {p!s} does not exist"
                )
            out.append(p)
        if not out:
            raise ValueError(f"--ncore-path {arg!s}: LST file is empty")
        return out

    raise ValueError(f"--ncore-path must end in .json or .lst, got: {arg!s}")
