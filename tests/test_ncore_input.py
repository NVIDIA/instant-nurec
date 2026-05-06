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

"""Branch-coverage tests for ``instant_nurec.ncore_input.resolve_ncore_paths``.

Covers the three top-level branches (``.json``, ``.lst``, anything else)
plus every per-line branch in the ``.lst`` parser:

  * absolute path
  * relative path (resolved against LST-file's directory)
  * ``~``-prefixed path (resolved against ``$HOME``)
  * blank line skipped
  * ``#``-prefixed comment line skipped
  * mix of absolute + relative + comment + blank in one LST
  * non-``.json`` line  → ValueError
  * missing-file line   → ValueError
  * empty LST           → ValueError
  * missing JSON arg    → ValueError
  * missing LST arg     → ValueError
  * unrecognised suffix → ValueError
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


from instant_nurec.ncore_input import resolve_ncore_paths  # noqa: E402


# ---------------------------------------------------------------------------
# .json single-sequence branch
# ---------------------------------------------------------------------------


def test_single_json_returns_one_resolved_path(tmp_path: Path) -> None:
    json_file = tmp_path / "seq.json"
    json_file.write_text("{}")
    out = resolve_ncore_paths(json_file)
    assert out == [json_file.resolve()]


def test_single_json_resolves_relative_input(tmp_path: Path, monkeypatch) -> None:
    """A relative ``.json`` path is resolved (anchored to cwd)."""
    json_file = tmp_path / "seq.json"
    json_file.write_text("{}")
    monkeypatch.chdir(tmp_path)
    out = resolve_ncore_paths(Path("seq.json"))
    assert out == [json_file.resolve()]
    assert out[0].is_absolute()


def test_single_json_expands_tilde(tmp_path: Path, monkeypatch) -> None:
    """``~/seq.json`` expands to the user's home dir before validation."""
    monkeypatch.setenv("HOME", str(tmp_path))
    json_file = tmp_path / "seq.json"
    json_file.write_text("{}")
    out = resolve_ncore_paths(Path("~/seq.json"))
    assert out == [json_file.resolve()]


def test_single_json_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not an existing JSON file"):
        resolve_ncore_paths(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# .lst multi-sequence branch — successful per-line resolutions
# ---------------------------------------------------------------------------


def test_lst_with_absolute_paths(tmp_path: Path) -> None:
    """Lines that are absolute paths are used as-is."""
    a = tmp_path / "a.json"
    b = tmp_path / "sub" / "b.json"
    a.write_text("{}")
    b.parent.mkdir()
    b.write_text("{}")

    lst = tmp_path / "manifest.lst"
    lst.write_text(f"{a}\n{b}\n")

    out = resolve_ncore_paths(lst)
    assert out == [a.resolve(), b.resolve()]


def test_lst_with_relative_paths_anchored_to_lst_dir(tmp_path: Path) -> None:
    """Relative lines anchor to the LST file's directory, not cwd."""
    lst_dir = tmp_path / "manifests"
    lst_dir.mkdir()
    a = lst_dir / "seqs" / "a.json"
    b = lst_dir / "seqs" / "b.json"
    a.parent.mkdir()
    a.write_text("{}")
    b.write_text("{}")

    lst = lst_dir / "all.lst"
    lst.write_text("seqs/a.json\nseqs/b.json\n")

    out = resolve_ncore_paths(lst)
    assert out == [a.resolve(), b.resolve()]


def test_lst_with_dotdot_relative_paths(tmp_path: Path) -> None:
    """Relative lines with ``..`` traverse correctly."""
    lst_dir = tmp_path / "lst-dir"
    lst_dir.mkdir()
    other = tmp_path / "elsewhere" / "seq.json"
    other.parent.mkdir()
    other.write_text("{}")

    lst = lst_dir / "all.lst"
    lst.write_text("../elsewhere/seq.json\n")

    out = resolve_ncore_paths(lst)
    assert out == [other.resolve()]


def test_lst_with_tilde_path_expands(tmp_path: Path, monkeypatch) -> None:
    """``~``-prefixed lines expand to ``$HOME`` (and become absolute)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    seq = tmp_path / "seq.json"
    seq.write_text("{}")

    lst = tmp_path / "all.lst"
    lst.write_text("~/seq.json\n")

    out = resolve_ncore_paths(lst)
    assert out == [seq.resolve()]


def test_lst_mixes_absolute_and_relative_paths(tmp_path: Path) -> None:
    """Per-line dispatch: absolute lines are kept, relative lines anchored."""
    abs_seq = tmp_path / "outside" / "abs.json"
    abs_seq.parent.mkdir()
    abs_seq.write_text("{}")

    lst_dir = tmp_path / "manifests"
    lst_dir.mkdir()
    rel_seq = lst_dir / "rel.json"
    rel_seq.write_text("{}")

    lst = lst_dir / "all.lst"
    lst.write_text(f"{abs_seq}\nrel.json\n")

    out = resolve_ncore_paths(lst)
    assert out == [abs_seq.resolve(), rel_seq.resolve()]


def test_lst_skips_blank_and_comment_lines(tmp_path: Path) -> None:
    """Blank lines and ``#``-prefixed lines are skipped."""
    seq = tmp_path / "seq.json"
    seq.write_text("{}")

    lst = tmp_path / "manifest.lst"
    lst.write_text(
        "# header comment\n"
        "\n"
        "  \n"  # whitespace-only
        f"{seq}\n"
        "# trailing comment\n"
    )

    out = resolve_ncore_paths(lst)
    assert out == [seq.resolve()]


def test_lst_strips_surrounding_whitespace(tmp_path: Path) -> None:
    """Leading/trailing whitespace on a path line is stripped."""
    seq = tmp_path / "seq.json"
    seq.write_text("{}")

    lst = tmp_path / "manifest.lst"
    lst.write_text(f"   {seq}   \n")

    out = resolve_ncore_paths(lst)
    assert out == [seq.resolve()]


def test_lst_passes_through_resolved_symlinks(tmp_path: Path) -> None:
    """Symlinks are resolved to canonical paths."""
    real = tmp_path / "real.json"
    real.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(real)

    lst = tmp_path / "all.lst"
    lst.write_text(f"{link}\n")

    out = resolve_ncore_paths(lst)
    assert out == [real.resolve()]


# ---------------------------------------------------------------------------
# .lst — error branches
# ---------------------------------------------------------------------------


def test_lst_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not an existing LST file"):
        resolve_ncore_paths(tmp_path / "nope.lst")


def test_lst_empty_after_stripping_comments_raises(tmp_path: Path) -> None:
    """A LST with only blank/comment lines is treated as empty."""
    lst = tmp_path / "all.lst"
    lst.write_text("# only a comment\n\n  \n")

    with pytest.raises(ValueError, match="LST file is empty"):
        resolve_ncore_paths(lst)


def test_lst_completely_empty_raises(tmp_path: Path) -> None:
    lst = tmp_path / "all.lst"
    lst.write_text("")

    with pytest.raises(ValueError, match="LST file is empty"):
        resolve_ncore_paths(lst)


def test_lst_line_with_non_json_suffix_raises(tmp_path: Path) -> None:
    txt = tmp_path / "wrong.txt"
    txt.write_text("not json")

    lst = tmp_path / "all.lst"
    lst.write_text(f"{txt}\n")

    with pytest.raises(ValueError, match="line 1.*not a .json file"):
        resolve_ncore_paths(lst)


def test_lst_line_pointing_to_missing_json_raises(tmp_path: Path) -> None:
    lst = tmp_path / "all.lst"
    lst.write_text(f"{tmp_path / 'absent.json'}\n")

    with pytest.raises(ValueError, match="line 1.*does not exist"):
        resolve_ncore_paths(lst)


def test_lst_error_reports_the_failing_line_number(tmp_path: Path) -> None:
    """A bad entry on line 3 mentions ``line 3`` in the error."""
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text("{}")
    b.write_text("{}")

    lst = tmp_path / "all.lst"
    lst.write_text(
        "# comment\n"
        f"{a}\n"
        f"{tmp_path / 'gone.json'}\n"  # line 3 — the bad one
        f"{b}\n"
    )

    with pytest.raises(ValueError, match="line 3.*does not exist"):
        resolve_ncore_paths(lst)


# ---------------------------------------------------------------------------
# Top-level error branch — unsupported suffix
# ---------------------------------------------------------------------------


def test_unrecognised_suffix_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "data.yaml"
    bogus.write_text("{}")
    with pytest.raises(ValueError, match="must end in .json or .lst"):
        resolve_ncore_paths(bogus)


def test_directory_path_raises(tmp_path: Path) -> None:
    """Directories are no longer auto-detected; user must pass the LST file directly."""
    (tmp_path / "debug.lst").write_text("\n")  # exists, but we passed the dir, not the LST
    with pytest.raises(ValueError, match="must end in .json or .lst"):
        resolve_ncore_paths(tmp_path)


def test_no_suffix_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "noextension"
    bogus.write_text("anything")
    with pytest.raises(ValueError, match="must end in .json or .lst"):
        resolve_ncore_paths(bogus)
