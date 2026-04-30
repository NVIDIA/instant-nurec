#!/usr/bin/env python3
"""
Generate Baseline Coverage Data (LCOV format)

Creates an LCOV coverage file listing all source files (.py, .cpp, .cu, .cuh, .slang)
with DA:line,0 entries for executable lines only (skips blanks, comments,
preprocessor directives, brace-only lines, etc.).

When merged with actual coverage data, files not touched by any test appear
as 0% covered — giving a complete picture.

Use --exclude-from to skip files already present in actual coverage data,
preventing LF inflation for already-covered files.

Usage:
    python generate_baseline_coverage.py -s /path/to/workspace -o baseline_coverage.dat
    python generate_baseline_coverage.py -s /path/to/workspace -o baseline.dat --exclude-from actual.dat
"""

import argparse
import os
import sys

from typing import Set

from line_classifier import get_executable_line_numbers


# Source file extensions to include
DEFAULT_EXTENSIONS = frozenset((".py", ".cpp", ".cu", ".cuh", ".slang", ".c", ".h", ".hpp"))

# Directories to skip (matched against directory name, not full path)
EXCLUDE_DIRS = frozenset(
    (
        ".cache",
        ".git",
        ".test_cache",
        "__pycache__",
        "deps",
        "node_modules",
        "testlogs",
    )
)

# Directory name suffixes to skip (e.g., genhtml output dirs like "combined_coverage_html")
EXCLUDE_DIR_SUFFIXES = ("_html",)

# File names to exclude
EXCLUDE_FILES = frozenset(("__init__.py",))


def extract_covered_files(lcov_path: str) -> Set[str]:
    """Extract the set of SF: paths from an existing LCOV file (fast, line-scan only)."""
    covered = set()
    with open(lcov_path, "r") as f:
        for line in f:
            if line.startswith("SF:"):
                covered.add(line[3:].rstrip("\n\r"))
    return covered


def _read_lines(filepath: str) -> list:
    """Read file lines, returning empty list on error."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return f.readlines()
    except (OSError, IOError):
        return []


def generate_baseline(
    workspace_root: str,
    output_path: str,
    extensions: frozenset,
    exclude_files: Set[str],
) -> None:
    """Walk workspace, emit LCOV baseline with DA entries for executable lines only."""
    total_files = 0
    total_lines = 0
    root_len = len(workspace_root.rstrip(os.sep)) + 1  # for fast relative path slicing

    with open(output_path, "w", buffering=8192) as out:
        for dirpath, dirnames, filenames in os.walk(workspace_root):
            # Prune excluded directories in-place
            dirnames[:] = [
                d
                for d in dirnames
                if d not in EXCLUDE_DIRS and not d.startswith("bazel-") and not d.endswith(EXCLUDE_DIR_SUFFIXES)
            ]

            for filename in filenames:
                if filename in EXCLUDE_FILES:
                    continue
                # Check extension
                _, ext = os.path.splitext(filename)
                if ext not in extensions:
                    continue

                abs_path = os.path.join(dirpath, filename)

                # Skip symlinks: prevents duplicate entries (symlink + real file)
                # and invalid SF: paths for symlinks pointing outside the workspace
                if os.path.islink(abs_path):
                    continue

                rel_path = abs_path[root_len:]  # fast slice instead of Path.relative_to

                # Skip files already covered
                if rel_path in exclude_files:
                    continue

                source_lines = _read_lines(abs_path)
                if not source_lines:
                    continue

                executable_lines = get_executable_line_numbers(source_lines, file_ext=ext)
                if not executable_lines:
                    continue

                num_executable = len(executable_lines)

                out.write("TN:\n")
                out.write(f"SF:{rel_path}\n")
                for line_num in sorted(executable_lines):
                    out.write(f"DA:{line_num},0\n")
                out.write(f"LF:{num_executable}\n")
                out.write("LH:0\n")
                out.write("end_of_record\n")

                total_files += 1
                total_lines += num_executable

    print(f"Baseline coverage: {total_files} files, {total_lines} executable lines")
    print(f"Output: {output_path}")
    if exclude_files:
        print(f"Excluded {len(exclude_files)} already-covered files")


def main():
    parser = argparse.ArgumentParser(
        description="Generate baseline LCOV coverage data with zero counts for all source files.",
    )
    parser.add_argument(
        "-s",
        "--source-dir",
        required=True,
        help="Workspace root directory to scan for source files",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="baseline_coverage.dat",
        help="Output LCOV file path (default: baseline_coverage.dat)",
    )
    parser.add_argument(
        "--exclude-from",
        default=None,
        dest="exclude_from",
        help="LCOV file whose SF: entries should be excluded from baseline "
        "(use to avoid inflating LF for already-covered files)",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=None,
        help=f"File extensions to include (default: {' '.join(sorted(DEFAULT_EXTENSIONS))})",
    )
    args = parser.parse_args()

    workspace_root = os.path.realpath(args.source_dir)
    if not os.path.isdir(workspace_root):
        print(f"ERROR: Source directory does not exist: {workspace_root}", file=sys.stderr)
        sys.exit(1)

    extensions = DEFAULT_EXTENSIONS
    if args.extensions:
        extensions = frozenset(ext if ext.startswith(".") else f".{ext}" for ext in args.extensions)

    exclude_files: Set[str] = set()
    if args.exclude_from:
        if not os.path.isfile(args.exclude_from):
            print(f"WARNING: --exclude-from file not found: {args.exclude_from}", file=sys.stderr)
        else:
            exclude_files = extract_covered_files(args.exclude_from)
            print(f"Loaded {len(exclude_files)} file paths to exclude from {args.exclude_from}")

    print(f"Scanning {workspace_root} for {', '.join(sorted(extensions))}")

    generate_baseline(workspace_root, args.output, extensions, exclude_files)


if __name__ == "__main__":
    main()
