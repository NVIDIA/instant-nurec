#!/bin/bash

# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

# Wrapper to call Ruff via Bazel for both import sorting and formatting

# --- begin runfiles.bash initialization v3 ---
# Copy-pasted from the Bazel Bash runfiles library v3.
set -uo pipefail; set +e; f=bazel_tools/tools/bash/runfiles/runfiles.bash
# shellcheck disable=SC1090
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
source "$0.runfiles/$f" 2>/dev/null || \
source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
{ echo>&2 "ERROR: cannot find $f"; exit 1; }; f=; set -e
# --- end runfiles.bash initialization v3 ---

# Get the path to the Ruff binary from Bazel
RUFF_BIN="$(rlocation multitool/tools/ruff/ruff)"

# Get the workspace root
WORKSPACE_ROOT=$(bazel info workspace)

# We are called with either
# - check-mode: `format --check --force-exclude --diff FILES...` (check mode)
# or
# - fix-mode: `format --force-exclude FILES...`

# Parse arguments to distinguish between --check (diff mode) and normal (fix mode)
# and collect the files to process.
FILES=""
CHECK_ARG=""
for arg in "$@"; do
    if [ "$arg" = "--check" ]; then
        CHECK_ARG="--check"
    else
        abs_path="$WORKSPACE_ROOT/$arg"
        if [ -f "$abs_path" ]; then
            FILES="$FILES $abs_path"
        fi
    fi
done

if [ -n "$CHECK_ARG" ]; then
    # Check mode: show what would change + fail on diff
    set -e
    $RUFF_BIN check --select I --force-exclude --diff $FILES
    $RUFF_BIN format --check --force-exclude --diff $FILES
else
    # Fix mode: apply changes
    $RUFF_BIN check --select I --fix $FILES
    $RUFF_BIN format --force-exclude $FILES
fi
