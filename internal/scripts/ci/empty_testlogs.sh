#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Empty bazel-testlogs folder to ensure fresh data after a test run.
# If the deleted content was still up-to-date, it will be quickly restored from
# disk cache by Bazel on later test runs.
#
# Usage:
#   empty_testlogs.sh <testlogs_dir>

set -euo pipefail

TESTLOGS_DIR="$1"

if [ -z "$TESTLOGS_DIR" ]; then
    echo "Usage: empty_testlogs.sh <testlogs_dir>" >&2
    exit 1
fi

# Refuse to operate on root
if [ "$TESTLOGS_DIR" = "/" ]; then
    echo "Refusing to empty root directory" >&2
    exit 1
fi

if [ -d "$TESTLOGS_DIR" ]; then
    # Bazel marks 'test.outputs' directories and children as read-only (possibly
    # a bug), so we fix permissions before deleting.
    find -L "$TESTLOGS_DIR" -type d -name "test.outputs" -exec chmod -R u+w {} +
    rm -rf "$TESTLOGS_DIR"/*
    echo "Emptied test logs directory: $TESTLOGS_DIR"
else
    echo "Test logs directory not found: $TESTLOGS_DIR"
fi
