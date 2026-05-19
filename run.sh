#!/usr/bin/env bash

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

# Thin wrapper around ``python run_inference.py`` that validates the user's
# ``--ncore-path`` and ``--output-dir`` arguments before invoking the CLI.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

NCORE_PATH=""
OUTPUT_DIR=""
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ncore-path)
            NCORE_PATH="$2"
            PASSTHROUGH+=("$1" "$2")
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            PASSTHROUGH+=("$1" "$2")
            shift 2
            ;;
        *)
            PASSTHROUGH+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$NCORE_PATH" ]] || [[ -z "$OUTPUT_DIR" ]]; then
    echo "usage: $0 --ncore-path <path> --output-dir <path> [--merge] [--n-gaussians N] [--log-level ...]" >&2
    exit 64
fi

if [[ ! -f "$NCORE_PATH" ]]; then
    echo "error: --ncore-path '$NCORE_PATH' does not exist or is not a file" >&2
    exit 65
fi
case "$NCORE_PATH" in
    *.json|*.lst) ;;
    *)
        echo "error: --ncore-path '$NCORE_PATH' must end in .json or .lst" >&2
        exit 65
        ;;
esac

mkdir -p "$OUTPUT_DIR"
if [[ ! -w "$OUTPUT_DIR" ]]; then
    echo "error: --output-dir '$OUTPUT_DIR' is not writable" >&2
    exit 73
fi

exec python run_inference.py "${PASSTHROUGH[@]}"
