#!/usr/bin/env bash
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
    echo "usage: $0 --ncore-path <path> --output-dir <path> [--merge {none,frustum-ownership}] [--log-level ...]" >&2
    exit 64
fi

if [[ ! -d "$NCORE_PATH" ]]; then
    echo "error: --ncore-path '$NCORE_PATH' does not exist or is not a directory" >&2
    exit 65
fi

mkdir -p "$OUTPUT_DIR"
if [[ ! -w "$OUTPUT_DIR" ]]; then
    echo "error: --output-dir '$OUTPUT_DIR' is not writable" >&2
    exit 73
fi

exec python run_inference.py "${PASSTHROUGH[@]}"
