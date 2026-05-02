#!/usr/bin/env bash
# Standalone Kelvin predict — environment setup.
#
# Phase 3 transition: this script bootstraps a venv and installs the
# Python-side dependencies. The compiled libs/ kernels (slang/CUDA) are still
# built via bazel for the moment; ``bazel run //instant_nurec:run -- ...``
# remains the launcher path. ``run_inference.py`` shells through to the same
# CLI entrypoint.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [[ ! -d ".venv" ]]; then
    python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# Bring the venv to a baseline of Python deps the standalone needs at import
# time. ``torch`` installs CPU-only by default; install a matching CUDA build
# manually if you need GPU at the Python level.
python -m pip install --upgrade pip
python -m pip install \
    "numpy" \
    "plyfile" \
    "pillow" \
    "pyyaml" \
    "tqdm" \
    "pydantic" \
    "torch" \
    "pandas" \
    "zarr"

cat <<'EOF'

instant_nurec setup complete.

Activate the venv with:

    source .venv/bin/activate

Run inference with:

    ./run.sh --ncore-path /path/to/ncorev4 --output-dir /tmp/out --merge {none,frustum-ownership}

Or directly:

    python run_inference.py --ncore-path /path/to/ncorev4 --output-dir /tmp/out --merge {none,frustum-ownership}

The compiled libs/ kernels are currently built via bazel; the canonical
``run_inference.py`` form will be self-contained once Phase 2 finishes
replacing them with torch-native equivalents.
EOF
