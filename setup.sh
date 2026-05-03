#!/usr/bin/env bash
# Standalone Kelvin predict — environment setup.
#
# Bootstraps a venv and ``pip install -e .``. All slang/CUDA kernels
# in ``libs/`` were replaced with pure-torch equivalents (Phase A);
# the only CUDA dependency left is whatever ``torch`` ships with the
# CUDA build you choose. ``run_inference.py`` is the canonical
# entrypoint.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [[ ! -d ".venv" ]]; then
    python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .

cat <<'EOF'

instant_nurec setup complete.

Activate the venv with:

    source .venv/bin/activate

Run inference with:

    ./run.sh --ncore-path /path/to/ncorev4 --output-dir /tmp/out --merge {none,frustum-ownership}

Or directly:

    python run_inference.py --ncore-path /path/to/ncorev4 --output-dir /tmp/out --merge {none,frustum-ownership}

If you need a CUDA torch build (the default install is CPU-only), reinstall
torch with the matching CUDA wheel after running this script:

    python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
EOF
