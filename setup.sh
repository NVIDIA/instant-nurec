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

# InstantNuRec Kelvin predict — environment setup.
#
# Bootstraps a venv and installs the locked dependency tree via ``uv sync``.
# The default workflow does not install the optional gsplat reference renderer;
# use ``uv sync --extra render`` before passing ``--render-preview`` or
# ``--render-video``.
# ``run_inference.py`` is the canonical entrypoint.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install it first:" >&2
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    echo "  or" >&2
    echo "    pip install uv" >&2
    exit 1
fi

# ``uv sync --frozen`` installs exactly what ``uv.lock`` records, fails if
# the lock and ``pyproject.toml`` disagree. This is the right gate for
# reproducibility — to update deps, run ``uv lock`` and commit the new lock.
uv sync --frozen

cat <<'EOF'

InstantNuRec setup complete.

Activate the venv with:

    source .venv/bin/activate

Run inference with:

    ./run.sh --ncore-path /path/to/sequence.json --output-dir /tmp/out [--merge] [--n-gaussians N]

Or directly:

    python run_inference.py --ncore-path /path/to/sequence.json --output-dir /tmp/out [--merge] [--n-gaussians N]

Each PLY also has .sky.npz and .sky.png outputs. To additionally render the
first context frame with the sky composited behind the Gaussians:

    uv sync --extra render
    ./run.sh --ncore-path /path/to/sequence.json --output-dir /tmp/out --render-preview

To render every original frame along the calibrated source trajectory (ffmpeg
with libx264 must be installed):

    ./run.sh --ncore-path /path/to/sequence.json --output-dir /tmp/out --merge --render-video
EOF
