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
# Bootstraps a venv and ``pip install -e .``. The only CUDA dependency
# is whatever ``torch`` ships with the CUDA build you choose.
# ``run_inference.py`` is the canonical entrypoint.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [[ ! -d ".venv" ]]; then
    python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip

# Install the pinned CUDA torch build first (the cu128 wheel is published
# on PyTorch's own index, not pypi.org). pyproject.toml pins
# ``torch==2.7.0+cu128`` so this index is required during install.
python -m pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --index-url https://download.pytorch.org/whl/cu128

python -m pip install -e .

cat <<'EOF'

instant_nurec setup complete.

Activate the venv with:

    source .venv/bin/activate

Run inference with:

    ./run.sh --ncore-path /path/to/ncorev4 --output-dir /tmp/out --merge {none,frustum-ownership}

Or directly:

    python run_inference.py --ncore-path /path/to/ncorev4 --output-dir /tmp/out --merge {none,frustum-ownership}
EOF
