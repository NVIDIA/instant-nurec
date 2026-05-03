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

#!/usr/bin/env python3
"""Standalone Kelvin predict entrypoint.

Invocation::

    python run_inference.py --ncore-path <path> --output-dir <path> --merge {none,frustum-ownership}

Phase 3 step 8.4: this is the canonical invocation. The Phase 1 ``bazel run
//instant_nurec:run -- ...`` form remains supported through Phase 3
transition; ``run_inference.py`` defers to ``instant_nurec.cli.main`` and so
runs whichever tree is currently importable on ``sys.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))


from instant_nurec.cli import main


if __name__ == "__main__":
    sys.exit(main())
