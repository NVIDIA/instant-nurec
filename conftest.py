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

# pytest auto-loads this before collection so the in-tree packages resolve
# without an editable install. ``internal/`` is on the path so the tests
# under ``internal/tests/`` can import the proprietary-architecture package
# (``instant_nurec_internal``); the public ``instant_nurec`` package no
# longer depends on it -- it's referenced only by the artifact-export
# script and its tests.

import sys

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent

for _p in (_REPO_ROOT, _REPO_ROOT / "internal"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
