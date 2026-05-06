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

# instant_nurec/model/<x>.py shims re-export from instant_nurec_internal.model.<x>
# during the move-to-internal transition (commits 2-7). The proprietary
# package lives in `<repo>/internal/`, which is excluded from the wheel
# (pyproject.toml:42-44). When installed as a wheel `instant_nurec_internal`
# is unavailable -- but so is the existing kelvin_full.pt pickle path that
# needs it. Commit 7 retires both, removing this hook.
import sys as _sys

from pathlib import Path as _Path


_INTERNAL_DIR = _Path(__file__).resolve().parent.parent / "internal"
if _INTERNAL_DIR.is_dir():
    _internal_str = str(_INTERNAL_DIR)
    if _internal_str not in _sys.path:
        _sys.path.insert(0, _internal_str)
