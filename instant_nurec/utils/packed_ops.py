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

from __future__ import annotations

import torch

from instant_nurec.utils._packed_ops_torch import (
    linstep_interleave as _linstep_interleave_torch,
)


@torch.no_grad()
def linstep_interleave(
    start: torch.Tensor,
    num_steps: torch.Tensor,
    step_size: torch.Tensor | int | float,
) -> torch.Tensor:
    """Returns interleaved per-pack arange-style sequences, one step_size apart.

    The bazel kernel also accepted a ``return_idx`` flag that returned a
    per-element pack-index tensor; the standalone never read that tensor
    so the parameter and the wrapper's ``ValuesAndPidx`` dataclass were
    deleted (final dead-code pass).
    """
    if start.numel() == 0:
        return torch.empty_like(start)
    return _linstep_interleave_torch(
        start.contiguous(),
        num_steps.contiguous().long(),
        step_size.contiguous() if isinstance(step_size, torch.Tensor) else step_size,
    )
