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

"""Branch-coverage tests for ``KelvinStaticCore`` and
``TraceableStaticCore``.

Real-architecture instantiation needs a full ``KelvinModelConfig`` + GPU;
that numerical-parity gate runs at artifact-export time
(``internal/scripts/export_kelvin_jit.py``). This file covers the
class-level invariants on CPU.
"""

from __future__ import annotations

import re
import sys

from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "internal"))


from instant_nurec.primitives.kelvin_primitive import KelvinSemanticClass  # noqa: E402
from instant_nurec_internal.jit.kelvin_static_core import (  # noqa: E402
    KelvinStaticCore,
    TraceableStaticCore,
)


def _make_static_core() -> KelvinStaticCore:
    """Build a ``KelvinStaticCore`` with no-op nn.Module placeholders.

    Numerical correctness requires real encoder/decoder/post_processing,
    which are GPU-bound; these placeholders cover the structural
    invariants only (parameter ownership, attribute access)."""

    class _Stub(nn.Module):
        pass

    encoder = _Stub()
    decoder = _Stub()
    post = _Stub()
    return KelvinStaticCore(
        encoder,
        decoder,
        post,
        scene_rescale=0.5,
        expected_b=1,
        expected_v=18,
        expected_h=448,
        expected_w=784,
    )


# ---------- KelvinStaticCore structural invariants ----------


def test_static_core_registers_submodules():
    core = _make_static_core()
    submodule_names = {name for name, _ in core.named_modules()}
    assert "encoder" in submodule_names
    assert "decoder" in submodule_names
    assert "post_processing" in submodule_names


def test_static_core_stores_scene_rescale_unchanged():
    core = _make_static_core()
    assert core.scene_rescale == 0.5


def test_static_core_semantic_constants_match_kelvin_semantic_class():
    """The hard-coded class indices in KelvinStaticCore must stay in sync
    with KelvinSemanticClass; if someone adds a new class and the indices
    shift, this test trips before the JIT artifact is silently miscompiled."""
    assert KelvinStaticCore._SEMANTIC_EGO == KelvinSemanticClass.EGO.value
    assert KelvinStaticCore._SEMANTIC_SKY == KelvinSemanticClass.SKY.value
    assert KelvinStaticCore._SEMANTIC_MOVABLE == KelvinSemanticClass.MOVABLE.value


# ---------- TraceableStaticCore ----------


def test_traceable_wraps_static_core_forward_tensors():
    """TraceableStaticCore.forward must call into
    KelvinStaticCore.forward_tensors exactly once with the same arguments.
    Numerical correctness of forward_tensors is gated by the export-time
    parity check."""

    class _NopStaticCore(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls: list[tuple] = []

        def forward_tensors(self, *args):
            self.calls.append(args)
            return tuple(torch.zeros(1) for _ in range(8))

    nop = _NopStaticCore()
    wrap = TraceableStaticCore(nop)

    rgb = torch.zeros(1, 2, 4, 4, 3)
    c2w = torch.eye(4)[None, None].expand(1, 2, 4, 4)
    fov = torch.zeros(1, 2, 2)
    rays = torch.zeros(1, 2, 4, 4, 6)
    distance_to_depth_scale = torch.ones(1, 2, 4, 4, 1)
    camera_idxs = torch.zeros(1, 2, dtype=torch.int64)

    out = wrap(rgb, c2w, fov, rays, distance_to_depth_scale, camera_idxs)

    assert len(nop.calls) == 1
    args = nop.calls[0]
    assert args[0] is rgb
    assert args[1] is c2w
    assert args[5] is camera_idxs
    assert len(out) == 8


def test_traceable_module_registers_static_core_as_submodule():
    """For torch.jit.trace to find the parameter tree, static_core must be
    a registered submodule (not a closure attribute)."""

    class _MiniCore(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2)

        def forward_tensors(self, *args):
            return (self.linear(torch.zeros(1, 2)),) * 8

    core = _MiniCore()
    wrap = TraceableStaticCore(core)
    submodule_names = {name for name, _ in wrap.named_modules()}
    assert "static_core" in submodule_names
    assert "static_core.linear" in submodule_names


# ---------- module boundary guard ----------


def test_module_does_not_import_kelvin_instant_nurec():
    """KelvinStaticCore must not depend on KelvinInstantNuRec -- the JIT
    artifact's load path stops importing the public class in commit 7."""
    import instant_nurec_internal.jit.kelvin_static_core as ksc_mod

    src = Path(ksc_mod.__file__).read_text()
    assert (
        re.search(
            r"^\s*(from .+ import .*KelvinInstantNuRec|import .+KelvinInstantNuRec)",
            src,
            re.MULTILINE,
        )
        is None
    )
