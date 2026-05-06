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

"""Branch-coverage tests for ``KelvinStaticCore``.

Real ``KelvinDAv3Encoder`` / ``KelvinDPTDecoder`` instantiation requires a full
``KelvinModelConfig`` and runs only on GPU; that numerical-parity gate runs at
artifact-export time (commit 5). This file covers ``KelvinStaticCore``'s
orchestration and packaging logic with mock submodules so the wiring is
verified end-to-end on CPU.
"""

from __future__ import annotations

import sys

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "internal"))


from instant_nurec_internal.jit.kelvin_static_core import (  # noqa: E402
    KelvinStaticCore,
    StaticLayerTensors,
)
from instant_nurec_internal.model.backbone.base import KelvinMultiscaleFeaturesLatent  # noqa: E402


class _FakeStaticLayer:
    """``KelvinStaticLayer`` stand-in -- ``unpack_optional`` requires a non-None
    object and ``len(layer)`` plus ``layer.device()`` are the methods
    ``KelvinStaticCore.forward`` calls when filling absent ``semantic_class`` /
    ``normals`` fields with zeros.
    """

    def __init__(self, n: int, with_semantic: bool = True, with_normals: bool = True):
        self.positions = torch.arange(n * 3, dtype=torch.float32).reshape(n, 3)
        self.rotations = torch.zeros(n, 4)
        self.scales = torch.ones(n, 3)
        self.densities = torch.full((n, 1), 0.5)
        self.rgb = torch.full((n, 3), 0.7)
        self.semantic_class = (
            torch.zeros(n, 1, dtype=torch.uint8) if with_semantic else None
        )
        self.normals = torch.full((n, 3), 0.1) if with_normals else None
        self._n = n

    def __len__(self) -> int:
        return self._n

    def device(self) -> torch.device:
        return torch.device("cpu")


class _FakeEncoder(nn.Module):
    def __init__(self, B: int, V: int, h: int, w: int, C: int):
        super().__init__()
        self.B, self.V, self.h, self.w, self.C = B, V, h, w, C
        self.calls: list[tuple] = []

    def encode(self, batches, scene_rescale):
        self.calls.append((batches, scene_rescale))
        feats = [torch.arange(self.B * self.V * self.h * self.w * self.C, dtype=torch.float32).reshape(
            self.B, self.V, self.h, self.w, self.C
        )]
        return KelvinMultiscaleFeaturesLatent(features=feats)


class _FakeDecoder(nn.Module):
    def __init__(self, returns: list):
        super().__init__()
        self._returns = returns
        self.calls: list[tuple] = []

    def decode(self, encoded_latent, context, cuboid_tracks, time_remappings, scene_rescale):
        self.calls.append((encoded_latent, context, cuboid_tracks, time_remappings, scene_rescale))
        return self._returns


class _FakePostProcessing(nn.Module):
    def __init__(self, n_affine: int, channels: int):
        super().__init__()
        self.n = n_affine
        self.c = channels
        self.transform_calls: list[tuple] = []
        self.decode_calls: list = []

    def transform_tokens(self, x, camera_idxs):
        self.transform_calls.append((x, camera_idxs))
        B = x.shape[0]
        return x, torch.zeros(B, self.n, self.c)

    def decode_affine(self, x):
        self.decode_calls.append(x)
        B = x.shape[0]
        affine_3 = torch.arange(B * self.n * 9, dtype=torch.float32).reshape(B, self.n, 3, 3)
        bias = torch.arange(
            B * self.n * 9, B * self.n * 9 + B * self.n * 3, dtype=torch.float32
        ).reshape(B, self.n, 3)
        return affine_3, bias


def _make_context(num_views: int = 2):
    """Minimal ``DataAndRenderingBatch``-like object for ``_build_time_remappings``."""
    timestamps = torch.tensor(
        [[v_idx * 1_000_000, v_idx * 1_000_000 + 100_000] for v_idx in range(num_views)],
        dtype=torch.int64,
    )  # (V, 2)
    rendering_camera = SimpleNamespace(timestamps_startend_us_cpu=timestamps)
    rendering = SimpleNamespace(camera=rendering_camera)

    meta = [SimpleNamespace(unique_sensor_idx=v_idx) for v_idx in range(num_views)]
    data_camera = SimpleNamespace(meta=meta)
    data = SimpleNamespace(camera=data_camera)

    return SimpleNamespace(data=data, rendering=rendering)


# ---------- structural / wiring tests ----------


def test_forward_returns_one_bundle_per_batch():
    encoder = _FakeEncoder(B=2, V=3, h=4, w=5, C=8)
    decoder_returns = [
        SimpleNamespace(static_layer=_FakeStaticLayer(7), dynamic_layers=[None]),
        SimpleNamespace(static_layer=_FakeStaticLayer(11), dynamic_layers=[None]),
    ]
    decoder = _FakeDecoder(returns=decoder_returns)
    post = _FakePostProcessing(n_affine=3, channels=8)

    core = KelvinStaticCore(encoder, decoder, post, scene_rescale=1.0)
    context = [_make_context(num_views=3), _make_context(num_views=3)]

    bundles, affine = core.forward(context)

    assert len(bundles) == 2
    assert isinstance(bundles[0], StaticLayerTensors)
    assert bundles[0].positions.shape == (7, 3)
    assert bundles[1].positions.shape == (11, 3)


def test_forward_passes_scene_rescale_to_encoder():
    encoder = _FakeEncoder(B=1, V=2, h=2, w=2, C=4)
    decoder = _FakeDecoder(returns=[SimpleNamespace(static_layer=_FakeStaticLayer(1), dynamic_layers=[None])])
    post = _FakePostProcessing(n_affine=1, channels=4)

    core = KelvinStaticCore(encoder, decoder, post, scene_rescale=0.25)
    core.forward([_make_context(num_views=2)])

    assert encoder.calls[0][1] == 0.25


def test_forward_calls_decoder_with_cuboid_tracks_none():
    encoder = _FakeEncoder(B=1, V=2, h=2, w=2, C=4)
    decoder = _FakeDecoder(returns=[SimpleNamespace(static_layer=_FakeStaticLayer(1), dynamic_layers=[None])])
    post = _FakePostProcessing(n_affine=1, channels=4)

    core = KelvinStaticCore(encoder, decoder, post, scene_rescale=1.0)
    core.forward([_make_context(num_views=2)])

    _, _, cuboid_tracks_arg, _, _ = decoder.calls[0]
    assert cuboid_tracks_arg is None


def test_forward_affine_matrix_concats_bias_as_last_column():
    encoder = _FakeEncoder(B=1, V=2, h=2, w=2, C=4)
    decoder = _FakeDecoder(returns=[SimpleNamespace(static_layer=_FakeStaticLayer(1), dynamic_layers=[None])])
    post = _FakePostProcessing(n_affine=1, channels=4)

    core = KelvinStaticCore(encoder, decoder, post, scene_rescale=1.0)
    _, affine = core.forward([_make_context(num_views=2)])

    assert affine.shape == (1, 1, 3, 4)


def test_forward_zero_fills_missing_semantic_class():
    decoder_returns = [
        SimpleNamespace(
            static_layer=_FakeStaticLayer(4, with_semantic=False), dynamic_layers=[None]
        )
    ]
    encoder = _FakeEncoder(B=1, V=2, h=2, w=2, C=4)
    decoder = _FakeDecoder(returns=decoder_returns)
    post = _FakePostProcessing(n_affine=1, channels=4)

    core = KelvinStaticCore(encoder, decoder, post, scene_rescale=1.0)
    bundles, _ = core.forward([_make_context(num_views=2)])

    assert bundles[0].semantic_class.shape == (4, 1)
    assert bundles[0].semantic_class.dtype == torch.uint8
    assert torch.equal(bundles[0].semantic_class, torch.zeros(4, 1, dtype=torch.uint8))


def test_forward_zero_fills_missing_normals():
    decoder_returns = [
        SimpleNamespace(
            static_layer=_FakeStaticLayer(4, with_normals=False), dynamic_layers=[None]
        )
    ]
    encoder = _FakeEncoder(B=1, V=2, h=2, w=2, C=4)
    decoder = _FakeDecoder(returns=decoder_returns)
    post = _FakePostProcessing(n_affine=1, channels=4)

    core = KelvinStaticCore(encoder, decoder, post, scene_rescale=1.0)
    bundles, _ = core.forward([_make_context(num_views=2)])

    assert bundles[0].normals.shape == (4, 3)
    assert torch.equal(bundles[0].normals, torch.zeros(4, 3))


def test_forward_preserves_present_semantic_and_normals():
    sl = _FakeStaticLayer(4, with_semantic=True, with_normals=True)
    sl.semantic_class = torch.tensor([[1], [2], [3], [4]], dtype=torch.uint8)
    sl.normals = torch.tensor([[0.1, 0.2, 0.3]] * 4)
    decoder_returns = [SimpleNamespace(static_layer=sl, dynamic_layers=[None])]
    encoder = _FakeEncoder(B=1, V=2, h=2, w=2, C=4)
    decoder = _FakeDecoder(returns=decoder_returns)
    post = _FakePostProcessing(n_affine=1, channels=4)

    core = KelvinStaticCore(encoder, decoder, post, scene_rescale=1.0)
    bundles, _ = core.forward([_make_context(num_views=2)])

    assert torch.equal(bundles[0].semantic_class, sl.semantic_class)
    assert torch.equal(bundles[0].normals, sl.normals)


# ---------- _grab_camera_idxs ----------


def test_grab_camera_idxs_stacks_per_batch_indices():
    ctx_a = SimpleNamespace(
        data=SimpleNamespace(
            camera=SimpleNamespace(
                meta=[SimpleNamespace(unique_sensor_idx=0), SimpleNamespace(unique_sensor_idx=1)]
            )
        ),
        rendering=None,
    )
    ctx_b = SimpleNamespace(
        data=SimpleNamespace(
            camera=SimpleNamespace(
                meta=[SimpleNamespace(unique_sensor_idx=2), SimpleNamespace(unique_sensor_idx=3)]
            )
        ),
        rendering=None,
    )

    out = KelvinStaticCore._grab_camera_idxs([ctx_a, ctx_b])

    assert out.shape == (2, 2)
    assert out.dtype == torch.int64
    assert torch.equal(out, torch.tensor([[0, 1], [2, 3]], dtype=torch.int64))


# ---------- StaticLayerTensors dataclass ----------


def test_static_layer_tensors_field_order_matches_kelvin_static_layer():
    """Locks the field order of the JIT-export tensor bundle. Reordering or
    removing fields here breaks the export script (commit 5) and the loader
    adapter (commit 7), since both rely on positional/named matching with
    ``KelvinStaticLayer``'s constructor."""
    fields = StaticLayerTensors.__dataclass_fields__
    assert list(fields) == [
        "positions",
        "rotations",
        "scales",
        "densities",
        "rgb",
        "semantic_class",
        "normals",
    ]


# ---------- importability of MagicMock-style usage ----------


def test_module_does_not_import_kelvin_instant_nurec():
    """KelvinStaticCore must not depend on KelvinInstantNuRec -- the JIT
    artifact's load path stops importing the public class in commit 7. If
    a future edit reintroduces the import, this test trips so the
    architectural boundary stays explicit."""
    import re

    import instant_nurec_internal.jit.kelvin_static_core as ksc_mod

    src = Path(ksc_mod.__file__).read_text()
    # Only flag actual imports; mentions in docstrings/comments are fine.
    assert re.search(r"^\s*(from .+ import .*KelvinInstantNuRec|import .+KelvinInstantNuRec)", src, re.MULTILINE) is None


# ---------- guards against accidental dynamic/sky output ----------


def test_forward_returns_only_static_tensors_and_affine():
    encoder = _FakeEncoder(B=1, V=2, h=2, w=2, C=4)
    decoder = _FakeDecoder(returns=[SimpleNamespace(static_layer=_FakeStaticLayer(1), dynamic_layers=[None])])
    post = _FakePostProcessing(n_affine=1, channels=4)

    core = KelvinStaticCore(encoder, decoder, post, scene_rescale=1.0)
    out = core.forward([_make_context(num_views=2)])

    # The contract: a 2-tuple (bundles, affine), nothing else.
    assert len(out) == 2
    bundles, affine = out
    assert isinstance(bundles, list)
    assert isinstance(affine, torch.Tensor)
