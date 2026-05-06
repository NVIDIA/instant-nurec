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

"""Branch-coverage tests for ``KelvinInstantNuRec._compute_affine_matrix`` and
``KelvinInstantNuRec._build_primitives`` -- the pure-Python helpers extracted
out of ``reconstruct()`` to set up the JIT boundary in follow-up commits.

Both helpers are exercised here without instantiating a full
``KelvinInstantNuRec`` (which would pull in the GPU-only encoder/decoder
stack); the methods are called via ``KelvinInstantNuRec.<method>`` with a
stub ``self`` for the instance-level case.
"""

from __future__ import annotations

import sys

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


from instant_nurec_internal.model.backbone.base import KelvinMultiscaleFeaturesLatent  # noqa: E402
from instant_nurec_internal.model.kelvin import KelvinInstantNuRec  # noqa: E402
from instant_nurec.primitives.kelvin_primitive import (  # noqa: E402
    KelvinDynamicLayer,
    KelvinStaticLayer,
)


def _make_static_layer(n: int) -> KelvinStaticLayer:
    return KelvinStaticLayer(
        positions=torch.zeros(n, 3),
        densities=torch.zeros(n, 1),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n, 4).contiguous(),
        scales=torch.ones(n, 3),
        rgb=torch.zeros(n, 3),
    )


def _make_dynamic_layer(n: int) -> KelvinDynamicLayer:
    return KelvinDynamicLayer(
        max_densities=torch.zeros(n, 1),
        keyframe_positions=torch.zeros(n, 3, 3),
        keyframe_timestamps_us=torch.zeros(n, 3),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n, 4).contiguous(),
        scales=torch.ones(n, 3),
        rgb=torch.zeros(n, 3),
    )


# ---------- _compute_affine_matrix ----------


class _AffineStub:
    """Minimal stand-in for ``self.post_processing`` -- captures the inputs
    handed to ``transform_tokens`` / ``decode_affine`` and returns canned
    outputs so the helper's only observable behavior (concat order + last-axis
    expansion of bias) can be asserted directly.
    """

    def __init__(self, n_affine_tokens: int, channels: int):
        self.n = n_affine_tokens
        self.c = channels
        self.transform_called_with: tuple | None = None
        self.decode_called_with: torch.Tensor | None = None

    def transform_tokens(self, x: torch.Tensor, camera_idxs: torch.Tensor):
        self.transform_called_with = (x, camera_idxs)
        B = x.shape[0]
        # match the documented (B, n, C) shape of the second return
        affine_token = torch.zeros(B, self.n, self.c)
        return x, affine_token

    def decode_affine(self, x: torch.Tensor):
        self.decode_called_with = x
        B = x.shape[0]
        # canned 3x3 + 3 outputs whose values let us verify the cat ordering
        affine_matrix_3 = torch.arange(B * self.n * 9, dtype=torch.float32).reshape(B, self.n, 3, 3)
        affine_bias = torch.arange(
            B * self.n * 9, B * self.n * 9 + B * self.n * 3, dtype=torch.float32
        ).reshape(B, self.n, 3)
        return affine_matrix_3, affine_bias


def test_compute_affine_matrix_concats_bias_as_last_column():
    B, V, h, w, C = 2, 3, 4, 5, 8
    encoded_latent = KelvinMultiscaleFeaturesLatent(features=[torch.zeros(B, V, h, w, C)])
    camera_idxs = torch.zeros(B, V, dtype=torch.int64)
    stub_self = SimpleNamespace(post_processing=_AffineStub(n_affine_tokens=3, channels=C))

    out = KelvinInstantNuRec._compute_affine_matrix(stub_self, encoded_latent, camera_idxs)

    assert out.shape == (B, 3, 3, 4)
    # Last column must equal the bias values supplied by the stub.
    expected_bias = stub_self.post_processing.decode_called_with  # x captured pre-cat
    assert expected_bias is not None  # decode_affine was invoked
    # The columns 0..2 are the 3x3 matrix, column 3 is the bias.
    matrix_part, bias_part = out[..., :3], out[..., 3]
    n = stub_self.post_processing.n
    expected_matrix = torch.arange(B * n * 9, dtype=torch.float32).reshape(B, n, 3, 3)
    expected_bias_values = torch.arange(
        B * n * 9, B * n * 9 + B * n * 3, dtype=torch.float32
    ).reshape(B, n, 3)
    assert torch.equal(matrix_part, expected_matrix)
    assert torch.equal(bias_part, expected_bias_values)


def test_compute_affine_matrix_passes_camera_idxs_through():
    B, V, h, w, C = 1, 2, 1, 1, 4
    encoded_latent = KelvinMultiscaleFeaturesLatent(features=[torch.zeros(B, V, h, w, C)])
    camera_idxs = torch.tensor([[7, 11]], dtype=torch.int64)
    stub_self = SimpleNamespace(post_processing=_AffineStub(n_affine_tokens=1, channels=C))

    KelvinInstantNuRec._compute_affine_matrix(stub_self, encoded_latent, camera_idxs)

    assert stub_self.post_processing.transform_called_with is not None
    _, passed_idxs = stub_self.post_processing.transform_called_with
    assert torch.equal(passed_idxs, camera_idxs)


def test_compute_affine_matrix_rearranges_deepest_to_three_axes():
    B, V, h, w, C = 2, 3, 4, 5, 6
    encoded_latent = KelvinMultiscaleFeaturesLatent(features=[torch.zeros(B, V, h, w, C)])
    camera_idxs = torch.zeros(B, V, dtype=torch.int64)
    stub_self = SimpleNamespace(post_processing=_AffineStub(n_affine_tokens=1, channels=C))

    KelvinInstantNuRec._compute_affine_matrix(stub_self, encoded_latent, camera_idxs)

    passed_x, _ = stub_self.post_processing.transform_called_with
    # rearrange "B V h w C -> B (V h w) C" should collapse spatial+view axes
    assert passed_x.shape == (B, V * h * w, C)


# ---------- _build_primitives ----------


def _make_decoder_return(static_n: int, dynamic_n: int):
    return SimpleNamespace(
        static_layer=_make_static_layer(static_n),
        dynamic_layers=[_make_dynamic_layer(dynamic_n)],
    )


def test_build_primitives_empty_context_returns_empty_list():
    sky = torch.zeros(0, 6, 4, 4, 3)
    affine = torch.zeros(0, 1, 3, 4)

    out = KelvinInstantNuRec._build_primitives(
        context=[], decoder_returns=[], sky_cubemaps=sky, affine_matrix=affine
    )

    assert out == []


def test_build_primitives_single_batch_indexes_correctly():
    cubemap_size = 4
    n_cameras = 2
    sky = torch.arange(6 * cubemap_size * cubemap_size * 3, dtype=torch.float32).reshape(
        1, 6, cubemap_size, cubemap_size, 3
    )
    affine = torch.arange(n_cameras * 3 * 4, dtype=torch.float32).reshape(1, n_cameras, 3, 4)
    decoder_returns = [_make_decoder_return(static_n=5, dynamic_n=2)]

    out = KelvinInstantNuRec._build_primitives(
        context=[None], decoder_returns=decoder_returns, sky_cubemaps=sky, affine_matrix=affine
    )

    assert len(out) == 1
    p = out[0]
    assert torch.equal(p.sky_cubemap, sky[0])
    assert torch.equal(p.affine_matrix, affine[0])
    assert p.static_layer is decoder_returns[0].static_layer
    assert p.dynamic_layers is decoder_returns[0].dynamic_layers


def test_build_primitives_multi_batch_preserves_per_batch_slicing():
    B = 3
    cubemap_size = 2
    n_cameras = 1
    sky = torch.randn(B, 6, cubemap_size, cubemap_size, 3)
    affine = torch.randn(B, n_cameras, 3, 4)
    decoder_returns = [_make_decoder_return(static_n=i + 1, dynamic_n=1) for i in range(B)]

    out = KelvinInstantNuRec._build_primitives(
        context=[None] * B, decoder_returns=decoder_returns, sky_cubemaps=sky, affine_matrix=affine
    )

    assert len(out) == B
    for bidx, primitive in enumerate(out):
        assert torch.equal(primitive.sky_cubemap, sky[bidx])
        assert torch.equal(primitive.affine_matrix, affine[bidx])
        assert primitive.static_layer is decoder_returns[bidx].static_layer


def test_build_primitives_raises_on_none_static_layer():
    sky = torch.zeros(1, 6, 4, 4, 3)
    affine = torch.zeros(1, 1, 3, 4)
    decoder_returns = [SimpleNamespace(static_layer=None, dynamic_layers=[_make_dynamic_layer(1)])]

    with pytest.raises(ValueError, match="empty optional"):
        KelvinInstantNuRec._build_primitives(
            context=[None], decoder_returns=decoder_returns, sky_cubemaps=sky, affine_matrix=affine
        )
