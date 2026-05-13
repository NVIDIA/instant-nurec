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

"""Branch-coverage tests for ``instant_nurec.model.jit_adapter``.

End-to-end inference exercises the adapter on GPU; here we cover the
shape-correctness and masking branches in isolation.
"""

from __future__ import annotations

import sys

from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


from instant_nurec.model.jit_adapter import (  # noqa: E402
    _PLACEHOLDER_SKY_CUBEMAP_SIZE,
    JITKelvinAdapter,
)
from instant_nurec.primitives.kelvin_primitive import (  # noqa: E402
    KelvinDynamicLayer,
    KelvinSemanticClass,
    KelvinStaticLayer,
)


# ---------- JITKelvinAdapter (CPU-only, mocked jit_module) ----------


class _FakeJITModule(torch.nn.Module):
    """Fake JIT module emitting per-pixel tensors so the adapter's
    flatten + gather logic can be exercised without GPU."""

    def __init__(self, B: int, V: int, H: int, W: int, n_cams: int, dynamic_pixel_idx: int = -1):
        super().__init__()
        self.B, self.V, self.H, self.W, self.n_cams = B, V, H, W, n_cams
        self._dynamic_pixel_idx = dynamic_pixel_idx
        self.calls: list[tuple] = []

    def forward(self, rgb, c2w, fov, rays, distance_to_depth_scale, camera_idxs):
        self.calls.append((rgb, c2w, fov, rays, distance_to_depth_scale, camera_idxs))
        B, V, H, W = self.B, self.V, self.H, self.W
        n_pixels = V * H * W

        gs_xyz = torch.arange(B * n_pixels * 3, dtype=torch.float32).reshape(B, V, H, W, 3)
        gs_rotations = torch.zeros(B, V, H, W, 4)
        gs_rotations[..., 0] = 1.0
        gs_scales = torch.ones(B, V, H, W, 3)
        gs_densities = torch.full((B, V, H, W, 1), 0.5)
        gs_rgb = torch.full((B, V, H, W, 3), 0.7)

        # Default: no dynamic pixels (all flagged as ROAD).
        semantic = torch.full((B, V, H, W), KelvinSemanticClass.ROAD.value, dtype=torch.int64)
        if self._dynamic_pixel_idx >= 0:
            flat_view = semantic.reshape(B, -1)
            flat_view[:, self._dynamic_pixel_idx] = KelvinSemanticClass.MOVABLE.value

        normals = torch.full((B, V, H, W, 3), 0.1)
        affine = torch.zeros(B, self.n_cams, 3, 4)
        affine[..., :3] = torch.eye(3)
        return gs_xyz, gs_rotations, gs_scales, gs_densities, gs_rgb, semantic, normals, affine


def _make_adapter(jit_module: _FakeJITModule, scene_rescale: float = 0.5) -> JITKelvinAdapter:
    """Adapter constructor reads buffers off the JIT module; attach mocks
    sized to match the fake module's per-pixel output."""
    from types import SimpleNamespace

    jit_module.static_core = SimpleNamespace(
        scene_rescale_buffer=torch.tensor(scene_rescale, dtype=torch.float32),
        expected_b=torch.tensor(jit_module.B, dtype=torch.int64),
        expected_v=torch.tensor(jit_module.V, dtype=torch.int64),
        expected_h=torch.tensor(jit_module.H, dtype=torch.int64),
        expected_w=torch.tensor(jit_module.W, dtype=torch.int64),
        decoder=SimpleNamespace(cuboids_dims_padding=torch.tensor([0.1, 0.1, 0.1])),
    )
    return JITKelvinAdapter(jit_module=jit_module)


def _fake_batch(V: int = 2, H: int = 4, W: int = 4):
    """Minimal DataAndRenderingBatch substitute that ``_extract_tensors`` and
    the masking branch can read from without a real dataloader."""
    from types import SimpleNamespace

    timestamps_startend_us = torch.tensor(
        [[0, 1_000_000]] * V, dtype=torch.int64
    )  # (V, 2)
    rays = torch.zeros(V, H, W, 6)
    rays[..., 5] = 1.0  # rays_dir = (0,0,1) so xyz = origin + depth*z
    distance_to_depth_scale = torch.ones(V, H, W, 1)

    poses = torch.zeros(V, 2, 7)
    poses[..., 6] = 1.0  # quaternion w=1

    # ``_extract_tensors`` only consumes ``resolution`` and ``focal_length`` off
    # the result of ``to_simple_pinhole_model_parameters`` (which gets
    # monkeypatched in the test fixture below), so a SimpleNamespace stand-in
    # is enough.
    sensor_params = [
        SimpleNamespace(resolution=(W, H), focal_length=(float(W), float(H)))
        for _ in range(V)
    ]

    rendering_camera = SimpleNamespace(
        rays=rays,
        rays_timestamps_us=torch.zeros(V, H, W, 1, dtype=torch.int64),
        distance_to_depth_scale=distance_to_depth_scale,
        poses_tquat_startend=poses,
        sensor_model_parameters=sensor_params,
        timestamps_startend_us_cpu=timestamps_startend_us,
    )
    rendering = SimpleNamespace(camera=rendering_camera)

    meta = [SimpleNamespace(unique_sensor_idx=v) for v in range(V)]
    labels = SimpleNamespace(rgb=torch.zeros(V, H, W, 3))
    data_camera = SimpleNamespace(meta=meta, labels=labels, b=V)
    data = SimpleNamespace(camera=data_camera)

    return SimpleNamespace(data=data, rendering=rendering)


@pytest.fixture(autouse=True)
def _stub_sensor_helpers(monkeypatch):
    """``_extract_tensors`` calls ``to_simple_pinhole_model_parameters`` to
    derive fov; bypass it with a passthrough so tests don't need real ncore
    sensor types. Also stub ``tquat_to_se3_matrix`` since the fake batch's
    pose tensor is not a real quaternion."""
    from instant_nurec.model import jit_adapter as adapter_mod

    monkeypatch.setattr(adapter_mod, "to_simple_pinhole_model_parameters", lambda p: p)

    def _identity_se3(q, unbatch):
        # q: (V, 7) -- ignore the actual quaternion math; fake an identity
        # transform with zero translation, shape (V, 4, 4).
        V = q.shape[0]
        m = torch.eye(4).expand(V, 4, 4).clone()
        return m

    monkeypatch.setattr(adapter_mod, "tquat_to_se3_matrix", _identity_se3)


def test_reconstruct_no_cuboid_tracks_returns_one_primitive_per_batch():
    V, H, W = 2, 4, 4
    jit = _FakeJITModule(B=1, V=V, H=H, W=W, n_cams=1)
    adapter = _make_adapter(jit)

    out = adapter.reconstruct([_fake_batch(V, H, W)], cuboid_tracks=None)

    assert len(out) == 1
    primitive = out[0]
    # No dynamic pixels in the fake jit module -> all V*H*W gaussians are static.
    assert len(primitive.static_layer) == V * H * W
    assert isinstance(primitive.static_layer, KelvinStaticLayer)
    assert isinstance(primitive.dynamic_layers, list)
    assert len(primitive.dynamic_layers) == 1
    assert isinstance(primitive.dynamic_layers[0], KelvinDynamicLayer)
    assert len(primitive.dynamic_layers[0]) == 0  # placeholder is empty


def test_reconstruct_drops_movable_pixels_in_semantic_only_mode():
    V, H, W = 2, 4, 4
    # Mark one pixel as MOVABLE -- semantic-only branch should drop it.
    jit = _FakeJITModule(B=1, V=V, H=H, W=W, n_cams=1, dynamic_pixel_idx=5)
    adapter = _make_adapter(jit)

    out = adapter.reconstruct([_fake_batch(V, H, W)], cuboid_tracks=None)

    assert len(out[0].static_layer) == V * H * W - 1


def test_reconstruct_uses_placeholder_sky_cubemap_shape():
    V, H, W = 2, 4, 4
    jit = _FakeJITModule(B=1, V=V, H=H, W=W, n_cams=1)
    adapter = _make_adapter(jit)
    out = adapter.reconstruct([_fake_batch(V, H, W)], cuboid_tracks=None)
    sky = out[0].sky_cubemap
    s = _PLACEHOLDER_SKY_CUBEMAP_SIZE
    assert sky.shape == (6, s, s, 3)
    assert torch.all(sky == 0)


def test_reconstruct_affine_matrix_shape_squeezed_to_per_camera():
    V, H, W, n_cams = 2, 4, 4, 3
    jit = _FakeJITModule(B=1, V=V, H=H, W=W, n_cams=n_cams)
    adapter = _make_adapter(jit)
    out = adapter.reconstruct([_fake_batch(V, H, W)], cuboid_tracks=None)
    assert out[0].affine_matrix.shape == (n_cams, 3, 4)


def test_reconstruct_passes_extracted_tensors_to_jit_module():
    V, H, W = 2, 4, 4
    jit = _FakeJITModule(B=1, V=V, H=H, W=W, n_cams=1)
    adapter = _make_adapter(jit)
    adapter.reconstruct([_fake_batch(V, H, W)], cuboid_tracks=None)

    rgb, c2w, fov, rays, distance_to_depth_scale, camera_idxs = jit.calls[0]
    # Every input is shape ``(1, V, ...)`` with the leading B=1 dim added by
    # the adapter's per-batch unsqueeze.
    assert rgb.shape == (1, V, H, W, 3)
    assert c2w.shape == (1, V, 4, 4)
    assert fov.shape == (1, V, 2)
    assert rays.shape == (1, V, H, W, 6)
    assert distance_to_depth_scale.shape == (1, V, H, W, 1)
    assert camera_idxs.shape == (1, V)


def test_reconstruct_static_layer_semantic_class_is_uint8():
    V, H, W = 2, 4, 4
    jit = _FakeJITModule(B=1, V=V, H=H, W=W, n_cams=1)
    adapter = _make_adapter(jit)
    out = adapter.reconstruct([_fake_batch(V, H, W)], cuboid_tracks=None)
    assert out[0].static_layer.semantic_class.dtype == torch.uint8


# ---------- prepare_context ----------


def test_prepare_context_passthrough():
    from types import SimpleNamespace

    context = [SimpleNamespace()]
    adapter = _make_adapter(_FakeJITModule(B=1, V=2, H=4, W=4, n_cams=1))
    assert adapter.prepare_context(context) is context


# ---------- _empty_dynamic_layer / _placeholder_sky_cubemap ----------


def test_empty_dynamic_layer_has_zero_gaussians_with_correct_dtypes():
    adapter = _make_adapter(_FakeJITModule(1, 1, 1, 1, 1))
    layer = adapter._empty_dynamic_layer(torch.device("cpu"))
    assert len(layer) == 0
    assert layer.keyframe_timestamps_us.dtype == torch.int64
    assert layer.rotations.dtype == torch.float32


def test_placeholder_sky_cubemap_dtype_and_shape():
    adapter = _make_adapter(_FakeJITModule(1, 1, 1, 1, 1))
    cube = adapter._placeholder_sky_cubemap(torch.device("cpu"), torch.float64)
    assert cube.shape == (6, _PLACEHOLDER_SKY_CUBEMAP_SIZE, _PLACEHOLDER_SKY_CUBEMAP_SIZE, 3)
    assert cube.dtype == torch.float64
    assert torch.all(cube == 0)


def test_pytest_collected(monkeypatch):
    """Sentinel: pytest must always pass at least one named test in this
    module to confirm the file isn't accidentally skipped by collection
    rules."""
    monkeypatch.setenv("__JIT_ADAPTER_TEST_SENTINEL__", "1")
    assert True


_ = pytest  # silence unused-import lint
