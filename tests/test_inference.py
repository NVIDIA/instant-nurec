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

"""Branch-coverage tests for ``instant_nurec.model.inference``.

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


from instant_nurec.model.inference import KelvinInferenceModel  # noqa: E402
from instant_nurec.model.static_core import KelvinPointQueryStaticOutput  # noqa: E402
from instant_nurec.primitives.kelvin_primitive import (  # noqa: E402
    KelvinDynamicLayer,
    KelvinSemanticClass,
    KelvinStaticLayer,
)


# ---------- KelvinInferenceModel (CPU-only, mocked static_core) ----------


class _FakeStaticCore(torch.nn.Module):
    """Fake source core emitting per-pixel tensors so the adapter's
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
        sky_cubemap = torch.full((B, 6, 8, 8, 3), 0.25)
        sky_cubemap_mask = torch.ones(B, 6, 8, 8, 1)
        return (
            gs_xyz,
            gs_rotations,
            gs_scales,
            gs_densities,
            gs_rgb,
            semantic,
            normals,
            affine,
            sky_cubemap,
            sky_cubemap_mask,
        )


class _FakePointQueryStaticCore(_FakeStaticCore):
    """Sparse-output counterpart used to cover point-query packaging."""

    def forward(self, rgb, c2w, fov, rays, distance_to_depth_scale, camera_idxs):
        dense = super().forward(rgb, c2w, fov, rays, distance_to_depth_scale, camera_idxs)
        xyz, rotations, scales, densities, color, semantic, normals, affine, sky_cubemap, sky_mask = dense
        source_indices = torch.tensor([[0, 5, 9]], dtype=torch.int64)
        return KelvinPointQueryStaticOutput(
            positions=xyz.reshape(self.B, -1, 3)[:, source_indices[0]],
            rotations=rotations.reshape(self.B, -1, 4)[:, source_indices[0]],
            scales=scales.reshape(self.B, -1, 3)[:, source_indices[0]],
            densities=densities.reshape(self.B, -1, 1)[:, source_indices[0]],
            rgb=color.reshape(self.B, -1, 3)[:, source_indices[0]],
            semantic_class=semantic.reshape(self.B, -1)[:, source_indices[0]],
            normals=normals.reshape(self.B, -1, 3)[:, source_indices[0]],
            affine_matrix=affine,
            source_indices=source_indices,
            sky_cubemap=sky_cubemap,
            sky_cubemap_mask=sky_mask,
        )


def _make_adapter(static_core: _FakeStaticCore, scene_rescale: float = 0.5) -> KelvinInferenceModel:
    """Build the inference wrapper around the small fake source core."""
    from types import SimpleNamespace

    static_core.decoder = SimpleNamespace(
        cuboids_dims_padding=torch.tensor([0.1, 0.1, 0.1]),
    )
    return KelvinInferenceModel(
        static_core=static_core,
        scene_rescale=scene_rescale,
        expected_frames=static_core.V,
        expected_height=static_core.H,
        expected_width=static_core.W,
    )


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
    from instant_nurec.model import inference as adapter_mod

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
    core = _FakeStaticCore(B=1, V=V, H=H, W=W, n_cams=1)
    adapter = _make_adapter(core)

    out = adapter.reconstruct([_fake_batch(V, H, W)], cuboid_tracks=None)

    assert len(out) == 1
    primitive = out[0]
    # No dynamic pixels in the fake core module -> all V*H*W gaussians are static.
    assert len(primitive.static_layer) == V * H * W
    assert isinstance(primitive.static_layer, KelvinStaticLayer)
    assert isinstance(primitive.dynamic_layers, list)
    assert len(primitive.dynamic_layers) == 1
    assert isinstance(primitive.dynamic_layers[0], KelvinDynamicLayer)
    assert len(primitive.dynamic_layers[0]) == 0  # placeholder is empty


def test_reconstruct_drops_movable_pixels_in_semantic_only_mode():
    V, H, W = 2, 4, 4
    # Mark one pixel as MOVABLE -- semantic-only branch should drop it.
    core = _FakeStaticCore(B=1, V=V, H=H, W=W, n_cams=1, dynamic_pixel_idx=5)
    adapter = _make_adapter(core)

    out = adapter.reconstruct([_fake_batch(V, H, W)], cuboid_tracks=None)

    assert len(out[0].static_layer) == V * H * W - 1


def test_reconstruct_with_tracks_keeps_unassociated_movable_gaussians(monkeypatch):
    from types import SimpleNamespace

    from instant_nurec.model import inference as inference_mod
    from instant_nurec.utils.types import TrackFlags

    class _AllMovableStaticCore(_FakeStaticCore):
        def forward(self, *args):
            output = list(super().forward(*args))
            output[5].fill_(KelvinSemanticClass.MOVABLE.value)
            return tuple(output)

    class _DynamicTrack:
        def ray_intersection(self, origins, directions, timestamps, **kwargs):
            del origins, directions, timestamps, kwargs
            return SimpleNamespace(
                intersections_tracks_idx=torch.zeros(2, 2, dtype=torch.int64),
                intersections_cnt=torch.tensor([1, 0], dtype=torch.int64),
            )

    monkeypatch.setattr(
        inference_mod.CuboidTracks.Ops,
        "subset_from_mask",
        lambda tracks, mask: _DynamicTrack(),
    )

    def _fake_warp(**kwargs):
        # The first MOVABLE Gaussian is associated with a dynamic track; the
        # second is parked/unassociated and must remain in the static export.
        assert torch.equal(kwargs["aux_tracks_idx"].reshape(-1), torch.tensor([0, -1]))
        return kwargs["aux_tracks_idx"] >= 0, []

    monkeypatch.setattr(inference_mod, "warp_points_with_cuboid_tracks", _fake_warp)

    core = _AllMovableStaticCore(B=1, V=1, H=1, W=2, n_cams=1)
    adapter = _make_adapter(core)
    tracks = SimpleNamespace(tracks_flags=torch.tensor([int(TrackFlags.DYNAMIC)]))

    primitive = adapter.reconstruct([_fake_batch(V=1, H=1, W=2)], [tracks])[0]

    assert len(primitive.static_layer) == 1
    assert torch.equal(primitive.static_layer.positions, torch.tensor([[3.0, 4.0, 5.0]]))
    assert primitive.static_layer.semantic_class.item() == KelvinSemanticClass.MOVABLE.value


def test_reconstruct_packages_sparse_point_query_output():
    V, H, W = 2, 4, 4
    core = _FakePointQueryStaticCore(B=1, V=V, H=H, W=W, n_cams=1)
    adapter = _make_adapter(core)

    out = adapter.reconstruct([_fake_batch(V, H, W)], cuboid_tracks=None)

    assert len(out[0].static_layer) == 3
    assert torch.equal(
        out[0].static_layer.positions,
        torch.arange(V * H * W * 3, dtype=torch.float32).reshape(-1, 3)[[0, 5, 9]],
    )


def test_sparse_dynamic_mask_gathers_aligned_source_rays_and_timestamps(monkeypatch):
    from types import SimpleNamespace

    from instant_nurec.model import inference as inference_mod
    from instant_nurec.utils.types import TrackFlags

    adapter = _make_adapter(_FakePointQueryStaticCore(B=1, V=1, H=2, W=3, n_cams=1))
    rendering = _fake_batch(V=1, H=2, W=3).rendering.camera
    rendering.rays = torch.arange(6 * 6, dtype=torch.float32).reshape(1, 2, 3, 6)
    rendering.rays_timestamps_us = torch.arange(6, dtype=torch.int64).reshape(1, 2, 3, 1)
    source_indices = torch.tensor([[4, 1]], dtype=torch.int64)
    captured = {}

    class _DynamicTrack:
        def ray_intersection(self, origins, directions, timestamps, **kwargs):
            captured["origins"] = origins
            captured["directions"] = directions
            captured["ray_timestamps"] = timestamps
            return SimpleNamespace(
                intersections_tracks_idx=torch.zeros(2, 2, dtype=torch.int64),
                intersections_cnt=torch.ones(2, dtype=torch.int64),
            )

    dynamic_track = _DynamicTrack()
    monkeypatch.setattr(
        inference_mod.CuboidTracks.Ops,
        "subset_from_mask",
        lambda tracks, mask: dynamic_track,
    )

    def _fake_warp(**kwargs):
        captured["points"] = kwargs["points"]
        captured["source_timestamps"] = kwargs["source_timestamps_us"]
        return torch.tensor([True, False]), []

    monkeypatch.setattr(inference_mod, "warp_points_with_cuboid_tracks", _fake_warp)
    xyz = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    semantic = torch.full((1, 2), KelvinSemanticClass.MOVABLE.value, dtype=torch.int64)
    tracks = SimpleNamespace(tracks_flags=torch.tensor([int(TrackFlags.DYNAMIC)]))

    dynamic_mask = adapter._compute_dynamic_mask(
        xyz,
        semantic,
        rendering,
        tracks,
        source_indices,
    )

    flat_rays = rendering.rays.reshape(-1, 6)[source_indices[0]]
    assert torch.equal(captured["origins"], flat_rays[:, :3])
    assert torch.equal(captured["directions"], flat_rays[:, 3:])
    assert torch.equal(captured["ray_timestamps"], torch.tensor([4, 1]))
    assert torch.equal(captured["source_timestamps"], torch.tensor([[4], [1]]))
    assert torch.equal(dynamic_mask, torch.tensor([True, False]))


def test_reconstruct_preserves_observation_derived_sky_cubemap():
    V, H, W = 2, 4, 4
    core = _FakeStaticCore(B=1, V=V, H=H, W=W, n_cams=1)
    adapter = _make_adapter(core)
    out = adapter.reconstruct([_fake_batch(V, H, W)], cuboid_tracks=None)
    sky = out[0].sky_cubemap
    assert sky.shape == (6, 8, 8, 3)
    assert torch.all(sky == 0.25)
    assert torch.all(out[0].sky_cubemap_mask == 1)


def test_reconstruct_affine_matrix_shape_squeezed_to_per_camera():
    V, H, W, n_cams = 2, 4, 4, 3
    core = _FakeStaticCore(B=1, V=V, H=H, W=W, n_cams=n_cams)
    adapter = _make_adapter(core)
    out = adapter.reconstruct([_fake_batch(V, H, W)], cuboid_tracks=None)
    assert out[0].affine_matrix.shape == (n_cams, 3, 4)


def test_reconstruct_passes_extracted_tensors_to_static_core():
    V, H, W = 2, 4, 4
    core = _FakeStaticCore(B=1, V=V, H=H, W=W, n_cams=1)
    adapter = _make_adapter(core)
    adapter.reconstruct([_fake_batch(V, H, W)], cuboid_tracks=None)

    rgb, c2w, fov, rays, distance_to_depth_scale, camera_idxs = core.calls[0]
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
    core = _FakeStaticCore(B=1, V=V, H=H, W=W, n_cams=1)
    adapter = _make_adapter(core)
    out = adapter.reconstruct([_fake_batch(V, H, W)], cuboid_tracks=None)
    assert out[0].static_layer.semantic_class.dtype == torch.uint8


def test_reconstruct_rejects_input_shape_that_does_not_match_public_config():
    adapter = _make_adapter(_FakeStaticCore(B=1, V=2, H=4, W=4, n_cams=1))

    with pytest.raises(ValueError, match="Input shape mismatch"):
        adapter.reconstruct([_fake_batch(V=1, H=4, W=4)], cuboid_tracks=None)


# ---------- prepare_context ----------


def test_prepare_context_passthrough():
    from types import SimpleNamespace

    context = [SimpleNamespace()]
    adapter = _make_adapter(_FakeStaticCore(B=1, V=2, H=4, W=4, n_cams=1))
    assert adapter.prepare_context(context) is context


# ---------- _empty_dynamic_layer ----------


def test_empty_dynamic_layer_has_zero_gaussians_with_correct_dtypes():
    adapter = _make_adapter(_FakeStaticCore(1, 1, 1, 1, 1))
    layer = adapter._empty_dynamic_layer(torch.device("cpu"))
    assert len(layer) == 0
    assert layer.keyframe_timestamps_us.dtype == torch.int64
    assert layer.rotations.dtype == torch.float32


def test_pytest_collected(monkeypatch):
    """Sentinel: pytest must always pass at least one named test in this
    module to confirm the file isn't accidentally skipped by collection
    rules."""
    monkeypatch.setenv("__INFERENCE_MODEL_TEST_SENTINEL__", "1")
    assert True


_ = pytest  # silence unused-import lint
