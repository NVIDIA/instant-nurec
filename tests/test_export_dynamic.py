# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from instant_nurec.cli import main
from instant_nurec.predict.export_dynamic import export_dynamic
from instant_nurec.predict.render_preview import gaussians_at_timestamp
from instant_nurec.primitives.kelvin_primitive import KelvinDynamicLayer
from tests.test_export_sky import _primitive


def _moving_primitive():
    primitive = _primitive()
    primitive.dynamic_layers = [KelvinDynamicLayer(
        rotations=torch.tensor([[1., 0., 0., 0.]]),
        scales=torch.ones(1, 3),
        rgb=torch.ones(1, 3),
        max_densities=torch.tensor([[0.8]]),
        keyframe_positions=torch.tensor([[[1., 0., 0.], [2., 0., 0.], [3., 0., 0.]]]),
        keyframe_timestamps_us=torch.tensor([[10, 20, 30]], dtype=torch.int64),
        falloff=False,
    )]
    return primitive


def test_dynamic_export_preserves_motion_and_transforms_to_world(tmp_path):
    primitive = _moving_primitive()
    world = torch.tensor([[0., -1., 0., 4.], [1., 0., 0., 5.], [0., 0., 1., 6.], [0., 0., 0., 1.]])
    original = primitive.dynamic_layers[0].keyframe_positions.clone()
    path = export_dynamic(primitive, SimpleNamespace(T_world_base=world), tmp_path / 'scene.ply')
    with np.load(path, allow_pickle=False) as data:
        assert data['format_version'].item() == 1
        assert 'layer_0_motion_time_bounds_us' not in data
        assert data['coordinate_frame'].item() == 'ncore_world'
        assert data['quaternion_format'].item() == 'wxyz'
        assert data['n_layers'].item() == 1
        np.testing.assert_allclose(data['layer_0_keyframe_positions'], [[[4., 6., 6.], [4., 7., 6.], [4., 8., 6.]]])
        np.testing.assert_array_equal(data['layer_0_keyframe_timestamps_us'], [[10, 20, 30]])
        assert data['layer_0_keyframe_timestamps_us'].dtype == np.int64
        assert not data['layer_0_falloff'].item()
        np.testing.assert_allclose(data['layer_0_max_densities'], [[0.8]])
        np.testing.assert_allclose(data['layer_0_rotations'], [[2**-0.5, 0., 0., 2**-0.5]], atol=1e-6)
    torch.testing.assert_close(primitive.dynamic_layers[0].keyframe_positions, original)


def test_dynamic_render_interpolates_motion_and_retains_static():
    primitive = _moving_primitive()
    snapshot = gaussians_at_timestamp(primitive, 15)
    torch.testing.assert_close(snapshot.positions, torch.tensor([[0., 0., 0.], [1.5, 0., 0.]]))
    torch.testing.assert_close(snapshot.densities, torch.tensor([[0.5], [0.8]]))
    primitive.dynamic_layers[0].falloff = True
    assert len(gaussians_at_timestamp(primitive, 1000)) == 1


def test_empty_dynamic_export_is_readable(tmp_path):
    path = export_dynamic(_primitive(), SimpleNamespace(T_world_base=torch.eye(4)), tmp_path / 'empty.ply')
    with np.load(path, allow_pickle=False) as data:
        assert data['layer_0_keyframe_positions'].shape == (0, 3, 3)
        assert data['layer_0_keyframe_timestamps_us'].dtype == np.int64


def test_point_query_dynamic_is_rejected_before_loading_data(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(['--model', 'pq-front', '--include-dynamic', '--ncore-path', 'missing.json', '--output-dir', str(tmp_path)])
    assert error.value.code == 2
    assert 'requires pa-front or pa-multiview' in capsys.readouterr().err


@pytest.mark.parametrize("source,track_kind", [
    ("tracks", "absent"), ("tracks", "empty"), ("tracks", "dynamic"),
    ("tracks", "static"), ("learned", "dynamic"),
])
def test_include_dynamic_preserves_static_layer_and_pr20_fallback(monkeypatch, source, track_kind):
    from dataclasses import fields
    from unittest.mock import Mock

    from instant_nurec.model import inference as inference_mod
    from instant_nurec.primitives.kelvin_primitive import KelvinStaticLayer
    from instant_nurec.utils.types import TrackFlags
    from tests.test_inference import _FakeStaticCore, _fake_batch, _make_adapter

    monkeypatch.setattr(inference_mod, 'to_simple_pinhole_model_parameters', lambda p: p)
    core = _FakeStaticCore(B=1, V=1, H=1, W=2, n_cams=1, dynamic_pixel_idx=0)
    model = _make_adapter(core)
    model.motion_source = source
    core.decoder.config = SimpleNamespace(dpt_chunk_size=1)
    core.decoder.context_motion_head = Mock(return_value=(torch.ones(1, 1, 1, 2, 3),
                                                        torch.ones(1, 1, 1, 2, 3) * 2))
    original_forward = core.forward

    def forward(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        if kwargs['retain_features']:
            output.encoded_latent = object()
        return output

    monkeypatch.setattr(core, 'forward', forward)
    tracks = None
    if track_kind != 'absent':
        flags = [] if track_kind == 'empty' else [int(TrackFlags.DYNAMIC if track_kind == 'dynamic' else TrackFlags.NONE)]
        tracks = [SimpleNamespace(n_tracks=len(flags), tracks_flags=torch.tensor(flags, dtype=torch.int64),
                                  tracks_packinfo=torch.tensor([[0, 2]], dtype=torch.int32),
                                  tracks_timestamps_us=torch.tensor([0, 1_000_000]))]
    associate = Mock(return_value=torch.tensor([[[[0], [-1]]]], dtype=torch.int32))
    monkeypatch.setattr(inference_mod, 'associate_points_with_cuboid_tracks', associate)

    def warp(**kwargs):
        return kwargs['tracks_idx'] != -1, [kwargs['points'] + 1 for _ in kwargs['target_timestamps_us_list']]

    monkeypatch.setattr(inference_mod, 'warp_points_with_cuboid_tracks', warp)
    batch = _fake_batch(V=1, H=1, W=2)
    before = model.reconstruct([batch], tracks)[0]
    model.include_dynamic = True
    after = model.reconstruct([batch], tracks)[0]
    for field in fields(KelvinStaticLayer):
        a, b = getattr(before.static_layer, field.name), getattr(after.static_layer, field.name)
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(before.sky_cubemap, after.sky_cubemap, rtol=0, atol=0)
    torch.testing.assert_close(before.affine_matrix, after.affine_matrix, rtol=0, atol=0)
    assert len(before.dynamic_layers[0]) == 0
    if source == 'learned' or track_kind in ('absent', 'empty'):
        assert len(after.static_layer) == 1
        assert len(after.dynamic_layers[0]) == 1
        core.decoder.context_motion_head.assert_called_once()
        associate.assert_not_called()
    else:
        assert len(after.static_layer) == (1 if track_kind == 'dynamic' else 2)
        assert len(after.dynamic_layers[0]) == (1 if track_kind == 'dynamic' else 0)
        core.decoder.context_motion_head.assert_not_called()


def test_track_bounds_roundtrip_keeps_endpoint_hold_after_export(tmp_path):
    primitive = _moving_primitive()
    layer = primitive.dynamic_layers[0]
    layer.motion_time_bounds_us = torch.tensor([[10, 30]])
    world = torch.eye(4)
    world[0, 3] = 4.
    path = export_dynamic(primitive, SimpleNamespace(T_world_base=world), tmp_path / 'bounded.ply')
    with np.load(path, allow_pickle=False) as data:
        assert data['format_version'].item() == 1
        assert data['layer_0_motion_time_bounds_us'].dtype == np.int64
        restored = KelvinDynamicLayer(**{
            name: torch.from_numpy(data[f'layer_0_{name}']) for name in (
                'keyframe_positions', 'keyframe_timestamps_us', 'rotations', 'scales', 'rgb',
                'max_densities', 'motion_time_bounds_us',
            )
        }, falloff=False)
    torch.testing.assert_close(restored.interpolate(40).positions, torch.tensor([[7., 0., 0.]]))
