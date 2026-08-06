# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

import instant_nurec.predict.render_preview as render_preview_module

from instant_nurec.predict.render_preview import composite_sky_and_affine
from instant_nurec.primitives.kelvin_primitive import KelvinInstantNuRecPrimitive, KelvinStaticLayer


def _identity_affine() -> torch.Tensor:
    return torch.eye(3, 4)


def test_transparent_foreground_reveals_sky() -> None:
    foreground = torch.zeros(2, 3, 3)
    opacity = torch.zeros(2, 3)
    sky = torch.full_like(foreground, 0.6)

    output = composite_sky_and_affine(foreground, opacity, sky, _identity_affine())

    torch.testing.assert_close(output, sky)


def test_opaque_foreground_hides_sky() -> None:
    foreground = torch.full((2, 3, 3), 0.2)
    opacity = torch.ones(2, 3, 1)
    sky = torch.full_like(foreground, 0.8)

    output = composite_sky_and_affine(foreground, opacity, sky, _identity_affine())

    torch.testing.assert_close(output, foreground)


def test_affine_is_applied_after_sky_compositing() -> None:
    foreground = torch.zeros(1, 1, 3)
    opacity = torch.zeros(1, 1)
    sky = torch.tensor([[[0.1, 0.2, 0.3]]])
    affine = torch.tensor(
        [
            [2.0, 0.0, 0.0, 0.1],
            [0.0, 1.0, 0.0, 0.2],
            [0.0, 0.0, 0.5, 0.3],
        ]
    )

    output = composite_sky_and_affine(foreground, opacity, sky, affine)

    torch.testing.assert_close(output, torch.tensor([[[0.3, 0.4, 0.45]]]))


def test_reference_preview_uses_end_pose_frame_rays_and_sensor_affine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rays = torch.arange(2 * 2 * 3 * 6, dtype=torch.float32).reshape(2, 2, 3, 6)
    poses = torch.zeros(2, 2, 7)
    poses[..., 6] = 1.0
    poses[0, 0, :3] = torch.tensor([1.0, 2.0, 3.0])
    poses[0, 1, :3] = torch.tensor([4.0, 5.0, 6.0])
    poses[1, 0, :3] = torch.tensor([10.0, 11.0, 12.0])
    poses[1, 1, :3] = torch.tensor([20.0, 21.0, 22.0])
    rendering = SimpleNamespace(
        rays=rays,
        poses_tquat_startend=poses,
        sensor_model_parameters=[object(), object()],
    )
    camera_data = SimpleNamespace(
        b=2,
        meta=[SimpleNamespace(unique_sensor_idx=9), SimpleNamespace(unique_sensor_idx=3)],
    )
    context = SimpleNamespace(
        data=SimpleNamespace(camera=camera_data),
        rendering=SimpleNamespace(camera=rendering),
    )

    static = KelvinStaticLayer(
        positions=torch.zeros(1, 3),
        densities=torch.full((1, 1), 0.5),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        scales=torch.ones(1, 3),
        rgb=torch.tensor([[1.5, -0.5, 0.25]]),
    )
    primitive = KelvinInstantNuRecPrimitive(
        static_layer=static,
        dynamic_layers=[],
        sky_cubemap=torch.zeros(6, 2, 2, 3),
        affine_matrix=torch.eye(3, 4).repeat(2, 1, 1),
    )
    primitive.affine_matrix[1] = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.1],
            [0.0, 1.0, 0.0, 0.2],
            [0.0, 0.0, 1.0, 0.3],
        ]
    )

    seen: dict[str, object] = {}
    camera_to_world = torch.eye(4)
    camera_to_world[:3, 3] = torch.tensor([2.0, 3.0, 4.0])

    def fake_tquat_to_se3_matrix(value: torch.Tensor, *, unbatch: bool) -> torch.Tensor:
        seen["pose"] = value.clone()
        seen["unbatch"] = unbatch
        return camera_to_world.clone()

    def fake_rasterization(**kwargs):
        seen["viewmats"] = kwargs["viewmats"].clone()
        seen["colors"] = kwargs["colors"].clone()
        height, width = kwargs["height"], kwargs["width"]
        return torch.zeros(1, height, width, 3), torch.zeros(1, height, width, 1), {}

    def fake_sample_sky_cubemap(cubemap: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        del cubemap
        seen["directions"] = directions.clone()
        return torch.full((*directions.shape[:-1], 3), 0.25)

    def fake_composite(
        foreground: torch.Tensor,
        opacity: torch.Tensor,
        sky_rgb: torch.Tensor,
        affine: torch.Tensor,
    ) -> torch.Tensor:
        del opacity, sky_rgb
        seen["affine"] = affine.clone()
        return foreground

    fake_gsplat = ModuleType("gsplat")
    setattr(fake_gsplat, "rasterization", fake_rasterization)
    pinhole = SimpleNamespace(focal_length=(100.0, 101.0), principal_point=(1.0, 0.5))
    monkeypatch.setitem(sys.modules, "gsplat", fake_gsplat)
    monkeypatch.setattr(render_preview_module, "tquat_to_se3_matrix", fake_tquat_to_se3_matrix)
    monkeypatch.setattr(render_preview_module, "to_simple_pinhole_model_parameters", lambda _: pinhole)
    monkeypatch.setattr(render_preview_module, "sample_sky_cubemap", fake_sample_sky_cubemap)
    monkeypatch.setattr(render_preview_module, "composite_sky_and_affine", fake_composite)

    output_path = tmp_path / "preview.png"
    stats = render_preview_module.render_reference_preview(primitive, context, output_path, frame_index=1)

    torch.testing.assert_close(seen["pose"], poses[1, 1])
    assert seen["unbatch"] is True
    torch.testing.assert_close(seen["viewmats"], torch.linalg.inv(camera_to_world).unsqueeze(0))
    torch.testing.assert_close(seen["colors"], primitive.static_layer.rgb)
    torch.testing.assert_close(seen["directions"], rays[1, ..., 3:])
    torch.testing.assert_close(seen["affine"], primitive.affine_matrix[1])
    assert stats.path == output_path
    assert stats.width == 3
    assert stats.height == 2
    assert output_path.is_file()
