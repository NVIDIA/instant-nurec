# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

from enum import Enum
from types import ModuleType, SimpleNamespace

import pytest
import torch

import instant_nurec.predict.render_preview as render_preview_module

from instant_nurec.predict.render_preview import composite_sky_and_affine
from instant_nurec.primitives.kelvin_primitive import KelvinInstantNuRecPrimitive, KelvinStaticLayer


def _identity_affine() -> torch.Tensor:
    return torch.eye(3, 4)


class _FakeFThetaPolynomialType(Enum):
    PIXELDIST_TO_ANGLE = 0
    ANGLE_TO_PIXELDIST = 1


class _FakeRollingShutterType(Enum):
    ROLLING_TOP_TO_BOTTOM = 0
    ROLLING_LEFT_TO_RIGHT = 1
    ROLLING_BOTTOM_TO_TOP = 2
    ROLLING_RIGHT_TO_LEFT = 3
    GLOBAL = 4


class _FakeFThetaCameraDistortionParameters:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


def _ftheta_parameters(*, external_distortion: object | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        reference_poly=SimpleNamespace(name="PIXELDIST_TO_ANGLE"),
        pixeldist_to_angle_poly=(0.0, 0.01, 0.0, 0.0, 0.0, 0.0),
        angle_to_pixeldist_poly=(0.0, 100.0, 0.0, 0.0, 0.0, 0.0),
        max_angle=1.25,
        linear_cde=(1.0, 0.01, -0.02),
        principal_point=(1.25, 0.75),
        shutter_type=SimpleNamespace(name="ROLLING_TOP_TO_BOTTOM"),
        external_distortion_parameters=external_distortion,
    )


def _render_inputs(
    sensor_parameters: object,
) -> tuple[KelvinInstantNuRecPrimitive, SimpleNamespace, torch.Tensor, torch.Tensor]:
    rays = torch.arange(1 * 2 * 3 * 6, dtype=torch.float32).reshape(1, 2, 3, 6)
    poses = torch.zeros(1, 2, 7)
    poses[..., 6] = 1.0
    poses[0, 0, :3] = torch.tensor([1.0, 2.0, 3.0])
    poses[0, 1, :3] = torch.tensor([4.0, 5.0, 6.0])
    rendering = SimpleNamespace(
        rays=rays,
        poses_tquat_startend=poses,
        sensor_model_parameters=[sensor_parameters],
    )
    context = SimpleNamespace(
        data=SimpleNamespace(
            camera=SimpleNamespace(
                b=1,
                meta=[SimpleNamespace(unique_sensor_idx=0)],
            )
        ),
        rendering=SimpleNamespace(camera=rendering),
    )
    primitive = KelvinInstantNuRecPrimitive(
        static_layer=KelvinStaticLayer(
            positions=torch.zeros(1, 3),
            densities=torch.full((1, 1), 0.5),
            rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            scales=torch.ones(1, 3),
            rgb=torch.tensor([[0.2, 0.2, 0.2]]),
        ),
        dynamic_layers=[],
        sky_cubemap=torch.zeros(6, 2, 2, 3),
        affine_matrix=torch.eye(3, 4).unsqueeze(0),
    )
    return primitive, context, rays, poses


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


def test_composited_frame_uses_calibrated_ftheta_rolling_shutter_and_exact_rays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primitive, context, rays, poses = _render_inputs(_ftheta_parameters())
    seen: dict[str, object] = {}
    camera_to_world_start = torch.eye(4)
    camera_to_world_start[:3, 3] = torch.tensor([2.0, 3.0, 4.0])
    camera_to_world_end = torch.eye(4)
    camera_to_world_end[:3, 3] = torch.tensor([5.0, 6.0, 7.0])

    def fake_tquat_to_se3_matrix(value: torch.Tensor, *, unbatch: bool) -> torch.Tensor:
        seen.setdefault("poses", []).append(value.clone())
        assert unbatch is True
        if torch.equal(value, poses[0, 0]):
            return camera_to_world_start.clone()
        if torch.equal(value, poses[0, 1]):
            return camera_to_world_end.clone()
        raise AssertionError(f"Unexpected pose: {value}")

    def fake_rasterization(**kwargs):
        seen["rasterization"] = kwargs
        height, width = kwargs["height"], kwargs["width"]
        foreground = torch.full((1, height, width, 3), 0.2)
        opacity = torch.full((1, height, width, 1), 0.5)
        return foreground, opacity, {}

    def fake_sample_sky_cubemap(cubemap: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        del cubemap
        seen["directions"] = directions.clone()
        return torch.full((*directions.shape[:-1], 3), 0.4)

    fake_gsplat = ModuleType("gsplat")
    setattr(fake_gsplat, "rasterization", fake_rasterization)
    fake_gsplat_rendering = ModuleType("gsplat.rendering")
    setattr(
        fake_gsplat_rendering,
        "FThetaCameraDistortionParameters",
        _FakeFThetaCameraDistortionParameters,
    )
    setattr(fake_gsplat_rendering, "FThetaPolynomialType", _FakeFThetaPolynomialType)
    setattr(fake_gsplat_rendering, "RollingShutterType", _FakeRollingShutterType)
    monkeypatch.setitem(sys.modules, "gsplat", fake_gsplat)
    monkeypatch.setitem(sys.modules, "gsplat.rendering", fake_gsplat_rendering)
    monkeypatch.setattr(render_preview_module, "tquat_to_se3_matrix", fake_tquat_to_se3_matrix)
    monkeypatch.setattr(render_preview_module, "sample_sky_cubemap", fake_sample_sky_cubemap)

    composed, opacity, sky = render_preview_module.render_composited_frame(primitive, context)

    kwargs = seen["rasterization"]
    assert isinstance(kwargs, dict)
    assert kwargs["camera_model"] == "ftheta"
    assert kwargs["packed"] is False
    assert kwargs["with_ut"] is True
    assert kwargs["with_eval3d"] is True
    assert kwargs["global_z_order"] is False
    assert kwargs["rolling_shutter"] is _FakeRollingShutterType.ROLLING_TOP_TO_BOTTOM
    torch.testing.assert_close(kwargs["rays"], rays)
    torch.testing.assert_close(
        kwargs["viewmats"],
        torch.linalg.inv(camera_to_world_start).unsqueeze(0),
    )
    torch.testing.assert_close(
        kwargs["viewmats_rs"],
        torch.linalg.inv(camera_to_world_end).unsqueeze(0),
    )
    torch.testing.assert_close(
        kwargs["Ks"],
        torch.tensor([[[100.0, 0.0, 1.25], [0.0, 100.0, 0.75], [0.0, 0.0, 1.0]]]),
    )
    ftheta_coeffs = kwargs["ftheta_coeffs"]
    assert isinstance(ftheta_coeffs, _FakeFThetaCameraDistortionParameters)
    assert ftheta_coeffs.reference_poly is _FakeFThetaPolynomialType.PIXELDIST_TO_ANGLE
    assert ftheta_coeffs.pixeldist_to_angle_poly == (0.0, 0.01, 0.0, 0.0, 0.0, 0.0)
    assert ftheta_coeffs.angle_to_pixeldist_poly == (0.0, 100.0, 0.0, 0.0, 0.0, 0.0)
    assert ftheta_coeffs.max_angle == 1.25
    assert ftheta_coeffs.linear_cde == (1.0, 0.01, -0.02)
    torch.testing.assert_close(seen["directions"], rays[0, ..., 3:])
    torch.testing.assert_close(opacity, torch.full((2, 3), 0.5))
    torch.testing.assert_close(sky, torch.full((2, 3, 3), 0.4))
    torch.testing.assert_close(composed, torch.full((2, 3, 3), 0.4))


def test_composited_frame_rejects_ftheta_external_windshield_distortion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primitive, context, _, _ = _render_inputs(_ftheta_parameters(external_distortion=SimpleNamespace()))
    fake_gsplat = ModuleType("gsplat")
    setattr(fake_gsplat, "rasterization", lambda **_: None)
    monkeypatch.setitem(sys.modules, "gsplat", fake_gsplat)

    with pytest.raises(NotImplementedError, match="external windshield distortion"):
        render_preview_module.render_composited_frame(primitive, context)


def test_composited_frame_rejects_non_ftheta_before_loading_gsplat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primitive, context, _, _ = _render_inputs(object())

    def unexpected_gsplat_import():
        raise AssertionError("unsupported cameras must fail before loading gsplat")

    monkeypatch.setattr(render_preview_module, "require_gsplat", unexpected_gsplat_import)

    with pytest.raises(NotImplementedError, match="supports NCore F-theta cameras only"):
        render_preview_module.render_composited_frame(primitive, context)
