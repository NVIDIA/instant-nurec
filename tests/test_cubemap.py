"""Branch-coverage tests for nre.nrm.utils.cubemap.

The module imports ``libs.vren.interface`` and ``ncore.impl.data.types``
at the top level for the unrelated ``unproject_to_sky_cubemap`` helper;
those are compiled .so / ncore Python bindings that aren't available in
the cpu-only test venv. We stub them via sys.modules so we can import
the pure-torch ``cubemap_ray_directions`` and ``rotate_sky_cubemap``
without pulling in the bazel runtime.
"""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _stub_compiled_imports(monkeypatch: pytest.MonkeyPatch):
    """Provide minimal sys.modules stubs so ``import nre.nrm.utils.cubemap``
    succeeds in the cpu-only test venv."""
    # libs.vren.interface — only camera_rays_to_image_points is referenced
    # at module load (as an import), and only used inside unproject_to_sky_cubemap
    # which we don't call in this suite.
    libs_mod = types.ModuleType("libs")
    vren_mod = types.ModuleType("libs.vren")
    vren_iface_mod = types.ModuleType("libs.vren.interface")
    vren_iface_mod.camera_rays_to_image_points = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "libs", libs_mod)
    monkeypatch.setitem(sys.modules, "libs.vren", vren_mod)
    monkeypatch.setitem(sys.modules, "libs.vren.interface", vren_iface_mod)

    # ncore.impl.data.types — only the CameraModelParameters type is referenced
    # for type hinting; a placeholder class is enough.
    ncore_mod = types.ModuleType("ncore")
    ncore_impl_mod = types.ModuleType("ncore.impl")
    ncore_impl_data_mod = types.ModuleType("ncore.impl.data")
    ncore_types_mod = types.ModuleType("ncore.impl.data.types")
    ncore_types_mod.CameraModelParameters = type("CameraModelParameters", (), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ncore", ncore_mod)
    monkeypatch.setitem(sys.modules, "ncore.impl", ncore_impl_mod)
    monkeypatch.setitem(sys.modules, "ncore.impl.data", ncore_impl_data_mod)
    monkeypatch.setitem(sys.modules, "ncore.impl.data.types", ncore_types_mod)

    # Force a fresh import (drop any prior cached version).
    sys.modules.pop("instant_nurec.nre.nrm.utils.cubemap", None)


# ---------------------------------------------------------------------------
# cubemap_ray_directions
# ---------------------------------------------------------------------------


def test_cubemap_ray_directions_shape():
    import torch

    from instant_nurec.nre.nrm.utils.cubemap import cubemap_ray_directions

    out = cubemap_ray_directions(8, device=torch.device("cpu"))
    assert out.shape == (6, 8, 8, 3)


def test_cubemap_ray_directions_unit_length():
    import torch

    from instant_nurec.nre.nrm.utils.cubemap import cubemap_ray_directions

    out = cubemap_ray_directions(4, device=torch.device("cpu"))
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_cubemap_ray_directions_face_centers_align_with_dominant_axis():
    """The center pixel of each face should produce a ray dominated by the
    expected axis. NRE face order is (right=+X, left=-X, top=-Y, bottom=+Y,
    front=+Z, back=-Z)."""
    import torch

    from instant_nurec.nre.nrm.utils.cubemap import cubemap_ray_directions

    H = 4
    out = cubemap_ray_directions(H, device=torch.device("cpu"))
    # take the geometric center pixel of each face — for even H the
    # (H/2, H/2) sample is offset slightly but still has the dominant
    # axis align with the face's signed direction.
    cx = cy = H // 2
    expected_axis_sign = [
        (0, +1),  # face 0 = +X
        (0, -1),  # face 1 = -X
        (1, -1),  # face 2 = -Y
        (1, +1),  # face 3 = +Y
        (2, +1),  # face 4 = +Z
        (2, -1),  # face 5 = -Z
    ]
    for f, (axis, sign) in enumerate(expected_axis_sign):
        ray = out[f, cy, cx]
        # dominant axis should be `axis`
        assert ray.abs().argmax().item() == axis, f"face {f}: dominant axis"
        # and have the right sign
        assert (ray[axis].item() > 0) == (sign > 0), f"face {f}: sign"


# ---------------------------------------------------------------------------
# rotate_sky_cubemap
# ---------------------------------------------------------------------------


def test_rotate_sky_cubemap_shape_preserved():
    import torch

    from instant_nurec.nre.nrm.utils.cubemap import rotate_sky_cubemap

    cube = torch.randn(6, 8, 8, 3)
    rot = torch.eye(3)
    out = rotate_sky_cubemap(cube, rot)
    assert out.shape == (6, 8, 8, 3)


def test_rotate_sky_cubemap_identity_recovers_input_within_aliasing():
    """Rotating by identity should give back the input (modulo bilinear
    self-sampling at face boundaries — using a constant-color cubemap to
    avoid that aliasing entirely)."""
    import torch

    from instant_nurec.nre.nrm.utils.cubemap import rotate_sky_cubemap

    # Constant per-face color so the cube is invariant under identity rotation
    # without any boundary-interp loss.
    cube = torch.zeros(6, 16, 16, 3)
    for f in range(6):
        cube[f, :, :, :] = float(f) / 5.0
    rot = torch.eye(3)
    out = rotate_sky_cubemap(cube, rot)
    assert torch.allclose(out, cube, atol=1e-5)


def test_rotate_sky_cubemap_zero_input_yields_zero():
    import torch

    from instant_nurec.nre.nrm.utils.cubemap import rotate_sky_cubemap

    cube = torch.zeros(6, 8, 8, 3)
    rot = torch.tensor(
        [
            [math.cos(0.4), -math.sin(0.4), 0.0],
            [math.sin(0.4), math.cos(0.4), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    out = rotate_sky_cubemap(cube, rot)
    assert torch.allclose(out, torch.zeros_like(cube))


def test_rotate_sky_cubemap_constant_color_is_constant_after_rotation():
    """A globally-constant cubemap (same color on every face) must still be
    constant under any rotation — this is a strong invariant."""
    import torch

    from instant_nurec.nre.nrm.utils.cubemap import rotate_sky_cubemap

    cube = torch.full((6, 8, 8, 3), 0.42)
    angle = math.radians(37.0)
    rot = torch.tensor(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ]
    )
    out = rotate_sky_cubemap(cube, rot)
    assert torch.allclose(out, torch.full_like(out, 0.42), atol=1e-5)


def test_rotate_sky_cubemap_supports_arbitrary_channel_count():
    """The torch impl uses cubemap.shape[-1] for the channel dimension —
    feature cubemaps with C != 3 should work too."""
    import torch

    from instant_nurec.nre.nrm.utils.cubemap import rotate_sky_cubemap

    cube = torch.randn(6, 8, 8, 5)  # 5-channel feature cubemap
    rot = torch.eye(3)
    out = rotate_sky_cubemap(cube, rot)
    assert out.shape == (6, 8, 8, 5)


# ---------------------------------------------------------------------------
# unproject_to_sky_cubemap
# ---------------------------------------------------------------------------


class _FakeImagePointsReturn:
    """Stand-in for the namedtuple returned by libs.vren camera_rays_to_image_points."""

    def __init__(self, n_rays: int, n_valid: int, image_w: int, image_h: int):
        # All-True valid_flag for the first n_valid; rest invalid.
        import torch

        valid = torch.zeros(n_rays, dtype=torch.bool)
        valid[:n_valid] = True
        self.valid_flag = valid
        # image_points: 2D coords inside the image
        pts = torch.zeros(n_rays, 2)
        pts[:, 0] = image_w / 2  # center x
        pts[:, 1] = image_h / 2  # center y
        self.image_points = pts


class _FakeCameraModelParameters:
    def __init__(self, w: int = 16, h: int = 16):
        import numpy as np

        self.resolution = np.array([w, h], dtype=np.float32)


@pytest.fixture
def _vren_with_fake_camera_rays(monkeypatch: pytest.MonkeyPatch):
    """Replace the no-op ``camera_rays_to_image_points`` stub with one that
    returns a ``_FakeImagePointsReturn`` for any input rays."""
    import sys

    def fake_camera_rays_to_image_points(camera_params, rays):
        n = rays.shape[0]
        # Mark the first half valid
        return _FakeImagePointsReturn(n_rays=n, n_valid=n // 2, image_w=16, image_h=16)

    sys.modules["libs.vren.interface"].camera_rays_to_image_points = (  # type: ignore[attr-defined]
        fake_camera_rays_to_image_points
    )
    yield


def test_unproject_to_sky_cubemap_returns_correct_shapes(_vren_with_fake_camera_rays):
    """With a fake camera_rays_to_image_points, verify the output shape
    matches the contract: feature (6, S, S, C) and mask (6, S, S, 1)."""
    import torch

    from instant_nurec.nre.nrm.utils.cubemap import unproject_to_sky_cubemap

    sky_size = 4
    N, H, W, C = 1, 8, 8, 3
    R_camera_world = torch.eye(3)[None].repeat(N, 1, 1)
    feature = torch.randn(N, H, W, C)
    feature_mask = torch.ones(N, H, W, 1)
    cam_params = [_FakeCameraModelParameters(w=W, h=H) for _ in range(N)]

    feat_out, mask_out = unproject_to_sky_cubemap(
        sky_cubemap_size=sky_size,
        R_camera_world=R_camera_world,
        camera_model_parameters=cam_params,
        feature=feature,
        feature_mask=feature_mask,
    )
    assert feat_out.shape == (6, sky_size, sky_size, C)
    assert mask_out.shape == (6, sky_size, sky_size, 1)


def test_unproject_to_sky_cubemap_zero_valid_rays_yields_empty_mask(monkeypatch):
    """If camera_rays_to_image_points marks nothing valid, the output mask
    is all False and the feature is all zero."""
    import sys
    import torch

    def fake_camera_rays_to_image_points(camera_params, rays):
        return _FakeImagePointsReturn(n_rays=rays.shape[0], n_valid=0, image_w=16, image_h=16)

    sys.modules["libs.vren.interface"].camera_rays_to_image_points = (  # type: ignore[attr-defined]
        fake_camera_rays_to_image_points
    )

    from instant_nurec.nre.nrm.utils.cubemap import unproject_to_sky_cubemap

    sky_size = 4
    R_camera_world = torch.eye(3)[None]
    feature = torch.randn(1, 8, 8, 3)
    feature_mask = torch.ones(1, 8, 8, 1)
    cam_params = [_FakeCameraModelParameters(w=8, h=8)]

    feat_out, mask_out = unproject_to_sky_cubemap(
        sky_cubemap_size=sky_size,
        R_camera_world=R_camera_world,
        camera_model_parameters=cam_params,
        feature=feature,
        feature_mask=feature_mask,
    )
    # All-zero feature, all-False mask
    assert torch.allclose(feat_out, torch.zeros_like(feat_out))
    assert not mask_out.any()


def test_unproject_to_sky_cubemap_multiple_views(_vren_with_fake_camera_rays):
    """Multi-view input flows through the per-view loop without error."""
    import torch

    from instant_nurec.nre.nrm.utils.cubemap import unproject_to_sky_cubemap

    sky_size = 4
    N, H, W, C = 3, 8, 8, 3
    R_camera_world = torch.eye(3)[None].repeat(N, 1, 1)
    feature = torch.randn(N, H, W, C)
    feature_mask = torch.ones(N, H, W, 1)
    cam_params = [_FakeCameraModelParameters(w=W, h=H) for _ in range(N)]

    feat_out, mask_out = unproject_to_sky_cubemap(
        sky_cubemap_size=sky_size,
        R_camera_world=R_camera_world,
        camera_model_parameters=cam_params,
        feature=feature,
        feature_mask=feature_mask,
    )
    assert feat_out.shape == (6, sky_size, sky_size, C)
    assert mask_out.shape == (6, sky_size, sky_size, 1)
