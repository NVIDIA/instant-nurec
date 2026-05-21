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

"""Branch-coverage tests for ``instant_nurec.utils.sensors.ncore_sensors_converters``.

The module converts ncore camera models into the in-tree dataclass
parameter types (these live in
``instant_nurec.utils.sensors.kernel_types``). The ncore types are
compiled extensions; we stub them via ``sys.modules`` and verify that
``CameraModelConverter.convert`` dispatches by isinstance, calls each
projection-specific factory with the right tensors, and assembles the
``CameraModelConverterResult`` correctly.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def stubbed_converters(monkeypatch):
    # The converter pulls dataclasses from
    # ``instant_nurec.utils.sensors.kernel_types`` (in-tree). We use
    # the real types and wrap their ``from_components`` with a capture
    # shim so the existing call-arg assertions still work.
    captured: dict = {}

    # ncore stubs (needed BEFORE kernel_types import — the converter package
    # __init__ pulls sensors.py which pulls ncore).
    ncore_mod = types.ModuleType("ncore")
    data_mod = types.ModuleType("ncore.data")
    sensors_ncore = types.ModuleType("ncore.sensors")

    class _AnglePolyType:
        ANGLE_TO_PIXELDIST = "ANGLE_TO_PIXELDIST"
        PIXELDIST_TO_ANGLE = "PIXELDIST_TO_ANGLE"

    class _FThetaCameraModelParameters:
        PolynomialType = _AnglePolyType

    class _NcoreReferencePolynomial:
        FORWARD = "FORWARD"
        BACKWARD = "BACKWARD"

    data_mod.FThetaCameraModelParameters = _FThetaCameraModelParameters
    data_mod.ReferencePolynomial = _NcoreReferencePolynomial
    data_mod.ConcreteCameraModelParametersUnion = object

    class _ShutterTypeNcore:
        # Match the in-tree kernel_types.ShutterType IntEnum values (1-5);
        # the converter does ShutterType(camera_model.shutter_type.value).
        ROLLING = type("RollingTag", (), {"value": 1})()

    data_mod.ShutterType = _ShutterTypeNcore

    class CameraModel:
        pass

    class FThetaCameraModel(CameraModel):
        pass

    class BivariateWindshieldModel:
        pass

    sensors_ncore.CameraModel = CameraModel
    sensors_ncore.FThetaCameraModel = FThetaCameraModel
    sensors_ncore.BivariateWindshieldModel = BivariateWindshieldModel
    ncore_mod.data = data_mod
    ncore_mod.sensors = sensors_ncore

    for name, mod in [
        ("ncore", ncore_mod),
        ("ncore.data", data_mod),
        ("ncore.sensors", sensors_ncore),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
    for cached in (
        "instant_nurec.utils.sensors.ncore_sensors_converters",
        "instant_nurec.utils.sensors.sensors",
        "instant_nurec.utils.sensors.kernel_types",
        "instant_nurec.utils.sensors.ray_gen",
        "instant_nurec.utils.sensors",
        "instant_nurec.utils.types",
    ):
        monkeypatch.delitem(sys.modules, cached, raising=False)

    import importlib

    # Now safe to import the in-tree types (converter package __init__ pulls
    # sensors.py → ncore.data, which is stubbed above).
    kt = importlib.import_module("instant_nurec.utils.sensors.kernel_types")

    def _wrap_from_components(real_cls, captured_name):
        original = real_cls.from_components

        def _capturing(*args, **kwargs):
            captured.setdefault("calls", []).append((captured_name, dict(kwargs)))
            return original(*args, **kwargs)

        return _capturing

    monkeypatch.setattr(
        kt.FThetaProjection,
        "from_components",
        _wrap_from_components(kt.FThetaProjection, "FThetaProjection"),
    )
    monkeypatch.setattr(
        kt.BivariateWindshieldDistortion,
        "from_components",
        _wrap_from_components(
            kt.BivariateWindshieldDistortion, "BivariateWindshieldDistortion"
        ),
    )

    FThetaProjection = kt.FThetaProjection  # noqa: F841
    BivariateWindshieldDistortion = kt.BivariateWindshieldDistortion  # noqa: F841
    FThetaPolynomialType = kt.FThetaPolynomialType
    ReferencePolynomial = kt.ReferencePolynomial
    ShutterType = kt.ShutterType

    converters = importlib.import_module("instant_nurec.utils.sensors.ncore_sensors_converters")
    return (
        converters,
        captured,
        FThetaCameraModel,
        BivariateWindshieldModel,
        ShutterType,
        ReferencePolynomial,
        FThetaPolynomialType,
        _NcoreReferencePolynomial,
        _AnglePolyType,
    )


def _shutter_obj(shutter_type_value):
    """A camera-model.shutter_type stand-in: has a .value attribute the
    converter passes to ``ShutterType(...)``."""
    obj = types.SimpleNamespace(value=shutter_type_value.value)
    return obj


def _attach_bivariate(m, BivariateModel):
    """Attach a stubbed BivariateWindshieldModel external distortion to a
    camera-model stand-in."""
    d = BivariateModel()
    d.horizontal_poly = torch.zeros(3)
    d.vertical_poly = torch.zeros(3)
    d.horizontal_poly_inverse = torch.zeros(3)
    d.vertical_poly_inverse = torch.zeros(3)
    d.reference_poly = "FORWARD"
    m.external_distortion = d
    return m


def _make_ftheta(model_cls, ShutterType, ref_poly):
    m = model_cls()
    m.principal_point = torch.tensor([320.0, 240.0])
    m.fw_poly = torch.zeros(8)
    m.bw_poly = torch.zeros(8)
    m.A = torch.eye(2)
    m.Ainv = torch.eye(2)
    m.dfw_poly = torch.zeros(7)
    m.dbw_poly = torch.zeros(7)
    m.reference_poly = ref_poly
    m.max_angle = 1.5
    m.newton_iterations = 5
    m.resolution = torch.tensor([640, 480])
    m.shutter_type = _shutter_obj(ShutterType.ROLLING_TOP_TO_BOTTOM)
    m.external_distortion = None
    return m


# ---------------------------------------------------------------------------
# convert() — projection branches
# ---------------------------------------------------------------------------


def test_convert_ftheta_angle_to_pixeldist_uses_forward_polynomial(stubbed_converters):
    (
        mod,
        captured,
        FThetaCameraModel,
        _BWModel,
        ShutterType,
        _RefPoly,
        FThetaPolynomialType,
        _NcoreRP,
        AnglePolyType,
    ) = stubbed_converters
    cam = _make_ftheta(FThetaCameraModel, ShutterType, ref_poly=AnglePolyType.ANGLE_TO_PIXELDIST)
    mod.CameraModelConverter.convert(cam)
    ftheta_call = next(c for c in captured["calls"] if c[0] == "FThetaProjection")
    assert ftheta_call[1]["reference_poly"] == FThetaPolynomialType.FORWARD


def test_convert_ftheta_pixeldist_to_angle_uses_backward_polynomial(stubbed_converters):
    (
        mod,
        captured,
        FThetaCameraModel,
        _BWModel,
        ShutterType,
        _RefPoly,
        FThetaPolynomialType,
        _NcoreRP,
        AnglePolyType,
    ) = stubbed_converters
    cam = _make_ftheta(FThetaCameraModel, ShutterType, ref_poly=AnglePolyType.PIXELDIST_TO_ANGLE)
    mod.CameraModelConverter.convert(cam)
    ftheta_call = next(c for c in captured["calls"] if c[0] == "FThetaProjection")
    assert ftheta_call[1]["reference_poly"] == FThetaPolynomialType.BACKWARD


def test_convert_unsupported_camera_type_raises(stubbed_converters):
    """Any non-FTheta camera model triggers ``TypeError`` — OpenCVPinhole /
    OpenCVFisheye are intentionally not supported on the input side."""
    (mod, *_) = stubbed_converters

    class _Mystery:
        pass

    with pytest.raises(TypeError, match="Unsupported camera model type"):
        mod.CameraModelConverter.convert(_Mystery())


def test_convert_default_device_is_cpu(stubbed_converters):
    mod = stubbed_converters[0]
    captured = stubbed_converters[1]
    FThetaCameraModel = stubbed_converters[2]
    ShutterType = stubbed_converters[4]
    AnglePolyType = stubbed_converters[8]
    cam = _make_ftheta(FThetaCameraModel, ShutterType, ref_poly=AnglePolyType.ANGLE_TO_PIXELDIST)
    mod.CameraModelConverter.convert(cam)
    ftheta_call = next(c for c in captured["calls"] if c[0] == "FThetaProjection")
    assert ftheta_call[1]["principal_point"].device == torch.device("cpu")


def test_convert_explicit_device_argument_is_propagated(stubbed_converters):
    mod = stubbed_converters[0]
    captured = stubbed_converters[1]
    FThetaCameraModel = stubbed_converters[2]
    ShutterType = stubbed_converters[4]
    AnglePolyType = stubbed_converters[8]
    cam = _make_ftheta(FThetaCameraModel, ShutterType, ref_poly=AnglePolyType.ANGLE_TO_PIXELDIST)
    mod.CameraModelConverter.convert(cam, device=torch.device("cpu"))
    ftheta_call = next(c for c in captured["calls"] if c[0] == "FThetaProjection")
    assert ftheta_call[1]["principal_point"].device.type == "cpu"


# ---------------------------------------------------------------------------
# _convert_external_distortion branches
# ---------------------------------------------------------------------------


def test_external_distortion_none_returns_no_external_distortion(stubbed_converters):
    mod = stubbed_converters[0]
    FThetaCameraModel = stubbed_converters[2]
    ShutterType = stubbed_converters[4]
    AnglePolyType = stubbed_converters[8]
    cam = _make_ftheta(FThetaCameraModel, ShutterType, ref_poly=AnglePolyType.ANGLE_TO_PIXELDIST)
    result = mod.CameraModelConverter.convert(cam)
    # NoExternalDistortion lives in instant_nurec's in-tree
    # kernel_types module.
    from instant_nurec.utils.sensors.kernel_types import NoExternalDistortion

    assert isinstance(result.external_distortion, NoExternalDistortion)


def test_external_distortion_bivariate_windshield_branch(stubbed_converters):
    (
        mod,
        _captured,
        FThetaCameraModel,
        BivariateWindshieldModel,
        ShutterType,
        ReferencePolynomial,
        _FThetaPolyType,
        _NcoreRP,
        AnglePolyType,
    ) = stubbed_converters
    cam = _attach_bivariate(
        _make_ftheta(FThetaCameraModel, ShutterType, ref_poly=AnglePolyType.ANGLE_TO_PIXELDIST),
        BivariateWindshieldModel,
    )
    result = mod.CameraModelConverter.convert(cam)
    from instant_nurec.utils.sensors.kernel_types import BivariateWindshieldDistortion

    assert isinstance(result.external_distortion, BivariateWindshieldDistortion)
    # FORWARD on the ncore side maps to ReferencePolynomial.FORWARD on the kernel side.
    assert result.external_distortion.reference_polynomial == ReferencePolynomial.FORWARD


def test_external_distortion_bivariate_backward_reference_polynomial(stubbed_converters):
    """If ncore reports the inverse reference polynomial, the kernel-side
    enum value should flip to BACKWARD."""
    (
        mod,
        _captured,
        FThetaCameraModel,
        BivariateWindshieldModel,
        ShutterType,
        ReferencePolynomial,
        _FThetaPolyType,
        _NcoreRP,
        AnglePolyType,
    ) = stubbed_converters
    cam = _attach_bivariate(
        _make_ftheta(FThetaCameraModel, ShutterType, ref_poly=AnglePolyType.ANGLE_TO_PIXELDIST),
        BivariateWindshieldModel,
    )
    cam.external_distortion.reference_poly = "BACKWARD"
    result = mod.CameraModelConverter.convert(cam)
    assert result.external_distortion.reference_polynomial == ReferencePolynomial.BACKWARD


def test_external_distortion_bivariate_duck_typed_uses_reference_polynomial_attr(
    stubbed_converters,
):
    """Parameter-object-like windshield models reach the converter via
    ``_looks_like_bivariate_windshield`` duck typing; some loaders spell the
    field ``reference_polynomial`` rather than ``reference_poly``. Both
    spellings must be honored so the converter and ray_gen paths agree."""
    (
        mod,
        _captured,
        FThetaCameraModel,
        _BWModel,
        ShutterType,
        ReferencePolynomial,
        _FThetaPolyType,
        _NcoreRP,
        AnglePolyType,
    ) = stubbed_converters
    cam = _make_ftheta(FThetaCameraModel, ShutterType, ref_poly=AnglePolyType.ANGLE_TO_PIXELDIST)
    duck = types.SimpleNamespace(
        horizontal_poly=torch.zeros(3),
        vertical_poly=torch.zeros(3),
        horizontal_poly_inverse=torch.zeros(3),
        vertical_poly_inverse=torch.zeros(3),
        reference_polynomial="FORWARD",
    )
    cam.external_distortion = duck
    result = mod.CameraModelConverter.convert(cam)
    assert result.external_distortion.reference_polynomial == ReferencePolynomial.FORWARD


def test_external_distortion_bivariate_duck_typed_missing_reference_raises(
    stubbed_converters,
):
    """A duck-typed windshield model with neither ``reference_poly`` nor
    ``reference_polynomial`` set must fail loud, mirroring the ray_gen path."""
    (
        mod,
        _captured,
        FThetaCameraModel,
        _BWModel,
        ShutterType,
        _ReferencePolynomial,
        _FThetaPolyType,
        _NcoreRP,
        AnglePolyType,
    ) = stubbed_converters
    cam = _make_ftheta(FThetaCameraModel, ShutterType, ref_poly=AnglePolyType.ANGLE_TO_PIXELDIST)
    duck = types.SimpleNamespace(
        horizontal_poly=torch.zeros(3),
        vertical_poly=torch.zeros(3),
        horizontal_poly_inverse=torch.zeros(3),
        vertical_poly_inverse=torch.zeros(3),
    )
    cam.external_distortion = duck
    with pytest.raises(ValueError, match="unrecognized reference_polynomial"):
        mod.CameraModelConverter.convert(cam)


def test_external_distortion_bivariate_unrecognized_reference_raises(stubbed_converters):
    """Unrecognized reference_poly strings should fail loud, not silently
    fall back to BACKWARD."""
    (
        mod,
        _captured,
        FThetaCameraModel,
        BivariateWindshieldModel,
        ShutterType,
        _ReferencePolynomial,
        _FThetaPolyType,
        _NcoreRP,
        AnglePolyType,
    ) = stubbed_converters
    cam = _attach_bivariate(
        _make_ftheta(FThetaCameraModel, ShutterType, ref_poly=AnglePolyType.ANGLE_TO_PIXELDIST),
        BivariateWindshieldModel,
    )
    cam.external_distortion.reference_poly = "BACKWARD_OR_OTHER"
    with pytest.raises(ValueError, match="unrecognized reference_polynomial"):
        mod.CameraModelConverter.convert(cam)


def test_external_distortion_unrecognized_type_returns_none(stubbed_converters):
    """If external_distortion is set but is not a BivariateWindshieldModel,
    the SUT falls through to the default ``return NoExternalDistortion()``."""
    mod = stubbed_converters[0]
    FThetaCameraModel = stubbed_converters[2]
    ShutterType = stubbed_converters[4]
    AnglePolyType = stubbed_converters[8]
    cam = _make_ftheta(FThetaCameraModel, ShutterType, ref_poly=AnglePolyType.ANGLE_TO_PIXELDIST)
    cam.external_distortion = object()  # not None, not BivariateWindshieldModel
    result = mod.CameraModelConverter.convert(cam)
    from instant_nurec.utils.sensors.kernel_types import NoExternalDistortion

    assert isinstance(result.external_distortion, NoExternalDistortion)


def test_module_exports_pose_and_dynamic_pose(stubbed_converters):
    """``__all__`` includes ``Pose`` and ``DynamicPose`` re-exported from
    instant_nurec's in-tree ``kernel_types`` ."""
    (mod, *_) = stubbed_converters
    from instant_nurec.utils.sensors.kernel_types import Pose, DynamicPose

    assert "Pose" in mod.__all__
    assert "DynamicPose" in mod.__all__
    assert mod.Pose is Pose
    assert mod.DynamicPose is DynamicPose
