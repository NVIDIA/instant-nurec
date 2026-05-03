"""Branch-coverage tests for ``instant_nurec.utils.sensors.ncore_sensors_converters``.

The module converts ncore camera models into the in-tree dataclass
parameter types (after Phase A.2/A.6 these live in
``instant_nurec.utils.sensors._kernel_types``). The ncore types are
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
    # After Phase A.6 the converter pulls dataclasses from
    # ``instant_nurec.utils.sensors._kernel_types`` (in-tree). We use
    # the real types and wrap their ``from_components`` with a capture
    # shim so the existing call-arg assertions still work.
    captured: dict = {}

    # ncore stubs (needed BEFORE _kernel_types import — the converter package
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
    # instant_nurec.utils.types pulls these unions in too.
    data_mod.ConcreteCameraModelParametersUnion = object
    data_mod.ConcreteLidarModelParametersUnion = object

    class _ShutterTypeNcore:
        # Match the in-tree _kernel_types.ShutterType IntEnum values (1-5);
        # the converter does ShutterType(camera_model.shutter_type.value).
        ROLLING = type("RollingTag", (), {"value": 1})()

    data_mod.ShutterType = _ShutterTypeNcore

    class CameraModel:
        pass

    class OpenCVPinholeCameraModel(CameraModel):
        pass

    class OpenCVFisheyeCameraModel(CameraModel):
        pass

    class FThetaCameraModel(CameraModel):
        pass

    class BivariateWindshieldModel:
        pass

    sensors_ncore.CameraModel = CameraModel
    sensors_ncore.OpenCVPinholeCameraModel = OpenCVPinholeCameraModel
    sensors_ncore.OpenCVFisheyeCameraModel = OpenCVFisheyeCameraModel
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
        "instant_nurec.utils.sensors._kernel_types",
        "instant_nurec.utils.sensors._image_points_to_world_rays_torch",
        "instant_nurec.utils.sensors",
        "instant_nurec.utils.types",
    ):
        monkeypatch.delitem(sys.modules, cached, raising=False)

    import importlib

    # Now safe to import the in-tree types (converter package __init__ pulls
    # sensors.py → ncore.data, which is stubbed above).
    kt = importlib.import_module("instant_nurec.utils.sensors._kernel_types")

    def _wrap_from_components(real_cls, captured_name):
        original = real_cls.from_components

        def _capturing(*args, **kwargs):
            captured.setdefault("calls", []).append((captured_name, dict(kwargs)))
            return original(*args, **kwargs)

        return _capturing

    monkeypatch.setattr(
        kt.OpenCVPinholeProjection,
        "from_components",
        _wrap_from_components(kt.OpenCVPinholeProjection, "OpenCVPinholeProjection"),
    )
    monkeypatch.setattr(
        kt.OpenCVFisheyeProjection,
        "from_components",
        _wrap_from_components(kt.OpenCVFisheyeProjection, "OpenCVFisheyeProjection"),
    )
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

    OpenCVPinholeProjection = kt.OpenCVPinholeProjection  # noqa: F841
    OpenCVFisheyeProjection = kt.OpenCVFisheyeProjection  # noqa: F841
    FThetaProjection = kt.FThetaProjection  # noqa: F841
    BivariateWindshieldDistortion = kt.BivariateWindshieldDistortion  # noqa: F841
    FThetaPolynomialType = kt.FThetaPolynomialType
    ReferencePolynomial = kt.ReferencePolynomial
    ShutterType = kt.ShutterType

    converters = importlib.import_module("instant_nurec.utils.sensors.ncore_sensors_converters")
    return (
        converters,
        captured,
        OpenCVPinholeCameraModel,
        OpenCVFisheyeCameraModel,
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


def _make_pinhole(model_cls, ShutterType, with_distortion=False, BivariateModel=None):
    m = model_cls()
    m.focal_length = torch.tensor([100.0, 100.0])
    m.principal_point = torch.tensor([320.0, 240.0])
    m.radial_coeffs = torch.zeros(6)
    m.tangential_coeffs = torch.zeros(2)
    m.thin_prism_coeffs = torch.zeros(4)
    m.resolution = torch.tensor([640, 480])
    m.shutter_type = _shutter_obj(ShutterType.ROLLING_TOP_TO_BOTTOM)
    m.external_distortion = None
    if with_distortion:
        d = BivariateModel()
        d.horizontal_poly = torch.zeros(5)
        d.vertical_poly = torch.zeros(5)
        d.horizontal_poly_inverse = torch.zeros(5)
        d.vertical_poly_inverse = torch.zeros(5)
        d.reference_poly = "FORWARD"
        m.external_distortion = d
    return m


def _make_fisheye(model_cls, ShutterType):
    m = model_cls()
    m.focal_length = torch.tensor([100.0, 100.0])
    m.principal_point = torch.tensor([320.0, 240.0])
    # forward_poly[[3, 5, 7, 9]] needs at least 10 elements.
    m.forward_poly = torch.arange(10, dtype=torch.float32)
    m.resolution = torch.tensor([640, 480])
    m.max_angle = 1.5
    m.newton_iterations = 5
    m.shutter_type = _shutter_obj(ShutterType.ROLLING_TOP_TO_BOTTOM)
    m.external_distortion = None
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


def test_convert_pinhole_returns_pinhole_projection(stubbed_converters):
    (
        mod,
        captured,
        OpenCVPinholeCameraModel,
        _Fisheye,
        _Ftheta,
        _BWModel,
        ShutterType,
        *_,
    ) = stubbed_converters
    cam = _make_pinhole(OpenCVPinholeCameraModel, ShutterType)
    result = mod.CameraModelConverter.convert(cam)
    names = [c[0] for c in captured["calls"]]
    assert "OpenCVPinholeProjection" in names
    assert result.resolution == (640, 480)
    assert result.shutter_type == ShutterType.ROLLING_TOP_TO_BOTTOM


def test_convert_fisheye_slices_forward_poly(stubbed_converters):
    (
        mod,
        captured,
        _Pinhole,
        OpenCVFisheyeCameraModel,
        _Ftheta,
        _BWModel,
        ShutterType,
        *_,
    ) = stubbed_converters
    cam = _make_fisheye(OpenCVFisheyeCameraModel, ShutterType)
    result = mod.CameraModelConverter.convert(cam)
    fisheye_call = next(c for c in captured["calls"] if c[0] == "OpenCVFisheyeProjection")
    fwd_poly = fisheye_call[1]["forward_poly"]
    # Indices [3,5,7,9] of arange(10) are [3,5,7,9].
    assert torch.equal(fwd_poly, torch.tensor([3.0, 5.0, 7.0, 9.0]))
    assert isinstance(result, mod.CameraModelConverterResult)


def test_convert_ftheta_angle_to_pixeldist_uses_forward_polynomial(stubbed_converters):
    (
        mod,
        captured,
        _Pinhole,
        _Fisheye,
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
        _Pinhole,
        _Fisheye,
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
    (mod, *_) = stubbed_converters

    class _Mystery:
        pass

    with pytest.raises(TypeError, match="Unsupported camera model type"):
        mod.CameraModelConverter.convert(_Mystery())


def test_convert_default_device_is_cpu(stubbed_converters):
    mod = stubbed_converters[0]
    captured = stubbed_converters[1]
    OpenCVPinholeCameraModel = stubbed_converters[2]
    ShutterType = stubbed_converters[6]
    cam = _make_pinhole(OpenCVPinholeCameraModel, ShutterType)
    mod.CameraModelConverter.convert(cam)
    pinhole_call = next(c for c in captured["calls"] if c[0] == "OpenCVPinholeProjection")
    assert pinhole_call[1]["focal_length"].device == torch.device("cpu")


def test_convert_explicit_device_argument_is_propagated(stubbed_converters):
    mod = stubbed_converters[0]
    captured = stubbed_converters[1]
    OpenCVPinholeCameraModel = stubbed_converters[2]
    ShutterType = stubbed_converters[6]
    cam = _make_pinhole(OpenCVPinholeCameraModel, ShutterType)
    mod.CameraModelConverter.convert(cam, device=torch.device("cpu"))
    pinhole_call = next(c for c in captured["calls"] if c[0] == "OpenCVPinholeProjection")
    assert pinhole_call[1]["focal_length"].device.type == "cpu"


# ---------------------------------------------------------------------------
# _convert_external_distortion branches
# ---------------------------------------------------------------------------


def test_external_distortion_none_returns_no_external_distortion(stubbed_converters):
    mod = stubbed_converters[0]
    OpenCVPinholeCameraModel = stubbed_converters[2]
    ShutterType = stubbed_converters[6]
    cam = _make_pinhole(OpenCVPinholeCameraModel, ShutterType)
    result = mod.CameraModelConverter.convert(cam)
    # After Phase A.6 NoExternalDistortion lives in instant_nurec's in-tree
    # _kernel_types module.
    from instant_nurec.utils.sensors._kernel_types import NoExternalDistortion

    assert isinstance(result.external_distortion, NoExternalDistortion)


def test_external_distortion_bivariate_windshield_branch(stubbed_converters):
    (
        mod,
        captured,
        OpenCVPinholeCameraModel,
        _Fisheye,
        _Ftheta,
        BivariateWindshieldModel,
        ShutterType,
        ReferencePolynomial,
        *_extra,
    ) = stubbed_converters
    cam = _make_pinhole(
        OpenCVPinholeCameraModel,
        ShutterType,
        with_distortion=True,
        BivariateModel=BivariateWindshieldModel,
    )
    result = mod.CameraModelConverter.convert(cam)
    bw_call = next(c for c in captured["calls"] if c[0] == "BivariateWindshieldDistortion")
    # FORWARD on the ncore side maps to ReferencePolynomial.FORWARD on the kernel side.
    assert bw_call[1]["reference_polynomial"] == ReferencePolynomial.FORWARD
    from instant_nurec.utils.sensors._kernel_types import BivariateWindshieldDistortion

    assert isinstance(result.external_distortion, BivariateWindshieldDistortion)


def test_external_distortion_bivariate_backward_reference_polynomial(stubbed_converters):
    """If ncore reports the inverse reference polynomial, the kernel-side
    enum value should flip to BACKWARD (the else branch of the ternary)."""
    (
        mod,
        captured,
        OpenCVPinholeCameraModel,
        _Fisheye,
        _Ftheta,
        BivariateWindshieldModel,
        ShutterType,
        ReferencePolynomial,
        *_extra,
    ) = stubbed_converters
    cam = _make_pinhole(
        OpenCVPinholeCameraModel,
        ShutterType,
        with_distortion=True,
        BivariateModel=BivariateWindshieldModel,
    )
    cam.external_distortion.reference_poly = "BACKWARD_OR_OTHER"
    mod.CameraModelConverter.convert(cam)
    bw_call = next(c for c in captured["calls"] if c[0] == "BivariateWindshieldDistortion")
    assert bw_call[1]["reference_polynomial"] == ReferencePolynomial.BACKWARD


def test_external_distortion_unrecognized_type_returns_none(stubbed_converters):
    """If external_distortion is set but is not a BivariateWindshieldModel,
    the SUT falls through to the default ``return NoExternalDistortion()``."""
    mod = stubbed_converters[0]
    OpenCVPinholeCameraModel = stubbed_converters[2]
    ShutterType = stubbed_converters[6]
    cam = _make_pinhole(OpenCVPinholeCameraModel, ShutterType)
    cam.external_distortion = object()  # not None, not BivariateWindshieldModel
    result = mod.CameraModelConverter.convert(cam)
    from instant_nurec.utils.sensors._kernel_types import NoExternalDistortion

    assert isinstance(result.external_distortion, NoExternalDistortion)


def test_module_exports_pose_and_dynamic_pose(stubbed_converters):
    """``__all__`` includes ``Pose`` and ``DynamicPose`` re-exported from
    instant_nurec's in-tree ``_kernel_types`` (after Phase A.6)."""
    (mod, *_) = stubbed_converters
    from instant_nurec.utils.sensors._kernel_types import Pose, DynamicPose

    assert "Pose" in mod.__all__
    assert "DynamicPose" in mod.__all__
    assert mod.Pose is Pose
    assert mod.DynamicPose is DynamicPose
