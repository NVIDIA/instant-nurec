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

"""Tests for NCore/HF bivariate windshield parameter handling."""

from __future__ import annotations

import sys
import types

from types import SimpleNamespace

import pytest
import torch


try:
    import ncore  # noqa: F401
except ModuleNotFoundError:
    ncore_mod = types.ModuleType("ncore")
    sensors_ncore = types.ModuleType("ncore.sensors")

    class FThetaCameraModel:
        pass

    sensors_ncore.FThetaCameraModel = FThetaCameraModel
    ncore_mod.sensors = sensors_ncore
    sys.modules["ncore"] = ncore_mod
    sys.modules["ncore.sensors"] = sensors_ncore

from instant_nurec.utils.sensors.kernel_types import (
    BivariateWindshieldDistortion,
    DynamicPose,
    ExternalDistortion,
    FThetaPolynomialType,
    FThetaProjection,
    NoExternalDistortion,
    Pose,
    ReferencePolynomial,
    ShutterType,
)
from instant_nurec.utils.sensors.ray_gen import (
    _ftheta_image_points_to_camera_rays,
    _ncore_external_distortion_to_distortion,
    camera_rays_to_image_points,
    image_points_to_world_rays_shutter_pose,
)


def _clip_windshield_coefficients() -> dict[str, list[float] | str]:
    """Windshield coefficients from the HF/ncorev4 smoke clip front-wide camera."""
    return {
        "horizontal_poly": [
            -0.001005438040010631,
            1.0016967058181763,
            0.00032004225067794323,
            0.0001555727212689817,
            0.004575578961521387,
            0.0014987860340625048,
        ],
        "horizontal_poly_inverse": [
            0.0010052943835034966,
            0.9983381032943726,
            -0.00031450000824406743,
            -0.00013237200619187206,
            -0.004510311875492334,
            -0.001468065893277526,
        ],
        "reference_poly": "FORWARD",
        "vertical_poly": [
            0.007004346698522568,
            0.0003755314683075994,
            -0.00244372827000916,
            0.0002474650682415813,
            0.0008841193630360067,
            1.0131069421768188,
            0.001544533995911479,
            0.00980763416737318,
            0.0006882359157316387,
            0.013353252783417702,
            0.0012939744628965855,
            0.010011252947151661,
            0.006461413111537695,
            -0.0024649337865412235,
            -0.0024882371071726084,
        ],
        "vertical_poly_inverse": [
            -0.006918848492205143,
            -0.0003612338041421026,
            0.0024752432946115732,
            -0.00023147383762989193,
            -0.000870318675879389,
            0.9872006773948669,
            -0.0014815711183473468,
            -0.009318409487605095,
            -0.00063826993573457,
            -0.012637471780180931,
            -0.0011527703609317541,
            -0.0093322629109025,
            -0.005527016706764698,
            0.0025377252604812384,
            0.0029134664218872786,
        ],
    }


def _distortion(reference_polynomial: ReferencePolynomial = ReferencePolynomial.FORWARD) -> BivariateWindshieldDistortion:
    coeffs = _clip_windshield_coefficients()
    return BivariateWindshieldDistortion.from_components(
        h_poly=torch.tensor(coeffs["horizontal_poly"], dtype=torch.float32),
        v_poly=torch.tensor(coeffs["vertical_poly"], dtype=torch.float32),
        h_poly_inv=torch.tensor(coeffs["horizontal_poly_inverse"], dtype=torch.float32),
        v_poly_inv=torch.tensor(coeffs["vertical_poly_inverse"], dtype=torch.float32),
        reference_polynomial=reference_polynomial,
    )


def _projection() -> FThetaProjection:
    return FThetaProjection.from_components(
        principal_point=torch.tensor([320.0, 240.0]),
        fw_poly=torch.tensor([0.0, 500.0]),
        bw_poly=torch.tensor([0.0, 0.002]),
        A=torch.eye(2),
        Ainv=torch.eye(2),
        dfw_poly=torch.tensor([500.0]),
        dbw_poly=torch.tensor([0.002]),
        reference_poly=FThetaPolynomialType.FORWARD,
        max_angle=1.5,
        newton_iterations=4,
        min_2d_norm=1e-6,
    )


def _reference_eval_poly_2d(
    coefficients: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    order: int,
) -> torch.Tensor:
    y_coeffs = []
    start_idx = 0
    for inner_order in range(order, -1, -1):
        result = torch.zeros_like(x)
        for coeff in reversed(coefficients[start_idx : start_idx + inner_order + 1]):
            result = result * x + coeff
        y_coeffs.append(result)
        start_idx += inner_order + 1

    result = torch.zeros_like(y)
    for coeff in reversed(y_coeffs):
        result = result * y + coeff
    return result


def _reference_distort(
    rays: torch.Tensor,
    h_poly: torch.Tensor,
    v_poly: torch.Tensor,
    h_order: int,
    v_order: int,
) -> torch.Tensor:
    ray_norm = torch.nn.functional.normalize(rays, dim=-1)
    phi = torch.asin(torch.clamp(ray_norm[..., 0], -1.0, 1.0))
    theta = torch.asin(torch.clamp(ray_norm[..., 1], -1.0, 1.0))
    x = torch.sin(_reference_eval_poly_2d(h_poly, phi, theta, h_order))
    y = torch.sin(_reference_eval_poly_2d(v_poly, phi, theta, v_order))
    z = torch.sqrt(torch.clamp(1.0 - x * x - y * y, min=0.0, max=1.0))
    z = z * torch.where(ray_norm[..., 2] < 0.0, -1.0, 1.0)
    return torch.stack([x, y, z], dim=-1)


def test_clip_windshield_coefficients_validate_triangular_orders() -> None:
    distortion = _distortion()

    assert distortion.reference_polynomial == ReferencePolynomial.FORWARD
    assert distortion.h_poly_degree == 2
    assert distortion.v_poly_degree == 4


def test_clip_windshield_distorts_like_nre_reference() -> None:
    distortion = _distortion()
    rays = torch.tensor(
        [
            [0.15, 0.04, 1.0],
            [-0.10, 0.12, 1.0],
            [0.05, -0.08, -1.0],
        ],
        dtype=torch.float32,
    )

    expected_distorted = _reference_distort(
        rays,
        distortion.h_poly,
        distortion.v_poly,
        distortion.h_poly_degree,
        distortion.v_poly_degree,
    )
    torch.testing.assert_close(distortion.distort_camera_rays(rays), expected_distorted)

    expected_undistorted = _reference_distort(
        rays,
        distortion.h_poly_inv,
        distortion.v_poly_inv,
        distortion.h_poly_degree,
        distortion.v_poly_degree,
    )
    torch.testing.assert_close(distortion.undistort_camera_rays(rays), expected_undistorted)


def test_backward_reference_swaps_forward_and_inverse_pairs() -> None:
    distortion = _distortion(reference_polynomial=ReferencePolynomial.BACKWARD)
    rays = torch.tensor([[0.15, 0.04, 1.0]], dtype=torch.float32)

    torch.testing.assert_close(
        distortion.distort_camera_rays(rays),
        _reference_distort(
            rays,
            distortion.h_poly_inv,
            distortion.v_poly_inv,
            distortion.h_poly_degree,
            distortion.v_poly_degree,
        ),
    )
    torch.testing.assert_close(
        distortion.undistort_camera_rays(rays),
        _reference_distort(
            rays,
            distortion.h_poly,
            distortion.v_poly,
            distortion.h_poly_degree,
            distortion.v_poly_degree,
        ),
    )


def test_windshield_inverse_polynomials_must_match_forward_order() -> None:
    with pytest.raises(ValueError, match="same bivariate order"):
        BivariateWindshieldDistortion.from_components(
            h_poly=torch.ones(3),
            v_poly=torch.ones(3),
            h_poly_inv=torch.ones(6),
            v_poly_inv=torch.ones(3),
            reference_polynomial=ReferencePolynomial.FORWARD,
        )


def test_image_points_to_world_rays_applies_windshield_undistortion() -> None:
    projection = _projection()
    distortion = _distortion()
    image_points = torch.tensor([[330.0, 250.0], [350.0, 260.0]])
    dynamic_pose = DynamicPose(
        start_pose=Pose(translation=torch.zeros(3), rotation=torch.tensor([0.0, 0.0, 0.0, 1.0])),
        end_pose=Pose(translation=torch.zeros(3), rotation=torch.tensor([0.0, 0.0, 0.0, 1.0])),
    )

    world_rays, _, _, _ = image_points_to_world_rays_shutter_pose(
        image_points=image_points,
        projection=projection,
        external_distortion=distortion,
        resolution=(640, 480),
        shutter_type=ShutterType.ROLLING_TOP_TO_BOTTOM,
        dynamic_pose=dynamic_pose,
    )

    camera_rays = _ftheta_image_points_to_camera_rays(image_points, projection)
    expected_directions = distortion.undistort_camera_rays(camera_rays)
    torch.testing.assert_close(world_rays[:, :3], torch.zeros((2, 3)))
    torch.testing.assert_close(world_rays[:, 3:], expected_directions, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# BivariateWindshieldDistortion validation/coercion branches
# ---------------------------------------------------------------------------


def test_windshield_rejects_non_1d_coefficients() -> None:
    with pytest.raises(ValueError, match="h_poly must be 1D"):
        BivariateWindshieldDistortion.from_components(
            h_poly=torch.ones((3, 2)),
            v_poly=torch.ones(3),
            h_poly_inv=torch.ones((3, 2)),
            v_poly_inv=torch.ones(3),
            reference_polynomial=ReferencePolynomial.FORWARD,
        )


def test_windshield_rejects_empty_coefficients() -> None:
    with pytest.raises(ValueError, match="at least one coefficient"):
        BivariateWindshieldDistortion.from_components(
            h_poly=torch.empty(0),
            v_poly=torch.ones(3),
            h_poly_inv=torch.empty(0),
            v_poly_inv=torch.ones(3),
            reference_polynomial=ReferencePolynomial.FORWARD,
        )


def test_windshield_rejects_non_triangular_coefficient_count() -> None:
    # 4 coefficients lie strictly between triangular sums 3 (order=1) and 6
    # (order=2), so no order satisfies (order+1)(order+2)/2 == 4.
    with pytest.raises(ValueError, match="triangular bivariate polynomial storage"):
        BivariateWindshieldDistortion.from_components(
            h_poly=torch.ones(4),
            v_poly=torch.ones(3),
            h_poly_inv=torch.ones(4),
            v_poly_inv=torch.ones(3),
            reference_polynomial=ReferencePolynomial.FORWARD,
        )


def test_windshield_v_poly_inv_order_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="v_poly_inv must have the same bivariate order"):
        BivariateWindshieldDistortion.from_components(
            h_poly=torch.ones(3),
            v_poly=torch.ones(3),
            h_poly_inv=torch.ones(3),
            v_poly_inv=torch.ones(6),
            reference_polynomial=ReferencePolynomial.FORWARD,
        )


def test_windshield_explicit_h_poly_degree_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="h_poly_degree does not match"):
        BivariateWindshieldDistortion(
            h_poly=torch.ones(3),  # order 1
            v_poly=torch.ones(3),
            h_poly_inv=torch.ones(3),
            v_poly_inv=torch.ones(3),
            reference_polynomial=ReferencePolynomial.FORWARD,
            h_poly_degree=2,  # mismatched
        )


def test_windshield_explicit_v_poly_degree_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="v_poly_degree does not match"):
        BivariateWindshieldDistortion(
            h_poly=torch.ones(3),
            v_poly=torch.ones(3),  # order 1
            h_poly_inv=torch.ones(3),
            v_poly_inv=torch.ones(3),
            reference_polynomial=ReferencePolynomial.FORWARD,
            v_poly_degree=2,  # mismatched
        )


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        (ReferencePolynomial.FORWARD, ReferencePolynomial.FORWARD),
        (ReferencePolynomial.BACKWARD, ReferencePolynomial.BACKWARD),
        ("FORWARD", ReferencePolynomial.FORWARD),
        ("backward", ReferencePolynomial.BACKWARD),
        ("ReferencePolynomial.BACKWARD", ReferencePolynomial.BACKWARD),
        (0, ReferencePolynomial.FORWARD),
        (1, ReferencePolynomial.BACKWARD),
        ("0", ReferencePolynomial.FORWARD),
        (SimpleNamespace(name="FORWARD"), ReferencePolynomial.FORWARD),
        (SimpleNamespace(value=1), ReferencePolynomial.BACKWARD),
    ],
)
def test_windshield_reference_polynomial_coercion_variants(raw_value, expected) -> None:
    distortion = BivariateWindshieldDistortion.from_components(
        h_poly=torch.ones(3),
        v_poly=torch.ones(3),
        h_poly_inv=torch.ones(3),
        v_poly_inv=torch.ones(3),
        reference_polynomial=raw_value,
    )
    assert distortion.reference_polynomial == expected


def test_windshield_reference_polynomial_unrecognized_raises() -> None:
    with pytest.raises(ValueError, match="unrecognized reference_polynomial"):
        BivariateWindshieldDistortion.from_components(
            h_poly=torch.ones(3),
            v_poly=torch.ones(3),
            h_poly_inv=torch.ones(3),
            v_poly_inv=torch.ones(3),
            reference_polynomial="SIDEWAYS",
        )


def test_windshield_reference_polynomial_none_raises() -> None:
    with pytest.raises(ValueError, match="unrecognized reference_polynomial"):
        BivariateWindshieldDistortion.from_components(
            h_poly=torch.ones(3),
            v_poly=torch.ones(3),
            h_poly_inv=torch.ones(3),
            v_poly_inv=torch.ones(3),
            reference_polynomial=None,
        )


# ---------------------------------------------------------------------------
# ray_gen._ncore_external_distortion_to_distortion branches
# ---------------------------------------------------------------------------


def test_ncore_external_distortion_missing_returns_no_external() -> None:
    ncore_params = SimpleNamespace()  # no external_distortion / _parameters attr
    result = _ncore_external_distortion_to_distortion(
        ncore_params, torch.device("cpu"), torch.float32
    )
    assert isinstance(result, NoExternalDistortion)


def test_ncore_external_distortion_explicit_none_returns_no_external() -> None:
    ncore_params = SimpleNamespace(
        external_distortion_parameters=None, external_distortion=None
    )
    result = _ncore_external_distortion_to_distortion(
        ncore_params, torch.device("cpu"), torch.float32
    )
    assert isinstance(result, NoExternalDistortion)


def test_ncore_external_distortion_falls_back_to_external_distortion_attr() -> None:
    coeffs = _clip_windshield_coefficients()
    external_distortion = SimpleNamespace(
        horizontal_poly=coeffs["horizontal_poly"],
        vertical_poly=coeffs["vertical_poly"],
        horizontal_poly_inverse=coeffs["horizontal_poly_inverse"],
        vertical_poly_inverse=coeffs["vertical_poly_inverse"],
        reference_polynomial=0,  # ncore newer-loader spelling, integer enum value
    )
    ncore_params = SimpleNamespace(
        external_distortion_parameters=None,
        external_distortion=external_distortion,
    )
    result = _ncore_external_distortion_to_distortion(
        ncore_params, torch.device("cpu"), torch.float32
    )
    assert isinstance(result, BivariateWindshieldDistortion)
    # The fallback path reads `reference_polynomial`, not `reference_poly`.
    assert result.reference_polynomial == ReferencePolynomial.FORWARD


def test_ncore_external_distortion_missing_reference_polynomial_raises() -> None:
    """Neither ``reference_poly`` nor ``reference_polynomial`` set — must fail
    loud rather than silently defaulting to FORWARD."""
    coeffs = _clip_windshield_coefficients()
    external_distortion = SimpleNamespace(
        horizontal_poly=coeffs["horizontal_poly"],
        vertical_poly=coeffs["vertical_poly"],
        horizontal_poly_inverse=coeffs["horizontal_poly_inverse"],
        vertical_poly_inverse=coeffs["vertical_poly_inverse"],
    )
    ncore_params = SimpleNamespace(external_distortion_parameters=external_distortion)
    with pytest.raises(ValueError, match="unrecognized reference_polynomial"):
        _ncore_external_distortion_to_distortion(
            ncore_params, torch.device("cpu"), torch.float32
        )


def test_ncore_external_distortion_unsupported_type_raises() -> None:
    bad_distortion = SimpleNamespace(some_other_field=1.0)
    ncore_params = SimpleNamespace(external_distortion_parameters=bad_distortion)
    with pytest.raises(NotImplementedError, match="unsupported external distortion parameters"):
        _ncore_external_distortion_to_distortion(
            ncore_params, torch.device("cpu"), torch.float32
        )


# ---------------------------------------------------------------------------
# ray_gen — external_distortion fallback branches
# ---------------------------------------------------------------------------


class _CustomExternalDistortion(ExternalDistortion):
    pass


def test_image_points_to_world_rays_rejects_unsupported_external_distortion() -> None:
    projection = _projection()
    dynamic_pose = DynamicPose(
        start_pose=Pose(translation=torch.zeros(3), rotation=torch.tensor([0.0, 0.0, 0.0, 1.0])),
        end_pose=Pose(translation=torch.zeros(3), rotation=torch.tensor([0.0, 0.0, 0.0, 1.0])),
    )
    with pytest.raises(NotImplementedError, match="unsupported external distortion"):
        image_points_to_world_rays_shutter_pose(
            image_points=torch.tensor([[320.0, 240.0]]),
            projection=projection,
            external_distortion=_CustomExternalDistortion(),
            resolution=(640, 480),
            shutter_type=ShutterType.GLOBAL,
            dynamic_pose=dynamic_pose,
        )


def test_camera_rays_to_image_points_rejects_unsupported_external_distortion() -> None:
    coeffs = _clip_windshield_coefficients()
    # Construct an ncore-params-like object whose external_distortion has none
    # of the windshield coefficient names — _ncore_external_distortion_to_distortion
    # raises NotImplementedError before we ever reach the fallback in
    # camera_rays_to_image_points.
    bad_distortion = SimpleNamespace(some_other_field=1.0)
    ncore_params = SimpleNamespace(
        angle_to_pixeldist_poly=[0.0, 500.0],
        pixeldist_to_angle_poly=[0.0, 0.002],
        linear_cde=[1.0, 0.0, 0.0],
        principal_point=[319.5, 239.5],
        reference_poly="ANGLE_TO_PIXELDIST",
        max_angle=1.5,
        resolution=[640, 480],
        external_distortion_parameters=bad_distortion,
    )
    rays = torch.tensor([[0.15, 0.04, 1.0]], dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="unsupported external distortion"):
        camera_rays_to_image_points(ncore_params, rays)
    # Silence unused-fixture lint by referring to coeffs.
    assert coeffs["reference_poly"] == "FORWARD"


def test_camera_rays_to_image_points_applies_ncore_parameter_windshield() -> None:
    coeffs = _clip_windshield_coefficients()
    external_distortion_parameters = SimpleNamespace(
        horizontal_poly=coeffs["horizontal_poly"],
        vertical_poly=coeffs["vertical_poly"],
        horizontal_poly_inverse=coeffs["horizontal_poly_inverse"],
        vertical_poly_inverse=coeffs["vertical_poly_inverse"],
        reference_poly=coeffs["reference_poly"],
    )
    ncore_params = SimpleNamespace(
        angle_to_pixeldist_poly=[0.0, 500.0],
        pixeldist_to_angle_poly=[0.0, 0.002],
        linear_cde=[1.0, 0.0, 0.0],
        principal_point=[319.5, 239.5],
        reference_poly="ANGLE_TO_PIXELDIST",
        max_angle=1.5,
        resolution=[640, 480],
        external_distortion_parameters=external_distortion_parameters,
    )
    rays = torch.tensor([[0.15, 0.04, 1.0], [-0.10, 0.12, 1.0]], dtype=torch.float32)

    image_points_return = camera_rays_to_image_points(ncore_params, rays)

    expected = camera_rays_to_image_points(_projection(), _distortion().distort_camera_rays(rays))
    torch.testing.assert_close(image_points_return.image_points, expected.image_points)
    assert image_points_return.valid_flag.all()
