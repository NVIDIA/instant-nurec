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

"""Pure-torch camera projection kernels:

* ``image_points_to_world_rays_shutter_pose`` — inverse projection +
  rolling-shutter pose interp
* ``camera_rays_to_image_points`` — forward camera projection (consumed
  by ``instant_nurec/utils/cubemap.py``).

FTheta is the supported projection model. External distortion supports
``NoExternalDistortion`` and NRE-compatible bivariate windshield models.
"""

from __future__ import annotations

import torch

from instant_nurec.utils.se3 import quat_xyzw_slerp, quat_xyzw_to_rotmat
from instant_nurec.utils.sensors.kernel_types import (
    BivariateWindshieldDistortion,
    DynamicPose,
    ExternalDistortion,
    FThetaPolynomialType,
    FThetaProjection,
    NoExternalDistortion,
    ShutterType,
)


def _eval_poly_horner(poly_coefficients: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Numerically-stable polynomial evaluation via Horner's method.

    Mirrors ``ncore.impl.sensors.common.eval_poly_horner``.
    """
    y = torch.zeros_like(x)
    for fi in torch.flip(poly_coefficients, dims=(0,)):
        y = y * x + fi
    return y


def _eval_poly_inverse_horner_newton(
    poly_coefficients: torch.Tensor,
    poly_derivative_coefficients: torch.Tensor,
    inverse_poly_approximation_coefficients: torch.Tensor,
    newton_iterations: int,
    y: torch.Tensor,
) -> torch.Tensor:
    """Newton-method inverse of a reference polynomial.

    Mirrors ``ncore.impl.sensors.common.eval_poly_inverse_horner_newton``.
    """
    x = _eval_poly_horner(inverse_poly_approximation_coefficients, y)
    for _ in range(newton_iterations):
        dfdx = _eval_poly_horner(poly_derivative_coefficients, x)
        residuals = _eval_poly_horner(poly_coefficients, x) - y
        x = x - residuals / dfdx
    return x


def _ftheta_image_points_to_camera_rays(
    image_points: torch.Tensor,
    projection: FThetaProjection,
) -> torch.Tensor:
    """FTheta inverse projection: image points → camera-frame rays.

    Mirrors ``ncore.impl.sensors.camera.FThetaCameraModel._image_points_to_camera_rays_impl``.
    """
    Ainv = projection.Ainv.to(device=image_points.device, dtype=image_points.dtype)
    pp = projection.principal_point.to(device=image_points.device, dtype=image_points.dtype)

    # Get f(theta)-weighted normalized 2d vectors (undoing the linear term A).
    image_points_dist = torch.einsum("ij,nj->ni", Ainv, image_points - pp)
    rdist = torch.linalg.norm(image_points_dist, dim=1, keepdim=True)

    bw_poly = projection.bw_poly.to(device=image_points.device, dtype=image_points.dtype)
    fw_poly = projection.fw_poly.to(device=image_points.device, dtype=image_points.dtype)
    dfw_poly = projection.dfw_poly.to(device=image_points.device, dtype=image_points.dtype)

    # Evaluate backward polynomial to get theta = f^-1(rdist).
    if int(projection.reference_poly) == int(FThetaPolynomialType.BACKWARD):
        # bw is reference, evaluate it directly.
        thetas = _eval_poly_horner(bw_poly, rdist)
    else:
        # fw is reference, invert via Newton on the bw_poly approximation.
        thetas = _eval_poly_inverse_horner_newton(
            fw_poly, dfw_poly, bw_poly, projection.newton_iterations, rdist
        )

    min_2d_norm = torch.tensor(
        projection.min_2d_norm, device=image_points.device, dtype=image_points.dtype
    )
    cam_rays = torch.hstack(
        (
            torch.sin(thetas) * image_points_dist / torch.maximum(rdist, min_2d_norm),
            torch.cos(thetas),
        )
    )
    cam_rays[rdist.flatten() < min_2d_norm, :] = torch.tensor(
        [[0, 0, 1]], device=image_points.device, dtype=image_points.dtype
    )
    return cam_rays


def _generate_all_pixel_image_points(
    resolution: tuple[int, int], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Generate (resolution[0]*resolution[1], 2) image-point coords for all pixels.

    Pixels are addressed at their centers (``index + 0.5``). Order: row-major,
    matching the slang kernel's ``tid = y * width + x`` indexing.
    """
    width, height = resolution
    x = torch.arange(width, device=device, dtype=dtype) + 0.5
    y = torch.arange(height, device=device, dtype=dtype) + 0.5
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).contiguous()


def _image_points_relative_frame_times(
    image_points: torch.Tensor,
    resolution: tuple[int, int],
    shutter_type: ShutterType,
) -> torch.Tensor:
    """Per-pixel relative time t ∈ [0, 1] for rolling-shutter compensation.

    Mirrors ncore's ``CameraModel.image_points_relative_frame_times`` —
    floor/ceil convention with ``(resolution - 1)`` normalization.
    """
    width, height = resolution
    if shutter_type == ShutterType.GLOBAL:
        return torch.zeros(image_points.shape[0], device=image_points.device, dtype=image_points.dtype)
    if shutter_type == ShutterType.ROLLING_TOP_TO_BOTTOM:
        return torch.floor(image_points[:, 1]) / (height - 1)
    if shutter_type == ShutterType.ROLLING_BOTTOM_TO_TOP:
        return (height - torch.ceil(image_points[:, 1])) / (height - 1)
    if shutter_type == ShutterType.ROLLING_LEFT_TO_RIGHT:
        return torch.floor(image_points[:, 0]) / (width - 1)
    if shutter_type == ShutterType.ROLLING_RIGHT_TO_LEFT:
        return (width - torch.ceil(image_points[:, 0])) / (width - 1)
    raise ValueError(f"Unknown shutter type: {shutter_type}")


def _numerically_stable_xy_norm(cam_rays: torch.Tensor) -> torch.Tensor:
    """Numerically stable ||(x, y)|| per ray for forward FTheta projection.

    Mirrors ``ncore.impl.sensors.camera.CameraModel._numerically_stable_xy_norm``.
    """
    xy_norms = torch.zeros_like(cam_rays[:, 0]).unsqueeze(1)
    abs_pts = torch.abs(cam_rays[:, :2])
    min_pts = torch.min(abs_pts, dim=1, keepdim=True).values
    max_pts = torch.max(abs_pts, dim=1, keepdim=True).values
    non_zero = max_pts > 0
    min_max_ratio = min_pts[non_zero] / max_pts[non_zero]
    xy_norms[non_zero, None] = max_pts[non_zero, None] * torch.sqrt(
        1 + torch.pow(min_max_ratio[:, None], 2)
    )
    return xy_norms


@torch._dynamo.disable  # numpy→list conversion confuses dynamo's from_numpy guards
def _ncore_ftheta_to_projection_and_resolution(
    ncore_params: object, device: torch.device, dtype: torch.dtype
) -> tuple[FThetaProjection, tuple[int, int]]:
    """Build a FThetaProjection + (width, height) from an ncore
    ``FThetaCameraModelParameters``. Mirrors the relevant portion of
    ``CameraModelConverter._convert_ftheta`` plus the FTheta initialisation
    in ``ncore.impl.sensors.camera.FThetaCameraModel.__init__``.

    Defaults match ncore's: ``newton_iterations=3``, ``min_2d_norm=1e-6``.
    """
    import numpy as np  # local import to avoid mandatory numpy dep if unused

    # Polynomials (already (N,) torch tensors or numpy). Convert to plain
    # Python lists before going through torch.tensor — direct numpy arrays
    # trip dynamo's ``___from_numpy`` guard machinery when this helper is
    # invoked from inside a ``@torch.compile``-decorated frame.
    fw_poly_list = [float(x) for x in ncore_params.angle_to_pixeldist_poly]
    bw_poly_list = [float(x) for x in ncore_params.pixeldist_to_angle_poly]

    fw_poly = torch.tensor(fw_poly_list, device=device, dtype=dtype)
    bw_poly = torch.tensor(bw_poly_list, device=device, dtype=dtype)

    # Linear term A = [[c, d], [e, 1]], Ainv = (1/(c-e*d)) * [[1, -d], [-e, c]].
    c, d_, e = (float(x) for x in ncore_params.linear_cde)
    A = torch.tensor([[c, d_], [e, 1.0]], device=device, dtype=dtype)
    det = c - e * d_
    Ainv = torch.tensor(
        [[1.0, -d_], [-e, c]], device=device, dtype=dtype
    ) / float(det)

    # Polynomial derivatives (coefficient-of-power-i = i * c_i for i>=1).
    dfw_poly = torch.tensor(
        [i * x for i, x in enumerate(fw_poly_list[1:], start=1)], device=device, dtype=dtype
    )
    dbw_poly = torch.tensor(
        [i * x for i, x in enumerate(bw_poly_list[1:], start=1)], device=device, dtype=dtype
    )

    # ncore convention: principal-point origin is at the centre of the first
    # pixel; CameraModel uses top-left-corner origin → +0.5 px.
    pp_list = [float(x) + 0.5 for x in ncore_params.principal_point]
    pp = torch.tensor(pp_list, device=device, dtype=dtype)

    # reference_poly mapping.
    poly_type_str = str(ncore_params.reference_poly)
    if "ANGLE_TO_PIXELDIST" in poly_type_str:
        reference_poly = FThetaPolynomialType.FORWARD
    else:
        reference_poly = FThetaPolynomialType.BACKWARD

    projection = FThetaProjection(
        principal_point=pp,
        fw_poly=fw_poly,
        bw_poly=bw_poly,
        A=A,
        Ainv=Ainv,
        dfw_poly=dfw_poly,
        dbw_poly=dbw_poly,
        reference_poly=reference_poly,
        max_angle=float(ncore_params.max_angle),
        newton_iterations=3,
        min_2d_norm=1e-6,
    )
    res_arr = np.asarray(ncore_params.resolution).astype(np.int64).flatten()
    resolution = (int(res_arr[0]), int(res_arr[1]))
    return projection, resolution


@torch._dynamo.disable  # numpy / dataclass conversion from ncore params is outside the compiled path
def _ncore_external_distortion_to_distortion(
    ncore_params: object, device: torch.device, dtype: torch.dtype
) -> ExternalDistortion:
    """Build an in-tree external distortion from ncore camera parameters."""
    external_distortion = getattr(ncore_params, "external_distortion_parameters", None)
    if external_distortion is None:
        external_distortion = getattr(ncore_params, "external_distortion", None)
    if external_distortion is None:
        return NoExternalDistortion()

    required = (
        "horizontal_poly",
        "vertical_poly",
        "horizontal_poly_inverse",
        "vertical_poly_inverse",
    )
    if not all(hasattr(external_distortion, name) for name in required):
        raise NotImplementedError(
            f"unsupported external distortion parameters: {type(external_distortion).__name__}"
        )

    def tensor(name: str) -> torch.Tensor:
        return torch.as_tensor(
            getattr(external_distortion, name),
            device=device,
            dtype=dtype,
        ).flatten()

    reference_polynomial = getattr(external_distortion, "reference_poly", None)
    if reference_polynomial is None:
        reference_polynomial = getattr(external_distortion, "reference_polynomial", None)

    return BivariateWindshieldDistortion.from_components(
        h_poly=tensor("horizontal_poly"),
        v_poly=tensor("vertical_poly"),
        h_poly_inv=tensor("horizontal_poly_inverse"),
        v_poly_inv=tensor("vertical_poly_inverse"),
        reference_polynomial=reference_polynomial,
    )


def camera_rays_to_image_points(
    camera_model_parameters: object,
    cam_rays: torch.Tensor,
) -> object:
    """Forward FTheta camera projection: camera-frame rays → image points + valid mask.

    Accepts an ncore ``FThetaCameraModelParameters`` (matching the libs API)
    and returns an object with ``.image_points`` and ``.valid_flag``
    attributes (matching ``ncore.sensors.CameraModel.ImagePointsReturn``).

    Mirrors ``ncore.impl.sensors.camera.FThetaCameraModel._camera_rays_to_image_points_impl``.
    """
    cam_rays = cam_rays.to(dtype=torch.float32).contiguous()
    device = cam_rays.device
    dtype = cam_rays.dtype

    if isinstance(camera_model_parameters, FThetaProjection):
        projection = camera_model_parameters
        resolution = None  # caller didn't pass resolution; skip image-bounds check
        external_distortion: ExternalDistortion = NoExternalDistortion()
    else:
        projection, resolution = _ncore_ftheta_to_projection_and_resolution(
            camera_model_parameters, device, dtype
        )
        external_distortion = _ncore_external_distortion_to_distortion(
            camera_model_parameters, device, dtype
        )

    if isinstance(external_distortion, BivariateWindshieldDistortion):
        cam_rays = external_distortion.distort_camera_rays(cam_rays)
    elif not isinstance(external_distortion, NoExternalDistortion):
        raise NotImplementedError(
            f"unsupported external distortion: {type(external_distortion).__name__}"
        )

    ray_xy_norms = _numerically_stable_xy_norm(cam_rays)
    eps = torch.finfo(torch.float32).eps
    ray_xy_norms = torch.where(
        ray_xy_norms <= 0.0,
        torch.tensor(eps, device=device, dtype=dtype),
        ray_xy_norms,
    )

    thetas_full = torch.atan2(ray_xy_norms[:, 0:1], cam_rays[:, 2:3])
    max_angle = projection.max_angle
    thetas = torch.clamp(thetas_full, max=max_angle)

    fw_poly = projection.fw_poly.to(device=device, dtype=dtype)
    bw_poly = projection.bw_poly.to(device=device, dtype=dtype)
    dbw_poly = projection.dbw_poly.to(device=device, dtype=dtype)

    if int(projection.reference_poly) == int(FThetaPolynomialType.BACKWARD):
        deltas = _eval_poly_inverse_horner_newton(
            bw_poly, dbw_poly, fw_poly, projection.newton_iterations, thetas
        )
    else:
        deltas = _eval_poly_horner(fw_poly, thetas)

    A = projection.A.to(device=device, dtype=dtype)
    pp = projection.principal_point.to(device=device, dtype=dtype)
    image_points = (
        torch.einsum("ij,nj->ni", A, deltas / ray_xy_norms * cam_rays[:, :2])
        + pp[None, :]
    )

    valid_thetas = thetas[:, 0] < max_angle
    if resolution is not None:
        width, height = resolution
        valid_x = (image_points[:, 0] >= 0.0) & (image_points[:, 0] < width)
        valid_y = (image_points[:, 1] >= 0.0) & (image_points[:, 1] < height)
        valid = valid_x & valid_y & valid_thetas
    else:
        valid = valid_thetas

    class _ImagePointsReturn:
        pass

    out = _ImagePointsReturn()
    out.image_points = image_points
    out.valid_flag = valid
    return out


def image_points_to_world_rays_shutter_pose(
    image_points: torch.Tensor | None,
    projection: object,
    external_distortion: ExternalDistortion,
    resolution: tuple[int, int],
    shutter_type: ShutterType,
    dynamic_pose: DynamicPose,
    start_timestamp_us: int | None = None,
    end_timestamp_us: int | None = None,
    return_timestamps: bool = False,
    return_poses: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Pure-torch replacement for the slang
    ``image_points_to_world_rays_shutter_pose``.

    Returns ``(world_rays (N, 6), timestamps_us (N,) or None, poses_t or
    None, poses_q or None)``. Camera ray gen is FTheta-only; ``return_poses``
    is not implemented.
    """
    if not isinstance(projection, FThetaProjection):
        raise NotImplementedError(
            f"only FThetaProjection supported, got {type(projection).__name__}"
        )
    if not isinstance(external_distortion, (NoExternalDistortion, BivariateWindshieldDistortion)):
        raise NotImplementedError(
            f"unsupported external distortion: {type(external_distortion).__name__}"
        )
    if return_poses:
        raise NotImplementedError("return_poses=True not supported.")

    device = dynamic_pose.start_pose.translation.device
    dtype = torch.float32

    if image_points is None:
        image_points = _generate_all_pixel_image_points(resolution, device, dtype)
    else:
        image_points = image_points.to(device=device, dtype=dtype).contiguous()

    n = image_points.shape[0]
    if n == 0:
        return (
            torch.empty((0, 6), device=device, dtype=dtype),
            torch.empty(0, device=device, dtype=torch.int64) if return_timestamps else None,
            None,
            None,
        )

    # Camera-frame rays via FTheta inverse projection.
    cam_rays = _ftheta_image_points_to_camera_rays(image_points, projection)
    if isinstance(external_distortion, BivariateWindshieldDistortion):
        cam_rays = external_distortion.undistort_camera_rays(cam_rays)

    # Per-pixel rolling-shutter interpolation parameter.
    t = _image_points_relative_frame_times(image_points, resolution, shutter_type)

    # Translation lerp.
    trans_s = dynamic_pose.start_pose.translation.to(device=device, dtype=dtype)
    trans_e = dynamic_pose.end_pose.translation.to(device=device, dtype=dtype)
    world_position = (1 - t).unsqueeze(-1) * trans_s + t.unsqueeze(-1) * trans_e

    # Rotation slerp.
    rot_s = dynamic_pose.start_pose.rotation.to(device=device, dtype=dtype)
    rot_e = dynamic_pose.end_pose.rotation.to(device=device, dtype=dtype)
    R_s = rot_s.unsqueeze(0).expand(n, -1)
    R_e = rot_e.unsqueeze(0).expand(n, -1)
    rot_quat = quat_xyzw_slerp(R_s, R_e, t)
    R_per_pixel = quat_xyzw_to_rotmat(rot_quat)

    # Camera-frame rays → world frame.
    world_directions = torch.bmm(R_per_pixel, cam_rays.unsqueeze(-1)).squeeze(-1)

    world_rays = torch.empty((n, 6), dtype=dtype, device=device)
    world_rays[:, :3] = world_position
    world_rays[:, 3:] = world_directions

    if return_timestamps:
        assert start_timestamp_us is not None and end_timestamp_us is not None
        ts = (
            start_timestamp_us
            + (t * (end_timestamp_us - start_timestamp_us)).to(torch.int64)
        )
    else:
        ts = None

    return world_rays, ts, None, None
