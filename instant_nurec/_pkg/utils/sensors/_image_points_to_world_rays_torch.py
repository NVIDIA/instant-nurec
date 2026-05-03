"""Pure-torch replacement for
``libs.sensors.kernels.cameras.image_points_to_world_rays_shutter_pose``.

FTheta + NoExternalDistortion only for now (the standalone predict baseline
uses ``camera_front_wide_120fov`` which is FTheta). Other camera models
fall through to ``NotImplementedError`` until a dataset that needs them
arrives.

The math is taken from ncore's pure-python ``CameraModel`` /
``FThetaCameraModel.image_points_to_world_rays_shutter_pose`` (NRE-side
copy at ``/storage/projects/nre/external/ncore/impl/sensors/camera.py``,
lines 1014-1112 for the rolling-shutter interp; lines 1347-1377 for the
FTheta inverse-projection).
"""

from __future__ import annotations

import torch

from instant_nurec._pkg.utils.sensors._kernel_types import (
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


def _quat_xyzw_slerp(
    quat_s: torch.Tensor, quat_e: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """Batched SLERP between two unit quaternions (XYZW).

    Mirrors ``ncore.impl.sensors.common.unitquat_slerp`` (shortest-arc).
    """
    cos_omega = torch.sum(quat_s * quat_e, dim=-1)
    quat_e = torch.where((cos_omega < 0).unsqueeze(-1), -quat_e, quat_e)
    cos_omega = torch.abs(cos_omega)

    nearby = cos_omega > (1.0 - 1e-3)
    cos_omega = torch.clamp(cos_omega, -1.0 + 1e-6, 1.0 - 1e-6)
    omega = torch.acos(cos_omega)
    alpha = torch.sin((1 - t) * omega)
    beta = torch.sin(t * omega)
    alpha = torch.where(nearby, (1 - t), alpha)
    beta = torch.where(nearby, t, beta)
    quat = alpha.unsqueeze(-1) * quat_s + beta.unsqueeze(-1) * quat_e
    return quat / torch.norm(quat, dim=-1, keepdim=True)


def _quat_xyzw_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """XYZW unit quaternion → 3x3 rotation matrix."""
    x = quat[..., 0]
    y = quat[..., 1]
    z = quat[..., 2]
    w = quat[..., 3]
    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w
    R = torch.empty(quat.shape[:-1] + (3, 3), dtype=quat.dtype, device=quat.device)
    R[..., 0, 0] = x2 - y2 - z2 + w2
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 1, 1] = -x2 + y2 - z2 + w2
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 2] = -x2 - y2 + z2 + w2
    return R


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
    is unused in the standalone and not implemented here.
    """
    if not isinstance(projection, FThetaProjection):
        raise NotImplementedError(
            f"Phase A.6 torch impl: only FThetaProjection supported, got "
            f"{type(projection).__name__}"
        )
    if not isinstance(external_distortion, NoExternalDistortion):
        raise NotImplementedError(
            f"Phase A.6 torch impl: only NoExternalDistortion supported, got "
            f"{type(external_distortion).__name__}"
        )
    if return_poses:
        raise NotImplementedError("return_poses=True not used in standalone predict.")

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
    rot_quat = _quat_xyzw_slerp(R_s, R_e, t)
    R_per_pixel = _quat_xyzw_to_rotmat(rot_quat)

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
