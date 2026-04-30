# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Python bindings for camera kernel operations.

This module provides thin wrappers around Slang camera kernels via slangtorch.
All functions are differentiable and support PyTorch autograd.
"""

import inspect

from typing import Any, Optional

import torch

from torch import Tensor

import libs.sensors.libcamera_slang_cc as camera_slang  # type: ignore # pycena: skip

from libs.sensors.kernels.cameras.parameters import (
    BivariateWindshieldDistortion,
    CameraProjection,
    ExternalDistortion,
    FThetaProjection,
    NoExternalDistortion,
    OpenCVFisheyeProjection,
    OpenCVPinholeProjection,
    ShutterType,
)
from libs.sensors.kernels.common import DynamicPose, Pose
from libs.slang_utils.utils import div_up


_THREADS_PER_BLOCK = 256
assert _THREADS_PER_BLOCK <= 1024, "_THREADS_PER_BLOCK must be less than or equal to 1024 for block level reductions"


def _count_nondiff_params(forward_fn) -> int:
    """Count non-differentiable params in an autograd forward() for backward's None tuple.

    The backward return has: explicit grads for the first differentiable tensors
    (up to and including control_times), then one None per remaining named param,
    then *grad_params for the variadic. This counts only the "remaining named" params.
    """
    sig = inspect.signature(forward_fn)
    named = [
        p.name
        for p in sig.parameters.values()
        if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    # named = [ctx, <diff tensors...>, control_times, <nondiff params...>]
    # Exclude: ctx (1), and the 4 explicitly handled entries (world_points/image_points,
    # control_translations, control_rotations, control_times)
    return len(named) - 1 - 4


# ============================================================================
# Parameter Extraction Helpers
# ============================================================================


def _get_kernel_suffix(projection: CameraProjection, external_distortion: ExternalDistortion) -> str:
    """Get kernel suffix from projection and distortion types."""
    if isinstance(projection, OpenCVPinholeProjection):
        proj = "opencv_pinhole"
    elif isinstance(projection, OpenCVFisheyeProjection):
        proj = "opencv_fisheye"
    elif isinstance(projection, FThetaProjection):
        proj = "ftheta"
    else:
        raise TypeError(f"Unsupported projection: {type(projection).__name__}")

    if isinstance(external_distortion, NoExternalDistortion):
        dist = "no_external"
    elif isinstance(external_distortion, BivariateWindshieldDistortion):
        dist = "bivariate_windshield"
    else:
        raise TypeError(f"Unsupported distortion: {type(external_distortion).__name__}")

    return f"{proj}_{dist}"


def _to_dev(t: Tensor, device: torch.device, dtype: torch.dtype, allow_device_transfer: bool = False) -> Tensor:
    """Move tensor to device/dtype and make contiguous.

    Args:
        t: Input tensor
        device: Target device
        dtype: Target dtype
        allow_device_transfer: If False, raises error when device/dtype transfer is needed.
            If True, allows implicit transfer.

    Raises:
        RuntimeError: If tensor requires device/dtype transfer and allow_device_transfer is False.
    """
    needs_transfer = t.device != device or t.dtype != dtype
    if needs_transfer:
        if not allow_device_transfer:
            raise RuntimeError(
                f"Tensor on {t.device} (dtype={t.dtype}) requires transfer to {device} (dtype={dtype}). "
                f"To allow implicit device transfer, set allow_device_transfer=True. "
                f"For best performance, ensure all tensors are on the target device before calling this function."
            )
        return t.to(device=device, dtype=dtype).contiguous()
    return t.contiguous()


def _extract_tensors(
    projection: CameraProjection,
    external_distortion: ExternalDistortion,
    device: torch.device,
    dtype: torch.dtype,
    allow_device_transfer: bool = False,
) -> tuple[list[Tensor], dict[str, Any], str]:
    """Extract all differentiable tensors and config from projection/distortion.

    Args:
        projection: Camera projection parameters
        external_distortion: External distortion parameters
        device: Target device
        dtype: Target dtype
        allow_device_transfer: If False, raises error when device/dtype transfer is needed.
            If True, allows implicit transfer.

    Returns:
        tensors: List of tensors in canonical order
        config: Dictionary of non-differentiable config values
        kernel_suffix: Kernel name suffix
    """
    tensors = []
    config = {}
    suffix = _get_kernel_suffix(projection, external_distortion)

    # Extract projection tensors
    if isinstance(projection, OpenCVPinholeProjection):
        # Single intrinsics tensor (16 floats) for efficiency - avoids overhead of 5 DiffTensorViews
        tensors.append(_to_dev(projection.intrinsics, device, dtype, allow_device_transfer))
    elif isinstance(projection, OpenCVFisheyeProjection):
        # Single intrinsics tensor (16 floats) for efficiency - avoids overhead of 5 DiffTensorViews
        tensors.append(_to_dev(projection.intrinsics, device, dtype, allow_device_transfer))
        config.update(
            {
                "fisheye_max_angle": projection.max_angle,
                "fisheye_newton_iterations": projection.newton_iterations,
                "fisheye_min_2d_norm": projection.min_2d_norm,
            }
        )
    elif isinstance(projection, FThetaProjection):
        # FThetaProjection packs all differentiable params into a single intrinsics tensor
        # Layout: [pp(2), fw_poly(N), bw_poly(N), A(4), Ainv(4), dfw_poly(N), dbw_poly(N)] where N=MAX_POLYNOMIAL_TERMS
        # Note: Derivatives are computed on-the-fly via eval_poly_derivative(); dfw/dbw arrays not used by Slang.
        intrinsics = _to_dev(projection.intrinsics, device, dtype, allow_device_transfer)
        tensors.append(intrinsics)  # Single packed tensor
        config.update(
            {
                "ftheta_reference_poly": int(projection.reference_poly),
                "ftheta_fw_poly_degree": projection.fw_poly_degree,
                "ftheta_bw_poly_degree": projection.bw_poly_degree,
                "ftheta_max_angle": projection.max_angle,
                "ftheta_newton_iterations": projection.newton_iterations,
                "ftheta_min_2d_norm": projection.min_2d_norm,
            }
        )
    else:
        raise TypeError(
            f"Unsupported projection type: {type(projection).__name__}. "
            f"Expected one of: OpenCVPinholeProjection, OpenCVFisheyeProjection, FThetaProjection"
        )

    # Extract distortion tensors
    if isinstance(external_distortion, BivariateWindshieldDistortion):
        # BivariateWindshieldDistortion packs all differentiable params into distortion_coeffs (40,)
        # Layout: [h_poly(10), v_poly(10), h_poly_inv(10), v_poly_inv(10)]
        tensors.append(_to_dev(external_distortion.distortion_coeffs, device, dtype, allow_device_transfer))
        config.update(
            {
                "ext_reference_polynomial": int(external_distortion.reference_polynomial),
                "ext_h_poly_degree": external_distortion.h_poly_degree,
                "ext_v_poly_degree": external_distortion.v_poly_degree,
            }
        )
    elif not isinstance(external_distortion, NoExternalDistortion):
        raise TypeError(
            f"Unsupported external distortion type: {type(external_distortion).__name__}. "
            f"Expected one of: NoExternalDistortion, BivariateWindshieldDistortion"
        )

    return tensors, config, suffix


def _build_kernel_args(
    tensors: list[Tensor],
    suffix: str,
    config: dict[str, Any],
    grad_tensors: Optional[list[Tensor]] = None,
) -> list[Any]:
    """Build kernel arguments from tensors and config.

    Args:
        tensors: List of parameter tensors
        suffix: Kernel suffix determining tensor interpretation
        config: Non-differentiable config values
        grad_tensors: Optional gradient tensors for backward pass
    """
    args = []
    grads = grad_tensors if grad_tensors else tensors
    idx = 0

    def _next() -> tuple[Any, tuple[Any]]:
        nonlocal idx
        t, g = tensors[idx], grads[idx]
        idx += 1
        return (t, (g,))

    # Projection args
    if suffix.startswith("opencv_pinhole"):
        args.append(_next())  # single packed intrinsics tensor (16,)
    elif suffix.startswith("opencv_fisheye"):
        args.append(_next())  # single packed intrinsics tensor (16,)
        args.extend([config["fisheye_max_angle"], config["fisheye_newton_iterations"], config["fisheye_min_2d_norm"]])
    elif suffix.startswith("ftheta"):
        args.append(_next())  # single packed intrinsics tensor
        args.extend(
            [
                config["ftheta_reference_poly"],
                config["ftheta_fw_poly_degree"],
                config["ftheta_bw_poly_degree"],
                config["ftheta_max_angle"],
                config["ftheta_newton_iterations"],
                config["ftheta_min_2d_norm"],
            ]
        )

    # Distortion args
    if suffix.endswith("bivariate_windshield"):
        args.append(_next())  # single packed distortion_coeffs tensor (40,)
        args.extend(
            [
                config["ext_reference_polynomial"],
                config["ext_h_poly_degree"],
                config["ext_v_poly_degree"],
            ]
        )

    return args


# ============================================================================
# Autograd Functions
# ============================================================================


class CameraRaysToImagePointsFunction(torch.autograd.Function):
    """Differentiable camera ray to image point projection."""

    @staticmethod
    def forward(ctx, camera_rays: Tensor, suffix: str, config: dict, *param_tensors) -> tuple[Tensor, Tensor]:
        N = camera_rays.shape[0]
        device, dtype = camera_rays.device, camera_rays.dtype

        camera_rays = camera_rays.contiguous()
        image_points = torch.empty((N, 2), device=device, dtype=dtype)
        valid_flags = torch.empty(N, device=device, dtype=torch.bool)

        blocks = div_up(N, _THREADS_PER_BLOCK)
        if blocks > 0:
            kernel = getattr(camera_slang, f"camera_rays_to_image_points_{suffix}")
            args: list[Any] = [(_THREADS_PER_BLOCK, 1, 1), (blocks, 1, 1)]
            args.extend(_build_kernel_args(list(param_tensors), suffix, config))
            args.extend(
                [(camera_rays, (camera_rays,)), (image_points, (image_points,)), valid_flags]
            )  # valid_flags is no_diff
            kernel(*args)

        ctx.save_for_backward(camera_rays, image_points, valid_flags, *param_tensors)
        ctx.suffix = suffix
        ctx.config = config
        ctx.N = N

        return image_points, valid_flags

    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore[override]
        grad_image_points, _grad_valid_flags = grad_outputs
        saved = ctx.saved_tensors
        camera_rays, image_points, valid_flags = saved[0], saved[1], saved[2]
        param_tensors = list(saved[3:])

        grad_image_points = grad_image_points.contiguous()
        grad_camera_rays = torch.zeros_like(camera_rays)
        grad_params = [torch.zeros_like(t) for t in param_tensors]

        blocks = div_up(ctx.N, _THREADS_PER_BLOCK)
        if blocks > 0:
            bwd_kernel = getattr(camera_slang, f"camera_rays_to_image_points_{ctx.suffix}_bwd_diff")
            args: list[Any] = [(_THREADS_PER_BLOCK, 1, 1), (blocks, 1, 1)]
            args.extend(_build_kernel_args(param_tensors, ctx.suffix, ctx.config, grad_params))
            args.extend([(camera_rays, (grad_camera_rays,)), (image_points, (grad_image_points,)), valid_flags])
            bwd_kernel(*args)

        return (grad_camera_rays, None, None, *grad_params)


class ImagePointsToCameraRaysFunction(torch.autograd.Function):
    """Differentiable image point to camera ray backprojection."""

    @staticmethod
    def forward(ctx, image_points: Tensor, suffix: str, config: dict, *param_tensors) -> Tensor:
        N = image_points.shape[0]
        device, dtype = image_points.device, image_points.dtype

        image_points = image_points.contiguous()
        camera_rays = torch.empty((N, 3), device=device, dtype=dtype)

        blocks = div_up(N, _THREADS_PER_BLOCK)
        if blocks > 0:
            kernel = getattr(camera_slang, f"image_points_to_camera_rays_{suffix}")
            args: list[Any] = [(_THREADS_PER_BLOCK, 1, 1), (blocks, 1, 1)]
            args.extend(_build_kernel_args(list(param_tensors), suffix, config))
            args.extend([(image_points, (image_points,)), (camera_rays, (camera_rays,))])
            kernel(*args)

        ctx.save_for_backward(image_points, camera_rays, *param_tensors)
        ctx.suffix = suffix
        ctx.config = config
        ctx.N = N

        return camera_rays

    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore[override]
        (grad_camera_rays,) = grad_outputs
        saved = ctx.saved_tensors
        image_points, camera_rays = saved[0], saved[1]
        param_tensors = list(saved[2:])

        grad_camera_rays = grad_camera_rays.contiguous()
        grad_image_points = torch.zeros_like(image_points)
        grad_params = [torch.zeros_like(t) for t in param_tensors]

        blocks = div_up(ctx.N, _THREADS_PER_BLOCK)
        if blocks > 0:
            bwd_kernel = getattr(camera_slang, f"image_points_to_camera_rays_{ctx.suffix}_bwd_diff")
            args: list[Any] = [(_THREADS_PER_BLOCK, 1, 1), (blocks, 1, 1)]
            args.extend(_build_kernel_args(param_tensors, ctx.suffix, ctx.config, grad_params))
            args.extend([(image_points, (grad_image_points,)), (camera_rays, (grad_camera_rays,))])
            bwd_kernel(*args)

        return (grad_image_points, None, None, *grad_params)


class ProjectWorldPointsFunction(torch.autograd.Function):
    """Differentiable world point projection with pose."""

    _nondiff_count: int = -1  # set after class body

    @staticmethod
    def forward(
        ctx,
        world_points: Tensor,
        control_translations: Tensor,
        control_rotations: Tensor,
        control_times: Tensor,
        suffix: str,
        config: dict,
        resolution: tuple[int, int],
        shutter_type: int,
        control_count: int,
        max_iterations: int,
        stop_mean_error_px: float,
        stop_delta_mean_error_px: float,
        initial_relative_time: float,
        use_shutter: bool,
        start_timestamp_us: int,
        end_timestamp_us: int,
        *param_tensors,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        N = world_points.shape[0]
        device, dtype = world_points.device, world_points.dtype

        world_points = world_points.contiguous()
        image_points = torch.empty((N, 2), device=device, dtype=dtype)
        valid_flags = torch.empty(N, device=device, dtype=torch.bool)
        timestamps_us = torch.empty(N, device=device, dtype=torch.int64)
        poses_translation = torch.empty((N, 3), device=device, dtype=dtype)
        poses_rotation = torch.empty((N, 4), device=device, dtype=dtype)

        blocks = div_up(N, _THREADS_PER_BLOCK)
        if blocks > 0:
            kernel_name = f"project_world_points_{'shutter' if use_shutter else 'mean'}_pose_{suffix}"
            kernel = getattr(camera_slang, kernel_name)

            args: list[Any] = [(_THREADS_PER_BLOCK, 1, 1), (blocks, 1, 1)]
            args.extend(_build_kernel_args(list(param_tensors), suffix, config))
            args.append((resolution[0], resolution[1]))  # uint2 as tuple
            if use_shutter:
                args.append(shutter_type)
            args.extend(
                [
                    (control_translations, (control_translations,)),
                    (control_rotations, (control_rotations,)),
                    control_times,
                    control_count,
                ]
            )
            if use_shutter:
                args.extend([max_iterations, stop_mean_error_px, stop_delta_mean_error_px, initial_relative_time])
            args.extend([start_timestamp_us, end_timestamp_us])
            args.extend(
                [
                    (world_points, (world_points,)),
                    (image_points, (image_points,)),
                    valid_flags,
                    timestamps_us,
                    poses_translation,
                    poses_rotation,
                ]
            )
            kernel(*args)

        ctx.save_for_backward(
            world_points,
            image_points,
            valid_flags,
            timestamps_us,
            poses_translation,
            poses_rotation,
            control_translations,
            control_rotations,
            control_times,
            *param_tensors,
        )
        ctx.suffix = suffix
        ctx.config = config
        ctx.resolution = resolution
        ctx.shutter_type = shutter_type
        ctx.control_count = control_count
        ctx.max_iterations = max_iterations
        ctx.stop_mean_error_px = stop_mean_error_px
        ctx.stop_delta_mean_error_px = stop_delta_mean_error_px
        ctx.initial_relative_time = initial_relative_time
        ctx.use_shutter = use_shutter
        ctx.start_timestamp_us = int(start_timestamp_us)
        ctx.end_timestamp_us = int(end_timestamp_us)
        ctx.N = N

        return image_points, valid_flags, timestamps_us, poses_translation, poses_rotation

    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore[override]
        # timestamps_us, poses_translation, poses_rotation are non-differentiable outputs
        (
            grad_image_points,
            _grad_valid_flags,
            _grad_timestamps_us,
            _grad_poses_trans,
            _grad_poses_rot,
        ) = grad_outputs
        saved = ctx.saved_tensors
        world_points, image_points, valid_flags = saved[0], saved[1], saved[2]
        timestamps_us, poses_translation, poses_rotation = saved[3], saved[4], saved[5]
        control_translations, control_rotations, control_times = saved[6], saved[7], saved[8]
        param_tensors = list(saved[9:])

        grad_image_points = grad_image_points.contiguous()
        grad_world_points = torch.zeros_like(world_points)
        grad_translations = torch.zeros_like(control_translations)
        grad_rotations = torch.zeros_like(control_rotations)
        grad_params = [torch.zeros_like(t) for t in param_tensors]

        blocks = div_up(ctx.N, _THREADS_PER_BLOCK)
        if blocks > 0:
            kernel_name = f"project_world_points_{'shutter' if ctx.use_shutter else 'mean'}_pose_{ctx.suffix}_bwd_diff"
            bwd_kernel = getattr(camera_slang, kernel_name)

            args: list[Any] = [(_THREADS_PER_BLOCK, 1, 1), (blocks, 1, 1)]
            args.extend(_build_kernel_args(param_tensors, ctx.suffix, ctx.config, grad_params))
            args.append((ctx.resolution[0], ctx.resolution[1]))  # uint2 as tuple
            if ctx.use_shutter:
                args.append(ctx.shutter_type)
            args.extend(
                [
                    (control_translations, (grad_translations,)),
                    (control_rotations, (grad_rotations,)),
                    control_times,
                    ctx.control_count,
                ]
            )
            if ctx.use_shutter:
                args.extend(
                    [
                        ctx.max_iterations,
                        ctx.stop_mean_error_px,
                        ctx.stop_delta_mean_error_px,
                        ctx.initial_relative_time,
                    ]
                )
            args.extend([ctx.start_timestamp_us, ctx.end_timestamp_us])
            args.extend(
                [
                    (world_points, (grad_world_points,)),
                    (image_points, (grad_image_points,)),
                    valid_flags,
                    timestamps_us,
                    poses_translation,
                    poses_rotation,
                ]
            )
            bwd_kernel(*args)

        return (
            grad_world_points,
            grad_translations,
            grad_rotations,
            None,
            *((None,) * ProjectWorldPointsFunction._nondiff_count),
            *grad_params,
        )


ProjectWorldPointsFunction._nondiff_count = _count_nondiff_params(ProjectWorldPointsFunction.forward)


class ImagePointsToWorldRaysFunction(torch.autograd.Function):
    """Differentiable image point to world ray backprojection with pose."""

    _nondiff_count: int = -1  # set after class body

    @staticmethod
    def forward(
        ctx,
        image_points: Tensor | None,
        control_translations: Tensor,
        control_rotations: Tensor,
        control_times: Tensor,
        suffix: str,
        config: dict,
        resolution: tuple[int, int],
        shutter_type: int,
        control_count: int,
        use_shutter: bool,
        start_timestamp_us: int,
        end_timestamp_us: int,
        *param_tensors,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        does_generate_elements = image_points is None
        N = image_points.shape[0] if image_points is not None else resolution[0] * resolution[1]
        device, dtype = control_translations.device, image_points.dtype if image_points is not None else torch.float32
        world_rays = torch.empty((N, 6), device=device, dtype=dtype)
        if image_points is None:
            image_points = torch.empty((1, 2), device=device, dtype=torch.float32)
        else:
            image_points = image_points.contiguous()
        timestamps_us = torch.empty(N, device=device, dtype=torch.int64)
        poses_translation = torch.empty((N, 3), device=device, dtype=dtype)
        poses_rotation = torch.empty((N, 4), device=device, dtype=dtype)

        blocks = div_up(N, _THREADS_PER_BLOCK)
        if blocks > 0:
            kernel_name = f"image_points_to_world_rays_{'shutter' if use_shutter else 'static'}_pose_{suffix}"
            kernel = getattr(camera_slang, kernel_name)

            args: list[Any] = [(_THREADS_PER_BLOCK, 1, 1), (blocks, 1, 1)]
            args.extend(_build_kernel_args(list(param_tensors), suffix, config))

            if use_shutter:
                args.append((resolution[0], resolution[1]))  # uint2 as tuple
                args.append(shutter_type)
                args.extend(
                    [
                        (control_translations, (control_translations,)),
                        (control_rotations, (control_rotations,)),
                        control_times,
                        control_count,
                        start_timestamp_us,
                        end_timestamp_us,
                        (image_points, (image_points,)),
                        does_generate_elements,
                    ]
                )
            else:
                # Static pose - just single translation/rotation plus timestamp
                args.extend(
                    [
                        (control_translations, (control_translations,)),
                        (control_rotations, (control_rotations,)),
                        start_timestamp_us,  # For static pose, use start_timestamp_us as the single timestamp
                        (image_points, (image_points,)),  # image points but no does_generate_elements
                    ]
                )
            args.extend(
                [
                    (world_rays, (world_rays,)),
                    timestamps_us,
                    poses_translation,
                    poses_rotation,
                ]
            )
            kernel(*args)

        ctx.save_for_backward(
            image_points,
            world_rays,
            timestamps_us,
            poses_translation,
            poses_rotation,
            control_translations,
            control_rotations,
            control_times,
            *param_tensors,
        )
        ctx.suffix = suffix
        ctx.config = config
        ctx.resolution = resolution
        ctx.shutter_type = shutter_type
        ctx.control_count = control_count
        ctx.use_shutter = use_shutter
        ctx.start_timestamp_us = int(start_timestamp_us)
        ctx.end_timestamp_us = int(end_timestamp_us)
        ctx.N = N
        ctx.does_generate_elements = does_generate_elements

        return world_rays, timestamps_us, poses_translation, poses_rotation

    @staticmethod
    def backward(ctx, *grad_outputs):  # type: ignore[override]
        # timestamps_us, poses_translation, poses_rotation are non-differentiable outputs
        grad_world_rays, _grad_timestamps_us, _grad_poses_trans, _grad_poses_rot = grad_outputs
        saved = ctx.saved_tensors
        image_points, world_rays = saved[0], saved[1]
        timestamps_us, poses_translation, poses_rotation = saved[2], saved[3], saved[4]
        control_translations, control_rotations, control_times = saved[5], saved[6], saved[7]
        param_tensors = list(saved[8:])

        if ctx.does_generate_elements:
            grad_image_points = torch.empty((1, 2), device=image_points.device, dtype=image_points.dtype)
        else:
            grad_image_points = torch.zeros_like(image_points)
        grad_world_rays = grad_world_rays.contiguous()
        grad_translations = torch.zeros_like(control_translations)
        grad_rotations = torch.zeros_like(control_rotations)
        grad_params = [torch.zeros_like(t) for t in param_tensors]

        blocks = div_up(ctx.N, _THREADS_PER_BLOCK)
        if blocks > 0:
            kernel_name = (
                f"image_points_to_world_rays_{'shutter' if ctx.use_shutter else 'static'}_pose_{ctx.suffix}_bwd_diff"
            )
            bwd_kernel = getattr(camera_slang, kernel_name)

            args: list[Any] = [(_THREADS_PER_BLOCK, 1, 1), (blocks, 1, 1)]
            args.extend(_build_kernel_args(param_tensors, ctx.suffix, ctx.config, grad_params))

            if ctx.use_shutter:
                args.append((ctx.resolution[0], ctx.resolution[1]))  # uint2 as tuple
                args.append(ctx.shutter_type)
                args.extend(
                    [
                        (control_translations, (grad_translations,)),
                        (control_rotations, (grad_rotations,)),
                        control_times,
                        ctx.control_count,
                        ctx.start_timestamp_us,
                        ctx.end_timestamp_us,
                        (image_points, (grad_image_points,)),
                        ctx.does_generate_elements,
                    ]
                )
            else:
                args.extend(
                    [
                        (control_translations, (grad_translations,)),
                        (control_rotations, (grad_rotations,)),
                        ctx.start_timestamp_us,
                        (image_points, (grad_image_points,)),
                    ]
                )
            args.extend(
                [
                    (world_rays, (grad_world_rays,)),
                    timestamps_us,
                    poses_translation,
                    poses_rotation,
                ]
            )
            bwd_kernel(*args)

        return (
            grad_image_points if not ctx.does_generate_elements else None,
            grad_translations,
            grad_rotations,
            None,
            *((None,) * ImagePointsToWorldRaysFunction._nondiff_count),
            *grad_params,
        )


ImagePointsToWorldRaysFunction._nondiff_count = _count_nondiff_params(ImagePointsToWorldRaysFunction.forward)


# ============================================================================
# Public API
# ============================================================================


def camera_rays_to_image_points(
    camera_rays: Tensor,
    projection: CameraProjection,
    external_distortion: ExternalDistortion,
    allow_device_transfer: bool = False,
) -> tuple[Tensor, Tensor]:
    """Project camera rays to image points. Fully differentiable.

    Args:
        camera_rays: (N, 3) normalized rays in camera frame
        projection: Camera projection parameters
        external_distortion: External distortion parameters
        allow_device_transfer: If False (default), raises error when projection/distortion
            tensors require device or dtype transfer. If True, allows implicit transfer.

    Returns:
        image_points: (N, 2) pixel coordinates
        valid_flags: (N,) bool validity mask
    """
    N = camera_rays.shape[0]
    if N == 0:
        return torch.empty((0, 2), device=camera_rays.device, dtype=camera_rays.dtype), torch.empty(
            0, device=camera_rays.device, dtype=torch.bool
        )

    tensors, config, suffix = _extract_tensors(
        projection, external_distortion, camera_rays.device, camera_rays.dtype, allow_device_transfer
    )
    image_points, valid_flags = CameraRaysToImagePointsFunction.apply(
        camera_rays.contiguous(), suffix, config, *tensors
    )
    return image_points, valid_flags


def image_points_to_camera_rays(
    image_points: Tensor,
    projection: CameraProjection,
    external_distortion: ExternalDistortion,
    allow_device_transfer: bool = False,
) -> Tensor:
    """Back-project image points to camera rays. Fully differentiable.

    Args:
        image_points: (N, 2) pixel coordinates
        projection: Camera projection parameters
        external_distortion: External distortion parameters
        allow_device_transfer: If False (default), raises error when projection/distortion
            tensors require device or dtype transfer. If True, allows implicit transfer.

    Returns:
        camera_rays: (N, 3) normalized directions in camera frame
    """
    N = image_points.shape[0]
    if N == 0:
        return torch.empty((0, 3), device=image_points.device, dtype=image_points.dtype)

    tensors, config, suffix = _extract_tensors(
        projection, external_distortion, image_points.device, image_points.dtype, allow_device_transfer
    )
    return ImagePointsToCameraRaysFunction.apply(image_points.contiguous(), suffix, config, *tensors)


def project_world_points_shutter_pose(
    world_points: Tensor,
    projection: CameraProjection,
    external_distortion: ExternalDistortion,
    resolution: tuple[int, int],
    shutter_type: ShutterType,
    dynamic_pose: DynamicPose,
    start_timestamp_us: int | None = None,
    end_timestamp_us: int | None = None,
    max_iterations: int = 10,
    stop_mean_error_px: float = 0.001,
    stop_delta_mean_error_px: float = 0.00001,
    initial_relative_time: float = 0.5,
    allow_device_transfer: bool = False,
    return_valid_flags: bool = False,
    return_timestamps: bool = False,
    return_poses: bool = False,
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    """Project world points with rolling shutter. Fully differentiable.

    Args:
        world_points: (N, 3) world coordinates
        projection: Camera projection parameters
        external_distortion: External distortion parameters
        resolution: (width, height) sensor resolution
        shutter_type: Rolling shutter behavior
        dynamic_pose: Time-varying pose
        start_timestamp_us: Start timestamp in microseconds for absolute timestamp computation
        end_timestamp_us: End timestamp in microseconds for absolute timestamp computation
        max_iterations: Max iterations for convergence
        stop_mean_error_px: Stopping criterion
        stop_delta_mean_error_px: Stopping criterion
        initial_relative_time: Initial guess for rolling shutter convergence in [0, 1]. Defaults to 0.5.
        allow_device_transfer: If False (default), raises error when projection/distortion
            tensors require device or dtype transfer. If True, allows implicit transfer.
        return_valid_flags: If True, compute and return validity mask
        return_timestamps: If True, compute and return absolute timestamps
        return_poses: If True, compute and return per-point poses

    Returns:
        image_points: (N, 2) pixel coordinates
        valid_flags: (N,) bool validity mask, or None if return_valid_flags=False
        timestamps_us: (N,) absolute timestamps in microseconds (int64), or None if return_timestamps=False
        poses_translation: (N, 3) per-point interpolated pose translations, or None if return_poses=False
        poses_rotation: (N, 4) per-point interpolated pose rotations (quaternions), or None if return_poses=False
    """
    if return_timestamps:
        assert start_timestamp_us is not None, "start_timestamp_us must be provided when return_timestamps=True"
        assert end_timestamp_us is not None, "end_timestamp_us must be provided when return_timestamps=True"

    N = world_points.shape[0]
    device, dtype = world_points.device, world_points.dtype

    if N == 0:
        return (
            torch.empty((0, 2), device=device, dtype=dtype),
            torch.empty(0, device=device, dtype=torch.bool) if return_valid_flags else None,
            torch.empty(0, device=device, dtype=torch.int64) if return_timestamps else None,
            torch.empty((0, 3), device=device, dtype=dtype) if return_poses else None,
            torch.empty((0, 4), device=device, dtype=dtype) if return_poses else None,
        )

    trajectory = dynamic_pose.to_trajectory()
    assert trajectory.control_count <= 2, (
        f"Rolling shutter kernel supports at most 2 control poses, got {trajectory.control_count}"
    )
    trans = _to_dev(
        torch.stack([p.translation for p in trajectory.control_poses]).contiguous(),
        device,
        dtype,
        allow_device_transfer,
    )
    rots = _to_dev(
        torch.stack([p.rotation for p in trajectory.control_poses]).contiguous(), device, dtype, allow_device_transfer
    )
    times = _to_dev(trajectory.control_times.contiguous(), device, dtype, allow_device_transfer)

    tensors, config, suffix = _extract_tensors(projection, external_distortion, device, dtype, allow_device_transfer)

    # Convert None timestamps to 0
    start_ts = 0 if start_timestamp_us is None else start_timestamp_us
    end_ts = 0 if end_timestamp_us is None else end_timestamp_us

    (
        image_points,
        valid_flags,
        timestamps_us,
        poses_translation,
        poses_rotation,
    ) = ProjectWorldPointsFunction.apply(
        world_points.contiguous(),
        trans,
        rots,
        times,
        suffix,
        config,
        resolution,
        int(shutter_type),
        trajectory.control_count,
        max_iterations,
        stop_mean_error_px,
        stop_delta_mean_error_px,
        initial_relative_time,
        True,
        start_ts,
        end_ts,
        *tensors,
    )
    return (
        image_points,
        valid_flags if return_valid_flags else None,
        timestamps_us if return_timestamps else None,
        poses_translation if return_poses else None,
        poses_rotation if return_poses else None,
    )


def project_world_points_mean_pose(
    world_points: Tensor,
    projection: CameraProjection,
    external_distortion: ExternalDistortion,
    dynamic_pose: DynamicPose,
    resolution: tuple[int, int],
    start_timestamp_us: int | None = None,
    end_timestamp_us: int | None = None,
    allow_device_transfer: bool = False,
    return_valid_flags: bool = False,
    return_timestamps: bool = False,
    return_poses: bool = False,
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    """Project world points using mean pose. Fully differentiable.

    Args:
        world_points: (N, 3) world coordinates
        projection: Camera projection parameters
        external_distortion: External distortion parameters
        dynamic_pose: Time-varying pose
        resolution: (width, height) sensor resolution
        start_timestamp_us: Start timestamp in microseconds for absolute timestamp computation
        end_timestamp_us: End timestamp in microseconds for absolute timestamp computation
        allow_device_transfer: If False (default), raises error when projection/distortion
            tensors require device or dtype transfer. If True, allows implicit transfer.
        return_valid_flags: If True, compute and return validity mask
        return_timestamps: If True, compute and return absolute timestamps
        return_poses: If True, compute and return per-point poses

    Returns:
        image_points: (N, 2) pixel coordinates
        valid_flags: (N,) bool validity mask, or None if return_valid_flags=False
        timestamps_us: (N,) absolute timestamps in microseconds (int64, midpoint for mean pose), or None if return_timestamps=False
        poses_translation: (N, 3) per-point interpolated pose translations (mean pose), or None if return_poses=False
        poses_rotation: (N, 4) per-point interpolated pose rotations (mean pose, quaternions), or None if return_poses=False
    """
    if return_timestamps:
        assert start_timestamp_us is not None, "start_timestamp_us must be provided when return_timestamps=True"
        assert end_timestamp_us is not None, "end_timestamp_us must be provided when return_timestamps=True"

    N = world_points.shape[0]
    device, dtype = world_points.device, world_points.dtype

    if N == 0:
        return (
            torch.empty((0, 2), device=device, dtype=dtype),
            torch.empty(0, device=device, dtype=torch.bool) if return_valid_flags else None,
            torch.empty(0, device=device, dtype=torch.int64) if return_timestamps else None,
            torch.empty((0, 3), device=device, dtype=dtype) if return_poses else None,
            torch.empty((0, 4), device=device, dtype=dtype) if return_poses else None,
        )

    trajectory = dynamic_pose.to_trajectory()
    assert trajectory.control_count <= 2, (
        f"Mean-pose kernel supports at most 2 control poses, got {trajectory.control_count}"
    )
    trans = _to_dev(
        torch.stack([p.translation for p in trajectory.control_poses]).contiguous(),
        device,
        dtype,
        allow_device_transfer,
    )
    rots = _to_dev(
        torch.stack([p.rotation for p in trajectory.control_poses]).contiguous(), device, dtype, allow_device_transfer
    )
    times = _to_dev(trajectory.control_times.contiguous(), device, dtype, allow_device_transfer)

    tensors, config, suffix = _extract_tensors(projection, external_distortion, device, dtype, allow_device_transfer)

    # Convert None timestamps to 0
    start_ts = 0 if start_timestamp_us is None else start_timestamp_us
    end_ts = 0 if end_timestamp_us is None else end_timestamp_us

    (
        image_points,
        valid_flags,
        timestamps_us,
        poses_translation,
        poses_rotation,
    ) = ProjectWorldPointsFunction.apply(
        world_points.contiguous(),
        trans,
        rots,
        times,
        suffix,
        config,
        resolution,
        0,
        trajectory.control_count,
        0,
        0.0,
        0.0,
        0.5,
        False,
        start_ts,
        end_ts,
        *tensors,
    )
    return (
        image_points,
        valid_flags if return_valid_flags else None,
        timestamps_us if return_timestamps else None,
        poses_translation if return_poses else None,
        poses_rotation if return_poses else None,
    )


def image_points_to_world_rays_static_pose(
    image_points: Tensor,
    projection: CameraProjection,
    external_distortion: ExternalDistortion,
    static_pose: Pose,
    timestamp_us: int | None = None,
    allow_device_transfer: bool = False,
    return_timestamps: bool = False,
    return_poses: bool = False,
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
    """Back-project image points to world rays with static pose. Fully differentiable.

    Args:
        image_points: (N, 2) pixel coordinates
        projection: Camera projection parameters
        external_distortion: External distortion parameters
        static_pose: Static pose
        timestamp_us: Timestamp in microseconds for the static pose
        allow_device_transfer: If False (default), raises error when projection/distortion
            tensors require device or dtype transfer. If True, allows implicit transfer.
        return_timestamps: If True, compute and return absolute timestamps
        return_poses: If True, compute and return per-point poses

    Returns:
        world_rays: (N, 6) [origin.xyz, direction.xyz] in world frame
        timestamps_us: (N,) absolute timestamps in microseconds (int64, same as timestamp_us for all points), or None if return_timestamps=False
        poses_translation: (N, 3) per-point pose translations (same as static pose), or None if return_poses=False
        poses_rotation: (N, 4) per-point pose rotations (same as static pose, quaternions), or None if return_poses=False
    """
    if return_timestamps:
        assert timestamp_us is not None, "timestamp_us must be provided when return_timestamps=True"

    N = image_points.shape[0]
    device, dtype = image_points.device, image_points.dtype

    if N == 0:
        return (
            torch.empty((0, 6), device=device, dtype=dtype),
            torch.empty(0, device=device, dtype=torch.int64) if return_timestamps else None,
            torch.empty((0, 3), device=device, dtype=dtype) if return_poses else None,
            torch.empty((0, 4), device=device, dtype=dtype) if return_poses else None,
        )

    # Static pose kernel expects (3,) and (4,) tensors, not batched
    trans = _to_dev(static_pose.translation.contiguous(), device, dtype, allow_device_transfer)
    rots = _to_dev(static_pose.rotation.contiguous(), device, dtype, allow_device_transfer)
    times = torch.tensor([0.0], device=device, dtype=dtype)

    tensors, config, suffix = _extract_tensors(projection, external_distortion, device, dtype, allow_device_transfer)

    # Convert None timestamp to 0
    ts = 0 if timestamp_us is None else timestamp_us

    world_rays, timestamps_us, poses_translation, poses_rotation = ImagePointsToWorldRaysFunction.apply(
        image_points.contiguous(),
        trans,
        rots,
        times,
        suffix,
        config,
        (0, 0),
        0,
        1,
        False,
        ts,
        0,  # end_timestamp_us not used for static pose
        *tensors,
    )
    return (
        world_rays,
        timestamps_us if return_timestamps else None,
        poses_translation if return_poses else None,
        poses_rotation if return_poses else None,
    )


def image_points_to_world_rays_shutter_pose(
    image_points: Tensor | None,
    projection: CameraProjection,
    external_distortion: ExternalDistortion,
    resolution: tuple[int, int],
    shutter_type: ShutterType,
    dynamic_pose: DynamicPose,
    start_timestamp_us: int | None = None,
    end_timestamp_us: int | None = None,
    allow_device_transfer: bool = False,
    return_timestamps: bool = False,
    return_poses: bool = False,
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
    """Back-project image points to world rays with rolling shutter. Fully differentiable.

    Args:
        image_points: (N, 2) pixel coordinates or None if elements are generated
        projection: Camera projection parameters
        external_distortion: External distortion parameters
        resolution: (width, height) sensor resolution
        shutter_type: Rolling shutter behavior
        dynamic_pose: Time-varying pose
        start_timestamp_us: Start timestamp in microseconds for absolute timestamp computation
        end_timestamp_us: End timestamp in microseconds for absolute timestamp computation
        allow_device_transfer: If False (default), raises error when projection/distortion
            tensors require device or dtype transfer. If True, allows implicit transfer.
        return_timestamps: If True, compute and return absolute timestamps
        return_poses: If True, compute and return per-point poses

    Returns:
        world_rays: (N, 6) [origin.xyz, direction.xyz] in world frame
        timestamps_us: (N,) absolute timestamps in microseconds (int64), or None if return_timestamps=False
        poses_translation: (N, 3) per-point interpolated pose translations, or None if return_poses=False
        poses_rotation: (N, 4) per-point interpolated pose rotations (quaternions), or None if return_poses=False
    """
    if return_timestamps:
        assert start_timestamp_us is not None, "start_timestamp_us must be provided when return_timestamps=True"
        assert end_timestamp_us is not None, "end_timestamp_us must be provided when return_timestamps=True"

    if image_points is not None:
        N = image_points.shape[0]
        device, dtype = image_points.device, image_points.dtype
    else:
        N = resolution[0] * resolution[1]
        dtype = torch.float32
        device = dynamic_pose.start_pose.translation.device

    if N == 0:
        return (
            torch.empty((0, 6), device=device, dtype=dtype),
            torch.empty(0, device=device, dtype=torch.int64) if return_timestamps else None,
            torch.empty((0, 3), device=device, dtype=dtype) if return_poses else None,
            torch.empty((0, 4), device=device, dtype=dtype) if return_poses else None,
        )

    trajectory = dynamic_pose.to_trajectory()
    assert trajectory.control_count <= 2, (
        f"Backprojection kernel supports at most 2 control poses, got {trajectory.control_count}"
    )
    trans = _to_dev(
        torch.stack([p.translation for p in trajectory.control_poses]).contiguous(),
        device,
        dtype,
        allow_device_transfer,
    )
    rots = _to_dev(
        torch.stack([p.rotation for p in trajectory.control_poses]).contiguous(), device, dtype, allow_device_transfer
    )
    times = _to_dev(trajectory.control_times.contiguous(), device, dtype, allow_device_transfer)

    tensors, config, suffix = _extract_tensors(projection, external_distortion, device, dtype, allow_device_transfer)

    # Convert None timestamps to 0
    start_ts = 0 if start_timestamp_us is None else start_timestamp_us
    end_ts = 0 if end_timestamp_us is None else end_timestamp_us

    (
        world_rays,
        timestamps_us,
        poses_translation,
        poses_rotation,
    ) = ImagePointsToWorldRaysFunction.apply(
        image_points,
        trans,
        rots,
        times,
        suffix,
        config,
        resolution,
        int(shutter_type),
        trajectory.control_count,
        True,
        start_ts,
        end_ts,
        *tensors,
    )
    return (
        world_rays,
        timestamps_us if return_timestamps else None,
        poses_translation if return_poses else None,
        poses_rotation if return_poses else None,
    )


def generate_image_points(
    resolution: tuple[int, int],
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Generate image points.

    Args:
        resolution: (width, height) sensor resolution

    Returns:
        image_points: (N, 2) pixel coordinates
    """
    w, h = resolution
    N = w * h
    image_points = torch.empty((N, 2), device=device, dtype=torch.float32)
    blocks = div_up(N, _THREADS_PER_BLOCK)
    if blocks > 0:
        camera_slang.generate_image_points_kernel((_THREADS_PER_BLOCK, 1, 1), (blocks, 1, 1), w, h, image_points)
    return image_points


__all__ = [
    "camera_rays_to_image_points",
    "image_points_to_camera_rays",
    "project_world_points_shutter_pose",
    "project_world_points_mean_pose",
    "image_points_to_world_rays_static_pose",
    "image_points_to_world_rays_shutter_pose",
    "generate_image_points",
]
