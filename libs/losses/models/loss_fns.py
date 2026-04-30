# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import math

from typing import Dict, List

import torch
import torch.nn.functional as F

from fused_ssim import fused_ssim

from libs.losses.kernel.constants import GRID_NUM_COLS, GRID_NUM_ROWS
from libs.losses.models.registry import register_loss_fn
from libs.losses.models.utils import torch_ssim
from nre.models.custom_modules import trunc_bce
from nre.models.post_processings.ppisp import BasePPISP, ColorCorrection, PiecewisePowerFunction, RadialFalloff
from nre.utils.lpips_network import LPIPSNetwork
from nre.utils.packed_ops import packed_weighted_sum


@register_loss_fn("bce_clipped")
def bce_loss_clipped(input: torch.Tensor, target: torch.Tensor, eps: float = 0.001) -> torch.Tensor:
    """
    Returns bce loss after directly clipping the `input` with (eps, 1-eps).
    No gradients on the clipped-out areas.
    """
    # NOTE: Due to pytorch `binary_cross_entropy` is not autocast-safe.
    with torch.amp.autocast(device_type="cuda", enabled=False):
        return F.binary_cross_entropy(input.float().clip(eps, 1 - eps), target.float(), reduction="none")


@register_loss_fn("bce_truncated")
def bce_loss_truncated(input: torch.Tensor, target: torch.Tensor, eps: float = 0.001) -> torch.Tensor:
    """
    Returns equivalent results as `bce_loss_clipped`,
    only that the clipped-out areas will still hold clipped gradients to avoid dead zone.
    See the documentation in `trunc_bce` for more details.
    """
    return trunc_bce(input, target, eps)


def dirac_delta_approx(x, mu=0, sigma=1e-5):
    """
    Approximates the Dirac delta function with a Gaussian distribution.

    Args:
        x (torch.Tensor): The input tensor.
        mu (float, optional): The mean of the Gaussian distribution. Defaults to 0.
        sigma (float, optional): The standard deviation of the Gaussian distribution. Defaults to 1e-5.

    Returns:
        torch.Tensor: The output tensor.
    """
    return (1 / (math.sqrt(2 * torch.pi * sigma**2))) * torch.exp(-((x - mu) ** 2) / (2 * sigma**2))


@register_loss_fn("los")
def line_of_sight_loss(
    rays_mask: torch.Tensor,
    gt_depth: torch.Tensor,
    weights: torch.Tensor,
    t_vals: torch.Tensor,
    pack_info: torch.Tensor,
    epsilon: float,
    depth_upper_bound: float | None = None,
) -> torch.Tensor:
    # evaluate sample-specific values unconditionally on all samples (unmasked)
    gt_depth_samples, t_vals = (
        gt_depth.squeeze().repeat_interleave(pack_info[:, 1], dim=0),
        t_vals.squeeze().detach(),
    )  # currently t_vals don't propagate gradients, still explicitly detaching to future-proof
    depth_mask = gt_depth_samples > 0

    # apply depth_upper_bound
    if depth_upper_bound is not None:
        depth_mask &= gt_depth_samples < depth_upper_bound

    empty_mask = t_vals < gt_depth_samples - epsilon
    near_mask = (t_vals > (gt_depth_samples - epsilon)) & (t_vals < gt_depth_samples + epsilon)

    empty_mask &= depth_mask
    near_mask &= depth_mask

    # evaluate per-ray loss values on the masked active rays only
    empty_loss = packed_weighted_sum(empty_mask.unsqueeze(-1).float(), weights.square(), pack_info[rays_mask, :])
    near_loss = packed_weighted_sum(
        near_mask.unsqueeze(-1).float(),
        (weights - dirac_delta_approx(t_vals - gt_depth_samples, sigma=epsilon / 3)).square(),
        pack_info[rays_mask, :],
    )

    """
    The far_loss term is not used as it's not needed for training.
    see (Eq 18) of the "Urban Radiance Fields" paper, as well as EmerNeRF implementation
    https://github.com/NVlabs/EmerNeRF/commit/17386ee765de69f9b815115af7930a464f2a3404#diff-2dba32fe1191d4d3b5eac91d8a1ba37d7a238fb6dc16f7abe4c718ceed6c2390R462

    far_mask = t_vals > gt_depth + epsilon
    far_mask &= depth_mask
    far_loss = torch.mean(packed_weighted_sum(
        far_mask.unsqueeze(-1).float(),
        weights.square(),
        pack_info
    )) # type: ignore
    """

    depth_error = empty_loss + near_loss
    return depth_error


@register_loss_fn("weights_reg")
def weight_regularization(weights_list: List[torch.Tensor], dim=1):
    return torch.mean(torch.cat([(weights**2).sum(dim) for weights in weights_list]))


@register_loss_fn("relu_sum")
def relu_sum(value: torch.Tensor, eps: float):
    return F.relu(value - eps).sum()


@register_loss_fn("total_variation_spatial")
def total_variation_spatial(x: torch.Tensor) -> torch.Tensor:
    """
    Compute total variation loss for a tensor x across spatial dimensions.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, D, H, W).

    Returns:
        tv_loss (torch.Tensor): Total variation loss, a tensor of shape (B,).
    """
    tv_z = x.diff(dim=2).square().mean(dim=(1, 2, 3, 4)) if x.shape[2] > 1 else torch.zeros(1, device=x.device)
    tv_y = x.diff(dim=3).square().mean(dim=(1, 2, 3, 4)) if x.shape[3] > 1 else torch.zeros(1, device=x.device)
    tv_x = x.diff(dim=4).square().mean(dim=(1, 2, 3, 4)) if x.shape[4] > 1 else torch.zeros(1, device=x.device)

    return tv_z + tv_y + tv_x


@register_loss_fn("total_variation_temporal")
def total_variation_temporal(x: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute total variation loss for a tensor x along its temporal dimension,
    assuming that the sequence is continous.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, D, H, W).

    Returns:
        tv_loss (torch.Tensor): Total variation loss, a tensor of shape (B,).
    """
    tv_t = x.diff(dim=0).square().mean(dim=(1, 2, 3, 4)) if x.shape[0] > 1 else torch.zeros(1, device=x.device)

    return tv_t * loss_mask


@register_loss_fn("identity_distance")
def identity_distance(grid: torch.Tensor) -> torch.Tensor:
    """
    Compute the distance of a tensor x from the identity transformation.
    Assumes that the tensor models an affine (B, M, N, ...) transformation
    with dimensions given by GRID_NUM_ROWS and GRID_NUM_COLS, i.e. MxN affine transformation matrix.

    Args:
        x (torch.Tensor): Input tensor of shape that can be reshaped into (B, M, N, ...)

    Returns:
        distance from identity transformation (torch.Tensor)
    """

    reshaped_grid = grid.view(grid.shape[0], GRID_NUM_ROWS, GRID_NUM_COLS, *grid.shape[2:])
    identity = torch.eye(GRID_NUM_ROWS, GRID_NUM_COLS, device=grid.device)
    identity = identity.view(1, GRID_NUM_ROWS, GRID_NUM_COLS, *([1] * len(grid.shape[2:])))

    # Calculate difference from identity for each transformation
    diff = reshaped_grid - identity

    # Calculate Frobenius norm for each transformation
    return torch.norm(diff, p="fro", dim=(1, 2))


def _smoothness_loss_across_cams(
    values: torch.Tensor,
    src_idcs: torch.Tensor,
    dst_idcs: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Compute total variation loss across consecutive frames of the same camera.
    Source and destination indices are expected to be pre-computed.

    Args:
        values: Tensor of shape (N,) containing one value per frame.
        src_idcs: Long tensor of shape (M,) containing source frame indices.
        dst_idcs: Long tensor of shape (M,) containing destination frame indices.
        beta: Float parameter for SmoothL1Loss (default: 0.1).

    Returns:
        Scalar tensor containing the smoothness loss normalized by number of frames.
    """
    assert src_idcs.size() == dst_idcs.size(), "Source and destination indices must have the same size"
    if src_idcs.numel() == 0:  # This implies dst_idcs is also empty due to the assert
        return torch.tensor(0.0, device=values.device)

    src_values = values[src_idcs]
    dst_values = values[dst_idcs]
    return torch.nn.functional.smooth_l1_loss(src_values, dst_values, beta=beta, reduction="mean")


def compute_exposure_losses(
    exposure_params: torch.Tensor, src_idcs: torch.Tensor, dst_idcs: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Compute exposure-related losses.

    Args:
        exposure_params: Tensor of shape (N,) containing exposure values per frame.
        src_idcs: Long tensor of shape (M,) containing source frame indices.
        dst_idcs: Long tensor of shape (M,) containing destination frame indices.

    Returns:
        Dict with scalar tensors for mean and smoothness losses.
    """
    losses = {}
    losses["exposure_mean"] = torch.abs(torch.mean(exposure_params))
    losses["exposure_smooth"] = _smoothness_loss_across_cams(exposure_params, src_idcs, dst_idcs, beta=0.05)
    return losses


def compute_vignetting_losses(vignetting_params: "RadialFalloff.PackedParams") -> Dict[str, torch.Tensor]:
    """Compute vignetting-related losses from packed parameters:
    - Center: Penalizes optical centers far from image center
    - Channel consistency: Enforces similar parameters across color channels
    - Non-positive: Ensures physically plausible darkening effect

    Args:
        vignetting_params: RadialFalloff.PackedParams of shape (num_cameras, 3, 5)
    """
    losses = {}

    # Unpack parameters using the PackedParams interface
    optical_centers = vignetting_params.optical_center  # (num_cameras, 3, 2)
    alphas = vignetting_params.alphas  # (num_cameras, 3, NUM_VIGNETTING_ALPHA_TERMS)

    # Center loss: penalize optical centers far from (0.5, 0.5)
    losses["vig_center"] = torch.mean((optical_centers - 0.5) ** 2)

    # Channel consistency: penalize differences between channel parameters for the same camera
    losses["vig_channel"] = torch.mean(torch.var(vignetting_params.data, dim=1))

    # Non-positive loss: ensure alpha values are negative for a darkening effect
    losses["vig_non_pos"] = torch.mean(F.relu(alphas))

    return losses


def compute_color_losses(
    color_params: torch.Tensor, src_idcs: torch.Tensor, dst_idcs: torch.Tensor, source_chroms: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Compute color correction-related losses from packed parameters.

    Args:
        color_params: torch.Tensor of shape (num_frames, 8)
        src_idcs: Long tensor of shape (M,) containing source frame indices.
        dst_idcs: Long tensor of shape (M,) containing destination frame indices.
        source_chroms: Tensor of shape (4, 2) containing source chromaticities.
    """
    losses = {}

    # Re-create homographies from packed params
    h = ColorCorrection.params_to_homography(color_params)
    target_chroms = ColorCorrection.apply_color_correction_rg(source_chroms, h)

    mean_target_chroms = target_chroms.mean(dim=0)

    losses["color_mean"] = torch.sum(torch.abs(mean_target_chroms - source_chroms))

    # Flatten chromaticities for smoothness loss
    target_chroms_flat = target_chroms.reshape(target_chroms.shape[0], -1)
    losses["color_smooth"] = _smoothness_loss_across_cams(target_chroms_flat, src_idcs, dst_idcs, beta=0.1)

    return losses


def compute_crf_losses(curve_points: "PiecewisePowerFunction.CurvePoints") -> Dict[str, torch.Tensor]:
    """Compute CRF-related losses from curve points.

    Args:
        crf_curve_points: CurvePoints object containing curve parameters for all cameras and RGB channels
                         Each parameter has shape (num_cameras, 3)
    """
    losses = {}

    # Compute inverse values for all curves using vectorized inverse function
    # Shape: (num_cameras, 3)
    ones_tensor = torch.ones_like(curve_points.gamma)

    # Apply inverse function directly to batched curve points
    inverse_values = PiecewisePowerFunction.inverse(curve_points, ones_tensor)
    max_inverse = torch.max(inverse_values)

    # Compute gamma-related loss
    # Shape: (num_cameras * 3,) -> scalar
    gamma_log_geom_mean = torch.mean(torch.log(curve_points.gamma.flatten()))
    log_target_gamma = math.log(1.0 / 2.2)  # sRGB reference (approximately)

    # Compute channel consistency loss
    # Key points for each curve: (x0,y0), (x1,y1), (shoulder_x,shoulder_y)
    # Shape: (num_cameras, 3, 3, 2) -> 3 key points, 2 coordinates each
    key_points = torch.stack(
        [
            torch.stack([curve_points.x0, curve_points.y0], dim=-1),  # (x0, y0)
            torch.stack([curve_points.x1, curve_points.y1], dim=-1),  # (x1, y1)
            torch.stack([curve_points.shoulder_x, curve_points.shoulder_y], dim=-1),  # (shoulder_x, shoulder_y)
        ],
        dim=2,
    )

    # Channel consistency loss: variance of RGB channels for each curve measurement
    # Shape: (num_cameras, 3, 3, 2) -> (num_cameras, 3, 2) -> (num_cameras,) -> scalar
    channel_variances = torch.var(key_points, dim=1)  # Variance across RGB channels
    crf_channel_consistency = torch.sum(channel_variances, dim=(1, 2))  # Sum across key points and coordinates

    # Combine losses
    losses["crf_range"] = torch.abs(max_inverse - 1.0)
    losses["crf_gamma"] = (gamma_log_geom_mean - log_target_gamma) ** 2
    losses["crf_channel"] = torch.mean(crf_channel_consistency)

    return losses


@register_loss_fn("ppisp_loss")
def ppisp_loss(lambdas: Dict[str, float], ppisp_model: BasePPISP) -> torch.Tensor:
    """Compute all PPISP-related losses and combine them with given lambdas."""
    losses: Dict[str, torch.Tensor] = {}

    # Gather source and destination indices for smoothness losses from the model
    src_idcs = ppisp_model.smoothness_src_indices
    dst_idcs = ppisp_model.smoothness_dst_indices

    # Get packed parameters from the model
    exposure_params = ppisp_model.packed_exposure_params
    vignetting_params = ppisp_model.packed_vignetting_params
    color_params = ppisp_model.packed_color_params
    crf_curve_points = ppisp_model.crf_curve_points

    # Compute losses for each component
    losses.update(compute_exposure_losses(exposure_params, src_idcs, dst_idcs))
    losses.update(compute_vignetting_losses(vignetting_params))
    losses.update(compute_color_losses(color_params, src_idcs, dst_idcs, ppisp_model.default_source_chroms))
    losses.update(compute_crf_losses(crf_curve_points))

    # Combine all losses with their respective lambdas
    total_loss = torch.sum(torch.stack([lambdas[k] * v for k, v in losses.items()]))
    return total_loss


@register_loss_fn("ssim")
def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    mask: torch.Tensor,
    mask_value: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    channel: int,
) -> torch.Tensor:
    img1 = img1 * mask + mask_value * (~mask)
    img2 = img2 * mask + mask_value * (~mask)

    # Fused-SSIM loss only supports window size of 11 (as per the original SSIM paper), for other window sizes we revert back to reference implementation
    if window_size == 11 and not img2.requires_grad:
        ssim_map = fused_ssim(img1, img2)
    else:
        ssim_map = torch_ssim(img1, img2, window=window, window_size=window_size, channel=channel)

    # Filter out the SSIM map with the mask so that the loss and hence bakward will only be applied to valid pixels
    # Average across the channels to obtain per ray SSIM loss and flatten it out
    num_mask = mask.sum()
    scale_factor = torch.where(
        num_mask > 0, 1 / num_mask, 0
    )  # pre-scale the mask to compute masked mean without using loss[mask], avoiding CUDA synchronization
    return (1 - ssim_map).mean(dim=1, keepdim=True) * mask * scale_factor


@register_loss_fn("depth_inverse_mse")
def depth_inverse_mse_loss(
    pred_value: torch.Tensor,
    target_value: torch.Tensor,
    eps: float,
    **kwargs,
) -> torch.Tensor:
    inv_pred_distance = torch.clamp(pred_value, min=eps).reciprocal()
    inv_target_distance = torch.clamp(target_value, min=eps).reciprocal()
    loss = (inv_pred_distance - inv_target_distance).pow(2)
    return loss


@register_loss_fn("log_l1")
def log_l1(pred_value: torch.Tensor, target_value: torch.Tensor, **kwargs) -> torch.Tensor:
    return torch.log(1.0 + (pred_value - target_value).abs())


@register_loss_fn("lpips")
def lpips_loss(img1: torch.Tensor, img2: torch.Tensor, lpips: LPIPSNetwork) -> torch.Tensor:
    # [B, 1, 1, 1] -> [B]
    return lpips.forward(img1, img2).view(-1)


@register_loss_fn("normal_cosine")
def normal_cosine_loss(pred_normal: torch.Tensor, gt_normal: torch.Tensor) -> torch.Tensor:
    """Compute 1 - cosine similarity between predicted and ground-truth normals. Requires normalized input."""
    assert pred_normal.shape == gt_normal.shape, "Predicted and ground-truth normals must have the same shape"
    assert pred_normal.shape[-1] == 3, "Normals must have 3 dimensions"
    return 1.0 - torch.sum(pred_normal * gt_normal, dim=-1)
