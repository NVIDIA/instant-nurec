# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import os

from pathlib import Path
from typing import Tuple

import click

# import imageio.v3 as iio
import numpy as np
import torch

from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToPILImage
from tqdm import tqdm


log = logging.getLogger(__name__)

DEFAULT_TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_mask_to_tensor(
    image_path: str, device: torch.device = DEFAULT_TORCH_DEVICE, invert: bool = False
) -> torch.Tensor:
    """Load an 2D image mask to a 2D boolean torch.Tensor with optional mask inversion"""
    image = Image.open(image_path)
    mask_tensor = torch.from_numpy(np.array(image.convert("L"))).to(device).bool()
    return mask_tensor if not invert else ~mask_tensor


def load_image_to_tensor(image_path: str, device: torch.device = DEFAULT_TORCH_DEVICE) -> torch.Tensor:
    """Load an RGB image into a float32 torch.Tensor of shape (C, H, W) with values within [0, 1.0]"""
    image = Image.open(image_path)
    return torch.from_numpy(np.array(image.convert("RGB"))).to(device).permute(2, 0, 1).float() / 255.0


def resize_mask(mask: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Resize a (H,W) torch.boolmask to a given resolution (height, width)"""
    assert mask.ndim == 2, "mask must be 2D (H, W)"
    assert mask.dtype == torch.bool, "mask must be bool"
    resize_op = transforms.Resize(
        size,
        transforms.InterpolationMode.BILINEAR,  # NEAREST leaves holes when upscaling
        max_size=None,
        antialias=True,
    )
    assert resize_op is not None  # Type narrowing for static analyzer
    resized_mask = resize_op(mask.unsqueeze(0)).squeeze(0)
    assert resized_mask.dtype == torch.bool
    return resized_mask


def overlay_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    color: torch.Tensor,
    alpha: torch.Tensor = torch.tensor(0.5, dtype=torch.float32),
    device: torch.device = DEFAULT_TORCH_DEVICE,
) -> torch.Tensor:
    """Overlay a mask on an image by blending pixels in the mask region with the given color.

    Args:
        image: (C, H, W) float32 tensor in [0, 1].
        mask: (H, W) bool tensor; True where the overlay is applied.
        color: (C,) float32 tensor in [0, 1].
        alpha: Scalar blend factor in [0, 1]; 0 = no overlay, 1 = full color.
        device: Device to run the computation on; all tensors are moved here.

    Returns:
        (C, H, W) float32 tensor on the given device.
    """
    image = image.to(device)
    mask = mask.to(device)
    color = color.to(device)
    alpha = alpha.to(device)

    assert mask.ndim == 2, "mask must be 2D (H, W)"
    assert image.ndim == 3, "image must be 3D (C, H, W)"
    assert mask.shape == image.shape[1:], "mask (H,W) must match image (C,H,W) spatial dims"
    assert mask.dtype == torch.bool, "mask must be bool"
    assert color.shape[0] == image.shape[0], "color (C,) must match image channels"
    assert alpha.ndim == 0, "alpha must be a scalar"
    assert alpha.dtype == torch.float32, "alpha must be float32"
    assert alpha.shape == torch.Size([]), "alpha must be a scalar"

    mask_expanded = mask.unsqueeze(0)  # (1, H, W)
    color_expanded = color.view(-1, 1, 1)  # (C, 1, 1)
    blended = (1 - alpha) * image + alpha * color_expanded
    return torch.where(mask_expanded, blended, image)


@click.command("export-mask-overlay")
@click.option(
    "--image-dir",
    type=str,
    help="Path to a directory containing one or more image sequences in subdirectories.",
    required=True,
)
@click.option(
    "--mask-dir",
    type=str,
    help=(
        "Path to a directory containing PNG masks, one mask per image sequence (non-zero pixels to be masked out)."
        " E.g. if seq1 and seq2 are two subdirectories in the image directory, "
        " then seq1.png and seq2.png should be present in the mask directory. "
        "The input masks are converted to grayscale and then to binary (zeros vs non-zeros). "
        "Non-zero pixels of the mask are masked out in the images, unless --invert-mask is used."
    ),
    required=True,
)
@click.option(
    "--output-dir",
    type=str,
    help=(
        "Path to an output directory. "
        "Images with the overlaid masks are saved to this directory following the structure of the image directory."
    ),
    required=True,
)
@click.option(
    "--mask-color",
    nargs=3,
    type=float,
    help="Desired RGB color of the mask overlay, in [0.0, 1.0]",
    default=(0.0, 0.0, 0.0),
    required=False,
)
@click.option(
    "--mask-alpha",
    type=float,
    help="Desired alpha value for the mask overlay, in [0.0, 1.0]",
    default=0.5,
    required=False,
)
@click.option(
    "--invert-mask",
    is_flag=True,
    help="Invert the mask before applying it",
)
@click.option(
    "--image-format",
    type=click.Choice(["png", "jpg"]),
    help="Image format to save the output images in (png or jpg)",
    default="jpg",
)
@click.option(
    "--jpeg-quality",
    type=int,
    help="JPEG quality to save the output images (0-100, higher is better quality)",
    default=90,
)
@click.option(
    "--png-compression",
    type=int,
    help="PNG compression level to save the output images in (0-9, 0: no compression, 9: highest compression)",
    default=1,
)
def export_mask_overlay(
    image_dir: str,
    mask_dir: str,
    output_dir: str,
    mask_color: Tuple[float, float, float],
    mask_alpha: float,
    invert_mask: bool,
    image_format: str,
    jpeg_quality: int,
    png_compression: int,
) -> None:
    """Overlay a set of binary masks on corresponding image sequences in subdirectories of an image root directory
    by blending a user-specified color to image pixels where each mask is non-zero.
    The masks are automatically resized when necessary to match the resolution of the images.

    Note that the overlaid mask can look pixelated, especially when upscaled, because it is a binary mask.
    """

    if not Path(image_dir).is_dir():
        raise FileNotFoundError(f"Missing directory {image_dir}")

    if not Path(mask_dir).is_dir():
        raise FileNotFoundError(f"Missing directory {mask_dir}")

    # Get list of subdirectories
    image_seq_names = [f for f in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, f))]
    log.info(f"Found {len(image_seq_names)} subdirectories in {image_dir}")

    for image_seq_name in image_seq_names:
        image_seq_dir = os.path.join(image_dir, image_seq_name)
        image_files = [
            f
            for f in os.listdir(image_seq_dir)
            if os.path.isfile(os.path.join(image_seq_dir, f)) and f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        log.info(f"Found {len(image_files)} images in {image_seq_dir}")

        mask_path = os.path.join(mask_dir, f"{image_seq_name}.png")

        log.info(f"Loading mask {mask_path}")
        mask = load_mask_to_tensor(mask_path, device=DEFAULT_TORCH_DEVICE, invert=invert_mask)
        log.info(f"Loaded mask: shape {tuple(mask.shape)} dtype {mask.dtype} on device {mask.device}")

        mask_color_tensor = torch.tensor(mask_color, dtype=torch.float32).clamp(0.0, 1.0).to(DEFAULT_TORCH_DEVICE)
        mask_alpha_tensor = torch.tensor(mask_alpha, dtype=torch.float32).clamp(0.0, 1.0).to(DEFAULT_TORCH_DEVICE)

        # Cache the resized mask to avoid resizing it for each image.
        resized_mask = mask.clone()

        for image_file in tqdm(image_files, desc=image_seq_name):
            # Load the image to overlay the mask on.
            image_path = os.path.join(image_seq_dir, image_file)
            image = load_image_to_tensor(image_path, device=DEFAULT_TORCH_DEVICE)
            height, width = image.shape[1], image.shape[2]

            # Ensure that the mask to overlay has the same resolution as the current image.
            if resized_mask.size() != image.size()[1:]:
                resized_mask = resize_mask(mask, (height, width))

            # Overlay the mask with a given color and alpha on the image.
            image_with_mask = overlay_mask(image, resized_mask, color=mask_color_tensor, alpha=mask_alpha_tensor)

            # Create the output sequence directory.
            output_seq_dir = os.path.join(output_dir, image_seq_name)
            os.makedirs(output_seq_dir, exist_ok=True)

            # Save the image.
            image_name = os.path.splitext(image_file)[0]
            output_image_path = os.path.join(output_seq_dir, f"{image_name}.{image_format}")
            ToPILImage()(image_with_mask).save(output_image_path, quality=jpeg_quality, compress_level=png_compression)

        log.info(f"{len(image_files)} images with mask overlay exported to {output_dir}")
