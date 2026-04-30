# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging

from typing import Sequence

import flow_vis
import imageio
import matplotlib as mpl
import numpy as np
import torch

from einops import rearrange
from matplotlib import cm
from pytorch_lightning.loggers.logger import DummyLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger


logger = logging.getLogger(__name__)


def scalar2img(
    scalar: np.ndarray,
    vmin: float | None = 0.1,
    vmax: float | None = 10.0,
    cmap: str = "turbo",
) -> np.ndarray:
    """
    Converts a (H, W) array into an RGB array suitable for viewing (by applying the colormap).

    Args:
        - scalar: 2d (H, W) array of scalars to be converted by a colormap
        - vmin: the value to correspond to start of the colormap (lower values clip), None uses `scalar.min()`
        - vmax: the value to correspond to end of the colormap (higher values clip), None uses `scalar.max()`
        - cmap: name of the matplotlib colormap to be used (see https://matplotlib.org/stable/users/explain/colors/colormaps.html)

    Returns:
        - A (H, W, 3) uint8 array of rgb colors
    """
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    # TODO: check why this doesn't work with type annotation
    colored_distance = (mapper.to_rgba(scalar)[:, :, :3] * 255).astype(np.uint8)  # type: ignore
    return colored_distance


def scalar2rgb(
    scalar: np.ndarray,
    vmin: float | None = 0.0,
    vmax: float | None = 1.0,
    cmap: str = "bwr",
) -> np.ndarray:
    """
    Converts a array into an RGB array suitable for viewing (by applying the colormap).

    Args:
        - scalar: (N, 1) array of scalars to be converted by a colormap
        - vmin: the value to correspond to start of the colormap (lower values clip), None uses `scalar.min()`
        - vmax: the value to correspond to end of the colormap (higher values clip), None uses `scalar.max()`
        - cmap: name of the matplotlib colormap to be used (see https://matplotlib.org/stable/users/explain/colors/colormaps.html)

    Returns:
        - A (N, 3) float16 array of rgb colors
    """
    normalizer = mpl.colors.Normalize(vmin=np.percentile(scalar, 5), vmax=np.percentile(scalar, 99))
    mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    colored_distance = (mapper.to_rgba(scalar)[:, :3]).astype(np.float16)  # type: ignore
    return colored_distance


def sem2img(sem_label, color_remap):
    color_remap[sem_label]
    return color_remap[sem_label].astype(np.uint8)


def flow2img(flow: np.ndarray, rad_max: float | None = None) -> np.ndarray:
    """
    Converts a UV flow field into an RGB image suitable for viewing.

    Args:
        - flow: (H, W, 2) array of UV flow vectors
        - rad_max: the maximum magnitude of the flow vectors, None uses max of flow radius
    """

    assert flow.ndim == 3, "input flow must have three dimensions"
    assert flow.shape[2] == 2, "input flow must have shape [H,W,2]"
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))

    if rad_max is None:
        rad_max = np.max(rad).item()
    else:
        # Renormalize the flow vectors
        renorm_mask = rad > rad_max
        u[renorm_mask] = u[renorm_mask] * (rad_max / rad[renorm_mask])
        v[renorm_mask] = v[renorm_mask] * (rad_max / rad[renorm_mask])

    u = u / (rad_max + 1e-5)
    v = v / (rad_max + 1e-5)
    return flow_vis.flow_uv_to_colors(u, v, False)


def instance2color(instance, color_map, h, w):
    n_inst = len(instance["scores"])
    instance_color = np.full((h, w, 3), 255, dtype=np.uint8)
    for inst in range(n_inst):
        instance_color[np.where(instance["instance_masks"][inst])] = color_map[instance["classes"][inst]]
    return instance_color


def generate_colors(
    values: np.ndarray,
    cmap_name: str = "jet_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
    return_hex: bool = False,
) -> np.ndarray:
    """Convert a range of values to the given color map.

    Args:
        values: a 1d array of values used to generate the color map, shape (num_values,).
        cmap_name: any color map of matplotlib.
        vmin: lower bound of the normalisation range used on the values: [vmin, vmax].
        vmax: upper bound of the normalisation range used on the values: [vmin, vmax].
        return_hex: if True, return the color map as hex values.

    Returns:
        colors as floats in the range [0, 1], shape (num_values, 3[r, g, b]).
            If `return_hex` is True: return hex values as string e.g. '#000080', shape (num_values,).
    """

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    rgba_colors = cmap(norm(values))
    if return_hex:
        return np.array([mpl.colors.rgb2hex(v) for v in rgba_colors])
    return rgba_colors[:, :3]  # remove the final column which is a constant alpha=1 value.


def save_im(im: torch.Tensor, h: int, filename: str) -> None:
    """Save an image to a file.

    Args:
        im (torch.Tensor): The image to save, tensor with shape [(h w) c].
        h (int): The height of the image.
        filename (str): The filename to save the image to.
    """
    np_im: np.ndarray = rearrange(im.cpu().numpy(), "(h w) c -> h w c", h=h)
    np_im = (np_im * 255).astype(np.uint8)
    imageio.v2.imsave(filename, np_im)


def log_image(
    logger: WandbLogger | TensorBoardLogger | DummyLogger | None,
    image: torch.Tensor | np.ndarray,
    caption: str,
    step: int | None = None,
) -> None:
    """Logs an image to the specified logger if available.

    Args:
        logger (WandbLogger | TensorBoardLogger | DummyLogger | None): Logger instance, or None to skip logging
        image (torch.Tensor | np.ndarray): Image to log
        caption (str): Name/tag for the image
        step (Optional[int]): Current training step
    """

    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image.copy())
        if image.dim() == 3:
            image = rearrange(image, "h w c -> c h w")
        elif image.dim() == 2:
            image = rearrange(image, "h w -> 1 h w")
    assert isinstance(image, torch.Tensor) and image.dim() == 3

    match logger:
        case WandbLogger():
            logger.log_image(caption, [image], step=step)

        case TensorBoardLogger():
            logger.experiment.add_image(caption, image, step, dataformats="CHW" if image.dim() == 3 else "HW")

        case DummyLogger() | None:
            pass


def make_image_grid(images: list[np.ndarray], grid_width: int, subsample: int = 1) -> np.ndarray:
    """
    Arrange a list of images into a grid with given width.

    Args:
        images (list of np.ndarray): List of images as numpy arrays (H, W, C or H, W).
        grid_width (int): Number of images per row in the grid.
        subsample (int): Stride-subsample each tile before compositing (default 1 = no-op).

    Returns:
        np.ndarray: The resulting grid image.
    """
    if subsample > 1:
        min_h = min(img.shape[0] for img in images)
        min_w = min(img.shape[1] for img in images)
        subsample = max(1, min(subsample, min_h, min_w))
        images = [img[::subsample, ::subsample] for img in images]

    # Pre-process dimensions in a single pass
    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)
    channels = max(img.shape[2] if img.ndim == 3 else 1 for img in images)

    # Calculate grid dimensions
    num_images = len(images)
    grid_height = (num_images + grid_width - 1) // grid_width

    # Pre-allocate the output grid
    grid_image = np.zeros(
        (grid_height * max_height, grid_width * max_width, channels),
        dtype=images[0].dtype,
    )

    for idx, img in enumerate(images):
        # Convert to 3D if needed
        if img.ndim == 2:
            img = img[..., None]

        h, w = img.shape[:2]
        row, col = divmod(idx, grid_width)

        # Calculate center offsets
        y_offset = (max_height - h) // 2
        x_offset = (max_width - w) // 2

        # Direct slice assignment (no intermediate padding)
        y_start = row * max_height + y_offset
        x_start = col * max_width + x_offset
        grid_image[y_start : y_start + h, x_start : x_start + w, : img.shape[2]] = img

    # If single-channel, squeeze to remove unnecessary dimensions
    return grid_image.squeeze(-1) if channels == 1 else grid_image


def save_video(
    video_file: str,
    images: Sequence[np.ndarray],
    fps: int = 30,
    macro_block_size: int = 1,
    format: str = "mp4",
) -> None:
    """Save a sequence of images as a video file, ensuring images have even dimensions for ffmpeg.

    Args:
        video_file (str): Path to save the video file.
        images (list[np.ndarray]): List of images to be saved as video frames.
        fps (int, optional): Frames per second for the video. Defaults to 30.
        macro_block_size (int, optional): Macro block size for the video codec. Defaults to 1.
        format (str, optional): Video format. Defaults to "mp4".
    """
    # Chop images if dimensions not divisible by 2 to fix ffmpeg errors
    processed_images = []
    unaligned_warning = False
    for img in images:
        # Remove last row if height is odd
        if img.shape[0] % 2 == 1:
            img = img[:-1, :]
        # Remove last column if width is odd
        if img.shape[1] % 2 == 1:
            img = img[:, :-1]
        processed_images.append(img)

        if img.shape[0] % 16 != 0:
            unaligned_warning = True

    if unaligned_warning:
        logger.warning("Video width not multiple of 16 - ffmpeg will warn about unaligned data")

    # Results in an Array is not the same as ArrayLike annotation error, so we suppress it
    imageio.v2.mimwrite(video_file, processed_images, fps=fps, macro_block_size=macro_block_size, format=format)  # type: ignore
    logger.info(f"Saved {len(images)} frames into '{video_file}'")


def draw_text_overlay(im: np.ndarray, overlay_text: str):
    from PIL import Image, ImageDraw, ImageFont

    # Convert numpy array to PIL Image
    pil_img = Image.fromarray(im)
    draw = ImageDraw.Draw(pil_img)

    # Try to use a default font, fallback to default if not available
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 36)
        except:
            font = ImageFont.load_default()

    # Get text size for positioning
    bbox = draw.textbbox((0, 0), overlay_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Position text in top right corner with padding
    x = im.shape[1] - text_width - 15
    y = 15

    # Draw text with black outline for better visibility
    outline_color = (0, 0, 0)
    text_color = (255, 255, 255)

    # Draw outline
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), overlay_text, font=font, fill=outline_color)

    # Draw main text
    draw.text((x, y), overlay_text, font=font, fill=text_color)

    # Convert back to numpy array and update the original array
    im[:] = np.array(pil_img)


DEFAULT_PS_VIEW_JSON_STR = """
{
    "farClipRatio": 1000.0,
    "nearClipRatio": 0.005,
    "fov": 45.0,
    "projectionMode": "Perspective",
    "viewMat": [0.0, -1.0, 0.0, -0.0, -0.0, 0.0, 1.0, -0.0, -1.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    "windowHeight": 1080,
    "windowWidth": 1920
}"""
