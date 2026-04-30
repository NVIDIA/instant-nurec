# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Tool for creating visual comparison GIFs between two images."""

import sys

from pathlib import Path

import click

from PIL import Image


@click.command()
@click.argument("input1", type=click.Path(exists=True))
@click.argument("input2", type=click.Path(exists=True))
@click.argument("out", type=click.Path())
@click.option(
    "--delay",
    type=int,
    default=500,
    show_default=True,
    help="Delay between frames in milliseconds.",
)
@click.option(
    "--loop",
    type=int,
    default=0,
    show_default=True,
    help="Number of times to loop (0 = infinite).",
)
def create_comparison_gif(
    input1: str,
    input2: str,
    out: str,
    delay: int,
    loop: int,
) -> None:
    """Create an animated GIF that alternates between two images for visual comparison.

    INPUT1 is the path to the first input image and INPUT2 the path to the second input image.
    Both must have the same size.

    OUT is the path and filename for the output GIF.
    """
    try:
        # Open both images
        img1 = Image.open(input1)
        img2 = Image.open(input2)

        # Ensure both images have the same size
        if img1.size != img2.size:
            print(
                f"Error: Images have different sizes. input1: {img1.size}, input2: {img2.size}.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Save as animated GIF
        img1.save(
            out,
            save_all=True,
            append_images=[img2],
            duration=delay,
            loop=loop,
            optimize=False,  # Don't optimize to preserve quality
        )

        print(f"Successfully created comparison GIF: {Path(out).absolute()}")

    except Exception as e:
        print(f"Error creating comparison GIF: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    create_comparison_gif()
