# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Constant definitions for Slang losses and script to generate -D flags for slangc compilation.

This module contains constant definitions with minimal dependencies, allowing it to be:
1. Imported by slang_losses.py for runtime use
2. Executed as a script to generate compile-time -D flags for slangc
"""

# Bilateral Grid constants, affine transformation matrix/grid dimensions
GRID_NUM_ROWS: int = 3
GRID_NUM_COLS: int = 4
GRID_NUM_CHANNELS: int = GRID_NUM_ROWS * GRID_NUM_COLS

# Block dimensions for slang_losses_kernel
BLOCK_THREADS = 256

# PPISP constants
NUM_VIGNETTING_ALPHA_TERMS = 3
DEFAULT_SOURCE_CHROMS_VALUES: list[list[float]] = [
    [0.0, 0.0],  # pure blue
    [1.0, 0.0],  # pure red
    [0.0, 1.0],  # pure green
    [1 / 3, 1 / 3],  # neutral gray
]
NUM_SOURCE_CHROMS = len(DEFAULT_SOURCE_CHROMS_VALUES)
EPSILON = 1e-5


def main():
    """Extract constants and print them as -D flags for slangc."""
    # Import RayFlags only when running as a script (not at module level to avoid circular import)
    from nre.utils.types import RayFlags

    # Extract block dimensions
    block_threads = BLOCK_THREADS

    # Extract grid affine matrix dimensions
    grid_num_rows = GRID_NUM_ROWS
    grid_num_cols = GRID_NUM_COLS
    grid_num_channels = GRID_NUM_CHANNELS

    # Extract RayFlags values
    rgb_label = int(RayFlags.RGB_LABEL)
    dropped = int(RayFlags.DROPPED)
    invalid = int(RayFlags.INVALID)
    sky_semantic = int(RayFlags.SKY_SEMANTIC)
    difixed = int(RayFlags.DIFIXED)
    synthetic = int(RayFlags.SYNTHETIC)

    # Print as -D flags (used by Slang via extra_cmd_args, and by CUDA via generated header)
    defines = [
        f"-DBLOCK_THREADS={block_threads}",
        f"-DGRID_NUM_ROWS={grid_num_rows}",
        f"-DGRID_NUM_COLS={grid_num_cols}",
        f"-DGRID_NUM_CHANNELS={grid_num_channels}",
        f"-DRGB_LABEL={rgb_label}",
        f"-DDROPPED={dropped}",
        f"-DINVALID={invalid}",
        f"-DSKY_SEMANTIC={sky_semantic}",
        f"-DDIFIXED={difixed}",
        f"-DSYNTHETIC={synthetic}",
        f"-DNUM_VIGNETTING_ALPHA_TERMS={NUM_VIGNETTING_ALPHA_TERMS}",
        f"-DEPSILON={EPSILON}",
    ]

    print(" ".join(defines))


if __name__ == "__main__":
    main()
