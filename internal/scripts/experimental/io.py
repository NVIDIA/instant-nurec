# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import io
import lzma
import struct

from typing import Union

import numpy as np


def save_pc_dat(file_path: str, lidar_data: np.ndarray) -> None:
    """
    Stores binary .dat / .dat.xz file representing a 2D single-precision array, usually representing
    a point-cloud (see load_pc_dat for format description).

    Args:
        file_path: path to .dat / .dat.xz file to store
        lidar_data: 2D single-precision array to serialize
    """

    if lidar_data.dtype is not np.dtype("float32"):
        raise ValueError("expecting single-precision array as input")

    def save(file: Union[io.BufferedWriter, lzma.LZMAFile]) -> None:
        n_rows, n_columns = lidar_data.shape
        lidar_data_flat = lidar_data.flatten()

        file.write(struct.pack("<i", n_rows))
        file.write(struct.pack("<i", n_columns))
        file.write(struct.pack("<%sf" % lidar_data_flat.size, *lidar_data_flat))

    if file_path.endswith(".dat"):
        with open(file_path, "wb") as file:
            save(file)
    elif file_path.endswith(".dat.xz"):
        with lzma.open(
            file_path,
            "wb",
            # Use fastest possible compression mode which still gives acceptable compression rates
            preset=0,
        ) as lzma_file:
            save(lzma_file)
    else:
        raise ValueError("invalid file format provided, supporting .dat / .dat.xz files only")


def load_pc_dat(file_path: str, allow_lookup_fallback: bool = True) -> np.ndarray:
    """
    Loads binary .dat / .dat.xz files representing a 2D single-precision array.
    Serialized 2D arrays usually represent a point-clouds with columns defined as

    [x_s, y_s, z_s, x_e, y_e, z_e, dist, intensity, dynamic_flag]

    - xys_s / xyz_e: the start / end point of world rays
    - dist: the norm of the ray
    - intensity: lidar intensity response value for this point
    - dynamic_flag:
      - -1: if the information is not available,
      -  0: static
      -  1: = dynamic

    Args:
        file_path: path to .dat / .dat.xz file to load.
        allow_lookup_fallback: If enabled, will fall back to .dat.xz/.dat, resp., in case loading .dat/.dat.xz fails (for backwards-compatibility).
    Return:
        lidar_data: loaded 2D single-precision array
    """

    def load(file: Union[io.BufferedReader, lzma.LZMAFile]) -> np.ndarray:
        # The first number denotes the number of points
        n_rows, n_columns = struct.unpack("<ii", file.read(8))
        # The remaining data are floats saved in little endian
        # Columns usually contain: x_s, y_s, z_s, x_e, y_e, z_e, d, intensity, dynamic_flag
        # Dynamic flag is set to -1 if the information is not available, 0 static, 1 = dynamic
        return np.array(struct.unpack("<%sf" % (n_rows * n_columns), file.read()), dtype=np.float32).reshape(
            n_rows, n_columns
        )

    if file_path.endswith(".dat"):
        try:
            with open(file_path, "rb") as file:
                lidar_data = load(file)
        except FileNotFoundError as e:
            if allow_lookup_fallback:
                with lzma.open(file_path + ".xz", "rb") as lzma_file:
                    lidar_data = load(lzma_file)
            else:
                raise e
    elif file_path.endswith(".dat.xz"):
        try:
            with lzma.open(file_path, "rb") as lzma_file:
                lidar_data = load(lzma_file)
        except FileNotFoundError as e:
            if allow_lookup_fallback:
                with open(file_path.replace(".dat.xz", ".dat"), "rb") as file:
                    lidar_data = load(file)
            else:
                raise e
    else:
        raise ValueError("invalid file format provided, supporting .dat / .dat.xz files only")

    return lidar_data
