# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# This script stores per-frame normal/depth inferences results into ncore aux data format.
# It expect the following folder structure:
# Depth:
# The same folder structure the same as 3D Reconstruction Pipeline:
# https://docs.google.com/document/d/1q9kzI4pKNV2bVAh5G-ebxSl3mCaeiTNNFBGIe_gh5X8/edit#heading=h.vzd17gv0bfj8
# eg:
# 00000000_timestamp_us.npy
# 00000000_dis.npy
# 00000001_timestamp_us.npy
# 00000001_dis.npy
# Normal:
# The 3d quantized normals encoded in the 3-channel png image as:
# normal = (image_data / 127.5 - 1).normalized
# <end_timestamp_us>.png
# eg:
# 1667510770555007.png

import argparse
import glob
import os

from pathlib import Path

import numpy as np

from PIL import Image
from tqdm import tqdm

from nre.utils.ncore_utils import AuxDataWriter


def write_depth_aux_data(
    writer: AuxDataWriter,
    camera_id: str,
    depth_inference_result_dir: str,
    store_depth_as_png,
    max_depth: float,
    method="monodepth",
):
    timestamp_files = glob.glob(os.path.join(depth_inference_result_dir, "*_timestamp_us.npy"))
    for i, timestamp_file in enumerate(tqdm(timestamp_files, "Processing depth data.")):
        depth_file = timestamp_file.replace("timestamp_us", "dist")
        assert os.path.exists(depth_file)
        timestamp = int(np.load(timestamp_file))
        depth_data = np.load(depth_file)

        if i == 0:
            writer.store_depth_meta(
                camera_id, [depth_data.shape[1], depth_data.shape[0]], store_depth_as_png, max_depth, method
            )

        writer.store_depth(camera_id, timestamp, depth_data)


def write_normal_aux_data(
    writer: AuxDataWriter,
    camera_id: str,
    normal_inference_result_dir: str,
    method: str,
):
    normal_image_files = glob.glob(os.path.join(normal_inference_result_dir, "*.png"))
    for i, normal_image_file in enumerate(tqdm(normal_image_files, "Processing normal data")):
        timestamp = int(Path(normal_image_file).stem)
        normals_quantized = np.array(Image.open(normal_image_file))

        normals = normals_quantized / 127.5 - 1
        if i == 0:
            writer.store_normal_meta(camera_id, [normals.shape[1], normals.shape[0]], method)

        writer.store_normal(camera_id, timestamp, normals)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Store depth or normal aux data")
    arg_parser.add_argument(
        "-d",
        "--input-depth-dir",
        default="",
        required=False,
        help="Directory that contains the depth inference results.",
    )
    arg_parser.add_argument(
        "-n",
        "--input-normal-dir",
        default="",
        required=False,
        help="Directory that contains the normal inference results.",
    )
    arg_parser.add_argument("--max-depth", type=float, default=250.0, help="Max depth for quantizing depth values.")
    arg_parser.add_argument(
        "--no-store-depth-as-png",
        action="store_true",
        help="By default depth will be quantized and stored as png."
        "This option disables that behavior and store the depth values as float16 directly.",
    )
    arg_parser.add_argument("-o", "--output-dir", required=True, help="Output directory for the aux data.")
    arg_parser.add_argument("--camera-id", required=True, help="Camera id.")
    arg_parser.add_argument("--base-name", required=True, help="Store base name.")
    arg_parser.add_argument("--sequence-id", required=False, help="Sequence id. If empty, base name will be used.")

    args = arg_parser.parse_args()

    assert os.path.exists(args.input_depth_dir) or os.path.exists(args.input_normal_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    sequence_id = args.sequence_id if args.sequence_id else args.base_name

    writer = AuxDataWriter(Path(args.output_dir), args.base_name, sequence_id, 0, 1)

    if args.input_depth_dir:
        write_depth_aux_data(
            writer,
            args.camera_id,
            args.input_depth_dir,
            store_depth_as_png=(not args.no_store_depth_as_png),
            max_depth=args.max_depth,
        )

    if args.input_normal_dir:
        write_normal_aux_data(
            writer,
            args.camera_id,
            args.input_normal_dir,
            method="mononormal",  # TODO(mengxiwu): add switch to script once parsing data from different source methods
        )

    writer.finalize()
