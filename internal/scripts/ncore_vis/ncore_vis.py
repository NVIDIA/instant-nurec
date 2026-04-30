# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import click

from internal.scripts.ncore_vis.viewer import NCoreViewer


@click.command()
@click.option(
    "--shard-file-pattern",
    type=str,
    help="Path to an output directory to dump into",
    required=True,
)
@click.option(
    "--ply-path",
    type=str,
    help="Path to a PLY point cloud file to overlay in the viewer",
    default=None,
)
def ncore_vis(shard_file_pattern: str, ply_path: str | None) -> None:
    viewer = NCoreViewer(shard_file_pattern, ply_path=ply_path)
    viewer.start_server()


if __name__ == "__main__":
    ncore_vis()
