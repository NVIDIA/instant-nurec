# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import glob
import os
import pickle

from pathlib import Path

import click
import numpy as np

from point_cloud_utils import TriangleMesh  # type: ignore
from scipy.spatial.transform import Rotation as R

from internal.scripts.experimental.io import load_pc_dat, save_pc_dat


@click.command()
@click.option(
    "--path",
    type=str,
    help="Path to the preprocessed NCORE sequence to update",
    default="/mount/zfs/users/rdelutio/new_waymo_selection/10275144660749673822_5755_561_5775_561/",
    required=True,
)
@click.option(
    "--output-dir",
    type=str,
    help="Path to the output target directory - if missing, will update the original sequence in place",
    required=True,
    default=None,
)
def mark_obj_as_dynamic(path: str, output_dir: str):
    # Load original frame data

    all_lidar_frames = sorted(glob.glob(os.path.join(path, "lidar", "*.dat*")))

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    for frame in all_lidar_frames:
        # Can have two endings so do stem twice
        frame_name = frame.split(os.sep)[-1].split(".")[0]

        pc_data = load_pc_dat(frame)
        meta_data = load_pkl(os.path.join(path, "lidar", frame_name + ".pkl"))
        # box.center_x, box.center_y, box.center_z, box.length, box.width, box.height, 0, 0, box.heading
        frame_labels = load_pkl(os.path.join(path, "labels", frame_name + ".pkl"))

        # Construct world -> lidar transformation
        T_rig_world = meta_data["ego_pose"]

        T_world_rig = np.linalg.inv(T_rig_world)
        dynamic_flag = pc_data[:, -1]

        # Transform points from world to lidar
        xyz_world_homogeneous = np.row_stack(
            [pc_data[:, 3:6].transpose(), np.ones(pc_data.shape[0], dtype=np.float32)]
        )  # 4 x N
        xyz_rig_homogeneous = T_world_rig @ xyz_world_homogeneous  # 4 x N

        xyz = xyz_rig_homogeneous[:3, :].transpose()  # N x 3
        for label in frame_labels["lidar_labels"]:
            bbox = label["3D_bbox"]
            dynamic_idx = is_within_3d_bbox(xyz, bbox, normals=None, return_points_in_bbox_frame=False)
            dynamic_flag[dynamic_idx] = 1.0

        # Set point-cloud dynamic flag and serialize updated point-cloud
        pc_data[:, -1] = dynamic_flag
        if output_dir is not None:
            save_path = Path(output_dir, frame_name + ".dat")
            save_pc_dat(str(save_path), pc_data)
            pc = TriangleMesh()
            print(pc_data[:, 3:6].shape)
            print(pc_data[:, -1].reshape(-1, 1).shape)
            pc.vertex_data.positions = pc_data[:, 3:6]
            pc.vertex_data.custom_attributes["dynamic_flag"] = pc_data[:, -1]
            pc.save(str(save_path.with_suffix(".ply")))
        else:
            save_pc_dat(frame, pc_data)


def load_pkl(path):
    """
    Load a .pkl object
    """
    file = open(path, "rb")
    return pickle.load(file)


def so3_trans_2_se3(so3, trans):
    """Create a 4x4 rigid transformation matrix given so3 rotation and translation.

    Args:
        so3: rotation matrix [n,3,3]
        trans: x, y, z translation [n, 3]

    Returns:
        np.ndarray: the constructed transformation matrix [n,4,4]
    """

    if so3.ndim > 2:
        T = np.eye(4)
        T = np.tile(T, (so3.shape[0], 1, 1))
        T[:, 0:3, 0:3] = so3
        T[:, 0:3, 3] = trans.reshape(
            -1,
            3,
        )

    else:
        T = np.eye(4)
        T[0:3, 0:3] = so3
        T[0:3, 3] = trans.reshape(
            3,
        )

    return T


def is_within_3d_bbox(points, box, normals=None, return_points_in_bbox_frame=False):
    """Checks whether a point is in a 3d box given a set of points and a box.
    Args:
        point: [N, 3] tensor. Inner dims are: [x, y, z].
        box: [11] tensor. Inner dims are: [center_x, center_y, center_z, length,
        width, height, ref_velocity[0], ref_velocity[1], heading_x, heading_y, heading_z].
        name: tf name scope.
    Returns:
        point_in_box; [N,] boolean array.
    """

    center = box[0:3]
    dim = box[3:6]
    rotation_angles = box[-3:]

    # Get the rotation matrix from the heading angle
    rotation = R.from_euler("xyz", rotation_angles, degrees=False).as_matrix()

    # [4, 4]
    transform = so3_trans_2_se3(rotation, center)
    # [4, 4]
    transform = np.linalg.inv(transform)
    # [3, 3]
    rotation = transform[0:3, 0:3]
    # [3]
    translation = transform[0:3, 3]

    # [M, 3]
    points_in_box_frames = np.matmul(rotation, points.transpose()).transpose() + translation

    # [M, 3]
    point_in_box = np.logical_and(
        np.logical_and(points_in_box_frames <= dim * 0.5, points_in_box_frames >= -dim * 0.5),
        np.all(np.not_equal(dim, 0), axis=-1, keepdims=True),
    )

    # [N, M]
    point_in_box = np.prod(point_in_box, axis=-1).astype(bool)

    if not return_points_in_bbox_frame:
        return point_in_box
    else:
        if normals is not None:
            T_normals = np.linalg.inv(transform).transpose()

            normals_in_bbox_frame = (
                np.matmul(T_normals[0:3, 0:3], normals[point_in_box, :].transpose()).transpose() + T_normals[0:3, 3]
            )

            return points_in_box_frames[point_in_box, :], normals_in_bbox_frame / np.linalg.norm(
                normals_in_bbox_frame, axis=1, keepdims=True
            )
        else:
            return points_in_box_frames[point_in_box, :]


if __name__ == "__main__":
    mark_obj_as_dynamic()
