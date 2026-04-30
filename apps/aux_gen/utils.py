# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Tuple

import matplotlib.cm
import numpy as np
import torch

from scipy.spatial import Delaunay


def plot_points_on_image(image, pixel_indices, dist_rs, psize=2):
    # TODO ZG: I don't know how to properly type the colormaps un matplotlib. check if easily solvable
    colors = matplotlib.cm.jet((dist_rs.squeeze() % 50.0) / 50.0)  # type: ignore
    for x_offset in range(-psize, psize):
        for y_offset in range(-psize, psize):
            y = pixel_indices[:, 1] + y_offset
            x = pixel_indices[:, 0] + x_offset
            y = y.clip(0, image.shape[0] - 1)
            x = x.clip(0, image.shape[1] - 1)
            image[y, x, :] = colors[:, :3] * 255
    return image


def plot_points_on_image_with_color(image, pixel_indices, colors, psize=2):
    """Plots points on a camera image with colors provided.

    Args:
        image: camera image, numpy array.
        pixel_indices: [N, 2] numpy array. The inner dims are
            [camera_x, camera_y].
        colors: [N, 3], numpy array. rgb value of points [0~255].
        psize: the point size, int.

    """
    h, w = image.shape[:2]
    for x_offset in range(-psize, psize):
        for y_offset in range(-psize, psize):
            y = pixel_indices[:, 1] + y_offset
            x = pixel_indices[:, 0] + x_offset
            y = y.clip(0, h - 1)
            x = x.clip(0, w - 1)
            image[y, x, :] = colors[:, :3]
    return image


class LidarSpinMesh:
    vertices: np.ndarray
    faces: np.ndarray

    def __init__(self, lidar_ray_starts: np.ndarray, lidar_ray_ends: np.ndarray) -> None:
        self.vertices = lidar_ray_ends
        self.faces = self._construct_mesh(lidar_ray_starts, lidar_ray_ends)

    def _construct_mesh(cls, lidar_ray_starts: np.ndarray, lidar_ray_ends: np.ndarray) -> np.ndarray:
        def _xyz_to_spherical(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Convert the point cloud to spherical coordinates, assuming that points are centered and the z axis points up"""
            if points.shape[1] != 3:
                raise ValueError(
                    f"Second dim of array points must be equal to 3. Shape of array points: {points.shape}"
                )
            radii = np.linalg.norm(points, axis=1)
            azimuth = np.arctan2(points[:, 1], points[:, 0])  # Ranges [-pi, +pi]
            elevation = np.arcsin(points[:, 2] / radii)
            return azimuth, elevation, radii

        # center points
        points = lidar_ray_ends - lidar_ray_starts
        azimuth, elevation, _ = _xyz_to_spherical(points)
        azimuth_elevation = np.stack([azimuth, elevation, np.zeros_like(azimuth)], axis=1)
        faces = Delaunay(azimuth_elevation[:, 0:2]).simplices
        return faces


def _quantile(tensor, q, dim=None, keepdim=False):
    """
    Computes the quantile of the input tensor along the specified dimension.  This function
    is a workaround for issue 64947 when there are more than 16 million elements:
    https://github.com/pytorch/pytorch/issues/64947

    Parameters:
    tensor (torch.Tensor): The input tensor.
    q (float): The quantile to compute, should be a float between 0 and 1.
    dim (int): The dimension to reduce. If None, the tensor is flattened.
    keepdim (bool): Whether to keep the reduced dimension in the output.
    Returns:
    torch.Tensor: The quantile value(s) along the specified dimension.
    """
    assert 0 <= q <= 1, "\n\nquantile value should be a float between 0 and 1.\n\n"

    if dim is None:
        tensor = tensor.flatten()
        dim = 0

    sorted_tensor, _ = torch.sort(tensor, dim=dim)
    num_elements = sorted_tensor.size(dim)
    index = q * (num_elements - 1)
    lower_index = int(index)
    upper_index = min(lower_index + 1, num_elements - 1)
    lower_value = sorted_tensor.select(dim, lower_index)
    upper_value = sorted_tensor.select(dim, upper_index)
    # linear interpolation
    weight = index - lower_index
    quantile_value = (1 - weight) * lower_value + weight * upper_value

    return quantile_value.unsqueeze(dim) if keepdim else quantile_value
