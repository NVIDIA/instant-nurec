# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import math

from typing import Tuple

import numpy as np
import scipy

from ncore.data import BBox3, FThetaCameraModelParameters, OpenCVPinholeCameraModelParameters, ShutterType
from ncore_internal.nvidia import approximate_polynomial_inverse, compute_max_radius


def calculate_bbox(bbox: list, t_lidar_bbox: np.ndarray, scale: float = 1.0) -> Tuple[BBox3, np.ndarray]:
    """Calculate the BBox3 given the box data from drivesim

    Args:
        bbox (list): The bounding box data from drivesim
        t_lidar_bbox (np.array): The transform from the lidar to the bounding box
        scale (float, optional): The bounding box scale. Defaults to 1.0.

    Returns:
        BBox3: Returns the BBox3 with centroid, dimensions and eulers.
        np.ndarray: Returns the BBox as a tensor.
    """
    eps = 1e-6

    p1 = t_lidar_bbox @ np.array([bbox[1], bbox[2], bbox[3], 1.0])
    p_x = t_lidar_bbox @ np.array([bbox[4], bbox[2], bbox[3], 1.0])
    p_y = t_lidar_bbox @ np.array([bbox[1], bbox[5], bbox[3], 1.0])
    p_z = t_lidar_bbox @ np.array([bbox[1], bbox[2], bbox[6], 1.0])
    p2 = t_lidar_bbox @ np.array([bbox[4], bbox[5], bbox[6], 1.0])
    centroid3 = [(p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5, (p1[2] + p2[2]) * 0.5]
    dim3 = [np.linalg.norm(p1 - p_x) * scale, np.linalg.norm(p1 - p_y) * scale, np.linalg.norm(p1 - p_z) * scale]
    # unit vectors for rotation matrix
    x: np.ndarray = (p_x - p1) / (dim3[0] / scale + eps)
    y: np.ndarray = (p_y - p1) / (dim3[1] / scale + eps)
    z: np.ndarray = (p_z - p1) / (dim3[2] / scale + eps)
    rotation_matrix = np.array([[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]], dtype=np.float32)
    eulers = scipy.spatial.transform.Rotation.from_matrix(rotation_matrix).as_euler("xyz", degrees=False)

    def tuple_of_floats(arr: np.ndarray | list) -> Tuple[float, float, float]:
        return (float(arr[0]), float(arr[1]), float(arr[2]))

    return BBox3(tuple_of_floats(centroid3), tuple_of_floats(dim3), tuple_of_floats(eulers)), np.array(
        centroid3 + dim3 + eulers.tolist()
    )


def create_ncore_camera_model(cam_params: dict) -> OpenCVPinholeCameraModelParameters | FThetaCameraModelParameters:
    resolution = [int(cam_params["renderProductResolution"][0]), int(cam_params["renderProductResolution"][1])]
    horiz_aperture = cam_params["cameraAperture"][0]
    vert_aperture = resolution[1] / resolution[0] * horiz_aperture
    focal_length = cam_params["cameraFocalLength"]

    width = resolution[0]
    height = resolution[1]

    if cam_params["cameraModel"] == "pinhole":
        fx = resolution[0] * focal_length / horiz_aperture
        fy = resolution[1] * focal_length / vert_aperture
        cx = resolution[0] * 0.5
        cy = resolution[1] * 0.5
        f_u = float(fx)
        f_v = float(fy)
        c_u = float(cx)
        c_v = float(cy)

        # No distortion
        k1, k2, k3 = 0, 0, 0
        p1, p2 = 0, 0

        return OpenCVPinholeCameraModelParameters(
            resolution=np.array([width, height], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([c_u, c_v], dtype=np.float32),
            focal_length=np.array([f_u, f_v], dtype=np.float32),
            radial_coeffs=np.array([k1, k2, k3, 0, 0, 0], dtype=np.float32),
            tangential_coeffs=np.array([p1, p2], dtype=np.float32),
            thin_prism_coeffs=np.array([0, 0, 0, 0], dtype=np.float32),
        )

    # drivesim's fisheyePolynomial is the ftheta polynomial model
    assert cam_params["cameraModel"] == "fisheyePolynomial"
    cx = cam_params["cameraFisheyeOpticalCentre"][0]
    cy = cam_params["cameraFisheyeOpticalCentre"][1]
    c_u = float(cx)
    c_v = float(cy)

    max_angle = math.radians(cam_params["cameraFisheyeMaxFOV"] / 2)
    pixeldist_to_angle_poly = np.array([float(p) for p in cam_params["cameraFisheyePolynomial"]], dtype=np.float32)
    angle_to_pixeldist_poly = approximate_polynomial_inverse(
        pixeldist_to_angle_poly,
        0.0,
        compute_max_radius(np.array(resolution, dtype=np.float64), np.array([c_u, c_v])),
    )

    # U and V coordinate of the principal point, following the NVIDIA default convention for FTheta camera models
    # in which the pixel indices represent the center of the pixel (not the top-left corners).
    # Principal point coordinates will be adapted internally in camera model APIs to reflect the :ref:`image
    # coordinate conventions <image_coordinate_conventions>`
    return FThetaCameraModelParameters(
        resolution=np.array(resolution, dtype=np.uint64),
        shutter_type=ShutterType.GLOBAL,
        principal_point=np.array([c_u, c_v], dtype=np.float32),
        # Indicating which of the two stored polynomials is the model's
        # *reference* polynomial (the other polynomial is only an approximation)
        reference_poly=FThetaCameraModelParameters.PolynomialType.PIXELDIST_TO_ANGLE,
        # Coefficients of the pixeldistances-to-angles polynomial (float32, [6,])
        pixeldist_to_angle_poly=pixeldist_to_angle_poly,
        # Coefficients of the angles-to-pixeldistances polynomial (float32, [6,])
        angle_to_pixeldist_poly=angle_to_pixeldist_poly,
        # Maximal extrinsic ray angle [rad] with the principal direction (float32)
        max_angle=max_angle,
    )
