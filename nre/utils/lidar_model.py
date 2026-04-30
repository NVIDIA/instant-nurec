# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import ncore.data
import ncore.data.v4
import ncore_internal.data.v3


def get_lidar_model_parameters_with_fallbacks(
    lidar_sensor: ncore_internal.data.v3.LidarSensor | ncore.data.LidarSensorProtocol,
    cwccw_fallback: bool = True,
) -> ncore.data.ConcreteLidarModelParametersUnion | None:
    """
    Robustly get lidar model parameters from the sensor, applying optional fallbacks if enabled
    """

    try:
        match lidar_sensor:
            case ncore_internal.data.v3.LidarSensor():
                # V3 native lidar sensor
                lidar_model_parameters = lidar_sensor.get_lidar_model_parameters()
            case ncore_internal.data.v3.SequenceLoaderV3.LidarSensor() | ncore.data.v4.SequenceLoaderV4.LidarSensor():
                # V4 compat sensor
                lidar_model_parameters = lidar_sensor.model_parameters
            case _:
                raise ValueError(f"Unsupported lidar sensor type: {type(lidar_sensor)}")
    except AssertionError as e:
        # Temporary WAR to support datasets generated with wrong conventions (prior to nrs/ncore!363)
        if not cwccw_fallback:
            raise AssertionError(
                f"{e} - Lidar model parameters may follow outdated conventions. "
                f"Consider enabling 'lidar_model_parameter_cwccw_fallback' in the config to apply a fallback."
            )

        match lidar_sensor:
            case ncore_internal.data.v3.LidarSensor():
                sensor_meta = lidar_sensor._sensor_meta
            case ncore_internal.data.v3.SequenceLoaderV3.LidarSensor():
                sensor_meta = lidar_sensor._sensor._sensor_meta
            case _:
                raise ValueError(
                    f"Unsupported lidar sensor type for cwccw fallback: {type(lidar_sensor)} - fallback only supported for V3 datasets"
                ) from e

        assert sensor_meta.lidar_model_type == ncore.data.RowOffsetStructuredSpinningLidarModelParameters.type(), (
            f"get_lidar_model_parameters_with_fallbacks: cwccw fallback only supported for RowOffsetStructuredSpinningLidarModelParameters"
        )
        lidar_model_parameters_dict = sensor_meta.lidar_model_parameters.copy()

        # Flip orientation of the spin direction (data was saved in the exact opposite order)
        lidar_model_parameters_dict["spinning_direction"] = (
            "cw" if lidar_model_parameters_dict["spinning_direction"] == "ccw" else "ccw"
        )

        # Remove obsolve FOV parameters
        for fov_key in [
            "fov_horiz_start_rad",
            "fov_horiz_end_rad",
            "fov_vert_start_rad",
            "fov_vert_end_rad",
            "fov_horiz_min_rad",
            "fov_vert_min_rad",
            "fov_horiz_max_rad",
            "fov_vert_max_rad",
        ]:
            lidar_model_parameters_dict.pop(fov_key, None)

        lidar_model_parameters = ncore.data.RowOffsetStructuredSpinningLidarModelParameters.from_dict(
            lidar_model_parameters_dict
        )

    return lidar_model_parameters
