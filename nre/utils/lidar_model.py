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

import json

import numpy as np
import torch

from omegaconf import DictConfig, ListConfig, OmegaConf

import ncore.data
import ncore.data.v4
import ncore_internal.data.v3

from libs.vren.lidars import (  # type: ignore
    preprocess_lidar_raygen_only,
)
from ncore.sensors import LidarModel
from nre.utils.misc import unpack_optional


class LidarModelBundle:
    """
    Wrapper around the meta data for rendering of a Lidar model.
    """

    @classmethod
    def load_from_config(cls, lidar_config: dict) -> LidarModelBundle:
        # Handle the case where lidar_config is a dictionary, but might contain OmegaConf objects
        processed_config = {}

        for key, value in lidar_config.items():
            if isinstance(value, (ListConfig, DictConfig)):
                processed_config[key] = OmegaConf.to_container(value, resolve=True)
            else:
                processed_config[key] = value

        data = json.dumps(processed_config)

        lidar_parameters = ncore.data.RowOffsetStructuredSpinningLidarModelParameters.from_json(data)
        return cls(lidar_parameters)

    def __init__(self, lidar_parameters: ncore.data.ConcreteLidarModelParametersUnion):
        self.lidar_parameters = lidar_parameters
        self.lidar_model = unpack_optional(LidarModel.maybe_from_parameters(lidar_parameters))
        # This model is just for converting elements to world rays, hard code the parameters for tile.
        self.vren_lidar = preprocess_lidar_raygen_only(lidar_parameters, device=torch.device("cuda"))
        self.elements = np.stack(
            np.meshgrid(
                np.arange(lidar_parameters.n_rows, dtype=np.uint16),
                np.arange(lidar_parameters.n_columns, dtype=np.uint16),
                indexing="ij",
            ),
            axis=-1,
        ).reshape((-1, 2))


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
