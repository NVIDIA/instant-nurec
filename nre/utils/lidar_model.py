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


def get_lidar_model_parameters(
    lidar_sensor: ncore.data.LidarSensorProtocol,
) -> ncore.data.ConcreteLidarModelParametersUnion | None:
    """Read lidar model parameters from a NCore V4 lidar sensor.

    Predict-only standalone reads ncorev4 only; the NRE-side V3 native
    sensor branch and the cwccw fallback (which fired only for V3
    datasets generated before `nrs/ncore!363`) were dropped together
    with the V3 sequence loader (Phase 1 step 4.3).
    """
    assert isinstance(lidar_sensor, ncore.data.v4.SequenceLoaderV4.LidarSensor), (
        f"Unsupported lidar sensor type: {type(lidar_sensor)} -- expected NCore V4 LidarSensor"
    )
    return lidar_sensor.model_parameters
