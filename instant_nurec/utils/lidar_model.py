# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import ncore.data
import ncore.data.v4


def get_lidar_model_parameters(
    lidar_sensor: ncore.data.LidarSensorProtocol,
) -> ncore.data.ConcreteLidarModelParametersUnion | None:
    """Read lidar model parameters from a NCore V4 lidar sensor."""
    assert isinstance(lidar_sensor, ncore.data.v4.SequenceLoaderV4.LidarSensor), (
        f"Unsupported lidar sensor type: {type(lidar_sensor)} -- expected NCore V4 LidarSensor"
    )
    return lidar_sensor.model_parameters
