# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from nre.config.base_schema import BaseConfigSchema, Field


class ProberConfig(BaseConfigSchema):
    """
    Configuration for TensorProber utility.

    Controls the debugging tensor prober utility that saves and loads tensors
    for debugging and testing purposes.
    """

    enabled: bool = Field(default=False, description="Enable or disable tensor saving/loading functionality")

    test_data_dir: str = Field(default="test_data", description="Directory where tensors are saved/loaded")

    every_n_steps: int = Field(default=5000, gt=0, description="Save tensors every n steps during training")

    batch_limit: int = Field(
        default=100000, ge=0, description="Limit the number of elements saved to this value. Set to 0 for no limit"
    )
