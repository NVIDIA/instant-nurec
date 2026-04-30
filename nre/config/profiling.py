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


class ProfilingActivities(BaseConfigSchema):
    cpu: bool = Field(default=True)
    cuda: bool = Field(default=True)


class ProfilingParams(BaseConfigSchema):
    start_step: int = Field(default=90)
    num_steps: int = Field(default=10)

    activities: ProfilingActivities = Field(default_factory=ProfilingActivities)

    record_shapes: bool = Field(default=True)
    profile_memory: bool = Field(default=True)

    with_flops: bool = Field(default=True)
    with_stack: bool = Field(default=True)
    with_modules: bool = Field(default=True)

    output_dir: str = Field(default="runtime_profiling")
    export_to_chrome: bool = Field(default=True)
    emit_nvtx: bool = Field(default=False)


class ProfilingConfig(BaseConfigSchema):
    enabled: bool = Field(default=False)
    frequency: int | None = Field(default=None)

    params: ProfilingParams = Field(default_factory=ProfilingParams)
