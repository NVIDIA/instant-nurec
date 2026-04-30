# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainValConfig:
    name: str
    config: Path


sqa_test_configs = {
    "car2sim": TrainValConfig(name="car2sim", config=Path("configs/apps/prod/Hyperion-8.1/car2sim.yaml")),
    "car2sim_gsplat": TrainValConfig(
        name="car2sim_gsplat", config=Path("configs/apps/prod/Hyperion-8.1/car2sim_gsplat.yaml")
    ),
    "car2sim_lidarfree": TrainValConfig(
        name="car2sim_lidarfree",
        config=Path("configs/apps/prod/Hyperion-8.1/car2sim_lidarfree.yaml"),
    ),
    "car2sim_lidarfree_gsplat": TrainValConfig(
        name="car2sim_lidarfree_gsplat",
        config=Path("configs/apps/prod/Hyperion-8.1/car2sim_lidarfree_gsplat.yaml"),
    ),
    "hyperion8.1_sqa_default": TrainValConfig(
        name="hyperion8.1_sqa_default", config=Path("configs/apps/prod/Hyperion-8.1/sqa_default.yaml")
    ),
    "hyperion8.1_sqa_default_gsplat": TrainValConfig(
        name="hyperion8.1_sqa_default_gsplat", config=Path("configs/apps/prod/Hyperion-8.1/sqa_default_gsplat.yaml")
    ),
    "hyperion8.1_sqa_lidar_default": TrainValConfig(
        name="hyperion8.1_sqa_lidar_default", config=Path("configs/apps/prod/Hyperion-8.1/sqa_lidar_default.yaml")
    ),
    "hyperion8.1_sqa_difix_distill": TrainValConfig(
        name="hyperion8.1_sqa_difix_distill", config=Path("configs/apps/prod/Hyperion-8.1/sqa_difix_distill.yaml")
    ),
    "hyperion8.1_sqa_difix_inference": TrainValConfig(
        name="hyperion8.1_sqa_difix_inference", config=Path("configs/apps/prod/Hyperion-8.1/sqa_difix_inference.yaml")
    ),
    "hyperion8.1_sqa_difix_distill_and_inference": TrainValConfig(
        name="hyperion8.1_sqa_difix_distill_and_inference",
        config=Path("configs/apps/prod/Hyperion-8.1/sqa_difix_distill_and_inference.yaml"),
    ),
    "waymo_sqa_default": TrainValConfig(
        name="waymo_sqa_default", config=Path("configs/apps/prod/Waymo/sqa_default.yaml")
    ),
}
