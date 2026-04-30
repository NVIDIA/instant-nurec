# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import unittest

from omegaconf import DictConfig

from nre.config.model import GsplatStrategyConfig
from nre.models.gaussians.strategies.base import BaseGaussianStrategy
from nre.models.gaussians.strategies.test_utils import make_trainer_cfg
from nre.models.nn_extensions import TypedModuleDict


class MockGaussianStrategy(BaseGaussianStrategy):
    def update_step_train_batch_end(
        self,
        epoch: int,
        global_step: int,
        batch,
        system,
        gaussians_nodes,
        **kwargs,
    ) -> None:
        pass


class TestBaseGaussianStrategy(unittest.TestCase):
    def test_base_class_instantiation(self):
        # Use minimal gsplat config
        config = GsplatStrategyConfig.model_validate(
            {
                "name": "gsplat",
            }
        )

        strategy = MockGaussianStrategy(
            config=config,
            trainer_config=make_trainer_cfg(),
            init_from_datasource=False,
            gaussians_nodes=TypedModuleDict(),
        )
        self.assertIsInstance(strategy, BaseGaussianStrategy)
