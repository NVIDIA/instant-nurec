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

from omegaconf import DictConfig  # type: ignore[import-not-found]

from nre.config.model import GsplatStrategyConfig
from nre.models.gaussians.strategies.gsplat import GSplatStrategy
from nre.models.gaussians.strategies.test_utils import make_trainer_cfg
from nre.models.nn_extensions import TypedModuleDict


class TestGSplatStrategy(unittest.TestCase):
    def test_gsplat_strategy_instantiation(self):
        strategy = GSplatStrategy(
            config=GsplatStrategyConfig.model_validate({"name": "gsplat"}),
            trainer_config=make_trainer_cfg(),
            init_from_datasource=False,
            gaussians_nodes=TypedModuleDict(),
        )
        self.assertIsInstance(strategy, GSplatStrategy)
