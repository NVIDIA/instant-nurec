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

from nre.models.base import BaseModel


class MockModel(BaseModel):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)


class TestBaseModel(unittest.TestCase):
    def test_base_class_instantiation(self):
        model = MockModel(config=DictConfig({}))
        self.assertIsInstance(model, BaseModel)
