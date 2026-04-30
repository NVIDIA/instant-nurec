# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import nre.systems.gaussians as gaussians
import nre.systems.nrend_test_systems as nrend_test_systems

from nre.difix.training_controller import TrainingDifixController
from nre.systems.base import BaseSystem, BaseSystemSO
from nre.systems.gaussians import GaussiansSystem
from nre.systems.registry import make


# These imports in __all__ are only used for documentation and shouldn't
# be used for relative imports. This is a temporary solution until
# we can make the autodiscovery of the modules work with sphinx
__all__ = [
    "BaseSystem",
    "BaseSystemSO",
    "GaussiansSystem",
    "TrainingDifixController",
]
