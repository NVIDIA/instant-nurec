# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import libs.losses.models.lambda_schedulers.lambda_schedulers as lambda_schedulers

from libs.losses.models.lambda_schedulers.lambda_schedulers import BaseLambdaScheduler, LinearLambdaScheduler
from libs.losses.models.lambda_schedulers.registry import make


# These imports in __all__ are only used for documentation and shouldn't
# be used for relative imports. This is a temporary solution until
# we can make the autodiscovery of the modules work with sphinx
__all__ = [
    "BaseLambdaScheduler",
    "LinearLambdaScheduler",
]
