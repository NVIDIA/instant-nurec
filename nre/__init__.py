# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Keep this file, otherwise find_packages() in setup.py will not find this package and its sub-packages,
# and they do not get installed with 'pip install .' (as copies) or 'pip install -e .' (as references
# to the source tree). This will lead to import errors outside of the repo root.
# In turn, find_namespace_packages() can be used in setup.py but it is very sensitive and would
# need too many exclusions to achieve the desired result.

import logging
import os

from nre.utils.colored_exceptions import enable_colored_exceptions
from nre.utils.misc import is_env_true


log_level = os.environ.get("LOGLEVEL", "INFO")
level = getattr(logging, log_level)
logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s", level=level)

if is_env_true("NRE_ENABLE_COLORED_EXCEPTIONS", True):
    enable_colored_exceptions()
