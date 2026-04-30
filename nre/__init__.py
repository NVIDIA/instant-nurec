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

import builtins
import logging
import os

from importlib import import_module

from nre.config.version import get_version
from nre.repo_root import __reporoot__
from nre.utils.colored_exceptions import enable_colored_exceptions
from nre.utils.debug.remote_debug import breakpoint_env
from nre.utils.misc import is_env_true


# Conditionally activate remote debugging based on environment variables
breakpoint_env()

# Note: version information is not available *intentionally* in some environments
# (like build / test sandboxes) to prevent build / test cache invalidations
__version__ = get_version()

log_level = os.environ.get("LOGLEVEL", "INFO")
level = getattr(logging, log_level)

# __PYCENA__ is set in builtins by the pycena C runtime before modules are imported.
# When running under pycena, module names are obfuscated, so we omit them from logs.
if getattr(builtins, "__PYCENA__", False):
    log_format = "[%(asctime)s][%(levelname)s] %(message)s"
else:
    log_format = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
logging.basicConfig(format=log_format, level=level)

# Enable colored exception handling if not explicitly disabled
if is_env_true("NRE_ENABLE_COLORED_EXCEPTIONS", True):
    enable_colored_exceptions()

logger = logging.getLogger(__name__)
# Try to import internal module if available
# This should be the only mention of internal modules in the non-internal code
try:
    import_module("nre.internal")  # noqa: F403
    logger.debug("NRE internals loaded")
except ImportError:
    logger.debug("NRE internals not present")
    pass
