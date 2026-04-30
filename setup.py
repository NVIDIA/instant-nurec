# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import importlib.util
import os

from setuptools import find_packages, setup


def get_version() -> str:
    # Import nre.config.version without loading the entire package.
    spec = importlib.util.spec_from_file_location("version", os.path.join("nre", "config", "version.py"))
    version = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(version)
    return version.get_version().semantic_string()


packages = find_packages()
print("Packages: ", packages)
setup(name="nre", version=get_version(), packages=packages)
