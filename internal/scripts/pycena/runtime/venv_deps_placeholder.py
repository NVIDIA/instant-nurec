# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Placeholder py_venv_binary used to build venvs for obfuscated binaries.

This script is never executed. The py_venv_binary targets (pycena_run_venv_deps,
pycena_nre_tools_venv_deps) exist solely to:
1. Declare pip package dependencies
2. Have aspect_rules_py build a venv at build time

The entrypoint.sh script activates the pre-built venv for the obfuscated C++ binary.
"""

if __name__ == "__main__":
    print("This script is not meant to be called directly. See entrypoint.sh.")
