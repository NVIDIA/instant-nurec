# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import sys


# HACK: Remove project root from sys.path if it exists
# torchrun adds the project root to the python path, which
# breaks importing our cuda python modules.
# The importing of cuda modules should be made more robust.
# Note: this assumes that the project root is two dir levels above.
project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))
while project_root in sys.path:
    sys.path.remove(project_root)

from nre.run.main import main
from nre.run.profile_dataloader import profile_dataloader
from nre.run.run_script import run_script
