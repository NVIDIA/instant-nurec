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

from pathlib import Path


__reporoot__ = (
    Path(__file__).parent.parent.resolve()
    if not "BUILD_WORKING_DIRECTORY" in os.environ
    else Path(os.environ["BUILD_WORKING_DIRECTORY"])
)
