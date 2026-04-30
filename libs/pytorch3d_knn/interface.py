# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import importlib

from typing import Any

# Pre-load dynamic torch dependencies, otherwise runtime-lookup will fail for torch-specific .so's
import torch


pytorch3d_knn: Any = importlib.import_module("libs.pytorch3d_knn.libpytorch3d_knn_cc")
