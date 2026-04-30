# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Layer 3: Orchestration - Configuration-driven loss coordination.

Empty __init__.py to avoid circular dependencies during module imports.
Import from submodules directly:
- libs.losses.orchestration.config          (LossItemConfig, LossConfig, LossReturn, LossAggregatorReturn, etc.)
- libs.losses.orchestration.loss_aggregator (LossAggregator)
"""
