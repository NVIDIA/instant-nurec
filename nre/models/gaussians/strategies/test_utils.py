# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

from nre.utils.trainer import TrainerConfig


def make_trainer_cfg() -> TrainerConfig:
    return TrainerConfig(
        max_epochs=1,
        check_val_every_n_epoch=1,
        precision="32",
        log_every_n_steps=1,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
