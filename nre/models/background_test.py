# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import tempfile

import torch

from omegaconf import DictConfig

from nre.config.model import SkyEnvMapBackgroundConfig
from nre.config.trainer import TrainerConfig
from nre.models.background import SkyEnvMapBackground


class MockTrainerConfig(TrainerConfig):
    def __init__(self):
        super().__init__(
            max_epochs=1,
            check_val_every_n_epoch=1,
            precision="32",
            log_every_n_steps=1,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
        )


def test_skyenvmapbackground_checkpoint_save_and_load() -> None:
    config = DictConfig(
        {
            "name": "sky-env-map",
            "envmap_type": "cubemap",
            "width": 512,
            "height": 512,
            "composite_in_linear_space": False,
            "min_grad_updates": 1000,
            "should_inpaint": False,
            "inpaint_threshold": 5e-2,
            "inpaint_kernel_size": 10,
        }
    )
    trainer_config = MockTrainerConfig()
    background = SkyEnvMapBackground(SkyEnvMapBackgroundConfig.model_validate(config), trainer_config)
    TARGET_N_GRAD_UPDATES = 42
    background.n_grad_updates = TARGET_N_GRAD_UPDATES
    background.to("cuda")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")
        torch.save(background.state_dict(), checkpoint_path)
        background_loaded = SkyEnvMapBackground(SkyEnvMapBackgroundConfig.model_validate(config), trainer_config)
        assert background_loaded.n_grad_updates == 1

        background_loaded.load_state_dict(torch.load(checkpoint_path))
        assert background_loaded.n_grad_updates == TARGET_N_GRAD_UPDATES
