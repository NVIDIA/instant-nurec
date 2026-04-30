# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import glob

import pytest

from omegaconf import DictConfig

from nre.nrm.config.dataset import NRMEpochSplitConfig
from nre.nrm.config.nrm import parse_typed_nrm_config


nrm_app_configs = glob.glob("configs/nrm/apps/**/[!_]*.yaml", recursive=True)

# Exclude any "options" subdirectories, as these contain config groups (not fully formed configs).
nrm_app_configs = [conf for conf in nrm_app_configs if "/options/" not in conf and "tmp" not in conf]


@pytest.mark.parametrize("config_name", nrm_app_configs)
def test_all_nrm_configs_are_valid(config_name: str) -> None:
    """
    A simple test which checks if all nrm configs are gramatically valid w.r.t. the config schema.
    It does not check if they actually lead to correct code execution.

    This test checks NRM (feedforward model) configs.
    """

    if config_name.startswith("configs/nrm/apps/pretrained/"):
        # Skip pretrained model configs as they require NGC authentication to download
        # the model config and don't follow the common NRM config schema directly
        pytest.skip("Skipping pretrained model config")

    hydra_args = [
        "out_dir=/does/not/matter",
    ]

    parse_typed_nrm_config(config_name=config_name, hydra_args=hydra_args)


def test_nrm_epoch_split_config_validation() -> None:
    """
    Test that NRMEpochSplitConfig properly validates epoch keys.
    """
    dataset_spec = DictConfig(
        {
            "name": "nrm-dataverse",
            "camera_subsampler": {
                "frame_width": 224,
                "frame_height": 224,
            },
            "frame_batch_samplers": {
                "my_sampler": {
                    "name": "varying_interval",
                    "n_samples_per_sequence": 1,
                    "n_frames_per_sample": 4,
                    "sequence_gap_timestamp_us_min": 8,
                    "sequence_gap_timestamp_us_max": 10,
                }
            },
            "supervision_frame_batch": {
                "n_frames_per_sample": 8,
            },
            "subset_spec": {
                "target": "re10k.RealEstate10K",
                "root_path": "/test/path",
                "annotation_json": "/test/train.json",
            },
        }
    )

    # Test valid epoch keys - should work
    valid_config = DictConfig(
        {
            "epoch_0": dataset_spec,
            "epoch_10": dataset_spec,
        }
    )

    # This should not raise an exception
    epoch_config = NRMEpochSplitConfig.model_validate(valid_config)
    assert epoch_config.root is not None

    # Test invalid epoch keys - no prefix
    invalid_config = DictConfig(
        {
            "epoch_0": dataset_spec,
            "invalid_key": dataset_spec,
        }
    )

    with pytest.raises(ValueError, match=r"Epoch split config keys must be 'epoch_{epoch_number}'"):
        NRMEpochSplitConfig.model_validate(invalid_config)

    # Test invalid epoch keys - non-numeric epoch number
    invalid_config = DictConfig(
        {
            "epoch_0": dataset_spec,
            "epoch_0_foo": dataset_spec,
        }
    )

    with pytest.raises(ValueError, match=r"Epoch split config keys must be 'epoch_{epoch_number}'"):
        NRMEpochSplitConfig.model_validate(invalid_config)
