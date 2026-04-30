# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from pathlib import Path

import numpy as np
import pytest

from python.runfiles import runfiles
from pytorch_lightning import seed_everything

from nre.nrm.config.dataset import (
    CameraSubsamplerConfig,
    DataverseNRMDatasetConfig,
    SupervisionFrameBatchParamsConfig,
    VaryingIntervalFrameBatchSamplerConfig,
)
from nre.nrm.datasets.nrm_dataverse import DataverseNRMDataset
from nre.nrm.datasets.registry import make


RUNFILES = runfiles.Create()

SHARED_PARAMS = dict(
    name="nrm-dataverse",
    frame_batch_samplers={
        "varying_interval": VaryingIntervalFrameBatchSamplerConfig(
            n_frames_per_sample=3,
            n_samples_per_sequence=1,
            sequence_gap_timestamp_us_min=3,
            sequence_gap_timestamp_us_max=10,
        )
    },
    supervision_frame_batch=SupervisionFrameBatchParamsConfig(n_frames_per_sample=3, sample_strategy="random"),
    camera_subsampler=CameraSubsamplerConfig(
        frame_width=256,
        frame_height=224,
    ),
)


@pytest.fixture
def test_data_dir() -> Path:
    """
    Returns the root directory of the test data.
    RUNFILES.Rlocation cannot be used to search for directories so we use the BUILD.bazel file as a proxy.
    """
    return Path(RUNFILES.Rlocation("test_data_dataverse/BUILD.bazel")).parent


@pytest.fixture
def re10k_config(test_data_dir: Path) -> DataverseNRMDatasetConfig:
    return DataverseNRMDatasetConfig(
        subset_spec=DataverseNRMDatasetConfig.Re10kParams(
            target="re10k.RealEstate10K",
            root_path=str(test_data_dir / "re10k"),
            annotation_json="train_all.json",
        ),
        **SHARED_PARAMS,  # type: ignore
    )


@pytest.fixture
def mvimgnet_config(test_data_dir: Path) -> DataverseNRMDatasetConfig:
    return DataverseNRMDatasetConfig(
        subset_spec=DataverseNRMDatasetConfig.MVImgNetParams(
            target="mvimgnet.MVImgNet",
            root_path=str(test_data_dir / "mvimgnet"),
        ),
        **SHARED_PARAMS,  # type: ignore
    )


@pytest.fixture
def dl3dv_config(test_data_dir: Path) -> DataverseNRMDatasetConfig:
    return DataverseNRMDatasetConfig(
        subset_spec=DataverseNRMDatasetConfig.DL3DVParams(
            target="dl3dv.DL3DV10K",
            root_path=str(test_data_dir / "dl3dv"),
        ),
        **SHARED_PARAMS,  # type: ignore
    )


@pytest.fixture
def subset_config(request) -> DataverseNRMDatasetConfig:
    seed_everything(0)  # needed for any NRM dataset to work
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize(
    "subset_config",
    [
        "re10k_config",
        "mvimgnet_config",
        "dl3dv_config",
    ],
    indirect=True,
)
def test_smoke_get_batch(
    subset_config: DataverseNRMDatasetConfig,
) -> None:
    dataset = DataverseNRMDataset(subset_config)

    # use getitem_allow_exceptions to avoid the exception handling in BaseNRMIndexableDataset.__getitem__
    # which hides potential failures and retries, causing broken tests to hang instead of failing
    batch = dataset.getitem_allow_exceptions(0, np.random.default_rng(seed=0))

    expected_height = subset_config.camera_subsampler.frame_height
    expected_width = subset_config.camera_subsampler.frame_width

    assert batch.context[0].rendering is not None
    assert batch.context[0].rendering.camera is not None
    assert batch.context[0].rendering.camera.poses_tquat_startend.shape == (3, 2, 7)
    assert batch.context[0].rendering.camera.rays.shape == (3, expected_height, expected_width, 6)
    assert batch.context[0].data.camera is not None
    assert len(batch.context[0].data.camera.meta) == 3
    assert batch.context[0].data.camera.meta[0].unique_sensor_idx == 0
    assert batch.context[0].data.camera.labels.rgb is not None
    assert batch.context[0].data.camera.labels.rgb.shape == (3, expected_height, expected_width, 3)

    assert batch.supervision is not None
    assert batch.supervision[0].rendering is not None
    assert batch.supervision[0].rendering.camera is not None
    assert batch.supervision[0].rendering.camera.poses_tquat_startend.shape == (3, 2, 7)
    assert batch.supervision[0].rendering.camera.rays.shape == (3, expected_height, expected_width, 6)
    assert batch.supervision[0].data.camera is not None
    assert len(batch.supervision[0].data.camera.meta) == 3
    assert batch.supervision[0].data.camera.meta[0].unique_sensor_idx == 0
    assert batch.supervision[0].data.camera.labels.rgb is not None
    assert batch.supervision[0].data.camera.labels.rgb.shape == (3, expected_height, expected_width, 3)

    assert batch.context_rig is not None
    assert batch.supervision_rig is not None


def test_construct_from_registry(test_data_dir: Path) -> None:
    """
    Smoke test to make sure DataverseNRMDataset can be constructed from the registry.
    Using only a single dataset to avoid the overhead of loading multiple datasets.
    """

    config = DataverseNRMDatasetConfig(
        subset_spec=DataverseNRMDatasetConfig.Re10kParams(
            target="re10k.RealEstate10K",
            root_path=str(test_data_dir / "re10k"),
            annotation_json="train_all.json",
        ),
        **SHARED_PARAMS,  # type: ignore
    )

    dataset = make("nrm-dataverse", config, "train")
    assert isinstance(dataset, DataverseNRMDataset)
    assert dataset.config == config
