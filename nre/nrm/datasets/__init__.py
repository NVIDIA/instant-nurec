# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from nre.nrm.datasets.datamodule import NRMDataModule
from nre.nrm.datasets.nrm_base import (
    BaseNRMDataset,
    BaseNRMIndexableDataset,
    BaseNRMIterableDataset,
    DummyNRMDataset,
    MixedNRMDataset,
)
from nre.nrm.datasets.nrm_dataverse import DataverseNRMDataset
from nre.nrm.datasets.nrm_ncore import NCoreNRMDataset
from nre.nrm.datasets.nrm_ncore_websocket import WebSocketNCoreNRMDataset
from nre.nrm.datasets.registry import make, register


# These imports in __all__ are only used for documentation and shouldn't
# be used for relative imports. This is a temporary solution until
# we can make the autodiscovery of the modules work with sphinx
__all__ = [
    "NRMDataModule",
    "BaseNRMDataset",
    "BaseNRMIndexableDataset",
    "BaseNRMIterableDataset",
    "DataverseNRMDataset",
    "WebSocketNCoreNRMDataset",
    "NCoreNRMDataset",
    "DummyNRMDataset",
    "MixedNRMDataset",
]
