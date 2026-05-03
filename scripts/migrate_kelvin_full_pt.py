# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One-off migration: re-pickle ``kelvin_full.pt`` after the NRM → InstantNuRec rename.

The pickled system was saved with the old fully-qualified class names
(``instant_nurec.model.system.GaussiansNRMSystem`` etc.). After the rename
those qualnames no longer resolve, so plain ``torch.load`` on the old
artifact fails. This script registers temporary aliases under the old
paths, loads the old ``.pt``, and writes a fresh copy whose qualnames
match the new class hierarchy. Delete this script once everyone has
migrated.

Usage::

    python scripts/migrate_kelvin_full_pt.py /path/to/old_kelvin_full.pt /path/to/new_kelvin_full.pt
"""

from __future__ import annotations

import argparse
import sys

import torch

import instant_nurec.config_schema.dataset as _new_dataset
import instant_nurec.config_schema.instantnurec as _new_schema
import instant_nurec.datasets.datamodule as _new_dm
import instant_nurec.datasets.instantnurec_base as _new_ds_base
import instant_nurec.datasets.instantnurec_ncore as _new_ds_ncore
import instant_nurec.model.kelvin as _new_kelvin
import instant_nurec.model.system as _new_system
import instant_nurec.primitives.base as _new_base
import instant_nurec.primitives.kelvin_primitive as _new_kp
import instant_nurec.utils.batch as _new_batch


def _install_legacy_aliases() -> None:
    """Map the old qualnames onto the new classes so pickle can resolve."""

    sys.modules["instant_nurec.config_schema.nrm"] = _new_schema
    sys.modules["instant_nurec.datasets.nrm_base"] = _new_ds_base
    sys.modules["instant_nurec.datasets.nrm_ncore"] = _new_ds_ncore

    _new_schema.GaussiansNRMSystemConfig = _new_schema.GaussiansInstantNuRecSystemConfig
    _new_schema.NRMConfig = _new_schema.InstantNuRecConfig

    _new_dataset.InstantNuRecSplitsConfig = _new_dataset.InstantNuRecSplitsConfig
    _new_dataset.NRMSplitsConfig = _new_dataset.InstantNuRecSplitsConfig
    _new_dataset.NCoreNRMDatasetConfig = _new_dataset.NCoreInstantNuRecDatasetConfig
    _new_dataset.NCoreNRMCuboidTracksParamsConfig = _new_dataset.NCoreInstantNuRecCuboidTracksParamsConfig

    _new_system.GaussiansNRMSystem = _new_system.GaussiansInstantNuRecSystem
    _new_kelvin.KelvinNRM = _new_kelvin.KelvinInstantNuRec
    _new_kp.KelvinNRMPrimitive = _new_kp.KelvinInstantNuRecPrimitive
    _new_base.BaseNRMPrimitive = _new_base.BaseInstantNuRecPrimitive
    _new_dm.NRMDataModule = _new_dm.InstantNuRecDataModule
    _new_ds_ncore.NCoreNRMDataset = _new_ds_ncore.NCoreInstantNuRecDataset
    _new_ds_base.NRMDataError = _new_ds_base.InstantNuRecDataError
    _new_batch.NRMDataBatch = _new_batch.InstantNuRecDataBatch


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("old_pt", help="Existing kelvin_full.pt with NRM qualnames.")
    p.add_argument("new_pt", help="Destination path for the migrated artifact.")
    args = p.parse_args()

    _install_legacy_aliases()

    print(f"Loading old artifact from {args.old_pt} ...", flush=True)
    system = torch.load(args.old_pt, map_location="cpu", weights_only=False)
    print(f"Loaded type: {type(system).__module__}.{type(system).__name__}", flush=True)

    print(f"Saving migrated artifact to {args.new_pt} ...", flush=True)
    torch.save(system, args.new_pt)
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
