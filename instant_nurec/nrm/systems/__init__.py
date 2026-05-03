# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import os

from typing import TYPE_CHECKING, Optional

import torch

from instant_nurec import _hf_mock
from instant_nurec.nrm.systems.gaussians_nrm import GaussiansNRMSystem


if TYPE_CHECKING:
    from instant_nurec.config_schema.nrm import NRMConfig


logger = logging.getLogger(__name__)


def _resolve_full_pt_path() -> Optional[str]:
    """Try the HF mock first (Phase 4 step 9 wire-up); fall back to the
    ``INSTANT_NUREC_FULL_PT`` env var if the mock can't satisfy the request.

    The mock's ``get_full_model_path`` already consumes
    ``INSTANT_NUREC_FULL_PT`` internally to seed the cache, so this preserves
    the previous behaviour while routing the lookup through the HF surface
    that the future-corp-published repo will hit.
    """
    try:
        return _hf_mock.get_full_model_path()
    except _hf_mock.HFMockError:
        env_path = os.environ.get("INSTANT_NUREC_FULL_PT")
        return env_path if env_path else None


def make(config: "NRMConfig", load_from_checkpoint: Optional[str] = None) -> GaussiansNRMSystem:
    """Predict-only standalone: NRE supported both Lightning checkpoint loading
    and weights-only loading; only the latter survives (the predict config
    always set resume_weights_only=true).

    Phase 1 step 5 + Phase 4 step 9: when the HF mock can resolve
    ``kelvin_full.pt`` (transitively from ``INSTANT_NUREC_FULL_PT`` or a
    pre-populated cache), skip construction + state_dict-load and torch.load
    the entire system. Otherwise, follow the build-from-config path and (if
    the env var is set but the file is missing) write the system to that
    path so the next run can use the fast path. This preserves parity
    bit-for-bit while exposing the load shortcut step 5 relies on.
    """
    full_pt_path = _resolve_full_pt_path()

    if full_pt_path and os.path.exists(full_pt_path):
        logger.info("Loading full system from %s (bypassing construct+state_dict path).", full_pt_path)
        loaded = torch.load(full_pt_path, map_location="cpu", weights_only=False)
        assert isinstance(loaded, GaussiansNRMSystem), (
            f"Expected GaussiansNRMSystem from {full_pt_path}, got {type(loaded).__name__}"
        )
        # Override the saved config-derived attributes with the new config: the
        # weight tensors are the only thing that needs to roundtrip via the .pt;
        # out_dir / run_id / merge flag / ncore path all change per invocation.
        from instant_nurec.datasets.datamodule import NRMDataModule  # local to avoid bootstrap loops

        loaded.out_dir = config.out_dir
        loaded.run_id = config.run_id
        loaded.config = config.system
        loaded.predict_config = config.predict
        loaded.export_preprocess = config.model.export_preprocess
        loaded.datamodule = NRMDataModule(config)
        return loaded

    system = GaussiansNRMSystem(config)
    if load_from_checkpoint is None:
        return system

    checkpoint = torch.load(load_from_checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    system.load_state_dict(state_dict, strict=True)

    write_path = os.environ.get("INSTANT_NUREC_FULL_PT")
    if write_path:
        logger.info("Writing full system to %s for the next run's fast path.", write_path)
        os.makedirs(os.path.dirname(write_path) or ".", exist_ok=True)
        torch.save(system, write_path)

    return system
