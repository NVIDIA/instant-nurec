"""Hydra-free NRMConfig loader.

Replaces the multi-step Hydra composition (`parse_pretrained_nrm_config` →
`parse_untyped_nrm_config` → `parse_typed_nrm_config`) with a single function
that:

  1. Reads the static portion of the config from `predict_config.yaml`.
  2. Resolves the pretrained checkpoint path via `create_model_registry`
     (same NGC registry NRE used; the cache lives at
     ``~/.cache/nrm/pretrained_models/kelvin_pa_front``).
  3. Injects the CLI-derived fields (output dir, ncore paths, merge toggle).
  4. Validates against `NRMConfig` and returns the typed instance.

`predict_config.yaml` is read once from the package directory; we never round-
trip through Hydra/OmegaConf, so the only YAML dep is `pyyaml`.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from nre.nrm.config.nrm import NRMConfig
from nre.utils.model_registry import create_model_registry


_PRETRAINED_MODEL_URL = (
    "https://api.ngc.nvidia.com/v2/org/nvstaging/team/nre/models/"
    "nrm-kelvin-pa/versions/1.0.0-front/files/nrm-kelvin-pa_1.0.0-front.ckpt"
)
_PRETRAINED_CACHE_DIR = Path(
    os.path.expandvars("$HOME/.cache/nrm/pretrained_models/kelvin_pa_front")
)
_PREDICT_CONFIG_PATH = Path(__file__).resolve().parent / "predict_config.yaml"


def _resolve_pretrained_checkpoint() -> str:
    """Return the absolute path to the pretrained kelvin_pa_front checkpoint.

    Downloads it to the local cache if it isn't there already (first-run
    only); subsequent calls just return the cached path. Mirrors what
    `parse_pretrained_nrm_config` does in NRE.
    """
    return create_model_registry(
        _PRETRAINED_MODEL_URL, _PRETRAINED_CACHE_DIR
    ).get_model()


def load_predict_config(
    *,
    ncore_path: Path,
    output_dir: Path,
    merge_enabled: bool,
) -> NRMConfig:
    """Load the static config and patch in CLI-derived fields.

    Args:
        ncore_path: ncorev4 dataset root (must contain ``debug.lst``).
        output_dir: directory for PLY output and parsed config.
        merge_enabled: whether to enable primitive merging
            (``--merge frustum-ownership`` vs ``--merge none``).

    Returns:
        Validated :class:`NRMConfig`.
    """
    with _PREDICT_CONFIG_PATH.open() as f:
        cfg: dict = yaml.safe_load(f)

    cfg["resume"] = _resolve_pretrained_checkpoint()
    cfg["out_dir"] = str(output_dir)
    cfg["dataset"]["predict"]["ncore_json_base_path"] = str(ncore_path)
    cfg["dataset"]["predict"]["ncore_json_list_path"] = str(ncore_path / "debug.lst")
    cfg["predict"]["primitive_merge"]["enabled"] = bool(merge_enabled)

    return NRMConfig.model_validate(cfg)
