"""Hydra-free, YAML-free NRMConfig loader.

Replaces the multi-step Hydra composition (`parse_pretrained_nrm_config` →
`parse_untyped_nrm_config` → `parse_typed_nrm_config`) with a single function
that:

  1. Holds the static portion of the config inline (sourced 1:1 from the
     hydra-produced ``parsed.yaml`` for the kelvin_pa_front pretrained
     predict run; Phase 1 step 4.4 dropped the YAML round-trip).
  2. Resolves the pretrained checkpoint path via ``create_model_registry``
     (same NGC registry NRE used; the cache lives at
     ``~/.cache/nrm/pretrained_models/kelvin_pa_front``).
  3. Injects the CLI-derived fields (output dir, ncore paths, merge toggle).
  4. Validates against ``NRMConfig`` and returns the typed instance.

No ``hydra-core``/``omegaconf``/``pyyaml`` imports.
"""

from __future__ import annotations

import os
from pathlib import Path

from instant_nurec._pkg.nrm.config.nrm import NRMConfig
from instant_nurec._pkg.utils.model_registry import create_model_registry


_PRETRAINED_MODEL_URL = (
    "https://api.ngc.nvidia.com/v2/org/nvstaging/team/nre/models/"
    "nrm-kelvin-pa/versions/1.0.0-front/files/nrm-kelvin-pa_1.0.0-front.ckpt"
)
_PRETRAINED_CACHE_DIR = Path(
    os.path.expandvars("$HOME/.cache/nrm/pretrained_models/kelvin_pa_front")
)


# Static portion of the NRMConfig consumed by the standalone Kelvin predict
# pipeline. CLI-driven fields are injected at load time:
#   - resume                                  (cached pretrained checkpoint path)
#   - out_dir                                 (--output-dir)
#   - dataset.predict.ncore_json_list_path    (--ncore-path/debug.lst)
#   - dataset.predict.ncore_json_base_path    (--ncore-path)
#   - predict.primitive_merge.enabled         (--merge)
_PREDICT_CONFIG: dict = {
    "seed": 38,
    "system": {
        "predict_num_workers": 4,
        "predict_batch_size": 8,
    },
    "dataset": {
        "predict": {
            "open_consolidated": True,
            "camera_max_fov_deg": 190.0,
            "n_camera_mask_dilation_iterations": 10,
            "camera_subsampler": {
                "frame_width": 784,
                "frame_height": 448,
            },
            "context_camera_ids": [
                "camera_front_wide_120fov",
            ],
            "frame_batch_sampler": {
                "n_frames_per_sample": 18,
                "n_samples_per_sequence": 8,
                "max_frame_gap_timestamp_us": 750000,
            },
            "supervision_camera_ids": [
                "camera_front_wide_120fov",
            ],
            "cuboid_tracks_params": {
                "lidar_id": "lidar_top_360fov",
                "track_min_travel_distance_m": 1.5,
                "track_min_centroid_rig_dist_m": 3.0,
                "track_extrapolate_timestamps_us": 1000000,
                "track_label_source": "AUTOLABEL",
            },
        },
    },
    "model": {
        "track_padding_m": [1.0, 1.0, 1.0],
        "scene_rescale": 0.15,
        "sky": {
            "cubemap_size": 448,
            "embed_dim": 384,
            "depth": 1,
            "checkpointing": True,
        },
        "patch_shape": [14, 14],
        "encoder": {
            "depth": 12,
            "n_heads": 12,
            "embed_dim": 1536,
            "take_block_indices": [5, 7, 9, 11],
            "aa_start_block_idx": 4,
            "checkpointing": "all",
        },
        "decoder": {
            "dpt_dim": 128,
            "dpt_reassemble_hidden_dims": [96, 192, 384, 768],
            "checkpointing": True,
            "dpt_chunk_size": 4,
            "time_encoding_dim": 256,
            "motion_depth": 4,
        },
        "activations": {
            "opacity_shift": -2.0,
            "scale_shift_log_ratio": -2.9,
            "scale_max": 0.045,
            "scale_min": 0.0,
        },
        "export_preprocess": {
            "density_prune_threshold": 0.01,
        },
    },
    "predict": {
        "chunk_size": 1,
        "primitive_merge": {
            "frustum_ownership_max_diff_m": 5.0,
        },
    },
}


def _resolve_pretrained_checkpoint() -> str:
    """Return the absolute path to the pretrained kelvin_pa_front checkpoint.

    Downloads it to the local cache if it isn't there already (first-run
    only); subsequent calls just return the cached path. Mirrors what
    ``parse_pretrained_nrm_config`` does in NRE.
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
    """Build the static config and patch in CLI-derived fields.

    Args:
        ncore_path: ncorev4 dataset root (must contain ``debug.lst``).
        output_dir: directory for PLY output and parsed config.
        merge_enabled: whether to enable primitive merging
            (``--merge frustum-ownership`` vs ``--merge none``).

    Returns:
        Validated :class:`NRMConfig`.
    """
    cfg: dict = _deep_copy_predict_config()

    cfg["resume"] = _resolve_pretrained_checkpoint()
    cfg["out_dir"] = str(output_dir)
    cfg["dataset"]["predict"]["ncore_json_base_path"] = str(ncore_path)
    cfg["dataset"]["predict"]["ncore_json_list_path"] = str(ncore_path / "debug.lst")
    cfg["predict"]["primitive_merge"]["enabled"] = bool(merge_enabled)

    return NRMConfig.model_validate(cfg)


def _deep_copy_predict_config() -> dict:
    """Return a fresh deep copy of ``_PREDICT_CONFIG`` so callers can mutate
    the CLI-driven fields without polluting the module-level template."""
    import copy

    return copy.deepcopy(_PREDICT_CONFIG)
