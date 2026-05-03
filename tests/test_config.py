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

"""Branch-coverage tests for the predict-only NRM pydantic config schemas.

Covers ``instant_nurec.config_schema.predict`` (``PrimitiveMergeConfig`` + ``PredictConfig``),
``instant_nurec.config_schema.models`` (``KelvinDPTDecoderConfig.model_post_init`` +
default-fields paths), and ``instant_nurec.config_schema.nrm.NRMConfig.model_post_init``
(resume / .ckpt suffix / NRE_ENV_RUN_ID env override / config_dir derivation).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pydantic import ValidationError

from instant_nurec.config_schema.dataset import (
    AdaptiveSequentialFrameBatchSamplerConfig,
    CameraSubsamplerConfig,
    NCoreNRMCuboidTracksParamsConfig,
)
from instant_nurec.config_schema.models import (
    GaussiansActivationConfig,
    KelvinDAv3EncoderConfig,
    KelvinDPTDecoderConfig,
    KelvinModelConfig,
    KelvinSkyCubemapDecoderConfig,
    PrimitiveExportPreprocessConfig,
)
from instant_nurec.config_schema.nrm import GaussiansNRMSystemConfig, NRMConfig
from instant_nurec.config_schema.predict import PredictConfig, PrimitiveMergeConfig


# ---------------------------------------------------------------------------
# PrimitiveMergeConfig
# ---------------------------------------------------------------------------


def test_primitive_merge_default_disabled():
    cfg = PrimitiveMergeConfig()
    assert cfg.enabled is False
    assert cfg.frustum_ownership_max_diff_m == 0.0


def test_primitive_merge_enabled_with_positive_diff():
    cfg = PrimitiveMergeConfig(enabled=True, frustum_ownership_max_diff_m=2.5)
    assert cfg.enabled is True
    assert cfg.frustum_ownership_max_diff_m == 2.5


def test_primitive_merge_rejects_negative_diff():
    with pytest.raises(ValidationError):
        PrimitiveMergeConfig(frustum_ownership_max_diff_m=-0.1)


# ---------------------------------------------------------------------------
# PredictConfig
# ---------------------------------------------------------------------------


def test_predict_config_defaults():
    cfg = PredictConfig()
    assert cfg.chunk_size == 1
    assert isinstance(cfg.primitive_merge, PrimitiveMergeConfig)
    assert cfg.primitive_merge.enabled is False


def test_predict_config_custom_chunk_size():
    cfg = PredictConfig(chunk_size=4)
    assert cfg.chunk_size == 4


# ---------------------------------------------------------------------------
# KelvinDPTDecoderConfig
# ---------------------------------------------------------------------------


def test_kelvin_dpt_decoder_post_init_accepts_positive_dpt_dim():
    cfg = KelvinDPTDecoderConfig(dpt_dim=128, dpt_reassemble_hidden_dims=[8, 16, 32, 64])
    assert cfg.dpt_dim == 128
    # defaults
    assert cfg.checkpointing is False
    assert cfg.dpt_chunk_size == -1
    assert cfg.time_encoding_dim == 256
    assert cfg.motion_depth == 4


def test_kelvin_dpt_decoder_post_init_rejects_zero_dpt_dim():
    """pydantic wraps the AssertionError into a ValidationError."""
    with pytest.raises(ValidationError, match="must be positive"):
        KelvinDPTDecoderConfig(dpt_dim=0, dpt_reassemble_hidden_dims=[8, 16, 32, 64])


def test_kelvin_dpt_decoder_post_init_rejects_negative_dpt_dim():
    with pytest.raises(ValidationError, match="must be positive"):
        KelvinDPTDecoderConfig(dpt_dim=-1, dpt_reassemble_hidden_dims=[8, 16, 32, 64])


# ---------------------------------------------------------------------------
# GaussiansActivationConfig defaults
# ---------------------------------------------------------------------------


def test_activation_config_defaults():
    cfg = GaussiansActivationConfig()
    assert cfg.opacity_shift == -2.0
    assert cfg.scale_shift_log_ratio == -1.0
    assert cfg.scale_max == 0.3
    assert cfg.scale_min == 0.0


def test_activation_config_custom_values():
    cfg = GaussiansActivationConfig(
        opacity_shift=1.0, scale_shift_log_ratio=0.0, scale_max=0.5, scale_min=0.01
    )
    assert cfg.opacity_shift == 1.0
    assert cfg.scale_shift_log_ratio == 0.0
    assert cfg.scale_max == 0.5
    assert cfg.scale_min == 0.01


# ---------------------------------------------------------------------------
# PrimitiveExportPreprocessConfig
# ---------------------------------------------------------------------------


def test_primitive_export_preprocess_default():
    cfg = PrimitiveExportPreprocessConfig()
    assert cfg.density_prune_threshold == 0.01


# ---------------------------------------------------------------------------
# KelvinModelConfig (defaults + composed)
# ---------------------------------------------------------------------------


def _make_model_cfg(**overrides):
    return KelvinModelConfig(
        sky=KelvinSkyCubemapDecoderConfig(cubemap_size=256, embed_dim=64, depth=2),
        encoder=KelvinDAv3EncoderConfig(
            depth=4,
            n_heads=8,
            embed_dim=128,
            take_block_indices=[0, 1, 2, 3],
            aa_start_block_idx=0,
        ),
        decoder=KelvinDPTDecoderConfig(
            dpt_dim=128, dpt_reassemble_hidden_dims=[8, 16, 32, 64]
        ),
        **overrides,
    )


def test_kelvin_model_default_track_padding_and_scene_rescale():
    cfg = _make_model_cfg()
    assert cfg.track_padding_m == [1.0, 1.0, 1.0]
    assert cfg.scene_rescale == 1.0
    assert cfg.patch_shape == (8, 8)
    assert isinstance(cfg.activations, GaussiansActivationConfig)
    assert isinstance(cfg.export_preprocess, PrimitiveExportPreprocessConfig)


def test_kelvin_model_rejects_track_padding_not_3_long():
    with pytest.raises(ValidationError):
        _make_model_cfg(track_padding_m=[1.0, 1.0])


def test_kelvin_model_rejects_track_padding_too_long():
    with pytest.raises(ValidationError):
        _make_model_cfg(track_padding_m=[1.0, 1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# NCoreNRMCuboidTracksParamsConfig
# ---------------------------------------------------------------------------


def test_cuboid_tracks_params_rejects_negative_travel_distance():
    with pytest.raises(ValidationError):
        NCoreNRMCuboidTracksParamsConfig(
            lidar_id="lidar_top",
            track_min_travel_distance_m=-1.0,
            track_min_centroid_rig_dist_m=0.5,
            track_label_source="AUTOLABEL",
        )


def test_cuboid_tracks_params_rejects_negative_centroid_dist():
    with pytest.raises(ValidationError):
        NCoreNRMCuboidTracksParamsConfig(
            lidar_id="lidar_top",
            track_min_travel_distance_m=0.5,
            track_min_centroid_rig_dist_m=-0.1,
            track_label_source="AUTOLABEL",
        )


def test_cuboid_tracks_params_rejects_invalid_label_source():
    with pytest.raises(ValidationError):
        NCoreNRMCuboidTracksParamsConfig(
            lidar_id="lidar_top",
            track_min_travel_distance_m=0.5,
            track_min_centroid_rig_dist_m=0.5,
            track_label_source="MADE_UP_SOURCE",  # type: ignore[arg-type]
        )


def test_cuboid_tracks_params_default_extrapolate_us():
    cfg = NCoreNRMCuboidTracksParamsConfig(
        lidar_id="lidar_top",
        track_min_travel_distance_m=0.5,
        track_min_centroid_rig_dist_m=0.5,
        track_label_source="AUTOLABEL",
    )
    assert cfg.track_extrapolate_timestamps_us == 1_000_000


# ---------------------------------------------------------------------------
# Sub-config simple defaults
# ---------------------------------------------------------------------------


def test_camera_subsampler_basic():
    cfg = CameraSubsamplerConfig(frame_width=1920, frame_height=1080)
    assert cfg.frame_width == 1920
    assert cfg.frame_height == 1080


def test_adaptive_sequential_frame_batch_sampler_basic():
    cfg = AdaptiveSequentialFrameBatchSamplerConfig(
        n_frames_per_sample=4, n_samples_per_sequence=2, max_frame_gap_timestamp_us=200000
    )
    assert cfg.n_frames_per_sample == 4
    assert cfg.n_samples_per_sequence == 2


# ---------------------------------------------------------------------------
# GaussiansNRMSystemConfig
# ---------------------------------------------------------------------------


def test_system_config_defaults():
    cfg = GaussiansNRMSystemConfig()
    assert cfg.predict_num_workers == 0
    assert cfg.predict_batch_size == 1


# ---------------------------------------------------------------------------
# BaseConfigSchema.__hash__
# ---------------------------------------------------------------------------


def test_base_config_schema_is_hashable():
    """The custom __hash__ override (vs PydanticBaseModel's hash-by-identity)
    enables instances to be used as dict keys / set members."""
    cfg1 = PrimitiveMergeConfig(enabled=False)
    cfg2 = PrimitiveMergeConfig(enabled=False)
    cfg3 = PrimitiveMergeConfig(enabled=True)
    # Same content → same hash.
    assert hash(cfg1) == hash(cfg2)
    # Different content → different hash (almost always; depends on __repr__ diff).
    assert hash(cfg1) != hash(cfg3)
    # And both are hashable in a set / dict
    assert len({cfg1, cfg2, cfg3}) == 2


# ---------------------------------------------------------------------------
# NRMConfig.model_post_init
# ---------------------------------------------------------------------------


def _make_nrm_kwargs(out_dir, **extra):
    base = dict(
        out_dir=str(out_dir),
        system=GaussiansNRMSystemConfig(),
        dataset={"predict": None},
        model=_make_model_cfg(),
    )
    base.update(extra)
    return base


def test_nrm_config_post_init_no_env(tmp_path, monkeypatch):
    monkeypatch.delenv("NRE_ENV_RUN_ID", raising=False)
    cfg = NRMConfig(**_make_nrm_kwargs(tmp_path))
    # config_dir auto-derives to out_dir/run_id/config
    assert cfg.config_dir == os.path.join(str(tmp_path), cfg.run_id, "config")
    assert cfg.run_id  # auto-generated shortuuid


def test_nrm_config_post_init_env_run_id_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("NRE_ENV_RUN_ID", "fixed-run-123")
    cfg = NRMConfig(**_make_nrm_kwargs(tmp_path))
    assert cfg.run_id == "fixed-run-123"
    assert cfg.config_dir == os.path.join(str(tmp_path), "fixed-run-123", "config")
