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

"""Branch-coverage tests for the predict-only InstantNuRec public pydantic
config schemas.

Architecture-side configs (encoder/decoder/sky/activations) live under
``instant_nurec_internal.config_schema.models`` and are covered by
``internal/tests/test_config_models.py`` -- they are not part of the
shipped surface.
"""

from __future__ import annotations

import sys

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pydantic import ValidationError

from instant_nurec.config_schema.dataset import (
    AdaptiveSequentialFrameBatchSamplerConfig,
    CameraSubsamplerConfig,
    NCoreInstantNuRecCuboidTracksParamsConfig,
)
from instant_nurec.config_schema.instantnurec import GaussiansInstantNuRecSystemConfig, InstantNuRecConfig
from instant_nurec.config_schema.models import (
    KelvinModelConfig,
    PrimitiveExportPreprocessConfig,
)
from instant_nurec.config_schema.predict import PredictConfig, PrimitiveMergeConfig


# ---------------------------------------------------------------------------
# PrimitiveMergeConfig
# ---------------------------------------------------------------------------


def test_primitive_merge_default_disabled():
    cfg = PrimitiveMergeConfig()
    assert cfg.enabled is False
    assert cfg.frustum_ownership_max_diff_m == 5.0


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
# PrimitiveExportPreprocessConfig
# ---------------------------------------------------------------------------


def test_primitive_export_preprocess_default():
    cfg = PrimitiveExportPreprocessConfig()
    assert cfg.density_prune_threshold == 0.01


# ---------------------------------------------------------------------------
# KelvinModelConfig (slim, post-architecture-split)
# ---------------------------------------------------------------------------


def test_kelvin_model_config_default_only_carries_export_preprocess():
    cfg = KelvinModelConfig()
    assert isinstance(cfg.export_preprocess, PrimitiveExportPreprocessConfig)
    # No architecture fields -- those live in
    # instant_nurec_internal.config_schema.models.KelvinFullModelConfig.
    assert not hasattr(cfg, "encoder")
    assert not hasattr(cfg, "decoder")
    assert not hasattr(cfg, "sky")
    assert not hasattr(cfg, "scene_rescale")
    assert not hasattr(cfg, "track_padding_m")


# ---------------------------------------------------------------------------
# NCoreInstantNuRecCuboidTracksParamsConfig
# ---------------------------------------------------------------------------


def test_cuboid_tracks_params_rejects_negative_travel_distance():
    with pytest.raises(ValidationError):
        NCoreInstantNuRecCuboidTracksParamsConfig(
            lidar_id="lidar_top",
            track_min_travel_distance_m=-1.0,
            track_min_centroid_rig_dist_m=0.5,
            track_label_source="AUTOLABEL",
        )


def test_cuboid_tracks_params_rejects_negative_centroid_dist():
    with pytest.raises(ValidationError):
        NCoreInstantNuRecCuboidTracksParamsConfig(
            lidar_id="lidar_top",
            track_min_travel_distance_m=0.5,
            track_min_centroid_rig_dist_m=-0.1,
            track_label_source="AUTOLABEL",
        )


def test_cuboid_tracks_params_rejects_invalid_label_source():
    with pytest.raises(ValidationError):
        NCoreInstantNuRecCuboidTracksParamsConfig(
            lidar_id="lidar_top",
            track_min_travel_distance_m=0.5,
            track_min_centroid_rig_dist_m=0.5,
            track_label_source="MADE_UP_SOURCE",  # type: ignore[arg-type]
        )


def test_cuboid_tracks_params_default_extrapolate_us():
    cfg = NCoreInstantNuRecCuboidTracksParamsConfig(
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
# GaussiansInstantNuRecSystemConfig
# ---------------------------------------------------------------------------


def test_system_config_defaults():
    cfg = GaussiansInstantNuRecSystemConfig()
    assert cfg.predict_num_workers == 4
    assert cfg.predict_batch_size == 8


# ---------------------------------------------------------------------------
# BaseConfigSchema.__hash__
# ---------------------------------------------------------------------------


def test_base_config_schema_is_hashable():
    """The custom __hash__ override (vs PydanticBaseModel's hash-by-identity)
    enables instances to be used as dict keys / set members."""
    cfg1 = PrimitiveMergeConfig(enabled=False)
    cfg2 = PrimitiveMergeConfig(enabled=False)
    cfg3 = PrimitiveMergeConfig(enabled=True)
    assert hash(cfg1) == hash(cfg2)
    assert hash(cfg1) != hash(cfg3)
    assert len({cfg1, cfg2, cfg3}) == 2


# ---------------------------------------------------------------------------
# InstantNuRecConfig.model_post_init
# ---------------------------------------------------------------------------


def _make_config_kwargs(out_dir, **extra):
    base = dict(
        out_dir=str(out_dir),
        system=GaussiansInstantNuRecSystemConfig(),
        dataset={"predict": None},
        model=KelvinModelConfig(),
    )
    base.update(extra)
    return base


def test_config_post_init_no_env(tmp_path, monkeypatch):
    monkeypatch.delenv("INSTANT_NUREC_RUN_ID", raising=False)
    cfg = InstantNuRecConfig(**_make_config_kwargs(tmp_path))
    assert cfg.run_id  # auto-generated shortuuid


def test_config_post_init_env_run_id_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("INSTANT_NUREC_RUN_ID", "fixed-run-123")
    cfg = InstantNuRecConfig(**_make_config_kwargs(tmp_path))
    assert cfg.run_id == "fixed-run-123"
