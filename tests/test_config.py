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
config schemas."""

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
    assert cfg.enable_voxelization is False
    assert cfg.voxel_size == 0.1
    assert cfg.target_n_gaussians == 2_000_000
    assert cfg.max_voxelization_iterations == 20


def test_primitive_merge_enabled_with_positive_diff():
    cfg = PrimitiveMergeConfig(enabled=True, frustum_ownership_max_diff_m=2.5)
    assert cfg.enabled is True
    assert cfg.frustum_ownership_max_diff_m == 2.5


def test_primitive_merge_rejects_negative_diff():
    with pytest.raises(ValidationError):
        PrimitiveMergeConfig(frustum_ownership_max_diff_m=-0.1)


def test_primitive_merge_enable_voxelization_with_explicit_initial_size():
    """``voxel_size`` is now the initial value for the iterative search."""
    cfg = PrimitiveMergeConfig(enable_voxelization=True, voxel_size=0.25)
    assert cfg.enable_voxelization is True
    assert cfg.voxel_size == 0.25


def test_primitive_merge_rejects_zero_voxel_size():
    with pytest.raises(ValidationError):
        PrimitiveMergeConfig(voxel_size=0.0)


def test_primitive_merge_rejects_negative_voxel_size():
    with pytest.raises(ValidationError):
        PrimitiveMergeConfig(voxel_size=-0.1)


def test_primitive_merge_target_n_gaussians_explicit():
    cfg = PrimitiveMergeConfig(target_n_gaussians=500_000)
    assert cfg.target_n_gaussians == 500_000


def test_primitive_merge_rejects_zero_target_n_gaussians():
    with pytest.raises(ValidationError):
        PrimitiveMergeConfig(target_n_gaussians=0)


def test_primitive_merge_rejects_negative_target_n_gaussians():
    with pytest.raises(ValidationError):
        PrimitiveMergeConfig(target_n_gaussians=-1)


def test_primitive_merge_max_voxelization_iterations_explicit():
    cfg = PrimitiveMergeConfig(max_voxelization_iterations=5)
    assert cfg.max_voxelization_iterations == 5


def test_primitive_merge_rejects_zero_max_voxelization_iterations():
    with pytest.raises(ValidationError):
        PrimitiveMergeConfig(max_voxelization_iterations=0)


# ---------------------------------------------------------------------------
# PredictConfig
# ---------------------------------------------------------------------------


def test_predict_config_defaults():
    cfg = PredictConfig()
    assert isinstance(cfg.primitive_merge, PrimitiveMergeConfig)
    assert cfg.primitive_merge.enabled is False


# ---------------------------------------------------------------------------
# PrimitiveExportPreprocessConfig
# ---------------------------------------------------------------------------


def test_primitive_export_preprocess_default():
    cfg = PrimitiveExportPreprocessConfig()
    assert cfg.density_prune_threshold == 0.01


# ---------------------------------------------------------------------------
# KelvinModelConfig
# ---------------------------------------------------------------------------


def test_kelvin_model_config_exposes_architecture_defaults():
    cfg = KelvinModelConfig()
    assert isinstance(cfg.export_preprocess, PrimitiveExportPreprocessConfig)
    assert cfg.encoder.embed_dim == 1536
    assert cfg.encoder.take_block_indices == [5, 7, 9, 11]
    assert cfg.decoder.dpt_dim == 128
    assert cfg.sky.cubemap_size == 448
    assert cfg.scene_rescale == 0.15
    assert cfg.track_padding_m == [1.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# NCoreInstantNuRecCuboidTracksParamsConfig
# ---------------------------------------------------------------------------


def test_cuboid_tracks_params_rejects_negative_travel_distance():
    with pytest.raises(ValidationError):
        NCoreInstantNuRecCuboidTracksParamsConfig(
            track_min_travel_distance_m=-1.0,
            track_min_centroid_rig_dist_m=0.5,
            track_label_source="AUTOLABEL",
        )


def test_cuboid_tracks_params_rejects_negative_centroid_dist():
    with pytest.raises(ValidationError):
        NCoreInstantNuRecCuboidTracksParamsConfig(
            track_min_travel_distance_m=0.5,
            track_min_centroid_rig_dist_m=-0.1,
            track_label_source="AUTOLABEL",
        )


def test_cuboid_tracks_params_rejects_invalid_label_source():
    with pytest.raises(ValidationError):
        NCoreInstantNuRecCuboidTracksParamsConfig(
            track_min_travel_distance_m=0.5,
            track_min_centroid_rig_dist_m=0.5,
            track_label_source="MADE_UP_SOURCE",  # type: ignore[arg-type]
        )


def test_cuboid_tracks_params_default_extrapolate_us():
    cfg = NCoreInstantNuRecCuboidTracksParamsConfig(
        track_min_travel_distance_m=0.5,
        track_min_centroid_rig_dist_m=0.5,
        track_label_source="AUTOLABEL",
    )
    assert cfg.track_extrapolate_timestamps_us == 1_000_000


# ---------------------------------------------------------------------------
# Sub-config simple defaults
# ---------------------------------------------------------------------------


def test_adaptive_sequential_frame_batch_sampler_basic():
    cfg = AdaptiveSequentialFrameBatchSamplerConfig(
        n_samples_per_sequence=2, max_frame_gap_timestamp_us=200000
    )
    assert cfg.n_frames_per_sample == 18
    assert cfg.n_samples_per_sequence == 2
    assert cfg.max_frame_gap_timestamp_us == 200000


def test_camera_subsampler_public_model_dimensions():
    cfg = CameraSubsamplerConfig()
    assert cfg.frame_width == 784
    assert cfg.frame_height == 448


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
