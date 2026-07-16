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

"""Branch-coverage tests for the public model architecture configs."""

from __future__ import annotations

import sys

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


from pydantic import ValidationError

from instant_nurec.config_schema.models import PrimitiveExportPreprocessConfig
from instant_nurec.config_schema.models import (
    GaussiansActivationConfig,
    KelvinDAv3EncoderConfig,
    KelvinDPTDecoderConfig,
    KelvinModelConfig,
    KelvinSkyCubemapDecoderConfig,
)


# ---------- KelvinDPTDecoderConfig.model_post_init ----------


def test_kelvin_dpt_decoder_post_init_accepts_positive_dpt_dim():
    cfg = KelvinDPTDecoderConfig(dpt_dim=128, dpt_reassemble_hidden_dims=[8, 16, 32, 64])
    assert cfg.dpt_dim == 128
    assert cfg.checkpointing is True
    assert cfg.dpt_chunk_size == 4
    assert cfg.time_encoding_dim == 256
    assert cfg.motion_depth == 4


def test_kelvin_dpt_decoder_post_init_rejects_zero_dpt_dim():
    with pytest.raises(ValidationError, match="must be positive"):
        KelvinDPTDecoderConfig(dpt_dim=0, dpt_reassemble_hidden_dims=[8, 16, 32, 64])


def test_kelvin_dpt_decoder_post_init_rejects_negative_dpt_dim():
    with pytest.raises(ValidationError, match="must be positive"):
        KelvinDPTDecoderConfig(dpt_dim=-1, dpt_reassemble_hidden_dims=[8, 16, 32, 64])


# ---------- GaussiansActivationConfig defaults ----------


def test_activation_config_defaults():
    cfg = GaussiansActivationConfig()
    assert cfg.opacity_shift == -2.0
    assert cfg.scale_shift_log_ratio == -2.9
    assert cfg.scale_max == 0.045
    assert cfg.scale_min == 0.0


def test_activation_config_custom_values():
    cfg = GaussiansActivationConfig(
        opacity_shift=1.0, scale_shift_log_ratio=0.0, scale_max=0.5, scale_min=0.01
    )
    assert cfg.opacity_shift == 1.0
    assert cfg.scale_shift_log_ratio == 0.0
    assert cfg.scale_max == 0.5
    assert cfg.scale_min == 0.01


# ---------- KelvinModelConfig (composed) ----------


def _make_full_model_cfg(**overrides):
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
    cfg = _make_full_model_cfg()
    assert cfg.track_padding_m == [1.0, 1.0, 1.0]
    assert cfg.scene_rescale == 0.15
    assert cfg.patch_shape == (14, 14)
    assert isinstance(cfg.activations, GaussiansActivationConfig)
    assert isinstance(cfg.export_preprocess, PrimitiveExportPreprocessConfig)


def test_kelvin_model_rejects_track_padding_not_3_long():
    with pytest.raises(ValidationError):
        _make_full_model_cfg(track_padding_m=[1.0, 1.0])


def test_kelvin_model_rejects_track_padding_too_long():
    with pytest.raises(ValidationError):
        _make_full_model_cfg(track_padding_m=[1.0, 1.0, 1.0, 1.0])
