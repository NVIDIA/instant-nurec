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

from __future__ import annotations

import os

import shortuuid

from instant_nurec.config_schema.base_schema import BaseConfigSchema, Field
from instant_nurec.config_schema.dataset import InstantNuRecSplitsConfig
from instant_nurec.config_schema.models import KelvinModelConfig, KelvinPointQueryCADecoderConfig
from instant_nurec.config_schema.predict import PredictConfig
from instant_nurec.pretrained import DEFAULT_MODEL_VARIANT, ModelVariant, get_model_profile


class GaussiansInstantNuRecSystemConfig(BaseConfigSchema):
    """Predict-only system config; just dataloader knobs."""

    predict_num_workers: int = Field(default=4, description="Number of workers for the predict dataloader per-node.")
    predict_batch_size: int = Field(default=8, description="Batch size for the predict dataloader. Typically set to 1.")


class InstantNuRecConfig(BaseConfigSchema):
    """Top-level predict configuration.

    All defaults are populated for the canonical kelvin-pa-front predict
    pipeline; only ``out_dir`` and ``dataset.predict.{ncore_json_*}``
    must be supplied per-invocation.
    """

    seed: int = Field(default=38, description="Random seed.")

    release_profile: ModelVariant = Field(
        default=DEFAULT_MODEL_VARIANT,
        description="Released checkpoint/input profile to use for inference.",
    )

    out_dir: str

    system: GaussiansInstantNuRecSystemConfig = Field(default_factory=GaussiansInstantNuRecSystemConfig)
    dataset: InstantNuRecSplitsConfig
    model: KelvinModelConfig = Field(default_factory=KelvinModelConfig)

    predict: PredictConfig = Field(
        default_factory=PredictConfig,
        description="Configuration for predict-time-only functionality such as primitive merging",
    )

    run_id: str = Field(
        default_factory=shortuuid.uuid,
        description=(
            "Unique identifier of this run; auto-generated as a shortuuid unless "
            "overridden via the INSTANT_NUREC_RUN_ID environment variable."
        ),
    )

    def model_post_init(self, __context) -> None:
        profile = get_model_profile(self.release_profile)
        configured_decoder_kind = (
            "point-query"
            if isinstance(self.model.decoder, KelvinPointQueryCADecoderConfig)
            else "pixel-aligned"
        )
        if configured_decoder_kind != profile.decoder_kind:
            raise ValueError(
                f"release_profile={self.release_profile!r} requires a "
                f"{profile.decoder_kind} decoder, but model.decoder is "
                f"{configured_decoder_kind}"
            )
        predict_dataset = self.dataset.predict
        if profile.decoder_kind == "point-query" and predict_dataset is not None:
            context_camera_ids = tuple(predict_dataset.context_camera_ids)
            if context_camera_ids != profile.context_camera_ids:
                required = ", ".join(profile.context_camera_ids)
                raise ValueError(
                    f"release_profile={self.release_profile!r} requires context camera(s): {required}"
                )

        if (env_run_id := os.environ.get("INSTANT_NUREC_RUN_ID")) is not None:
            self.run_id = env_run_id
