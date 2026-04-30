# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from pathlib import Path
from typing import List

from pxr import Sdf, Usd, UsdUtils

from nre.utils.io.utils import initialize_usd_stage
from nre.utils.types import NamedUSDStage


def update_render_settings(stage: Usd.Stage, referenced_layer: Sdf.Layer) -> None:
    if "renderSettings" not in referenced_layer.customLayerData:
        return  # Do nothing if render settings are not present in the referenced layer
    new_render_settings = referenced_layer.customLayerData["renderSettings"]

    current_render_settings = stage.GetRootLayer().customLayerData.get("renderSettings", {})
    if current_render_settings is None:
        current_render_settings = {}

    current_render_settings.update(new_render_settings)
    stage.SetMetadataByDictKey("customLayerData", "renderSettings", current_render_settings)


def update_animation_settings(stage: Usd.Stage, referenced_layer: Sdf.Layer) -> None:
    USD_DEFAULT_TIME_CODE = 0.0
    USD_DEFAULT_TIME_CODES_PER_SECOND = 24.0

    if referenced_layer.startTimeCode != USD_DEFAULT_TIME_CODE:
        current_start_time = stage.GetStartTimeCode()
        new_start_time = referenced_layer.startTimeCode
        if current_start_time == USD_DEFAULT_TIME_CODE:
            current_start_time = new_start_time
        new_start_time_code = min(current_start_time, new_start_time)
        stage.SetStartTimeCode(new_start_time_code)

    if referenced_layer.endTimeCode != USD_DEFAULT_TIME_CODE:
        current_end_time = stage.GetEndTimeCode()
        new_end_time = referenced_layer.endTimeCode
        if current_end_time == USD_DEFAULT_TIME_CODE:
            current_end_time = new_end_time
        new_end_time_code = max(current_end_time, new_end_time)
        stage.SetEndTimeCode(new_end_time_code)

    if referenced_layer.timeCodesPerSecond != USD_DEFAULT_TIME_CODES_PER_SECOND:
        current_time_codes_per_second = stage.GetTimeCodesPerSecond()
        new_time_codes_per_second = referenced_layer.timeCodesPerSecond
        if current_time_codes_per_second == USD_DEFAULT_TIME_CODES_PER_SECOND:
            stage.SetTimeCodesPerSecond(new_time_codes_per_second)
        elif current_time_codes_per_second != new_time_codes_per_second:
            raise ValueError(
                f"TimeCodesPerSecond mismatch: existing value {current_time_codes_per_second} and new value {new_time_codes_per_second}"
            )

    if "absoluteTimeOffsetMicroSec" in referenced_layer.customLayerData:
        new_absolute_time_code_offset = referenced_layer.customLayerData["absoluteTimeOffsetMicroSec"]
        current_absolute_time_code_offset = stage.GetMetadataByDictKey("customLayerData", "absoluteTimeOffsetMicroSec")
        if not current_absolute_time_code_offset:
            stage.SetMetadataByDictKey("customLayerData", "absoluteTimeOffsetMicroSec", new_absolute_time_code_offset)
        elif new_absolute_time_code_offset != current_absolute_time_code_offset:
            raise ValueError(
                f"absoluteTimeOffsetMicroSec mismatch: existing value {current_absolute_time_code_offset} and new value {new_absolute_time_code_offset}"
            )


def serialize_usd_default_layer(references: List[NamedUSDStage]) -> Usd.Stage:
    stage = initialize_usd_stage()

    # The delegate captures all errors about dangling references, effectively silencing them.
    delegate = UsdUtils.CoalescingDiagnosticDelegate()
    for named_stage in references:
        prim = stage.OverridePrim(f"/World/{Path(named_stage.filename).stem}")
        # Assume that all reference paths are in the same directory, so that they are also valid relative file paths.
        prim.GetReferences().AddReference(named_stage.filename)

        # Bubble up any render and animation settings as they are not referenced from the prim.
        referenced_layer = named_stage.stage.GetRootLayer()
        update_render_settings(stage, referenced_layer)
        update_animation_settings(stage, referenced_layer)

    return stage
