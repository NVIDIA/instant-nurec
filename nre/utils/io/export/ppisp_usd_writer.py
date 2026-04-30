# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
PPISP USD Writer

Export PPISP (Physically Plausible Image Signal Processing) shader to USD RenderProducts.

This module provides functions to add PPISP post-processing shaders to USD RenderProducts
for each camera, enabling ISP pipeline rendering in USD-compatible renderers.

PPISP pipeline stages:
1. Exposure compensation (per-frame)
2. Vignetting correction (per-camera, per-channel)
3. Color correction / homography (per-frame)
4. Camera Response Function (per-camera, per-channel)
"""

from __future__ import annotations

import logging

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from pxr import Gf, Sdf, Usd, UsdShade, Vt

from nre.utils.types import NamedSerialized


if TYPE_CHECKING:
    from nre.models.post_processings.ppisp import BasePPISP
    from nre.utils.types import RigTrajectories


log = logging.getLogger(__name__)

# PPISP parameter dimensions (matching nrePPISPPostProcessing.h)
NUM_CHANNELS = 3
NUM_VIGNETTING_OPTICAL_CENTER = 2
NUM_VIGNETTING_ALPHA_TERMS = 3
NUM_VIGNETTING_PARAMS = NUM_VIGNETTING_OPTICAL_CENTER + NUM_VIGNETTING_ALPHA_TERMS
NUM_CRF_PARAMS = 7
NUM_HOMOGRAPHY_PARAMS = 8

# PPISP SPG file paths (relative to this module)
PPISP_SPG_DIR = Path(__file__).parent
PPISP_SPG_SLANG_FILE = "ppisp_usd_spg.slang"
PPISP_SPG_LUA_FILE = "ppisp_usd_spg.slang.lua"
PPISP_SPG_USDA_FILE = "ppisp_usd_spg.slang.usda"

# Shared RenderVar names for SPG pipeline
PPISP_INPUT_RENDER_VAR = "HdrColor"  # Input AOV name
PPISP_OUTPUT_RENDER_VAR = "PPISPColor"  # Output AOV name
LDR_COLOR_RENDER_VAR = "LdrColor"  # Display AOV wired to PPISP result


def _add_ldr_color_render_var(stage: Usd.Stage, render_product_path: str, ppisp_output_path: Sdf.Path) -> str:
    """Create a LdrColor RenderVar wired to the PPISP output (omni:rtx:aov connects to PPISP.outputs:PPISPColor).
    Returns the path to the created RenderVar for appending to orderedVars."""
    ldr_var_path = f"{render_product_path}/{LDR_COLOR_RENDER_VAR}"
    ldr_var = stage.DefinePrim(ldr_var_path, "RenderVar")
    ldr_var.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set(LDR_COLOR_RENDER_VAR)
    aov_attr = ldr_var.CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque)
    aov_attr.SetConnections([ppisp_output_path])
    return ldr_var_path


def add_ppisp_shader_to_render_product(
    stage: Usd.Stage,
    render_product_path: str,
    camera_index: int,
    ppisp: BasePPISP,
    frame_index: int = 0,
) -> Usd.Prim:
    """
    Add a PPISP shader to a RenderProduct for a specific camera.

    Creates a Shader prim with PPISP parameters using the SPG (Slang Post Graph) format.
    References ppisp_usd_spg.usda and connects to the RenderProduct's HdrColor input,
    outputting to a LdrColor RenderVar (wired to the PPISP result).

    Args:
        stage: USD stage containing the RenderProduct
        render_product_path: Path to the RenderProduct prim (e.g., "/Render/front_camera")
        camera_index: Index of the camera in the PPISP model (for vignetting/CRF params)
        ppisp: PPISP module containing the ISP parameters
        frame_index: Frame index for per-frame parameters (exposure, color). Default 0.

    Returns:
        The created PPISP Shader prim

    Raises:
        ValueError: If the RenderProduct doesn't exist or camera_index is out of bounds
    """
    render_product = stage.GetPrimAtPath(render_product_path)
    if not render_product.IsValid():
        raise ValueError(f"RenderProduct not found at path: {render_product_path}")

    if camera_index < 0 or camera_index >= ppisp.num_cameras:
        raise ValueError(f"camera_index {camera_index} out of bounds [0, {ppisp.num_cameras})")

    if frame_index < 0 or frame_index >= ppisp.num_frames:
        raise ValueError(f"frame_index {frame_index} out of bounds [0, {ppisp.num_frames})")

    # Add opaque omni:rtx:aov to input RenderVar (no connection, just declared)
    input_var_path = f"{render_product_path}/{PPISP_INPUT_RENDER_VAR}"
    input_var_prim = stage.GetPrimAtPath(input_var_path)
    if input_var_prim.IsValid():
        input_var_prim.CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque)

    # Create PPISP Shader via UsdShade and reference SPG definition
    ppisp_shader_path = f"{render_product_path}/PPISP"
    shader = UsdShade.Shader.Define(stage, ppisp_shader_path)
    shader.GetPrim().GetReferences().AddReference(PPISP_SPG_USDA_FILE)

    # Set parameters via UsdShade inputs
    _set_exposure_params(shader, ppisp, frame_index)
    _set_vignetting_params(shader, ppisp, camera_index)
    _set_color_params(shader, ppisp, frame_index)
    _set_crf_params(shader, ppisp, camera_index)

    # Connect HdrColor input to input RenderVar (Omniverse-specific; no UsdShade on RenderVar)
    hdr_input = shader.CreateInput(PPISP_INPUT_RENDER_VAR, Sdf.ValueTypeNames.Token)
    hdr_input.GetAttr().SetConnections([Sdf.Path(f"../{PPISP_INPUT_RENDER_VAR}.omni:rtx:aov")])

    # Declare output for LdrColor to connect to
    shader.CreateOutput(PPISP_OUTPUT_RENDER_VAR, Sdf.ValueTypeNames.Token)

    # LdrColor RenderVar wired to PPISP result (display AOV)
    ppisp_output_path = shader.GetPath().AppendProperty(f"outputs:{PPISP_OUTPUT_RENDER_VAR}")
    ldr_var_path = _add_ldr_color_render_var(stage, render_product_path, ppisp_output_path)

    # Update orderedVars to include LdrColor
    ordered_vars_rel = render_product.GetRelationship("orderedVars")
    if ordered_vars_rel:
        existing_targets = list(ordered_vars_rel.GetTargets())
        existing_targets.append(Sdf.Path(ldr_var_path))
        ordered_vars_rel.SetTargets(existing_targets)

    log.info(f"Added PPISP shader to {render_product_path} for camera {camera_index}")
    return shader.GetPrim()


def add_animated_ppisp_shader_to_render_product(
    stage: Usd.Stage,
    render_product_path: str,
    camera_index: int,
    ppisp: BasePPISP,
    camera_frame_mapping: CameraFrameMapping,
    usd_timestamp_offset_us: int,
    usd_time_code_scale: float,
) -> Usd.Prim:
    """
    Add a PPISP shader to a RenderProduct with time-sampled per-frame parameters.

    Creates a Shader prim with PPISP parameters using the SPG format.
    Per-camera parameters (vignetting, CRF) are static; per-frame parameters
    (exposure, color homography) are animated using USD time samples.

    Args:
        stage: USD stage containing the RenderProduct
        render_product_path: Path to the RenderProduct prim
        camera_index: Index of the camera in the PPISP model (for vignetting/CRF params)
        ppisp: PPISP module containing the ISP parameters
        camera_frame_mapping: Mapping from timestamps to PPISP frame indices for this camera
        usd_timestamp_offset_us: Timestamp offset in microseconds for time code calculation
        usd_time_code_scale: Scale factor (timeCodesPerSecond * 1e-6) for timestamp to time code

    Returns:
        The created PPISP Shader prim

    Raises:
        ValueError: If the RenderProduct doesn't exist or camera_index is out of bounds
    """
    render_product = stage.GetPrimAtPath(render_product_path)
    if not render_product.IsValid():
        raise ValueError(f"RenderProduct not found at path: {render_product_path}")

    if camera_index < 0 or camera_index >= ppisp.num_cameras:
        raise ValueError(f"camera_index {camera_index} out of bounds [0, {ppisp.num_cameras})")

    # Add opaque omni:rtx:aov to input RenderVar
    input_var_path = f"{render_product_path}/{PPISP_INPUT_RENDER_VAR}"
    input_var_prim = stage.GetPrimAtPath(input_var_path)
    if input_var_prim.IsValid():
        input_var_prim.CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque)

    # Create PPISP Shader via UsdShade and reference SPG definition
    ppisp_shader_path = f"{render_product_path}/PPISP"
    shader = UsdShade.Shader.Define(stage, ppisp_shader_path)
    shader.GetPrim().GetReferences().AddReference(PPISP_SPG_USDA_FILE)

    # Set per-camera static parameters (vignetting, CRF)
    _set_vignetting_params(shader, ppisp, camera_index)
    _set_crf_params(shader, ppisp, camera_index)

    # Set per-frame animated parameters (exposure, color homography)
    _set_animated_exposure_params(shader, ppisp, camera_frame_mapping, usd_timestamp_offset_us, usd_time_code_scale)
    _set_animated_color_params(shader, ppisp, camera_frame_mapping, usd_timestamp_offset_us, usd_time_code_scale)

    # Connect HdrColor input to input RenderVar (Omniverse-specific; no UsdShade on RenderVar)
    hdr_input = shader.CreateInput(PPISP_INPUT_RENDER_VAR, Sdf.ValueTypeNames.Token)
    hdr_input.GetAttr().SetConnections([Sdf.Path(f"../{PPISP_INPUT_RENDER_VAR}.omni:rtx:aov")])

    # Declare output for LdrColor to connect to
    shader.CreateOutput(PPISP_OUTPUT_RENDER_VAR, Sdf.ValueTypeNames.Token)

    # LdrColor RenderVar wired to PPISP result (display AOV)
    ppisp_output_path = shader.GetPath().AppendProperty(f"outputs:{PPISP_OUTPUT_RENDER_VAR}")
    ldr_var_path = _add_ldr_color_render_var(stage, render_product_path, ppisp_output_path)

    # Update orderedVars to include LdrColor
    ordered_vars_rel = render_product.GetRelationship("orderedVars")
    if ordered_vars_rel:
        existing_targets = list(ordered_vars_rel.GetTargets())
        existing_targets.append(Sdf.Path(ldr_var_path))
        ordered_vars_rel.SetTargets(existing_targets)

    num_frames = len(camera_frame_mapping.timestamp_to_ppisp_frame_idx)
    log.info(f"Added animated PPISP shader to {render_product_path} for camera {camera_index} with {num_frames} frames")
    return shader.GetPrim()


def _set_exposure_params(shader: UsdShade.Shader, ppisp: BasePPISP, frame_index: int) -> None:
    """Set exposure offset parameter for the given frame."""
    exposure_params = ppisp.packed_exposure_params.cpu().numpy()
    exposure_offset = float(exposure_params[frame_index])
    shader.CreateInput("exposureOffset", Sdf.ValueTypeNames.Float).Set(exposure_offset)


def _set_vignetting_params(shader: UsdShade.Shader, ppisp: BasePPISP, camera_index: int) -> None:
    """Set vignetting parameters for the given camera (per-channel)."""
    vignetting_params = ppisp.packed_vignetting_params
    # Shape: (num_cameras, 3, 5) -> [center_x, center_y, alpha1, alpha2, alpha3]

    for channel in range(NUM_CHANNELS):
        channel_suffix = ["R", "G", "B"][channel]

        # Optical center
        optical_center = vignetting_params.optical_center[camera_index, channel].cpu().numpy()
        shader.CreateInput(f"vignettingCenter{channel_suffix}", Sdf.ValueTypeNames.Float2).Set(
            Gf.Vec2f(float(optical_center[0]), float(optical_center[1]))
        )

        # Alpha terms
        alphas = vignetting_params.alphas[camera_index, channel].cpu().numpy()
        shader.CreateInput(f"vignettingAlpha1{channel_suffix}", Sdf.ValueTypeNames.Float).Set(float(alphas[0]))
        shader.CreateInput(f"vignettingAlpha2{channel_suffix}", Sdf.ValueTypeNames.Float).Set(float(alphas[1]))
        shader.CreateInput(f"vignettingAlpha3{channel_suffix}", Sdf.ValueTypeNames.Float).Set(float(alphas[2]))


def _set_color_params(shader: UsdShade.Shader, ppisp: BasePPISP, frame_index: int) -> None:
    """Set color correction / homography parameters for the given frame."""
    color_params = ppisp.packed_color_params.cpu().numpy()
    # Shape: (num_frames, 8) -> 3x3 homography matrix (last element is 1.0)
    homography = color_params[frame_index]

    # Store as 3x3 matrix (row-major)
    # The homography is stored as 8 values: [h00, h01, h02, h10, h11, h12, h20, h21]
    # with h22 = 1.0 implicitly
    # Gf.Matrix3d constructor takes 9 separate doubles
    matrix = Gf.Matrix3d(
        float(homography[0]),
        float(homography[1]),
        float(homography[2]),
        float(homography[3]),
        float(homography[4]),
        float(homography[5]),
        float(homography[6]),
        float(homography[7]),
        1.0,
    )
    shader.CreateInput("colorHomography", Sdf.ValueTypeNames.Matrix3d).Set(matrix)


def _set_crf_params(shader: UsdShade.Shader, ppisp: BasePPISP, camera_index: int) -> None:
    """Set Camera Response Function parameters for the given camera (per-channel)."""
    crf_curve_points = ppisp.crf_curve_points
    # CRF parameters per camera, per channel

    for channel in range(NUM_CHANNELS):
        channel_suffix = ["R", "G", "B"][channel]

        # Extract curve points for this camera/channel
        x0 = float(crf_curve_points.x0[camera_index, channel].cpu().item())
        y0 = float(crf_curve_points.y0[camera_index, channel].cpu().item())
        slope_p0 = float(crf_curve_points.slope_p0[camera_index, channel].cpu().item())
        y0_pre_gamma = float(crf_curve_points.y0_pre_gamma[camera_index, channel].cpu().item())
        slope_line = float(crf_curve_points.slope_line[camera_index, channel].cpu().item())
        gamma = float(crf_curve_points.gamma[camera_index, channel].cpu().item())
        x1 = float(crf_curve_points.x1[camera_index, channel].cpu().item())
        y1 = float(crf_curve_points.y1[camera_index, channel].cpu().item())
        slope_p1 = float(crf_curve_points.slope_p1[camera_index, channel].cpu().item())
        shoulder_x = float(crf_curve_points.shoulder_x[camera_index, channel].cpu().item())
        shoulder_y = float(crf_curve_points.shoulder_y[camera_index, channel].cpu().item())

        shader.CreateInput(f"crfX0{channel_suffix}", Sdf.ValueTypeNames.Float).Set(x0)
        shader.CreateInput(f"crfY0{channel_suffix}", Sdf.ValueTypeNames.Float).Set(y0)
        shader.CreateInput(f"crfSlopeP0{channel_suffix}", Sdf.ValueTypeNames.Float).Set(slope_p0)
        shader.CreateInput(f"crfY0PreGamma{channel_suffix}", Sdf.ValueTypeNames.Float).Set(y0_pre_gamma)
        shader.CreateInput(f"crfSlopeLine{channel_suffix}", Sdf.ValueTypeNames.Float).Set(slope_line)
        shader.CreateInput(f"crfGamma{channel_suffix}", Sdf.ValueTypeNames.Float).Set(gamma)
        shader.CreateInput(f"crfX1{channel_suffix}", Sdf.ValueTypeNames.Float).Set(x1)
        shader.CreateInput(f"crfY1{channel_suffix}", Sdf.ValueTypeNames.Float).Set(y1)
        shader.CreateInput(f"crfSlopeP1{channel_suffix}", Sdf.ValueTypeNames.Float).Set(slope_p1)
        shader.CreateInput(f"crfShoulderX{channel_suffix}", Sdf.ValueTypeNames.Float).Set(shoulder_x)
        shader.CreateInput(f"crfShoulderY{channel_suffix}", Sdf.ValueTypeNames.Float).Set(shoulder_y)


# -----------------------------------------------------------------------------
# Animated Parameter Writing (Time-Sampled)
# -----------------------------------------------------------------------------


def _set_animated_exposure_params(
    shader: UsdShade.Shader,
    ppisp: BasePPISP,
    camera_frame_mapping: CameraFrameMapping,
    usd_timestamp_offset_us: int,
    usd_time_code_scale: float,
) -> None:
    """Write time-sampled exposure offset parameters for all frames of a camera.

    Time samples are authored on the single attribute inputs:exposureOffset (no separate
    .timeSamples attribute). A default value is set from the first frame so evaluation
    is well-defined when no time sample exists.

    Args:
        shader: PPISP shader (UsdShade) to write attributes to
        ppisp: PPISP module containing exposure parameters
        camera_frame_mapping: Mapping containing timestamps and PPISP frame indices
        usd_timestamp_offset_us: Timestamp offset in microseconds
        usd_time_code_scale: Scale factor for converting timestamps to USD time codes
    """
    exposure_params = ppisp.packed_exposure_params.cpu().numpy()
    exposure_input = shader.CreateInput("exposureOffset", Sdf.ValueTypeNames.Float)
    exposure_attr = exposure_input.GetAttr()

    first_default_set = False
    for timestamp_us, ppisp_frame_idx in camera_frame_mapping.timestamp_to_ppisp_frame_idx.items():
        if ppisp_frame_idx >= len(exposure_params):
            log.warning(
                f"PPISP frame index {ppisp_frame_idx} out of bounds for exposure params "
                f"(length {len(exposure_params)}), skipping"
            )
            continue

        exposure_offset = float(exposure_params[ppisp_frame_idx])
        usd_time_code = usd_time_code_scale * (timestamp_us - usd_timestamp_offset_us)
        if not first_default_set:
            exposure_attr.Set(exposure_offset)
            first_default_set = True
        exposure_attr.Set(exposure_offset, usd_time_code)


def _set_animated_color_params(
    shader: UsdShade.Shader,
    ppisp: BasePPISP,
    camera_frame_mapping: CameraFrameMapping,
    usd_timestamp_offset_us: int,
    usd_time_code_scale: float,
) -> None:
    """Write time-sampled color homography parameters for all frames of a camera.

    Time samples are authored on the single attribute inputs:colorHomography (no separate
    .timeSamples attribute). A default value is set from the first frame so evaluation
    is well-defined when no time sample exists.

    Args:
        shader: PPISP shader (UsdShade) to write attributes to
        ppisp: PPISP module containing color parameters
        camera_frame_mapping: Mapping containing timestamps and PPISP frame indices
        usd_timestamp_offset_us: Timestamp offset in microseconds
        usd_time_code_scale: Scale factor for converting timestamps to USD time codes
    """
    color_params = ppisp.packed_color_params.cpu().numpy()
    color_input = shader.CreateInput("colorHomography", Sdf.ValueTypeNames.Matrix3d)
    color_attr = color_input.GetAttr()

    first_default_set = False
    for timestamp_us, ppisp_frame_idx in camera_frame_mapping.timestamp_to_ppisp_frame_idx.items():
        if ppisp_frame_idx >= len(color_params):
            log.warning(
                f"PPISP frame index {ppisp_frame_idx} out of bounds for color params "
                f"(length {len(color_params)}), skipping"
            )
            continue

        homography = color_params[ppisp_frame_idx]
        matrix = Gf.Matrix3d(
            float(homography[0]),
            float(homography[1]),
            float(homography[2]),
            float(homography[3]),
            float(homography[4]),
            float(homography[5]),
            float(homography[6]),
            float(homography[7]),
            1.0,
        )
        usd_time_code = usd_time_code_scale * (timestamp_us - usd_timestamp_offset_us)
        if not first_default_set:
            color_attr.Set(matrix)
            first_default_set = True
        color_attr.Set(matrix, usd_time_code)


def add_ppisp_to_all_render_products(
    stage: Usd.Stage,
    ppisp: BasePPISP,
    camera_name_to_index: dict[str, int],
    render_scope_path: str = "/Render",
    frame_index: int = 0,
    camera_frame_mappings: Optional[Dict[str, CameraFrameMapping]] = None,
    usd_timestamp_offset_us: int = 0,
    usd_time_code_scale: Optional[float] = None,
) -> list[Usd.Prim]:
    """
    Add PPISP shaders to all RenderProducts in the Render scope.

    Iterates over RenderProducts and adds PPISP shaders using the camera-to-index mapping.
    If camera_frame_mappings is provided, per-frame parameters (exposure, color) are animated
    using USD time samples; otherwise, static values at frame_index are used.

    Args:
        stage: USD stage containing the RenderProducts
        ppisp: PPISP module containing the ISP parameters
        camera_name_to_index: Mapping from camera name to PPISP camera index
        render_scope_path: Path to the Render scope (default "/Render")
        frame_index: Frame index for static per-frame parameters (ignored if animated)
        camera_frame_mappings: Optional mapping from camera name to CameraFrameMapping
            for animated per-frame parameters. If provided, enables animation.
        usd_timestamp_offset_us: Timestamp offset in microseconds (for animation)
        usd_time_code_scale: Scale factor (timeCodesPerSecond * 1e-6). If None, computed from stage.

    Returns:
        List of created PPISP shader prims
    """
    render_scope = stage.GetPrimAtPath(render_scope_path)
    if not render_scope.IsValid():
        log.warning(f"Render scope not found at {render_scope_path}")
        return []

    # Compute time code scale from stage if not provided
    if camera_frame_mappings is not None and usd_time_code_scale is None:
        usd_time_code_scale = stage.GetTimeCodesPerSecond() * 1e-6

    use_animation = camera_frame_mappings is not None and len(camera_frame_mappings) > 0

    created_shaders = []
    for child in render_scope.GetChildren():
        if child.GetTypeName() == "RenderProduct":
            camera_name = child.GetName()
            if camera_name in camera_name_to_index:
                camera_index = camera_name_to_index[camera_name]

                if use_animation and camera_frame_mappings is not None and camera_name in camera_frame_mappings:
                    # Use animated version with time-sampled per-frame parameters
                    shader = add_animated_ppisp_shader_to_render_product(
                        stage=stage,
                        render_product_path=str(child.GetPath()),
                        camera_index=camera_index,
                        ppisp=ppisp,
                        camera_frame_mapping=camera_frame_mappings[camera_name],
                        usd_timestamp_offset_us=usd_timestamp_offset_us,
                        usd_time_code_scale=usd_time_code_scale,  # type: ignore
                    )
                else:
                    # Use static version with single frame_index
                    shader = add_ppisp_shader_to_render_product(
                        stage=stage,
                        render_product_path=str(child.GetPath()),
                        camera_index=camera_index,
                        ppisp=ppisp,
                        frame_index=frame_index,
                    )
                created_shaders.append(shader)
            else:
                log.warning(f"Camera '{camera_name}' not found in camera_name_to_index mapping")

    if use_animation:
        log.info(f"Added animated PPISP shaders to {len(created_shaders)} RenderProducts")
    else:
        log.info(f"Added PPISP shaders to {len(created_shaders)} RenderProducts")
    return created_shaders


# -----------------------------------------------------------------------------
# PPISP Detection and Mapping
# -----------------------------------------------------------------------------


def get_ppisp_from_model(model: Any) -> Optional[BasePPISP]:
    """
    Extract PPISP module from model's post-processings if available.

    Args:
        model: The Gaussian model that may have post-processings

    Returns:
        The BasePPISP module if found, None otherwise
    """
    from nre.models.post_processing import PPISPPostProcessing

    if not hasattr(model, "post_processings"):
        return None

    for pp in model.post_processings:
        if isinstance(pp, PPISPPostProcessing):
            return pp.ppisp

    return None


def build_camera_name_to_index_mapping(rig_trajectories: RigTrajectories) -> Dict[str, int]:
    """
    Build a mapping from camera logical sensor name to PPISP camera index.

    The camera index corresponds to the order cameras appear in the rig trajectories,
    which matches how PPISP parameters are indexed.

    Args:
        rig_trajectories: RigTrajectories object from datasource

    Returns:
        Dictionary mapping camera name to index
    """
    camera_name_to_index: Dict[str, int] = {}
    camera_index = 0

    for rig_trajectory in rig_trajectories.rig_trajectories:
        for camera_unique_name in rig_trajectory.cameras_frame_timestamps_us:
            camera_data = rig_trajectories.camera_calibrations[camera_unique_name]
            logical_name = camera_data.logical_sensor_name
            if logical_name not in camera_name_to_index:
                camera_name_to_index[logical_name] = camera_index
                camera_index += 1

    return camera_name_to_index


@dataclass
class CameraFrameMapping:
    """Mapping from timestamps to PPISP frame indices for a single camera.

    Attributes:
        camera_name: Logical sensor name (used in USD RenderProduct paths)
        camera_unique_id: Unique sensor ID (used in rig_trajectory)
        linear_start_frame_idx: Starting PPISP frame index for this camera
        frame_timestamps_us: List of (end) timestamps for each frame in this camera
        timestamp_to_ppisp_frame_idx: Dict mapping timestamp to PPISP frame index
    """

    camera_name: str
    camera_unique_id: str
    linear_start_frame_idx: int
    frame_timestamps_us: List[int]
    timestamp_to_ppisp_frame_idx: Dict[int, int]


@dataclass
class TrainingFrameFilterConfig:
    """Configuration for filtering training frames from validation frames.

    Mirrors the validation frame selection logic from NCORESequentialDataset.
    Training frames are those NOT selected for validation.

    Two modes are supported (mutually exclusive):
    1. val_frame_step mode: validation frames are at indices [start::step]
       -> training frames are all other frames
    2. val_exclude_frame_step mode: validation frames EXCLUDE indices [start::step]
       -> training frames are at indices [start::step]

    Attributes:
        val_frame_start: Start index for validation frame selection (default 0)
        val_frame_step: Step for validation frame selection (mode 1)
        val_exclude_frame_start: Start index for exclusion pattern (mode 2)
        val_exclude_frame_step: Step for exclusion pattern (mode 2)
    """

    val_frame_start: Optional[int] = 0
    val_frame_step: Optional[int] = None
    val_exclude_frame_start: Optional[int] = None
    val_exclude_frame_step: Optional[int] = None

    def get_training_frame_local_indices(self, num_frames: int) -> List[int]:
        """Compute local frame indices that are training frames (not validation).

        Args:
            num_frames: Total number of frames for this camera

        Returns:
            List of local frame indices that are training frames
        """
        all_indices = list(range(num_frames))

        if self.val_frame_step is not None:
            # Mode 1: validation = frames at [start::step], training = everything else
            start = self.val_frame_start if self.val_frame_start is not None else 0
            val_indices = set(all_indices[start :: self.val_frame_step])
            training_indices = [i for i in all_indices if i not in val_indices]
        elif self.val_exclude_frame_step is not None:
            # Mode 2: validation = frames NOT at [start::step], training = frames at [start::step]
            start = self.val_exclude_frame_start if self.val_exclude_frame_start is not None else 0
            training_indices = all_indices[start :: self.val_exclude_frame_step]
        else:
            # No filtering - all frames are training frames
            training_indices = all_indices

        return training_indices


def build_camera_frame_mappings(
    rig_trajectories: RigTrajectories,
    training_filter: Optional[TrainingFrameFilterConfig] = None,
) -> Dict[str, CameraFrameMapping]:
    """
    Build per-camera mappings from timestamps to PPISP frame indices.

    Uses cameras_linear_start_frame_indices from rig_trajectory to compute the
    correct PPISP frame index for each camera frame. The end timestamp of each
    frame is used for matching (consistent with training code).

    If training_filter is provided, only training frames (not validation frames)
    are included in the mapping. This ensures PPISP animation only uses frames
    that have trained parameter values.

    Args:
        rig_trajectories: RigTrajectories object containing camera frame info
        training_filter: Optional config to filter out validation frames.
            If None, all frames are included.

    Returns:
        Dict mapping camera logical name to CameraFrameMapping
    """
    camera_mappings: Dict[str, CameraFrameMapping] = {}

    for rig_trajectory in rig_trajectories.rig_trajectories:
        linear_start_indices = rig_trajectory.cameras_linear_start_frame_indices
        if linear_start_indices is None:
            log.warning("cameras_linear_start_frame_indices not available, PPISP animation disabled")
            return {}

        for camera_unique_id in rig_trajectory.cameras_frame_timestamps_us:
            camera_data = rig_trajectories.camera_calibrations[camera_unique_id]
            camera_name = camera_data.logical_sensor_name

            if camera_name in camera_mappings:
                # Camera already processed (from another rig trajectory with same logical name)
                continue

            linear_start_idx = linear_start_indices.get(camera_unique_id, 0)

            # Get frame timestamps (Nx2 tensor with start/end timestamps)
            timestamps_tensor = rig_trajectory.cameras_frame_timestamps_us[camera_unique_id]
            # Use end timestamps (column 1) for frame matching
            end_timestamps = timestamps_tensor[:, 1].numpy().tolist()
            num_frames = len(end_timestamps)

            # Determine which local frame indices to include
            if training_filter is not None:
                training_local_indices = training_filter.get_training_frame_local_indices(num_frames)
            else:
                training_local_indices = list(range(num_frames))

            # Build timestamp -> PPISP frame index mapping (only for training frames)
            timestamp_to_frame: Dict[int, int] = {}
            included_timestamps: List[int] = []
            for local_idx in training_local_indices:
                end_ts = int(end_timestamps[local_idx])
                ppisp_frame_idx = linear_start_idx + local_idx
                timestamp_to_frame[end_ts] = ppisp_frame_idx
                included_timestamps.append(end_ts)

            camera_mappings[camera_name] = CameraFrameMapping(
                camera_name=camera_name,
                camera_unique_id=camera_unique_id,
                linear_start_frame_idx=linear_start_idx,
                frame_timestamps_us=included_timestamps,
                timestamp_to_ppisp_frame_idx=timestamp_to_frame,
            )

            if training_filter is not None:
                log.debug(
                    f"Camera '{camera_name}': linear_start={linear_start_idx}, "
                    f"total_frames={num_frames}, training_frames={len(training_local_indices)}, "
                    f"frame_range=[{linear_start_idx}, {linear_start_idx + num_frames - 1}]"
                )
            else:
                log.debug(
                    f"Camera '{camera_name}': linear_start={linear_start_idx}, "
                    f"num_frames={num_frames}, "
                    f"frame_range=[{linear_start_idx}, {linear_start_idx + num_frames - 1}]"
                )

    return camera_mappings


# -----------------------------------------------------------------------------
# PPISP SPG File Handling
# -----------------------------------------------------------------------------


def get_ppisp_spg_files() -> List[NamedSerialized]:
    """
    Load all PPISP SPG files (slang, lua, usda) as serialized data.

    Returns:
        List of NamedSerialized objects for each SPG file
    """
    spg_files: List[NamedSerialized] = []

    for filename in [PPISP_SPG_SLANG_FILE, PPISP_SPG_LUA_FILE, PPISP_SPG_USDA_FILE]:
        file_path = PPISP_SPG_DIR / filename
        if file_path.exists():
            with open(file_path, "rb") as f:
                file_data = f.read()
            spg_files.append(NamedSerialized(filename=filename, serialized=file_data))
            log.info(f"Loaded PPISP SPG file: {filename}")
        else:
            log.warning(f"PPISP SPG file not found: {file_path}")

    return spg_files
