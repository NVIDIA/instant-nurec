# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Extract all camera intrinsics, extrinsics, and external distortion parameters from a trained USDZ artifact.

Iterates over every camera in the rig trajectories and writes a single JSON file
containing the full parameter set for each camera, keyed by unique sensor ID.

Usage:
    python extract_camera_params.py /path/to/model.usdz -o camera_params.json
    python extract_camera_params.py /path/to/model.usdz --filter cam_front
"""

from __future__ import annotations

import json
import logging

from pathlib import Path
from typing import Any

import click
import numpy as np

import ncore.data

from nre.artifact.artifact import Artifact
from nre.utils.types import RigTrajectories


log = logging.getLogger(__name__)


def _safe_get(obj: Any, attr: str, *, convert: str = "tolist") -> Any:
    """Safely read an attribute, returning None when the attribute is absent or conversion fails.

    Args:
        obj: The object to read from.
        attr: Attribute name.
        convert: Conversion to apply — "tolist" calls .tolist(), "float" wraps in float(),
                 "str" wraps in str(), "raw" returns the value as-is.
    """
    if not hasattr(obj, attr):
        log.debug("Attribute '%s' not found on %s", attr, type(obj).__name__)
        return None
    val: Any = getattr(obj, attr)
    try:
        if convert == "tolist":
            return val.tolist() if hasattr(val, "tolist") else val
        if convert == "float":
            return float(val)
        if convert == "str":
            return str(val)
        return val
    except Exception:
        log.warning("Failed to convert '%s' on %s", attr, type(obj).__name__, exc_info=True)
        return str(val)


def _ndarray_to_list(v: Any) -> Any:
    """Recursively convert numpy arrays and torch tensors to JSON-friendly lists."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    try:
        import torch

        if isinstance(v, torch.Tensor):
            return v.detach().cpu().numpy().tolist()
    except ImportError:
        pass
    if isinstance(v, dict):
        return {k: _ndarray_to_list(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_ndarray_to_list(item) for item in v]
    return v


def _extract_ftheta(params: Any) -> dict[str, Any]:
    return {
        "principal_point": _safe_get(params, "principal_point"),
        "pixeldist_to_angle_poly": _safe_get(params, "pixeldist_to_angle_poly"),
        "angle_to_pixeldist_poly": _safe_get(params, "angle_to_pixeldist_poly"),
        "reference_poly": _safe_get(params, "reference_poly", convert="str"),
        "max_angle_rad": _safe_get(params, "max_angle", convert="float"),
        "linear_cde": _safe_get(params, "linear_cde"),
    }


def _extract_opencv_pinhole(params: Any) -> dict[str, Any]:
    return {
        "principal_point": _safe_get(params, "principal_point"),
        "focal_length": _safe_get(params, "focal_length"),
        "radial_coeffs": _safe_get(params, "radial_coeffs"),
        "tangential_coeffs": _safe_get(params, "tangential_coeffs"),
        "thin_prism_coeffs": _safe_get(params, "thin_prism_coeffs"),
    }


def _extract_opencv_fisheye(params: Any) -> dict[str, Any]:
    return {
        "principal_point": _safe_get(params, "principal_point"),
        "focal_length": _safe_get(params, "focal_length"),
        "radial_coeffs": _safe_get(params, "radial_coeffs"),
        "max_angle_rad": _safe_get(params, "max_angle", convert="float"),
    }


def _extract_external_distortion(ext: Any) -> dict[str, Any]:
    return {
        "type": "bivariate-windshield",
        "reference_poly": _safe_get(ext, "reference_poly", convert="str"),
        "horizontal_poly": _safe_get(ext, "horizontal_poly"),
        "vertical_poly": _safe_get(ext, "vertical_poly"),
        "horizontal_poly_inverse": _safe_get(ext, "horizontal_poly_inverse"),
        "vertical_poly_inverse": _safe_get(ext, "vertical_poly_inverse"),
    }


def _extract_camera(unique_id: str, calib: Any) -> dict[str, Any]:
    """Extract all parameters from a single camera calibration.

    Every attribute access goes through _safe_get so that a missing or renamed
    field yields a None + log warning instead of crashing the whole extraction.
    """
    params = getattr(calib, "camera_model_parameters", None)
    if params is None:
        log.error("Camera '%s' has no camera_model_parameters — skipping intrinsics.", unique_id)
        return {"_error": "missing camera_model_parameters"}

    camera_type: str | None = None
    try:
        camera_type = params.type()
    except Exception:
        log.warning("Could not determine camera model type for '%s'.", unique_id, exc_info=True)

    entry: dict[str, Any] = {
        "logical_sensor_name": _safe_get(calib, "logical_sensor_name", convert="raw"),
        "sequence_id": _safe_get(calib, "sequence_id", convert="raw"),
        "unique_sensor_idx": _safe_get(calib, "unique_sensor_idx", convert="raw"),
        "camera_model_type": camera_type,
        "resolution": _safe_get(params, "resolution"),
        "shutter_type": _safe_get(params, "shutter_type", convert="str"),
        "T_sensor_rig": _ndarray_to_list(getattr(calib, "T_sensor_rig", None)),
    }

    intrinsics_extracted = False
    try:
        if isinstance(params, ncore.data.FThetaCameraModelParameters):
            entry["intrinsics"] = _extract_ftheta(params)
            intrinsics_extracted = True
        elif isinstance(params, ncore.data.OpenCVPinholeCameraModelParameters):
            entry["intrinsics"] = _extract_opencv_pinhole(params)
            intrinsics_extracted = True
        elif isinstance(params, ncore.data.OpenCVFisheyeCameraModelParameters):
            entry["intrinsics"] = _extract_opencv_fisheye(params)
            intrinsics_extracted = True
    except Exception:
        log.warning(
            "Typed extraction failed for '%s' (%s), falling back to to_dict().", unique_id, camera_type, exc_info=True
        )

    if not intrinsics_extracted:
        try:
            entry["intrinsics"] = _ndarray_to_list(params.to_dict())
        except Exception:
            log.error("to_dict() fallback also failed for '%s'.", unique_id, exc_info=True)
            entry["intrinsics"] = None

    try:
        ext = getattr(params, "external_distortion_parameters", None)
        if ext is not None:
            entry["external_distortion"] = _extract_external_distortion(ext)
        else:
            entry["external_distortion"] = None
    except Exception:
        log.warning("Failed to extract external_distortion for '%s'.", unique_id, exc_info=True)
        entry["external_distortion"] = None

    return entry


def extract_all_cameras(artifact_path: Path, filter_name: str | None = None) -> dict[str, Any]:
    """Load a USDZ artifact and extract camera parameters for all (or filtered) cameras.

    Args:
        artifact_path: Path to the .usdz file.
        filter_name: If provided, only include cameras whose logical_sensor_name contains this substring.

    Returns:
        Dictionary keyed by unique sensor ID with full camera parameters.
    """
    artifact = Artifact(artifact_path)
    rig_trajectories = RigTrajectories.from_dict(artifact.rig_trajectories)

    calibrations = getattr(rig_trajectories, "camera_calibrations", None)
    if calibrations is None:
        log.error("RigTrajectories has no 'camera_calibrations' attribute — ncore API may have changed.")
        return {}

    result: dict[str, Any] = {}
    for unique_id, calib in calibrations.items():
        logical_name = getattr(calib, "logical_sensor_name", unique_id)
        if filter_name and filter_name not in logical_name:
            continue
        log.info("Extracting: %s (logical: %s)", unique_id, logical_name)
        try:
            result[unique_id] = _extract_camera(unique_id, calib)
        except Exception:
            log.error("Unexpected failure extracting '%s' — skipping.", unique_id, exc_info=True)
            result[unique_id] = {"_error": f"extraction failed for {unique_id}"}

    return result


@click.command()
@click.argument("usdz_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output JSON path. Defaults to <usdz_stem>_camera_params.json.",
)
@click.option(
    "--filter",
    "filter_name",
    type=str,
    default=None,
    help="Only include cameras whose logical_sensor_name contains this substring.",
)
@click.option("--pretty/--compact", default=True, help="Pretty-print JSON output (default: pretty).")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def main(usdz_path: Path, output: Path | None, filter_name: str | None, pretty: bool, verbose: bool) -> None:
    """Extract camera parameters from a trained USDZ artifact to JSON."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s: %(message)s")

    cameras = extract_all_cameras(usdz_path, filter_name=filter_name)

    if not cameras:
        log.warning("No cameras found%s.", f" matching '{filter_name}'" if filter_name else "")
        return

    log.info("Extracted %d camera(s).", len(cameras))

    if output is None:
        output = usdz_path.with_name(f"{usdz_path.stem}_camera_params.json")

    indent = 2 if pretty else None
    output.write_text(json.dumps(cameras, indent=indent, default=str))
    log.info("Written to %s", output)


if __name__ == "__main__":
    main()
