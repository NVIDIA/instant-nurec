# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import json
import logging

from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import click
import numpy as np
import torch

import ncore_internal.impl.common.transformations as ncore_internal_transformations

from ncore.data import BivariateWindshieldModelParameters, FThetaCameraModelParameters, ReferencePolynomial, ShutterType
from nre.artifact import Artifact
from nre.utils.types import RigTrajectories


log = logging.getLogger(__name__)


# --- Rig JSON parsing ---
def _load_rig_json(rig_json_path: str) -> Dict[str, Dict]:
    """Load and parse camera sensors from an NDAS rig JSON file.

    Returns:
        Dictionary mapping camera_id to sensor dict with 'extrinsics' and 'intrinsics' keys.
    """
    log.info(f"Loading rig JSON: {rig_json_path}")
    with open(rig_json_path) as f:
        rig_data = json.load(f)

    sensors = rig_data["rig"]["sensors"]
    cameras: Dict[str, Dict] = {}
    for sensor in sensors:
        name = sensor.get("name", "")
        if not name.startswith("camera:"):
            continue
        camera_id = name.replace(":", "_")

        nominal = sensor["nominalSensor2Rig_FLU"]
        roll, pitch, yaw = nominal["roll-pitch-yaw"]
        tx, ty, tz = nominal["t"]

        correction_rpy = sensor.get("correction_sensor_R_FLU", {}).get("roll-pitch-yaw", [0.0, 0.0, 0.0])
        correction_t = sensor.get("correction_rig_T", [0.0, 0.0, 0.0])

        props = sensor.get("properties", {})
        if props is None or props.get("Model") != "ftheta":
            log.warning(f"Skipping {camera_id}: not an ftheta camera")
            continue

        intrinsics: Dict = {
            "width": int(props["width"]),
            "height": int(props["height"]),
            "cx": float(props["cx"]),
            "cy": float(props["cy"]),
            "polynomial": [float(c) for c in props["polynomial"].split()],
            "polynomial_type": props["polynomial-type"],
            "linear_cde": [
                float(props.get("linear-c", "1")),
                float(props.get("linear-d", "0")),
                float(props.get("linear-e", "0")),
            ],
        }

        # Parse windshield distortion if present.
        # Expects the rig JSON to be augmented with inverse polys via DW WindshieldModelInversion
        ws_h = props.get("windshield-horizontal-polynomial")
        ws_v = props.get("windshield-vertical-polynomial")
        ws_h_inv = props.get("windshield-horizontal-polynomial-approx-inverse")
        ws_v_inv = props.get("windshield-vertical-polynomial-approx-inverse")
        if ws_h and ws_v:
            if not (ws_h_inv and ws_v_inv):
                raise ValueError(
                    f"Camera {camera_id} has forward windshield polynomials but no inverse. "
                    f"Run rig_format_adjuster --add-inverse-windshield on the rig JSON first."
                )
            intrinsics["windshield"] = {
                "horizontal_poly": [float(c) for c in ws_h.split()],
                "vertical_poly": [float(c) for c in ws_v.split()],
                "horizontal_poly_inverse": [float(c) for c in ws_h_inv.split()],
                "vertical_poly_inverse": [float(c) for c in ws_v_inv.split()],
                "polynomial_type": props.get("windshield-polynomial-type", "forward"),
            }

        cameras[camera_id] = {
            "extrinsics": {
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
                "tx": tx,
                "ty": ty,
                "tz": tz,
                "correction_rpy": correction_rpy,
                "correction_t": correction_t,
            },
            "intrinsics": intrinsics,
        }

    log.info(f"Loaded {len(cameras)} cameras from rig JSON: {list(cameras.keys())}")
    return cameras


# --- Extrinsics ---
def _rpy_to_T_sensor_rig(
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
    tx: float,
    ty: float,
    tz: float,
    correction_rpy: Optional[List[float]] = None,
    correction_t: Optional[List[float]] = None,
) -> np.ndarray:
    """Convert NDAS roll-pitch-yaw extrinsics to a 4x4 T_sensor_rig matrix.

    Reproduces SensorExtrinsicProfile::loadSensorExtrinsics from DW:
      1. R_sensor = R_nomFLU @ sensor2sensorFLU
      2. R_sensor = R_sensor @ sensor2sensorFLU^T @ R_corrFLU @ sensor2sensorFLU  (if correction)
      3. t = nominal_t + correction_t
    """
    # Maps camera coords (z-fwd, x-right, y-down) to FLU (x-fwd, y-left, z-up)
    sensor2sensorFLU = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32)

    R_nomFLU = ncore_internal_transformations.euler_2_so3(
        np.array([roll_deg, pitch_deg, yaw_deg], dtype=np.float32), degrees=True, seq="xyz"
    )
    R_sensor = R_nomFLU @ sensor2sensorFLU

    if correction_rpy is not None and any(c != 0.0 for c in correction_rpy):
        R_corrFLU = ncore_internal_transformations.euler_2_so3(
            np.array(correction_rpy, dtype=np.float32), degrees=True, seq="xyz"
        )
        R_sensor = R_sensor @ sensor2sensorFLU.T @ R_corrFLU @ sensor2sensorFLU

    t = np.array([tx, ty, tz], dtype=np.float32)
    if correction_t is not None:
        t = t + np.array(correction_t, dtype=np.float32)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R_sensor
    T[:3, 3] = t
    return T


# --- Intrinsics ---
def _compute_inverse_poly(
    coeffs: List[float], range_min: float, range_max: float, num_samples: int = 500, degree: int = 0
) -> List[float]:
    """Compute inverse polynomial via least-squares fitting (replicates DW's computeInversePoly).

    Samples the forward polynomial at uniform points, swaps (x, y), and fits an inverse
    polynomial with c0=0 constraint.
    """
    if degree == 0:
        degree = len(coeffs) - 1  # -1 because we prepend c0=0, keeping total length = len(coeffs)

    xs = np.linspace(range_min, range_max, num_samples, dtype=np.float64)
    ys = np.zeros_like(xs)
    for i, c in enumerate(coeffs):
        ys += c * xs**i

    # Fit inv_poly(ys) ≈ xs with c0=0: inv_poly(y) = c1*y + c2*y^2 + ...
    A = np.column_stack([ys**i for i in range(1, degree + 1)])
    inv_coeffs, _, _, _ = np.linalg.lstsq(A, xs, rcond=None)

    return [0.0] + inv_coeffs.tolist()


def _create_camera_model_parameters(
    intrinsics: Dict,
) -> FThetaCameraModelParameters:
    """Create FThetaCameraModelParameters from parsed NDAS rig intrinsics.

    From the augmented rig JSON we get:
      - ftheta polynomial (forward direction) and its type
      - principal point, resolution, linear_cde
      - windshield forward and inverse polynomials

    Computed here:
      - inverse ftheta polynomial (via least-squares, replicating DW's computeInversePoly)
      - max_angle (from polynomial evaluation at max pixel distance)
      - shutter_type: all NDAS cameras are rolling shutter, and the polynomial type is either
        pixeldistance2angle or angle2pixeldistance.
    """
    width, height = intrinsics["width"], intrinsics["height"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    polynomial = intrinsics["polynomial"]
    polynomial_type = intrinsics["polynomial_type"]

    # rig.json does not contain shutter type info, as all NDAS cameras are rolling shutter
    shutter_type = ShutterType.ROLLING_TOP_TO_BOTTOM

    corners = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32)
    max_pixel_dist = float(np.max(np.linalg.norm(corners - np.array([cx, cy], dtype=np.float32), axis=1)))

    if polynomial_type == "pixeldistance-to-angle":
        reference_poly = FThetaCameraModelParameters.PolynomialType.PIXELDIST_TO_ANGLE
        max_angle = float(np.polyval(np.array(polynomial, dtype=np.float64)[::-1], max_pixel_dist))
        pixeldist_to_angle_poly = polynomial
        angle_to_pixeldist_poly = _compute_inverse_poly(polynomial, 0.0, max_pixel_dist)
    elif polynomial_type == "angle-to-pixeldistance":
        reference_poly = FThetaCameraModelParameters.PolynomialType.ANGLE_TO_PIXELDIST
        deriv_coeffs = [i * c for i, c in enumerate(polynomial)][1:]
        lo, hi = 0.0, np.pi
        for _ in range(100):
            mid = (lo + hi) / 2.0
            if sum(c * mid**i for i, c in enumerate(deriv_coeffs)) > 0:
                lo = mid
            else:
                hi = mid
        max_angle = (lo + hi) / 2.0
        angle_to_pixeldist_poly = polynomial
        pixeldist_to_angle_poly = _compute_inverse_poly(polynomial, 0.0, max_angle)
    else:
        raise ValueError(f"Unknown polynomial type: {polynomial_type}")

    # Windshield distortion: both forward and inverse polys from the augmented rig JSON.
    # The rig JSON must be preprocessed with DW's WindshieldModelInversion to compute the inverse polynomials.
    external_distortion: Optional[BivariateWindshieldModelParameters] = None
    ws = intrinsics.get("windshield")
    if ws is not None:
        ws_poly_type = ws["polynomial_type"]
        assert ws_poly_type == "forward", f"Expected forward windshield polynomial type, got: {ws_poly_type}"

        external_distortion = BivariateWindshieldModelParameters(
            reference_poly=ReferencePolynomial.FORWARD,
            horizontal_poly=np.array(ws["horizontal_poly"], dtype=np.float32),
            vertical_poly=np.array(ws["vertical_poly"], dtype=np.float32),
            horizontal_poly_inverse=np.array(ws["horizontal_poly_inverse"], dtype=np.float32),
            vertical_poly_inverse=np.array(ws["vertical_poly_inverse"], dtype=np.float32),
        )
        log.info("Windshield distortion loaded from augmented rig JSON")

    return FThetaCameraModelParameters(
        resolution=np.array([width, height], dtype=np.uint64),
        shutter_type=shutter_type,
        principal_point=np.array([cx, cy], dtype=np.float32),
        reference_poly=reference_poly,
        pixeldist_to_angle_poly=np.array(pixeldist_to_angle_poly, dtype=np.float32),
        angle_to_pixeldist_poly=np.array(angle_to_pixeldist_poly, dtype=np.float32),
        max_angle=max_angle,
        linear_cde=np.array(intrinsics["linear_cde"], dtype=np.float32),
        external_distortion_parameters=external_distortion,
    )


# --- CLI command ---
@click.command("export-custom-rig-trajectory")
@click.option("--artifact-path", type=str, default=None, help="Path to a USDZ artifact file.")
@click.option(
    "--reference-rig-trajectory",
    type=str,
    default=None,
    help="Path to a reference rig_trajectories.json file. Alternative to --artifact-path.",
)
@click.option(
    "--rig-json",
    type=str,
    default=None,
    help="Path to an NDAS rig JSON file. If provided, replaces camera calibrations. "
    "If omitted, the training rig trajectories are extracted as-is.",
)
@click.option("--output", type=str, required=True, help="Path to write the output rig_trajectories.json.")
def export_custom_rig_trajectory(
    artifact_path: Optional[str],
    reference_rig_trajectory: Optional[str],
    rig_json: Optional[str],
    output: str,
) -> None:
    """Create a custom rig_trajectories.json for use with 'nre render --custom-rig-trajectory'.

    Reads training rig trajectories from a USDZ artifact (--artifact-path) or a JSON file
    (--reference-rig-trajectory). Optionally replaces camera calibrations with those from an
    NDAS rig JSON (--rig-json). Without --rig-json, extracts the training rig trajectories as-is.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Load reference rig trajectories
    if artifact_path is not None and reference_rig_trajectory is not None:
        raise click.UsageError("Specify either --artifact-path or --reference-rig-trajectory, not both.")
    if artifact_path is None and reference_rig_trajectory is None:
        raise click.UsageError("One of --artifact-path or --reference-rig-trajectory is required.")

    if artifact_path is not None:
        log.info(f"Loading rig trajectories from artifact {artifact_path}")
        source_data = Artifact(Path(artifact_path)).rig_trajectories
    else:
        assert reference_rig_trajectory is not None
        log.info(f"Loading reference rig trajectories from {reference_rig_trajectory}")
        with open(reference_rig_trajectory) as f:
            source_data = json.load(f)

    source_rt = RigTrajectories.from_dict(source_data)
    log.info(f"Source cameras: {list(source_rt.camera_calibrations.keys())}")

    # If no rig JSON, just write the training trajectories
    if rig_json is None:
        log.info("No --rig-json provided, extracting training rig trajectories as-is")
        _write_output(source_rt, output)
        return

    # Parse NDAS rig JSON and auto-map to reference cameras by matching ID prefix before '@'.
    # Cameras that match a reference camera reuse its key and metadata. Cameras with no match
    # in the reference are treated as new: they get a synthesized key ("{rig_id}@{sequence_id}")
    # and borrow trajectory timestamps, indices and poses from the first reference camera.
    rig_cameras = _load_rig_json(rig_json)
    source_cam_ids = list(source_rt.camera_calibrations.keys())
    default_cam_id = source_cam_ids[0]
    default_calib = source_rt.camera_calibrations[default_cam_id]
    cam_mapping: Dict[str, str] = {}
    new_cam_ids: List[str] = []
    for rig_cam_id in rig_cameras:
        matched_cam_ids = [rcid for rcid in source_cam_ids if rcid.split("@")[0] == rig_cam_id]
        if len(matched_cam_ids) > 1:
            raise ValueError(f"Rig camera '{rig_cam_id}' matched multiple source cameras: {matched_cam_ids}")
        if matched_cam_ids:
            cam_mapping[rig_cam_id] = matched_cam_ids[0]
        else:
            new_cam_id = f"{rig_cam_id}@{default_calib.sequence_id}"
            cam_mapping[rig_cam_id] = new_cam_id
            new_cam_ids.append(new_cam_id)
            log.info(f"New camera '{rig_cam_id}' not in reference, will borrow timestamps from '{default_cam_id}'")

    log.info(f"Camera mapping: {cam_mapping}")

    # Build new camera calibrations with rig extrinsics/intrinsics
    target_calibrations: OrderedDict[str, RigTrajectories.CameraCalibration] = OrderedDict()
    target_cam_ids = set(cam_mapping.values())
    for rig_cam_id, target_cam_id in cam_mapping.items():
        source_calib = source_rt.camera_calibrations.get(target_cam_id)
        if source_calib is not None:
            seq_id = source_calib.sequence_id
            sensor_idx = source_calib.unique_sensor_idx
        else:
            # New camera not in training: reuse the default camera's sequence_id and
            # unique_sensor_idx so it renders with valid trained ISP parameters.
            # unique_sensor_idx indexes into the post-processing model's per-camera
            # parameter array, so using an out-of-range index would crash the render
            # subcommand.  Reusing the default's index means duplicated indices, which
            # breaks the RigTrajectories uniqueness convention but avoids the crash.
            # TODO: the ISP models should accept unique_sensor_idx=None (the
            # rendering API already supports it) so novel cameras don't need to
            # borrow a training index.
            seq_id = default_calib.sequence_id
            sensor_idx = default_calib.unique_sensor_idx

        rig_extrinsics = rig_cameras[rig_cam_id]["extrinsics"]
        target_calibrations[target_cam_id] = RigTrajectories.CameraCalibration(
            sequence_id=seq_id,
            logical_sensor_name=rig_cam_id,
            unique_sensor_idx=sensor_idx,
            T_sensor_rig=torch.from_numpy(
                _rpy_to_T_sensor_rig(
                    rig_extrinsics["roll"],
                    rig_extrinsics["pitch"],
                    rig_extrinsics["yaw"],
                    rig_extrinsics["tx"],
                    rig_extrinsics["ty"],
                    rig_extrinsics["tz"],
                    # Parameter corrections are optional in the NDAS rig JSON.
                    correction_rpy=rig_extrinsics.get("correction_rpy"),
                    correction_t=rig_extrinsics.get("correction_t"),
                )
            ),
            camera_model_parameters=_create_camera_model_parameters(
                rig_cameras[rig_cam_id]["intrinsics"],
            ),
        )

    # Build trajectories: keep entries for matched cameras, and for new cameras borrow
    # timestamps / poses from the default reference camera.
    target_trajectories: List[RigTrajectories.RigTrajectory] = []
    for source_trajectory in source_rt.rig_trajectories:
        # Populate dicts with matched cameras from the source trajectory
        # Filters out dict items of unmatched cameras from the source trajectory.
        ts_dict = {k: v for k, v in source_trajectory.cameras_frame_timestamps_us.items() if k in target_cam_ids}
        poses_dict = (
            {k: v for k, v in source_trajectory.cameras_frame_T_rig_worlds.items() if k in target_cam_ids}
            if source_trajectory.cameras_frame_T_rig_worlds
            else None
        )

        # For new cameras not in the reference, clone trajectory data from the default camera.
        # Raise ValueError if the default camera has no timestamps, since we have nothing to
        # borrow for the new cameras.
        if new_cam_ids and default_cam_id not in source_trajectory.cameras_frame_timestamps_us:
            raise ValueError(
                f"Default source camera '{default_cam_id}' has no timestamps in this trajectory; "
                f"cannot borrow timestamps for new cameras {new_cam_ids}"
            )
        # rig poses interpolated at each frame are stored in the
        # source trajectory for the default camera. This situation is default.
        source_per_frame_poses = (
            source_trajectory.cameras_frame_T_rig_worlds.get(default_cam_id)
            if source_trajectory.cameras_frame_T_rig_worlds is not None
            else None
        )
        for new_cam_id in new_cam_ids:
            ts_dict[new_cam_id] = source_trajectory.cameras_frame_timestamps_us[default_cam_id].clone()
            if poses_dict is not None and source_per_frame_poses is not None:
                poses_dict[new_cam_id] = source_per_frame_poses.clone()

        # cameras_linear_start_frame_indices: set to None.  unique_frame_idx in
        # rendering can only refer to training views and should not be used with
        # --custom-rig-trajectory, whose purpose is to feed novel views.  Feeding
        # training frame indices would violate the RigTrajectories class invariant
        # because indices would not necessarily be unique across novel cameras.
        # We avoid this dilemma by setting the field to None.
        target_trajectories.append(
            replace(
                source_trajectory,
                cameras_frame_timestamps_us=ts_dict,
                cameras_linear_start_frame_indices=None,
                cameras_frame_T_rig_worlds=poses_dict,
            )
        )

    target_rt = replace(source_rt, rig_trajectories=target_trajectories, camera_calibrations=target_calibrations)
    _write_output(target_rt, output)


def _write_output(rig_trajectories: RigTrajectories, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(rig_trajectories.to_dict(), f, indent=4)
    log.info(f"Wrote {output_path} with cameras: {list(rig_trajectories.camera_calibrations.keys())}")
