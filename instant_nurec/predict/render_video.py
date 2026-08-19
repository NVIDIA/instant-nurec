# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream a calibrated source-camera trajectory to an H.264 video."""

from __future__ import annotations

import shutil
import subprocess
import tempfile

from dataclasses import dataclass
from pathlib import Path

import torch

from tqdm import tqdm

from instant_nurec.predict.render_preview import render_composited_frame
from instant_nurec.primitives.kelvin_primitive import KelvinInstantNuRecPrimitive
from instant_nurec.utils.batch import (
    CameraFrameLabels,
    CameraFreePoseViewGeometry,
    DataAndRenderingBatch,
    DataBatch,
    FrameMeta,
    RenderingBatch,
)
from instant_nurec.utils.types import RigTrajectories


@dataclass(frozen=True, slots=True)
class RenderVideoStats:
    path: Path
    width: int
    height: int
    frame_count: int
    fps: float
    duration_seconds: float
    background_fraction: float
    sky_contribution_mean: float


def require_ffmpeg() -> str:
    """Return an ffmpeg with libx264 or fail before reconstruction begins."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "Full calibrated video rendering requires ffmpeg on PATH. Install ffmpeg, "
            "then rerun with --merge --render-video."
        )
    probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        message = probe.stderr.strip() or "no error output"
        raise RuntimeError(f"Could not query ffmpeg encoders: {message}")
    if "libx264" not in probe.stdout:
        raise RuntimeError("Full calibrated video rendering requires an ffmpeg build with the libx264 encoder.")
    return ffmpeg


def infer_source_fps(frame_timestamps_startend_us: torch.Tensor) -> float:
    """Infer fixed-rate playback from the median source-frame cadence."""

    if frame_timestamps_startend_us.ndim != 2 or frame_timestamps_startend_us.shape[1] != 2:
        raise ValueError(
            f"Expected source timestamps with shape (frames, 2), got {tuple(frame_timestamps_startend_us.shape)}"
        )
    if len(frame_timestamps_startend_us) < 2:
        raise ValueError("At least two source frames are required to render a video")
    starts = frame_timestamps_startend_us[:, 0].to(dtype=torch.float64, device="cpu")
    cadence_us = torch.diff(starts)
    if bool((cadence_us <= 0).any()):
        raise ValueError("Source camera frame timestamps must be strictly increasing")
    return 1_000_000.0 / float(cadence_us.median().item())


def _ffmpeg_command(
    ffmpeg: str,
    *,
    width: int,
    height: int,
    fps: float,
    output_path: Path,
) -> list[str]:
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{fps:.8f}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def _reserve_partial_path(path: Path) -> Path:
    """Reserve a unique output path beside the final video."""

    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.",
        suffix=".partial.mp4",
        dir=path.parent,
        delete=False,
    ) as partial_file:
        return Path(partial_file.name)


def _ffmpeg_error(returncode: int, stderr_log) -> RuntimeError:
    stderr_log.flush()
    stderr_log.seek(0)
    stderr = stderr_log.read().decode("utf-8", errors="replace").strip()
    return RuntimeError(f"ffmpeg exited with status {returncode}: {stderr or 'no error output'}")


def _close_stdin(process: subprocess.Popen) -> None:
    if process.stdin is None or process.stdin.closed:
        return
    try:
        process.stdin.close()
    except (BrokenPipeError, OSError):
        pass


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        _close_stdin(process)
        return
    process.kill()
    _close_stdin(process)
    try:
        process.wait()
    except OSError:
        pass


def _single_camera_sequence(
    rig: RigTrajectories,
) -> tuple[str, RigTrajectories.CameraCalibration, torch.Tensor]:
    if len(rig.camera_calibrations) != 1:
        raise ValueError(
            f"Full video rendering expects exactly one camera calibration, got {list(rig.camera_calibrations)}"
        )
    sensor_id, calibration = next(iter(rig.camera_calibrations.items()))
    candidates = [
        trajectory for trajectory in rig.rig_trajectories if trajectory.sequence_id == calibration.sequence_id
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one trajectory for render camera {sensor_id!r}, found {len(candidates)}")
    return sensor_id, calibration, candidates[0].cameras_frame_timestamps_us[sensor_id]


def _frame_context(
    geometry: CameraFreePoseViewGeometry,
    *,
    unique_sensor_idx: int,
    unique_frame_idx: int,
    device: torch.device,
) -> DataAndRenderingBatch:
    camera = DataBatch.Camera(
        meta=[
            FrameMeta(
                unique_sensor_idx=unique_sensor_idx,
                unique_frame_idx=unique_frame_idx,
            )
        ],
        labels=CameraFrameLabels(),
    ).to(device)
    rendering = geometry.to_rendering_data(camera)
    return DataAndRenderingBatch(
        data=DataBatch(camera=camera),
        rendering=RenderingBatch(camera=rendering),
    )


@torch.inference_mode()
def render_reference_video(
    primitive: KelvinInstantNuRecPrimitive,
    full_camera_rig: RigTrajectories,
    path: Path,
) -> RenderVideoStats:
    """Render every source exposure while keeping only one frame's rays live."""

    ffmpeg = require_ffmpeg()
    sensor_id, calibration, timestamps = _single_camera_sequence(full_camera_rig)
    fps = infer_source_fps(timestamps)
    frame_count = len(timestamps)
    device = primitive.device()
    geometry = CameraFreePoseViewGeometry.from_rig_trajectories(full_camera_rig).to(device=device)
    frame_range = geometry.sensor_ids_to_frame_range[sensor_id]
    if len(frame_range) != frame_count:
        raise ValueError(f"Geometry frame count {len(frame_range)} does not match source timestamps {frame_count}")

    width, height = (int(value) for value in calibration.camera_model_parameters.resolution)
    if width % 2 or height % 2:
        raise ValueError(f"H.264 yuv420p output requires even dimensions, got {width}x{height}")

    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = _reserve_partial_path(path)
    process: subprocess.Popen | None = None
    completed = False
    background_sum = 0.0
    sky_contribution_sum = 0.0
    with tempfile.TemporaryFile(mode="w+b") as stderr_log:
        try:
            process = subprocess.Popen(
                _ffmpeg_command(
                    ffmpeg,
                    width=width,
                    height=height,
                    fps=fps,
                    output_path=partial_path,
                ),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=stderr_log,
            )
            if process.stdin is None:  # pragma: no cover - subprocess contract
                raise RuntimeError("Failed to open ffmpeg input pipe")

            try:
                for unique_frame_idx in tqdm(
                    frame_range,
                    total=frame_count,
                    desc="Rendering calibrated video",
                    unit="frame",
                ):
                    context = _frame_context(
                        geometry,
                        unique_sensor_idx=calibration.unique_sensor_idx,
                        unique_frame_idx=unique_frame_idx,
                        device=device,
                    )
                    composed, opacity, sky = render_composited_frame(primitive, context)
                    if composed.shape != (height, width, 3):
                        raise ValueError(
                            f"Rendered frame has shape {tuple(composed.shape)}, expected {(height, width, 3)}"
                        )
                    pixels = (
                        composed.mul(255.0)
                        .round_()
                        .clamp_(0.0, 255.0)
                        .to(dtype=torch.uint8, device="cpu")
                        .contiguous()
                        .numpy()
                    )
                    process.stdin.write(pixels.tobytes())
                    background_sum += float((1.0 - opacity).mean().item())
                    sky_contribution_sum += float(((1.0 - opacity[..., None]) * sky).mean().item())

                process.stdin.close()
            except BrokenPipeError as exc:
                returncode = process.wait()
                raise _ffmpeg_error(returncode, stderr_log) from exc

            returncode = process.wait()
            if returncode != 0:
                raise _ffmpeg_error(returncode, stderr_log)
            partial_path.replace(path)
            completed = True
        finally:
            if process is not None:
                _stop_process(process)
            if not completed:
                partial_path.unlink(missing_ok=True)

    duration_seconds = frame_count / fps
    return RenderVideoStats(
        path=path,
        width=width,
        height=height,
        frame_count=frame_count,
        fps=fps,
        duration_seconds=duration_seconds,
        background_fraction=background_sum / frame_count,
        sky_contribution_mean=sky_contribution_sum / frame_count,
    )
