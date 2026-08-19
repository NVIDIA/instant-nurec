# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import instant_nurec.predict.render_video as render_video


def test_infer_source_fps_uses_median_start_cadence() -> None:
    timestamps = torch.tensor(
        [[10, 20], [33_343, 33_353], [66_677, 66_687], [100_010, 100_020]],
        dtype=torch.int64,
    )

    assert render_video.infer_source_fps(timestamps) == pytest.approx(1_000_000 / 33_333)


@pytest.mark.parametrize(
    "timestamps, message",
    [
        (torch.zeros(1, 2, dtype=torch.int64), "At least two"),
        (torch.tensor([[2, 3], [2, 4]]), "strictly increasing"),
        (torch.zeros(2, 3, dtype=torch.int64), "shape"),
    ],
)
def test_infer_source_fps_rejects_invalid_timeline(
    timestamps: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        render_video.infer_source_fps(timestamps)


def test_require_ffmpeg_reports_missing_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render_video.shutil, "which", lambda _: None)

    with pytest.raises(RuntimeError, match="ffmpeg on PATH"):
        render_video.require_ffmpeg()


def test_require_ffmpeg_reports_missing_libx264(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render_video.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        render_video.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="h264_nvenc\n", stderr=""),
    )

    with pytest.raises(RuntimeError, match="libx264 encoder"):
        render_video.require_ffmpeg()


def test_partial_video_paths_are_unique_and_share_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "scene.render.mp4"
    first = render_video._reserve_partial_path(output)
    second = render_video._reserve_partial_path(output)
    try:
        assert first.parent == output.parent
        assert second.parent == output.parent
        assert first != second
        assert first.name.endswith(".partial.mp4")
        assert second.name.endswith(".partial.mp4")
    finally:
        first.unlink(missing_ok=True)
        second.unlink(missing_ok=True)


def test_render_reference_video_streams_every_frame_and_atomically_finishes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    timestamps = torch.tensor(
        [[0, 100], [33_333, 33_433], [133_333, 133_433]],
        dtype=torch.int64,
    )
    calibration = SimpleNamespace(
        sequence_id="context-main",
        unique_sensor_idx=7,
        camera_model_parameters=SimpleNamespace(resolution=(4, 2)),
    )
    trajectory = SimpleNamespace(
        sequence_id="context-main",
        cameras_frame_timestamps_us={"front": timestamps},
    )
    rig = SimpleNamespace(
        camera_calibrations=OrderedDict([("front", calibration)]),
        rig_trajectories=[trajectory],
    )
    geometry = SimpleNamespace(sensor_ids_to_frame_range={"front": range(3)})
    geometry.to = lambda *, device: geometry
    geometry_type = SimpleNamespace(from_rig_trajectories=lambda _: geometry)
    monkeypatch.setattr(render_video, "CameraFreePoseViewGeometry", geometry_type)
    monkeypatch.setattr(render_video, "require_ffmpeg", lambda: "/usr/bin/ffmpeg")

    frame_context_calls: list[int] = []

    def fake_frame_context(_geometry, *, unique_sensor_idx, unique_frame_idx, device):
        assert _geometry is geometry
        assert unique_sensor_idx == 7
        assert device == torch.device("cpu")
        frame_context_calls.append(unique_frame_idx)
        return SimpleNamespace(frame=unique_frame_idx)

    monkeypatch.setattr(render_video, "_frame_context", fake_frame_context)

    def fake_render(_primitive, context):
        value = 0.25 + context.frame * 0.1
        return (
            torch.full((2, 4, 3), value),
            torch.full((2, 4), 0.25),
            torch.full((2, 4, 3), 0.2),
        )

    monkeypatch.setattr(render_video, "render_composited_frame", fake_render)

    popen_calls: list[list[str]] = []
    popen_kwargs: list[dict[str, object]] = []
    processes: list[_FakeProcess] = []

    class _CapturePipe:
        def __init__(self) -> None:
            self.data = bytearray()
            self.closed = False

        def write(self, value: bytes) -> int:
            self.data.extend(value)
            return len(value)

        def close(self) -> None:
            self.closed = True

    class _FakeProcess:
        def __init__(self, command: list[str], **kwargs: object) -> None:
            popen_calls.append(command)
            popen_kwargs.append(kwargs)
            self.stdin = _CapturePipe()
            self.returncode: int | None = None
            processes.append(self)

        def poll(self) -> int | None:
            return self.returncode

        def wait(self) -> int:
            self.returncode = 0
            Path(popen_calls[-1][-1]).write_bytes(b"encoded-mp4")
            return 0

        def kill(self) -> None:
            self.returncode = -9

    monkeypatch.setattr(render_video.subprocess, "Popen", _FakeProcess)

    primitive = SimpleNamespace(device=lambda: torch.device("cpu"))
    output = tmp_path / "scene.render.mp4"
    stats = render_video.render_reference_video(primitive, rig, output)

    assert frame_context_calls == [0, 1, 2]
    assert len(popen_calls) == 1
    assert "4x2" in popen_calls[0]
    partial_path = Path(popen_calls[0][-1])
    assert partial_path.parent == output.parent
    assert partial_path.name.endswith(".partial.mp4")
    assert partial_path != output.with_suffix(".partial.mp4")
    assert popen_kwargs[0]["stderr"] is not render_video.subprocess.PIPE
    assert len(processes[0].stdin.data) == 3 * 4 * 2 * 3
    assert output.read_bytes() == b"encoded-mp4"
    assert not partial_path.exists()
    assert stats.path == output
    assert stats.frame_count == 3
    assert stats.fps == pytest.approx(1_000_000 / 33_333)
    assert stats.duration_seconds == pytest.approx(3 / stats.fps)
    assert stats.width == 4
    assert stats.height == 2
    assert stats.background_fraction == pytest.approx(0.75)
    assert stats.sky_contribution_mean == pytest.approx(0.15)


def test_render_reference_video_reports_ffmpeg_stderr_on_broken_pipe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    timestamps = torch.tensor([[0, 100], [33_333, 33_433]], dtype=torch.int64)
    calibration = SimpleNamespace(
        sequence_id="context-main",
        unique_sensor_idx=7,
        camera_model_parameters=SimpleNamespace(resolution=(4, 2)),
    )
    trajectory = SimpleNamespace(
        sequence_id="context-main",
        cameras_frame_timestamps_us={"front": timestamps},
    )
    rig = SimpleNamespace(
        camera_calibrations=OrderedDict([("front", calibration)]),
        rig_trajectories=[trajectory],
    )
    geometry = SimpleNamespace(sensor_ids_to_frame_range={"front": range(2)})
    geometry.to = lambda *, device: geometry
    geometry_type = SimpleNamespace(from_rig_trajectories=lambda _: geometry)
    monkeypatch.setattr(render_video, "CameraFreePoseViewGeometry", geometry_type)
    monkeypatch.setattr(render_video, "require_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(render_video, "_frame_context", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(
        render_video,
        "render_composited_frame",
        lambda *args: (
            torch.zeros(2, 4, 3),
            torch.zeros(2, 4),
            torch.zeros(2, 4, 3),
        ),
    )

    class _BrokenPipe:
        closed = False

        def write(self, value: bytes) -> int:
            del value
            raise BrokenPipeError("ffmpeg closed stdin")

        def close(self) -> None:
            self.closed = True

    class _FailedProcess:
        def __init__(self, command: list[str], **kwargs: object) -> None:
            del command
            self.stdin = _BrokenPipe()
            self.returncode = 1
            stderr_log = kwargs["stderr"]
            stderr_log.write(b"Unknown encoder 'libx264'\n")
            stderr_log.flush()

        def poll(self) -> int:
            return self.returncode

        def wait(self) -> int:
            return self.returncode

        def kill(self) -> None:
            raise AssertionError("Exited ffmpeg process must not be killed")

    monkeypatch.setattr(render_video.subprocess, "Popen", _FailedProcess)

    output = tmp_path / "scene.render.mp4"
    with pytest.raises(RuntimeError, match="status 1: Unknown encoder 'libx264'") as exc_info:
        render_video.render_reference_video(SimpleNamespace(device=lambda: torch.device("cpu")), rig, output)

    assert isinstance(exc_info.value.__cause__, BrokenPipeError)
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []
