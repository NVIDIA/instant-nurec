# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from types import SimpleNamespace

import ncore.data
import numpy as np
import pytest
import torch

from instant_nurec.config_schema.dataset import NCoreInstantNuRecDatasetConfig
from instant_nurec.datasets.instantnurec_ncore import NCoreInstantNuRecDataset
from instant_nurec.utils.types import FrameConversion, RigTrajectories


CAMERA_ID = "camera_front_wide_120fov"


@dataclass
class _FakeCameraParameters:
    name: str


class _FakeCameraSensor:
    def __init__(self, starts: np.ndarray, ends: np.ndarray) -> None:
        self.model_parameters = _FakeCameraParameters("native-ftheta")
        self.T_sensor_rig = np.array(
            [
                [1.0, 0.0, 0.0, 1.5],
                [0.0, 1.0, 0.0, -0.25],
                [0.0, 0.0, 1.0, 0.75],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self._starts = starts
        self._ends = ends
        self.timestamp_calls: list[ncore.data.FrameTimepoint] = []

    def get_frames_timestamps_us(self, timepoint: ncore.data.FrameTimepoint) -> np.ndarray:
        self.timestamp_calls.append(timepoint)
        if timepoint == ncore.data.FrameTimepoint.START:
            return self._starts
        if timepoint == ncore.data.FrameTimepoint.END:
            return self._ends
        raise AssertionError(f"Unexpected frame timepoint: {timepoint}")

    def get_frame_image_array(self, frame_idx: int) -> np.ndarray:
        raise AssertionError(f"load_full_camera_rig must not load image {frame_idx}")


def _dataset(source_path, *, cameras: list[str] | None = None) -> NCoreInstantNuRecDataset:
    cameras = [CAMERA_ID] if cameras is None else cameras
    return NCoreInstantNuRecDataset(
        NCoreInstantNuRecDatasetConfig(
            ncore_json_paths=[str(source_path)],
            context_camera_ids=cameras,
            supervision_camera_ids=cameras,
        ),
        frame_width=784,
        frame_height=448,
        n_frames_per_sample=18,
    )


def _reference_rig() -> RigTrajectories:
    poses = torch.eye(4, dtype=torch.float64).repeat(3, 1, 1)
    poses[:, 0, 3] = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64)
    world_to_scene = np.eye(4, dtype=np.float32)
    world_to_scene[:3, 3] = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    T_world_base = torch.eye(4, dtype=torch.float64)
    T_world_base[:3, 3] = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float64)
    trajectory = RigTrajectories.RigTrajectory(
        sequence_id="context-main",
        cameras_frame_timestamps_us={CAMERA_ID: torch.tensor([[10, 20]], dtype=torch.int64)},
        T_rig_worlds=poses,
        T_rig_world_timestamps_us=torch.tensor([0, 100_000, 200_000], dtype=torch.int64),
    )
    calibration = RigTrajectories.CameraCalibration(
        sequence_id="context-main",
        unique_sensor_idx=0,
        T_sensor_rig=torch.eye(4, dtype=torch.float32),
        camera_model_parameters=_FakeCameraParameters("sampled-reference"),  # type: ignore[arg-type]
    )
    return RigTrajectories(
        T_world_base=T_world_base,
        world_to_scene=FrameConversion(matrix=world_to_scene),
        rig_trajectories=[trajectory],
        camera_calibrations=OrderedDict([(CAMERA_ID, calibration)]),
    )


def test_load_full_camera_rig_keeps_all_exposures_without_images_or_rays(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "sequence.json"
    source_path.write_text("{}")
    dataset = _dataset(source_path)
    reference_rig = _reference_rig()
    sensor = _FakeCameraSensor(
        starts=np.array([25_542, 58_876, 92_210], dtype=np.int64),
        ends=np.array([56_101, 89_435, 122_769], dtype=np.int64),
    )
    transformed_parameters = _FakeCameraParameters("inference-resize-crop")
    seen: dict[str, object] = {}

    def fake_load(source, camera_ids):
        seen["source"] = source
        seen["camera_ids"] = camera_ids
        return SimpleNamespace(
            camera_sensors={camera_ids[0]: sensor},
            T_rig_worlds_with_timestamps_us=(
                np.tile(np.eye(4, dtype=np.float64), (3, 1, 1)),
                np.array([0, 100_000, 200_000], dtype=np.int64),
            ),
        )

    class _FakeSubsampler:
        def apply_camera_parameters(self, parameters):
            seen["calibration_input"] = parameters
            return transformed_parameters

    monkeypatch.setattr(dataset, "_get_loaders_and_sensors", fake_load)
    monkeypatch.setattr(dataset, "_build_camera_subsampler", _FakeSubsampler)

    full_rig = dataset.load_full_camera_rig(str(source_path), reference_rig)

    assert str(seen["source"]).endswith("sequence.json")
    assert [str(camera) for camera in seen["camera_ids"]] == [CAMERA_ID]
    assert sensor.timestamp_calls == [ncore.data.FrameTimepoint.START, ncore.data.FrameTimepoint.END]
    assert seen["calibration_input"] == _FakeCameraParameters("native-ftheta")
    assert seen["calibration_input"] is not sensor.model_parameters

    assert full_rig.T_world_base is reference_rig.T_world_base
    assert full_rig.world_to_scene is reference_rig.world_to_scene
    assert list(full_rig.camera_calibrations) == [CAMERA_ID]
    full_calibration = full_rig.camera_calibrations[CAMERA_ID]
    assert full_calibration.sequence_id == "context-main"
    assert full_calibration.unique_sensor_idx == 0
    assert full_calibration.camera_model_parameters is transformed_parameters
    torch.testing.assert_close(full_calibration.T_sensor_rig, torch.from_numpy(sensor.T_sensor_rig))

    assert len(full_rig.rig_trajectories) == 1
    full_trajectory = full_rig.rig_trajectories[0]
    reference_trajectory = reference_rig.rig_trajectories[0]
    assert full_trajectory.T_rig_worlds is reference_trajectory.T_rig_worlds
    assert full_trajectory.T_rig_world_timestamps_us is reference_trajectory.T_rig_world_timestamps_us
    torch.testing.assert_close(
        full_trajectory.cameras_frame_timestamps_us[CAMERA_ID],
        torch.tensor(
            [[25_542, 56_101], [58_876, 89_435], [92_210, 122_769]],
            dtype=torch.int64,
        ),
    )


def test_load_full_camera_rig_requires_camera_for_multicamera_dataset(tmp_path) -> None:
    source_path = tmp_path / "sequence.json"
    source_path.write_text("{}")
    dataset = _dataset(source_path, cameras=[CAMERA_ID, "camera_cross_left_120fov"])

    with pytest.raises(ValueError, match="camera_id is required"):
        dataset.load_full_camera_rig(str(source_path), _reference_rig())


def test_load_full_camera_rig_rejects_mismatched_exposure_counts(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "sequence.json"
    source_path.write_text("{}")
    dataset = _dataset(source_path)
    sensor = _FakeCameraSensor(
        starts=np.array([10, 20], dtype=np.int64),
        ends=np.array([15], dtype=np.int64),
    )

    def fake_load(source, camera_ids):
        del source
        return SimpleNamespace(
            camera_sensors={camera_ids[0]: sensor},
            T_rig_worlds_with_timestamps_us=(
                np.tile(np.eye(4, dtype=np.float64), (3, 1, 1)),
                np.array([0, 100_000, 200_000], dtype=np.int64),
            ),
        )

    monkeypatch.setattr(dataset, "_get_loaders_and_sensors", fake_load)

    with pytest.raises(ValueError, match="START/END timestamp counts differ"):
        dataset.load_full_camera_rig(str(source_path), _reference_rig())


def test_load_full_camera_rig_pads_pose_coverage_at_source_boundaries(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "sequence.json"
    source_path.write_text("{}")
    dataset = _dataset(source_path)
    reference_rig = _reference_rig()
    sensor = _FakeCameraSensor(
        starts=np.array([-50, 250_000], dtype=np.int64),
        ends=np.array([0, 250_100], dtype=np.int64),
    )

    def fake_load(source, camera_ids):
        del source
        return SimpleNamespace(
            camera_sensors={camera_ids[0]: sensor},
            T_rig_worlds_with_timestamps_us=(
                np.tile(np.eye(4, dtype=np.float64), (3, 1, 1)),
                np.array([0, 100_000, 200_000], dtype=np.int64),
            ),
        )

    monkeypatch.setattr(dataset, "_get_loaders_and_sensors", fake_load)
    monkeypatch.setattr(
        dataset,
        "_subsample_camera_model_parameters",
        lambda camera_sensor, camera_subsampler: camera_sensor.model_parameters,
    )

    full_rig = dataset.load_full_camera_rig(str(source_path), reference_rig)
    trajectory = full_rig.rig_trajectories[0]

    torch.testing.assert_close(
        trajectory.T_rig_world_timestamps_us,
        torch.tensor([-51, 0, 100_000, 200_000, 250_101], dtype=torch.int64),
    )
    torch.testing.assert_close(trajectory.T_rig_worlds[0], reference_rig.rig_trajectories[0].T_rig_worlds[0])
    torch.testing.assert_close(trajectory.T_rig_worlds[-1], reference_rig.rig_trajectories[0].T_rig_worlds[-1])


def test_load_full_camera_rig_rejects_truncated_reconstruction(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "sequence.json"
    source_path.write_text("{}")
    dataset = _dataset(source_path)
    sensor = _FakeCameraSensor(
        starts=np.array([0, 120_000_000], dtype=np.int64),
        ends=np.array([30_000, 120_030_000], dtype=np.int64),
    )

    def fake_load(source, camera_ids):
        del source
        return SimpleNamespace(
            camera_sensors={camera_ids[0]: sensor},
            T_rig_worlds_with_timestamps_us=(
                np.tile(np.eye(4, dtype=np.float64), (2, 1, 1)),
                np.array([0, 120_030_000], dtype=np.int64),
            ),
        )

    monkeypatch.setattr(dataset, "_get_loaders_and_sensors", fake_load)

    with pytest.raises(ValueError, match=r"Rerun with --max-chunks 9"):
        dataset.load_full_camera_rig(str(source_path), _reference_rig())
