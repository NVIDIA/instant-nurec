# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest
import torch

from PIL import Image

from instant_nurec.predict.export_sky import SKY_FACE_AXES, SKY_FACE_ORDER, export_sky
from instant_nurec.primitives.kelvin_primitive import (
    KelvinDynamicLayer,
    KelvinInstantNuRecPrimitive,
    KelvinStaticLayer,
)
from instant_nurec.utils.types import FrameConversion, RigTrajectories


def _primitive() -> KelvinInstantNuRecPrimitive:
    static = KelvinStaticLayer(
        positions=torch.zeros(1, 3),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        scales=torch.ones(1, 3),
        densities=torch.full((1, 1), 0.5),
        rgb=torch.full((1, 3), 0.25),
    )
    dynamic = KelvinDynamicLayer(
        rotations=torch.empty(0, 4),
        scales=torch.empty(0, 3),
        rgb=torch.empty(0, 3),
        max_densities=torch.empty(0, 1),
        keyframe_positions=torch.empty(0, 3, 3),
        keyframe_timestamps_us=torch.empty(0, 3, dtype=torch.int64),
    )
    cubemap = torch.linspace(0.0, 1.0, 6 * 4 * 4 * 3).reshape(6, 4, 4, 3)
    mask = torch.ones(6, 4, 4, 1)
    return KelvinInstantNuRecPrimitive(
        static_layer=static,
        dynamic_layers=[dynamic],
        sky_cubemap=cubemap,
        sky_cubemap_mask=mask,
        affine_matrix=torch.eye(3, 4).unsqueeze(0),
    )


def _rig(T_world_base: torch.Tensor | None = None) -> RigTrajectories:
    camera_calibrations = OrderedDict(
        [
            (
                "camera_front",
                RigTrajectories.CameraCalibration(
                    sequence_id="main",
                    unique_sensor_idx=7,
                    T_sensor_rig=torch.eye(4),
                    camera_model_parameters=object(),  # type: ignore[arg-type]
                ),
            )
        ]
    )
    return RigTrajectories(
        T_world_base=(
            torch.eye(4, dtype=torch.float64) if T_world_base is None else T_world_base
        ),
        world_to_scene=FrameConversion(matrix=np.eye(4, dtype=np.float32)),
        rig_trajectories=[],
        camera_calibrations=camera_calibrations,
    )


def test_export_sky_writes_safe_sidecar_and_preview(tmp_path) -> None:
    ply_path = tmp_path / "scene.ply"

    sidecar_path, preview_path = export_sky(_primitive(), _rig(), ply_path)

    assert sidecar_path == tmp_path / "scene.sky.npz"
    assert preview_path == tmp_path / "scene.sky.png"
    with np.load(sidecar_path, allow_pickle=False) as bundle:
        assert bundle["sky_cubemap"].shape == (6, 4, 4, 3)
        assert bundle["sky_cubemap"].dtype == np.float16
        assert bundle["sky_cubemap_mask"].shape == (6, 4, 4, 1)
        assert bundle["affine_matrix"].shape == (1, 3, 4)
        assert bundle["affine_sensor_indices"].tolist() == [7]
        assert bundle["affine_sensor_ids"].tolist() == ["camera_front"]
        assert tuple(bundle["face_order"].tolist()) == SKY_FACE_ORDER
        assert tuple(bundle["face_axes"].tolist()) == SKY_FACE_AXES
        assert bundle["uv_convention"].item() == "u_left_to_right_v_top_to_bottom"
        assert bundle["coordinate_frame"].item() == "ncore_world"
        assert bundle["sky_source"].item() == "observed_rgb_semantics"
        assert bundle["sky_observed_fraction"].item() == pytest.approx(1.0)
    with Image.open(preview_path) as preview:
        assert preview.size == (12, 8)
        assert preview.mode == "RGB"


def test_export_sky_rotates_only_environment_state_to_world(tmp_path) -> None:
    primitive = _primitive()
    T_world_base = torch.tensor(
        [
            [-1.0, 0.0, 0.0, 2.0],
            [0.0, -1.0, 0.0, 3.0],
            [0.0, 0.0, 1.0, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )

    sidecar_path, _ = export_sky(primitive, _rig(T_world_base), tmp_path / "scene.ply")

    expected = primitive.rigid_transform(T_world_base.float()).sky_cubemap
    with np.load(sidecar_path, allow_pickle=False) as bundle:
        torch.testing.assert_close(
            torch.from_numpy(bundle["sky_cubemap"].astype(np.float32)),
            expected,
            atol=5e-4,
            rtol=5e-4,
        )


def test_export_sky_preserves_canonical_values_outside_display_range(tmp_path) -> None:
    primitive = _primitive()
    primitive.sky_cubemap[0, 0, 0] = torch.tensor([-0.25, 1.25, 2.0])

    sidecar_path, preview_path = export_sky(primitive, _rig(), tmp_path / "scene.ply")

    with np.load(sidecar_path, allow_pickle=False) as bundle:
        np.testing.assert_allclose(
            bundle["sky_cubemap"][0, 0, 0].astype(np.float32),
            [-0.25, 1.25, 2.0],
        )
    with Image.open(preview_path) as preview:
        pixels = np.asarray(preview)
        assert pixels.min() == 0
        assert pixels.max() == 255
