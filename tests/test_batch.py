"""Branch-coverage tests for nre.utils.batch.

The module wraps a substantial dependency graph (slang kernels, ncore camera
models, lt.SE3-shaped tensors). We stub the kernel + ncore surface via
``sys.modules`` and exercise the testable dataclass logic with plain torch
tensors.

Coverage focus: ``generate_grid_2d_indices`` + the dataclass __post_init__ /
``collate_fn`` / ``to`` / ``__getitem__`` paths for ``RenderingData``,
``FrameMeta``, ``CameraFrameLabels``, ``LidarFrameLabels``, ``DataBatch.{Camera,Lidar}``,
``DataBatch``, ``RenderingBatch``, ``DataAndRenderingBatch``.

The CameraFreePoseViewGeometry class (which uses ncore.PoseInterpolator +
slang image_points_to_world_rays_shutter_pose at runtime) is left for
end-to-end coverage.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def stubbed_batch(monkeypatch):
    # lietorch (nre.utils.types pulls it).
    lietorch_mod = types.ModuleType("lietorch")

    class _FakeSE3:
        pass

    lietorch_mod.SE3 = _FakeSE3
    monkeypatch.setitem(sys.modules, "lietorch", lietorch_mod)

    # libs.geometry.kernels.pose
    libs_mod = types.ModuleType("libs")
    geometry_pkg = types.ModuleType("libs.geometry")
    geometry_kernels_pkg = types.ModuleType("libs.geometry.kernels")
    geometry_pose_mod = types.ModuleType("libs.geometry.kernels.pose")
    geometry_pose_mod.se3pose_from_matrix = lambda M: M  # passthrough
    geometry_kernels_pkg.pose = geometry_pose_mod
    geometry_pkg.kernels = geometry_kernels_pkg

    # libs.sensors.kernels.cameras + .common + .pose_calib + .cameras.parameters
    sensors_pkg = types.ModuleType("libs.sensors")
    sensors_kernels_pkg = types.ModuleType("libs.sensors.kernels")
    cameras_pkg = types.ModuleType("libs.sensors.kernels.cameras")
    parameters_mod = types.ModuleType("libs.sensors.kernels.cameras.parameters")
    common_mod = types.ModuleType("libs.sensors.kernels.common")
    pose_calib_mod = types.ModuleType("libs.sensors.kernels.pose_calib")

    pose_calib_mod.compute_poses_and_timestamps = lambda *a, **k: (
        torch.zeros(1, 2, 4, 4),
        torch.zeros(1, 2),
    )

    class CameraProjection:
        pass

    class ExternalDistortion:
        pass

    class NoExternalDistortion(ExternalDistortion):
        pass

    class _Projection(CameraProjection):
        @classmethod
        def from_components(cls, **kwargs):
            return cls()

    parameters_mod.CameraProjection = CameraProjection
    parameters_mod.ExternalDistortion = ExternalDistortion
    parameters_mod.NoExternalDistortion = NoExternalDistortion
    parameters_mod.OpenCVPinholeProjection = type("P", (_Projection,), {})
    parameters_mod.OpenCVFisheyeProjection = type("Q", (_Projection,), {})
    parameters_mod.FThetaProjection = type("R", (_Projection,), {})
    parameters_mod.BivariateWindshieldDistortion = type("BWD", (ExternalDistortion,), {})

    from enum import Enum

    parameters_mod.FThetaPolynomialType = Enum("FThetaPT", ["FORWARD", "BACKWARD"])
    parameters_mod.ReferencePolynomial = Enum("RefP", ["FORWARD", "BACKWARD"])
    parameters_mod.ShutterType = Enum("ShutterType", ["ROLLING", "GLOBAL"])

    common_mod.Pose = type("Pose", (), {})
    common_mod.DynamicPose = type("DynamicPose", (), {})

    cameras_pkg.parameters = parameters_mod
    cameras_pkg.image_points_to_world_rays_shutter_pose = lambda *a, **k: torch.zeros(1, 6)
    sensors_kernels_pkg.cameras = cameras_pkg
    sensors_kernels_pkg.common = common_mod
    sensors_kernels_pkg.pose_calib = pose_calib_mod
    sensors_pkg.kernels = sensors_kernels_pkg

    libs_mod.geometry = geometry_pkg
    libs_mod.sensors = sensors_pkg

    # ncore stubs.
    ncore_mod = types.ModuleType("ncore")
    ncore_data_mod = types.ModuleType("ncore.data")
    ncore_data_mod.ConcreteCameraModelParametersUnion = object
    ncore_data_mod.ConcreteLidarModelParametersUnion = object
    ncore_data_mod.FThetaCameraModelParameters = type(
        "FCP", (), {"PolynomialType": Enum("PolyType", ["ANGLE_TO_PIXELDIST"])}
    )
    ncore_data_mod.ReferencePolynomial = Enum("NcRP", ["FORWARD", "BACKWARD"])

    class _ShutterTypeNcore:
        ROLLING = type("Tag", (), {"value": 0})()

    ncore_data_mod.ShutterType = _ShutterTypeNcore
    ncore_mod.data = ncore_data_mod

    impl_mod = types.ModuleType("ncore.impl")
    common_ncore_mod = types.ModuleType("ncore.impl.common")
    transformations_mod = types.ModuleType("ncore.impl.common.transformations")

    class _PoseInterpolator:
        def __init__(self, T_rig_world, ts):
            pass

        def interpolate_to_timestamps(self, ts):
            import numpy as np

            return np.tile(np.eye(4), (1, 2, 1, 1))

    transformations_mod.PoseInterpolator = _PoseInterpolator
    common_ncore_mod.transformations = transformations_mod
    impl_mod.common = common_ncore_mod
    ncore_mod.impl = impl_mod

    sensors_ncore = types.ModuleType("ncore.sensors")

    class CameraModel:
        @classmethod
        def from_parameters(cls, params, device=None, dtype=None):
            obj = cls()
            obj.resolution = torch.tensor([4, 3])
            return obj

        def pixels_to_camera_rays(self, indices):
            n = indices.shape[0]
            return torch.zeros(n, 3)

        def get_parameters(self):
            return self

    sensors_ncore.CameraModel = CameraModel
    sensors_ncore.OpenCVPinholeCameraModel = type("OPC", (CameraModel,), {})
    sensors_ncore.OpenCVFisheyeCameraModel = type("OFC", (CameraModel,), {})
    sensors_ncore.FThetaCameraModel = type("FTC", (CameraModel,), {})
    sensors_ncore.BivariateWindshieldModel = type("BWM", (), {})
    ncore_mod.sensors = sensors_ncore

    for name, mod in [
        ("lietorch", lietorch_mod),
        ("libs", libs_mod),
        ("libs.geometry", geometry_pkg),
        ("libs.geometry.kernels", geometry_kernels_pkg),
        ("libs.geometry.kernels.pose", geometry_pose_mod),
        ("libs.sensors", sensors_pkg),
        ("libs.sensors.kernels", sensors_kernels_pkg),
        ("libs.sensors.kernels.cameras", cameras_pkg),
        ("libs.sensors.kernels.cameras.parameters", parameters_mod),
        ("libs.sensors.kernels.common", common_mod),
        ("libs.sensors.kernels.pose_calib", pose_calib_mod),
        ("ncore", ncore_mod),
        ("ncore.data", ncore_data_mod),
        ("ncore.impl", impl_mod),
        ("ncore.impl.common", common_ncore_mod),
        ("ncore.impl.common.transformations", transformations_mod),
        ("ncore.sensors", sensors_ncore),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    for cached in (
        "instant_nurec.utils.batch",
        "instant_nurec.utils.types",
        "instant_nurec.utils.sensors",
        "instant_nurec.utils.sensors.sensors",
        "instant_nurec.utils.sensors.ncore_sensors_converters",
    ):
        monkeypatch.delitem(sys.modules, cached, raising=False)

    import importlib

    mod = importlib.import_module("instant_nurec.utils.batch")
    return mod


# ---------------------------------------------------------------------------
# generate_grid_2d_indices
# ---------------------------------------------------------------------------


def test_generate_grid_2d_indices_shape_and_values(stubbed_batch):
    mod = stubbed_batch
    out = mod.generate_grid_2d_indices((3, 2))  # w=3, h=2
    assert out.shape == (6, 2)
    assert out.dtype == torch.int16
    # Default xy ordering: (x, y) for (0,0),(1,0),(2,0),(0,1),(1,1),(2,1) — exact order is
    # determined by torch.meshgrid + flatten; verify the unique-pair set.
    pairs = {tuple(p.tolist()) for p in out}
    assert pairs == {(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)}


# ---------------------------------------------------------------------------
# RenderingData
# ---------------------------------------------------------------------------


def _make_rendering_data(mod, *, B=1, H=4, W=5):
    return mod.RenderingData(
        rays=torch.zeros(B, H, W, 6),
        sensor_model_parameters=[None] * B,
        poses_tquat_startend=torch.zeros(B, 2, 7),
        timestamps_startend_us=torch.zeros(B, 2, dtype=torch.int64),
        timestamps_startend_us_cpu=torch.zeros(B, 2, dtype=torch.int64),
    )


def test_rendering_data_b_property(stubbed_batch):
    rd = _make_rendering_data(stubbed_batch, B=3)
    assert rd.b == 3


def test_rendering_data_post_init_rejects_wrong_rays_shape(stubbed_batch):
    mod = stubbed_batch
    with pytest.raises(AssertionError, match="Rays must be a 4D tensor"):
        mod.RenderingData(
            rays=torch.zeros(1, 4, 5),  # 3D, not 4D
            sensor_model_parameters=[None],
            poses_tquat_startend=torch.zeros(1, 2, 7),
            timestamps_startend_us=torch.zeros(1, 2, dtype=torch.int64),
            timestamps_startend_us_cpu=torch.zeros(1, 2, dtype=torch.int64),
        )


def test_rendering_data_post_init_rejects_wrong_param_count(stubbed_batch):
    mod = stubbed_batch
    with pytest.raises(AssertionError, match="length B"):
        mod.RenderingData(
            rays=torch.zeros(2, 4, 5, 6),
            sensor_model_parameters=[None],  # 1, but B=2
            poses_tquat_startend=torch.zeros(2, 2, 7),
            timestamps_startend_us=torch.zeros(2, 2, dtype=torch.int64),
            timestamps_startend_us_cpu=torch.zeros(2, 2, dtype=torch.int64),
        )


def test_rendering_data_post_init_rejects_wrong_pose_shape(stubbed_batch):
    mod = stubbed_batch
    with pytest.raises(AssertionError, match="Poses must be"):
        mod.RenderingData(
            rays=torch.zeros(1, 4, 5, 6),
            sensor_model_parameters=[None],
            poses_tquat_startend=torch.zeros(1, 3, 7),  # wrong middle dim
            timestamps_startend_us=torch.zeros(1, 2, dtype=torch.int64),
            timestamps_startend_us_cpu=torch.zeros(1, 2, dtype=torch.int64),
        )


def test_rendering_data_post_init_rejects_wrong_ts_shape(stubbed_batch):
    mod = stubbed_batch
    with pytest.raises(AssertionError, match="Timestamps must be"):
        mod.RenderingData(
            rays=torch.zeros(1, 4, 5, 6),
            sensor_model_parameters=[None],
            poses_tquat_startend=torch.zeros(1, 2, 7),
            timestamps_startend_us=torch.zeros(1, 3, dtype=torch.int64),
            timestamps_startend_us_cpu=torch.zeros(1, 2, dtype=torch.int64),
        )


def test_rendering_data_post_init_rejects_wrong_ts_cpu_shape(stubbed_batch):
    mod = stubbed_batch
    with pytest.raises(AssertionError, match="CPU timestamps must be"):
        mod.RenderingData(
            rays=torch.zeros(1, 4, 5, 6),
            sensor_model_parameters=[None],
            poses_tquat_startend=torch.zeros(1, 2, 7),
            timestamps_startend_us=torch.zeros(1, 2, dtype=torch.int64),
            timestamps_startend_us_cpu=torch.zeros(1, 3, dtype=torch.int64),
        )


def test_rendering_data_post_init_rejects_wrong_rays_ts_shape(stubbed_batch):
    mod = stubbed_batch
    with pytest.raises(AssertionError, match="Rays timestamps"):
        mod.RenderingData(
            rays=torch.zeros(1, 4, 5, 6),
            sensor_model_parameters=[None],
            poses_tquat_startend=torch.zeros(1, 2, 7),
            timestamps_startend_us=torch.zeros(1, 2, dtype=torch.int64),
            timestamps_startend_us_cpu=torch.zeros(1, 2, dtype=torch.int64),
            rays_timestamps_us=torch.zeros(1, 4, 5, 2),  # last dim should be 1
        )


def test_rendering_data_post_init_rejects_wrong_depth_scale_shape(stubbed_batch):
    mod = stubbed_batch
    with pytest.raises(AssertionError, match="Depth to distance"):
        mod.RenderingData(
            rays=torch.zeros(1, 4, 5, 6),
            sensor_model_parameters=[None],
            poses_tquat_startend=torch.zeros(1, 2, 7),
            timestamps_startend_us=torch.zeros(1, 2, dtype=torch.int64),
            timestamps_startend_us_cpu=torch.zeros(1, 2, dtype=torch.int64),
            _distance_to_depth_scale=torch.zeros(1, 3, 5, 1),  # wrong height
        )


def test_rendering_data_to_moves_tensors(stubbed_batch):
    rd = _make_rendering_data(stubbed_batch)
    moved = rd.to(torch.float64)
    assert moved.rays.dtype == torch.float64
    assert moved.poses_tquat_startend.dtype == torch.float64
    # CPU copy stays the same dtype (it's int64).
    assert moved.timestamps_startend_us_cpu.dtype == torch.int64


def test_rendering_data_to_with_optional_fields(stubbed_batch):
    mod = stubbed_batch
    rd = mod.RenderingData(
        rays=torch.zeros(1, 4, 5, 6),
        sensor_model_parameters=[None],
        poses_tquat_startend=torch.zeros(1, 2, 7),
        timestamps_startend_us=torch.zeros(1, 2, dtype=torch.int64),
        timestamps_startend_us_cpu=torch.zeros(1, 2, dtype=torch.int64),
        rays_timestamps_us=torch.zeros(1, 4, 5, 1),
        _distance_to_depth_scale=torch.zeros(1, 4, 5, 1),
    )
    moved = rd.to(torch.float64)
    assert moved.rays_timestamps_us.dtype == torch.float64
    assert moved._distance_to_depth_scale.dtype == torch.float64


def test_rendering_data_getitem_int(stubbed_batch):
    rd = _make_rendering_data(stubbed_batch, B=3)
    sub = rd[1]
    assert sub.b == 1


def test_rendering_data_getitem_slice(stubbed_batch):
    rd = _make_rendering_data(stubbed_batch, B=4)
    sub = rd[1:3]
    assert sub.b == 2


def test_rendering_data_getitem_with_optional_fields(stubbed_batch):
    mod = stubbed_batch
    rd = mod.RenderingData(
        rays=torch.zeros(3, 4, 5, 6),
        sensor_model_parameters=[None] * 3,
        poses_tquat_startend=torch.zeros(3, 2, 7),
        timestamps_startend_us=torch.zeros(3, 2, dtype=torch.int64),
        timestamps_startend_us_cpu=torch.zeros(3, 2, dtype=torch.int64),
        rays_timestamps_us=torch.zeros(3, 4, 5, 1),
        _distance_to_depth_scale=torch.zeros(3, 4, 5, 1),
    )
    sub = rd[0:2]
    assert sub.rays_timestamps_us.shape[0] == 2
    assert sub._distance_to_depth_scale.shape[0] == 2


def test_rendering_data_collate_fn_handles_optional_fields(stubbed_batch):
    rd1 = _make_rendering_data(stubbed_batch, B=1)
    rd2 = _make_rendering_data(stubbed_batch, B=2)
    out = stubbed_batch.RenderingData.collate_fn([rd1, rd2])
    assert out.b == 3
    assert out.rays_timestamps_us is None
    assert out._distance_to_depth_scale is None


def test_rendering_data_collate_fn_with_all_optional_fields_present(stubbed_batch):
    mod = stubbed_batch
    rd1 = mod.RenderingData(
        rays=torch.zeros(1, 4, 5, 6),
        sensor_model_parameters=[None],
        poses_tquat_startend=torch.zeros(1, 2, 7),
        timestamps_startend_us=torch.zeros(1, 2, dtype=torch.int64),
        timestamps_startend_us_cpu=torch.zeros(1, 2, dtype=torch.int64),
        rays_timestamps_us=torch.zeros(1, 4, 5, 1),
        _distance_to_depth_scale=torch.zeros(1, 4, 5, 1),
    )
    rd2 = mod.RenderingData(
        rays=torch.zeros(1, 4, 5, 6),
        sensor_model_parameters=[None],
        poses_tquat_startend=torch.zeros(1, 2, 7),
        timestamps_startend_us=torch.zeros(1, 2, dtype=torch.int64),
        timestamps_startend_us_cpu=torch.zeros(1, 2, dtype=torch.int64),
        rays_timestamps_us=torch.zeros(1, 4, 5, 1),
        _distance_to_depth_scale=torch.zeros(1, 4, 5, 1),
    )
    out = mod.RenderingData.collate_fn([rd1, rd2])
    assert out.rays_timestamps_us is not None
    assert out._distance_to_depth_scale is not None


def test_rendering_data_distance_to_depth_scale_lazy_compute(stubbed_batch):
    """First access computes via CameraModel.from_parameters; second returns cached."""
    rd = _make_rendering_data(stubbed_batch, B=1, H=3, W=4)
    out1 = rd.distance_to_depth_scale
    assert out1.shape == (1, 3, 4, 1)
    # Cached on subsequent access.
    out2 = rd.distance_to_depth_scale
    assert out1 is out2


def test_rendering_data_distance_to_depth_scale_returns_precomputed(stubbed_batch):
    mod = stubbed_batch
    pre = torch.ones(1, 4, 5, 1)
    rd = mod.RenderingData(
        rays=torch.zeros(1, 4, 5, 6),
        sensor_model_parameters=[None],
        poses_tquat_startend=torch.zeros(1, 2, 7),
        timestamps_startend_us=torch.zeros(1, 2, dtype=torch.int64),
        timestamps_startend_us_cpu=torch.zeros(1, 2, dtype=torch.int64),
        _distance_to_depth_scale=pre,
    )
    out = rd.distance_to_depth_scale
    assert out is pre


# ---------------------------------------------------------------------------
# FrameMeta
# ---------------------------------------------------------------------------


def test_frame_meta_post_init_builds_tensor_and_str(stubbed_batch):
    fm = stubbed_batch.FrameMeta(unique_sensor_idx=2, unique_frame_idx=5)
    assert fm.unique_sensor_idx_str == "2"
    assert fm.unique_frame_idx_tensor is not None
    assert int(fm.unique_frame_idx_tensor[0].item()) == 5


def test_frame_meta_post_init_skips_tensor_for_minus_one(stubbed_batch):
    """unique_frame_idx == -1 → tensor is None (sentinel for 'no frame')."""
    fm = stubbed_batch.FrameMeta(unique_sensor_idx=0, unique_frame_idx=-1)
    assert fm.unique_frame_idx_tensor is None


def test_frame_meta_to_preserves_none_tensor(stubbed_batch):
    fm = stubbed_batch.FrameMeta(unique_sensor_idx=0, unique_frame_idx=-1)
    moved = fm.to(torch.float32)
    assert moved.unique_frame_idx_tensor is None


def test_frame_meta_to_moves_tensor(stubbed_batch):
    fm = stubbed_batch.FrameMeta(unique_sensor_idx=0, unique_frame_idx=3)
    moved = fm.to(torch.int64)
    assert moved.unique_frame_idx_tensor.dtype == torch.int64


def test_frame_meta_collate_fn_returns_list_of_moved_metas(stubbed_batch):
    fm1 = stubbed_batch.FrameMeta(unique_sensor_idx=0, unique_frame_idx=1)
    fm2 = stubbed_batch.FrameMeta(unique_sensor_idx=1, unique_frame_idx=2)
    out = stubbed_batch.FrameMeta.collate_fn([fm1, fm2])
    assert isinstance(out, list)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# CameraFrameLabels
# ---------------------------------------------------------------------------


def _make_cam_labels(stubbed_batch, **overrides):
    base = dict(
        flags=torch.zeros(1, 4, 5, 1, dtype=torch.int32),
        rgb=torch.zeros(1, 4, 5, 3, dtype=torch.float32),
        metric_distance=torch.zeros(1, 4, 5, 1, dtype=torch.float32),
        normals=torch.zeros(1, 4, 5, 3, dtype=torch.float32),
    )
    base.update(overrides)
    return stubbed_batch.CameraFrameLabels(**base)


def test_cam_labels_post_init_accepts_valid_inputs(stubbed_batch):
    _ = _make_cam_labels(stubbed_batch)


def test_cam_labels_post_init_rejects_bad_flags_shape(stubbed_batch):
    mod = stubbed_batch
    with pytest.raises(AssertionError, match="Flags must be a 4D"):
        mod.CameraFrameLabels(flags=torch.zeros(1, 4, 5, dtype=torch.int32))


def test_cam_labels_post_init_rejects_wrong_flags_dtype(stubbed_batch):
    mod = stubbed_batch
    with pytest.raises(AssertionError, match="Flags must be a int32"):
        mod.CameraFrameLabels(flags=torch.zeros(1, 4, 5, 1, dtype=torch.float32))


def test_cam_labels_post_init_rejects_bad_rgb_shape(stubbed_batch):
    mod = stubbed_batch
    with pytest.raises(AssertionError, match="RGB must be"):
        mod.CameraFrameLabels(rgb=torch.zeros(1, 4, 5, 4, dtype=torch.float32))


def test_cam_labels_post_init_rejects_wrong_rgb_dtype(stubbed_batch):
    mod = stubbed_batch
    with pytest.raises(AssertionError, match="RGB must be a float32"):
        mod.CameraFrameLabels(rgb=torch.zeros(1, 4, 5, 3, dtype=torch.float64))


def test_cam_labels_post_init_rejects_bad_metric_distance_shape(stubbed_batch):
    mod = stubbed_batch
    with pytest.raises(AssertionError, match="Metric distance must be"):
        mod.CameraFrameLabels(metric_distance=torch.zeros(1, 4, 5, 2, dtype=torch.float32))


def test_cam_labels_post_init_rejects_wrong_metric_distance_dtype(stubbed_batch):
    mod = stubbed_batch
    with pytest.raises(AssertionError, match="Metric distance must be a float32"):
        mod.CameraFrameLabels(metric_distance=torch.zeros(1, 4, 5, 1, dtype=torch.float64))


def test_cam_labels_post_init_rejects_bad_normals_shape(stubbed_batch):
    mod = stubbed_batch
    with pytest.raises(AssertionError, match="Normals must be"):
        mod.CameraFrameLabels(normals=torch.zeros(1, 4, 5, 4, dtype=torch.float32))


def test_cam_labels_post_init_rejects_wrong_normals_dtype(stubbed_batch):
    mod = stubbed_batch
    with pytest.raises(AssertionError, match="Normals must be a float32"):
        mod.CameraFrameLabels(normals=torch.zeros(1, 4, 5, 3, dtype=torch.float64))


def test_cam_labels_get_mask_flags_all(stubbed_batch):
    """Mask is True iff all bits of `flags.value` are set in self.flags."""
    mod = stubbed_batch
    flags_tensor = torch.tensor([[[[0b110]]]], dtype=torch.int32)  # has bits 1 and 2

    class _F:
        value = 0b010

    cl = mod.CameraFrameLabels(flags=flags_tensor)
    mask = cl.get_mask_flags_all(_F())
    assert mask.all()


def test_cam_labels_get_mask_flags_all_requires_flags(stubbed_batch):
    mod = stubbed_batch
    cl = mod.CameraFrameLabels()  # no flags

    class _F:
        value = 0b1

    with pytest.raises(AssertionError, match="flags are required"):
        cl.get_mask_flags_all(_F())


def test_cam_labels_to_passes_through_optional_fields(stubbed_batch):
    """All four fields support .to() with None passthrough."""
    cl = stubbed_batch.CameraFrameLabels()  # all None
    moved = cl.to(torch.float32)
    assert moved.flags is None and moved.rgb is None
    assert moved.metric_distance is None and moved.normals is None


def test_cam_labels_to_moves_tensors(stubbed_batch):
    cl = _make_cam_labels(stubbed_batch)
    moved = cl.to(torch.device("cpu"))
    # .to(device) with no dtype change — types preserved.
    assert moved.rgb.dtype == torch.float32
    assert moved.normals.dtype == torch.float32


def test_cam_labels_getitem_int_and_slice(stubbed_batch):
    cl = stubbed_batch.CameraFrameLabels(rgb=torch.zeros(3, 4, 5, 3, dtype=torch.float32))
    sub_int = cl[1]
    assert sub_int.rgb.shape[0] == 1
    sub_slice = cl[0:2]
    assert sub_slice.rgb.shape[0] == 2


def test_cam_labels_getitem_passes_through_none(stubbed_batch):
    cl = stubbed_batch.CameraFrameLabels()
    sub = cl[0]
    assert sub.flags is None
    assert sub.rgb is None
    assert sub.metric_distance is None
    assert sub.normals is None


def test_cam_labels_collate_fn_with_all_none(stubbed_batch):
    a = stubbed_batch.CameraFrameLabels()
    b = stubbed_batch.CameraFrameLabels()
    out = stubbed_batch.CameraFrameLabels.collate_fn([a, b])
    assert out.flags is None and out.rgb is None


def test_cam_labels_collate_fn_zero_fills_missing_metric_distance(stubbed_batch):
    """If at least one sample has metric_distance and others don't, the
    others get filled with zeros_like instead of being treated as None."""
    a = stubbed_batch.CameraFrameLabels(
        metric_distance=torch.ones(1, 4, 5, 1, dtype=torch.float32)
    )
    b = stubbed_batch.CameraFrameLabels()  # no metric_distance
    out = stubbed_batch.CameraFrameLabels.collate_fn([a, b])
    assert out.metric_distance is not None
    assert out.metric_distance.shape == (2, 4, 5, 1)


# ---------------------------------------------------------------------------
# LidarFrameLabels
# ---------------------------------------------------------------------------


def test_lidar_labels_post_init_accepts_valid_inputs(stubbed_batch):
    _ = stubbed_batch.LidarFrameLabels(
        flags=torch.zeros(1, 4, 5, 1, dtype=torch.int32),
        distance=torch.zeros(1, 4, 5, 1, dtype=torch.float32),
    )


def test_lidar_labels_post_init_rejects_bad_flags_shape(stubbed_batch):
    with pytest.raises(AssertionError, match="Flags must be a 4D"):
        stubbed_batch.LidarFrameLabels(flags=torch.zeros(1, 4, 5, dtype=torch.int32))


def test_lidar_labels_post_init_rejects_wrong_flags_dtype(stubbed_batch):
    with pytest.raises(AssertionError, match="Flags must be a int32"):
        stubbed_batch.LidarFrameLabels(flags=torch.zeros(1, 4, 5, 1, dtype=torch.float32))


def test_lidar_labels_post_init_rejects_bad_distance_shape(stubbed_batch):
    with pytest.raises(AssertionError, match="Distance must be"):
        stubbed_batch.LidarFrameLabels(distance=torch.zeros(1, 4, 5, 2, dtype=torch.float32))


def test_lidar_labels_post_init_rejects_wrong_distance_dtype(stubbed_batch):
    with pytest.raises(AssertionError, match="Distance must be a float32"):
        stubbed_batch.LidarFrameLabels(distance=torch.zeros(1, 4, 5, 1, dtype=torch.float64))


def test_lidar_labels_to_moves_tensors(stubbed_batch):
    ll = stubbed_batch.LidarFrameLabels(
        flags=torch.zeros(1, 4, 5, 1, dtype=torch.int32),
        distance=torch.zeros(1, 4, 5, 1, dtype=torch.float32),
    )
    moved = ll.to(torch.device("cpu"))
    # .to(device) with no dtype change — types preserved.
    assert moved.distance.dtype == torch.float32


def test_lidar_labels_to_passes_through_none(stubbed_batch):
    moved = stubbed_batch.LidarFrameLabels().to(torch.float32)
    assert moved.flags is None and moved.distance is None


def test_lidar_labels_getitem(stubbed_batch):
    ll = stubbed_batch.LidarFrameLabels(distance=torch.zeros(3, 4, 5, 1, dtype=torch.float32))
    assert ll[0].distance.shape[0] == 1
    assert ll[0:2].distance.shape[0] == 2


def test_lidar_labels_getitem_with_none_fields(stubbed_batch):
    sub = stubbed_batch.LidarFrameLabels()[0]
    assert sub.flags is None and sub.distance is None


def test_lidar_labels_collate_fn(stubbed_batch):
    a = stubbed_batch.LidarFrameLabels(distance=torch.zeros(1, 4, 5, 1, dtype=torch.float32))
    b = stubbed_batch.LidarFrameLabels(distance=torch.ones(1, 4, 5, 1, dtype=torch.float32))
    out = stubbed_batch.LidarFrameLabels.collate_fn([a, b])
    assert out.distance.shape == (2, 4, 5, 1)


# ---------------------------------------------------------------------------
# DataBatch (Camera, Lidar) + composition
# ---------------------------------------------------------------------------


def test_databatch_camera_collate_to_getitem(stubbed_batch):
    mod = stubbed_batch
    fm = mod.FrameMeta(unique_sensor_idx=0, unique_frame_idx=0)
    cam_labels = _make_cam_labels(stubbed_batch)
    cam = mod.DataBatch.Camera(meta=[fm], labels=cam_labels)
    assert cam.b == 1

    cam2 = mod.DataBatch.Camera(meta=[fm], labels=cam_labels)
    out = mod.DataBatch.Camera.collate_fn([cam, cam2])
    assert len(out.meta) == 2

    moved = cam.to(torch.device("cpu"))
    assert isinstance(moved, mod.DataBatch.Camera)

    sub_int = cam[0]
    assert isinstance(sub_int, mod.DataBatch.Camera)
    sub_slice = cam[0:1]
    assert isinstance(sub_slice, mod.DataBatch.Camera)


def test_databatch_lidar_collate_to_getitem(stubbed_batch):
    mod = stubbed_batch
    fm = mod.FrameMeta(unique_sensor_idx=0, unique_frame_idx=0)
    ll = mod.LidarFrameLabels(distance=torch.zeros(1, 4, 5, 1, dtype=torch.float32))
    lid = mod.DataBatch.Lidar(meta=[fm], labels=ll)
    assert lid.b == 1

    lid2 = mod.DataBatch.Lidar(meta=[fm], labels=ll)
    out = mod.DataBatch.Lidar.collate_fn([lid, lid2])
    assert len(out.meta) == 2

    moved = lid.to(torch.device("cpu"))
    assert isinstance(moved, mod.DataBatch.Lidar)

    sub_int = lid[0]
    assert isinstance(sub_int, mod.DataBatch.Lidar)
    sub_slice = lid[0:1]
    assert isinstance(sub_slice, mod.DataBatch.Lidar)


def test_databatch_collate_fn_combines_camera_and_lidar(stubbed_batch):
    mod = stubbed_batch
    fm = mod.FrameMeta(unique_sensor_idx=0, unique_frame_idx=0)
    cam = mod.DataBatch.Camera(meta=[fm], labels=_make_cam_labels(stubbed_batch))
    lid = mod.DataBatch.Lidar(
        meta=[fm], labels=mod.LidarFrameLabels(distance=torch.zeros(1, 4, 5, 1, dtype=torch.float32))
    )
    db = mod.DataBatch(camera=cam, lidar=lid)
    out = mod.DataBatch.collate_fn([db, db])
    assert out.camera is not None
    assert out.lidar is not None


def test_databatch_collate_fn_drops_camera_or_lidar_when_any_missing(stubbed_batch):
    mod = stubbed_batch
    fm = mod.FrameMeta(unique_sensor_idx=0, unique_frame_idx=0)
    cam = mod.DataBatch.Camera(meta=[fm], labels=_make_cam_labels(stubbed_batch))
    db_with = mod.DataBatch(camera=cam, lidar=None)
    db_without = mod.DataBatch(camera=None, lidar=None)
    out = mod.DataBatch.collate_fn([db_with, db_without])
    assert out.camera is None
    assert out.lidar is None


def test_databatch_to_moves_subbatches(stubbed_batch):
    mod = stubbed_batch
    fm = mod.FrameMeta(unique_sensor_idx=0, unique_frame_idx=0)
    cam = mod.DataBatch.Camera(meta=[fm], labels=_make_cam_labels(stubbed_batch))
    lid = mod.DataBatch.Lidar(
        meta=[fm], labels=mod.LidarFrameLabels(distance=torch.zeros(1, 4, 5, 1, dtype=torch.float32))
    )
    db = mod.DataBatch(camera=cam, lidar=lid)
    moved = db.to(torch.device("cpu"))
    assert moved.camera is not None
    assert moved.lidar is not None


def test_databatch_to_passes_through_none(stubbed_batch):
    db = stubbed_batch.DataBatch(camera=None, lidar=None)
    moved = db.to(torch.float32)
    assert moved.camera is None and moved.lidar is None


# ---------------------------------------------------------------------------
# RenderingBatch
# ---------------------------------------------------------------------------


def test_rendering_batch_collate_combines(stubbed_batch):
    rd = _make_rendering_data(stubbed_batch)
    rb = stubbed_batch.RenderingBatch(camera=rd, lidar=rd)
    out = stubbed_batch.RenderingBatch.collate_fn([rb, rb])
    assert out.camera is not None and out.lidar is not None


def test_rendering_batch_collate_drops_when_any_missing(stubbed_batch):
    mod = stubbed_batch
    rd = _make_rendering_data(stubbed_batch)
    rb_with = mod.RenderingBatch(camera=rd, lidar=None)
    rb_without = mod.RenderingBatch(camera=None, lidar=None)
    out = mod.RenderingBatch.collate_fn([rb_with, rb_without])
    assert out.camera is None
    assert out.lidar is None


def test_rendering_batch_to_with_optional_fields(stubbed_batch):
    rd = _make_rendering_data(stubbed_batch)
    rb = stubbed_batch.RenderingBatch(camera=rd, lidar=rd)
    moved = rb.to(torch.float64)
    assert moved.camera is not None
    moved2 = stubbed_batch.RenderingBatch().to(torch.float32)
    assert moved2.camera is None and moved2.lidar is None


# ---------------------------------------------------------------------------
# DataAndRenderingBatch
# ---------------------------------------------------------------------------


def _make_data_and_rendering(stubbed_batch, *, with_rendering=True):
    mod = stubbed_batch
    fm = mod.FrameMeta(unique_sensor_idx=0, unique_frame_idx=0)
    cam = mod.DataBatch.Camera(meta=[fm], labels=_make_cam_labels(stubbed_batch))
    db = mod.DataBatch(camera=cam, lidar=None)
    rb = mod.RenderingBatch(camera=_make_rendering_data(stubbed_batch)) if with_rendering else None
    return mod.DataAndRenderingBatch(data=db, rendering=rb)


def test_data_and_rendering_collate_all_with_rendering(stubbed_batch):
    a = _make_data_and_rendering(stubbed_batch, with_rendering=True)
    b = _make_data_and_rendering(stubbed_batch, with_rendering=True)
    out = stubbed_batch.DataAndRenderingBatch.collate_fn([a, b])
    assert out.rendering is not None


def test_data_and_rendering_collate_all_without_rendering(stubbed_batch):
    a = _make_data_and_rendering(stubbed_batch, with_rendering=False)
    b = _make_data_and_rendering(stubbed_batch, with_rendering=False)
    out = stubbed_batch.DataAndRenderingBatch.collate_fn([a, b])
    assert out.rendering is None


def test_data_and_rendering_collate_rejects_mixed_rendering_state(stubbed_batch):
    a = _make_data_and_rendering(stubbed_batch, with_rendering=True)
    b = _make_data_and_rendering(stubbed_batch, with_rendering=False)
    with pytest.raises(ValueError, match="either a rendering or no rendering"):
        stubbed_batch.DataAndRenderingBatch.collate_fn([a, b])


def test_data_and_rendering_to_moves_subobjects(stubbed_batch):
    d = _make_data_and_rendering(stubbed_batch, with_rendering=True)
    moved = d.to(torch.device("cpu"))
    assert moved.rendering is not None
    d2 = _make_data_and_rendering(stubbed_batch, with_rendering=False)
    moved2 = d2.to(torch.device("cpu"))
    assert moved2.rendering is None


def test_data_and_rendering_pin_memory_walks_dataclass_tree(stubbed_batch, monkeypatch):
    """pin_memory walks dataclass fields; if a field has a callable
    pin_memory attr, it's invoked and the result replaces the field.

    The real torch.Tensor.pin_memory() requires a CUDA device. We monkey-patch
    it on the test instance so the SUT's traversal can run end-to-end on CPU."""
    d = _make_data_and_rendering(stubbed_batch, with_rendering=False)

    # Patch torch.Tensor.pin_memory to a no-op that returns the tensor itself.
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)

    out = d.pin_memory()
    assert out is d
