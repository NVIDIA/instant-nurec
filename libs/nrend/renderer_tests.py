# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os

import pytest
import torch

from python.runfiles import runfiles

from libs.nrend.renderer_test_case import RendererTestCase  # type: ignore


def _discover_test_asset_paths() -> list[str]:
    RUNFILES = runfiles.Create()
    test_assets_dir = os.path.dirname(RUNFILES.Rlocation("nrend_test_assets/somefile.msgpack"))
    test_asset_paths = [
        os.path.join(test_assets_dir, f)
        for f in os.listdir(test_assets_dir)
        if os.path.isfile(os.path.join(test_assets_dir, f)) and str(f).endswith(".msgpack")
    ]
    assert len(test_asset_paths) > 1, "Failed to load nrend test-cases"
    return test_asset_paths


def _generate_test_id(asset_path: str) -> str:
    """Generate a clean test ID from the asset path - filename without extension."""
    filename = os.path.basename(asset_path)
    return os.path.splitext(filename)[0]


test_device = torch.device("cuda:1") if torch.cuda.device_count() > 1 else torch.device("cuda:0")
default_device = torch.device("cuda:0")


def test_multi_gpu_configuration():
    """
    Verify GPU configuration

    Verify that all GPUs remain exposed to this multi-gpu test when TEST_TOTAL_GPUS has been defined to request GPU load
    balancing in our Bazel test infrastructure.
    """
    test_total_gpus = os.environ.get("TEST_TOTAL_GPUS")
    if test_total_gpus is not None:
        cuda_device_count = torch.cuda.device_count()
        assert cuda_device_count == int(test_total_gpus), (
            f"GPU configuration mismatch: TEST_TOTAL_GPUS={test_total_gpus} but "
            f"torch.cuda.device_count()={cuda_device_count}. "
            "This multi-gpu test expects all GPUs to be visible."
        )


@pytest.mark.skipif(test_device == default_device, reason="Cannot test multi-GPU correctness on a single GPU")
def test_multi_gpu_correctness():
    """
    This is a dummy test to raise a skip warning if we're running on a single GPU.

    The real body of the test is in `test_renderer` below, which makes stronger assertions when two devices are available.
    We do not parameterize test_renderer over the device to avoid running a weaker version of the same test again.
    """
    pass


@pytest.mark.skipif(torch.cuda.get_device_capability() < (8, 0), reason="NRend requires cuda CC >= 8.0 - skipping test")
@pytest.mark.parametrize("asset_path", _discover_test_asset_paths(), ids=_generate_test_id)
def test_renderer(asset_path: str):
    # set the default device explicitly
    with torch.cuda.device(test_device):
        test_case = RendererTestCase.from_file(asset_path, device=test_device)
        test_case.run()


def _find_gaussian_test_asset() -> str:
    """Find a Gaussian-based test asset (3dgut, 3dgrt, or 3dgs)."""
    asset_paths = _discover_test_asset_paths()
    # Look for particle-based model assets only (not composite models like dnsg)
    gaussian_keywords = ["3dgut", "3dgrt", "3dgs"]
    for path in asset_paths:
        if any(kw in os.path.basename(path) for kw in gaussian_keywords):
            return path
    raise RuntimeError("No Gaussian-based test asset found")


def _find_test_asset_by_name(name: str) -> str:
    """Find a test asset by exact name match."""
    asset_paths = _discover_test_asset_paths()
    for path in asset_paths:
        if os.path.basename(path) == name:
            return path
    raise RuntimeError(f"No test asset found matching '{name}'")


@pytest.mark.skipif(torch.cuda.get_device_capability() < (8, 0), reason="NRend requires cuda CC >= 8.0 - skipping test")
@pytest.mark.parametrize(
    "asset_name",
    [
        "nrend_test_3dgrt_colmap_0.2.715_250623.msgpack",
        "nrend_test_3dgut_colmap_25.8.81.msgpack",
    ],
)
def test_cumulated_weights_basic(asset_name: str):
    """
    Test that cumulated weights output is computed correctly when enabled.

    Verifies:
    - scene_data tensor is returned with correct shape (num_particles, 1)
    - scene_data contains non-zero values (some particles are hit)
    - scene_data is empty when the feature is disabled
    """
    try:
        from nrend.renderer import Renderer  # type: ignore
    except ImportError:
        from libs.nrend.renderer import Renderer  # type: ignore

    asset_path = _find_test_asset_by_name(asset_name)
    test_case = RendererTestCase.from_file(asset_path, device=default_device)

    # Enable cumulated weights in render settings
    render_settings_with_weights = dict(test_case.renderer)
    if "outputs" not in render_settings_with_weights:
        render_settings_with_weights["outputs"] = {}
    if "scene" not in render_settings_with_weights["outputs"]:
        render_settings_with_weights["outputs"]["scene"] = {}
    render_settings_with_weights["outputs"]["scene"]["enable_cumulated_weights"] = True

    # Create renderer with cumulated weights enabled
    renderer = Renderer(
        model=test_case.model,
        render_settings=render_settings_with_weights,
        track_instances_uid_map=test_case.track_instances_uid,
        log_level=Renderer.LogLevel.DEBUG,
    )
    assert renderer.valid(), "Renderer should be valid"

    # Prepare sensor model if present (same as RendererTestCase.run())
    frames_sensor_model = None
    if test_case.frames_sensor_model is not None:
        try:
            frames_sensor_model = Renderer.prepare_and_cache_camera_model(
                test_case.frames_sensor_model, test_case.device
            )
        except Exception:
            lidar_tiling = render_settings_with_weights.get("tiling", {}).get("lidar", {})
            frames_sensor_model = Renderer.prepare_and_cache_lidar_model(
                test_case.frames_sensor_model,
                n_bins_elevation=lidar_tiling.get("n_bins_elevation", 16),
                max_pts_per_tile=lidar_tiling.get("tile_size_elevation", 16)
                * lidar_tiling.get("tile_size_azimuth", 16),
                device=test_case.device,
            ).parameters

    # Render
    _, _, _, _, scene_data = renderer.render(
        frame_id=test_case.frame_id,
        frame_width=test_case.frame_width,
        frame_height=test_case.frame_height,
        frame_start_timestamp=test_case.frame_start_timestamp,
        frame_end_timestamp=test_case.frame_end_timestamp,
        rays_origin=test_case.rays_origin,
        rays_direction=test_case.rays_direction,
        rays_timestamp=test_case.rays_timestamp,
        frames_sensor_model=frames_sensor_model,
        frames_sensor_ids=test_case.frames_sensor_ids,
        frames_sensor_start_pose=test_case.frames_sensor_start_pose,
        frames_sensor_end_pose=test_case.frames_sensor_end_pose,
        num_active_track_instances=test_case.num_active_track_instances,
        active_track_instances_ids=test_case.active_track_instances_ids,
        active_track_instances_start_pose=test_case.active_track_instances_start_pose,
        active_track_instances_end_pose=test_case.active_track_instances_end_pose,
    )

    # Verify scene_data is returned with expected properties
    assert scene_data.numel() > 0, "scene_data should not be empty when cumulated weights is enabled"
    assert scene_data.dim() == 2, "scene_data should be 2D (num_particles, 1)"
    assert scene_data.shape[1] == 1, "scene_data should have 1 column for cumulated weights"
    assert scene_data.dtype == torch.float32, "scene_data should be float32"
    assert scene_data.device.type == "cuda", "scene_data should be on CUDA"

    # Verify some particles were hit (non-zero weights)
    assert (scene_data > 0).any(), "Some particles should have non-zero cumulated weights"

    # Verify all weights are non-negative
    assert (scene_data >= 0).all(), "All cumulated weights should be non-negative"


@pytest.mark.skipif(torch.cuda.get_device_capability() < (8, 0), reason="NRend requires cuda CC >= 8.0 - skipping test")
@pytest.mark.parametrize(
    "asset_name",
    [
        "nrend_test_3dgut_colmap_25.8.81.msgpack",  # Only 3dgut supports visibility
    ],
)
def test_visibility_basic(asset_name: str):
    """
    Test that visibility output is computed correctly when enabled.

    Verifies:
    - scene_data tensor is returned with correct shape (num_particles, 1)
    - scene_data contains values that are 0 or 1 (binary visibility)
    - Some particles are visible (have visibility=1)
    """
    try:
        from nrend.renderer import Renderer  # type: ignore
    except ImportError:
        from libs.nrend.renderer import Renderer  # type: ignore

    asset_path = _find_test_asset_by_name(asset_name)
    test_case = RendererTestCase.from_file(asset_path, device=default_device)

    # Enable visibility in render settings
    render_settings_with_visibility = dict(test_case.renderer)
    if "outputs" not in render_settings_with_visibility:
        render_settings_with_visibility["outputs"] = {}
    if "scene" not in render_settings_with_visibility["outputs"]:
        render_settings_with_visibility["outputs"]["scene"] = {}
    render_settings_with_visibility["outputs"]["scene"]["enable_visibility"] = True

    # Create renderer with visibility enabled
    renderer = Renderer(
        model=test_case.model,
        render_settings=render_settings_with_visibility,
        track_instances_uid_map=test_case.track_instances_uid,
        log_level=Renderer.LogLevel.DEBUG,
    )
    assert renderer.valid(), "Renderer should be valid"

    # Prepare sensor model if present
    frames_sensor_model = None
    if test_case.frames_sensor_model is not None:
        try:
            frames_sensor_model = Renderer.prepare_and_cache_camera_model(
                test_case.frames_sensor_model, test_case.device
            )
        except Exception:
            lidar_tiling = render_settings_with_visibility.get("tiling", {}).get("lidar", {})
            frames_sensor_model = Renderer.prepare_and_cache_lidar_model(
                test_case.frames_sensor_model,
                n_bins_elevation=lidar_tiling.get("n_bins_elevation", 16),
                max_pts_per_tile=lidar_tiling.get("tile_size_elevation", 16)
                * lidar_tiling.get("tile_size_azimuth", 16),
                device=test_case.device,
            ).parameters

    # Render
    _, _, _, _, scene_data = renderer.render(
        frame_id=test_case.frame_id,
        frame_width=test_case.frame_width,
        frame_height=test_case.frame_height,
        frame_start_timestamp=test_case.frame_start_timestamp,
        frame_end_timestamp=test_case.frame_end_timestamp,
        rays_origin=test_case.rays_origin,
        rays_direction=test_case.rays_direction,
        rays_timestamp=test_case.rays_timestamp,
        frames_sensor_model=frames_sensor_model,
        frames_sensor_ids=test_case.frames_sensor_ids,
        frames_sensor_start_pose=test_case.frames_sensor_start_pose,
        frames_sensor_end_pose=test_case.frames_sensor_end_pose,
        num_active_track_instances=test_case.num_active_track_instances,
        active_track_instances_ids=test_case.active_track_instances_ids,
        active_track_instances_start_pose=test_case.active_track_instances_start_pose,
        active_track_instances_end_pose=test_case.active_track_instances_end_pose,
    )

    # Verify scene_data is returned with expected properties
    assert scene_data.numel() > 0, "scene_data should not be empty when visibility is enabled"
    assert scene_data.dim() == 2, "scene_data should be 2D (num_particles, 1)"
    assert scene_data.shape[1] == 1, "scene_data should have 1 column for visibility"
    assert scene_data.dtype == torch.float32, "scene_data should be float32"
    assert scene_data.device.type == "cuda", "scene_data should be on CUDA"

    # Verify visibility values are binary (0 or 1)
    assert ((scene_data == 0) | (scene_data == 1)).all(), "Visibility should be 0 or 1"

    # Verify some particles are visible
    assert (scene_data == 1).any(), "Some particles should be visible"


@pytest.mark.skipif(torch.cuda.get_device_capability() < (8, 0), reason="NRend requires cuda CC >= 8.0 - skipping test")
@pytest.mark.parametrize(
    "asset_name",
    [
        "nrend_test_3dgut_colmap_25.8.81.msgpack",  # Only 3dgut supports both features
    ],
)
def test_visibility_and_cumulated_weights(asset_name: str):
    """
    Test that both visibility and cumulated weights can be enabled together.

    Verifies:
    - scene_data tensor has shape (num_particles, 2)
    - Column 0: cumulated weights (non-negative floats)
    - Column 1: visibility (0 or 1)
    """
    try:
        from nrend.renderer import Renderer  # type: ignore
    except ImportError:
        from libs.nrend.renderer import Renderer  # type: ignore

    asset_path = _find_test_asset_by_name(asset_name)
    test_case = RendererTestCase.from_file(asset_path, device=default_device)

    # Enable both features in render settings
    render_settings_both = dict(test_case.renderer)
    if "outputs" not in render_settings_both:
        render_settings_both["outputs"] = {}
    if "scene" not in render_settings_both["outputs"]:
        render_settings_both["outputs"]["scene"] = {}
    render_settings_both["outputs"]["scene"]["enable_cumulated_weights"] = True
    render_settings_both["outputs"]["scene"]["enable_visibility"] = True

    # Create renderer with both features enabled
    renderer = Renderer(
        model=test_case.model,
        render_settings=render_settings_both,
        track_instances_uid_map=test_case.track_instances_uid,
        log_level=Renderer.LogLevel.DEBUG,
    )
    assert renderer.valid(), "Renderer should be valid"

    # Prepare sensor model if present
    frames_sensor_model = None
    if test_case.frames_sensor_model is not None:
        try:
            frames_sensor_model = Renderer.prepare_and_cache_camera_model(
                test_case.frames_sensor_model, test_case.device
            )
        except Exception:
            lidar_tiling = render_settings_both.get("tiling", {}).get("lidar", {})
            frames_sensor_model = Renderer.prepare_and_cache_lidar_model(
                test_case.frames_sensor_model,
                n_bins_elevation=lidar_tiling.get("n_bins_elevation", 16),
                max_pts_per_tile=lidar_tiling.get("tile_size_elevation", 16)
                * lidar_tiling.get("tile_size_azimuth", 16),
                device=test_case.device,
            ).parameters

    # Render
    _, _, _, _, scene_data = renderer.render(
        frame_id=test_case.frame_id,
        frame_width=test_case.frame_width,
        frame_height=test_case.frame_height,
        frame_start_timestamp=test_case.frame_start_timestamp,
        frame_end_timestamp=test_case.frame_end_timestamp,
        rays_origin=test_case.rays_origin,
        rays_direction=test_case.rays_direction,
        rays_timestamp=test_case.rays_timestamp,
        frames_sensor_model=frames_sensor_model,
        frames_sensor_ids=test_case.frames_sensor_ids,
        frames_sensor_start_pose=test_case.frames_sensor_start_pose,
        frames_sensor_end_pose=test_case.frames_sensor_end_pose,
        num_active_track_instances=test_case.num_active_track_instances,
        active_track_instances_ids=test_case.active_track_instances_ids,
        active_track_instances_start_pose=test_case.active_track_instances_start_pose,
        active_track_instances_end_pose=test_case.active_track_instances_end_pose,
    )

    # Verify scene_data has both columns
    assert scene_data.numel() > 0, "scene_data should not be empty"
    assert scene_data.dim() == 2, "scene_data should be 2D"
    assert scene_data.shape[1] == 2, "scene_data should have 2 columns (cumulated_weights, visibility)"

    # Verify cumulated weights (column 0)
    cumulated_weights = scene_data[:, 0]
    assert (cumulated_weights >= 0).all(), "Cumulated weights should be non-negative"
    assert (cumulated_weights > 0).any(), "Some particles should have non-zero cumulated weights"

    # Verify visibility (column 1)
    visibility = scene_data[:, 1]
    assert ((visibility == 0) | (visibility == 1)).all(), "Visibility should be 0 or 1"
    assert (visibility == 1).any(), "Some particles should be visible"

    # Test parse_scene_data method
    parsed = renderer.parse_scene_data(scene_data)
    assert "cumulated_weights" in parsed, "Parsed should contain cumulated_weights"
    assert "visibility" in parsed, "Parsed should contain visibility"
    assert torch.equal(parsed["cumulated_weights"], cumulated_weights), "Parsed cumulated_weights should match"
    assert torch.equal(parsed["visibility"], visibility), "Parsed visibility should match"
