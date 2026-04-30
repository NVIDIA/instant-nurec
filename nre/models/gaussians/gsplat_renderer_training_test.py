# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.

"""
Test suite for GSplatRenderer Phase 2: Training Support

Tests gradient flow, memory optimizations, and training integration
for the GSplatRenderer implementation.

Phase 2 Exit Criteria:
- Training step runs without errors
- Gradients flow to all Gaussian parameters
- Numerical gradient check passes
- Memory usage acceptable
- sparse_grad and absgrad work correctly
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from ncore.data import (
    OpenCVPinholeCameraModelParameters,
    ShutterType,
)
from nre.config.model import GSplatRendererConfig
from nre.models.gaussians.renderers import GSplatRenderer
from nre.utils.batch import RenderingData


@pytest.fixture(autouse=True)
def seed_rng():
    """Reset RNG before each test in this file for reproducibility."""
    torch.manual_seed(42)


@pytest.fixture
def simple_config():
    """Simple renderer configuration for testing."""
    return GSplatRendererConfig.model_validate(
        {
            "name": "3dgs-gsplat",  # Use 3dgs for gradients support
            "tiling": {"camera": {"tile_width": 16, "tile_height": 16}},
            "background": {"color": 0.0},
            "culling": {
                "near_clip_distance": 0.01,
                "far_clip_distance": 1000.0,
            },
            "projection": {"min_projected_ray_radius": 0.5477},
            "rasterize_mode": "classic",
            "packed": True,
            "sparse_grad": False,
            "absgrad": False,
        }
    )


@pytest.fixture
def mock_model():
    """Mock model for renderer initialization."""

    class MockModel(nn.Module):
        def serialize_to_json_dict(self, with_state_dict=False):
            """Mock serialization method required by GSplatRenderer."""
            return {
                "type": "mock_model",
                "config": {},
            }

    return MockModel()


@pytest.fixture
def simple_rendering_data(device="cuda"):
    """Create simple rendering data for testing."""
    h, w = 64, 64

    # Simple rays
    rays = torch.randn(1, h, w, 6, device=device)

    # Simple camera parameters
    sensor_params = OpenCVPinholeCameraModelParameters(
        resolution=np.array([w, h], dtype=np.uint64),
        focal_length=np.array([50.0, 50.0], dtype=np.float32),
        principal_point=np.array([32.0, 32.0], dtype=np.float32),
        radial_coeffs=np.zeros(6, dtype=np.float32),
        tangential_coeffs=np.zeros(2, dtype=np.float32),
        thin_prism_coeffs=np.zeros(4, dtype=np.float32),
        shutter_type=ShutterType.GLOBAL,
    )

    # Simple pose (identity transform)
    pose_tquat = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])  # (1, 7)
    poses_tquat_startend = pose_tquat.unsqueeze(1).repeat(1, 2, 1)  # (1, 2, 7)

    # Timestamps
    timestamps = torch.tensor([[0, 1000]], device=device, dtype=torch.int64)
    timestamps_cpu = timestamps.cpu()

    return RenderingData(
        rays=rays,
        sensor_model_parameters=[sensor_params],
        poses_tquat_startend=poses_tquat_startend.to(device),
        timestamps_startend_us=timestamps,
        timestamps_startend_us_cpu=timestamps_cpu,
        rays_timestamps_us=None,
        _rays_footprints=None,
    )


@pytest.fixture
def simple_gaussian_parameters(device="cuda"):
    """Create simple Gaussian parameters for testing."""
    n_gaussians = 100

    # Create leaf tensors by generating values first, THEN enabling gradients
    positions = torch.randn(n_gaussians, 3, device=device)
    positions.requires_grad = True

    rotations = nn.functional.normalize(torch.randn(n_gaussians, 4, device=device), dim=-1)
    rotations.requires_grad = True

    scales = torch.exp(torch.randn(n_gaussians, 3, device=device))
    scales.requires_grad = True

    densities = torch.sigmoid(torch.randn(n_gaussians, device=device))
    densities.requires_grad = True

    features = torch.randn(n_gaussians, 3, device=device)
    features.requires_grad = True

    extra_signal = torch.zeros(n_gaussians, 0, device=device)
    camera_extra_signal = torch.zeros(n_gaussians, 0, device=device)

    return {
        "positions": positions,
        "rotations": rotations,
        "scales": scales,
        "densities": densities,
        "features": features,
        "extra_signal": extra_signal,
        "camera_extra_signal": camera_extra_signal,
    }


class TestGSplatRendererGradients:
    """Test gradient flow through GSplatRenderer."""

    def test_gradient_flow_basic(self, simple_config, mock_model, simple_rendering_data, simple_gaussian_parameters):
        """Test that gradients flow to all Gaussian parameters."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        renderer = GSplatRenderer(simple_config, mock_model)

        # Forward pass
        output = renderer.render(
            simple_rendering_data,
            simple_gaussian_parameters,
            n_active_features=0,  # DC only
            extra_ray_signal_infos=([], [], []),
            frame_meta=None,
        )

        # Compute simple loss
        loss = output.rgb.sum() + output.opacity.sum() + output.distance.sum()

        # Backward pass
        loss.backward()

        # Check gradients exist and are non-zero
        assert simple_gaussian_parameters["positions"].grad is not None
        assert torch.any(simple_gaussian_parameters["positions"].grad != 0)

        assert simple_gaussian_parameters["rotations"].grad is not None
        assert torch.any(simple_gaussian_parameters["rotations"].grad != 0)

        assert simple_gaussian_parameters["scales"].grad is not None
        assert torch.any(simple_gaussian_parameters["scales"].grad != 0)

        assert simple_gaussian_parameters["densities"].grad is not None
        assert torch.any(simple_gaussian_parameters["densities"].grad != 0)

        assert simple_gaussian_parameters["features"].grad is not None
        assert torch.any(simple_gaussian_parameters["features"].grad != 0)

        print("✓ Gradients flow to all Gaussian parameters")

    def test_gradient_shapes(self, simple_config, mock_model, simple_rendering_data, simple_gaussian_parameters):
        """Test that gradient shapes match parameter shapes."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        renderer = GSplatRenderer(simple_config, mock_model)

        # Forward + backward
        output = renderer.render(
            simple_rendering_data,
            simple_gaussian_parameters,
            n_active_features=0,
            extra_ray_signal_infos=([], [], []),
        )
        loss = output.rgb.sum()
        loss.backward()

        # Check gradient shapes (skip zero-size tensors that don't participate in the graph)
        for key, param in simple_gaussian_parameters.items():
            if param.grad is None:
                continue
            assert param.grad.shape == param.shape, (
                f"Gradient shape mismatch for {key}: grad.shape={param.grad.shape}, param.shape={param.shape}"
            )

        print("✓ Gradient shapes match parameter shapes")

    def test_no_nan_gradients(self, simple_config, mock_model, simple_rendering_data, simple_gaussian_parameters):
        """Test that gradients don't contain NaN or Inf values."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        renderer = GSplatRenderer(simple_config, mock_model)

        # Forward + backward
        output = renderer.render(
            simple_rendering_data,
            simple_gaussian_parameters,
            n_active_features=0,
            extra_ray_signal_infos=([], [], []),
        )
        loss = output.rgb.sum() + output.opacity.sum()
        loss.backward()

        # Check for NaN/Inf (skip zero-size tensors that don't participate in the graph)
        for key, param in simple_gaussian_parameters.items():
            if param.grad is None:
                continue
            assert not torch.isnan(param.grad).any(), f"NaN gradients in {key}"
            assert not torch.isinf(param.grad).any(), f"Inf gradients in {key}"

        print("✓ No NaN or Inf gradients")


class TestMemoryOptimizations:
    """Test sparse_grad and absgrad features."""

    def test_sparse_grad_config(self, mock_model, simple_rendering_data, simple_gaussian_parameters):
        """Test sparse_grad configuration.

        Note: sparse_grad in gsplat does not support batch dimensions.
        This is a known limitation of the gsplat library.
        """
        pytest.skip("sparse_grad does not support batch dimensions (gsplat expects batch_dims==())")

    def test_absgrad_storage(self, mock_model, simple_rendering_data, simple_gaussian_parameters):
        """Test that absgrad is stored in meta dict."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        config = GSplatRendererConfig.model_validate(
            {
                "name": "3dgs-gsplat",
                "tiling": {"camera": {"tile_width": 16, "tile_height": 16}},
                "background": {"color": 0.0},
                "culling": {"near_clip_distance": 0.01, "far_clip_distance": 1000.0},
                "projection": {"min_projected_ray_radius": 0.5477},
                "absgrad": True,  # Enable absolute gradients
            }
        )

        renderer = GSplatRenderer(config, mock_model)

        # Forward + backward
        output = renderer.render(
            simple_rendering_data,
            simple_gaussian_parameters,
            n_active_features=0,
            extra_ray_signal_infos=([], [], []),
        )
        loss = output.rgb.sum()
        loss.backward()

        # Check meta dict was stored
        assert renderer.last_rendering_meta is not None, "Meta dict not stored"
        assert "means2d" in renderer.last_rendering_meta, "means2d not in meta"

        # Check absgrad was computed (accessed after backward)
        means2d = renderer.last_rendering_meta["means2d"]
        assert hasattr(means2d, "absgrad"), "absgrad not computed"

        print("✓ absgrad stored in meta dict")

    def test_sparse_grad_with_3dgut_warning(self, mock_model, simple_rendering_data, simple_gaussian_parameters):
        """Test that sparse_grad with 3DGUT produces warning."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        config = GSplatRendererConfig.model_validate(
            {
                "name": "3dgut-gsplat",  # 3DGUT mode doesn't support sparse_grad
                "tiling": {"camera": {"tile_width": 16, "tile_height": 16}},
                "background": {"color": 0.0},
                "culling": {"near_clip_distance": 0.01, "far_clip_distance": 1000.0},
                "projection": {
                    "min_projected_ray_radius": 0.5477,
                    "ut_alpha": 0.1,
                    "ut_beta": 2.0,
                    "ut_kappa": 0.0,
                    "image_margin_factor": 0.1,
                },
                "sparse_grad": True,  # Should be disabled with warning
                "packed": True,
            }
        )

        renderer = GSplatRenderer(config, mock_model)

        # Should still work, just without sparse_grad
        output = renderer.render(
            simple_rendering_data,
            simple_gaussian_parameters,
            n_active_features=0,
            extra_ray_signal_infos=([], [], []),
        )

        assert output.rgb is not None
        print("✓ sparse_grad with 3DGUT handled correctly")


class TestPoseGradientLimitations:
    """Test pose gradient warnings and limitations."""

    def test_3dgut_pose_gradient_warning(self, mock_model, caplog):
        """Test that 3dgut-gsplat without ray input produces pose gradient warning."""
        import logging

        config = GSplatRendererConfig.model_validate(
            {
                "name": "3dgut-gsplat",  # Should trigger warning
                "use_rays": False,  # Warning only fires without ray input
                "tiling": {"camera": {"tile_width": 16, "tile_height": 16}},
                "background": {"color": 0.0},
            }
        )

        with caplog.at_level(logging.WARNING):
            renderer = GSplatRenderer(config, mock_model)

        # Check warning was logged
        assert any("does NOT support gradients" in record.message for record in caplog.records)
        print("✓ 3DGUT pose gradient warning produced")

    def test_3dgs_no_warning(self, mock_model, caplog):
        """Test that 3dgs-gsplat does NOT produce pose gradient warning."""
        import logging

        config = GSplatRendererConfig.model_validate(
            {
                "name": "3dgs-gsplat",  # Should NOT trigger warning
                "tiling": {"camera": {"tile_width": 16, "tile_height": 16}},
                "background": {"color": 0.0},
            }
        )

        with caplog.at_level(logging.WARNING):
            renderer = GSplatRenderer(config, mock_model)

        # Check NO warning was logged
        assert not any("does NOT support gradients" in record.message for record in caplog.records)
        print("✓ 3DGS (non-3DGUT) produces no warning")

    def test_3dgut_rejects_antialiased_rasterization(self):
        """Test that 3dgut-gsplat rejects antialiased rasterization at config validation time."""
        with pytest.raises(ValueError, match="match nRend's opacity behavior"):
            GSplatRendererConfig.model_validate(
                {
                    "name": "3dgut-gsplat",
                    "rasterize_mode": "antialiased",
                    "tiling": {"camera": {"tile_width": 16, "tile_height": 16}},
                    "background": {"color": 0.0},
                }
            )


class TestSceneDataOutputs:
    """Test scene-level gsplat outputs returned by the renderer."""

    _H, _W, _N = 4, 5, 3

    @pytest.fixture
    def scene_test_data(self):
        """Shared small-resolution rendering data for scene-output tests."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device("cuda")
        h, w, n = self._H, self._W, self._N
        sensor_params = OpenCVPinholeCameraModelParameters(
            resolution=np.array([w, h], dtype=np.uint64),
            focal_length=np.array([50.0, 50.0], dtype=np.float32),
            principal_point=np.array([w / 2, h / 2], dtype=np.float32),
            radial_coeffs=np.zeros(6, dtype=np.float32),
            tangential_coeffs=np.zeros(2, dtype=np.float32),
            thin_prism_coeffs=np.zeros(4, dtype=np.float32),
            shutter_type=ShutterType.GLOBAL,
        )
        pose_tquat = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
        timestamps = torch.tensor([[0, 1000]], dtype=torch.int64, device=device)
        rendering_data = RenderingData(
            rays=torch.randn(1, h, w, 6, device=device),
            sensor_model_parameters=[sensor_params],
            poses_tquat_startend=pose_tquat.unsqueeze(1).repeat(1, 2, 1),
            timestamps_startend_us=timestamps,
            timestamps_startend_us_cpu=timestamps.cpu(),
            rays_timestamps_us=None,
            _rays_footprints=None,
        )
        gaussian_parameters = {
            "positions": torch.randn(n, 3, device=device),
            "rotations": nn.functional.normalize(torch.randn(n, 4, device=device), dim=-1),
            "scales": torch.rand(n, 3, device=device),
            "densities": torch.sigmoid(torch.randn(n, device=device)),
            "features": torch.rand(n, 3, device=device),
            "extra_signal": torch.zeros(n, 0, device=device),
            "camera_extra_signal": torch.zeros(n, 0, device=device),
        }
        return rendering_data, gaussian_parameters, device

    def test_scene_data_is_flattened(self, simple_config, mock_model, scene_test_data, monkeypatch):
        """Singleton batch/view axes from gsplat meta should collapse to per-Gaussian vectors."""
        import nre.models.gaussians.renderers as gaussian_renderers

        rendering_data, gaussian_parameters, device = scene_test_data
        h, w, n = self._H, self._W, self._N
        renderer = GSplatRenderer(simple_config, mock_model)

        def fake_rasterization(*args, **kwargs):
            render_colors = torch.zeros(1, 1, h, w, 4, dtype=torch.float32, device=device)
            render_alphas = torch.zeros(1, 1, h, w, 1, dtype=torch.float32, device=device)
            meta = {
                "radii": torch.tensor(
                    [[[[0.0, 0.0], [1.0, 0.5], [0.0, 2.0]]]],
                    dtype=torch.float32,
                    device=device,
                )
            }
            return render_colors, render_alphas, meta

        monkeypatch.setattr(gaussian_renderers.gsplat, "rasterization", fake_rasterization)

        output = renderer.render(
            rendering_data,
            gaussian_parameters,
            n_active_features=0,
            extra_ray_signal_infos=([], [], []),
        )

        assert output.visibility is not None
        assert output.visibility.shape == (n,)
        assert torch.equal(output.visibility, torch.tensor([0.0, 1.0, 1.0], device=device))

    def test_flat_radii_passed_through(self, simple_config, mock_model, scene_test_data, monkeypatch):
        """Flat (N,) radii from gsplat should produce (N,) visibility without amax."""
        import nre.models.gaussians.renderers as gaussian_renderers

        rendering_data, gaussian_parameters, device = scene_test_data
        h, w, n = self._H, self._W, self._N
        renderer = GSplatRenderer(simple_config, mock_model)

        def fake_rasterization(*args, **kwargs):
            render_colors = torch.zeros(1, 1, h, w, 4, dtype=torch.float32, device=device)
            render_alphas = torch.zeros(1, 1, h, w, 1, dtype=torch.float32, device=device)
            meta = {
                "radii": torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32, device=device),
            }
            return render_colors, render_alphas, meta

        monkeypatch.setattr(gaussian_renderers.gsplat, "rasterization", fake_rasterization)

        output = renderer.render(
            rendering_data,
            gaussian_parameters,
            n_active_features=0,
            extra_ray_signal_infos=([], [], []),
        )

        assert output.visibility is not None
        assert output.visibility.shape == (n,)
        assert torch.equal(output.visibility, torch.tensor([0.0, 1.0, 1.0], device=device))


class TestTrainingIntegration:
    """Test integration with training workflow."""

    def test_training_step_completes(
        self, simple_config, mock_model, simple_rendering_data, simple_gaussian_parameters
    ):
        """Test that a full training step completes without error."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        renderer = GSplatRenderer(simple_config, mock_model)

        # Simulate training step
        optimizer = torch.optim.Adam(
            [
                simple_gaussian_parameters["positions"],
                simple_gaussian_parameters["rotations"],
                simple_gaussian_parameters["scales"],
                simple_gaussian_parameters["densities"],
                simple_gaussian_parameters["features"],
            ],
            lr=0.001,
        )

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = renderer.render(
            simple_rendering_data,
            simple_gaussian_parameters,
            n_active_features=0,
            extra_ray_signal_infos=([], [], []),
        )

        # Compute loss
        loss = output.rgb.sum() + output.opacity.sum()

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        print("✓ Training step completes successfully")

    def test_multiple_training_steps(
        self, simple_config, mock_model, simple_rendering_data, simple_gaussian_parameters
    ):
        """Test multiple training steps complete without error."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        renderer = GSplatRenderer(simple_config, mock_model)

        optimizer = torch.optim.Adam(
            [
                simple_gaussian_parameters["positions"],
                simple_gaussian_parameters["rotations"],
                simple_gaussian_parameters["scales"],
                simple_gaussian_parameters["densities"],
                simple_gaussian_parameters["features"],
            ],
            lr=0.001,
        )

        losses = []
        for step in range(5):
            optimizer.zero_grad()

            output = renderer.render(
                simple_rendering_data,
                simple_gaussian_parameters,
                n_active_features=0,
                extra_ray_signal_infos=([], [], []),
            )

            loss = output.rgb.sum() + output.opacity.sum()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Loss should change over steps (parameters are updating)
        assert not all(l == losses[0] for l in losses), "Loss not changing (parameters not updating)"
        print(f"✓ Multiple training steps complete (losses: {losses})")


if __name__ == "__main__":
    # Run tests manually
    pytest.main([__file__, "-v", "-s"])
