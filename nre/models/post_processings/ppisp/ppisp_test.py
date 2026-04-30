# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import copy
import gc
import unittest

import torch
import torch.nn as nn

from apex.optimizers import FusedAdam
from torch.nn.functional import softplus

from libs.losses.models.loss_fns import (
    compute_color_losses,
    compute_crf_losses,
    compute_exposure_losses,
    compute_vignetting_losses,
)
from nre.models.post_processings.ppisp import (
    CRF,
    ColorCorrection,
    ExposureOffset,
    PiecewisePowerFunction,
    RadialFalloff,
    Vignetting,
    softplus_inverse,
)


class TestExposureOffset(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Default toy model and data
        self.num_frames = 5
        self.rgb_size = 12
        self.model = ExposureOffset(self.device, [self.num_frames], self.num_frames)
        self.frame_idcs = torch.randint(0, self.num_frames, (self.rgb_size,), device=self.device)
        self.rgb = torch.rand(self.rgb_size, 3, device=self.device)

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def test_output_shape(self):
        output = self.model(self.rgb, self.frame_idcs)
        self.assertEqual(output.shape, self.rgb.shape)

    def test_initial_forward_unchanged(self):
        output = self.model(self.rgb, self.frame_idcs)
        torch.testing.assert_close(output, self.rgb)

    def test_gradient_flow(self):
        # Set random initial exposure offsets
        with torch.no_grad():
            self.model.exposure_params.data = torch.randn(self.num_frames, 1, device=self.device) * 0.1

        # Create a simple loss using the exposure_offset values
        offsets = self.model.exposure_params
        loss = offsets.sum()
        loss.backward()

        self.assertIsNotNone(self.model.exposure_params.grad)
        self.assertGreater(self.model.exposure_params.grad.norm().item(), 0.0, "Gradient norm should be non-zero")

    def test_convergence(self):
        assert self.device == "cuda", "FusedAdam requires CUDA"

        num_frames = 128
        rgb_size = num_frames * 128
        batch_size = 512
        num_epochs = 100

        # Generate smooth sine wave values in [-1, 1]. This represents a strong exposure correction.
        t = torch.linspace(0, 1.3 * torch.pi, num_frames, device=self.device)
        reference_values = torch.sin(t)
        # Ensure zero-mean not to conflict with loss function
        reference_values -= reference_values.mean()

        target_model = ExposureOffset(self.device, [num_frames], num_frames)
        target_model.eval()
        with torch.no_grad():
            target_model.exposure_params.data = reference_values.reshape(-1, 1)

        self.model = ExposureOffset(self.device, [num_frames], num_frames)

        all_frame_idcs = torch.repeat_interleave(torch.arange(num_frames, device=self.device), rgb_size // num_frames)
        all_rgb = torch.rand(rgb_size, 3, device=self.device)  # Random RGB in [0, 1]

        with torch.no_grad():
            all_target_rgb = target_model(all_rgb, all_frame_idcs)

        initial_lr = 0.1
        optimizer = FusedAdam(self.model.parameters(), lr=initial_lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=initial_lr / 100)

        lambdas = {
            "exposure_mean": 0.00001,
            "exposure_smooth": 0.002,
        }

        # Indices for smoothness loss
        src_idcs = torch.arange(num_frames - 1, device=self.device)
        dst_idcs = torch.arange(1, num_frames, device=self.device)

        print("\nExposure offset convergence test:")
        for epoch in range(num_epochs):
            epoch_loss = 0.0

            num_batches = rgb_size // batch_size
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size

                batch_frame_idcs = all_frame_idcs[start_idx:end_idx]
                batch_rgb = all_rgb[start_idx:end_idx]
                batch_target_rgb = all_target_rgb[start_idx:end_idx]

                optimizer.zero_grad()
                output = self.model(batch_rgb, batch_frame_idcs)

                mse_loss = nn.MSELoss()(output, batch_target_rgb)
                reg_losses = compute_exposure_losses(
                    self.model.exposure_params.squeeze(-1), src_idcs=src_idcs, dst_idcs=dst_idcs
                )
                reg_loss = torch.sum(torch.stack([lambdas[key] * value for key, value in reg_losses.items()]))
                loss = mse_loss + reg_loss

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            scheduler.step()

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}, Avg Loss: {epoch_loss / num_batches:.6f}, "
                    f"Zero-mean reg loss: {reg_losses['exposure_mean'].item():.6f}"
                )

        # Verify exposure offsets converged to reference values
        torch.testing.assert_close(self.model.exposure_params.data, reference_values.reshape(-1, 1), rtol=0, atol=1e-2)

        # Verify output matches target for a sample batch
        sample_size = batch_size
        sample_frame_idcs = all_frame_idcs[:sample_size]
        sample_rgb = all_rgb[:sample_size]
        sample_target_rgb = all_target_rgb[:sample_size]

        self.model.eval()
        with torch.no_grad():
            output = self.model(sample_rgb, sample_frame_idcs)
        torch.testing.assert_close(output, sample_target_rgb, rtol=0, atol=1e-2)


class TestVignetting(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.num_cameras = 3
        self.rgb_size = 12
        self.model = Vignetting(self.device, self.num_cameras)
        self.cam_idcs = torch.randint(0, self.num_cameras, (self.rgb_size,), device=self.device)
        self.rgb = torch.rand(self.rgb_size, 3, device=self.device)
        self.coords_xy = torch.rand(self.rgb_size, 2, device=self.device)

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def test_output_shape(self):
        output = self.model(self.rgb, self.cam_idcs, self.coords_xy)
        self.assertEqual(output.shape, self.rgb.shape)

    def test_initial_forward_unchanged(self):
        # All alpha parameters start at zero, so output should equal input
        output = self.model(self.rgb, self.cam_idcs, self.coords_xy)
        torch.testing.assert_close(output, self.rgb)

    def test_known_alpha_values(self):
        # Set up a single camera vignetting correction
        self.model = Vignetting(self.device, 1)

        with torch.no_grad():
            # Set same values for all channels
            for channel in range(3):
                self.model.falloff_curves[0][channel].optical_center.data = torch.tensor([0.5, 0.5], device=self.device)
                self.model.falloff_curves[0][channel].alpha.data = torch.tensor([-2.0, 0.0, 0.0], device=self.device)

        # Test points
        coords = torch.tensor(
            [
                [0.5, 0.5],  # At center: r2 = 0
                [0.5, 0.0],  # Top edge: r2 = 0.25
            ],
            device=self.device,
        )
        cam_idcs = torch.zeros(2, device=self.device, dtype=torch.long)
        rgb = torch.ones(2, 3, device=self.device)

        output = self.model(rgb, cam_idcs, coords)
        expected = torch.tensor(
            [
                [1.0, 1.0, 1.0],  # Center: no change
                [0.5, 0.5, 0.5],  # Edge: 50% vignetting
            ],
            device=self.device,
        )
        torch.testing.assert_close(output, expected)

    def test_convergence(self):
        assert self.device == "cuda", "FusedAdam requires CUDA"

        num_cameras = 4
        rgb_size = num_cameras * 256
        batch_size = 512
        num_epochs = 200

        target_model = Vignetting(self.device, num_cameras)
        target_model.eval()
        with torch.no_grad():
            for cam_idx in range(num_cameras):
                # Random optical centers in [0.3, 0.7] range
                optical_center = 0.4 * torch.rand(2, device=self.device) + 0.3

                # Random alpha values in [-2.0, -0.5] range
                alpha = -1.5 * torch.rand(3, device=self.device) - 0.5

                # All values are identical across channels
                for channel in range(3):
                    target_model.falloff_curves[cam_idx][channel].optical_center.data = optical_center
                    target_model.falloff_curves[cam_idx][channel].alpha.data = alpha

        self.model = Vignetting(self.device, num_cameras)

        all_cam_idcs = torch.repeat_interleave(torch.arange(num_cameras, device=self.device), rgb_size // num_cameras)
        all_coords_xy = torch.rand(rgb_size, 2, device=self.device)
        all_rgb = torch.rand(rgb_size, 3, device=self.device)

        with torch.no_grad():
            all_target_rgb = target_model(all_rgb, all_cam_idcs, all_coords_xy)

        initial_lr = 0.2
        optimizer = FusedAdam(self.model.parameters(), lr=initial_lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=initial_lr / 100)

        lambdas = {
            "vig_center": 0.0,  # The target optical centers are not centered.
            "vig_channel": 0.01,
            "vig_non_pos": 0.01,
        }

        print("\nVignetting convergence test:")
        for epoch in range(num_epochs):
            epoch_loss = 0.0

            num_batches = rgb_size // batch_size
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, rgb_size)

                batch_cam_idcs = all_cam_idcs[start_idx:end_idx]
                batch_coords_xy = all_coords_xy[start_idx:end_idx]
                batch_rgb = all_rgb[start_idx:end_idx]
                batch_target_rgb = all_target_rgb[start_idx:end_idx]

                optimizer.zero_grad()
                output = self.model(batch_rgb, batch_cam_idcs, batch_coords_xy)

                mse_loss = nn.MSELoss()(output, batch_target_rgb)

                # Pack vignetting parameters for loss computation
                packed_vignetting_params = []
                for cam_curves in self.model.falloff_curves:
                    cam_params = []
                    for channel_curve in cam_curves:
                        packed_channel_params = torch.cat([channel_curve.optical_center, channel_curve.alpha])
                        cam_params.append(packed_channel_params)
                    packed_vignetting_params.append(torch.stack(cam_params, dim=0))
                packed_vignetting_params = RadialFalloff.PackedParams(torch.stack(packed_vignetting_params, dim=0))

                reg_losses = compute_vignetting_losses(packed_vignetting_params)
                reg_loss = torch.sum(torch.stack([lambdas[key] * value for key, value in reg_losses.items()]))
                loss = mse_loss + reg_loss

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            scheduler.step()

            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Avg Loss: {epoch_loss / num_batches:.6f}")

        # Verify optical centers converged to target values.
        # Note: We don't verify alpha parameters because they're not uniquely identifiable.
        # Different combinations of alpha values can produce almost the same vignetting effect,
        # making them structurally non-identifiable parameters. Convergence would be extremely slow.
        for cam_idx in range(num_cameras):
            for channel in range(3):
                torch.testing.assert_close(
                    self.model.falloff_curves[cam_idx][channel].optical_center.data,
                    target_model.falloff_curves[cam_idx][channel].optical_center.data,
                    rtol=0,
                    atol=1e-2,
                )

        # Verify output matches target for a sample batch
        sample_size = batch_size
        sample_cam_idcs = all_cam_idcs[:sample_size]
        sample_coords_xy = all_coords_xy[:sample_size]
        sample_rgb = all_rgb[:sample_size]
        sample_target_rgb = all_target_rgb[:sample_size]

        self.model.eval()
        with torch.no_grad():
            output = self.model(sample_rgb, sample_cam_idcs, sample_coords_xy)
        torch.testing.assert_close(output, sample_target_rgb, rtol=0, atol=1e-2)


class TestColorCorrection(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.num_frames = 3
        self.model = ColorCorrection(self.device, [self.num_frames], self.num_frames)
        self.rgb_size = 12
        self.frame_idcs = torch.randint(0, self.num_frames, (self.rgb_size,), device=self.device)
        self.rgb = torch.rand(self.rgb_size, 3, device=self.device)

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def test_output_shape(self):
        output = self.model(self.rgb, self.frame_idcs)
        self.assertEqual(output.shape, self.rgb.shape)

    def test_identity_homography(self):
        h = self.model.get_all_homographies()
        expected = torch.eye(3, device=self.device).unsqueeze(0).expand(self.num_frames, -1, -1)
        torch.testing.assert_close(h, expected)

    def test_initial_forward_unchanged(self):
        output = self.model(self.rgb, self.frame_idcs)
        torch.testing.assert_close(output, self.rgb)

    def test_intensity_preservation(self):
        # Create RGB values in [0.3, 0.7] range, away from saturation, which would distort test
        self.rgb = torch.rand_like(self.rgb) * 0.4 + 0.3

        # Slightly perturb the homography parameters
        with torch.no_grad():
            # Add small random perturbations to the homography parameters
            perturbation = (torch.rand(self.num_frames, 8, device=self.device) - 0.5) * 0.1
            self.model.color_params.data += perturbation

        output = self.model(self.rgb, self.frame_idcs)

        input_intensity = self.rgb.sum(dim=1)
        output_intensity = output.sum(dim=1)

        torch.testing.assert_close(input_intensity, output_intensity, rtol=1e-5, atol=1e-5)

    def test_target_chroms_from_homography(self):
        num_frames = 2
        self.model = ColorCorrection(self.device, [num_frames], num_frames)

        # Define source chromaticities (primaries + neutral gray)
        source_chroms = ColorCorrection.get_default_source_chroms(torch.device(self.device))

        # Set specific homography matrices with known effects
        with torch.no_grad():
            # Frame 0: Identity homography (no change)
            # Already initialized as identity

            # Frame 1: Translation homography (shift all points by [0.1, -0.1])
            # For translation, we need to set h[0,2] = 0.1, h[1,2] = -0.1
            # In the flattened form, these are the 3rd and 6th elements (0-indexed)
            # Start with identity matrix values (first 8 elements of flattened identity matrix)
            translation_homography_params = torch.eye(3, device=self.device).flatten()[:8]
            translation_homography_params[2] = 0.1  # h[0,2] = 0.1 (tx = 0.1)
            translation_homography_params[5] = -0.1  # h[1,2] = -0.1 (ty = -0.1)
            self.model.color_params.data[1] = translation_homography_params

        h = self.model.get_all_homographies()
        computed_target_chroms = ColorCorrection.apply_color_correction_rg(source_chroms, h)

        # Expected results:
        # Frame 0: Same as source (identity transform)
        # Frame 1: Each point shifted by [0.1, -0.1]
        expected_frame0 = source_chroms
        expected_frame1 = source_chroms.clone()
        expected_frame1[:, 0] += 0.1  # Add 0.1 to x-coordinates
        expected_frame1[:, 1] -= 0.1  # Subtract 0.1 from y-coordinates

        expected_target_chroms = torch.stack([expected_frame0, expected_frame1])

        # Verify mapping accuracy
        torch.testing.assert_close(computed_target_chroms, expected_target_chroms, rtol=1e-5, atol=1e-5)

    def test_homography_from_chrom_pairs(self):
        num_frames = 3
        self.model = ColorCorrection(self.device, [num_frames], num_frames)

        # Define source chromaticities (primaries + neutral gray)
        source_chroms = ColorCorrection.get_default_source_chroms(torch.device(self.device))

        # Create random target chromaticities with small perturbations from source
        with torch.no_grad():
            perturbations = (torch.rand(num_frames, 4, 2, device=self.device) - 0.5) * 0.2
            target_chroms = source_chroms.unsqueeze(0) + perturbations  # Shape: (num_frames, 4, 2)
            target_chroms = torch.clamp(target_chroms, 0.05, 0.95)

        for frame_idx in range(num_frames):
            h = ColorCorrection.get_h_from_chrom_pairs(source_chroms, target_chroms[frame_idx])
            self.model.color_params.data[frame_idx] = h.flatten()[:8]

        h = self.model.get_all_homographies()
        computed_target_chroms = ColorCorrection.apply_color_correction_rg(
            source_chroms, h
        )  # Shape: (num_frames, 4, 2)

        # Verify mapping accuracy
        torch.testing.assert_close(computed_target_chroms, target_chroms, rtol=1e-4, atol=1e-4)

    def test_convergence(self):
        assert self.device == "cuda", "FusedAdam requires CUDA"

        num_frames = 32
        rgb_size = num_frames * 256
        batch_size = 2048
        num_epochs = 3000

        # Define source chromaticities (primaries + neutral gray)
        source_chroms = ColorCorrection.get_default_source_chroms(torch.device(self.device))

        # Create a circular pattern of offsets that will create a smooth circular offset over successive frames.
        target_model = ColorCorrection(self.device, [num_frames], num_frames)
        target_model.eval()

        with torch.no_grad():
            radius = 0.05  # This represents a strong color correction.
            t = torch.linspace(0, 1.3 * torch.pi, num_frames, device=self.device)
            phases = torch.tensor([0, torch.pi / 4, torch.pi / 2, 3 * torch.pi / 4], device=self.device)
            frequencies = torch.tensor([1.0, 1.1, 0.9, 1.2], device=self.device)

            # Generate circular paths for all 4 vectors
            # Shape: [num_frames, 4, 2]
            offsets = torch.zeros(num_frames, 4, 2, device=self.device)
            for i in range(4):
                theta = frequencies[i] * t + phases[i]
                offsets[:, i, 0] = radius * torch.cos(theta)  # x coordinate
                offsets[:, i, 1] = radius * torch.sin(theta)  # y coordinate

            # Ensure zero mean across frames for each vector and dimension
            offsets -= offsets.mean(dim=0, keepdim=True)

            target_chroms = source_chroms.unsqueeze(0) + offsets  # Shape: (num_frames, 4, 2)

            for frame_idx in range(num_frames):
                h = ColorCorrection.get_h_from_chrom_pairs(source_chroms, target_chroms[frame_idx])
                target_model.color_params.data[frame_idx] = h.flatten()[:8]

            h = target_model.get_all_homographies()
            target_chroms = ColorCorrection.apply_color_correction_rg(source_chroms, h)

        self.model = ColorCorrection(self.device, [num_frames], num_frames)

        all_frame_idcs = torch.repeat_interleave(torch.arange(num_frames, device=self.device), rgb_size // num_frames)
        all_rgb = torch.rand(rgb_size, 3, device=self.device)

        with torch.no_grad():
            all_target_rgb = target_model(all_rgb, all_frame_idcs)

        initial_lr = 0.01
        optimizer = FusedAdam(self.model.parameters(), lr=initial_lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=initial_lr / 10)

        lambdas = {
            "color_mean": 0.0001,
            "color_smooth": 0.001,
        }

        # Pre-compute indices for smoothness losses across frames.
        # Assuming all frames are from the same camera for this test
        src_idcs = torch.arange(0, num_frames - 1, device=self.device)
        dst_idcs = torch.arange(1, num_frames, device=self.device)

        print("\nColor correction convergence test:")
        for epoch in range(num_epochs):
            epoch_loss = 0.0

            num_batches = rgb_size // batch_size
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, rgb_size)

                batch_frame_idcs = all_frame_idcs[start_idx:end_idx]
                batch_rgb = all_rgb[start_idx:end_idx]
                batch_target_rgb = all_target_rgb[start_idx:end_idx]

                optimizer.zero_grad()
                output = self.model(batch_rgb, batch_frame_idcs)

                mse_loss = nn.MSELoss()(output, batch_target_rgb)
                reg_losses = compute_color_losses(
                    self.model.color_params, src_idcs, dst_idcs, self.model.default_source_chroms
                )
                reg_loss = torch.sum(torch.stack([lambdas[key] * value for key, value in reg_losses.items()]))

                loss = mse_loss + reg_loss
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            scheduler.step()

            if epoch % 300 == 0:
                print(
                    f"Epoch {epoch}, Avg Loss: {epoch_loss / num_batches:.6f}, "
                    f"Zero-mean reg loss: {reg_losses['color_mean'].item():.6f}"
                )

        # Compare normalized homography matrices instead of raw data
        trained_h = self.model.get_all_homographies()
        target_h = target_model.get_all_homographies()
        torch.testing.assert_close(trained_h, target_h, rtol=0, atol=1e-2)

        # Verify output matches target for a sample batch
        sample_size = batch_size
        sample_frame_idcs = all_frame_idcs[:sample_size]
        sample_rgb = all_rgb[:sample_size]
        sample_target_rgb = all_target_rgb[:sample_size]

        self.model.eval()
        with torch.no_grad():
            output = self.model(sample_rgb, sample_frame_idcs)
        torch.testing.assert_close(output, sample_target_rgb, rtol=0, atol=1e-2)


class TestPiecewisePowerFunction(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = PiecewisePowerFunction(self.device)
        self.input_data = torch.rand(1000, device=self.device) * 3.1 - 0.1

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def test_output_shape(self):
        output_data = self.model(self.input_data)
        self.assertEqual(self.input_data.shape, output_data.shape)

    def test_gradient_flow(self):
        output = self.model(self.input_data)

        # Compute dummy loss
        loss = output.mean()
        loss.backward()

        # Verify that gradients are computed for all parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient for {name} is None")
                self.assertGreater(param.grad.norm().item(), 0, f"Gradient for {name} is NaN or zero")

    def test_convergence(self):
        assert self.device == "cuda", "FusedAdam requires CUDA"

        batch_size = 512
        num_epochs = 400

        target_model = PiecewisePowerFunction(
            self.device,
            x0=0.4,
            y0=0.3,
            y1=0.5,
            toe_length=3.0,
            shoulder_length=2.0,
            shoulder_overshoot=0.05,
            gamma=1.0 / 3.2,
        )
        target_model.eval()
        with torch.no_grad():
            all_target_output = target_model(self.input_data)

        initial_lr = 0.2
        optimizer = FusedAdam(self.model.parameters(), lr=initial_lr)

        print("\nPiecewise power function convergence test:")
        for epoch in range(num_epochs):
            epoch_loss = 0.0

            num_batches = self.input_data.shape[0] // batch_size
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, self.input_data.shape[0])

                batch_input = self.input_data[start_idx:end_idx]
                batch_target = all_target_output[start_idx:end_idx]

                optimizer.zero_grad()
                output = self.model(batch_input)

                loss = nn.MSELoss()(output, batch_target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if epoch % 40 == 0:
                print(f"Epoch {epoch}, Avg Loss: {epoch_loss / num_batches:.6f}")

        # Verify output matches target for a sample batch
        sample_size = batch_size
        sample_input = self.input_data[:sample_size]
        sample_target = all_target_output[:sample_size]

        self.model.eval()
        with torch.no_grad():
            output = self.model(sample_input)
        torch.testing.assert_close(output, sample_target, rtol=0, atol=1e-2)


class TestCRF(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_cameras = 2
        self.rgb_size = 1000
        self.model = CRF(self.device, self.num_cameras)
        self.cam_idcs = torch.repeat_interleave(
            torch.arange(self.num_cameras, device=self.device), self.rgb_size // self.num_cameras
        )
        self.rgb = torch.rand(self.rgb_size, 3, device=self.device) * 2.1 - 0.1

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def test_output_shape(self):
        output = self.model(self.rgb, self.cam_idcs)
        self.assertEqual(output.shape, self.rgb.shape)

    def test_clamped_forward(self):
        # Output should always be in [0, 1]
        output = self.model(self.rgb, self.cam_idcs)
        self.assertTrue(torch.all(output >= 0) and torch.all(output <= 1))

    def test_convergence(self):
        assert self.device == "cuda", "FusedAdam requires CUDA"

        batch_size = 512
        num_epochs = 200

        # Create target model with specific CRF parameters
        target_model = CRF(self.device, self.num_cameras)
        target_model.eval()
        with torch.no_grad():
            for cam_idx in range(self.num_cameras):
                # Create single CRF function per camera
                crf = PiecewisePowerFunction(
                    device=self.device,
                    x0=0.05 + 0.07 * cam_idx,
                    y0=0.18 + 0.03 * cam_idx,
                    y1=0.75 + 0.08 * cam_idx,
                    toe_length=1.5 - 0.3 * cam_idx,
                    shoulder_length=0.8 + 0.25 * cam_idx,
                    shoulder_overshoot=0.05 + 0.05 * cam_idx,
                    gamma=1.0 / (2.0 + 0.1 * cam_idx),
                )

                # Assign same curve to all channels. Create a copy so later modifications don't
                # mirror across channels.
                for channel in range(3):
                    target_model.curves[cam_idx][channel] = copy.deepcopy(crf)

        # Calculate and normalize target model's dynamic ranges
        with torch.no_grad():
            dynamic_ranges = []
            for camera_curves in target_model.curves:
                for curve in camera_curves:
                    # Compute curve points and use static inverse method
                    raw_params_accessor = PiecewisePowerFunction.RawParams(curve.raw_params)
                    curve_points = PiecewisePowerFunction.crf_curve_points(raw_params_accessor)
                    dynamic_range = PiecewisePowerFunction.inverse(curve_points, torch.tensor(1.0, device=curve.device))
                    dynamic_ranges.append(dynamic_range)

            max_dynamic_range = max(dynamic_ranges)
            print(f"dyn range max: {max_dynamic_range}")
            scale_factor = 1.0 / max_dynamic_range.item()
            print(f"scale factor: {scale_factor}")
            for camera_curves in target_model.curves:
                for curve in camera_curves:
                    # Get current x0_offset value from curve points
                    raw_params_accessor = PiecewisePowerFunction.RawParams(curve.raw_params)
                    curve_points = PiecewisePowerFunction.crf_curve_points(raw_params_accessor)
                    current_x0_offset = curve_points.x0 / (1.0 + softplus(raw_params_accessor.toe_length_raw))

                    # Compute new x0_offset and convert back to raw parameter
                    new_x0_offset = current_x0_offset * scale_factor
                    new_x0_offset_raw = softplus_inverse(new_x0_offset.item())

                    # Update the raw parameter
                    curve.raw_params.data[0] = new_x0_offset_raw

        with torch.no_grad():
            all_target_rgb = target_model(self.rgb, self.cam_idcs)

        initial_lr = 0.1
        optimizer = FusedAdam(self.model.parameters(), lr=initial_lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=initial_lr / 100)

        lambdas = {
            "crf_range": 0.00001,
            "crf_gamma": 0.0001,
            "crf_channel": 0.001,
        }

        print("\nCRF convergence test:")
        for epoch in range(num_epochs):
            epoch_loss = 0.0

            num_batches = self.rgb_size // batch_size
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, self.rgb_size)

                batch_cam_idcs = self.cam_idcs[start_idx:end_idx]
                batch_rgb = self.rgb[start_idx:end_idx]
                batch_target_rgb = all_target_rgb[start_idx:end_idx]

                optimizer.zero_grad()
                output = self.model(batch_rgb, batch_cam_idcs)

                mse_loss = nn.MSELoss()(output, batch_target_rgb)

                # Compute curve points for all cameras and channels
                # Collect curve points separately for each parameter
                all_x0, all_y0, all_slope_p0 = [], [], []
                all_y0_pre_gamma, all_slope_line, all_gamma = [], [], []
                all_x1, all_y1, all_slope_p1 = [], [], []
                all_shoulder_x, all_shoulder_y = [], []

                for cam_idx in range(self.num_cameras):
                    cam_x0, cam_y0, cam_slope_p0 = [], [], []
                    cam_y0_pre_gamma, cam_slope_line, cam_gamma = [], [], []
                    cam_x1, cam_y1, cam_slope_p1 = [], [], []
                    cam_shoulder_x, cam_shoulder_y = [], []

                    for channel in range(3):
                        curve = self.model.curves[cam_idx][channel]
                        raw_params_accessor = PiecewisePowerFunction.RawParams(curve.raw_params)
                        curve_points = PiecewisePowerFunction.crf_curve_points(raw_params_accessor)

                        # Collect individual parameters
                        cam_x0.append(curve_points.x0)
                        cam_y0.append(curve_points.y0)
                        cam_slope_p0.append(curve_points.slope_p0)
                        cam_y0_pre_gamma.append(curve_points.y0_pre_gamma)
                        cam_slope_line.append(curve_points.slope_line)
                        cam_gamma.append(curve_points.gamma)
                        cam_x1.append(curve_points.x1)
                        cam_y1.append(curve_points.y1)
                        cam_slope_p1.append(curve_points.slope_p1)
                        cam_shoulder_x.append(curve_points.shoulder_x)
                        cam_shoulder_y.append(curve_points.shoulder_y)

                    # Stack channel data for this camera
                    all_x0.append(torch.stack(cam_x0))
                    all_y0.append(torch.stack(cam_y0))
                    all_slope_p0.append(torch.stack(cam_slope_p0))
                    all_y0_pre_gamma.append(torch.stack(cam_y0_pre_gamma))
                    all_slope_line.append(torch.stack(cam_slope_line))
                    all_gamma.append(torch.stack(cam_gamma))
                    all_x1.append(torch.stack(cam_x1))
                    all_y1.append(torch.stack(cam_y1))
                    all_slope_p1.append(torch.stack(cam_slope_p1))
                    all_shoulder_x.append(torch.stack(cam_shoulder_x))
                    all_shoulder_y.append(torch.stack(cam_shoulder_y))

                # Create a single CurvePoints object with stacked parameters
                packed_curve_points = PiecewisePowerFunction.CurvePoints(
                    x0=torch.stack(all_x0),
                    y0=torch.stack(all_y0),
                    slope_p0=torch.stack(all_slope_p0),
                    y0_pre_gamma=torch.stack(all_y0_pre_gamma),
                    slope_line=torch.stack(all_slope_line),
                    gamma=torch.stack(all_gamma),
                    x1=torch.stack(all_x1),
                    y1=torch.stack(all_y1),
                    slope_p1=torch.stack(all_slope_p1),
                    shoulder_x=torch.stack(all_shoulder_x),
                    shoulder_y=torch.stack(all_shoulder_y),
                )

                reg_losses = compute_crf_losses(packed_curve_points)
                reg_loss = torch.sum(torch.stack([lambdas[key] * value for key, value in reg_losses.items()]))
                loss = mse_loss + reg_loss

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            scheduler.step()

            if epoch % 20 == 0:
                print(
                    f"Epoch {epoch}, Avg Loss: {epoch_loss / num_batches:.6f}, "
                    f"Dynamic range reg loss: {reg_losses['crf_range'].item():.6f}"
                )

        # Verify output matches target for a sample batch
        sample_size = batch_size
        sample_cam_idcs = self.cam_idcs[:sample_size]
        sample_rgb = self.rgb[:sample_size]
        sample_target_rgb = all_target_rgb[:sample_size]

        self.model.eval()
        with torch.no_grad():
            output = self.model(sample_rgb, sample_cam_idcs)
        torch.testing.assert_close(output, sample_target_rgb, rtol=0, atol=5.0e-2)
