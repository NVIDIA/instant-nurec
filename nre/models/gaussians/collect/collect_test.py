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
import time
import unittest

from collections import namedtuple
from dataclasses import fields, is_dataclass

import torch
import torch.nn as nn

from libs.slang_utils.utils import profile
from nre.models.gaussians.collect import (
    CreateGaussianParameterCollector,
    DensityActivation,
    DirectTracksCalibData,
    IndividualRemapTimeInputEmbeddingConfig,
    IndividualRemapTimeInputEmbeddingData,
    InputEmbeddingData,
    LayerConfigDeformable,
    LayerConfigRigid,
    LayerConfigSH,
    LayerDataDeformable,
    LayerDataRigid,
    LayerDataSH,
    LayersConfig,
    LayersData,
    RotationActivation,
    ScaleActivation,
    SceneContractorData,
    TracksInterpolationData,
    TracksTimestampsGlobalData,
    TracksTimestampsPerTrackData,
    WeightedInstanceInputEmbeddingData,
)
from nre.utils.types import AABB3D, SceneContractor


device = torch.device("cuda")
# Deterministic random for reproducibility
torch.manual_seed(123)


class TestCollector(unittest.TestCase):
    def _apply_scene_contraction(
        self, xyzs: torch.Tensor, instance_idx: torch.Tensor, scene_contractor_config: SceneContractorData
    ) -> torch.Tensor:
        """Apply scene contraction using SceneContractor.to_contracted_space().

        We use the Python SceneContractor implementation here for ground truth comparison.
        """
        # Create SceneContractor from config
        aabb = AABB3D(scene_contractor_config.aabb_blb, scene_contractor_config.aabb_trf)
        scene_contractor = SceneContractor(
            degree=scene_contractor_config.degree,
            aabb=aabb,
            is_single=False,  # Multiple instances
            is_merf=scene_contractor_config.is_merf,
        )

        # Apply per-sample contraction by indexing the contractor per instance
        contracted_xyzs = scene_contractor[instance_idx].to_contracted_space(xyzs)

        # Clamp to [0, 1] as done in Slang kernel
        return torch.clamp(contracted_xyzs, 0.0, 1.0)

    def test_errors(self):
        layers_config = LayersConfig(
            layers=[
                LayerConfigSH(
                    rotation_activation=RotationActivation.NORMALIZE,
                    scale_activation=ScaleActivation.EXP,
                    density_activation=DensityActivation.SIGMOID,
                    fourier_features_dim=1,
                    embed_config=None,
                ),
            ],
            extra_signal_dim=0,
            camera_extra_signal_dim=20,
            lidar_extra_signal_dim=0,
            albedo_dim=3,
            specular_dim=45,
        )

        collector = CreateGaussianParameterCollector(layers_config)

        nb_gaussians = 1000
        layers_data = LayersData(
            layers=[
                LayerDataSH(
                    positions=torch.randn(nb_gaussians, 3, device=device),
                    rotations=torch.randn(nb_gaussians, 4, device=device),
                    scales=torch.randn(nb_gaussians, 3, device=device),
                    densities=torch.randn(nb_gaussians, 1, device=device),
                    extra_signal=torch.randn(nb_gaussians, 0, device=device),
                    camera_extra_signal=torch.randn(nb_gaussians, 20, device=device),
                    lidar_extra_signal=torch.randn(nb_gaussians, 0, device=device),
                    features_albedo=torch.randn(nb_gaussians, 3, device=device),
                    features_specular=torch.randn(nb_gaussians, 45, device=device),
                    embed_data=None,
                )
            ],
            frame_timestamp_us=None,
        )

        # Test successful collection.
        results = collector.collect(layers_data)
        self.assertEqual(results.positions.shape, layers_data.layers[0].positions.shape)
        self.assertEqual(results.rotations.shape, layers_data.layers[0].rotations.shape)
        self.assertEqual(results.scales.shape, layers_data.layers[0].scales.shape)
        self.assertEqual(results.densities.shape, layers_data.layers[0].densities.shape)
        self.assertEqual(results.extra_signal.shape, layers_data.layers[0].extra_signal.shape)
        self.assertEqual(results.camera_extra_signal.shape, layers_data.layers[0].camera_extra_signal.shape)
        self.assertEqual(results.lidar_extra_signal.shape, layers_data.layers[0].lidar_extra_signal.shape)
        self.assertEqual(
            results.features.shape,
            (
                nb_gaussians,
                layers_data.layers[0].features_albedo.shape[1] + layers_data.layers[0].features_specular.shape[1],
            ),
        )

        tensor_attributes = [
            field.name
            for field in fields(LayerDataSH)
            if isinstance(getattr(layers_data.layers[0], field.name), torch.Tensor)
        ]

        # For mismatch in number of gaussians.
        for name in tensor_attributes:
            layers_data_modified = copy.deepcopy(layers_data)
            modified_value = getattr(layers_data_modified.layers[0], name)
            modified_value = modified_value[:-1]
            setattr(layers_data_modified.layers[0], name, modified_value)
            with self.assertRaises(AssertionError):
                collector.collect(layers_data_modified)

        # For mismatch in component dimensions.
        for name in tensor_attributes:
            layers_data_modified = copy.deepcopy(layers_data)
            modified_value = getattr(layers_data_modified.layers[0], name)
            if modified_value.shape[1] == 0:
                continue
            modified_value = modified_value[:, :-1]
            setattr(layers_data_modified.layers[0], name, modified_value)
            with self.assertRaises(AssertionError):
                collector.collect(layers_data_modified)

        # For zero-sized layers:
        layer_data_zeroed = copy.deepcopy(layers_data)
        for name in tensor_attributes:
            modified_value = getattr(layer_data_zeroed.layers[0], name)
            modified_value = modified_value[0:0]
            setattr(layer_data_zeroed.layers[0], name, modified_value)
        results = collector.collect(layer_data_zeroed)
        for field in fields(results):
            result = getattr(results, field.name)
            self.assertEqual(result.shape[0], 0)

    def test_base_layer(self):
        layers_nb_gaussians = [1000, 10000, 100000]
        layers_config = LayersConfig(
            layers=[
                LayerConfigSH(
                    rotation_activation=RotationActivation.NORMALIZE,
                    scale_activation=ScaleActivation.EXP,
                    density_activation=DensityActivation.SIGMOID,
                    fourier_features_dim=1,
                    embed_config=None,
                ),
            ]
            * len(layers_nb_gaussians),
            extra_signal_dim=0,
            camera_extra_signal_dim=20,
            lidar_extra_signal_dim=0,
            albedo_dim=3,
            specular_dim=45,
        )

        collector = CreateGaussianParameterCollector(layers_config)

        layers_data = LayersData(
            layers=[
                LayerDataSH(
                    positions=torch.randn(nb_gaussians, 3, device=device),
                    rotations=torch.randn(nb_gaussians, 4, device=device),
                    scales=torch.randn(nb_gaussians, 3, device=device),
                    densities=torch.randn(nb_gaussians, 1, device=device),
                    extra_signal=torch.randn(nb_gaussians, 0, device=device),
                    camera_extra_signal=torch.randn(nb_gaussians, 20, device=device),
                    lidar_extra_signal=torch.randn(nb_gaussians, 0, device=device),
                    features_albedo=torch.randn(nb_gaussians, 3, device=device),
                    features_specular=torch.randn(nb_gaussians, 45, device=device),
                    embed_data=None,
                )
                for nb_gaussians in layers_nb_gaussians
            ],
            frame_timestamp_us=None,
        )

        # Test successful collection.
        results = collector.collect(layers_data)

        offset = 0
        for i in range(len(layers_nb_gaussians)):
            nb_gaussians = layers_nb_gaussians[i]

            positions = results.positions[offset : offset + nb_gaussians]
            original_positions = layers_data.layers[i].positions
            self.assertTrue(torch.equal(positions, original_positions))

            rotations = results.rotations[offset : offset + nb_gaussians]
            original_rotations = torch.nn.functional.normalize(layers_data.layers[i].rotations)
            self.assertTrue(torch.allclose(rotations, original_rotations))

            scales = results.scales[offset : offset + nb_gaussians]
            original_scales = torch.exp(layers_data.layers[i].scales)
            self.assertTrue(torch.allclose(scales, original_scales))

            densities = results.densities[offset : offset + nb_gaussians]
            original_densities = torch.sigmoid(layers_data.layers[i].densities)
            self.assertTrue(torch.allclose(densities, original_densities))

            extra_signal = results.extra_signal[offset : offset + nb_gaussians]
            original_extra_signal = layers_data.layers[i].extra_signal
            self.assertTrue(torch.equal(extra_signal, original_extra_signal))

            camera_extra_signal = results.camera_extra_signal[offset : offset + nb_gaussians]
            original_camera_extra_signal = layers_data.layers[i].camera_extra_signal
            self.assertTrue(torch.equal(camera_extra_signal, original_camera_extra_signal))

            lidar_extra_signal = results.lidar_extra_signal[offset : offset + nb_gaussians]
            original_lidar_extra_signal = layers_data.layers[i].lidar_extra_signal
            self.assertTrue(torch.equal(lidar_extra_signal, original_lidar_extra_signal))

            features = results.features[offset : offset + nb_gaussians]
            original_features = torch.cat(
                [layers_data.layers[i].features_albedo, layers_data.layers[i].features_specular], dim=1
            )
            self.assertTrue(torch.equal(features, original_features))

            offset += nb_gaussians

    def test_backward_pass(self):
        layers_nb_gaussians = [1000, 10000, 100000]
        layers_config = LayersConfig(
            layers=[
                LayerConfigSH(
                    rotation_activation=RotationActivation.NORMALIZE,
                    scale_activation=ScaleActivation.EXP,
                    density_activation=DensityActivation.SIGMOID,
                    fourier_features_dim=1,
                    embed_config=None,
                ),
            ]
            * len(layers_nb_gaussians),
            extra_signal_dim=0,
            camera_extra_signal_dim=20,
            lidar_extra_signal_dim=0,
            albedo_dim=3,
            specular_dim=45,
        )

        collector = CreateGaussianParameterCollector(layers_config)

        layers_data = LayersData(
            layers=[
                LayerDataSH(
                    positions=torch.nn.Parameter(torch.randn(nb_gaussians, 3, device=device)),
                    rotations=torch.nn.Parameter(torch.randn(nb_gaussians, 4, device=device)),
                    scales=torch.nn.Parameter(torch.randn(nb_gaussians, 3, device=device)),
                    densities=torch.nn.Parameter(torch.randn(nb_gaussians, 1, device=device)),
                    extra_signal=torch.nn.Parameter(torch.randn(nb_gaussians, 0, device=device)),
                    camera_extra_signal=torch.nn.Parameter(torch.randn(nb_gaussians, 20, device=device)),
                    lidar_extra_signal=torch.nn.Parameter(torch.randn(nb_gaussians, 0, device=device)),
                    features_albedo=torch.nn.Parameter(torch.randn(nb_gaussians, 3, device=device)),
                    features_specular=torch.nn.Parameter(torch.randn(nb_gaussians, 45, device=device)),
                    embed_data=None,
                )
                for nb_gaussians in layers_nb_gaussians
            ],
            frame_timestamp_us=None,
        )

        results = collector.collect(layers_data)

        # Test that gradients which are just copies are correct.
        results_positions_grad = torch.randn_like(results.positions)
        results_extra_signal_grad = torch.randn_like(results.extra_signal)
        results_camera_extra_signal_grad = torch.randn_like(results.camera_extra_signal)
        results_lidar_extra_signal_grad = torch.randn_like(results.lidar_extra_signal)
        results_features_grad = torch.randn_like(results.features)

        torch.autograd.backward(
            [
                results.positions,
                results.extra_signal,
                results.camera_extra_signal,
                results.lidar_extra_signal,
                results.features,
            ],
            [
                results_positions_grad,
                results_extra_signal_grad,
                results_camera_extra_signal_grad,
                results_lidar_extra_signal_grad,
                results_features_grad,
            ],
        )

        offset = 0
        for i in range(len(layers_nb_gaussians)):
            nb_gaussians = layers_nb_gaussians[i]
            layer_data = layers_data.layers[i]

            results_positions = results_positions_grad[offset : offset + nb_gaussians]
            self.assertTrue(torch.equal(layer_data.positions.grad, results_positions))

            results_camera_extra_signal = results_camera_extra_signal_grad[offset : offset + nb_gaussians]
            self.assertTrue(torch.equal(layer_data.camera_extra_signal.grad, results_camera_extra_signal))

            results_features_albedo = results_features_grad[offset : offset + nb_gaussians, :3]
            self.assertTrue(torch.equal(layer_data.features_albedo.grad, results_features_albedo))
            results_features_specular = results_features_grad[offset : offset + nb_gaussians, 3:]
            self.assertTrue(torch.equal(layer_data.features_specular.grad, results_features_specular))

            offset += nb_gaussians

    def _get_standalone_collector(self):
        layers_config = LayersConfig(
            layers=[
                LayerConfigSH(
                    rotation_activation=RotationActivation.NORMALIZE,
                    scale_activation=ScaleActivation.EXP,
                    density_activation=DensityActivation.SIGMOID,
                    fourier_features_dim=1,
                    embed_config=None,
                ),
            ],
            extra_signal_dim=0,
            camera_extra_signal_dim=20,
            lidar_extra_signal_dim=0,
            albedo_dim=3,
            specular_dim=45,
        )

        collector = CreateGaussianParameterCollector(layers_config)
        return collector

    def _get_test_scene_contractor(self, n_instances=1):
        """Helper method to create a test SceneContractorData."""
        # Simple AABB centered at origin with range [-1, 1] for each instance
        aabb_blb = torch.tensor([[-1.0, -1.0, -1.0]] * n_instances, device=device, dtype=torch.float32)
        aabb_trf = torch.tensor([[1.0, 1.0, 1.0]] * n_instances, device=device, dtype=torch.float32)
        return SceneContractorData(
            aabb_blb=aabb_blb,
            aabb_trf=aabb_trf,
            degree=2.0,
            is_merf=False,
        )

    def test_tracks_calib(self):
        # Create the fake rotations.
        tracks_nb_poses = [123, 456, 789]
        total_count = sum(tracks_nb_poses)

        gradient_mask = torch.full((total_count,), True, dtype=torch.bool, device=device)
        offset = 0
        for count in tracks_nb_poses:
            gradient_mask[offset] = False
            gradient_mask[offset + count - 1] = False
            offset += count
        tracks_delta_q = torch.nn.functional.normalize(torch.randn(total_count, 4, device=device))
        tracks_delta_q.requires_grad = True
        tracks_delta_t = torch.randn(total_count, 3, device=device)
        tracks_delta_t.requires_grad = True
        tracks_poses = torch.zeros(total_count, 4, device=device)

        tracks_q = torch.nn.functional.normalize(torch.randn(total_count, 4, device=device))
        tracks_t = torch.randn(total_count, 3, device=device)
        tracks_poses = torch.cat([tracks_t, tracks_q], dim=1)

        # Compute the ground truth poses.
        import lietorch as lt

        current_tracks_delta_q = torch.where(gradient_mask.view(-1, 1), tracks_delta_q, tracks_delta_q.detach())
        current_tracks_delta_t = torch.where(gradient_mask.view(-1, 1), tracks_delta_t, tracks_delta_t.detach())

        current_tracks_delta_transform = lt.SE3.InitFromVec(
            torch.cat(
                [current_tracks_delta_t, current_tracks_delta_q / current_tracks_delta_q.norm(dim=1, keepdim=True)],
                dim=1,
            )
        )

        ground_truth_poses = current_tracks_delta_transform * lt.SE3.InitFromVec(tracks_poses)
        ground_truth_poses = ground_truth_poses.vec()

        # Calibrate the poses.
        collector = self._get_standalone_collector()

        tracks_calib_data = DirectTracksCalibData(
            tracks_poses=tracks_poses,
            gradient_mask=gradient_mask,
            tracks_delta_q=tracks_delta_q,
            tracks_delta_t=tracks_delta_t,
        )
        output_poses = collector.calibrate_tracks_poses(tracks_calib_data)

        self.assertTrue(torch.allclose(output_poses, ground_truth_poses, atol=1e-5))

        # Backward pass.
        ground_truth_poses_grad = torch.randn_like(output_poses)
        output_poses.backward(ground_truth_poses_grad)
        input_grad_delta_t = tracks_delta_t.grad
        input_grad_delta_q = tracks_delta_q.grad

        tracks_delta_t.grad = None
        tracks_delta_q.grad = None

        ground_truth_poses.backward(ground_truth_poses_grad)

        ground_truth_input_grad_delta_t = tracks_delta_t.grad
        ground_truth_input_grad_delta_q = tracks_delta_q.grad
        # Since we don't implement tangent space propagation yet, to match lietorch's
        # results, we need to remove the parts of the gradient orthogonal to the tangent space.
        # This is done by projecting the gradient onto the tangent space.
        input_grad_delta_q = (
            input_grad_delta_q - (input_grad_delta_q * tracks_delta_q).sum(dim=1, keepdim=True) * tracks_delta_q
        )

        self.assertTrue(torch.allclose(input_grad_delta_t, ground_truth_input_grad_delta_t, atol=1e-5))
        self.assertTrue(torch.allclose(input_grad_delta_q, ground_truth_input_grad_delta_q, atol=1e-5))

    def test_input_embeddings(self):
        """Test input embedding computation with IndividualRemapTimeInputEmbedding."""
        n_samples = 1000
        n_instances = 10
        instance_dim = 1  # Must be 1 for Slang [ForceUnroll] in differentiable code

        # Create input data
        xyzs = torch.randn(n_samples, 3, device=device, dtype=torch.float32)
        xyzs.requires_grad = True

        instance_idx = torch.randint(0, n_instances, (n_samples,), device=device, dtype=torch.int32)

        instance_embedding = nn.Embedding(n_instances, instance_dim, device=device, dtype=torch.float32)

        # Time embedding config
        timestamps_ranges = torch.tensor(
            [[1000 * i, 1000 * (i + 10)] for i in range(n_instances)], dtype=torch.int64, device=device
        )
        time_embedding_config = IndividualRemapTimeInputEmbeddingConfig(
            timestamps_us_ranges=timestamps_ranges,
            remap_min=0.0,
            remap_max=1.0,
        )

        # Use single timestamp for all samples
        timestamp_us = 5000

        # Get scene contractor for applying contraction
        scene_contractor_config = self._get_test_scene_contractor(n_instances=n_instances)

        # Compute ground truth in Python
        ground_truth_embeddings = []

        # xyz
        contracted_xyzs = self._apply_scene_contraction(xyzs, instance_idx, scene_contractor_config)
        ground_truth_embeddings.append(contracted_xyzs)

        # instance embedding
        instance_emb = instance_embedding(instance_idx)
        ground_truth_embeddings.append(instance_emb)

        # time embedding (single timestamp for all samples)
        ranges = timestamps_ranges[instance_idx]
        ratio = (float(timestamp_us) - ranges[:, 0].float()) / (ranges[:, 1].float() - ranges[:, 0].float())
        time_emb = torch.clamp(ratio, 0, 1).unsqueeze(-1)  # remap_min=0, remap_max=1
        ground_truth_embeddings.append(time_emb)

        ground_truth = torch.cat(ground_truth_embeddings, dim=-1)

        # Compute using Slang collector
        collector = self._get_standalone_collector()

        input_embedding_data = InputEmbeddingData(
            xyzs=xyzs,
            instance_idx=instance_idx,
            timestamps_us=timestamp_us,
            instance_embedding_weights=WeightedInstanceInputEmbeddingData(
                instance_embedding_weights=instance_embedding.weight
            ),
            time_embedding_config=time_embedding_config,
            scene_contractor=scene_contractor_config,
        )

        output = collector.prepare_input_embeddings(input_embedding_data)

        self.assertEqual(output.shape, (n_samples, 3 + instance_dim + 1))
        self.assertTrue(torch.allclose(output, ground_truth, atol=1e-5))

        # Test backward pass
        xyzs.grad = None
        instance_embedding.weight.grad = None

        ground_truth_grad = torch.randn_like(output)
        output.backward(ground_truth_grad)
        output_grad_xyz = xyzs.grad
        output_grad_emb = instance_embedding.weight.grad

        xyzs.grad = None
        instance_embedding.weight.grad = None

        ground_truth.backward(ground_truth_grad)
        ground_truth_grad_xyz = xyzs.grad
        ground_truth_grad_emb = instance_embedding.weight.grad

        self.assertTrue(torch.allclose(output_grad_xyz, ground_truth_grad_xyz, atol=1e-5))
        self.assertTrue(torch.allclose(output_grad_emb, ground_truth_grad_emb, atol=1e-5))

    def test_input_embeddings_with_timestamps_delta(self):
        """Test input embedding computation with IndividualRemapTimeInputEmbedding and timestamps_delta."""
        n_samples = 1000
        n_instances = 10
        instance_dim = 1  # Must be 1 for Slang [ForceUnroll] in differentiable code

        # Create input data
        xyzs = torch.randn(n_samples, 3, device=device, dtype=torch.float32)
        xyzs.requires_grad = True

        instance_idx = torch.randint(0, n_instances, (n_samples,), device=device, dtype=torch.int32)

        instance_embedding = nn.Embedding(n_instances, instance_dim, device=device, dtype=torch.float32)

        # Time embedding config
        timestamps_ranges = torch.tensor(
            [[1000 * i, 1000 * (i + 10)] for i in range(n_instances)], dtype=torch.int64, device=device
        )
        time_embedding_config = IndividualRemapTimeInputEmbeddingConfig(
            timestamps_us_ranges=timestamps_ranges,
            remap_min=0.0,
            remap_max=1.0,
        )

        # Use single timestamp for all samples
        timestamp_us = 5000

        # Per-sample deltas
        timestamps_delta = torch.randint(-500, 501, (n_samples,), device=device, dtype=torch.int64)

        # Get scene contractor for applying contraction
        scene_contractor_config = self._get_test_scene_contractor(n_instances=n_instances)

        # Compute ground truth in Python for all 3 variants: [t, t-delta, t+delta]
        ground_truth_variants = []

        time_offsets = [
            torch.zeros_like(timestamps_delta),
            -timestamps_delta,
            timestamps_delta,
        ]  # [t, t-delta, t+delta]

        for time_offset in time_offsets:
            contracted_xyzs = self._apply_scene_contraction(xyzs, instance_idx, scene_contractor_config)

            # instance embedding (same for all variants)
            instance_emb = instance_embedding(instance_idx)

            # time embedding with offset
            adjusted_timestamp = float(timestamp_us) + time_offset.float()
            ranges = timestamps_ranges[instance_idx]
            ratio = (adjusted_timestamp - ranges[:, 0].float()) / (ranges[:, 1].float() - ranges[:, 0].float())
            time_emb = torch.clamp(ratio, 0, 1).unsqueeze(-1)  # remap_min=0, remap_max=1

            # Concatenate this variant
            variant = torch.cat([contracted_xyzs, instance_emb, time_emb], dim=-1)
            ground_truth_variants.append(variant)

        # Stack all variants along batch dimension (dim=0) to get [3*N, base_dim]
        ground_truth = torch.cat(ground_truth_variants, dim=0)

        # Compute using Slang collector
        collector = self._get_standalone_collector()

        input_embedding_data = InputEmbeddingData(
            xyzs=xyzs,
            instance_idx=instance_idx,
            timestamps_us=timestamp_us,
            timestamps_delta=timestamps_delta,
            instance_embedding_weights=WeightedInstanceInputEmbeddingData(
                instance_embedding_weights=instance_embedding.weight
            ),
            time_embedding_config=time_embedding_config,
            scene_contractor=scene_contractor_config,
        )

        output = collector.prepare_input_embeddings(input_embedding_data)

        base_dim = 3 + instance_dim + 1
        self.assertEqual(output.shape, (n_samples * 3, base_dim))
        self.assertTrue(torch.allclose(output, ground_truth, atol=1e-5))

        # Test backward pass
        xyzs.grad = None
        instance_embedding.weight.grad = None

        ground_truth_grad = torch.randn_like(output)
        output.backward(ground_truth_grad)
        output_grad_xyz = xyzs.grad.clone() if xyzs.grad is not None else None
        output_grad_emb = instance_embedding.weight.grad.clone() if instance_embedding.weight.grad is not None else None

        xyzs.grad = None
        instance_embedding.weight.grad = None

        ground_truth.backward(ground_truth_grad)
        ground_truth_grad_xyz = xyzs.grad
        ground_truth_grad_emb = instance_embedding.weight.grad

        self.assertTrue(torch.allclose(output_grad_xyz, ground_truth_grad_xyz, atol=1e-5))
        self.assertTrue(torch.allclose(output_grad_emb, ground_truth_grad_emb, atol=1e-5))

    def test_tracks_interpolation(self):
        def get_translation(t):
            return torch.tensor([1.0, 2.0, 3.0], device=device) * t.unsqueeze(1)

        def get_rotation(t):
            half_angle = t.unsqueeze(1) * (10 * 0.5 * torch.pi / 180.0)
            c = torch.cos(half_angle)
            s = torch.sin(half_angle)
            return torch.cat([s, torch.zeros_like(s), torch.zeros_like(s), c], dim=1)

        def get_tracks_poses(factors):
            track0 = torch.cat([torch.zeros_like(get_translation(factors[0])), get_rotation(factors[0])], dim=1)
            track1 = torch.cat([get_translation(factors[1]), get_rotation(factors[1])], dim=1)
            tracks_poses = torch.cat([track0, track1], dim=0)
            return tracks_poses

        factors = torch.arange(4, device=device).expand(2, 4)
        tracks_poses = get_tracks_poses(factors)
        tracks_timestamps = torch.tensor(
            [1000, 2000, 3000, 4000, 3000, 4000, 5000, 6000], dtype=torch.int64, device=device
        )
        tracks_packinfo = torch.tensor([[0, 4], [4, 4]], dtype=torch.int32, device=device)

        InterpolationTest = namedtuple("InterpolationTest", ["timestamp", "expected_factors", "expected_inside"])
        tests = [
            InterpolationTest(timestamp=0, expected_factors=[0.0, 0.0], expected_inside=[False, False]),
            InterpolationTest(timestamp=1000, expected_factors=[0.0, 0.0], expected_inside=[True, False]),
            InterpolationTest(timestamp=1400, expected_factors=[0.4, 0.0], expected_inside=[True, False]),
            InterpolationTest(timestamp=1500, expected_factors=[0.5, 0.0], expected_inside=[True, False]),
            InterpolationTest(timestamp=1600, expected_factors=[0.6, 0.0], expected_inside=[True, False]),
            InterpolationTest(timestamp=2000, expected_factors=[1.0, 0.0], expected_inside=[True, False]),
            InterpolationTest(timestamp=2400, expected_factors=[1.4, 0.0], expected_inside=[True, False]),
            InterpolationTest(timestamp=2500, expected_factors=[1.5, 0.0], expected_inside=[True, False]),
            InterpolationTest(timestamp=2600, expected_factors=[1.6, 0.0], expected_inside=[True, False]),
            InterpolationTest(timestamp=3000, expected_factors=[2.0, 0.0], expected_inside=[True, True]),
            InterpolationTest(timestamp=3400, expected_factors=[2.4, 0.4], expected_inside=[True, True]),
            InterpolationTest(timestamp=3500, expected_factors=[2.5, 0.5], expected_inside=[True, True]),
            InterpolationTest(timestamp=3600, expected_factors=[2.6, 0.6], expected_inside=[True, True]),
            InterpolationTest(timestamp=4000, expected_factors=[3.0, 1.0], expected_inside=[True, True]),
            InterpolationTest(timestamp=4400, expected_factors=[3.0, 1.4], expected_inside=[False, True]),
            InterpolationTest(timestamp=4500, expected_factors=[3.0, 1.5], expected_inside=[False, True]),
            InterpolationTest(timestamp=4600, expected_factors=[3.0, 1.6], expected_inside=[False, True]),
            InterpolationTest(timestamp=5000, expected_factors=[3.0, 2.0], expected_inside=[False, True]),
            InterpolationTest(timestamp=5400, expected_factors=[3.0, 2.4], expected_inside=[False, True]),
            InterpolationTest(timestamp=5500, expected_factors=[3.0, 2.5], expected_inside=[False, True]),
            InterpolationTest(timestamp=5600, expected_factors=[3.0, 2.6], expected_inside=[False, True]),
            InterpolationTest(timestamp=6000, expected_factors=[3.0, 3.0], expected_inside=[False, True]),
            InterpolationTest(timestamp=6400, expected_factors=[3.0, 3.0], expected_inside=[False, False]),
        ]

        collector = self._get_standalone_collector()

        for test in tests:
            for nearest_neighbor in [False, True]:
                timestamps_data = TracksTimestampsGlobalData(timestamp=test.timestamp)
                tracks_interpolation_data = TracksInterpolationData(
                    tracks_poses=tracks_poses,
                    tracks_timestamps=tracks_timestamps,
                    tracks_packinfo=tracks_packinfo,
                    timestamps_data=timestamps_data,
                    nearest_neighbor=nearest_neighbor,
                )
                results_interp, results_inside_interp = collector.interpolate_tracks_poses(tracks_interpolation_data)

                expected_inside = getattr(test, "expected_inside")
                self.assertEqual(expected_inside, results_inside_interp.tolist())

                expected_factors = test.expected_factors
                expected_factors = torch.tensor(expected_factors, device=device).unsqueeze(1)
                if nearest_neighbor:
                    # In case of a tie we round to the left / smallest value.
                    expected_factors = torch.ceil(expected_factors - 0.5)
                expected_poses = get_tracks_poses(expected_factors)
                self.assertTrue(torch.allclose(results_interp, expected_poses))

        # Backward pass.
        # We are just going to test that gradients pass through in nearest neighbor mode.
        tracks_poses.requires_grad = True
        tracks_poses.grad = None
        timestamps = torch.tensor([2900, 4100], dtype=torch.int64, device=device)
        timestamps_data = TracksTimestampsPerTrackData(timestamps=timestamps)
        matching_indices = [2, 5]
        tracks_interpolation_data = TracksInterpolationData(
            tracks_poses=tracks_poses,
            tracks_timestamps=tracks_timestamps,
            tracks_packinfo=tracks_packinfo,
            timestamps_data=timestamps_data,
            nearest_neighbor=True,
        )
        results_interp, results_inside_interp = collector.interpolate_tracks_poses(tracks_interpolation_data)
        results_interp_grad = torch.randn_like(results_interp)
        results_interp.backward(results_interp_grad)
        tracks_poses_grad = tracks_poses.grad
        for i in range(len(matching_indices)):
            tracks_packinfo_start = tracks_packinfo[i, 0].item()
            tracks_packinfo_length = tracks_packinfo[i, 1].item()
            for j in range(tracks_packinfo_length):
                pose_index = tracks_packinfo_start + j
                ground_truth = results_interp_grad[i]
                if pose_index != matching_indices[i]:
                    ground_truth = torch.zeros_like(ground_truth)
                self.assertTrue(torch.equal(tracks_poses_grad[pose_index], ground_truth))

    def test_rigid_layer(self):
        layers_config = LayersConfig(
            layers=[
                LayerConfigRigid(
                    rotation_activation=RotationActivation.NORMALIZE,
                    scale_activation=ScaleActivation.EXP,
                    density_activation=DensityActivation.SIGMOID,
                    fourier_features_dim=1,
                    embed_config=None,
                ),
            ],
            extra_signal_dim=0,
            camera_extra_signal_dim=20,
            lidar_extra_signal_dim=0,
            albedo_dim=3,
            specular_dim=45,
        )

        collector = CreateGaussianParameterCollector(layers_config)

        nb_tracks = 2
        tracks_poses = torch.cat(
            [
                torch.randn(nb_tracks, 3, device=device),
                torch.zeros(nb_tracks, 3, device=device),
                torch.ones(nb_tracks, 1, device=device),
            ],
            dim=1,
        )
        keep_mask = torch.tensor([True, False], device=device, dtype=torch.bool)
        tracks_ids = torch.tensor([0, 0, 1], device=device, dtype=torch.int32)

        nb_gaussians = 3
        positions = torch.randn(nb_gaussians, 3, device=device)
        rotations = torch.randn(nb_gaussians, 4, device=device)
        scales = torch.randn(nb_gaussians, 3, device=device)
        densities = torch.randn(nb_gaussians, 1, device=device)
        extra_signal = torch.randn(nb_gaussians, 0, device=device)
        camera_extra_signal = torch.randn(nb_gaussians, 20, device=device)
        lidar_extra_signal = torch.randn(nb_gaussians, 0, device=device)
        features_albedo = torch.randn(nb_gaussians, 3, device=device)
        features_specular = torch.randn(nb_gaussians, 45, device=device)
        embed_data = None

        layers_data = LayersData(
            layers=[
                LayerDataRigid(
                    positions=positions,
                    rotations=rotations,
                    scales=scales,
                    densities=densities,
                    extra_signal=extra_signal,
                    camera_extra_signal=camera_extra_signal,
                    lidar_extra_signal=lidar_extra_signal,
                    features_albedo=features_albedo,
                    features_specular=features_specular,
                    embed_data=embed_data,
                    poses=tracks_poses,
                    keep_mask=keep_mask,
                    tracks_ids=tracks_ids,
                ),
            ],
            frame_timestamp_us=None,
        )

        for tensor in [
            positions,
            rotations,
            scales,
            densities,
            extra_signal,
            camera_extra_signal,
            lidar_extra_signal,
            features_albedo,
            features_specular,
            tracks_poses,
        ]:
            tensor.requires_grad = True
            tensor.grad = None

        # Forward pass.
        results = collector.collect(layers_data)
        ground_truth_positions = positions + tracks_poses[tracks_ids][:, :3]
        self.assertTrue(torch.allclose(results.positions, ground_truth_positions))
        enabled_densities = results.densities[keep_mask[tracks_ids]]
        disabled_densities = results.densities[~keep_mask[tracks_ids]]
        self.assertEqual(disabled_densities.shape, (1, 1))
        self.assertTrue(torch.all(disabled_densities == 0))
        self.assertTrue(torch.all(enabled_densities != 0))

        # Backward pass.
        buffers = [getattr(results, field.name) for field in fields(results)]
        gradients = [torch.randn_like(buffer) for buffer in buffers]
        torch.autograd.backward(buffers, gradients)

        for tensor in [extra_signal, lidar_extra_signal]:
            self.assertEqual(tensor.grad, None)

        for tensor in [
            positions,
            rotations,
            scales,
            densities,
            camera_extra_signal,
            features_albedo,
            features_specular,
        ]:
            self.assertTrue(torch.all(tensor.grad[keep_mask[tracks_ids]] != 0))
            self.assertTrue(torch.all(tensor.grad[~keep_mask[tracks_ids]] == 0))

        self.assertTrue(torch.all(tracks_poses.grad[keep_mask] != 0))
        self.assertTrue(torch.all(tracks_poses.grad[~keep_mask] == 0))

        self.assertTrue(torch.all(positions.grad[keep_mask[tracks_ids]] == gradients[0][keep_mask[tracks_ids]]))

    def test_deformable_layer(self):
        layers_config = LayersConfig(
            layers=[
                LayerConfigDeformable(
                    rotation_activation=RotationActivation.NORMALIZE,
                    scale_activation=ScaleActivation.EXP,
                    density_activation=DensityActivation.SIGMOID,
                    fourier_features_dim=1,
                    embed_config=None,
                ),
            ],
            extra_signal_dim=0,
            camera_extra_signal_dim=20,
            lidar_extra_signal_dim=0,
            albedo_dim=3,
            specular_dim=45,
        )

        collector = CreateGaussianParameterCollector(layers_config)

        nb_tracks = 2
        tracks_poses = torch.cat(
            [
                torch.randn(nb_tracks, 3, device=device),
                torch.zeros(nb_tracks, 3, device=device),
                torch.ones(nb_tracks, 1, device=device),
            ],
            dim=1,
        )
        keep_mask = torch.tensor([True, False], device=device, dtype=torch.bool)
        tracks_ids = torch.tensor([0, 0, 1], device=device, dtype=torch.int32)

        nb_gaussians = 3
        positions = torch.randn(nb_gaussians, 3, device=device)
        rotations = torch.randn(nb_gaussians, 4, device=device)
        scales = torch.randn(nb_gaussians, 3, device=device)
        densities = torch.randn(nb_gaussians, 1, device=device)
        extra_signal = torch.randn(nb_gaussians, 0, device=device)
        camera_extra_signal = torch.randn(nb_gaussians, 20, device=device)
        lidar_extra_signal = torch.randn(nb_gaussians, 0, device=device)
        features_albedo = torch.randn(nb_gaussians, 3, device=device)
        features_specular = torch.randn(nb_gaussians, 45, device=device)
        embed_data = None
        deform_positions = torch.randn(nb_gaussians, 3, device=device)
        deform_rotations = torch.randn(nb_gaussians, 4, device=device)

        layers_data = LayersData(
            layers=[
                LayerDataDeformable(
                    positions=positions,
                    rotations=rotations,
                    scales=scales,
                    densities=densities,
                    extra_signal=extra_signal,
                    camera_extra_signal=camera_extra_signal,
                    lidar_extra_signal=lidar_extra_signal,
                    features_albedo=features_albedo,
                    features_specular=features_specular,
                    embed_data=embed_data,
                    poses=tracks_poses,
                    keep_mask=keep_mask,
                    tracks_ids=tracks_ids,
                    deform_positions=deform_positions,
                    deform_rotations=deform_rotations,
                ),
            ],
            frame_timestamp_us=None,
        )

        for tensor in [
            positions,
            rotations,
            scales,
            densities,
            extra_signal,
            camera_extra_signal,
            lidar_extra_signal,
            features_albedo,
            features_specular,
            tracks_poses,
            deform_positions,
            deform_rotations,
        ]:
            tensor.requires_grad = True
            tensor.grad = None

        # Forward pass.
        results = collector.collect(layers_data)
        ground_truth_positions = positions + deform_positions + tracks_poses[tracks_ids][:, :3]
        self.assertTrue(torch.allclose(results.positions, ground_truth_positions))
        enabled_densities = results.densities[keep_mask[tracks_ids]]
        disabled_densities = results.densities[~keep_mask[tracks_ids]]
        self.assertEqual(disabled_densities.shape, (1, 1))
        self.assertTrue(torch.all(disabled_densities == 0))
        self.assertTrue(torch.all(enabled_densities != 0))

        # Backward pass.
        buffers = [getattr(results, field.name) for field in fields(results)]
        gradients = [torch.randn_like(buffer) for buffer in buffers]
        torch.autograd.backward(buffers, gradients)

        for tensor in [extra_signal, lidar_extra_signal]:
            self.assertEqual(tensor.grad, None)

        for tensor in [
            positions,
            rotations,
            scales,
            densities,
            camera_extra_signal,
            features_albedo,
            features_specular,
        ]:
            self.assertTrue(torch.all(tensor.grad[keep_mask[tracks_ids]] != 0))
            self.assertTrue(torch.all(tensor.grad[~keep_mask[tracks_ids]] == 0))

        self.assertTrue(torch.all(tracks_poses.grad[keep_mask] != 0))
        self.assertTrue(torch.all(tracks_poses.grad[~keep_mask] == 0))

        self.assertTrue(torch.all(positions.grad[keep_mask[tracks_ids]] == gradients[0][keep_mask[tracks_ids]]))
        self.assertTrue(torch.all(deform_positions.grad[keep_mask[tracks_ids]] == gradients[0][keep_mask[tracks_ids]]))

    def test_performance(self):
        layers_nb_gaussians = [303609, 57965, 102079]
        layers_config = LayersConfig(
            layers=[
                LayerConfigSH(
                    rotation_activation=RotationActivation.NORMALIZE,
                    scale_activation=ScaleActivation.EXP,
                    density_activation=DensityActivation.SIGMOID,
                    fourier_features_dim=1,
                    embed_config=None,
                ),
                LayerConfigSH(
                    rotation_activation=RotationActivation.NORMALIZE,
                    scale_activation=ScaleActivation.EXP,
                    density_activation=DensityActivation.SIGMOID,
                    fourier_features_dim=1,
                    embed_config=None,
                ),
                LayerConfigSH(
                    rotation_activation=RotationActivation.NORMALIZE,
                    scale_activation=ScaleActivation.EXP,
                    density_activation=DensityActivation.SIGMOID,
                    fourier_features_dim=20,
                    embed_config=IndividualRemapTimeInputEmbeddingConfig(
                        timestamps_us_ranges=torch.tensor([[0, 1000000]], dtype=torch.int64, device=device),
                        remap_min=0.0,
                        remap_max=1.0,
                    ),
                ),
            ],
            extra_signal_dim=0,
            camera_extra_signal_dim=20,
            lidar_extra_signal_dim=0,
            albedo_dim=3,
            specular_dim=45,
        )

        layers_data = LayersData(
            layers=[
                LayerDataSH(
                    positions=torch.nn.Parameter(torch.randn(layers_nb_gaussians[0], 3, device=device)),
                    rotations=torch.nn.Parameter(torch.randn(layers_nb_gaussians[0], 4, device=device)),
                    scales=torch.nn.Parameter(torch.randn(layers_nb_gaussians[0], 3, device=device)),
                    densities=torch.nn.Parameter(torch.randn(layers_nb_gaussians[0], 1, device=device)),
                    extra_signal=torch.nn.Parameter(torch.randn(layers_nb_gaussians[0], 0, device=device)),
                    camera_extra_signal=torch.nn.Parameter(torch.randn(layers_nb_gaussians[0], 20, device=device)),
                    lidar_extra_signal=torch.nn.Parameter(torch.randn(layers_nb_gaussians[0], 0, device=device)),
                    features_albedo=torch.nn.Parameter(torch.randn(layers_nb_gaussians[0], 3, device=device)),
                    features_specular=torch.nn.Parameter(torch.randn(layers_nb_gaussians[0], 45, device=device)),
                    embed_data=None,
                ),
                LayerDataSH(
                    positions=torch.nn.Parameter(torch.randn(layers_nb_gaussians[1], 3, device=device)),
                    rotations=torch.nn.Parameter(torch.randn(layers_nb_gaussians[1], 4, device=device)),
                    scales=torch.nn.Parameter(torch.randn(layers_nb_gaussians[1], 3, device=device)),
                    densities=torch.nn.Parameter(torch.randn(layers_nb_gaussians[1], 1, device=device)),
                    extra_signal=torch.nn.Parameter(torch.randn(layers_nb_gaussians[1], 0, device=device)),
                    camera_extra_signal=torch.nn.Parameter(torch.randn(layers_nb_gaussians[1], 20, device=device)),
                    lidar_extra_signal=torch.nn.Parameter(torch.randn(layers_nb_gaussians[1], 0, device=device)),
                    features_albedo=torch.nn.Parameter(torch.randn(layers_nb_gaussians[1], 3, device=device)),
                    features_specular=torch.nn.Parameter(torch.randn(layers_nb_gaussians[1], 45, device=device)),
                    embed_data=None,
                ),
                LayerDataSH(
                    positions=torch.nn.Parameter(torch.randn(layers_nb_gaussians[2], 3, device=device)),
                    rotations=torch.nn.Parameter(torch.randn(layers_nb_gaussians[2], 4, device=device)),
                    scales=torch.nn.Parameter(torch.randn(layers_nb_gaussians[2], 3, device=device)),
                    densities=torch.nn.Parameter(torch.randn(layers_nb_gaussians[2], 1, device=device)),
                    extra_signal=torch.nn.Parameter(torch.randn(layers_nb_gaussians[2], 0, device=device)),
                    camera_extra_signal=torch.nn.Parameter(torch.randn(layers_nb_gaussians[2], 20, device=device)),
                    lidar_extra_signal=torch.nn.Parameter(torch.randn(layers_nb_gaussians[2], 0, device=device)),
                    features_albedo=torch.nn.Parameter(torch.randn(layers_nb_gaussians[2], 20, 3, device=device)),
                    features_specular=torch.nn.Parameter(torch.randn(layers_nb_gaussians[2], 45, device=device)),
                    embed_data=IndividualRemapTimeInputEmbeddingData(
                        instance_idx=torch.zeros(layers_nb_gaussians[2], dtype=torch.int32, device=device),
                    ),
                ),
            ],
            frame_timestamp_us=500000,
        )

        def zero_grad(obj):
            if isinstance(obj, torch.Tensor):
                obj.grad = None
            elif is_dataclass(obj):
                for field in fields(obj):
                    zero_grad(getattr(obj, field.name))
            elif isinstance(obj, list):
                for item in obj:
                    zero_grad(item)

        with profile("Creation"):
            collector = CreateGaussianParameterCollector(layers_config)

        NB_WARMUP = 10
        NB_MEASURE = 100

        with profile("Warmup"):
            for _ in range(NB_WARMUP):
                results = collector.collect(layers_data)
                buffers = [getattr(results, field.name) for field in fields(results)]
                gradients = [torch.randn_like(buffer) for buffer in buffers]
                torch.autograd.backward(buffers, gradients)

        with profile("Benchmark"):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            for _ in range(NB_MEASURE):
                with profile("Zero grad"):
                    zero_grad(layers_data)
                with profile("Forward"):
                    results = collector.collect(layers_data)
                with profile("Backward"):
                    buffers = [getattr(results, field.name) for field in fields(results)]
                    gradients = [torch.randn_like(buffer) for buffer in buffers]
                    torch.autograd.backward(buffers, gradients)
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            print(f"Collect took {(end_time - start_time) / NB_MEASURE * 1e3:.6f} ms")

        with profile("Benchmark Sync"):
            for _ in range(NB_MEASURE):
                with profile("Zero grad"):
                    zero_grad(layers_data)
                    torch.cuda.synchronize()
                with profile("Forward"):
                    results = collector.collect(layers_data)
                    torch.cuda.synchronize()
                with profile("Backward"):
                    buffers = [getattr(results, field.name) for field in fields(results)]
                    gradients = [torch.randn_like(buffer) for buffer in buffers]
                    torch.autograd.backward(buffers, gradients)
                    torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
