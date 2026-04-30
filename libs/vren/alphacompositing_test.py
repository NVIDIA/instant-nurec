# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import unittest

import numpy as np
import torch

from nre.models.custom_modules import vol_rend_from_alphas, weights_from_alphas
from nre.utils.packed_ops import packed_weighted_sum
from nre.utils.tests import NonDeterministicTestCase
from nre.utils.types import RadianceEmbeddingType


class TestVolumeRendering(NonDeterministicTestCase):
    def setUp(self):
        n_samples = 13000
        n_data_dim = 10
        n_rays = 256

        # Random sample per rays
        rand_n = np.random.rand(n_rays)
        samples_per_ray = np.floor(rand_n * n_samples / np.sum(rand_n))
        missing = np.random.choice(n_rays, int(n_samples - np.sum(samples_per_ray)), replace=False)
        samples_per_ray[missing] += 1
        samples_per_ray = torch.from_numpy(samples_per_ray).to(torch.int)

        self.occupancy_threshold = 0.1
        assert torch.sum(samples_per_ray).item() == n_samples, (
            "Combined number of samples per ray has to equal n_samples"
        )

        self.alphas = torch.rand((n_samples,), requires_grad=True).cuda()
        self.ts = torch.rand((n_samples,)).cuda()
        self.data = torch.rand((n_samples, n_data_dim), requires_grad=True).cuda()
        # Torch doesn't have exclusive cumsum so need to this manually
        cumsum_samples = torch.cumsum(samples_per_ray, 0)[:, None].roll(1, 0)
        cumsum_samples[0] = 0
        self.pack_info = torch.cat([cumsum_samples, samples_per_ray[:, None]], dim=1).cuda().int()
        self.vr_samples = torch.tensor([a - 1 if a > 0 else a for a in samples_per_ray.tolist()]).cuda().int()

    def reference_alpha_to_weights(self, occupancy_threshold=0.0):
        output = []

        for ray in self.pack_info:
            alphas = self.alphas[ray[0] : ray[0] + ray[1]].reshape(1, -1)
            alphas_shifted = torch.cat([torch.ones((1, 1)).cuda(), 1 - alphas + 1e-15], dim=-1)  # [N, T+t+1]
            weights = (alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]).squeeze()
            mask = torch.cumprod(alphas_shifted, dim=-1)[..., :-1].squeeze() > occupancy_threshold
            if ray[1] == 1:
                weights = weights.unsqueeze(0)
                mask = mask.unsqueeze(0)
            output.append(weights * mask)

        return torch.cat(output)

    def test_compare_weights_from_alphas_to_torch(self):
        weights_torch = self.reference_alpha_to_weights()

        # Compare the weights
        _, _, weights_vren = weights_from_alphas(self.alphas, self.pack_info, 0.0)

        self._compareTensor(
            weights_torch.detach().cpu(), weights_vren.detach().cpu(), absolute_decimal=5, relative_decimal=-1
        )

        # Generate the GT weightes: compute the loss and compare the gradients
        gt_weights = torch.rand_like(weights_torch, requires_grad=False)
        loss_torch = torch.mean(torch.square(weights_torch - gt_weights))
        loss_ours = torch.mean(torch.square(weights_vren - gt_weights))

        grad_torch = torch.autograd.grad(loss_torch, (self.alphas), grad_outputs=torch.ones_like(loss_torch))[0]
        grad_ours = torch.autograd.grad(loss_ours, (self.alphas), grad_outputs=torch.ones_like(loss_ours))[0]
        self._compareTensor(
            grad_torch.detach().cpu(),
            grad_ours.detach().cpu(),
            absolute_decimal=5,
            relative_decimal=-1,
            ratio_of_permitted_failures=0.04,
        )

    def test_occupancy_threshold(self):
        # When occupancy threshold is given, only the samples before the occupancy should be converted to weights
        # and get the gradient
        _, _, weights_vren = weights_from_alphas(self.alphas, self.pack_info, self.occupancy_threshold)

        weights_torch = self.reference_alpha_to_weights(self.occupancy_threshold)

        # Compare the weights
        self._compareTensor(weights_torch.detach().cpu(), weights_vren.detach().cpu())

        # Generate the GT weights: compute the loss and compare the gradients
        gt_weights = torch.rand_like(weights_torch, requires_grad=False)
        loss_torch = torch.mean(torch.square(weights_torch - gt_weights))
        loss_ours = torch.mean(torch.square(weights_vren - gt_weights))

        grad_torch = torch.autograd.grad(loss_torch, (self.alphas), grad_outputs=torch.ones_like(loss_torch))[0]
        grad_ours = torch.autograd.grad(loss_ours, (self.alphas), grad_outputs=torch.ones_like(loss_ours))[0]
        self._compareTensor(grad_torch.detach().cpu(), grad_ours.detach().cpu(), absolute_decimal=5, relative_decimal=5)

    def test_split_kernels_to_reference_implementation(self):
        # Perform volume rendering with the reference implementation
        vol_rend = vol_rend_from_alphas(
            self.alphas,
            RadianceEmbeddingType.RGB,
            self.data[:, :3].contiguous(),
            self.ts,
            self.pack_info,
            self.occupancy_threshold,
        )
        reference_rgb = vol_rend.rgb
        reference_weights = vol_rend.sample_weights

        # Perform volume rendering in two stages with split implementation
        _, _, weights_split = weights_from_alphas(self.alphas, self.pack_info, self.occupancy_threshold)
        rgb_split = packed_weighted_sum(self.data[:, :3].contiguous(), weights_split, self.pack_info)

        # Compare the weights
        self._compareTensor(reference_weights.detach().cpu(), weights_split.detach().cpu())
        self._compareTensor(
            reference_rgb.detach().cpu(), rgb_split.detach().cpu(), absolute_decimal=7, relative_decimal=7
        )

        # Generate the GT rgb: compute the loss and compare the gradients
        gt_rgb = torch.rand_like(reference_rgb, requires_grad=False)
        loss_reference = torch.mean(torch.square(reference_rgb - gt_rgb))
        loss_split = torch.mean(torch.square(rgb_split - gt_rgb))

        grad_reference = torch.autograd.grad(
            loss_reference, (self.alphas), grad_outputs=torch.ones_like(loss_reference)
        )[0]

        grad_split = torch.autograd.grad(loss_split, (self.alphas), grad_outputs=torch.ones_like(loss_split))[0]

        self._compareTensor(
            grad_reference.detach().cpu(), grad_split.detach().cpu(), absolute_decimal=5, relative_decimal=5
        )

    def test_zero_sized_feature_input(self):
        feats = torch.empty_like(self.data[:, :0], requires_grad=True)
        vol_rend = vol_rend_from_alphas(
            self.alphas,
            RadianceEmbeddingType.EMPTY,
            feats,
            self.ts,
            self.pack_info,
            self.occupancy_threshold,
        )

        grad_feats = torch.autograd.grad(vol_rend.opacity, feats, torch.ones_like(vol_rend.opacity), only_inputs=True)[
            0
        ]

        assert [*vol_rend.radiance_embedding.shape] == [len(self.pack_info), 0]
        assert [*grad_feats.shape] == [len(self.data), 0]

    def test_zero_sized_pack_info(self):
        pack_info = torch.empty((0, 2), dtype=torch.long, device=self.data.device)

        # Test `weights_from_alphas` with zero-sized pack info
        samples, opacity, weights = weights_from_alphas(self.alphas, pack_info, self.occupancy_threshold)
        assert [*samples.shape] == [*opacity.shape] == [*weights.shape] == [0]

        grad_alphas = torch.autograd.grad(
            opacity,
            self.alphas,
            torch.ones_like(opacity),
            only_inputs=True,
        )[0]

        assert (grad_alphas == 0).all()

        # Test `vol_rend_from_alphas` with zero-sized pack info
        vol_rend = vol_rend_from_alphas(
            self.alphas,
            RadianceEmbeddingType.RGB,
            self.data[:, :3].contiguous(),
            self.ts,
            pack_info,
            self.occupancy_threshold,
        )

        assert vol_rend.n_vr_samples == 0
        assert [*vol_rend.opacity.shape] == [0]
        assert [*vol_rend.distance.shape] == [0]
        assert [*vol_rend.radiance_embedding.shape] == [0, 3]
        assert [*vol_rend.sample_weights.shape] == [0]
        assert [*vol_rend.sample_transmittance.shape] == [0]
        assert [*vol_rend.vr_samples.shape] == [0]

        grad_alphas = torch.autograd.grad(
            vol_rend.opacity, self.alphas, torch.ones_like(vol_rend.opacity), only_inputs=True, retain_graph=True
        )[0]
        grad_feats = torch.autograd.grad(
            vol_rend.opacity, self.alphas, torch.ones_like(vol_rend.opacity), only_inputs=True, retain_graph=True
        )[0]

        assert (grad_alphas == 0).all() and (grad_feats == 0).all()


if __name__ == "__main__":
    unittest.main()
