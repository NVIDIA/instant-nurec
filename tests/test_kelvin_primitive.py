# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Branch-coverage tests for KelvinStaticLayer.voxelize (kl-optimal)."""

import torch

from instant_nurec.primitives.kelvin_primitive import KelvinStaticLayer


def _identity_quat_wxyz(n: int) -> torch.Tensor:
    q = torch.zeros(n, 4)
    q[:, 0] = 1.0
    return q


def _make_layer(
    positions: torch.Tensor,
    *,
    with_semantic: bool = False,
    with_normals: bool = False,
    scales: torch.Tensor | None = None,
) -> KelvinStaticLayer:
    n = positions.shape[0]
    return KelvinStaticLayer(
        positions=positions,
        densities=torch.full((n, 1), 0.5),
        rotations=_identity_quat_wxyz(n),
        scales=scales if scales is not None else torch.full((n, 3), 0.1),
        rgb=torch.zeros(n, 3),
        semantic_class=(torch.zeros(n, 1, dtype=torch.uint8) if with_semantic else None),
        normals=(torch.tensor([[0.0, 0.0, 1.0]]).expand(n, 3).contiguous() if with_normals else None),
    )


class TestKelvinStaticLayerVoxelize:
    def test_default_confidence_is_ones(self):
        """confidence=None branch -> built-in ones tensor; reduces co-located Gaussians."""
        positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        layer = _make_layer(positions)
        out = layer.voxelize(voxel_size=0.1)
        # Two Gaussians at origin collapse to one; the third at (1,0,0) is its own voxel.
        assert len(out) == 2

    def test_explicit_confidence_selects_high_weight(self):
        """confidence != None branch -> softmax weights bias toward high-confidence Gaussians."""
        positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        scales = torch.tensor([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]])
        layer = _make_layer(positions, scales=scales)
        # Strongly bias toward Gaussian #1 -> its scales dominate the merged output.
        confidence = torch.tensor([0.0, 50.0])
        out = layer.voxelize(voxel_size=0.1, confidence=confidence)
        assert len(out) == 1
        # Merged scale should be close to the high-confidence Gaussian's [0.9, 0.9, 0.9].
        # (Spread term contribution is ~0 because positions coincide.)
        assert out.scales.max().item() > 0.5

    def test_no_semantic_no_normals(self):
        """semantic_class=None + normals=None branches -> outputs stay None."""
        positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        layer = _make_layer(positions)
        out = layer.voxelize(voxel_size=0.1)
        assert out.semantic_class is None
        assert out.normals is None

    def test_with_semantic_picks_max_confidence_class(self):
        """semantic_class is not None branch -> argmax_indices selects the right class."""
        positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        layer = _make_layer(positions, with_semantic=True)
        # Overwrite semantic_class so the two co-located Gaussians have distinct IDs.
        layer.semantic_class = torch.tensor([[3], [7]], dtype=torch.uint8)
        confidence = torch.tensor([0.0, 10.0])  # Gaussian #1 wins.
        out = layer.voxelize(voxel_size=0.1, confidence=confidence)
        assert out.semantic_class is not None
        assert out.semantic_class.shape == (1, 1)
        assert out.semantic_class[0, 0].item() == 7

    def test_with_normals_renormalizes_to_unit(self):
        """normals is not None branch -> renormalize keeps unit length when inputs agree."""
        positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        layer = _make_layer(positions, with_normals=True)
        out = layer.voxelize(voxel_size=0.1)
        assert out.normals is not None
        assert out.normals.shape == (1, 3)
        torch.testing.assert_close(torch.norm(out.normals, dim=1), torch.ones(1), atol=1e-5, rtol=1e-5)

    def test_single_gaussian_per_voxel_passthrough(self):
        """Distinct voxels -> output count == input count (no merging)."""
        positions = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        layer = _make_layer(positions)
        out = layer.voxelize(voxel_size=0.5)
        assert len(out) == 3

    def test_returns_same_class_subtype(self):
        """The returned object is constructed via ``self.__class__`` -> exact type preserved."""
        positions = torch.tensor([[0.0, 0.0, 0.0]])
        layer = _make_layer(positions)
        out = layer.voxelize(voxel_size=0.1)
        assert type(out) is KelvinStaticLayer

    def test_density_and_rgb_weighted_average(self):
        """Position/density/rgb fields are scatter_added with softmax weights; co-located
        Gaussians yield the per-axis weighted mean."""
        positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        layer = _make_layer(positions)
        layer.densities = torch.tensor([[0.2], [0.8]])
        layer.rgb = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        out = layer.voxelize(voxel_size=0.1)
        # With confidence=ones, weights are uniform softmax(1) -> 0.5 each.
        torch.testing.assert_close(out.densities, torch.tensor([[0.5]]), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out.rgb, torch.tensor([[0.5, 0.5, 0.0]]), atol=1e-5, rtol=1e-5)
