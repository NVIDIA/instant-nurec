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

"""Branch-coverage tests for KelvinPrimitiveMerge._maybe_voxelize_static_layer.

The full ``merge_primitives_and_batch`` flow needs a fully-populated batch
+ rig trajectories which is exercised end-to-end by integration runs and
is out of scope for the unit tests; what's testable here is the
voxelization hook itself, which is the only behavior added by this
commit.
"""

import logging

import torch

from instant_nurec.config_schema.predict import PrimitiveMergeConfig
from instant_nurec.predict.primitive_merge import KelvinPrimitiveMerge
from instant_nurec.primitives.kelvin_primitive import (
    KelvinDynamicLayer,
    KelvinInstantNuRecPrimitive,
    KelvinStaticLayer,
)


def _identity_quat_wxyz(n: int) -> torch.Tensor:
    q = torch.zeros(n, 4)
    q[:, 0] = 1.0
    return q


def _make_static_layer(positions: torch.Tensor) -> KelvinStaticLayer:
    n = positions.shape[0]
    return KelvinStaticLayer(
        positions=positions,
        densities=torch.full((n, 1), 0.5),
        rotations=_identity_quat_wxyz(n),
        scales=torch.full((n, 3), 0.1),
        rgb=torch.zeros(n, 3),
    )


def _make_dynamic_layer(n: int = 2) -> KelvinDynamicLayer:
    return KelvinDynamicLayer(
        rotations=_identity_quat_wxyz(n),
        scales=torch.full((n, 3), 0.1),
        rgb=torch.zeros(n, 3),
        max_densities=torch.full((n, 1), 0.5),
        keyframe_positions=torch.zeros(n, 3, 3),
        keyframe_timestamps_us=torch.tensor([[0.0, 1.0, 2.0]]).expand(n, 3).contiguous(),
    )


def _make_primitive(positions: torch.Tensor) -> KelvinInstantNuRecPrimitive:
    return KelvinInstantNuRecPrimitive(
        static_layer=_make_static_layer(positions),
        dynamic_layers=[_make_dynamic_layer(n=2)],
        sky_cubemap=torch.zeros(6, 4, 4, 3),
        affine_matrix=torch.eye(3, 4).unsqueeze(0),
    )


class TestMaybeVoxelizeStaticLayer:
    def test_disabled_returns_primitive_unchanged(self, caplog):
        """enable_voxelization=False -> static_layer object is the same instance, no log."""
        positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        primitive = _make_primitive(positions)
        merger = KelvinPrimitiveMerge(PrimitiveMergeConfig(enable_voxelization=False))
        original_layer = primitive.static_layer
        with caplog.at_level(logging.INFO, logger="instant_nurec.predict.primitive_merge"):
            out = merger._maybe_voxelize_static_layer(primitive)
        assert out is primitive
        assert out.static_layer is original_layer
        assert not any("Voxelization" in r.message for r in caplog.records)

    def test_enabled_voxelizes_static_layer(self, caplog):
        """enable_voxelization=True -> static_layer reduced + reduction log emitted."""
        # Two Gaussians at origin collapse into one voxel; one Gaussian elsewhere stays its own voxel.
        positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        primitive = _make_primitive(positions)
        merger = KelvinPrimitiveMerge(PrimitiveMergeConfig(enable_voxelization=True, voxel_size=0.1))
        n_before = len(primitive.static_layer)
        with caplog.at_level(logging.INFO, logger="instant_nurec.predict.primitive_merge"):
            out = merger._maybe_voxelize_static_layer(primitive)
        assert out is primitive  # same dataclass; only ``static_layer`` field is replaced.
        assert len(out.static_layer) == 2
        assert len(out.static_layer) < n_before
        assert any("Voxelization" in r.message for r in caplog.records)

    def test_enabled_with_voxel_size_passthrough(self):
        """voxel_size is forwarded into the layer's voxelize(...) call."""
        # Two Gaussians 0.4 apart should stay distinct at voxel_size=0.1 but merge at voxel_size=1.0.
        positions = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
        primitive_small = _make_primitive(positions)
        merger_small = KelvinPrimitiveMerge(PrimitiveMergeConfig(enable_voxelization=True, voxel_size=0.1))
        out_small = merger_small._maybe_voxelize_static_layer(primitive_small)
        assert len(out_small.static_layer) == 2

        primitive_big = _make_primitive(positions)
        merger_big = KelvinPrimitiveMerge(PrimitiveMergeConfig(enable_voxelization=True, voxel_size=1.0))
        out_big = merger_big._maybe_voxelize_static_layer(primitive_big)
        assert len(out_big.static_layer) == 1
